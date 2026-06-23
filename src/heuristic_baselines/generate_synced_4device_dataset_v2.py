"""
Generate synced 4-device 5-min dataset v2.

Changes vs the earlier synced_4device_dataset:
  - Four source windows must have both start and end times within a tolerance
    (default 2 s), instead of being selected by maximum overlap.
  - Dataset inclusion uses only:
      1) enough PPG samples to reconstruct the common 50 Hz grid, and
      2) trustworthy ECG label QC.
  - PPG SQI, motion, PPG peak quality, and PPG-derived HRV are saved only as
    metadata. They do not remove windows.
  - window_qc_pass_strict is removed to avoid confusion with inclusion rules.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import config  # noqa: E402
from algorithms import hrv  # noqa: E402
from generate_synced_4device_dataset import (  # noqa: E402
    CHANNELS,
    DEFAULT_MAX_IBI_CORRECTION_RATIO,
    DEFAULT_MAX_MOTION_FRACTION,
    DEFAULT_MAX_PPG_IBI_CORRECTION_RATIO,
    DEFAULT_MAX_RMSSD_MS,
    DEFAULT_MIN_PPG_SQI,
    DEVICES,
    _ecg_label_and_qc,
    _ppg_channel_qc,
    _ppg_peaks_and_qc,
    _preprocess_for_peaks,
    _resample_to_grid,
)
from io_utils import normalize_participant_id  # noqa: E402


def _source_path(root: Path, pid: str, dev: str) -> Path:
    return root / pid / f"alignment_windows_{pid}_{dev}.npz"


def _open_sources(root: Path, pid: str) -> dict[str, np.lib.npyio.NpzFile]:
    sources = {}
    for dev in DEVICES:
        path = _source_path(root, pid, dev)
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[dev] = np.load(path, allow_pickle=True)
    return sources


def _close_sources(sources: dict[str, np.lib.npyio.NpzFile]) -> None:
    for z in sources.values():
        z.close()


def _match_start_end_windows(
    sources: dict[str, np.lib.npyio.NpzFile],
    *,
    tolerance_ms: float,
) -> list[dict[str, int | float]]:
    """Match source windows whose start and end times are both within tolerance."""
    t0 = {dev: np.asarray(sources[dev]["t0_ms"], dtype=float) for dev in DEVICES}
    t1 = {dev: np.asarray(sources[dev]["t1_ms"], dtype=float) for dev in DEVICES}
    ref = DEVICES[0]
    used = {dev: set() for dev in DEVICES}
    rows: list[dict[str, int | float]] = []

    for ref_idx, (s, e) in enumerate(zip(t0[ref], t1[ref])):
        match: dict[str, int | float] = {
            "target_t0_ms": float(s),
            "target_t1_ms": float(e),
            ref: ref_idx,
        }
        ok = True
        max_start_diff = 0.0
        max_end_diff = 0.0
        for dev in DEVICES[1:]:
            start_diff = np.abs(t0[dev] - s)
            end_diff = np.abs(t1[dev] - e)
            score = np.maximum(start_diff, end_diff)
            idx = int(np.argmin(score))
            sd = float(start_diff[idx])
            ed = float(end_diff[idx])
            if sd > tolerance_ms or ed > tolerance_ms or idx in used[dev]:
                ok = False
                break
            match[dev] = idx
            max_start_diff = max(max_start_diff, sd)
            max_end_diff = max(max_end_diff, ed)
        if ok:
            for dev, idx in match.items():
                if dev in used:
                    used[dev].add(int(idx))
            match["max_start_diff_ms"] = max_start_diff
            match["max_end_diff_ms"] = max_end_diff
            rows.append(match)
    return rows


def _motion_threshold_and_fraction(
    z: np.lib.npyio.NpzFile,
    source_indices: np.ndarray,
    *,
    percentile: float,
    seg_sec: float,
) -> tuple[float, np.ndarray]:
    fs = float(np.asarray(z["ppg_fs"]).item()) if "ppg_fs" in z.files else 100.0
    seg_n = max(1, int(round(seg_sec * fs)))
    ax = np.asarray(z["accel_x"])
    ay = np.asarray(z["accel_y"])
    az = np.asarray(z["accel_z"])
    vals: list[float] = []
    fracs = np.zeros(len(source_indices), dtype=np.float32)

    for i in range(ax.shape[0]):
        mag = np.sqrt(ax[i] ** 2 + ay[i] ** 2 + az[i] ** 2)
        n_seg = len(mag) // seg_n
        if n_seg == 0:
            vals.append(float(np.std(mag)))
        else:
            vals.extend(float(np.std(mag[j * seg_n:(j + 1) * seg_n])) for j in range(n_seg))
    threshold = float(np.percentile(vals, percentile)) if vals else float("nan")

    for out_i, src_i in enumerate(source_indices):
        mag = np.sqrt(ax[src_i] ** 2 + ay[src_i] ** 2 + az[src_i] ** 2)
        n_seg = len(mag) // seg_n
        if n_seg == 0:
            stds = np.array([float(np.std(mag))])
        else:
            stds = np.array([float(np.std(mag[j * seg_n:(j + 1) * seg_n])) for j in range(n_seg)])
        fracs[out_i] = float(np.mean(stds > threshold))
    del ax, ay, az
    return threshold, fracs


def _empty_object_array(shape: tuple[int, ...]) -> np.ndarray:
    arr = np.empty(shape, dtype=object)
    arr.fill(np.array([], dtype=np.float32))
    return arr


def generate_participant_v2(
    pid: str,
    *,
    source_root: Path,
    out_dir: Path,
    temp_dir: Path,
    window_sec: int,
    target_fs: float,
    start_end_tolerance_sec: float,
    min_valid_sample_ratio: float,
    preprocess_mode: str,
    motion_seg_sec: float,
    motion_percentile: float,
    ecg_min_valid_ibi_ratio: float,
    max_windows: int | None,
) -> Path | None:
    sources = _open_sources(source_root, pid)
    try:
        target_len = int(round(window_sec * target_fs))
        sample_step_ms = 1000.0 / target_fs
        max_gap_ms = max(float(start_end_tolerance_sec * 1000.0), sample_step_ms * 2.0)
        matches = _match_start_end_windows(
            sources,
            tolerance_ms=start_end_tolerance_sec * 1000.0,
        )
        if max_windows is not None:
            matches = matches[:max_windows]
        if not matches:
            out_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "participant": pid,
                "n_start_end_aligned": 0,
                "n_ecg_label_pass": 0,
                "n_kept": 0,
                "kept_pct_of_start_end_aligned": 0.0,
                "target_len": int(round(window_sec * target_fs)),
                "target_fs": target_fs,
                "start_end_tolerance_sec": start_end_tolerance_sec,
                "max_source_sample_gap_ms": max(float(start_end_tolerance_sec * 1000.0), (1000.0 / target_fs) * 2.0),
                "min_valid_sample_ratio": min_valid_sample_ratio,
                "preprocess_mode": preprocess_mode,
                "mean_max_start_diff_ms": np.nan,
                "mean_max_end_diff_ms": np.nan,
                "mean_ecg_valid_ibi_ratio": np.nan,
                "mean_ecg_ibi_correction_ratio": np.nan,
                "mean_ecg_qrs_sqi": np.nan,
                "mean_motion_fraction": np.nan,
                "mean_ppg_sqi": np.nan,
                "mean_ppg_valid_sample_ratio": np.nan,
                "mean_ppg_ibi_correction_ratio": np.nan,
                "ppg_quality_flag_pct_metadata_only": np.nan,
                "skip_reason": "no_source_windows_with_start_and_end_within_tolerance",
            }
            pd.DataFrame([summary]).to_csv(out_dir / f"synced_4device_v2_{pid}_summary.csv", index=False)
            print(f"[SKIP] {pid}: no windows satisfy start/end tolerance")
            return None

        n_match = len(matches)
        source_indices_all = np.zeros((n_match, len(DEVICES)), dtype=np.int32)
        source_start_offset_all = np.zeros((n_match, len(DEVICES)), dtype=np.float32)
        source_end_offset_all = np.zeros((n_match, len(DEVICES)), dtype=np.float32)
        max_start_diff_ms = np.zeros(n_match, dtype=np.float32)
        max_end_diff_ms = np.zeros(n_match, dtype=np.float32)
        t0_all = np.zeros(n_match, dtype=np.float64)
        t1_all = np.zeros(n_match, dtype=np.float64)

        for wi, idxs in enumerate(matches):
            target_t0 = float(idxs["target_t0_ms"])
            target_t1 = float(idxs["target_t1_ms"])
            t0_all[wi] = target_t0
            t1_all[wi] = target_t1
            max_start_diff_ms[wi] = float(idxs["max_start_diff_ms"])
            max_end_diff_ms[wi] = float(idxs["max_end_diff_ms"])
            for di, dev in enumerate(DEVICES):
                src_i = int(idxs[dev])
                source_indices_all[wi, di] = src_i
                source_start_offset_all[wi, di] = float((sources[dev]["t0_ms"][src_i] - target_t0) / 1000.0)
                source_end_offset_all[wi, di] = float((sources[dev]["t1_ms"][src_i] - target_t1) / 1000.0)

        print(f"[{pid}] start/end-aligned windows: {n_match}")

        earring = sources[DEVICES[0]]
        ecg_dict = {
            k: np.asarray(earring[k])
            for k in ("n_rr", "rr_intervals_ms", "r_peak_samples", "ecg", "ecg_valid_len", "t0_ms")
            if k in earring.files
        }

        ecg_keep = np.ones(n_match, dtype=bool)
        ecg_peak_times_all = _empty_object_array((n_match,))
        ecg_peak_indices_all = _empty_object_array((n_match,))
        ecg_peak_amplitudes_all = _empty_object_array((n_match,))
        ecg_rr_all = _empty_object_array((n_match,))
        ecg_rr_corrected_all = _empty_object_array((n_match,))
        ecg_rmssd_all = np.full(n_match, np.nan, dtype=np.float32)
        ecg_sdnn_all = np.full(n_match, np.nan, dtype=np.float32)
        ecg_valid_sample_ratio_all = np.zeros(n_match, dtype=np.float32)
        ecg_valid_ibi_ratio_all = np.zeros(n_match, dtype=np.float32)
        ecg_ibi_correction_ratio_all = np.full(n_match, np.nan, dtype=np.float32)
        ecg_qc_pass_all = np.zeros(n_match, dtype=bool)
        ecg_qc_reason_all = np.empty(n_match, dtype=object)
        ecg_label_qc_pass_all = np.zeros(n_match, dtype=bool)
        ecg_label_qc_reason_all = np.empty(n_match, dtype=object)
        ecg_qrs_sqi_all = np.full(n_match, np.nan, dtype=np.float32)
        ecg_peak_count_all = np.zeros(n_match, dtype=np.int32)
        ecg_hr_bpm_all = np.full(n_match, np.nan, dtype=np.float32)
        ecg_rr_cv_all = np.full(n_match, np.nan, dtype=np.float32)

        for wi, idxs in enumerate(matches):
            ref_idx = int(idxs[DEVICES[0]])
            label = _ecg_label_and_qc(
                ecg_dict,
                ref_idx,
                source_t0_ms=float(sources[DEVICES[0]]["t0_ms"][ref_idx]),
                target_t0_ms=float(idxs["target_t0_ms"]),
                window_ms=window_sec * 1000.0,
                target_fs=target_fs,
                ibi_min_ms=hrv.IBI_MIN_MS,
                ibi_max_ms=hrv.IBI_MAX_MS,
                min_valid_ibi_ratio=ecg_min_valid_ibi_ratio,
            )
            ecg_peak_times_all[wi] = label["ecg_r_peak_times_rel_ms"]
            ecg_peak_indices_all[wi] = label["ecg_r_peak_indices_50hz"]
            ecg_peak_amplitudes_all[wi] = label["ecg_r_peak_amplitudes_raw"]
            ecg_rr_all[wi] = label["ecg_rr_intervals_ms"]
            ecg_rr_corrected_all[wi] = label["ecg_rr_intervals_corrected_ms"]
            ecg_rmssd_all[wi] = label["ecg_rmssd_ms"]
            ecg_sdnn_all[wi] = label["ecg_sdnn_ms"]
            ecg_valid_sample_ratio_all[wi] = label["ecg_valid_sample_ratio"]
            ecg_valid_ibi_ratio_all[wi] = label["ecg_valid_ibi_ratio"]
            ecg_ibi_correction_ratio_all[wi] = label["ecg_ibi_correction_ratio"]
            ecg_qc_pass_all[wi] = label["ecg_qc_pass"]
            ecg_qc_reason_all[wi] = label["ecg_qc_reason"]
            ecg_label_qc_pass_all[wi] = label["ecg_label_qc_pass"]
            ecg_label_qc_reason_all[wi] = label["ecg_label_qc_reason"]
            ecg_qrs_sqi_all[wi] = label["ecg_qrs_sqi"]
            ecg_peak_count_all[wi] = label["ecg_peak_count"]
            ecg_hr_bpm_all[wi] = label["ecg_hr_bpm"]
            ecg_rr_cv_all[wi] = label["ecg_rr_cv"]
            if not label["ecg_label_qc_pass"]:
                ecg_keep[wi] = False

        del ecg_dict
        ecg_kept_idx = np.where(ecg_keep)[0]
        if ecg_kept_idx.size == 0:
            raise RuntimeError(f"{pid}: all start/end-aligned windows failed ECG label QC")

        n_ecg = int(ecg_kept_idx.size)
        print(f"[{pid}] ECG-label-kept windows: {n_ecg}/{n_match}")

        temp_dir.mkdir(parents=True, exist_ok=True)
        tmp = temp_dir / f"{pid}_v2_work"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

        ppg = np.lib.format.open_memmap(
            tmp / "ppg_50hz.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_ecg, len(DEVICES), len(CHANNELS), target_len),
        )
        ppg_valid_mask = np.lib.format.open_memmap(
            tmp / "ppg_valid_mask_50hz.npy",
            mode="w+",
            dtype=bool,
            shape=(n_ecg, len(DEVICES), len(CHANNELS), target_len),
        )

        ppg_valid_sample_ratio = np.zeros((n_ecg, len(DEVICES), len(CHANNELS)), dtype=np.float32)
        ppg_valid_ibi_ratio = np.zeros_like(ppg_valid_sample_ratio)
        ppg_sqi_arr = np.zeros_like(ppg_valid_sample_ratio)
        ppg_ibi_correction_ratio = np.full_like(ppg_valid_sample_ratio, np.nan)
        ppg_rmssd = np.full_like(ppg_valid_sample_ratio, np.nan)
        ppg_sdnn = np.full_like(ppg_valid_sample_ratio, np.nan)
        ppg_quality_flag = np.zeros((n_ecg, len(DEVICES), len(CHANNELS)), dtype=bool)
        motion_fraction = np.zeros((n_ecg, len(DEVICES)), dtype=np.float32)
        motion_threshold = np.zeros(len(DEVICES), dtype=np.float32)

        ppg_peak_times = _empty_object_array((n_ecg, len(DEVICES), len(CHANNELS)))
        ppg_peak_indices = _empty_object_array((n_ecg, len(DEVICES), len(CHANNELS)))
        ppg_peak_amplitudes = _empty_object_array((n_ecg, len(DEVICES), len(CHANNELS)))
        ppg_ibi = _empty_object_array((n_ecg, len(DEVICES), len(CHANNELS)))
        ppg_ibi_corrected = _empty_object_array((n_ecg, len(DEVICES), len(CHANNELS)))
        ppg_quality_reason = np.empty((n_ecg, len(DEVICES), len(CHANNELS)), dtype=object)

        source_indices = source_indices_all[ecg_kept_idx]
        source_start_offset_sec = source_start_offset_all[ecg_kept_idx]
        source_end_offset_sec = source_end_offset_all[ecg_kept_idx]
        t0_ms = t0_all[ecg_kept_idx]
        t1_ms = t1_all[ecg_kept_idx]

        for di, dev in enumerate(DEVICES):
            print(f"[{pid}] motion metadata {dev}")
            threshold, frac = _motion_threshold_and_fraction(
                sources[dev],
                source_indices[:, di],
                percentile=motion_percentile,
                seg_sec=motion_seg_sec,
            )
            motion_threshold[di] = threshold
            motion_fraction[:, di] = frac

        ppg_sample_keep = np.ones(n_ecg, dtype=bool)
        for di, dev in enumerate(DEVICES):
            print(f"[{pid}] PPG metadata/input {dev}")
            z = sources[dev]
            ppg_t_ms = np.asarray(z["ppg_t_ms"])
            for ci, ch in enumerate(CHANNELS):
                print(f"[{pid}]   channel {ch}")
                ch_data = np.asarray(z[ch])
                for out_i in range(n_ecg):
                    src_i = int(source_indices[out_i, di])
                    grid_ms = t0_ms[out_i] + np.arange(target_len, dtype=np.float64) * sample_step_ms
                    raw_50, valid_mask, valid_ratio = _resample_to_grid(
                        ch_data[src_i],
                        ppg_t_ms[src_i],
                        grid_ms,
                        max_gap_ms=max_gap_ms,
                    )
                    ppg[out_i, di, ci] = raw_50
                    ppg_valid_mask[out_i, di, ci] = valid_mask
                    ppg_valid_sample_ratio[out_i, di, ci] = valid_ratio
                    if valid_ratio < min_valid_sample_ratio:
                        ppg_sample_keep[out_i] = False

                    peak_signal = _preprocess_for_peaks(raw_50, target_fs, preprocess_mode)
                    peak_info = _ppg_peaks_and_qc(
                        raw_50,
                        peak_signal,
                        fs=target_fs,
                        valid_sample_ratio=valid_ratio,
                    )
                    ppg_peak_times[out_i, di, ci] = peak_info["ppg_peak_times_rel_ms"]
                    ppg_peak_indices[out_i, di, ci] = peak_info["ppg_peak_indices_50hz"]
                    ppg_peak_amplitudes[out_i, di, ci] = peak_info["ppg_peak_amplitudes_raw"]
                    ppg_ibi[out_i, di, ci] = peak_info["ppg_ibi_ms"]
                    ppg_ibi_corrected[out_i, di, ci] = peak_info["ppg_ibi_corrected_ms"]
                    ppg_ibi_correction_ratio[out_i, di, ci] = peak_info["ppg_ibi_correction_ratio"]
                    ppg_rmssd[out_i, di, ci] = peak_info["ppg_rmssd_ms"]
                    ppg_sdnn[out_i, di, ci] = peak_info["ppg_sdnn_ms"]
                    ppg_valid_ibi_ratio[out_i, di, ci] = peak_info["ppg_valid_ibi_ratio"]
                    ppg_sqi_arr[out_i, di, ci] = peak_info["ppg_sqi"]

                    # Metadata only: not used for inclusion.
                    qflag, qreason = _ppg_channel_qc(
                        valid_sample_ratio=valid_ratio,
                        valid_ibi_ratio=peak_info["ppg_valid_ibi_ratio"],
                        correction_ratio=peak_info["ppg_ibi_correction_ratio"],
                        rmssd_ms=peak_info["ppg_rmssd_ms"],
                        sqi=peak_info["ppg_sqi"],
                        motion_fraction=float(motion_fraction[out_i, di]),
                        min_valid_sample_ratio=min_valid_sample_ratio,
                        min_valid_ibi_ratio=ecg_min_valid_ibi_ratio,
                        min_sqi=DEFAULT_MIN_PPG_SQI,
                        max_motion_fraction=DEFAULT_MAX_MOTION_FRACTION,
                    )
                    ppg_quality_flag[out_i, di, ci] = qflag
                    ppg_quality_reason[out_i, di, ci] = qreason
                del ch_data
            del ppg_t_ms

        final_idx = np.where(ppg_sample_keep)[0]
        if final_idx.size == 0:
            raise RuntimeError(f"{pid}: all ECG-valid windows failed PPG sample coverage inclusion")
        print(f"[{pid}] final inclusion windows: {len(final_idx)}/{n_match}")

        cfg = {
            "participant": pid,
            "dataset_version": "v2_start_end_tolerance_ppg_sample_ecg_label_inclusion",
            "devices": DEVICES,
            "channels": CHANNELS,
            "window_sec": window_sec,
            "target_fs": target_fs,
            "target_len": target_len,
            "start_end_tolerance_sec": start_end_tolerance_sec,
            "alignment_method": "source_window_start_and_end_within_tolerance_then_absolute_time_grid_interpolation",
            "max_source_sample_gap_ms": max_gap_ms,
            "min_valid_sample_ratio": min_valid_sample_ratio,
            "preprocess_mode_for_peak_metadata": preprocess_mode,
            "motion_seg_sec": motion_seg_sec,
            "motion_percentile": motion_percentile,
            "ecg_min_valid_ibi_ratio": ecg_min_valid_ibi_ratio,
            "min_ppg_sqi_metadata_only": DEFAULT_MIN_PPG_SQI,
            "max_ibi_correction_ratio": DEFAULT_MAX_IBI_CORRECTION_RATIO,
            "max_ppg_ibi_correction_ratio_metadata_only": DEFAULT_MAX_PPG_IBI_CORRECTION_RATIO,
            "max_rmssd_ms": DEFAULT_MAX_RMSSD_MS,
            "max_motion_fraction_metadata_only": DEFAULT_MAX_MOTION_FRACTION,
            "primary_label": "ecg_r_peak_times_rel_ms",
            "inclusion_rule": "start_end_aligned && ecg_label_qc_pass && ppg_valid_sample_ratio>=min_valid_sample_ratio for all device/channel",
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"synced_4device_v2_{pid}.npz"
        print(f"[{pid}] saving {out_path}")
        idx = final_idx
        np.savez_compressed(
            out_path,
            config_json=np.array(json.dumps(cfg, ensure_ascii=False)),
            participant=np.array(pid),
            devices=np.array(DEVICES),
            channels=np.array(CHANNELS),
            kept_original_rows=ecg_kept_idx[idx].astype(np.int32),
            source_indices=source_indices[idx],
            source_start_offset_sec=source_start_offset_sec[idx],
            source_end_offset_sec=source_end_offset_sec[idx],
            max_start_diff_ms=max_start_diff_ms[ecg_kept_idx][idx],
            max_end_diff_ms=max_end_diff_ms[ecg_kept_idx][idx],
            t0_ms=t0_ms[idx],
            t1_ms=t1_ms[idx],
            ppg_50hz=ppg[idx],
            ppg_valid_mask_50hz=ppg_valid_mask[idx],
            ppg_valid_sample_ratio=ppg_valid_sample_ratio[idx],
            ppg_valid_ibi_ratio=ppg_valid_ibi_ratio[idx],
            ppg_sqi=ppg_sqi_arr[idx],
            ppg_peak_times_rel_ms=ppg_peak_times[idx],
            ppg_peak_indices_50hz=ppg_peak_indices[idx],
            ppg_peak_amplitudes_raw=ppg_peak_amplitudes[idx],
            ppg_ibi_ms=ppg_ibi[idx],
            ppg_ibi_corrected_ms=ppg_ibi_corrected[idx],
            ppg_ibi_correction_ratio=ppg_ibi_correction_ratio[idx],
            ppg_rmssd_ms=ppg_rmssd[idx],
            ppg_sdnn_ms=ppg_sdnn[idx],
            ppg_quality_flag=ppg_quality_flag[idx],
            ppg_quality_reason=ppg_quality_reason[idx],
            motion_fraction=motion_fraction[idx],
            motion_threshold=motion_threshold,
            ecg_r_peak_times_rel_ms=ecg_peak_times_all[ecg_kept_idx][idx],
            ecg_r_peak_indices_50hz=ecg_peak_indices_all[ecg_kept_idx][idx],
            ecg_r_peak_amplitudes_raw=ecg_peak_amplitudes_all[ecg_kept_idx][idx],
            ecg_rr_intervals_ms=ecg_rr_all[ecg_kept_idx][idx],
            ecg_rr_intervals_corrected_ms=ecg_rr_corrected_all[ecg_kept_idx][idx],
            ecg_rmssd_ms=ecg_rmssd_all[ecg_kept_idx][idx],
            ecg_sdnn_ms=ecg_sdnn_all[ecg_kept_idx][idx],
            ecg_valid_sample_ratio=ecg_valid_sample_ratio_all[ecg_kept_idx][idx],
            ecg_valid_ibi_ratio=ecg_valid_ibi_ratio_all[ecg_kept_idx][idx],
            ecg_ibi_correction_ratio=ecg_ibi_correction_ratio_all[ecg_kept_idx][idx],
            ecg_qc_pass=ecg_qc_pass_all[ecg_kept_idx][idx],
            ecg_qc_reason=ecg_qc_reason_all[ecg_kept_idx][idx],
            ecg_label_qc_pass=ecg_label_qc_pass_all[ecg_kept_idx][idx],
            ecg_label_qc_reason=ecg_label_qc_reason_all[ecg_kept_idx][idx],
            ecg_qrs_sqi=ecg_qrs_sqi_all[ecg_kept_idx][idx],
            ecg_peak_count=ecg_peak_count_all[ecg_kept_idx][idx],
            ecg_hr_bpm=ecg_hr_bpm_all[ecg_kept_idx][idx],
            ecg_rr_cv=ecg_rr_cv_all[ecg_kept_idx][idx],
            ecg_rpeak_detector_agreement=np.full(len(idx), np.nan, dtype=np.float32),
        )

        summary = {
            "participant": pid,
            "n_start_end_aligned": n_match,
            "n_ecg_label_pass": int(n_ecg),
            "n_kept": int(len(idx)),
            "kept_pct_of_start_end_aligned": 100.0 * len(idx) / n_match,
            "target_len": target_len,
            "target_fs": target_fs,
            "start_end_tolerance_sec": start_end_tolerance_sec,
            "max_source_sample_gap_ms": max_gap_ms,
            "min_valid_sample_ratio": min_valid_sample_ratio,
            "preprocess_mode": preprocess_mode,
            "mean_max_start_diff_ms": float(np.nanmean(max_start_diff_ms[ecg_kept_idx][idx])),
            "mean_max_end_diff_ms": float(np.nanmean(max_end_diff_ms[ecg_kept_idx][idx])),
            "mean_ecg_valid_ibi_ratio": float(np.nanmean(ecg_valid_ibi_ratio_all[ecg_kept_idx][idx])),
            "mean_ecg_ibi_correction_ratio": float(np.nanmean(ecg_ibi_correction_ratio_all[ecg_kept_idx][idx])),
            "mean_ecg_qrs_sqi": float(np.nanmean(ecg_qrs_sqi_all[ecg_kept_idx][idx])),
            "mean_motion_fraction": float(np.nanmean(motion_fraction[idx])),
            "mean_ppg_sqi": float(np.nanmean(ppg_sqi_arr[idx])),
            "mean_ppg_valid_sample_ratio": float(np.nanmean(ppg_valid_sample_ratio[idx])),
            "mean_ppg_ibi_correction_ratio": float(np.nanmean(ppg_ibi_correction_ratio[idx])),
            "ppg_quality_flag_pct_metadata_only": float(np.mean(ppg_quality_flag[idx]) * 100.0),
        }
        pd.DataFrame([summary]).to_csv(out_dir / f"synced_4device_v2_{pid}_summary.csv", index=False)
        shutil.rmtree(tmp)
        print(f"[SAVED] {out_path} kept={len(idx)}/{n_match}")
        return out_path
    finally:
        _close_sources(sources)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synced 4-device dataset v2.")
    ap.add_argument("--participant", required=True)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--out-dir", default=str(config.HEURISTIC_RESULT_ROOT / "synced_4device_dataset_v2"))
    ap.add_argument("--temp-dir", default=str(config.HEURISTIC_RESULT_ROOT / "synced_4device_dataset_v2_tmp"))
    ap.add_argument("--window-sec", type=int, default=300)
    ap.add_argument("--target-fs", type=float, default=50.0)
    ap.add_argument("--start-end-tolerance-sec", type=float, default=2.0)
    ap.add_argument("--min-valid-sample-ratio", type=float, default=0.50)
    ap.add_argument("--preprocess-mode", choices=("bandpass", "raw"), default="bandpass")
    ap.add_argument("--motion-seg-sec", type=float, default=10.0)
    ap.add_argument("--motion-percentile", type=float, default=75.0)
    ap.add_argument("--ecg-min-valid-ibi-ratio", type=float, default=0.80)
    ap.add_argument("--max-windows", type=int, default=None)
    args = ap.parse_args()

    generate_participant_v2(
        normalize_participant_id(args.participant),
        source_root=Path(args.source_root).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        temp_dir=Path(args.temp_dir).resolve(),
        window_sec=args.window_sec,
        target_fs=args.target_fs,
        start_end_tolerance_sec=args.start_end_tolerance_sec,
        min_valid_sample_ratio=args.min_valid_sample_ratio,
        preprocess_mode=args.preprocess_mode,
        motion_seg_sec=args.motion_seg_sec,
        motion_percentile=args.motion_percentile,
        ecg_min_valid_ibi_ratio=args.ecg_min_valid_ibi_ratio,
        max_windows=args.max_windows,
    )


if __name__ == "__main__":
    main()
