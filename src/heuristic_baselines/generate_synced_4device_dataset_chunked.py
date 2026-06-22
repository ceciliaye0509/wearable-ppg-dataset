"""
Chunked/memmap generator for very large participants.

This is the low-memory companion to generate_synced_4device_dataset.py. It is
intended for participants such as P3 whose compressed source NPZ files are too
large to load all devices and arrays at once.
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
    _match_overlapping_windows,
    _ppg_channel_qc,
    _ppg_peaks_and_qc,
    _preprocess_for_peaks,
    _resample_to_grid,
)
from io_utils import merged_windows_npz, normalize_participant_id  # noqa: E402


def _open_sources(pid: str) -> dict[str, np.lib.npyio.NpzFile]:
    sources = {}
    for dev in DEVICES:
        path = merged_windows_npz(config.HEURISTIC_WINDOWS_ROOT, pid, dev)
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file for {pid} {dev}: {path}")
        sources[dev] = np.load(path, allow_pickle=True)
    return sources


def _close_sources(sources: dict[str, np.lib.npyio.NpzFile]) -> None:
    for z in sources.values():
        z.close()


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


def _save_npz_compressed(out_path: Path, arrays: dict[str, object]) -> None:
    np.savez_compressed(out_path, **arrays)


def generate_participant_chunked(
    pid: str,
    *,
    out_dir: Path,
    temp_dir: Path,
    window_sec: int,
    target_fs: float,
    align_tolerance_sec: float,
    min_valid_sample_ratio: float,
    preprocess_mode: str,
    motion_seg_sec: float,
    motion_percentile: float,
    ecg_min_valid_ibi_ratio: float,
    max_windows: int | None,
) -> Path:
    sources = _open_sources(pid)
    try:
        target_len = int(round(window_sec * target_fs))
        window_ms = float(window_sec * 1000.0)
        sample_step_ms = 1000.0 / target_fs
        max_gap_ms = max(float(align_tolerance_sec * 1000.0), sample_step_ms * 2.0)

        meta = {
            dev: {
                "t0_ms": np.asarray(sources[dev]["t0_ms"], dtype=float),
                "t1_ms": np.asarray(sources[dev]["t1_ms"], dtype=float),
            }
            for dev in DEVICES
        }
        matches = _match_overlapping_windows(
            meta,
            window_ms=window_ms,
            min_overlap_ratio=min_valid_sample_ratio,
        )
        if max_windows is not None:
            matches = matches[:max_windows]
        if not matches:
            raise RuntimeError(f"{pid}: no four-device synced windows")

        n_match = len(matches)
        source_indices_all = np.zeros((n_match, len(DEVICES)), dtype=np.int32)
        source_start_offset_all = np.zeros((n_match, len(DEVICES)), dtype=np.float32)
        t0_all = np.zeros(n_match, dtype=np.float64)
        t1_all = np.zeros(n_match, dtype=np.float64)
        for wi, idxs in enumerate(matches):
            target_t0 = float(idxs["target_t0_ms"])
            t0_all[wi] = target_t0
            t1_all[wi] = float(idxs["target_t1_ms"])
            for di, dev in enumerate(DEVICES):
                src_i = int(idxs[dev])
                source_indices_all[wi, di] = src_i
                source_start_offset_all[wi, di] = float((meta[dev]["t0_ms"][src_i] - target_t0) / 1000.0)

        print(f"[{pid}] matched windows: {n_match}")

        # ECG first, so the large PPG memmaps are allocated only for ECG-valid labels.
        earring = sources[DEVICES[0]]
        ecg_dict = {k: np.asarray(earring[k]) for k in (
            "n_rr",
            "rr_intervals_ms",
            "r_peak_samples",
            "ecg",
            "ecg_valid_len",
            "t0_ms",
        ) if k in earring.files}

        ecg_keep = np.ones(n_match, dtype=bool)
        ecg_peak_times_all = np.empty(n_match, dtype=object)
        ecg_peak_indices_all = np.empty(n_match, dtype=object)
        ecg_peak_amplitudes_all = np.empty(n_match, dtype=object)
        ecg_rr_all = np.empty(n_match, dtype=object)
        ecg_rr_corrected_all = np.empty(n_match, dtype=object)
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
                source_t0_ms=float(meta[DEVICES[0]]["t0_ms"][ref_idx]),
                target_t0_ms=float(idxs["target_t0_ms"]),
                window_ms=window_ms,
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
            if not label["ecg_qc_pass"]:
                ecg_keep[wi] = False

        del ecg_dict
        kept_match_idx = np.where(ecg_keep)[0]
        if kept_match_idx.size == 0:
            raise RuntimeError(f"{pid}: all windows failed ECG label QC")

        n = int(kept_match_idx.size)
        print(f"[{pid}] ECG-kept windows: {n}/{n_match}")

        temp_dir.mkdir(parents=True, exist_ok=True)
        tmp = temp_dir / f"{pid}_chunked_work"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

        ppg = np.lib.format.open_memmap(
            tmp / "ppg_50hz.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n, len(DEVICES), len(CHANNELS), target_len),
        )
        ppg_valid_mask = np.lib.format.open_memmap(
            tmp / "ppg_valid_mask_50hz.npy",
            mode="w+",
            dtype=bool,
            shape=(n, len(DEVICES), len(CHANNELS), target_len),
        )

        ppg_valid_sample_ratio = np.zeros((n, len(DEVICES), len(CHANNELS)), dtype=np.float32)
        ppg_valid_ibi_ratio = np.zeros_like(ppg_valid_sample_ratio)
        ppg_sqi_arr = np.zeros_like(ppg_valid_sample_ratio)
        ppg_ibi_correction_ratio = np.full_like(ppg_valid_sample_ratio, np.nan)
        ppg_rmssd = np.full_like(ppg_valid_sample_ratio, np.nan)
        ppg_sdnn = np.full_like(ppg_valid_sample_ratio, np.nan)
        ppg_channel_qc_pass = np.zeros((n, len(DEVICES), len(CHANNELS)), dtype=bool)
        ppg_device_qc_pass = np.zeros((n, len(DEVICES)), dtype=bool)
        motion_fraction = np.zeros((n, len(DEVICES)), dtype=np.float32)
        motion_threshold = np.zeros(len(DEVICES), dtype=np.float32)

        ppg_peak_times = np.empty((n, len(DEVICES), len(CHANNELS)), dtype=object)
        ppg_peak_indices = np.empty_like(ppg_peak_times)
        ppg_peak_amplitudes = np.empty_like(ppg_peak_times)
        ppg_ibi = np.empty_like(ppg_peak_times)
        ppg_ibi_corrected = np.empty_like(ppg_peak_times)
        ppg_qc_reason = np.empty_like(ppg_peak_times)

        source_indices = source_indices_all[kept_match_idx]
        source_start_offset_sec = source_start_offset_all[kept_match_idx]
        t0_ms = t0_all[kept_match_idx]
        t1_ms = t1_all[kept_match_idx]

        for di, dev in enumerate(DEVICES):
            print(f"[{pid}] motion QC {dev}")
            threshold, frac = _motion_threshold_and_fraction(
                sources[dev],
                source_indices[:, di],
                percentile=motion_percentile,
                seg_sec=motion_seg_sec,
            )
            motion_threshold[di] = threshold
            motion_fraction[:, di] = frac

        final_keep = np.ones(n, dtype=bool)
        for di, dev in enumerate(DEVICES):
            print(f"[{pid}] PPG {dev}")
            z = sources[dev]
            ppg_t_ms = np.asarray(z["ppg_t_ms"])
            for ci, ch in enumerate(CHANNELS):
                print(f"[{pid}]   channel {ch}")
                ch_data = np.asarray(z[ch])
                for out_i, match_i in enumerate(kept_match_idx):
                    src_i = int(source_indices[out_i, di])
                    target_t0 = float(t0_ms[out_i])
                    grid_ms = target_t0 + np.arange(target_len, dtype=np.float64) * sample_step_ms
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
                        final_keep[out_i] = False
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
                    chan_pass, chan_reason = _ppg_channel_qc(
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
                    ppg_channel_qc_pass[out_i, di, ci] = chan_pass
                    ppg_qc_reason[out_i, di, ci] = chan_reason
                del ch_data
            del ppg_t_ms

        ppg.flush()
        ppg_valid_mask.flush()

        ppg_device_qc_pass[:] = np.any(ppg_channel_qc_pass, axis=2)
        window_qc_pass_strict = np.zeros(n, dtype=bool)
        window_qc_reason_strict = np.empty(n, dtype=object)
        for i in range(n):
            reasons = []
            if not ecg_label_qc_pass_all[kept_match_idx[i]]:
                reasons.append("ecg_label_qc_failed")
            if not bool(np.all(ppg_device_qc_pass[i])):
                reasons.append("ppg_device_qc_failed")
            window_qc_pass_strict[i] = len(reasons) == 0
            window_qc_reason_strict[i] = "ok" if not reasons else ";".join(reasons)

        if not bool(np.all(final_keep)):
            failed = int(np.sum(~final_keep))
            raise RuntimeError(
                f"{pid}: {failed} windows failed post-PPG inclusion QC; "
                "this chunked writer expects all ECG-kept windows to satisfy sample coverage."
            )

        cfg = {
            "participant": pid,
            "devices": DEVICES,
            "channels": CHANNELS,
            "window_sec": window_sec,
            "target_fs": target_fs,
            "target_len": target_len,
            "align_tolerance_sec": align_tolerance_sec,
            "alignment_method": "absolute_time_grid_interpolation",
            "max_source_sample_gap_ms": max_gap_ms,
            "min_valid_sample_ratio": min_valid_sample_ratio,
            "preprocess_mode_for_peak_qc": preprocess_mode,
            "motion_seg_sec": motion_seg_sec,
            "motion_percentile": motion_percentile,
            "ecg_min_valid_ibi_ratio": ecg_min_valid_ibi_ratio,
            "min_ppg_sqi": DEFAULT_MIN_PPG_SQI,
            "max_ibi_correction_ratio": DEFAULT_MAX_IBI_CORRECTION_RATIO,
            "max_ppg_ibi_correction_ratio": DEFAULT_MAX_PPG_IBI_CORRECTION_RATIO,
            "max_rmssd_ms": DEFAULT_MAX_RMSSD_MS,
            "max_motion_fraction": DEFAULT_MAX_MOTION_FRACTION,
            "primary_label": "ecg_r_peak_times_rel_ms",
            "chunked_memmap_generation": True,
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"synced_4device_{pid}.npz"
        print(f"[{pid}] saving {out_path}")
        _save_npz_compressed(
            out_path,
            {
                "config_json": np.array(json.dumps(cfg, ensure_ascii=False)),
                "participant": np.array(pid),
                "devices": np.array(DEVICES),
                "channels": np.array(CHANNELS),
                "kept_original_rows": kept_match_idx.astype(np.int32),
                "source_indices": source_indices,
                "source_start_offset_sec": source_start_offset_sec,
                "t0_ms": t0_ms,
                "t1_ms": t1_ms,
                "ppg_50hz": ppg,
                "ppg_valid_mask_50hz": ppg_valid_mask,
                "ppg_valid_sample_ratio": ppg_valid_sample_ratio,
                "ppg_valid_ibi_ratio": ppg_valid_ibi_ratio,
                "ppg_sqi": ppg_sqi_arr,
                "ppg_peak_times_rel_ms": ppg_peak_times,
                "ppg_peak_indices_50hz": ppg_peak_indices,
                "ppg_peak_amplitudes_raw": ppg_peak_amplitudes,
                "ppg_ibi_ms": ppg_ibi,
                "ppg_ibi_corrected_ms": ppg_ibi_corrected,
                "ppg_ibi_correction_ratio": ppg_ibi_correction_ratio,
                "ppg_rmssd_ms": ppg_rmssd,
                "ppg_sdnn_ms": ppg_sdnn,
                "ppg_channel_qc_pass": ppg_channel_qc_pass,
                "ppg_device_qc_pass": ppg_device_qc_pass,
                "ppg_qc_reason": ppg_qc_reason,
                "motion_fraction": motion_fraction,
                "motion_threshold": motion_threshold,
                "ecg_r_peak_times_rel_ms": ecg_peak_times_all[kept_match_idx],
                "ecg_r_peak_indices_50hz": ecg_peak_indices_all[kept_match_idx],
                "ecg_r_peak_amplitudes_raw": ecg_peak_amplitudes_all[kept_match_idx],
                "ecg_rr_intervals_ms": ecg_rr_all[kept_match_idx],
                "ecg_rr_intervals_corrected_ms": ecg_rr_corrected_all[kept_match_idx],
                "ecg_rmssd_ms": ecg_rmssd_all[kept_match_idx],
                "ecg_sdnn_ms": ecg_sdnn_all[kept_match_idx],
                "ecg_valid_sample_ratio": ecg_valid_sample_ratio_all[kept_match_idx],
                "ecg_valid_ibi_ratio": ecg_valid_ibi_ratio_all[kept_match_idx],
                "ecg_ibi_correction_ratio": ecg_ibi_correction_ratio_all[kept_match_idx],
                "ecg_qc_pass": ecg_qc_pass_all[kept_match_idx],
                "ecg_qc_reason": ecg_qc_reason_all[kept_match_idx],
                "ecg_label_qc_pass": ecg_label_qc_pass_all[kept_match_idx],
                "ecg_label_qc_reason": ecg_label_qc_reason_all[kept_match_idx],
                "ecg_qrs_sqi": ecg_qrs_sqi_all[kept_match_idx],
                "ecg_peak_count": ecg_peak_count_all[kept_match_idx],
                "ecg_hr_bpm": ecg_hr_bpm_all[kept_match_idx],
                "ecg_rr_cv": ecg_rr_cv_all[kept_match_idx],
                "ecg_rpeak_detector_agreement": np.full(n, np.nan, dtype=np.float32),
                "window_qc_pass_strict": window_qc_pass_strict,
                "window_qc_reason_strict": window_qc_reason_strict,
            },
        )

        summary = {
            "participant": pid,
            "n_aligned": n_match,
            "n_kept": n,
            "kept_pct": 100.0 * n / n_match,
            "target_len": target_len,
            "target_fs": target_fs,
            "align_tolerance_sec": align_tolerance_sec,
            "max_source_sample_gap_ms": max_gap_ms,
            "preprocess_mode": preprocess_mode,
            "mean_ecg_valid_ibi_ratio": float(np.nanmean(ecg_valid_ibi_ratio_all[kept_match_idx])),
            "mean_ecg_ibi_correction_ratio": float(np.nanmean(ecg_ibi_correction_ratio_all[kept_match_idx])),
            "mean_ecg_qrs_sqi": float(np.nanmean(ecg_qrs_sqi_all[kept_match_idx])),
            "mean_motion_fraction": float(np.nanmean(motion_fraction)),
            "mean_ppg_sqi": float(np.nanmean(ppg_sqi_arr)),
            "mean_ppg_valid_sample_ratio": float(np.nanmean(ppg_valid_sample_ratio)),
            "mean_ppg_ibi_correction_ratio": float(np.nanmean(ppg_ibi_correction_ratio)),
            "strict_window_pass_pct": float(np.mean(window_qc_pass_strict) * 100.0),
        }
        pd.DataFrame([summary]).to_csv(out_dir / f"synced_4device_{pid}_summary.csv", index=False)
        print(f"[SAVED] {out_path} kept={n}/{n_match}")
        shutil.rmtree(tmp)
        return out_path
    finally:
        _close_sources(sources)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synced 4-device dataset for large participants.")
    ap.add_argument("--participant", default="P3")
    ap.add_argument("--out-dir", default=str(config.HEURISTIC_RESULT_ROOT / "synced_4device_dataset"))
    ap.add_argument("--temp-dir", default=str(config.HEURISTIC_RESULT_ROOT / "synced_4device_dataset_tmp"))
    ap.add_argument("--window-sec", type=int, default=300)
    ap.add_argument("--target-fs", type=float, default=50.0)
    ap.add_argument("--align-tolerance-sec", type=float, default=2.0)
    ap.add_argument("--min-valid-sample-ratio", type=float, default=0.50)
    ap.add_argument("--preprocess-mode", choices=("bandpass", "raw"), default="bandpass")
    ap.add_argument("--motion-seg-sec", type=float, default=10.0)
    ap.add_argument("--motion-percentile", type=float, default=75.0)
    ap.add_argument("--ecg-min-valid-ibi-ratio", type=float, default=0.80)
    ap.add_argument("--max-windows", type=int, default=None)
    args = ap.parse_args()

    generate_participant_chunked(
        normalize_participant_id(args.participant),
        out_dir=Path(args.out_dir).resolve(),
        temp_dir=Path(args.temp_dir).resolve(),
        window_sec=args.window_sec,
        target_fs=args.target_fs,
        align_tolerance_sec=args.align_tolerance_sec,
        min_valid_sample_ratio=args.min_valid_sample_ratio,
        preprocess_mode=args.preprocess_mode,
        motion_seg_sec=args.motion_seg_sec,
        motion_percentile=args.motion_percentile,
        ecg_min_valid_ibi_ratio=args.ecg_min_valid_ibi_ratio,
        max_windows=args.max_windows,
    )


if __name__ == "__main__":
    main()
