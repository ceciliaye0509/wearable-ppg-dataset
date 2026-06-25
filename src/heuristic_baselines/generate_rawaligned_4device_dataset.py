"""
Generate raw-timeline aligned 4-device PPG dataset with ECG R-peak labels.

This generator is separate from the earlier 5min_windowed-based v2 generator.
It follows the updated dataset definition:

  - start from raw PPG/ECG timelines;
  - cut windows on one shared absolute-time grid;
  - resample all four PPG devices and both channels to a common 50 Hz grid;
  - include windows using only PPG sample coverage and ECG label QC;
  - keep PPG SQI, motion, PPG peaks, and PPG-derived HRV as metadata only;
  - use ECG R-peak location arrays as the primary label.

Run with the project venv so SciPy and NeuroKit2 are available, e.g.

    /Users/jiqiyu/Desktop/Daily_HRV/venv/bin/python \
      src/heuristic_baselines/generate_rawaligned_4device_dataset.py \
      --dataset-name synced_4device_rawaligned_strict_reference \
      --stride-sec 300
"""
from __future__ import annotations

import argparse
import json
import math
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
    _ppg_channel_qc,
    _ppg_peaks_and_qc,
    _preprocess_for_peaks,
    _resample_to_grid,
    _simple_ecg_qrs_sqi,
)
from io_utils import normalize_participant_id  # noqa: E402

try:
    import neurokit2 as nk
except ImportError as exc:  # pragma: no cover
    raise SystemExit("neurokit2 is required. Run with /Users/jiqiyu/Desktop/Daily_HRV/venv/bin/python") from exc


RAW_ROOT = config.HEURISTIC_HF_SUBMISSION_ROOT / "raw_data"
MAX_PEAKS_PER_WINDOW = 1200


def _participant_ids(raw_root: Path, raw: str | None) -> list[str]:
    if raw:
        return [normalize_participant_id(p) for p in raw.split(",") if p.strip()]
    return sorted((p.name for p in raw_root.iterdir() if p.is_dir()), key=lambda x: int(x[1:]))


def _raw_path(raw_root: Path, pid: str, name: str) -> Path:
    return raw_root / pid / f"{pid}_{name}_raw.npz"


def _load_ppg_raw(raw_root: Path, pid: str) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for dev in DEVICES:
        path = _raw_path(raw_root, pid, dev)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as z:
            out[dev] = {k: np.asarray(z[k]) for k in z.files}
    return out


def _load_ecg_raw(raw_root: Path, pid: str) -> dict[str, np.ndarray]:
    path = raw_root / pid / f"{pid}_polar_ecg_raw.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _infer_fs_ms(t_ms: np.ndarray) -> float:
    t = np.asarray(t_ms, dtype=np.float64)
    dt = np.diff(t[np.isfinite(t)])
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        return float("nan")
    return float(1000.0 / np.median(dt))


def _empty_object_array(shape: tuple[int, ...]) -> np.ndarray:
    arr = np.empty(shape, dtype=object)
    arr.fill(np.array([], dtype=np.float32))
    return arr


def _boundary_offsets_ms(times_ms: np.ndarray, t0: float, t1: float) -> tuple[float, float, int]:
    t = np.asarray(times_ms, dtype=np.float64)
    lo = int(np.searchsorted(t, t0, side="left"))
    hi = int(np.searchsorted(t, t1, side="right"))
    if hi <= lo:
        return float("inf"), float("inf"), 0
    first = float(t[lo])
    last = float(t[hi - 1])
    return first - t0, t1 - last, hi - lo


def _candidate_windows(
    ppg_raw: dict[str, dict[str, np.ndarray]],
    ecg_raw: dict[str, np.ndarray],
    *,
    window_sec: int,
    stride_sec: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    starts = [float(ppg_raw[d]["timestamp"][0]) for d in DEVICES]
    ends = [float(ppg_raw[d]["timestamp"][-1]) for d in DEVICES]
    starts.append(float(ecg_raw["time_ms"][0]))
    ends.append(float(ecg_raw["time_ms"][-1]))
    common_start = max(starts)
    common_end = min(ends)
    window_ms = window_sec * 1000.0
    stride_ms = stride_sec * 1000.0
    first_t0 = math.ceil(common_start / stride_ms) * stride_ms
    last_t0 = common_end - window_ms
    if last_t0 < first_t0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, {
            "common_start_ms": common_start,
            "common_end_ms": common_end,
            "common_duration_sec": max(0.0, (common_end - common_start) / 1000.0),
        }
    t0 = np.arange(first_t0, last_t0 + 0.5 * stride_ms, stride_ms, dtype=np.float64)
    t1 = t0 + window_ms
    return t0, t1, {
        "common_start_ms": common_start,
        "common_end_ms": common_end,
        "common_duration_sec": max(0.0, (common_end - common_start) / 1000.0),
    }


def _ecg_fs_from_raw(ecg_raw: dict[str, np.ndarray]) -> float:
    t_ms = np.asarray(ecg_raw["time_ms"], dtype=np.float64)
    return _infer_fs_ms(t_ms)


def _ecg_label_for_window(
    ecg_raw: dict[str, np.ndarray],
    ecg_fs: float,
    *,
    t0: float,
    t1: float,
    target_fs: float,
    min_valid_ibi_ratio: float,
    min_hr_bpm: float,
    max_hr_bpm: float,
    min_ecg_valid_sample_ratio: float,
    detector_method: str,
) -> dict[str, object]:
    window_ms = t1 - t0
    ecg = np.asarray(ecg_raw["ecg_uv"], dtype=np.float64)
    raw_t = np.asarray(ecg_raw["time_ms"], dtype=np.float64)
    lo = int(np.searchsorted(raw_t, t0, side="left"))
    hi = int(np.searchsorted(raw_t, t1, side="right"))
    seg = ecg[lo:hi]
    expected_n = max(1, int(round((window_ms / 1000.0) * ecg_fs)))
    valid_ratio = float(np.isfinite(seg).sum() / expected_n) if seg.size else 0.0

    peak_times = np.array([], dtype=np.float64)
    local_peak_idx = np.array([], dtype=np.int64)
    clean = np.array([], dtype=np.float64)
    if seg.size >= 50 and np.isfinite(ecg_fs) and ecg_fs > 0 and np.isfinite(seg).all():
        try:
            clean = nk.ecg_clean(seg, sampling_rate=ecg_fs, method="pantompkins1985")
            _, info = nk.ecg_peaks(clean, sampling_rate=ecg_fs, method="pantompkins1985")
            local_peak_idx = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64).ravel()
            local_peak_idx = local_peak_idx[(local_peak_idx >= 0) & (local_peak_idx < seg.size)]
            if local_peak_idx.size:
                peak_times = raw_t[lo + local_peak_idx]
        except Exception:
            peak_times = np.array([], dtype=np.float64)
            local_peak_idx = np.array([], dtype=np.int64)

    rel_ms = peak_times - t0
    peak_count = int(rel_ms.size)
    rr = np.diff(peak_times).astype(np.float64) if peak_times.size >= 2 else np.array([], dtype=np.float64)
    valid_rr = (rr >= hrv.IBI_MIN_MS) & (rr <= hrv.IBI_MAX_MS)
    valid_ibi_ratio = float(valid_rr.mean()) if rr.size else 0.0
    nn = rr[valid_rr]
    if nn.size >= 3:
        corrected, corr_ratio = hrv._correct_ibi_artifacts_with_ratio(nn)
        diff = np.diff(corrected)
        rmssd = float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan")
        sdnn = float(np.std(corrected, ddof=1)) if corrected.size > 1 else float("nan")
        hr_bpm = float(60000.0 / np.nanmean(corrected)) if np.nanmean(corrected) > 0 else float("nan")
        rr_cv = float(np.nanstd(corrected) / np.nanmean(corrected)) if np.nanmean(corrected) > 0 else float("nan")
    else:
        corrected = np.array([], dtype=np.float64)
        corr_ratio = float("nan")
        rmssd = float("nan")
        sdnn = float("nan")
        hr_bpm = float("nan")
        rr_cv = float("nan")

    qrs_sqi = _simple_ecg_qrs_sqi(clean if clean.size else seg, local_peak_idx, ecg_fs)
    peak_amp = []
    for p in local_peak_idx:
        if 0 <= int(p) < seg.size:
            peak_amp.append(float(seg[int(p)]))
    min_peak_count = int(math.floor((window_ms / 1000.0) * min_hr_bpm / 60.0))
    qc_pass = bool(
        valid_ratio >= min_ecg_valid_sample_ratio
        and valid_ibi_ratio >= min_valid_ibi_ratio
        and peak_count >= max(3, min_peak_count)
        and np.isfinite(rmssd)
        and (not np.isfinite(corr_ratio) or corr_ratio <= DEFAULT_MAX_IBI_CORRECTION_RATIO)
        and (not np.isfinite(hr_bpm) or (min_hr_bpm <= hr_bpm <= max_hr_bpm))
        and (not np.isfinite(rmssd) or rmssd <= DEFAULT_MAX_RMSSD_MS)
    )
    reasons: list[str] = []
    if valid_ratio < min_ecg_valid_sample_ratio:
        reasons.append("low_ecg_sample_ratio")
    if valid_ibi_ratio < min_valid_ibi_ratio:
        reasons.append("low_ecg_valid_ibi_ratio")
    if peak_count < max(3, min_peak_count):
        reasons.append("too_few_ecg_peaks")
    if not np.isfinite(rmssd):
        reasons.append("invalid_ecg_hrv")
    if np.isfinite(corr_ratio) and corr_ratio > DEFAULT_MAX_IBI_CORRECTION_RATIO:
        reasons.append("high_ecg_ibi_correction_ratio")
    if np.isfinite(hr_bpm) and not (min_hr_bpm <= hr_bpm <= max_hr_bpm):
        reasons.append("ecg_hr_out_of_range")
    if np.isfinite(rmssd) and rmssd > DEFAULT_MAX_RMSSD_MS:
        reasons.append("ecg_rmssd_too_high")
    if detector_method == "bad_ecg_fs":
        reasons.append(detector_method)

    return {
        "ecg_r_peak_times_rel_ms": rel_ms.astype(np.float32),
        "ecg_r_peak_indices_50hz": np.rint(rel_ms / 1000.0 * target_fs).astype(np.int32),
        "ecg_r_peak_amplitudes_raw": np.asarray(peak_amp, dtype=np.float32),
        "ecg_rr_intervals_ms": rr.astype(np.float32),
        "ecg_rr_intervals_corrected_ms": corrected.astype(np.float32),
        "ecg_rmssd_ms": rmssd,
        "ecg_sdnn_ms": sdnn,
        "ecg_valid_sample_ratio": valid_ratio,
        "ecg_valid_ibi_ratio": valid_ibi_ratio,
        "ecg_ibi_correction_ratio": corr_ratio,
        "ecg_qc_pass": qc_pass,
        "ecg_qc_reason": "ok" if qc_pass else ";".join(reasons),
        "ecg_label_qc_pass": qc_pass,
        "ecg_label_qc_reason": "ok" if qc_pass else ";".join(reasons),
        "ecg_qrs_sqi": qrs_sqi,
        "ecg_peak_count": peak_count,
        "ecg_hr_bpm": hr_bpm,
        "ecg_rr_cv": rr_cv,
    }


def _motion_thresholds_from_raw(
    ppg_raw: dict[str, dict[str, np.ndarray]],
    *,
    percentile: float,
    seg_sec: float,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for dev, d in ppg_raw.items():
        t = np.asarray(d["timestamp"], dtype=np.float64)
        fs = _infer_fs_ms(t)
        seg_n = max(1, int(round(seg_sec * fs))) if np.isfinite(fs) else 1000
        mag = np.sqrt(
            np.asarray(d["accel_x"], dtype=np.float64) ** 2
            + np.asarray(d["accel_y"], dtype=np.float64) ** 2
            + np.asarray(d["accel_z"], dtype=np.float64) ** 2
        )
        n_seg = mag.size // seg_n
        if n_seg <= 0:
            vals = np.array([np.nanstd(mag)], dtype=np.float64)
        else:
            vals = np.array([np.nanstd(mag[i * seg_n:(i + 1) * seg_n]) for i in range(n_seg)], dtype=np.float64)
        thresholds[dev] = float(np.nanpercentile(vals, percentile)) if vals.size else float("nan")
    return thresholds


def _motion_fraction_window(
    dev_raw: dict[str, np.ndarray],
    *,
    t0: float,
    t1: float,
    threshold: float,
    seg_sec: float,
) -> float:
    t = np.asarray(dev_raw["timestamp"], dtype=np.float64)
    lo = int(np.searchsorted(t, t0, side="left"))
    hi = int(np.searchsorted(t, t1, side="right"))
    if hi <= lo:
        return float("nan")
    fs = _infer_fs_ms(t[lo:hi])
    seg_n = max(1, int(round(seg_sec * fs))) if np.isfinite(fs) else 1000
    mag = np.sqrt(
        np.asarray(dev_raw["accel_x"][lo:hi], dtype=np.float64) ** 2
        + np.asarray(dev_raw["accel_y"][lo:hi], dtype=np.float64) ** 2
        + np.asarray(dev_raw["accel_z"][lo:hi], dtype=np.float64) ** 2
    )
    n_seg = mag.size // seg_n
    if n_seg <= 0:
        stds = np.array([np.nanstd(mag)], dtype=np.float64)
    else:
        stds = np.array([np.nanstd(mag[i * seg_n:(i + 1) * seg_n]) for i in range(n_seg)], dtype=np.float64)
    return float(np.nanmean(stds > threshold)) if np.isfinite(threshold) else float("nan")


def _write_readme(out_dir: Path, dataset_name: str, cfg: dict[str, object], summary: pd.DataFrame) -> None:
    total = int(summary["n_kept"].sum()) if "n_kept" in summary else 0
    participants = int((summary["n_kept"] > 0).sum()) if "n_kept" in summary else 0
    text = f"""# {dataset_name}

This dataset was generated from `snowballlab/Multisite-PPG/raw_data` by aligning
raw timelines first and then cutting shared absolute-time windows.

## Core Parameters

```json
{json.dumps(cfg, indent=2, ensure_ascii=False)}
```

## Inclusion Rule

A window is kept only if:

1. all four PPG devices have samples near the shared window start and end within
   `alignment_tolerance_sec`;
2. every device/channel has `ppg_valid_sample_ratio >= min_valid_sample_ratio`
   after 50 Hz resampling;
3. ECG label QC passes.

PPG SQI, motion fraction, PPG peak quality, and PPG-derived HRV are metadata
only. They are not used to remove windows.

## Current Output

- kept windows: {total}
- participants with kept windows: {participants}
- primary label: `ecg_r_peak_times_rel_ms`
- model input: `ppg_50hz` with shape `(N, 4, 2, target_len)`
- missing-sample mask: `ppg_valid_mask_50hz`

## Important Fields

| Field | Meaning |
|---|---|
| `ppg_50hz` | 4-device, 2-channel PPG resampled to the shared 50 Hz grid |
| `ppg_valid_mask_50hz` | True where a resampled point is supported by nearby raw samples |
| `ppg_valid_sample_ratio` | PPG sample coverage used for inclusion |
| `ecg_r_peak_times_rel_ms` | primary label: ECG R-peak times relative to window start |
| `ecg_rr_intervals_ms` | ECG RR intervals from Pan-Tompkins R-peaks |
| `ecg_rmssd_ms`, `ecg_sdnn_ms` | ECG-derived HRV labels from corrected RR intervals |
| `ecg_label_qc_pass`, `ecg_label_qc_reason` | final ECG label trustworthiness flag/reason |
| `ecg_qrs_sqi` | simple raw-ECG QRS prominence/consistency SQI |
| `ppg_peak_times_rel_ms`, `ppg_ibi_ms`, `ppg_rmssd_ms` | PPG-derived metadata, not inclusion criteria |
| `ppg_quality_flag`, `ppg_quality_reason` | metadata-only PPG quality indicators |
| `motion_fraction` | metadata-only accelerometer motion fraction |

## Summary

See `{dataset_name}_summary.csv` for per-participant counts and quality means.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def generate_participant(
    pid: str,
    *,
    raw_root: Path,
    out_dir: Path,
    dataset_name: str,
    window_sec: int,
    stride_sec: int,
    target_fs: float,
    alignment_tolerance_sec: float,
    min_valid_sample_ratio: float,
    max_source_gap_ms: float,
    preprocess_mode: str,
    motion_seg_sec: float,
    motion_percentile: float,
    ecg_min_valid_ibi_ratio: float,
    min_ecg_valid_sample_ratio: float,
    min_hr_bpm: float,
    max_hr_bpm: float,
    max_windows: int | None,
) -> Path | None:
    print(f"[{pid}] loading raw data")
    ppg_raw = _load_ppg_raw(raw_root, pid)
    ecg_raw = _load_ecg_raw(raw_root, pid)
    target_len = int(round(window_sec * target_fs))
    sample_step_ms = 1000.0 / target_fs
    window_ms = window_sec * 1000.0
    tolerance_ms = alignment_tolerance_sec * 1000.0

    t0_all, t1_all, overlap_info = _candidate_windows(
        ppg_raw,
        ecg_raw,
        window_sec=window_sec,
        stride_sec=stride_sec,
    )
    if max_windows is not None:
        t0_all = t0_all[:max_windows]
        t1_all = t1_all[:max_windows]

    out_dir.mkdir(parents=True, exist_ok=True)
    if t0_all.size == 0:
        summary = {
            "participant": pid,
            "n_candidate_windows": 0,
            "n_boundary_aligned": 0,
            "n_ppg_sample_pass": 0,
            "n_ecg_label_pass": 0,
            "n_kept": 0,
            **overlap_info,
            "skip_reason": "no_raw_common_overlap_for_window",
        }
        pd.DataFrame([summary]).to_csv(out_dir / f"{dataset_name}_{pid}_summary.csv", index=False)
        print(f"[SKIP] {pid}: no raw common overlap")
        return None

    ecg_fs = _ecg_fs_from_raw(ecg_raw)
    ecg_detector = "window_pantompkins1985"
    print(f"[{pid}] ECG fs={ecg_fs:.3f}, detector={ecg_detector}")
    motion_threshold = _motion_thresholds_from_raw(
        ppg_raw,
        percentile=motion_percentile,
        seg_sec=motion_seg_sec,
    )

    boundary_keep = []
    max_start_diff_ms = []
    max_end_diff_ms = []
    ppg_sample_keep_pre = []
    ecg_keep_pre = []
    labels: list[dict[str, object]] = []

    print(f"[{pid}] screening {t0_all.size} candidate windows")
    for wi, (t0, t1) in enumerate(zip(t0_all, t1_all)):
        start_offsets = []
        end_offsets = []
        boundary_ok = True
        for dev in DEVICES:
            so, eo, _n = _boundary_offsets_ms(ppg_raw[dev]["timestamp"], float(t0), float(t1))
            start_offsets.append(so)
            end_offsets.append(eo)
            if so > tolerance_ms or eo > tolerance_ms:
                boundary_ok = False
        boundary_keep.append(boundary_ok)
        max_start_diff_ms.append(float(np.nanmax(start_offsets)))
        max_end_diff_ms.append(float(np.nanmax(end_offsets)))
        if not boundary_ok:
            labels.append({})
            ppg_sample_keep_pre.append(False)
            ecg_keep_pre.append(False)
            continue
        label = _ecg_label_for_window(
            ecg_raw,
            ecg_fs,
            t0=float(t0),
            t1=float(t1),
            target_fs=target_fs,
            min_valid_ibi_ratio=ecg_min_valid_ibi_ratio,
            min_hr_bpm=min_hr_bpm,
            max_hr_bpm=max_hr_bpm,
            min_ecg_valid_sample_ratio=min_ecg_valid_sample_ratio,
            detector_method=ecg_detector,
        )
        labels.append(label)
        ecg_keep_pre.append(bool(label["ecg_label_qc_pass"]))
        ppg_sample_keep_pre.append(True)  # exact per-channel check happens while resampling

    boundary_keep_arr = np.asarray(boundary_keep, dtype=bool)
    ecg_keep_arr = np.asarray(ecg_keep_pre, dtype=bool)
    prelim_idx = np.where(boundary_keep_arr & ecg_keep_arr)[0]
    if prelim_idx.size == 0:
        summary = {
            "participant": pid,
            "n_candidate_windows": int(t0_all.size),
            "n_boundary_aligned": int(boundary_keep_arr.sum()),
            "n_ppg_sample_pass": 0,
            "n_ecg_label_pass": int(ecg_keep_arr.sum()),
            "n_kept": 0,
            **overlap_info,
            "skip_reason": "no_window_after_boundary_and_ecg_qc",
        }
        pd.DataFrame([summary]).to_csv(out_dir / f"{dataset_name}_{pid}_summary.csv", index=False)
        print(f"[SKIP] {pid}: no windows after boundary + ECG QC")
        return None

    n_pre = int(prelim_idx.size)
    ppg = np.zeros((n_pre, len(DEVICES), len(CHANNELS), target_len), dtype=np.float32)
    ppg_valid_mask = np.zeros_like(ppg, dtype=bool)
    ppg_valid_sample_ratio = np.zeros((n_pre, len(DEVICES), len(CHANNELS)), dtype=np.float32)
    ppg_valid_ibi_ratio = np.zeros_like(ppg_valid_sample_ratio)
    ppg_sqi_arr = np.zeros_like(ppg_valid_sample_ratio)
    ppg_ibi_correction_ratio = np.full_like(ppg_valid_sample_ratio, np.nan)
    ppg_rmssd = np.full_like(ppg_valid_sample_ratio, np.nan)
    ppg_sdnn = np.full_like(ppg_valid_sample_ratio, np.nan)
    ppg_quality_flag = np.zeros((n_pre, len(DEVICES), len(CHANNELS)), dtype=bool)
    ppg_quality_reason = np.empty((n_pre, len(DEVICES), len(CHANNELS)), dtype=object)
    motion_fraction = np.zeros((n_pre, len(DEVICES)), dtype=np.float32)

    ppg_peak_times = _empty_object_array((n_pre, len(DEVICES), len(CHANNELS)))
    ppg_peak_indices = _empty_object_array((n_pre, len(DEVICES), len(CHANNELS)))
    ppg_peak_amplitudes = _empty_object_array((n_pre, len(DEVICES), len(CHANNELS)))
    ppg_ibi = _empty_object_array((n_pre, len(DEVICES), len(CHANNELS)))
    ppg_ibi_corrected = _empty_object_array((n_pre, len(DEVICES), len(CHANNELS)))

    final_keep = np.ones(n_pre, dtype=bool)
    print(f"[{pid}] resampling PPG for {n_pre} ECG-valid windows")
    for out_i, wi in enumerate(prelim_idx):
        t0 = float(t0_all[wi])
        grid_ms = t0 + np.arange(target_len, dtype=np.float64) * sample_step_ms
        for di, dev in enumerate(DEVICES):
            motion_fraction[out_i, di] = _motion_fraction_window(
                ppg_raw[dev],
                t0=t0,
                t1=float(t1_all[wi]),
                threshold=motion_threshold[dev],
                seg_sec=motion_seg_sec,
            )
            dev_t = np.asarray(ppg_raw[dev]["timestamp"], dtype=np.float64)
            seg_lo = int(np.searchsorted(dev_t, t0 - max_source_gap_ms, side="left"))
            seg_hi = int(np.searchsorted(dev_t, float(t1_all[wi]) + max_source_gap_ms, side="right"))
            seg_t = dev_t[seg_lo:seg_hi]
            for ci, ch in enumerate(CHANNELS):
                seg_x = np.asarray(ppg_raw[dev][ch][seg_lo:seg_hi], dtype=np.float64)
                raw_50, valid_mask, valid_ratio = _resample_to_grid(
                    seg_x,
                    seg_t,
                    grid_ms,
                    max_gap_ms=max_source_gap_ms,
                )
                ppg[out_i, di, ci] = raw_50
                ppg_valid_mask[out_i, di, ci] = valid_mask
                ppg_valid_sample_ratio[out_i, di, ci] = valid_ratio
                if valid_ratio < min_valid_sample_ratio:
                    final_keep[out_i] = False
                peak_signal = _preprocess_for_peaks(raw_50, target_fs, preprocess_mode)
                peak_info = _ppg_peaks_and_qc(raw_50, peak_signal, fs=target_fs, valid_sample_ratio=valid_ratio)
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

    keep_idx = np.where(final_keep)[0]
    if keep_idx.size == 0:
        summary = {
            "participant": pid,
            "n_candidate_windows": int(t0_all.size),
            "n_boundary_aligned": int(boundary_keep_arr.sum()),
            "n_ppg_sample_pass": 0,
            "n_ecg_label_pass": int(ecg_keep_arr.sum()),
            "n_kept": 0,
            **overlap_info,
            "skip_reason": "no_window_after_ppg_sample_coverage",
        }
        pd.DataFrame([summary]).to_csv(out_dir / f"{dataset_name}_{pid}_summary.csv", index=False)
        print(f"[SKIP] {pid}: no windows after PPG sample coverage")
        return None

    kept_wi = prelim_idx[keep_idx]
    kept_labels = [labels[int(wi)] for wi in kept_wi]
    out_path = out_dir / f"{dataset_name}_{pid}.npz"
    cfg = {
        "dataset_name": dataset_name,
        "participant": pid,
        "source": "snowballlab/Multisite-PPG/raw_data local mirror",
        "devices": DEVICES,
        "channels": CHANNELS,
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "target_fs": target_fs,
        "target_len": target_len,
        "alignment_tolerance_sec": alignment_tolerance_sec,
        "min_valid_sample_ratio": min_valid_sample_ratio,
        "max_source_gap_ms": max_source_gap_ms,
        "preprocess_mode_for_peak_metadata": preprocess_mode,
        "motion_seg_sec": motion_seg_sec,
        "motion_percentile": motion_percentile,
        "ecg_detector": "window_neurokit2_pantompkins1985",
        "ecg_fs_inferred_hz": ecg_fs,
        "ecg_min_valid_ibi_ratio": ecg_min_valid_ibi_ratio,
        "min_ecg_valid_sample_ratio": min_ecg_valid_sample_ratio,
        "primary_label": "ecg_r_peak_times_rel_ms",
        "inclusion_rule": "raw_boundary_aligned && ecg_label_qc_pass && ppg_valid_sample_ratio>=min_valid_sample_ratio for all device/channel",
        "ppg_quality_is_metadata_only": True,
    }
    print(f"[{pid}] saving {out_path} kept={keep_idx.size}/{t0_all.size}")
    np.savez_compressed(
        out_path,
        config_json=np.array(json.dumps(cfg, ensure_ascii=False)),
        participant=np.array(pid),
        devices=np.array(DEVICES),
        channels=np.array(CHANNELS),
        t0_ms=t0_all[kept_wi],
        t1_ms=t1_all[kept_wi],
        max_start_diff_ms=np.asarray(max_start_diff_ms, dtype=np.float32)[kept_wi],
        max_end_diff_ms=np.asarray(max_end_diff_ms, dtype=np.float32)[kept_wi],
        ppg_50hz=ppg[keep_idx],
        ppg_valid_mask_50hz=ppg_valid_mask[keep_idx],
        ppg_valid_sample_ratio=ppg_valid_sample_ratio[keep_idx],
        ppg_valid_ibi_ratio=ppg_valid_ibi_ratio[keep_idx],
        ppg_sqi=ppg_sqi_arr[keep_idx],
        ppg_peak_times_rel_ms=ppg_peak_times[keep_idx],
        ppg_peak_indices_50hz=ppg_peak_indices[keep_idx],
        ppg_peak_amplitudes_raw=ppg_peak_amplitudes[keep_idx],
        ppg_ibi_ms=ppg_ibi[keep_idx],
        ppg_ibi_corrected_ms=ppg_ibi_corrected[keep_idx],
        ppg_ibi_correction_ratio=ppg_ibi_correction_ratio[keep_idx],
        ppg_rmssd_ms=ppg_rmssd[keep_idx],
        ppg_sdnn_ms=ppg_sdnn[keep_idx],
        ppg_quality_flag=ppg_quality_flag[keep_idx],
        ppg_quality_reason=ppg_quality_reason[keep_idx],
        motion_fraction=motion_fraction[keep_idx],
        motion_threshold=np.array([motion_threshold[d] for d in DEVICES], dtype=np.float32),
        ecg_r_peak_times_rel_ms=np.array([x["ecg_r_peak_times_rel_ms"] for x in kept_labels], dtype=object),
        ecg_r_peak_indices_50hz=np.array([x["ecg_r_peak_indices_50hz"] for x in kept_labels], dtype=object),
        ecg_r_peak_amplitudes_raw=np.array([x["ecg_r_peak_amplitudes_raw"] for x in kept_labels], dtype=object),
        ecg_rr_intervals_ms=np.array([x["ecg_rr_intervals_ms"] for x in kept_labels], dtype=object),
        ecg_rr_intervals_corrected_ms=np.array([x["ecg_rr_intervals_corrected_ms"] for x in kept_labels], dtype=object),
        ecg_rmssd_ms=np.array([x["ecg_rmssd_ms"] for x in kept_labels], dtype=np.float32),
        ecg_sdnn_ms=np.array([x["ecg_sdnn_ms"] for x in kept_labels], dtype=np.float32),
        ecg_valid_sample_ratio=np.array([x["ecg_valid_sample_ratio"] for x in kept_labels], dtype=np.float32),
        ecg_valid_ibi_ratio=np.array([x["ecg_valid_ibi_ratio"] for x in kept_labels], dtype=np.float32),
        ecg_ibi_correction_ratio=np.array([x["ecg_ibi_correction_ratio"] for x in kept_labels], dtype=np.float32),
        ecg_qc_pass=np.array([x["ecg_qc_pass"] for x in kept_labels], dtype=bool),
        ecg_qc_reason=np.array([x["ecg_qc_reason"] for x in kept_labels], dtype=object),
        ecg_label_qc_pass=np.array([x["ecg_label_qc_pass"] for x in kept_labels], dtype=bool),
        ecg_label_qc_reason=np.array([x["ecg_label_qc_reason"] for x in kept_labels], dtype=object),
        ecg_qrs_sqi=np.array([x["ecg_qrs_sqi"] for x in kept_labels], dtype=np.float32),
        ecg_peak_count=np.array([x["ecg_peak_count"] for x in kept_labels], dtype=np.int32),
        ecg_hr_bpm=np.array([x["ecg_hr_bpm"] for x in kept_labels], dtype=np.float32),
        ecg_rr_cv=np.array([x["ecg_rr_cv"] for x in kept_labels], dtype=np.float32),
        ecg_rpeak_detector_agreement=np.full(keep_idx.size, np.nan, dtype=np.float32),
    )

    summary = {
        "participant": pid,
        "n_candidate_windows": int(t0_all.size),
        "n_boundary_aligned": int(boundary_keep_arr.sum()),
        "n_ppg_sample_pass": int(final_keep.sum()),
        "n_ecg_label_pass": int(ecg_keep_arr.sum()),
        "n_kept": int(keep_idx.size),
        "kept_pct_of_candidate": float(100.0 * keep_idx.size / t0_all.size),
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "target_fs": target_fs,
        "target_len": target_len,
        "alignment_tolerance_sec": alignment_tolerance_sec,
        "max_source_gap_ms": max_source_gap_ms,
        "min_valid_sample_ratio": min_valid_sample_ratio,
        "mean_max_start_diff_ms": float(np.nanmean(np.asarray(max_start_diff_ms)[kept_wi])),
        "mean_max_end_diff_ms": float(np.nanmean(np.asarray(max_end_diff_ms)[kept_wi])),
        "mean_ppg_valid_sample_ratio": float(np.nanmean(ppg_valid_sample_ratio[keep_idx])),
        "mean_ppg_sqi": float(np.nanmean(ppg_sqi_arr[keep_idx])),
        "mean_ppg_ibi_correction_ratio": float(np.nanmean(ppg_ibi_correction_ratio[keep_idx])),
        "mean_motion_fraction": float(np.nanmean(motion_fraction[keep_idx])),
        "mean_ecg_valid_ibi_ratio": float(np.nanmean([x["ecg_valid_ibi_ratio"] for x in kept_labels])),
        "mean_ecg_ibi_correction_ratio": float(np.nanmean([x["ecg_ibi_correction_ratio"] for x in kept_labels])),
        "mean_ecg_qrs_sqi": float(np.nanmean([x["ecg_qrs_sqi"] for x in kept_labels])),
        "mean_ecg_hr_bpm": float(np.nanmean([x["ecg_hr_bpm"] for x in kept_labels])),
        **overlap_info,
        "skip_reason": "",
    }
    pd.DataFrame([summary]).to_csv(out_dir / f"{dataset_name}_{pid}_summary.csv", index=False)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate raw-aligned 4-device ECG-label dataset.")
    ap.add_argument("--participants", default=None)
    ap.add_argument("--exclude", default="P2,P14,P16,P17")
    ap.add_argument("--raw-root", default=str(RAW_ROOT))
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--window-sec", type=int, default=300)
    ap.add_argument("--stride-sec", type=int, default=300)
    ap.add_argument("--target-fs", type=float, default=50.0)
    ap.add_argument("--alignment-tolerance-sec", type=float, default=2.0)
    ap.add_argument("--min-valid-sample-ratio", type=float, default=0.50)
    ap.add_argument("--max-source-gap-ms", type=float, default=500.0)
    ap.add_argument("--preprocess-mode", choices=("bandpass", "raw"), default="bandpass")
    ap.add_argument("--motion-seg-sec", type=float, default=10.0)
    ap.add_argument("--motion-percentile", type=float, default=75.0)
    ap.add_argument("--ecg-min-valid-ibi-ratio", type=float, default=0.80)
    ap.add_argument("--min-ecg-valid-sample-ratio", type=float, default=0.95)
    ap.add_argument("--min-hr-bpm", type=float, default=30.0)
    ap.add_argument("--max-hr-bpm", type=float, default=200.0)
    ap.add_argument("--max-windows", type=int, default=None)
    args = ap.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (config.HEURISTIC_RESULT_ROOT / args.dataset_name).resolve()
    participants = _participant_ids(raw_root, args.participants)
    excluded = {normalize_participant_id(p) for p in args.exclude.split(",") if p.strip()}
    participants = [p for p in participants if p not in excluded]

    cfg = {
        "dataset_name": args.dataset_name,
        "raw_root": str(raw_root),
        "participants": participants,
        "excluded": sorted(excluded),
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "target_fs": args.target_fs,
        "target_len": int(round(args.window_sec * args.target_fs)),
        "alignment_tolerance_sec": args.alignment_tolerance_sec,
        "min_valid_sample_ratio": args.min_valid_sample_ratio,
        "max_source_gap_ms": args.max_source_gap_ms,
        "preprocess_mode": args.preprocess_mode,
        "motion_seg_sec": args.motion_seg_sec,
        "motion_percentile": args.motion_percentile,
        "ecg_detector": "window_neurokit2_pantompkins1985",
        "ecg_min_valid_ibi_ratio": args.ecg_min_valid_ibi_ratio,
        "min_ecg_valid_sample_ratio": args.min_ecg_valid_sample_ratio,
        "primary_label": "ecg_r_peak_times_rel_ms",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[rawaligned] participants={participants}")
    print(f"[rawaligned] out_dir={out_dir}")
    paths = []
    for pid in participants:
        try:
            path = generate_participant(
                pid,
                raw_root=raw_root,
                out_dir=out_dir,
                dataset_name=args.dataset_name,
                window_sec=args.window_sec,
                stride_sec=args.stride_sec,
                target_fs=args.target_fs,
                alignment_tolerance_sec=args.alignment_tolerance_sec,
                min_valid_sample_ratio=args.min_valid_sample_ratio,
                max_source_gap_ms=args.max_source_gap_ms,
                preprocess_mode=args.preprocess_mode,
                motion_seg_sec=args.motion_seg_sec,
                motion_percentile=args.motion_percentile,
                ecg_min_valid_ibi_ratio=args.ecg_min_valid_ibi_ratio,
                min_ecg_valid_sample_ratio=args.min_ecg_valid_sample_ratio,
                min_hr_bpm=args.min_hr_bpm,
                max_hr_bpm=args.max_hr_bpm,
                max_windows=args.max_windows,
            )
            if path is not None:
                paths.append(path)
        except Exception as exc:
            summary = {
                "participant": pid,
                "n_candidate_windows": 0,
                "n_boundary_aligned": 0,
                "n_ppg_sample_pass": 0,
                "n_ecg_label_pass": 0,
                "n_kept": 0,
                "skip_reason": f"exception:{type(exc).__name__}:{exc}",
            }
            pd.DataFrame([summary]).to_csv(out_dir / f"{args.dataset_name}_{pid}_summary.csv", index=False)
            print(f"[ERROR] {pid}: {exc!r}")

    summaries = []
    for csv_path in sorted(out_dir.glob(f"{args.dataset_name}_P*_summary.csv")):
        summaries.append(pd.read_csv(csv_path))
    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        summary_path = out_dir / f"{args.dataset_name}_summary.csv"
        combined.to_csv(summary_path, index=False)
        _write_readme(out_dir, args.dataset_name, cfg, combined)
        print(f"[SAVED] {summary_path}")
        print(f"[SAVED] {out_dir / 'README.md'}")
    print(f"[DONE] participant files={len(paths)}")


if __name__ == "__main__":
    main()
