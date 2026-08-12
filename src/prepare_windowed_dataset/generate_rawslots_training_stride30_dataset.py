"""
Generate a rawslots training_stride30 dataset.

This variant keeps the existing raw-aligned 5-min / stride-30 ECG-QC windowing
standard, but stores two PPG input views:

- rawslots: nearest raw sample assigned to a fixed 100 Hz window grid, no
  waveform interpolation, empty slots are NaN;
- resampled: linear-interpolated PPG on the same grid, for paired comparison.

No legacy field aliases are written. ECG HRV labels are explicitly named as
corrected or uncorrected.
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
_SRC_ROOT = _PKG_ROOT.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from prepare_windowed_dataset import config  # noqa: E402
from prepare_windowed_dataset.generate_rawaligned_4device_dataset import (  # noqa: E402
    CHANNELS,
    DEVICES,
    MAX_PEAKS_PER_WINDOW,
    OUTPUT_ROOT,
    RAW_ROOT,
    _accel_gravity_reference,
    _accel_mean_magnitude_window,
    _accel_motion_mean_magnitude_window,
    _boundary_offsets_ms,
    _candidate_windows,
    _ecg_fs_from_raw,
    _ecg_label_for_window,
    _load_ecg_raw,
    _load_ppg_raw,
    _object_array_cast,
    _participant_ids,
    _resample_to_grid,
)
from heuristic_baselines.algorithms import hrv  # noqa: E402


DATASET_KIND = "rawslots_training_stride30"


def _hrv_from_intervals(intervals_ms: np.ndarray) -> dict[str, float]:
    rr = np.asarray(intervals_ms, dtype=np.float64)
    rr = rr[np.isfinite(rr)]
    if rr.size < 3:
        return {
            "rmssd": float("nan"),
            "sdnn": float("nan"),
            "hr_bpm": float("nan"),
            "mean_ibi": float("nan"),
            "rr_cv": float("nan"),
        }
    diff = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan")
    sdnn = float(np.std(rr, ddof=1)) if rr.size > 1 else float("nan")
    mean_ibi = float(np.mean(rr))
    hr_bpm = float(60000.0 / mean_ibi) if mean_ibi > 0 else float("nan")
    rr_cv = float(np.std(rr, ddof=1) / mean_ibi) if rr.size > 1 and mean_ibi > 0 else float("nan")
    return {"rmssd": rmssd, "sdnn": sdnn, "hr_bpm": hr_bpm, "mean_ibi": mean_ibi, "rr_cv": rr_cv}


def _canonical_ecg_label(label: dict[str, object], *, t0: float) -> dict[str, object]:
    rel = np.asarray(label["ecg_r_peak_times_rel_ms"], dtype=np.float64)
    rr_uncorrected = np.asarray(label["ecg_rr_intervals_ms"], dtype=np.float64)
    rr_corrected = np.asarray(label["ecg_rr_intervals_corrected_ms"], dtype=np.float64)
    uncorrected_stats = _hrv_from_intervals(rr_uncorrected)
    corrected_stats = _hrv_from_intervals(rr_corrected)
    return {
        "ecg_r_peak_times_rel_ms": rel.astype(np.float32),
        "ecg_r_peak_times_abs_ms": (float(t0) + rel).astype(np.float64),
        "ecg_r_peak_indices_grid": np.asarray(label["ecg_r_peak_indices_grid"], dtype=np.int32),
        "ecg_r_peak_amplitudes_raw": np.asarray(label["ecg_r_peak_amplitudes_raw"], dtype=np.float32),
        "ecg_rr_intervals_uncorrected_ms": rr_uncorrected.astype(np.float32),
        "ecg_rmssd_uncorrected_ms": uncorrected_stats["rmssd"],
        "ecg_sdnn_uncorrected_ms": uncorrected_stats["sdnn"],
        "ecg_hr_bpm_uncorrected": uncorrected_stats["hr_bpm"],
        "ecg_mean_ibi_uncorrected_ms": uncorrected_stats["mean_ibi"],
        "ecg_rr_cv_uncorrected": uncorrected_stats["rr_cv"],
        "ecg_rr_intervals_corrected_ms": rr_corrected.astype(np.float32),
        "ecg_rmssd_corrected_ms": corrected_stats["rmssd"],
        "ecg_sdnn_corrected_ms": corrected_stats["sdnn"],
        "ecg_hr_bpm_corrected": corrected_stats["hr_bpm"],
        "ecg_mean_ibi_corrected_ms": corrected_stats["mean_ibi"],
        "ecg_rr_cv_corrected": corrected_stats["rr_cv"],
        "ecg_label_qc_pass": bool(label["ecg_label_qc_pass"]),
        "ecg_label_qc_reason": str(label["ecg_label_qc_reason"]),
        "ecg_valid_sample_ratio": float(label["ecg_valid_sample_ratio"]),
        "ecg_valid_ibi_ratio": float(label["ecg_valid_ibi_ratio"]),
        "ecg_ibi_correction_ratio": float(label["ecg_ibi_correction_ratio"]),
        "ecg_qrs_sqi": float(label["ecg_qrs_sqi"]),
        "ecg_peak_count": int(label["ecg_peak_count"]),
        "ecg_rpeak_detector_agreement": float("nan"),
    }


def _rawslots_to_grid(
    values: np.ndarray,
    times_ms: np.ndarray,
    grid_ms: np.ndarray,
    *,
    slot_tolerance_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    sig = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_ms, dtype=np.float64)
    valid = np.isfinite(sig) & np.isfinite(t)
    out_values = np.full(grid_ms.size, np.nan, dtype=np.float32)
    out_timestamps = np.full(grid_ms.size, np.nan, dtype=np.float64)
    out_mask = np.zeros(grid_ms.size, dtype=bool)
    if valid.sum() == 0:
        return out_values, out_mask, out_timestamps, 0.0

    t = t[valid]
    sig = sig[valid]
    step_ms = float(np.median(np.diff(grid_ms))) if grid_ms.size > 1 else 10.0
    slot_idx = np.rint((t - float(grid_ms[0])) / step_ms).astype(np.int64)
    in_range = (slot_idx >= 0) & (slot_idx < grid_ms.size)
    if not np.any(in_range):
        return out_values, out_mask, out_timestamps, 0.0

    t = t[in_range]
    sig = sig[in_range]
    slot_idx = slot_idx[in_range]
    dist = np.abs(t - grid_ms[slot_idx])
    close = dist <= slot_tolerance_ms
    if not np.any(close):
        return out_values, out_mask, out_timestamps, 0.0

    t = t[close]
    sig = sig[close]
    slot_idx = slot_idx[close]
    dist = dist[close]

    order = np.lexsort((dist, slot_idx))
    slot_sorted = slot_idx[order]
    first = np.r_[True, np.diff(slot_sorted) != 0]
    keep = order[first]
    chosen_slots = slot_idx[keep]
    out_values[chosen_slots] = sig[keep].astype(np.float32)
    out_timestamps[chosen_slots] = t[keep]
    out_mask[chosen_slots] = True
    return out_values, out_mask, out_timestamps, float(out_mask.mean())


def _write_readme(out_dir: Path, dataset_name: str, cfg: dict[str, object], summary: pd.DataFrame) -> None:
    total = int(summary["n_kept"].sum()) if "n_kept" in summary else 0
    text = f"""# {dataset_name}

This dataset is a rawslots variant of the raw-aligned training stride-30 set.

## Core Parameters

```json
{json.dumps(cfg, indent=2, ensure_ascii=False)}
```

## Inclusion Rule

The windowing, ECG QC, boundary alignment, and PPG sample-coverage thresholds
match the current training_stride30 generation standard. PPG sample coverage is
evaluated on `ppg_rawslot_mask` for the no-interpolation view.

## PPG Fields

| Field | Shape | Meaning |
|---|---|---|
| `ppg_window_t0_ms` | `(N,)` | Window start Unix ms |
| `ppg_window_t1_ms` | `(N,)` | Window end Unix ms |
| `ppg_grid_timestamp_ms` | `(N, L)` | Absolute 100 Hz grid timestamps |
| `ppg_rawslot_values` | `(N, D, 2, L)` | Raw PPG samples assigned to nearest grid slot; empty slots are NaN |
| `ppg_rawslot_mask` | `(N, D, 2, L)` | True where a finite raw sample is assigned |
| `ppg_rawslot_timestamp_ms` | `(N, D, 2, L)` | Absolute Unix ms timestamp of the assigned raw sample; empty slots are NaN |
| `ppg_rawslot_valid_sample_ratio` | `(N, D, 2)` | Mean rawslot mask |
| `ppg_resampled_values` | `(N, D, 2, L)` | Linear-interpolated PPG on the same grid; unsupported points are NaN |
| `ppg_resampled_mask` | `(N, D, 2, L)` | True where resampled value has nearby raw-sample support |
| `ppg_resampled_valid_sample_ratio` | `(N, D, 2)` | Mean resampled mask |

## ECG Labels

Corrected labels are the default training labels:

- `ecg_rr_intervals_corrected_ms`
- `ecg_rmssd_corrected_ms`
- `ecg_sdnn_corrected_ms`
- `ecg_hr_bpm_corrected`
- `ecg_mean_ibi_corrected_ms`
- `ecg_rr_cv_corrected`

Uncorrected labels are included for diagnosis and ablation:

- `ecg_rr_intervals_uncorrected_ms`
- `ecg_rmssd_uncorrected_ms`
- `ecg_sdnn_uncorrected_ms`
- `ecg_hr_bpm_uncorrected`
- `ecg_mean_ibi_uncorrected_ms`
- `ecg_rr_cv_uncorrected`

No legacy aliases such as `ppg_resampled`, `ecg_rmssd_ms`, or `t0_ms` are written.

## Summary

- kept windows: {total}
- participants with kept windows: {int((summary["n_kept"] > 0).sum()) if "n_kept" in summary else 0}

See `{dataset_name}_summary.csv` for per-participant counts.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def generate_participant(
    pid: str,
    *,
    raw_root: Path,
    out_dir: Path,
    dataset_name: str,
    devices: list[str],
    window_sec: int,
    stride_sec: int,
    target_fs: float,
    alignment_tolerance_sec: float,
    min_valid_sample_ratio: float,
    max_source_gap_ms: float,
    rawslot_tolerance_ms: float,
    ecg_min_valid_ibi_ratio: float,
    min_ecg_valid_sample_ratio: float,
    min_hr_bpm: float,
    max_hr_bpm: float,
    max_windows: int | None,
) -> Path | None:
    print(f"[{pid}] loading raw data")
    ppg_raw = _load_ppg_raw(raw_root, pid, devices)
    ecg_raw = _load_ecg_raw(raw_root, pid)
    target_len = int(round(window_sec * target_fs))
    sample_step_ms = 1000.0 / target_fs
    window_ms = window_sec * 1000.0
    tolerance_ms = alignment_tolerance_sec * 1000.0

    t0_all, t1_all, overlap_info = _candidate_windows(
        ppg_raw, ecg_raw, devices=devices, window_sec=window_sec, stride_sec=stride_sec
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
        return None

    ecg_fs = _ecg_fs_from_raw(ecg_raw)
    ecg_detector = "window_pantompkins1985"
    print(f"[{pid}] ECG fs={ecg_fs:.3f}, detector={ecg_detector}")

    boundary_keep: list[bool] = []
    max_start_diff_ms: list[float] = []
    max_end_diff_ms: list[float] = []
    ecg_keep_pre: list[bool] = []
    labels: list[dict[str, object]] = []
    print(f"[{pid}] screening {t0_all.size} candidate windows")
    for t0, t1 in zip(t0_all, t1_all):
        start_offsets = []
        end_offsets = []
        boundary_ok = True
        for dev in devices:
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
        labels.append(_canonical_ecg_label(label, t0=float(t0)))
        ecg_keep_pre.append(bool(label["ecg_label_qc_pass"]))

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
        return None

    n_pre = int(prelim_idx.size)
    shape = (n_pre, len(devices), len(CHANNELS), target_len)
    ppg_rawslot_values = np.full(shape, np.nan, dtype=np.float32)
    ppg_rawslot_mask = np.zeros(shape, dtype=bool)
    ppg_rawslot_timestamp_ms = np.full(shape, np.nan, dtype=np.float64)
    ppg_rawslot_valid_sample_ratio = np.zeros((n_pre, len(devices), len(CHANNELS)), dtype=np.float32)
    ppg_resampled_values = np.full(shape, np.nan, dtype=np.float32)
    ppg_resampled_mask = np.zeros(shape, dtype=bool)
    ppg_resampled_valid_sample_ratio = np.zeros((n_pre, len(devices), len(CHANNELS)), dtype=np.float32)
    ppg_grid_timestamp_ms = np.zeros((n_pre, target_len), dtype=np.float64)
    accel_mean_mag = np.zeros((n_pre, len(devices)), dtype=np.float32)
    accel_motion_mean_mag = np.zeros((n_pre, len(devices)), dtype=np.float32)
    accel_gravity_reference = {dev: _accel_gravity_reference(ppg_raw[dev]) for dev in devices}

    final_keep = np.ones(n_pre, dtype=bool)
    print(f"[{pid}] building rawslots/resampled PPG for {n_pre} ECG-valid windows")
    for out_i, wi in enumerate(prelim_idx):
        t0 = float(t0_all[wi])
        t1 = float(t1_all[wi])
        grid_ms = t0 + np.arange(target_len, dtype=np.float64) * sample_step_ms
        ppg_grid_timestamp_ms[out_i] = grid_ms
        for di, dev in enumerate(devices):
            accel_mean_mag[out_i, di] = _accel_mean_magnitude_window(ppg_raw[dev], t0=t0, t1=t1)
            accel_motion_mean_mag[out_i, di] = _accel_motion_mean_magnitude_window(
                ppg_raw[dev],
                t0=t0,
                t1=t1,
                gravity_reference=accel_gravity_reference[dev],
            )
            dev_t = np.asarray(ppg_raw[dev]["timestamp"], dtype=np.float64)
            seg_lo = int(np.searchsorted(dev_t, t0 - max_source_gap_ms, side="left"))
            seg_hi = int(np.searchsorted(dev_t, t1 + max_source_gap_ms, side="right"))
            seg_t = dev_t[seg_lo:seg_hi]
            for ci, ch in enumerate(CHANNELS):
                seg_x = np.asarray(ppg_raw[dev][ch][seg_lo:seg_hi], dtype=np.float64)
                raw_v, raw_m, raw_ts, raw_ratio = _rawslots_to_grid(
                    seg_x,
                    seg_t,
                    grid_ms,
                    slot_tolerance_ms=rawslot_tolerance_ms,
                )
                ppg_rawslot_values[out_i, di, ci] = raw_v
                ppg_rawslot_mask[out_i, di, ci] = raw_m
                ppg_rawslot_timestamp_ms[out_i, di, ci] = raw_ts
                ppg_rawslot_valid_sample_ratio[out_i, di, ci] = raw_ratio

                resampled, resampled_mask, resampled_ratio = _resample_to_grid(
                    seg_x,
                    seg_t,
                    grid_ms,
                    max_gap_ms=max_source_gap_ms,
                )
                ppg_resampled_values[out_i, di, ci] = np.where(resampled_mask, resampled, np.nan)
                ppg_resampled_mask[out_i, di, ci] = resampled_mask
                ppg_resampled_valid_sample_ratio[out_i, di, ci] = resampled_ratio
                if raw_ratio < min_valid_sample_ratio:
                    final_keep[out_i] = False

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
            "skip_reason": "no_window_after_rawslot_sample_coverage",
        }
        pd.DataFrame([summary]).to_csv(out_dir / f"{dataset_name}_{pid}_summary.csv", index=False)
        return None

    kept_wi = prelim_idx[keep_idx]
    kept_labels = [labels[int(wi)] for wi in kept_wi]
    cfg = {
        "dataset_name": dataset_name,
        "dataset_kind": DATASET_KIND,
        "participant": pid,
        "source": "snowballlab/Multisite-PPG/raw_data local mirror",
        "devices": devices,
        "channels": CHANNELS,
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "target_fs": target_fs,
        "target_len": target_len,
        "alignment_tolerance_sec": alignment_tolerance_sec,
        "min_valid_sample_ratio": min_valid_sample_ratio,
        "max_source_gap_ms": max_source_gap_ms,
        "rawslot_tolerance_ms": rawslot_tolerance_ms,
        "ecg_detector": "window_neurokit2_pantompkins1985",
        "ecg_fs_inferred_hz": ecg_fs,
        "ecg_min_valid_ibi_ratio": ecg_min_valid_ibi_ratio,
        "min_ecg_valid_sample_ratio": min_ecg_valid_sample_ratio,
        "min_hr_bpm": min_hr_bpm,
        "max_hr_bpm": max_hr_bpm,
        "primary_training_labels": ["ecg_rmssd_corrected_ms", "ecg_sdnn_corrected_ms"],
        "diagnostic_labels": ["ecg_rmssd_uncorrected_ms", "ecg_sdnn_uncorrected_ms"],
        "legacy_aliases_written": False,
    }
    out_path = out_dir / f"{dataset_name}_{pid}.npz"
    np.savez_compressed(
        out_path,
        config_json=json.dumps(cfg, ensure_ascii=False),
        participant=np.array(pid),
        devices=np.asarray(devices, dtype=object),
        channels=np.asarray(CHANNELS, dtype=object),
        target_fs=np.array(target_fs, dtype=np.float64),
        target_len=np.array(target_len, dtype=np.int32),
        window_sec=np.array(window_sec, dtype=np.int32),
        stride_sec=np.array(stride_sec, dtype=np.int32),
        ppg_window_t0_ms=t0_all[kept_wi].astype(np.float64),
        ppg_window_t1_ms=t1_all[kept_wi].astype(np.float64),
        ppg_grid_timestamp_ms=ppg_grid_timestamp_ms[keep_idx].astype(np.float64, copy=False),
        max_start_diff_ms=np.asarray(max_start_diff_ms, dtype=np.float32)[kept_wi],
        max_end_diff_ms=np.asarray(max_end_diff_ms, dtype=np.float32)[kept_wi],
        ppg_rawslot_values=ppg_rawslot_values[keep_idx].astype(np.float32, copy=False),
        ppg_rawslot_mask=ppg_rawslot_mask[keep_idx].astype(bool, copy=False),
        ppg_rawslot_timestamp_ms=ppg_rawslot_timestamp_ms[keep_idx].astype(np.float64, copy=False),
        ppg_rawslot_valid_sample_ratio=ppg_rawslot_valid_sample_ratio[keep_idx].astype(np.float32, copy=False),
        ppg_resampled_values=ppg_resampled_values[keep_idx].astype(np.float32, copy=False),
        ppg_resampled_mask=ppg_resampled_mask[keep_idx].astype(bool, copy=False),
        ppg_resampled_valid_sample_ratio=ppg_resampled_valid_sample_ratio[keep_idx].astype(np.float32, copy=False),
        accel_mean_mag=accel_mean_mag[keep_idx].astype(np.float32, copy=False),
        accel_motion_mean_mag=accel_motion_mean_mag[keep_idx].astype(np.float32, copy=False),
        ecg_r_peak_times_rel_ms=_object_array_cast([x["ecg_r_peak_times_rel_ms"] for x in kept_labels], np.float32),
        ecg_r_peak_times_abs_ms=_object_array_cast([x["ecg_r_peak_times_abs_ms"] for x in kept_labels], np.float64),
        ecg_r_peak_indices_grid=_object_array_cast([x["ecg_r_peak_indices_grid"] for x in kept_labels], np.int32),
        ecg_r_peak_amplitudes_raw=_object_array_cast([x["ecg_r_peak_amplitudes_raw"] for x in kept_labels], np.float32),
        ecg_rr_intervals_uncorrected_ms=_object_array_cast([x["ecg_rr_intervals_uncorrected_ms"] for x in kept_labels], np.float32),
        ecg_rmssd_uncorrected_ms=np.array([x["ecg_rmssd_uncorrected_ms"] for x in kept_labels], dtype=np.float32),
        ecg_sdnn_uncorrected_ms=np.array([x["ecg_sdnn_uncorrected_ms"] for x in kept_labels], dtype=np.float32),
        ecg_hr_bpm_uncorrected=np.array([x["ecg_hr_bpm_uncorrected"] for x in kept_labels], dtype=np.float32),
        ecg_mean_ibi_uncorrected_ms=np.array([x["ecg_mean_ibi_uncorrected_ms"] for x in kept_labels], dtype=np.float32),
        ecg_rr_cv_uncorrected=np.array([x["ecg_rr_cv_uncorrected"] for x in kept_labels], dtype=np.float32),
        ecg_rr_intervals_corrected_ms=_object_array_cast([x["ecg_rr_intervals_corrected_ms"] for x in kept_labels], np.float32),
        ecg_rmssd_corrected_ms=np.array([x["ecg_rmssd_corrected_ms"] for x in kept_labels], dtype=np.float32),
        ecg_sdnn_corrected_ms=np.array([x["ecg_sdnn_corrected_ms"] for x in kept_labels], dtype=np.float32),
        ecg_hr_bpm_corrected=np.array([x["ecg_hr_bpm_corrected"] for x in kept_labels], dtype=np.float32),
        ecg_mean_ibi_corrected_ms=np.array([x["ecg_mean_ibi_corrected_ms"] for x in kept_labels], dtype=np.float32),
        ecg_rr_cv_corrected=np.array([x["ecg_rr_cv_corrected"] for x in kept_labels], dtype=np.float32),
        ecg_label_qc_pass=np.array([x["ecg_label_qc_pass"] for x in kept_labels], dtype=bool),
        ecg_label_qc_reason=np.array([x["ecg_label_qc_reason"] for x in kept_labels], dtype=object),
        ecg_valid_sample_ratio=np.array([x["ecg_valid_sample_ratio"] for x in kept_labels], dtype=np.float32),
        ecg_valid_ibi_ratio=np.array([x["ecg_valid_ibi_ratio"] for x in kept_labels], dtype=np.float32),
        ecg_ibi_correction_ratio=np.array([x["ecg_ibi_correction_ratio"] for x in kept_labels], dtype=np.float32),
        ecg_qrs_sqi=np.array([x["ecg_qrs_sqi"] for x in kept_labels], dtype=np.float32),
        ecg_peak_count=np.array([x["ecg_peak_count"] for x in kept_labels], dtype=np.int32),
        ecg_rpeak_detector_agreement=np.full(keep_idx.size, np.nan, dtype=np.float32),
    )

    summary = {
        "participant": pid,
        "n_candidate_windows": int(t0_all.size),
        "n_boundary_aligned": int(boundary_keep_arr.sum()),
        "n_ecg_label_pass": int(ecg_keep_arr.sum()),
        "n_ppg_sample_pass": int(final_keep.sum()),
        "n_kept": int(keep_idx.size),
        "rawslot_valid_sample_ratio_mean": float(np.nanmean(ppg_rawslot_valid_sample_ratio[keep_idx])),
        "resampled_valid_sample_ratio_mean": float(np.nanmean(ppg_resampled_valid_sample_ratio[keep_idx])),
        "ecg_rmssd_corrected_mean": float(np.nanmean([x["ecg_rmssd_corrected_ms"] for x in kept_labels])),
        "ecg_sdnn_corrected_mean": float(np.nanmean([x["ecg_sdnn_corrected_ms"] for x in kept_labels])),
        **overlap_info,
        "skip_reason": "",
    }
    pd.DataFrame([summary]).to_csv(out_dir / f"{dataset_name}_{pid}_summary.csv", index=False)
    print(f"[SAVED] {out_path} windows={keep_idx.size}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate rawslots training_stride30 dataset.")
    ap.add_argument("--participants", default=None)
    ap.add_argument("--exclude", default="P2,P14,P16,P17")
    ap.add_argument("--devices", default=",".join(DEVICES))
    ap.add_argument("--raw-root", default=str(RAW_ROOT))
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--window-sec", type=int, default=300)
    ap.add_argument("--stride-sec", type=int, default=30)
    ap.add_argument("--target-fs", type=float, default=100.0)
    ap.add_argument("--alignment-tolerance-sec", type=float, default=2.0)
    ap.add_argument("--min-valid-sample-ratio", type=float, default=0.90)
    ap.add_argument("--max-source-gap-ms", type=float, default=500.0)
    ap.add_argument("--rawslot-tolerance-ms", type=float, default=5.0)
    ap.add_argument("--ecg-min-valid-ibi-ratio", type=float, default=1.00)
    ap.add_argument("--min-ecg-valid-sample-ratio", type=float, default=1.0)
    ap.add_argument("--min-hr-bpm", type=float, default=30.0)
    ap.add_argument("--max-hr-bpm", type=float, default=200.0)
    ap.add_argument("--max-windows", type=int, default=None)
    args = ap.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (OUTPUT_ROOT / args.dataset_name).resolve()
    participants = _participant_ids(raw_root, args.participants)
    excluded = {config.normalize_participant_id(p) for p in args.exclude.split(",") if p.strip()}
    participants = [p for p in participants if p not in excluded]
    device_lookup = {d.lower(): d for d in DEVICES}
    devices = [device_lookup.get(d.strip().lower(), d.strip()) for d in args.devices.split(",") if d.strip()]
    unknown_devices = sorted(set(devices) - set(DEVICES))
    if unknown_devices:
        raise SystemExit(f"Unknown devices: {unknown_devices}. Valid devices: {DEVICES}")

    cfg = {
        "dataset_name": args.dataset_name,
        "dataset_kind": DATASET_KIND,
        "raw_root": str(raw_root),
        "participants": participants,
        "excluded": sorted(excluded),
        "devices": devices,
        "channels": list(CHANNELS),
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "target_fs": args.target_fs,
        "target_len": int(round(args.window_sec * args.target_fs)),
        "alignment_tolerance_sec": args.alignment_tolerance_sec,
        "min_valid_sample_ratio": args.min_valid_sample_ratio,
        "max_source_gap_ms": args.max_source_gap_ms,
        "rawslot_tolerance_ms": args.rawslot_tolerance_ms,
        "ecg_detector": "window_neurokit2_pantompkins1985",
        "ecg_min_valid_ibi_ratio": args.ecg_min_valid_ibi_ratio,
        "min_ecg_valid_sample_ratio": args.min_ecg_valid_sample_ratio,
        "primary_training_labels": ["ecg_rmssd_corrected_ms", "ecg_sdnn_corrected_ms"],
        "legacy_aliases_written": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[rawslots] participants={participants}")
    print(f"[rawslots] devices={devices}")
    print(f"[rawslots] out_dir={out_dir}")
    paths = []
    for pid in participants:
        out_path = out_dir / f"{args.dataset_name}_{pid}.npz"
        if out_path.exists():
            paths.append(out_path)
            print(f"[SKIP] {pid}: already exists")
            continue
        try:
            path = generate_participant(
                pid,
                raw_root=raw_root,
                out_dir=out_dir,
                dataset_name=args.dataset_name,
                devices=devices,
                window_sec=args.window_sec,
                stride_sec=args.stride_sec,
                target_fs=args.target_fs,
                alignment_tolerance_sec=args.alignment_tolerance_sec,
                min_valid_sample_ratio=args.min_valid_sample_ratio,
                max_source_gap_ms=args.max_source_gap_ms,
                rawslot_tolerance_ms=args.rawslot_tolerance_ms,
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

    summaries = [pd.read_csv(p) for p in sorted(out_dir.glob(f"{args.dataset_name}_P*_summary.csv"))]
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
