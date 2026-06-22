"""
Generate a four-device synced 5-min dataset for HRV/peak-label modeling.

This script is intentionally separate from the heuristic HRV evaluation
pipeline. Its job is to create a training/evaluation dataset with:

  - one common 5-min window shared by Earring/Ring/Necklace/Watch
  - uniformly resampled PPG at 50 Hz, length 15000
  - ECG R-peak location arrays as the primary ground-truth label
  - derived ECG RR/RMSSD/SDNN and ECG/PPG QC metadata
  - raw peak amplitudes saved as arrays for later label experiments

Default choices match the current project decision:
  - target_len = 15000 (300 s * 50 Hz)
  - device alignment tolerance = 2 s
  - minimum valid sample ratio = 0.50
  - unified preprocessing mode = bandpass for peak/QC baselines

The model can later experiment with different label representations derived
from the saved ECG peak locations: peak times, RR intervals, peak-probability
sequences, or scalar HRV metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import config  # noqa: E402
from algorithms import hrv  # noqa: E402
from algorithms.sqa import ppg_sqi  # noqa: E402
from io_utils import merged_windows_npz, normalize_participant_id  # noqa: E402
from preprocess import bandpass_filter  # noqa: E402

DEVICES = ("Earring", "Ring", "Necklace", "Watch")
CHANNELS = ("ppg_green", "ppg_ir")
DEFAULT_MIN_PPG_SQI = 0.40
DEFAULT_MAX_IBI_CORRECTION_RATIO = 0.20
DEFAULT_MAX_PPG_IBI_CORRECTION_RATIO = 0.50
DEFAULT_MAX_RMSSD_MS = 200.0
DEFAULT_MAX_MOTION_FRACTION = 0.50


def _participant_ids(raw: str | None) -> list[str]:
    if raw:
        return [normalize_participant_id(p) for p in raw.split(",") if p.strip()]
    root = config.HEURISTIC_WINDOWS_ROOT
    return sorted((p.name for p in root.iterdir() if p.is_dir()), key=lambda x: int(x[1:]))


def _load_npzs(pid: str) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for dev in DEVICES:
        path = merged_windows_npz(config.HEURISTIC_WINDOWS_ROOT, pid, dev)
        if not path.is_file():
            return {}
        with np.load(path, allow_pickle=True) as z:
            out[dev] = {k: np.asarray(z[k]) for k in z.files}
    return out


def _overlap_ms(a0: float, a1: float, b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.minimum(a1, b1) - np.maximum(a0, b0))


def _match_overlapping_windows(
    data: dict[str, dict[str, np.ndarray]],
    *,
    window_ms: float,
    min_overlap_ratio: float,
) -> list[dict[str, int | float]]:
    """Use Earring as target windows and find best-overlap source windows for all devices."""
    ref = DEVICES[0]
    rows: list[dict[str, int | float]] = []
    min_overlap_ms = window_ms * min_overlap_ratio
    t0_all = {dev: np.asarray(data[dev]["t0_ms"], dtype=float) for dev in DEVICES}
    t1_all = {dev: np.asarray(data[dev]["t1_ms"], dtype=float) for dev in DEVICES}

    for ref_idx, target_t0 in enumerate(t0_all[ref]):
        target_t1 = target_t0 + window_ms
        match: dict[str, int | float] = {
            "target_t0_ms": float(target_t0),
            "target_t1_ms": float(target_t1),
            ref: ref_idx,
        }
        ok = True
        for dev in DEVICES[1:]:
            overlaps = _overlap_ms(target_t0, target_t1, t0_all[dev], t1_all[dev])
            idx = int(np.argmax(overlaps))
            if float(overlaps[idx]) < min_overlap_ms:
                ok = False
                break
            match[dev] = idx
        if ok:
            rows.append(match)
    return rows


def _resample_to_grid(
    values: np.ndarray,
    times_ms: np.ndarray,
    grid_ms: np.ndarray,
    *,
    max_gap_ms: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Interpolate source samples onto the shared absolute time grid."""
    sig = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_ms, dtype=np.float64)
    valid = np.isfinite(sig) & np.isfinite(t)
    if valid.sum() < 2:
        empty = np.zeros(len(grid_ms), dtype=np.float32)
        return empty, np.zeros(len(grid_ms), dtype=bool), 0.0

    t = t[valid]
    sig = sig[valid]
    order = np.argsort(t)
    t = t[order]
    sig = sig[order]
    unique = np.r_[True, np.diff(t) > 0]
    t = t[unique]
    sig = sig[unique]
    if t.size < 2:
        empty = np.zeros(len(grid_ms), dtype=np.float32)
        return empty, np.zeros(len(grid_ms), dtype=bool), 0.0

    y = np.interp(grid_ms, t, sig, left=np.nan, right=np.nan)
    nearest_idx = np.searchsorted(t, grid_ms)
    left_idx = np.clip(nearest_idx - 1, 0, t.size - 1)
    right_idx = np.clip(nearest_idx, 0, t.size - 1)
    nearest_dist = np.minimum(np.abs(grid_ms - t[left_idx]), np.abs(grid_ms - t[right_idx]))
    mask = np.isfinite(y) & (nearest_dist <= max_gap_ms)
    valid_ratio = float(mask.mean()) if mask.size else 0.0
    y = np.where(mask, y, 0.0)
    return y.astype(np.float32), mask.astype(bool), valid_ratio


def _preprocess_for_peaks(x_50hz: np.ndarray, fs: float, mode: str) -> np.ndarray:
    sig = np.asarray(x_50hz, dtype=np.float64)
    if mode == "raw":
        return sig
    if mode == "bandpass":
        return bandpass_filter(sig, 0.7, 3.5, fs)
    raise ValueError(f"Unknown preprocess mode: {mode}")


def _motion_fraction(
    dev_data: dict[str, np.ndarray],
    idx: int,
    threshold: float,
    *,
    seg_sec: float,
) -> float:
    fs = float(np.asarray(dev_data["ppg_fs"]).item()) if "ppg_fs" in dev_data else 100.0
    seg_n = max(1, int(round(seg_sec * fs)))
    ax = np.asarray(dev_data["accel_x"][idx], dtype=np.float64)
    ay = np.asarray(dev_data["accel_y"][idx], dtype=np.float64)
    az = np.asarray(dev_data["accel_z"][idx], dtype=np.float64)
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    n_seg = len(mag) // seg_n
    if n_seg == 0:
        stds = np.array([float(np.std(mag))])
    else:
        stds = np.array([float(np.std(mag[i * seg_n:(i + 1) * seg_n])) for i in range(n_seg)])
    return float(np.mean(stds > threshold))


def _motion_thresholds(data: dict[str, dict[str, np.ndarray]], percentile: float, seg_sec: float) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for dev in DEVICES:
        d = data[dev]
        fs = float(np.asarray(d["ppg_fs"]).item()) if "ppg_fs" in d else 100.0
        seg_n = max(1, int(round(seg_sec * fs)))
        vals: list[float] = []
        for i in range(len(d["t0_ms"])):
            mag = np.sqrt(d["accel_x"][i] ** 2 + d["accel_y"][i] ** 2 + d["accel_z"][i] ** 2)
            n_seg = len(mag) // seg_n
            if n_seg == 0:
                vals.append(float(np.std(mag)))
            else:
                vals.extend(float(np.std(mag[j * seg_n:(j + 1) * seg_n])) for j in range(n_seg))
        thresholds[dev] = float(np.percentile(vals, percentile)) if vals else float("nan")
    return thresholds


def _ecg_label_and_qc(
    dev_data: dict[str, np.ndarray],
    idx: int,
    *,
    source_t0_ms: float,
    target_t0_ms: float,
    window_ms: float,
    target_fs: float,
    ibi_min_ms: float,
    ibi_max_ms: float,
    min_valid_ibi_ratio: float,
) -> dict[str, object]:
    n_rr = int(dev_data["n_rr"][idx])
    rr = np.asarray(dev_data["rr_intervals_ms"][idx][:n_rr], dtype=np.float64)
    r_samples = np.asarray(dev_data["r_peak_samples"][idx][:n_rr + 1], dtype=np.int64)
    ecg = np.asarray(dev_data["ecg"][idx], dtype=np.float64)
    ecg_valid_len = int(dev_data.get("ecg_valid_len", np.array([len(ecg)]))[idx])
    ecg_valid_sample_ratio = float(min(ecg_valid_len, len(ecg)) / len(ecg)) if len(ecg) else 0.0

    if rr.size and r_samples.size >= 2:
        duration_s = float(np.nansum(rr) / 1000.0)
        sample_span = int(r_samples[min(len(r_samples) - 1, rr.size)] - r_samples[0])
        ecg_fs = sample_span / duration_s if duration_s > 0 and sample_span > 0 else 130.0
    else:
        ecg_fs = 130.0

    valid_rr = (rr >= ibi_min_ms) & (rr <= ibi_max_ms)
    ecg_valid_ibi_ratio = float(valid_rr.mean()) if rr.size else 0.0
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

    r_peak_times_rel_ms = (r_samples.astype(np.float64) / ecg_fs) * 1000.0
    r_peak_times_rel_ms = r_peak_times_rel_ms + (source_t0_ms - target_t0_ms)
    in_window = (r_peak_times_rel_ms >= 0.0) & (r_peak_times_rel_ms <= window_ms)
    r_peak_times_rel_ms = r_peak_times_rel_ms[in_window]
    r_peak_indices_50hz = np.rint(r_peak_times_rel_ms / 1000.0 * target_fs).astype(np.int32)

    amp = []
    r_samples_in_window = r_samples[in_window]
    for s in r_samples_in_window:
        if 0 <= int(s) < len(ecg):
            amp.append(float(ecg[int(s)]))
    amp_arr = np.asarray(amp, dtype=np.float32)
    peak_count = int(len(r_peak_times_rel_ms))
    qrs_sqi = _simple_ecg_qrs_sqi(ecg, r_samples_in_window, ecg_fs)

    qc_pass = bool(
        ecg_valid_sample_ratio >= 0.50
        and ecg_valid_ibi_ratio >= min_valid_ibi_ratio
        and (not np.isfinite(corr_ratio) or corr_ratio <= DEFAULT_MAX_IBI_CORRECTION_RATIO)
        and (not np.isfinite(hr_bpm) or (30.0 <= hr_bpm <= 200.0))
        and (not np.isfinite(rmssd) or rmssd <= DEFAULT_MAX_RMSSD_MS)
        and np.isfinite(rmssd)
        and peak_count >= 3
    )
    reason = "ok"
    if not qc_pass:
        reasons = []
        if ecg_valid_sample_ratio < 0.50:
            reasons.append("low_ecg_sample_ratio")
        if ecg_valid_ibi_ratio < min_valid_ibi_ratio:
            reasons.append("low_ecg_valid_ibi_ratio")
        if np.isfinite(corr_ratio) and corr_ratio > DEFAULT_MAX_IBI_CORRECTION_RATIO:
            reasons.append("high_ecg_ibi_correction_ratio")
        if np.isfinite(hr_bpm) and not (30.0 <= hr_bpm <= 200.0):
            reasons.append("ecg_hr_out_of_range")
        if np.isfinite(rmssd) and rmssd > DEFAULT_MAX_RMSSD_MS:
            reasons.append("ecg_rmssd_too_high")
        if not np.isfinite(rmssd):
            reasons.append("invalid_ecg_hrv")
        if peak_count < 3:
            reasons.append("too_few_ecg_peaks")
        reason = ";".join(reasons)

    return {
        "ecg_r_peak_times_rel_ms": r_peak_times_rel_ms.astype(np.float32),
        "ecg_r_peak_indices_50hz": r_peak_indices_50hz,
        "ecg_r_peak_amplitudes_raw": amp_arr,
        "ecg_rr_intervals_ms": rr.astype(np.float32),
        "ecg_rr_intervals_corrected_ms": corrected.astype(np.float32),
        "ecg_rmssd_ms": rmssd,
        "ecg_sdnn_ms": sdnn,
        "ecg_valid_sample_ratio": ecg_valid_sample_ratio,
        "ecg_valid_ibi_ratio": ecg_valid_ibi_ratio,
        "ecg_ibi_correction_ratio": corr_ratio,
        "ecg_qc_pass": qc_pass,
        "ecg_qc_reason": reason,
        "ecg_label_qc_pass": qc_pass,
        "ecg_label_qc_reason": reason,
        "ecg_qrs_sqi": qrs_sqi,
        "ecg_peak_count": peak_count,
        "ecg_hr_bpm": hr_bpm,
        "ecg_rr_cv": rr_cv,
    }


def _simple_ecg_qrs_sqi(ecg: np.ndarray, r_samples: np.ndarray, fs: float) -> float:
    """Simple raw-ECG QRS quality score from local R-peak prominence consistency."""
    sig = np.asarray(ecg, dtype=np.float64)
    samples = np.asarray(r_samples, dtype=np.int64)
    samples = samples[(samples >= 0) & (samples < sig.size)]
    if sig.size == 0 or samples.size < 3 or not np.isfinite(fs) or fs <= 0:
        return float("nan")

    local_half = max(3, int(round(0.08 * fs)))
    noise_half = max(local_half + 1, int(round(0.30 * fs)))
    scores = []
    for s in samples:
        lo = max(0, int(s) - noise_half)
        hi = min(sig.size, int(s) + noise_half + 1)
        local_lo = max(0, int(s) - local_half)
        local_hi = min(sig.size, int(s) + local_half + 1)
        noise_win = sig[lo:hi]
        local_win = sig[local_lo:local_hi]
        if noise_win.size < 5 or local_win.size < 3:
            continue
        baseline = float(np.nanmedian(noise_win))
        mad = float(np.nanmedian(np.abs(noise_win - baseline))) * 1.4826
        local_peak = float(np.nanmax(np.abs(local_win - baseline)))
        if not np.isfinite(local_peak) or not np.isfinite(mad):
            continue
        scores.append(local_peak / (mad + 1e-6))
    if len(scores) < 3:
        return float("nan")

    scores_arr = np.asarray(scores, dtype=np.float64)
    snr_score = float(np.clip(np.nanmedian(scores_arr) / 10.0, 0.0, 1.0))
    consistency = 1.0 - float(
        np.clip(np.nanstd(scores_arr) / (np.nanmean(scores_arr) + 1e-6), 0.0, 1.0)
    )
    return float(np.clip(0.7 * snr_score + 0.3 * consistency, 0.0, 1.0))


def _ppg_peaks_and_qc(
    raw_50hz: np.ndarray,
    peak_signal: np.ndarray,
    *,
    fs: float,
    valid_sample_ratio: float,
) -> dict[str, object]:
    peaks = hrv.detect_ppg_peaks(peak_signal, fs)
    peak_times = peaks.astype(np.float64) / fs * 1000.0
    amps = raw_50hz[peaks] if peaks.size else np.array([], dtype=np.float32)
    if peaks.size >= 2:
        ibi = np.diff(peaks) / fs * 1000.0
        valid_ibi = (ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)
        valid_ibi_ratio = float(np.mean(valid_ibi))
        ibi_valid = ibi[valid_ibi]
    else:
        ibi = np.array([], dtype=np.float64)
        ibi_valid = np.array([], dtype=np.float64)
        valid_ibi_ratio = 0.0
    if ibi_valid.size >= 3:
        corrected, corr_ratio = hrv._correct_ibi_artifacts_with_ratio(ibi_valid)
        diff = np.diff(corrected)
        rmssd = float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan")
        sdnn = float(np.std(corrected, ddof=1)) if corrected.size > 1 else float("nan")
    else:
        corrected = np.array([], dtype=np.float64)
        corr_ratio = float("nan")
        rmssd = float("nan")
        sdnn = float("nan")
    try:
        sqi = float(ppg_sqi(peak_signal, fs, peaks))
    except Exception:
        sqi = float("nan")
    return {
        "ppg_peak_times_rel_ms": peak_times.astype(np.float32),
        "ppg_peak_indices_50hz": peaks.astype(np.int32),
        "ppg_peak_amplitudes_raw": np.asarray(amps, dtype=np.float32),
        "ppg_ibi_ms": ibi.astype(np.float32),
        "ppg_ibi_corrected_ms": corrected.astype(np.float32),
        "ppg_ibi_correction_ratio": corr_ratio,
        "ppg_rmssd_ms": rmssd,
        "ppg_sdnn_ms": sdnn,
        "ppg_valid_sample_ratio": valid_sample_ratio,
        "ppg_valid_ibi_ratio": valid_ibi_ratio,
        "ppg_sqi": sqi,
    }


def _ppg_channel_qc(
    *,
    valid_sample_ratio: float,
    valid_ibi_ratio: float,
    correction_ratio: float,
    rmssd_ms: float,
    sqi: float,
    motion_fraction: float,
    min_valid_sample_ratio: float,
    min_valid_ibi_ratio: float,
    min_sqi: float,
    max_motion_fraction: float,
) -> tuple[bool, str]:
    reasons = []
    if valid_sample_ratio < min_valid_sample_ratio:
        reasons.append("low_ppg_sample_ratio")
    if valid_ibi_ratio < min_valid_ibi_ratio:
        reasons.append("low_ppg_valid_ibi_ratio")
    if np.isfinite(correction_ratio) and correction_ratio > DEFAULT_MAX_PPG_IBI_CORRECTION_RATIO:
        reasons.append("high_ppg_ibi_correction_ratio")
    if np.isfinite(rmssd_ms) and rmssd_ms > DEFAULT_MAX_RMSSD_MS:
        reasons.append("ppg_rmssd_too_high")
    if not np.isfinite(sqi) or sqi < min_sqi:
        reasons.append("low_ppg_sqi")
    if motion_fraction >= max_motion_fraction:
        reasons.append("motion_artifact")
    if not np.isfinite(rmssd_ms):
        reasons.append("invalid_ppg_hrv")
    return (len(reasons) == 0, "ok" if not reasons else ";".join(reasons))


def generate_participant(
    pid: str,
    *,
    out_dir: Path,
    window_sec: int,
    target_fs: float,
    align_tolerance_sec: float,
    min_valid_sample_ratio: float,
    preprocess_mode: str,
    motion_seg_sec: float,
    motion_percentile: float,
    ecg_min_valid_ibi_ratio: float,
    max_windows: int | None,
) -> Path | None:
    data = _load_npzs(pid)
    if not data:
        print(f"[SKIP] {pid}: missing one or more device files")
        return None

    target_len = int(round(window_sec * target_fs))
    window_ms = float(window_sec * 1000.0)
    sample_step_ms = 1000.0 / target_fs
    max_gap_ms = max(float(align_tolerance_sec * 1000.0), sample_step_ms * 2.0)
    matches = _match_overlapping_windows(
        data,
        window_ms=window_ms,
        min_overlap_ratio=min_valid_sample_ratio,
    )
    if max_windows is not None:
        matches = matches[:max_windows]
    if not matches:
        print(f"[SKIP] {pid}: no four-device synced windows")
        return None

    motion_threshold = _motion_thresholds(data, motion_percentile, motion_seg_sec)
    ppg = np.zeros((len(matches), len(DEVICES), len(CHANNELS), target_len), dtype=np.float32)
    ppg_valid_mask = np.zeros((len(matches), len(DEVICES), len(CHANNELS), target_len), dtype=bool)
    ppg_valid_sample_ratio = np.zeros((len(matches), len(DEVICES), len(CHANNELS)), dtype=np.float32)
    ppg_valid_ibi_ratio = np.zeros_like(ppg_valid_sample_ratio)
    ppg_sqi_arr = np.zeros_like(ppg_valid_sample_ratio)
    motion_fraction = np.zeros((len(matches), len(DEVICES)), dtype=np.float32)
    source_indices = np.zeros((len(matches), len(DEVICES)), dtype=np.int32)
    source_start_offset_sec = np.zeros((len(matches), len(DEVICES)), dtype=np.float32)
    t0_ms = np.zeros(len(matches), dtype=np.float64)
    t1_ms = np.zeros(len(matches), dtype=np.float64)

    ppg_peak_times = np.empty((len(matches), len(DEVICES), len(CHANNELS)), dtype=object)
    ppg_peak_indices = np.empty_like(ppg_peak_times)
    ppg_peak_amplitudes = np.empty_like(ppg_peak_times)
    ppg_ibi = np.empty_like(ppg_peak_times)
    ppg_ibi_corrected = np.empty_like(ppg_peak_times)
    ppg_ibi_correction_ratio = np.full((len(matches), len(DEVICES), len(CHANNELS)), np.nan, dtype=np.float32)
    ppg_rmssd = np.full_like(ppg_ibi_correction_ratio, np.nan)
    ppg_sdnn = np.full_like(ppg_ibi_correction_ratio, np.nan)
    ppg_channel_qc_pass = np.zeros((len(matches), len(DEVICES), len(CHANNELS)), dtype=bool)
    ppg_device_qc_pass = np.zeros((len(matches), len(DEVICES)), dtype=bool)
    ppg_qc_reason = np.empty((len(matches), len(DEVICES), len(CHANNELS)), dtype=object)

    ecg_peak_times = np.empty(len(matches), dtype=object)
    ecg_peak_indices = np.empty(len(matches), dtype=object)
    ecg_peak_amplitudes = np.empty(len(matches), dtype=object)
    ecg_rr = np.empty(len(matches), dtype=object)
    ecg_rr_corrected = np.empty(len(matches), dtype=object)
    ecg_rmssd = np.full(len(matches), np.nan, dtype=np.float32)
    ecg_sdnn = np.full(len(matches), np.nan, dtype=np.float32)
    ecg_valid_sample_ratio = np.zeros(len(matches), dtype=np.float32)
    ecg_valid_ibi_ratio = np.zeros(len(matches), dtype=np.float32)
    ecg_ibi_correction_ratio = np.full(len(matches), np.nan, dtype=np.float32)
    ecg_qc_pass = np.zeros(len(matches), dtype=bool)
    ecg_qc_reason = np.empty(len(matches), dtype=object)
    ecg_label_qc_pass = np.zeros(len(matches), dtype=bool)
    ecg_label_qc_reason = np.empty(len(matches), dtype=object)
    ecg_qrs_sqi = np.full(len(matches), np.nan, dtype=np.float32)
    ecg_peak_count = np.zeros(len(matches), dtype=np.int32)
    ecg_hr_bpm = np.full(len(matches), np.nan, dtype=np.float32)
    ecg_rr_cv = np.full(len(matches), np.nan, dtype=np.float32)
    ecg_rpeak_detector_agreement = np.full(len(matches), np.nan, dtype=np.float32)
    window_qc_pass_strict = np.zeros(len(matches), dtype=bool)
    window_qc_reason_strict = np.empty(len(matches), dtype=object)

    keep = np.ones(len(matches), dtype=bool)

    for wi, idxs in enumerate(matches):
        target_t0 = float(idxs["target_t0_ms"])
        target_t1 = float(idxs["target_t1_ms"])
        grid_ms = target_t0 + np.arange(target_len, dtype=np.float64) * sample_step_ms
        ref_idx = int(idxs[DEVICES[0]])
        t0_ms[wi] = target_t0
        t1_ms[wi] = target_t1
        for di, dev in enumerate(DEVICES):
            dev_data = data[dev]
            idx = int(idxs[dev])
            source_indices[wi, di] = idx
            source_start_offset_sec[wi, di] = float((dev_data["t0_ms"][idx] - target_t0) / 1000.0)
            motion_fraction[wi, di] = _motion_fraction(
                dev_data, idx, motion_threshold[dev], seg_sec=motion_seg_sec
            )
            for ci, ch in enumerate(CHANNELS):
                raw_50, valid_mask, valid_ratio = _resample_to_grid(
                    dev_data[ch][idx],
                    dev_data["ppg_t_ms"][idx],
                    grid_ms,
                    max_gap_ms=max_gap_ms,
                )
                ppg[wi, di, ci] = raw_50
                ppg_valid_mask[wi, di, ci] = valid_mask
                ppg_valid_sample_ratio[wi, di, ci] = valid_ratio
                if valid_ratio < min_valid_sample_ratio:
                    keep[wi] = False
                peak_signal = _preprocess_for_peaks(raw_50, target_fs, preprocess_mode)
                peak_info = _ppg_peaks_and_qc(
                    raw_50,
                    peak_signal,
                    fs=target_fs,
                    valid_sample_ratio=valid_ratio,
                )
                ppg_peak_times[wi, di, ci] = peak_info["ppg_peak_times_rel_ms"]
                ppg_peak_indices[wi, di, ci] = peak_info["ppg_peak_indices_50hz"]
                ppg_peak_amplitudes[wi, di, ci] = peak_info["ppg_peak_amplitudes_raw"]
                ppg_ibi[wi, di, ci] = peak_info["ppg_ibi_ms"]
                ppg_ibi_corrected[wi, di, ci] = peak_info["ppg_ibi_corrected_ms"]
                ppg_ibi_correction_ratio[wi, di, ci] = peak_info["ppg_ibi_correction_ratio"]
                ppg_rmssd[wi, di, ci] = peak_info["ppg_rmssd_ms"]
                ppg_sdnn[wi, di, ci] = peak_info["ppg_sdnn_ms"]
                ppg_valid_ibi_ratio[wi, di, ci] = peak_info["ppg_valid_ibi_ratio"]
                ppg_sqi_arr[wi, di, ci] = peak_info["ppg_sqi"]
                chan_pass, chan_reason = _ppg_channel_qc(
                    valid_sample_ratio=valid_ratio,
                    valid_ibi_ratio=peak_info["ppg_valid_ibi_ratio"],
                    correction_ratio=peak_info["ppg_ibi_correction_ratio"],
                    rmssd_ms=peak_info["ppg_rmssd_ms"],
                    sqi=peak_info["ppg_sqi"],
                    motion_fraction=motion_fraction[wi, di],
                    min_valid_sample_ratio=min_valid_sample_ratio,
                    min_valid_ibi_ratio=ecg_min_valid_ibi_ratio,
                    min_sqi=DEFAULT_MIN_PPG_SQI,
                    max_motion_fraction=DEFAULT_MAX_MOTION_FRACTION,
                )
                ppg_channel_qc_pass[wi, di, ci] = chan_pass
                ppg_qc_reason[wi, di, ci] = chan_reason

        # ECG is shared across device files; use Earring as the canonical source
        # after four-device alignment.
        label = _ecg_label_and_qc(
            data[DEVICES[0]],
            ref_idx,
            source_t0_ms=float(data[DEVICES[0]]["t0_ms"][ref_idx]),
            target_t0_ms=target_t0,
            window_ms=window_ms,
            target_fs=target_fs,
            ibi_min_ms=hrv.IBI_MIN_MS,
            ibi_max_ms=hrv.IBI_MAX_MS,
            min_valid_ibi_ratio=ecg_min_valid_ibi_ratio,
        )
        ecg_peak_times[wi] = label["ecg_r_peak_times_rel_ms"]
        ecg_peak_indices[wi] = label["ecg_r_peak_indices_50hz"]
        ecg_peak_amplitudes[wi] = label["ecg_r_peak_amplitudes_raw"]
        ecg_rr[wi] = label["ecg_rr_intervals_ms"]
        ecg_rr_corrected[wi] = label["ecg_rr_intervals_corrected_ms"]
        ecg_rmssd[wi] = label["ecg_rmssd_ms"]
        ecg_sdnn[wi] = label["ecg_sdnn_ms"]
        ecg_valid_sample_ratio[wi] = label["ecg_valid_sample_ratio"]
        ecg_valid_ibi_ratio[wi] = label["ecg_valid_ibi_ratio"]
        ecg_ibi_correction_ratio[wi] = label["ecg_ibi_correction_ratio"]
        ecg_qc_pass[wi] = label["ecg_qc_pass"]
        ecg_qc_reason[wi] = label["ecg_qc_reason"]
        ecg_label_qc_pass[wi] = label["ecg_label_qc_pass"]
        ecg_label_qc_reason[wi] = label["ecg_label_qc_reason"]
        ecg_qrs_sqi[wi] = label["ecg_qrs_sqi"]
        ecg_peak_count[wi] = label["ecg_peak_count"]
        ecg_hr_bpm[wi] = label["ecg_hr_bpm"]
        ecg_rr_cv[wi] = label["ecg_rr_cv"]
        if not label["ecg_qc_pass"]:
            keep[wi] = False
        strict_reasons = []
        if not label["ecg_label_qc_pass"]:
            strict_reasons.append("ecg_label_qc_failed")
        ppg_device_qc_pass[wi] = np.any(ppg_channel_qc_pass[wi], axis=1)
        if not bool(np.all(ppg_device_qc_pass[wi])):
            strict_reasons.append("ppg_device_qc_failed")
        window_qc_pass_strict[wi] = len(strict_reasons) == 0
        window_qc_reason_strict[wi] = "ok" if not strict_reasons else ";".join(strict_reasons)

    kept_idx = np.where(keep)[0]
    if kept_idx.size == 0:
        print(f"[SKIP] {pid}: all aligned windows failed dataset inclusion QC")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"synced_4device_{pid}.npz"
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
    }

    np.savez_compressed(
        out_path,
        config_json=np.array(json.dumps(cfg, ensure_ascii=False)),
        participant=np.array(pid),
        devices=np.array(DEVICES),
        channels=np.array(CHANNELS),
        kept_original_rows=kept_idx.astype(np.int32),
        source_indices=source_indices[kept_idx],
        source_start_offset_sec=source_start_offset_sec[kept_idx],
        t0_ms=t0_ms[kept_idx],
        t1_ms=t1_ms[kept_idx],
        ppg_50hz=ppg[kept_idx],
        ppg_valid_mask_50hz=ppg_valid_mask[kept_idx],
        ppg_valid_sample_ratio=ppg_valid_sample_ratio[kept_idx],
        ppg_valid_ibi_ratio=ppg_valid_ibi_ratio[kept_idx],
        ppg_sqi=ppg_sqi_arr[kept_idx],
        ppg_peak_times_rel_ms=ppg_peak_times[kept_idx],
        ppg_peak_indices_50hz=ppg_peak_indices[kept_idx],
        ppg_peak_amplitudes_raw=ppg_peak_amplitudes[kept_idx],
        ppg_ibi_ms=ppg_ibi[kept_idx],
        ppg_ibi_corrected_ms=ppg_ibi_corrected[kept_idx],
        ppg_ibi_correction_ratio=ppg_ibi_correction_ratio[kept_idx],
        ppg_rmssd_ms=ppg_rmssd[kept_idx],
        ppg_sdnn_ms=ppg_sdnn[kept_idx],
        ppg_channel_qc_pass=ppg_channel_qc_pass[kept_idx],
        ppg_device_qc_pass=ppg_device_qc_pass[kept_idx],
        ppg_qc_reason=ppg_qc_reason[kept_idx],
        motion_fraction=motion_fraction[kept_idx],
        motion_threshold=np.array([motion_threshold[d] for d in DEVICES], dtype=np.float32),
        ecg_r_peak_times_rel_ms=ecg_peak_times[kept_idx],
        ecg_r_peak_indices_50hz=ecg_peak_indices[kept_idx],
        ecg_r_peak_amplitudes_raw=ecg_peak_amplitudes[kept_idx],
        ecg_rr_intervals_ms=ecg_rr[kept_idx],
        ecg_rr_intervals_corrected_ms=ecg_rr_corrected[kept_idx],
        ecg_rmssd_ms=ecg_rmssd[kept_idx],
        ecg_sdnn_ms=ecg_sdnn[kept_idx],
        ecg_valid_sample_ratio=ecg_valid_sample_ratio[kept_idx],
        ecg_valid_ibi_ratio=ecg_valid_ibi_ratio[kept_idx],
        ecg_ibi_correction_ratio=ecg_ibi_correction_ratio[kept_idx],
        ecg_qc_pass=ecg_qc_pass[kept_idx],
        ecg_qc_reason=ecg_qc_reason[kept_idx],
        ecg_label_qc_pass=ecg_label_qc_pass[kept_idx],
        ecg_label_qc_reason=ecg_label_qc_reason[kept_idx],
        ecg_qrs_sqi=ecg_qrs_sqi[kept_idx],
        ecg_peak_count=ecg_peak_count[kept_idx],
        ecg_hr_bpm=ecg_hr_bpm[kept_idx],
        ecg_rr_cv=ecg_rr_cv[kept_idx],
        ecg_rpeak_detector_agreement=ecg_rpeak_detector_agreement[kept_idx],
        window_qc_pass_strict=window_qc_pass_strict[kept_idx],
        window_qc_reason_strict=window_qc_reason_strict[kept_idx],
    )

    summary = {
        "participant": pid,
        "n_aligned": len(matches),
        "n_kept": int(kept_idx.size),
        "kept_pct": 100.0 * kept_idx.size / len(matches),
        "target_len": target_len,
        "target_fs": target_fs,
        "align_tolerance_sec": align_tolerance_sec,
        "max_source_sample_gap_ms": max_gap_ms,
        "preprocess_mode": preprocess_mode,
        "mean_ecg_valid_ibi_ratio": float(np.nanmean(ecg_valid_ibi_ratio[kept_idx])),
        "mean_ecg_ibi_correction_ratio": float(np.nanmean(ecg_ibi_correction_ratio[kept_idx])),
        "mean_ecg_qrs_sqi": float(np.nanmean(ecg_qrs_sqi[kept_idx])),
        "mean_motion_fraction": float(np.nanmean(motion_fraction[kept_idx])),
        "mean_ppg_sqi": float(np.nanmean(ppg_sqi_arr[kept_idx])),
        "mean_ppg_valid_sample_ratio": float(np.nanmean(ppg_valid_sample_ratio[kept_idx])),
        "mean_ppg_ibi_correction_ratio": float(np.nanmean(ppg_ibi_correction_ratio[kept_idx])),
        "strict_window_pass_pct": float(np.mean(window_qc_pass_strict[kept_idx]) * 100.0),
    }
    pd.DataFrame([summary]).to_csv(out_dir / f"synced_4device_{pid}_summary.csv", index=False)
    print(f"[SAVED] {out_path} kept={kept_idx.size}/{len(matches)}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate four-device synced peak-label dataset.")
    ap.add_argument("--participants", default=None, help="Comma-separated IDs. Default: all available.")
    ap.add_argument("--exclude", default="P2,P14,P16,P17")
    ap.add_argument("--out-dir", default=str(config.HEURISTIC_RESULT_ROOT / "synced_4device_dataset"))
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

    participants = _participant_ids(args.participants)
    excluded = {normalize_participant_id(p) for p in args.exclude.split(",") if p.strip()}
    participants = [p for p in participants if p not in excluded]
    out_dir = Path(args.out_dir).resolve()

    print("[synced_dataset] participants=", participants)
    print(
        "[synced_dataset] unified method="
        f"{args.preprocess_mode}, target_fs={args.target_fs}, "
        f"target_len={int(args.window_sec * args.target_fs)}, "
        f"align_tol={args.align_tolerance_sec}s"
    )
    paths = []
    for pid in participants:
        path = generate_participant(
            pid,
            out_dir=out_dir,
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
        if path is not None:
            paths.append(path)

    summaries = []
    for path in paths:
        csv_path = path.with_name(path.stem + "_summary.csv")
        if csv_path.is_file():
            summaries.append(pd.read_csv(csv_path))
    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        combined.to_csv(out_dir / "synced_4device_summary.csv", index=False)
        print(f"[SAVED] {out_dir / 'synced_4device_summary.csv'}")


if __name__ == "__main__":
    main()
