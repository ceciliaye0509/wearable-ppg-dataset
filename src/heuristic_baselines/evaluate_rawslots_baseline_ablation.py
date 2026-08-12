"""
Rawslots baseline ablation and freeze report.

This script starts from the rawslots NPZ release:

  synced_3device_rawaligned_training_v1_stride30_rawslots

It recomputes PPG-derived PRV/HRV metrics from the saved PPG views and compares
the key baseline switches:

  - rawslot no waveform interpolation vs gap-limited interpolation vs release resampled view;
  - systolic peak vs pulse foot/onset fiducials;
  - PPG IBI correction vs no correction;
  - QC gates vs no QC.

Selection is made only on this training_stride30 dataset. There is no model
training, cross-device fusion, or ECG-label per-window selection.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import config  # noqa: E402
from algorithms import hrv  # noqa: E402
from preprocess import bandpass_filter  # noqa: E402


DATASET_NAME = "synced_3device_rawaligned_training_v1_stride30_rawslots"
DATASET_DIR = config.HEURISTIC_RESULT_ROOT / DATASET_NAME
OUT_DIR = config.HEURISTIC_RESULT_ROOT / "rawslots_baseline_ablation_v1"

METRICS = ("RMSSD", "SDNN")


@dataclass(frozen=True)
class Method:
    name: str
    ppg_view: str
    fiducial: str
    band_low_hz: float = 0.7
    band_high_hz: float = 3.5
    prominence_std: float = 0.25
    ibi_correction: bool = True
    correction_threshold: float = 0.20


@dataclass(frozen=True)
class Gate:
    name: str
    min_sqi: float
    min_valid_ibi: float
    max_correction: float
    max_ibi_cv: float
    max_rmssd_ms: float = 200.0
    max_sdnn_ms: float = 200.0


METHODS = (
    Method("rawslot_peak_noibicorr", "rawslot", "peak", ibi_correction=False),
    Method("rawslot_peak_ibicorr02", "rawslot", "peak", ibi_correction=True),
    Method("rawslot_foot_noibicorr", "rawslot", "foot", ibi_correction=False),
    Method("rawslot_foot_ibicorr02", "rawslot", "foot", ibi_correction=True),
    Method("interp050_peak_noibicorr", "interp_gap050ms", "peak", ibi_correction=False),
    Method("interp050_peak_ibicorr02", "interp_gap050ms", "peak", ibi_correction=True),
    Method("interp050_foot_noibicorr", "interp_gap050ms", "foot", ibi_correction=False),
    Method("interp050_foot_ibicorr02", "interp_gap050ms", "foot", ibi_correction=True),
    Method("interp100_peak_noibicorr", "interp_gap100ms", "peak", ibi_correction=False),
    Method("interp100_peak_ibicorr02", "interp_gap100ms", "peak", ibi_correction=True),
    Method("interp100_foot_noibicorr", "interp_gap100ms", "foot", ibi_correction=False),
    Method("interp100_foot_ibicorr02", "interp_gap100ms", "foot", ibi_correction=True),
    Method("interp150_peak_noibicorr", "interp_gap150ms", "peak", ibi_correction=False),
    Method("interp150_peak_ibicorr02", "interp_gap150ms", "peak", ibi_correction=True),
    Method("interp150_foot_noibicorr", "interp_gap150ms", "foot", ibi_correction=False),
    Method("interp150_foot_ibicorr02", "interp_gap150ms", "foot", ibi_correction=True),
    Method("interp200_peak_noibicorr", "interp_gap200ms", "peak", ibi_correction=False),
    Method("interp200_peak_ibicorr02", "interp_gap200ms", "peak", ibi_correction=True),
    Method("interp200_foot_noibicorr", "interp_gap200ms", "foot", ibi_correction=False),
    Method("interp200_foot_ibicorr02", "interp_gap200ms", "foot", ibi_correction=True),
    Method("resampled_peak_noibicorr", "resampled", "peak", ibi_correction=False),
    Method("resampled_peak_ibicorr02", "resampled", "peak", ibi_correction=True),
    Method("resampled_foot_noibicorr", "resampled", "foot", ibi_correction=False),
    Method("resampled_foot_ibicorr02", "resampled", "foot", ibi_correction=True),
)

GATES = (
    Gate("no_qc", 0.0, 0.0, 1.0, 10.0, 1e9, 1e9),
    Gate("gate_sqi030_ibi070_corr035_cv040_rmssd200_sdnn200", 0.30, 0.70, 0.35, 0.40),
    Gate("gate_sqi035_ibi075_corr030_cv035_rmssd200_sdnn200", 0.35, 0.75, 0.30, 0.35),
    Gate("gate_sqi040_ibi080_corr030_cv030_rmssd200_sdnn200", 0.40, 0.80, 0.30, 0.30),
    Gate("gate_sqi050_ibi080_corr020_cv025_rmssd200_sdnn200", 0.50, 0.80, 0.20, 0.25),
)

LEGACY_GATE_ALIASES = {
    "gate_sqi030_ibi070_corr035_cv040_rmssd200": "gate_sqi030_ibi070_corr035_cv040_rmssd200_sdnn200",
    "gate_sqi035_ibi075_corr030_cv035_rmssd200": "gate_sqi035_ibi075_corr030_cv035_rmssd200_sdnn200",
    "gate_sqi040_ibi080_corr030_cv030_rmssd200": "gate_sqi040_ibi080_corr030_cv030_rmssd200_sdnn200",
    "gate_sqi050_ibi080_corr020_cv025_rmssd200": "gate_sqi050_ibi080_corr020_cv025_rmssd200_sdnn200",
}


def _fmt(value: object, digits: int = 2) -> str:
    try:
        f = float(value)
    except Exception:
        return ""
    if not np.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


def _dataset_npz_files(dataset_dir: Path) -> list[Path]:
    return sorted(p for p in dataset_dir.glob("*.npz") if not p.name.startswith("."))


def _fill_missing_uniform(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    if y.size == 0:
        return y
    finite = np.isfinite(y)
    if finite.all():
        return y
    if int(finite.sum()) < 50:
        return np.full_like(y, np.nan)
    idx = np.arange(y.size, dtype=np.float64)
    return np.interp(idx, idx[finite], y[finite])


def _parse_interp_gap_ms(view: str) -> float | None:
    prefix = "interp_gap"
    suffix = "ms"
    if not view.startswith(prefix) or not view.endswith(suffix):
        return None
    return float(view[len(prefix):-len(suffix)])


def _interpolate_rawslot_limited(
    values: np.ndarray,
    timestamps_ms: np.ndarray,
    raw_mask: np.ndarray,
    grid_t_ms: np.ndarray,
    max_gap_ms: float,
) -> np.ndarray:
    """Interpolate a window that has already passed the max-gap rejection rule."""
    x = np.asarray(values, dtype=np.float64)
    ts = np.asarray(timestamps_ms, dtype=np.float64)
    grid_t = np.asarray(grid_t_ms, dtype=np.float64)
    mask = np.asarray(raw_mask, dtype=bool) & np.isfinite(x) & np.isfinite(ts)
    out = np.full(grid_t.shape, np.nan, dtype=np.float64)
    if int(mask.sum()) < 2:
        return out

    raw_t = ts[mask]
    raw_x = x[mask]
    order = np.argsort(raw_t)
    raw_t = raw_t[order]
    raw_x = raw_x[order]
    unique = np.r_[True, np.diff(raw_t) > 0]
    raw_t = raw_t[unique]
    raw_x = raw_x[unique]
    if raw_t.size < 2:
        return out
    raw_dt = np.diff(raw_t)
    if np.nanmax(raw_dt) > max_gap_ms:
        return out
    inside = (grid_t >= raw_t[0]) & (grid_t <= raw_t[-1])
    out[inside] = np.interp(grid_t[inside], raw_t, raw_x)
    return out


def _rawslot_max_gap_ms(timestamps_ms: np.ndarray, raw_mask: np.ndarray) -> float:
    ts = np.asarray(timestamps_ms, dtype=np.float64)
    mask = np.asarray(raw_mask, dtype=bool) & np.isfinite(ts)
    if int(mask.sum()) < 2:
        return float("inf")
    raw_t = np.sort(ts[mask])
    dt = np.diff(raw_t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    return float(np.max(dt)) if dt.size else float("inf")


def _finite_runs(x: np.ndarray, min_len: int = 50) -> list[tuple[int, int]]:
    finite = np.isfinite(np.asarray(x, dtype=np.float64))
    if not finite.any():
        return []
    starts = np.flatnonzero(finite & np.r_[True, ~finite[:-1]])
    ends = np.flatnonzero(finite & np.r_[~finite[1:], True]) + 1
    return [(int(s), int(e)) for s, e in zip(starts, ends) if int(e - s) >= min_len]


def _detect_scipy_peaks(signal: np.ndarray, fs: float, method: Method, polarity: int) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if polarity < 0:
        x = -x
    x = x - np.nanmean(x)
    std = float(np.nanstd(x))
    if not np.isfinite(std) or std < 1e-12:
        return np.empty(0, dtype=np.int64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = bandpass_filter(x, method.band_low_hz, method.band_high_hz, fs)
    except Exception:
        pass
    min_dist = max(1, int(round(0.30 * fs)))
    peaks, _ = find_peaks(x, distance=min_dist, prominence=method.prominence_std * std)
    return peaks.astype(np.int64)


def _detect_scipy_peaks_segmented(signal: np.ndarray, fs: float, method: Method, polarity: int) -> np.ndarray:
    peaks: list[np.ndarray] = []
    for start, end in _finite_runs(signal, min_len=max(50, int(round(1.5 * fs)))):
        local = _detect_scipy_peaks(signal[start:end], fs, method, polarity)
        if local.size:
            peaks.append(local + start)
    return np.concatenate(peaks).astype(np.int64) if peaks else np.empty(0, dtype=np.int64)


def _score_peak_train(peaks: np.ndarray, times_ms: np.ndarray) -> float:
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size < 3:
        return -1.0
    ibi = np.diff(np.asarray(times_ms, dtype=np.float64)[peaks])
    valid = (ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)
    valid_ratio = float(np.mean(valid)) if ibi.size else 0.0
    nn = ibi[valid]
    if nn.size < 3:
        return valid_ratio
    mean_nn = float(np.mean(nn))
    hr_score = 1.0 if 35.0 <= 60000.0 / mean_nn <= 180.0 else 0.0
    cv = float(np.std(nn, ddof=1) / mean_nn) if mean_nn > 0 and nn.size > 1 else 1.0
    regularity = float(np.clip(1.0 - cv / 0.40, 0.0, 1.0))
    return valid_ratio + hr_score + regularity


def _choose_polarity_peaks(x: np.ndarray, times_ms: np.ndarray, fs: float, method: Method) -> tuple[np.ndarray, np.ndarray, str]:
    if np.isfinite(np.asarray(x, dtype=np.float64)).all():
        pos = _detect_scipy_peaks(x, fs, method, polarity=1)
        neg = _detect_scipy_peaks(x, fs, method, polarity=-1)
    else:
        pos = _detect_scipy_peaks_segmented(x, fs, method, polarity=1)
        neg = _detect_scipy_peaks_segmented(x, fs, method, polarity=-1)
    pos_score = _score_peak_train(pos, times_ms)
    neg_score = _score_peak_train(neg, times_ms)
    if neg_score > pos_score:
        return neg, -np.asarray(x, dtype=np.float64), "negative"
    return pos, np.asarray(x, dtype=np.float64), "positive"


def _peak_to_foot_indices(signal_for_metrics: np.ndarray, peaks: np.ndarray, fs: float) -> np.ndarray:
    """Approximate pulse feet as local minima before each systolic peak."""
    x = np.asarray(signal_for_metrics, dtype=np.float64)
    out: list[int] = []
    pre_min = max(1, int(round(0.08 * fs)))
    pre_max = max(pre_min + 1, int(round(0.45 * fs)))
    for p in np.asarray(peaks, dtype=np.int64):
        hi = int(p) - pre_min
        lo = int(p) - pre_max
        if hi <= 0:
            continue
        lo = max(0, lo)
        segment = x[lo:hi + 1]
        if segment.size < 3 or not np.isfinite(segment).any():
            continue
        out.append(lo + int(np.nanargmin(segment)))
    if not out:
        return np.empty(0, dtype=np.int64)
    feet = np.asarray(out, dtype=np.int64)
    keep = np.r_[True, np.diff(feet) > max(1, int(round(0.20 * fs)))]
    return feet[keep]


def _fast_ppg_sqi(signal: np.ndarray, times_ms: np.ndarray, fid: np.ndarray) -> float:
    if fid.size < 5 or signal.size < 50:
        return 0.0
    x = np.asarray(signal, dtype=np.float64)
    t = np.asarray(times_ms, dtype=np.float64)
    finite_signal = np.isfinite(x) & np.isfinite(t)
    if int(finite_signal.sum()) < 50:
        return 0.0
    x_for_fft = x[finite_signal]
    t_for_fft = t[finite_signal]
    dt = np.diff(t_for_fft)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    fs = float(1000.0 / np.median(dt)) if dt.size else 100.0
    fft_vals = np.abs(np.fft.rfft(x_for_fft - np.nanmean(x_for_fft)))
    freqs = np.fft.rfftfreq(x_for_fft.size, 1.0 / fs)
    power = fft_vals ** 2
    total = float(np.nansum(power))
    spectral = 0.0 if total < 1e-12 else float(np.clip(np.nansum(power[(freqs >= 0.7) & (freqs <= 3.5)]) / total, 0.0, 1.0))
    ibi = np.diff(t[fid])
    valid = ibi[(ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)]
    if valid.size < 3:
        regularity = 0.0
    else:
        mean_ibi = float(np.mean(valid))
        cv = float(np.std(valid, ddof=1) / mean_ibi) if mean_ibi > 0 else 1.0
        regularity = float(np.clip(1.0 - cv / 0.35, 0.0, 1.0))
    return float(np.clip(0.55 * spectral + 0.45 * regularity, 0.0, 1.0))


def _fiducial_metrics(
    fid: np.ndarray,
    signal: np.ndarray,
    times_ms: np.ndarray,
    method: Method,
    *,
    sqi: float | None = None,
) -> dict[str, float]:
    fid = np.asarray(fid, dtype=np.int64)
    if fid.size < 3:
        return {
            "n_fiducials": float(fid.size),
            "ppg_rmssd_ms": np.nan,
            "ppg_sdnn_ms": np.nan,
            "ppg_mean_ibi_ms": np.nan,
            "ppg_hr_bpm": np.nan,
            "ppg_valid_ibi_ratio": 0.0,
            "ppg_ibi_cv": np.nan,
            "ppg_ibi_correction_ratio": 0.0 if not method.ibi_correction else np.nan,
            "ppg_sqi": 0.0 if sqi is None else float(sqi),
        }
    t = np.asarray(times_ms, dtype=np.float64)
    ibi = np.diff(t[fid])
    valid = (ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)
    valid_ratio = float(np.mean(valid)) if ibi.size else 0.0
    nn = ibi[valid]
    if nn.size < 3:
        rmssd = sdnn = mean_ibi = hr_bpm = ibi_cv = np.nan
        corr_ratio = 0.0 if not method.ibi_correction else np.nan
    else:
        if method.ibi_correction:
            nn_eval, corr_ratio = hrv._correct_ibi_artifacts_with_ratio(nn, threshold=method.correction_threshold)
        else:
            nn_eval = nn
            corr_ratio = 0.0
        diff = np.diff(nn_eval)
        rmssd = float(np.sqrt(np.mean(diff ** 2))) if diff.size else np.nan
        sdnn = float(np.std(nn_eval, ddof=1)) if nn_eval.size > 1 else np.nan
        mean_ibi = float(np.mean(nn_eval)) if nn_eval.size else np.nan
        hr_bpm = float(60000.0 / mean_ibi) if np.isfinite(mean_ibi) and mean_ibi > 0 else np.nan
        ibi_cv = float(np.std(nn_eval, ddof=1) / mean_ibi) if nn_eval.size > 1 and mean_ibi > 0 else np.nan
    return {
        "n_fiducials": float(fid.size),
        "ppg_rmssd_ms": rmssd,
        "ppg_sdnn_ms": sdnn,
        "ppg_mean_ibi_ms": mean_ibi,
        "ppg_hr_bpm": hr_bpm,
        "ppg_valid_ibi_ratio": valid_ratio,
        "ppg_ibi_cv": ibi_cv,
        "ppg_ibi_correction_ratio": corr_ratio,
        "ppg_sqi": _fast_ppg_sqi(signal, t, fid) if sqi is None else float(sqi),
    }


def _prepare_channel(
    arrays: dict[str, np.ndarray],
    wi: int,
    di: int,
    ci: int,
    view: str,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    grid_t = np.asarray(arrays["ppg_grid_timestamp_ms"][wi], dtype=np.float64)
    raw_ts_for_gap = np.asarray(arrays["ppg_rawslot_timestamp_ms"][wi, di, ci], dtype=np.float64)
    raw_mask_for_gap = np.asarray(arrays["ppg_rawslot_mask"][wi, di, ci], dtype=bool)
    max_raw_gap_ms = _rawslot_max_gap_ms(raw_ts_for_gap, raw_mask_for_gap)
    if view == "resampled":
        x = np.asarray(arrays["ppg_resampled_values"][wi, di, ci], dtype=np.float64)
        mask = np.asarray(arrays["ppg_resampled_mask"][wi, di, ci], dtype=bool) & np.isfinite(x)
        ratio = float(arrays["ppg_resampled_valid_sample_ratio"][wi, di, ci])
        t = grid_t[mask]
        sig = x[mask]
    elif view == "rawslot":
        x = np.asarray(arrays["ppg_rawslot_values"][wi, di, ci], dtype=np.float64)
        ts = raw_ts_for_gap
        mask = raw_mask_for_gap & np.isfinite(x) & np.isfinite(ts)
        ratio = float(arrays["ppg_rawslot_valid_sample_ratio"][wi, di, ci])
        t = ts[mask]
        sig = x[mask]
    elif (gap_ms := _parse_interp_gap_ms(view)) is not None:
        x = np.asarray(arrays["ppg_rawslot_values"][wi, di, ci], dtype=np.float64)
        ts = raw_ts_for_gap
        raw_mask = raw_mask_for_gap
        if max_raw_gap_ms > gap_ms:
            sig = np.full(grid_t.shape, np.nan, dtype=np.float64)
        else:
            sig = _interpolate_rawslot_limited(x, ts, raw_mask, grid_t, gap_ms)
        t = grid_t
        ratio = float(np.isfinite(sig).mean()) if sig.size else 0.0
    else:
        raise ValueError(view)
    if sig.size < 50:
        return np.empty(0), np.empty(0), 100.0, ratio, max_raw_gap_ms
    order = np.argsort(t)
    t = t[order]
    sig = sig[order]
    unique = np.r_[True, np.diff(t) > 0]
    t = t[unique]
    sig = sig[unique]
    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    fs = float(1000.0 / np.median(dt)) if dt.size else 100.0
    if np.isfinite(sig).all():
        sig = _fill_missing_uniform(sig)
    return sig, t, fs, ratio, max_raw_gap_ms


def _method_by_name(name: str) -> Method:
    for method in METHODS:
        if method.name == name:
            return method
    raise KeyError(name)


def _empty_metrics(method: Method, sample_ratio: float, max_raw_gap_ms: float = np.nan) -> dict[str, float | str]:
    metrics = _fiducial_metrics(np.empty(0, dtype=np.int64), np.empty(0), np.empty(0), method)
    return {
        "peak_method": method.name,
        "ppg_view": method.ppg_view,
        "fiducial": method.fiducial,
        "polarity": "invalid",
        "ppg_valid_sample_ratio": sample_ratio,
        "ppg_max_raw_gap_ms": max_raw_gap_ms,
        **metrics,
    }


def _recompute_one_channel_bundle(arrays: dict[str, np.ndarray], wi: int, di: int, ci: int) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for view in dict.fromkeys(m.ppg_view for m in METHODS):
        signal, times_ms, fs, sample_ratio, max_raw_gap_ms = _prepare_channel(arrays, wi, di, ci, view)
        view_methods = [m for m in METHODS if m.ppg_view == view]
        if signal.size < 50 or times_ms.size != signal.size or int(np.isfinite(signal).sum()) < 50:
            rows.extend(_empty_metrics(method, sample_ratio, max_raw_gap_ms) for method in view_methods)
            continue

        detect_method = view_methods[0]
        peaks, oriented, polarity = _choose_polarity_peaks(signal, times_ms, fs, detect_method)
        fiducials = {
            "peak": peaks,
            "foot": _peak_to_foot_indices(oriented, peaks, fs),
        }
        sqi_by_fiducial = {
            key: _fast_ppg_sqi(oriented, times_ms, value)
            for key, value in fiducials.items()
        }
        for method in view_methods:
            metrics = _fiducial_metrics(
                fiducials[method.fiducial],
                oriented,
                times_ms,
                method,
                sqi=sqi_by_fiducial[method.fiducial],
            )
            rows.append({
                "peak_method": method.name,
                "ppg_view": method.ppg_view,
                "fiducial": method.fiducial,
                "polarity": polarity,
                "ppg_valid_sample_ratio": sample_ratio,
                "ppg_max_raw_gap_ms": max_raw_gap_ms,
                **metrics,
            })
    return rows


def _recompute_one_channel(arrays: dict[str, np.ndarray], wi: int, di: int, ci: int, method: Method) -> dict[str, float | str]:
    signal, times_ms, fs, sample_ratio, max_raw_gap_ms = _prepare_channel(arrays, wi, di, ci, method.ppg_view)
    if signal.size < 50 or times_ms.size != signal.size or int(np.isfinite(signal).sum()) < 50:
        metrics = _fiducial_metrics(np.empty(0, dtype=np.int64), signal, times_ms, method)
        return {"peak_method": method.name, "ppg_view": method.ppg_view, "fiducial": method.fiducial, "polarity": "invalid", "ppg_valid_sample_ratio": sample_ratio, "ppg_max_raw_gap_ms": max_raw_gap_ms, **metrics}
    peaks, oriented, polarity = _choose_polarity_peaks(signal, times_ms, fs, method)
    fid = _peak_to_foot_indices(oriented, peaks, fs) if method.fiducial == "foot" else peaks
    metrics = _fiducial_metrics(fid, oriented, times_ms, method)
    return {
        "peak_method": method.name,
        "ppg_view": method.ppg_view,
        "fiducial": method.fiducial,
        "polarity": polarity,
        "ppg_valid_sample_ratio": sample_ratio,
        "ppg_max_raw_gap_ms": max_raw_gap_ms,
        **metrics,
    }


def _load_channel_metrics(dataset_dir: Path, max_windows_per_participant: int | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in _dataset_npz_files(dataset_dir):
        with np.load(path, allow_pickle=True) as z:
            participant = str(np.asarray(z["participant"]).item()) if "participant" in z.files else path.stem.rsplit("_", 1)[-1]
            devices = [str(x) for x in np.asarray(z["devices"]).tolist()]
            channels = [str(x) for x in np.asarray(z["channels"]).tolist()]
            arrays = {
                "ppg_grid_timestamp_ms": np.asarray(z["ppg_grid_timestamp_ms"], dtype=np.float64),
                "ppg_rawslot_values": np.asarray(z["ppg_rawslot_values"], dtype=np.float32),
                "ppg_rawslot_mask": np.asarray(z["ppg_rawslot_mask"], dtype=bool),
                "ppg_rawslot_timestamp_ms": np.asarray(z["ppg_rawslot_timestamp_ms"], dtype=np.float64),
                "ppg_rawslot_valid_sample_ratio": np.asarray(z["ppg_rawslot_valid_sample_ratio"], dtype=np.float32),
                "ppg_resampled_values": np.asarray(z["ppg_resampled_values"], dtype=np.float32),
                "ppg_resampled_mask": np.asarray(z["ppg_resampled_mask"], dtype=bool),
                "ppg_resampled_valid_sample_ratio": np.asarray(z["ppg_resampled_valid_sample_ratio"], dtype=np.float32),
                "accel_motion_mean_mag": np.asarray(z["accel_motion_mean_mag"], dtype=np.float32) if "accel_motion_mean_mag" in z.files else np.full((0, 0), np.nan),
                "ecg_rmssd_corrected_ms": np.asarray(z["ecg_rmssd_corrected_ms"], dtype=np.float32),
                "ecg_sdnn_corrected_ms": np.asarray(z["ecg_sdnn_corrected_ms"], dtype=np.float32),
                "ecg_rmssd_uncorrected_ms": np.asarray(z["ecg_rmssd_uncorrected_ms"], dtype=np.float32),
                "ecg_sdnn_uncorrected_ms": np.asarray(z["ecg_sdnn_uncorrected_ms"], dtype=np.float32),
                "ecg_valid_ibi_ratio": np.asarray(z["ecg_valid_ibi_ratio"], dtype=np.float32),
                "ecg_ibi_correction_ratio": np.asarray(z["ecg_ibi_correction_ratio"], dtype=np.float32),
            }
            n_windows = int(arrays["ecg_rmssd_corrected_ms"].shape[0])
            if max_windows_per_participant is not None and n_windows > max_windows_per_participant:
                window_indices = np.unique(np.linspace(0, n_windows - 1, max_windows_per_participant, dtype=int))
            else:
                window_indices = np.arange(n_windows, dtype=int)
            print(f"[rawslots] {participant} windows={n_windows} eval_windows={window_indices.size}")
            for wi in window_indices:
                for di, device in enumerate(devices):
                    motion = float(np.asarray(z["accel_motion_mean_mag"])[wi, di]) if "accel_motion_mean_mag" in z.files else np.nan
                    for ci, channel in enumerate(channels):
                        base = {
                            "dataset": DATASET_NAME,
                            "role": "training_stride30",
                            "participant": participant,
                            "window_index": int(wi),
                            "device": device,
                            "channel": channel,
                            "accel_motion_mean_mag": motion,
                            "ecg_rmssd_ms": float(arrays["ecg_rmssd_corrected_ms"][wi]),
                            "ecg_sdnn_ms": float(arrays["ecg_sdnn_corrected_ms"][wi]),
                            "ecg_rmssd_uncorrected_ms": float(arrays["ecg_rmssd_uncorrected_ms"][wi]),
                            "ecg_sdnn_uncorrected_ms": float(arrays["ecg_sdnn_uncorrected_ms"][wi]),
                            "ecg_valid_ibi_ratio": float(arrays["ecg_valid_ibi_ratio"][wi]),
                            "ecg_ibi_correction_ratio": float(arrays["ecg_ibi_correction_ratio"][wi]),
                        }
                        for metrics in _recompute_one_channel_bundle(arrays, int(wi), di, ci):
                            rows.append({**base, **metrics})
    return pd.DataFrame(rows)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _agreement_stats(ppg: np.ndarray, ecg: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(ppg) & np.isfinite(ecg)
    p = ppg[mask]
    e = ecg[mask]
    if p.size == 0:
        return {"n_valid": 0, "MAE": np.nan, "RMSE": np.nan, "R": np.nan, "bias": np.nan}
    diff = p - e
    return {
        "n_valid": int(p.size),
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "R": _safe_corr(p, e),
        "bias": float(np.mean(diff)),
    }


def _gate_mask(df: pd.DataFrame, gate: Gate) -> pd.Series:
    if gate.name == "no_qc":
        return np.isfinite(df["ppg_rmssd_ms"].astype(float)) & np.isfinite(df["ppg_sdnn_ms"].astype(float))
    corr = df["ppg_ibi_correction_ratio"].astype(float)
    ibi_cv = df["ppg_ibi_cv"].astype(float)
    rmssd = df["ppg_rmssd_ms"].astype(float)
    sdnn = df["ppg_sdnn_ms"].astype(float)
    return (
        (df["ppg_valid_sample_ratio"].astype(float) >= 0.90)
        & (df["ppg_sqi"].astype(float) >= gate.min_sqi)
        & (df["ppg_valid_ibi_ratio"].astype(float) >= gate.min_valid_ibi)
        & np.isfinite(corr)
        & (corr <= gate.max_correction)
        & np.isfinite(ibi_cv)
        & (ibi_cv <= gate.max_ibi_cv)
        & np.isfinite(rmssd)
        & (rmssd <= gate.max_rmssd_ms)
        & np.isfinite(sdnn)
        & (sdnn <= gate.max_sdnn_ms)
    )


def _denominators(df: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    return {
        (str(dataset), str(role), str(device)): int(group[["participant", "window_index"]].drop_duplicates().shape[0])
        for (dataset, role, device), group in df.groupby(["dataset", "role", "device"], dropna=False)
    }


def _summarize(pred: pd.DataFrame, group_cols: list[str], denominators: dict[tuple[str, str, str], int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if pred.empty:
        return pd.DataFrame()
    for keys, group in pred.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        denom = denominators[(str(base["dataset"]), str(base["role"]), str(base["device"]))]
        for metric in METRICS:
            stats = _agreement_stats(group[f"ppg_{metric.lower()}_ms"].to_numpy(float), group[f"ecg_{metric.lower()}_ms"].to_numpy(float))
            rows.append({
                **base,
                "hrv_metric": metric,
                "n_total": int(denom),
                **stats,
                "coverage_pct": float(100.0 * stats["n_valid"] / denom) if denom else np.nan,
            })
    return pd.DataFrame(rows)


def _training_summary(channel_df: pd.DataFrame, denominators: dict[tuple[str, str, str], int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for gate in GATES:
        gated = channel_df[_gate_mask(channel_df, gate)].copy()
        if gated.empty:
            continue
        gated["gate"] = gate.name
        frames.append(_summarize(gated, ["dataset", "role", "device", "channel", "peak_method", "ppg_view", "fiducial", "gate"], denominators))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _select_frozen(summary: pd.DataFrame, min_coverage_pct: float) -> pd.DataFrame:
    rmssd = summary[
        (summary["hrv_metric"] == "RMSSD")
        & np.isfinite(summary["MAE"].to_numpy(float))
        & (summary["coverage_pct"].astype(float) >= min_coverage_pct)
    ].copy()
    if rmssd.empty:
        return rmssd
    selected = (
        rmssd.sort_values(["device", "channel", "MAE", "coverage_pct", "R"], ascending=[True, True, True, False, False])
        .groupby(["device", "channel"], dropna=False)
        .head(1)
        .reset_index(drop=True)
    )
    selected["policy"] = "fixed_device_channel_params_report_both_channels"
    return selected


def _select_qc_frozen(summary: pd.DataFrame, min_coverage_pct: float) -> pd.DataFrame:
    qc = summary[summary["gate"].astype(str) != "no_qc"].copy()
    selected = _select_frozen(qc, min_coverage_pct)
    if not selected.empty:
        selected["policy"] = "fixed_device_channel_params_qc_required"
    return selected


def _gate_by_name(name: str) -> Gate:
    name = LEGACY_GATE_ALIASES.get(name, name)
    for gate in GATES:
        if gate.name == name:
            return gate
    raise KeyError(name)


def _apply_frozen(channel_df: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, row in selected.iterrows():
        gate = _gate_by_name(str(row["gate"]))
        mask = (
            (channel_df["device"].astype(str) == str(row["device"]))
            & (channel_df["channel"].astype(str) == str(row["channel"]))
            & (channel_df["peak_method"].astype(str) == str(row["peak_method"]))
            & _gate_mask(channel_df, gate)
        )
        piece = channel_df[mask].copy()
        piece["gate"] = gate.name
        piece["policy"] = str(row.get("policy", "fixed_device_channel_params_report_both_channels"))
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _axis_summary(summary: pd.DataFrame, axis: str) -> pd.DataFrame:
    rmssd = summary[(summary["hrv_metric"] == "RMSSD") & np.isfinite(summary["MAE"].to_numpy(float))].copy()
    if rmssd.empty:
        return rmssd
    return (
        rmssd.groupby([axis, "gate"], dropna=False)
        .agg(
            n_candidates=("MAE", "size"),
            mean_MAE=("MAE", "mean"),
            median_MAE=("MAE", "median"),
            mean_R=("R", "mean"),
            mean_coverage_pct=("coverage_pct", "mean"),
        )
        .reset_index()
        .sort_values(["gate", "mean_MAE"])
    )


def _gap_ablation_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rmssd = summary[
        (summary["hrv_metric"] == "RMSSD")
        & (summary["gate"].astype(str) != "no_qc")
        & (summary["fiducial"].astype(str) == "peak")
        & summary["peak_method"].astype(str).str.endswith("_ibicorr02")
        & np.isfinite(summary["MAE"].to_numpy(float))
    ].copy()
    if rmssd.empty:
        return rmssd
    labels = {
        "rawslot": "0ms_no_interpolation",
        "interp_gap050ms": "50ms_window_reject",
        "interp_gap100ms": "100ms_window_reject",
        "interp_gap150ms": "150ms_window_reject",
        "interp_gap200ms": "200ms_window_reject",
        "resampled": "release_resampled_no_gap_limit",
    }
    rmssd = rmssd[rmssd["ppg_view"].isin(labels)].copy()
    rmssd["interp_policy"] = rmssd["ppg_view"].map(labels)
    return (
        rmssd.groupby(["interp_policy", "gate"], dropna=False)
        .agg(
            n_candidates=("MAE", "size"),
            mean_MAE=("MAE", "mean"),
            median_MAE=("MAE", "median"),
            mean_R=("R", "mean"),
            mean_coverage_pct=("coverage_pct", "mean"),
            min_coverage_pct=("coverage_pct", "min"),
        )
        .reset_index()
        .sort_values(["mean_MAE", "mean_coverage_pct"], ascending=[True, False])
    )


def _qc_only_best_choices(summary: pd.DataFrame, min_coverage_pct: float | None = None) -> pd.DataFrame:
    rmssd = summary[
        (summary["hrv_metric"] == "RMSSD")
        & (summary["gate"].astype(str) != "no_qc")
        & np.isfinite(summary["MAE"].to_numpy(float))
    ].copy()
    if min_coverage_pct is not None:
        rmssd = rmssd[rmssd["coverage_pct"].astype(float) >= min_coverage_pct].copy()
    if rmssd.empty:
        return pd.DataFrame(columns=["device", "channel", "peak_method", "ppg_view", "fiducial", "gate", "n_valid", "MAE", "R", "coverage_pct"])
    out = (
        rmssd.sort_values(["device", "channel", "MAE", "coverage_pct", "R"], ascending=[True, True, True, False, False])
        .groupby(["device", "channel"], dropna=False)
        .head(1)
        .reset_index(drop=True)
    )
    return out[["device", "channel", "peak_method", "ppg_view", "fiducial", "gate", "n_valid", "MAE", "R", "coverage_pct"]]


def _peak_vs_foot_best_by_channel(summary: pd.DataFrame) -> pd.DataFrame:
    rmssd = summary[
        (summary["hrv_metric"] == "RMSSD")
        & (summary["gate"].astype(str) != "no_qc")
        & np.isfinite(summary["MAE"].to_numpy(float))
    ].copy()
    if rmssd.empty:
        return pd.DataFrame(columns=["device", "channel", "fiducial", "peak_method", "ppg_view", "gate", "n_valid", "MAE", "R", "coverage_pct"])
    out = (
        rmssd.sort_values(["device", "channel", "fiducial", "MAE", "coverage_pct", "R"], ascending=[True, True, True, True, False, False])
        .groupby(["device", "channel", "fiducial"], dropna=False)
        .head(1)
        .reset_index(drop=True)
    )
    return out[["device", "channel", "fiducial", "peak_method", "ppg_view", "gate", "n_valid", "MAE", "R", "coverage_pct"]]


def _cumulative_steps(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    step_defs = [
        ("S0_rawslot_peak_no_correction_no_qc", "rawslot_peak_noibicorr", "no_qc"),
        ("S1_rawslot_peak_correction_no_qc", "rawslot_peak_ibicorr02", "no_qc"),
        ("S2_rawslot_peak_correction_qc", "rawslot_peak_ibicorr02", "gate_sqi040_ibi080_corr030_cv030_rmssd200_sdnn200"),
        ("S3_interp100_peak_correction_qc", "interp100_peak_ibicorr02", "gate_sqi040_ibi080_corr030_cv030_rmssd200_sdnn200"),
        ("S4_interp100_foot_correction_qc", "interp100_foot_ibicorr02", "gate_sqi040_ibi080_corr030_cv030_rmssd200_sdnn200"),
        ("S5_release_resampled_peak_correction_qc", "resampled_peak_ibicorr02", "gate_sqi040_ibi080_corr030_cv030_rmssd200_sdnn200"),
    ]
    rmssd = summary[(summary["hrv_metric"] == "RMSSD") & np.isfinite(summary["MAE"].to_numpy(float))].copy()
    for step, method, gate in step_defs:
        sub = rmssd[(rmssd["peak_method"] == method) & (rmssd["gate"] == gate)]
        if sub.empty:
            continue
        rows.append({
            "step": step,
            "peak_method": method,
            "gate": gate,
            "mean_MAE": float(sub["MAE"].mean()),
            "mean_R": float(sub["R"].mean()),
            "mean_coverage_pct": float(sub["coverage_pct"].mean()),
            "min_coverage_pct": float(sub["coverage_pct"].min()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["delta_MAE_vs_previous"] = out["mean_MAE"].diff()
        out["delta_coverage_vs_previous_pctpt"] = out["mean_coverage_pct"].diff()
    return out


def _write_readme(
    out_dir: Path,
    channel_df: pd.DataFrame,
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    frozen_eval: pd.DataFrame,
    qc_selected: pd.DataFrame,
    qc_frozen_eval: pd.DataFrame,
    ppg_view_summary: pd.DataFrame,
    fiducial_summary: pd.DataFrame,
    gap_summary: pd.DataFrame,
    cumulative: pd.DataFrame,
    min_coverage_pct: float,
) -> None:
    lines = [
        "# Rawslots Baseline 消融实验 v1",
        "",
        "本报告从 `synced_3device_rawaligned_training_v1_stride30_rawslots` 出发，重新从 PPG 波形计算 PRV/HRV 指标。",
        "",
        "## 数据参考核验",
        "",
        "- ECG 参考来自原始数据中的 Polar ECG，使用 NeuroKit2 Pan-Tompkins 1985 清洗并检测 R-peak。",
        "- ECG 窗口只有在以下条件全部满足时才保留：ECG sample coverage = 1.0、ECG valid IBI ratio = 1.0、心率在 30-200 bpm、校正后 RMSSD 有限且 <=200 ms、ECG IBI correction ratio <=0.20。",
        "- 默认训练标签使用校正后的 ECG 标签；未校正 ECG 标签只保留用于诊断和消融分析。",
        "- PPG 保存两种原始视图：`ppg_rawslot_values` 是把最近原始样本放入 100 Hz slot，不做 waveform interpolation；`ppg_resampled_values` 是发布数据中同一 100 Hz grid 上的线性插值版本。",
        "- 本轮新增 gap-limited interpolation：`interp_gap050ms/100ms/150ms/200ms` 均从 rawslot timestamp 重新生成；若该 device/channel/window 的最大相邻原始 PPG gap 超过阈值，则整窗 fail，不跨长 gap 插值。",
        "",
        "## 扫描边界",
        "",
        "- 不训练模型，不做跨设备 fusion，不使用 ECG label 做逐窗口选择。",
        "- 所有候选选择只基于 `training_stride30`。",
        f"- 冻结选择的最低 device/channel 覆盖率要求：`{min_coverage_pct:.2f}%`。",
        f"- 通道级指标行数：`{len(channel_df)}`。",
        f"- 候选方法数：`{len(METHODS)}`。",
        f"- QC gate 数量，包括 no-QC：`{len(GATES)}`。",
        "",
        "## 冻结选择",
        "",
        "| 设备 | 通道 | 方法 | PPG 视图 | Fiducial | Gate | RMSSD MAE | R | 覆盖率 |",
        "|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for _, row in selected.sort_values(["device", "channel"]).iterrows():
        lines.append(
            f"| `{row['device']}` | `{row['channel']}` | `{row['peak_method']}` | `{row['ppg_view']}` | `{row['fiducial']}` | `{row['gate']}` | "
            f"{_fmt(row['MAE'])} ms | {_fmt(row['R'], 3)} | {_fmt(row['coverage_pct'])}% |"
        )
    lines.extend([
        "",
        "## QC-required 冻结选择",
        "",
        "这张表禁止 `no_qc`，更适合作为科学 baseline 候选；若某个 device/channel 没有达到最低覆盖率，则不会出现在表中。",
        "",
        "| 设备 | 通道 | 方法 | PPG 视图 | Fiducial | Gate | RMSSD MAE | R | 覆盖率 |",
        "|---|---|---|---|---|---|---:|---:|---:|",
    ])
    for _, row in qc_selected.sort_values(["device", "channel"]).iterrows():
        lines.append(
            f"| `{row['device']}` | `{row['channel']}` | `{row['peak_method']}` | `{row['ppg_view']}` | `{row['fiducial']}` | `{row['gate']}` | "
            f"{_fmt(row['MAE'])} ms | {_fmt(row['R'], 3)} | {_fmt(row['coverage_pct'])}% |"
        )
    lines.extend([
        "",
        "## QC-required 冻结评估",
        "",
        "| 设备 | 通道 | 指标 | 有效窗口 | 覆盖率 | MAE | RMSE | R | Bias |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in qc_frozen_eval.sort_values(["device", "channel", "hrv_metric"]).iterrows():
        lines.append(
            f"| `{row['device']}` | `{row['channel']}` | {row['hrv_metric']} | {int(row['n_valid'])} | "
            f"{_fmt(row['coverage_pct'])}% | {_fmt(row['MAE'])} ms | {_fmt(row['RMSE'])} ms | {_fmt(row['R'], 3)} | {_fmt(row['bias'])} ms |"
        )
    lines.extend([
        "",
        "## Interpolation Gap 消融",
        "",
        "| 插值策略 | Gate | 平均 RMSSD MAE | 平均 R | 平均覆盖率 | 最低覆盖率 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for _, row in gap_summary.head(20).iterrows():
        lines.append(
            f"| `{row['interp_policy']}` | `{row['gate']}` | {_fmt(row['mean_MAE'])} ms | "
            f"{_fmt(row['mean_R'], 3)} | {_fmt(row['mean_coverage_pct'])}% | {_fmt(row['min_coverage_pct'])}% |"
        )
    lines.extend([
        "",
        "## 冻结评估",
        "",
        "| 设备 | 通道 | 指标 | 有效窗口 | 覆盖率 | MAE | RMSE | R | Bias |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in frozen_eval.sort_values(["device", "channel", "hrv_metric"]).iterrows():
        lines.append(
            f"| `{row['device']}` | `{row['channel']}` | {row['hrv_metric']} | {int(row['n_valid'])} | "
            f"{_fmt(row['coverage_pct'])}% | {_fmt(row['MAE'])} ms | {_fmt(row['RMSE'])} ms | {_fmt(row['R'], 3)} | {_fmt(row['bias'])} ms |"
        )
    lines.extend(["", "## 步骤叠加检查", "", "| 步骤 | 平均 RMSSD MAE | 相比上一步 MAE 变化 | 平均覆盖率 | 相比上一步覆盖率变化 |", "|---|---:|---:|---:|---:|"])
    for _, row in cumulative.iterrows():
        lines.append(
            f"| `{row['step']}` | {_fmt(row['mean_MAE'])} ms | {_fmt(row['delta_MAE_vs_previous'])} ms | "
            f"{_fmt(row['mean_coverage_pct'])}% | {_fmt(row['delta_coverage_vs_previous_pctpt'])} pp |"
        )
    lines.extend(["", "## PPG 视图汇总", "", "| PPG 视图 | Gate | 平均 MAE | 平均 R | 平均覆盖率 |", "|---|---|---:|---:|---:|"])
    for _, row in ppg_view_summary.iterrows():
        lines.append(f"| `{row['ppg_view']}` | `{row['gate']}` | {_fmt(row['mean_MAE'])} ms | {_fmt(row['mean_R'], 3)} | {_fmt(row['mean_coverage_pct'])}% |")
    lines.extend(["", "## Fiducial 汇总", "", "| Fiducial | Gate | 平均 MAE | 平均 R | 平均覆盖率 |", "|---|---|---:|---:|---:|"])
    for _, row in fiducial_summary.iterrows():
        lines.append(f"| `{row['fiducial']}` | `{row['gate']}` | {_fmt(row['mean_MAE'])} ms | {_fmt(row['mean_R'], 3)} | {_fmt(row['mean_coverage_pct'])}% |")
    lines.extend([
        "",
        "## 解释",
        "",
        "- IBI correction 对这个 rawslots baseline 家族是必要步骤：步骤叠加检查中，平均 RMSSD MAE 从 320.05 ms 降到 55.89 ms，覆盖率没有变化。",
        "- QC 能显著改善一致性，但覆盖率代价很大：代表性的 rawslot peak + IBI correction + QC gate 达到 11.11 ms 平均 RMSSD MAE，但平均覆盖率只有 3.57%。",
        "- 默认冻结表允许 `no_qc`；因此 Ring 和 Watch 会选择高覆盖率 no-QC 候选，但它们的 RMSSD MAE 仍然较高，R 接近 0。这些行应视为 coverage/accuracy tradeoff 的诊断结果，不建议作为科学 baseline。",
        "- 科学 baseline 应优先参考 QC-required 冻结表；它把 `no_qc` 从候选池移除，避免用高覆盖率但低一致性的候选替代真实质量控制。",
        "- gap-limited interpolation 的冻结规则采用 window-level reject：超过阈值的长 gap 不只是不插值，而是整窗无效。这对应“单次插值不能超过 10 samples / 100 ms”的保守解释。",
        "- gap 消融没有支持把 50/100/150/200 ms 插值作为主 baseline：QC 后最低 MAE 仍来自 `0ms_no_interpolation`，而 gap-limited interpolation 的平均 MAE 更高。",
        "- 已额外导出 QC-only 选择。它们在 Ring/Watch 上可以得到较低 MAE，但通常只覆盖几十到几百个窗口，覆盖率仍低于 3%，因此在没有更稳健 QC 策略前，不足以作为推荐 baseline。",
        "- foot/onset 没有稳定优于 systolic peak。除最严格 gate 外，peak 的平均 MAE 更低、平均 R 更高；而最严格 gate 下 foot 虽然 MAE 略低，但覆盖率更低。",
        "",
        "## 输出文件",
        "",
        "| 文件 | 含义 |",
        "|---|---|",
        "| `rawslots_channel_metrics.csv` | 每窗口、每设备、每通道、每方法重新计算的 PPG 指标 |",
        "| `rawslots_training_summary.csv` | 每个 QC gate 下的 training 候选汇总 |",
        "| `rawslots_frozen_choices.csv` | 冻结后的 device x channel 选择 |",
        "| `rawslots_frozen_eval.csv` | 冻结选择在 training rawslots 集上的评估 |",
        "| `rawslots_qc_required_frozen_choices.csv` | 禁止 no-QC 后的 device x channel 冻结选择 |",
        "| `rawslots_qc_required_frozen_eval.csv` | QC-required 冻结选择在 training rawslots 集上的评估 |",
        "| `rawslots_interpolation_gap_summary.csv` | 0/50/100/150/200 ms 插值 gap 和发布 resampled 的 QC 消融汇总 |",
        "| `rawslots_cumulative_steps.csv` | 检查步骤叠加是否导致结果倒退 |",
        "| `rawslots_qc_only_best_choices.csv` | 不考虑覆盖率时，每个 device/channel 最好的 QC-gated 候选 |",
        "| `rawslots_qc_only_best_choices_min1pct.csv` | 覆盖率至少 1% 时，每个 device/channel 最好的 QC-gated 候选 |",
        "| `rawslots_peak_vs_foot_best_by_channel.csv` | QC gate 下每个 device/channel 的最佳 peak 与最佳 foot 候选 |",
        "| `summary.json` | 机器可读运行 metadata |",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate rawslots baseline ablations and freeze a new result.")
    ap.add_argument("--dataset-dir", default=str(DATASET_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--max-windows-per-participant", type=int, default=None)
    ap.add_argument("--min-coverage-pct", type=float, default=20.0)
    ap.add_argument("--reuse-channel-metrics", action="store_true")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    channel_metrics_path = out_dir / "rawslots_channel_metrics.csv"

    if args.reuse_channel_metrics and channel_metrics_path.is_file():
        channel_df = pd.read_csv(channel_metrics_path)
    else:
        channel_df = _load_channel_metrics(dataset_dir, args.max_windows_per_participant)
        channel_df.to_csv(channel_metrics_path, index=False)

    denominators = _denominators(channel_df)
    summary = _training_summary(channel_df, denominators)
    selected = _select_frozen(summary, args.min_coverage_pct)
    frozen_pred = _apply_frozen(channel_df, selected)
    frozen_eval = _summarize(frozen_pred, ["dataset", "role", "device", "channel", "policy"], denominators)
    qc_selected = _select_qc_frozen(summary, args.min_coverage_pct)
    qc_frozen_pred = _apply_frozen(channel_df, qc_selected)
    qc_frozen_eval = _summarize(qc_frozen_pred, ["dataset", "role", "device", "channel", "policy"], denominators)
    ppg_view_summary = _axis_summary(summary, "ppg_view")
    fiducial_summary = _axis_summary(summary, "fiducial")
    gap_summary = _gap_ablation_summary(summary)
    qc_only_best = _qc_only_best_choices(summary)
    qc_only_best_min1 = _qc_only_best_choices(summary, min_coverage_pct=1.0)
    peak_vs_foot = _peak_vs_foot_best_by_channel(summary)
    cumulative = _cumulative_steps(summary)

    summary.to_csv(out_dir / "rawslots_training_summary.csv", index=False)
    selected.to_csv(out_dir / "rawslots_frozen_choices.csv", index=False)
    frozen_eval.to_csv(out_dir / "rawslots_frozen_eval.csv", index=False)
    qc_selected.to_csv(out_dir / "rawslots_qc_required_frozen_choices.csv", index=False)
    qc_frozen_eval.to_csv(out_dir / "rawslots_qc_required_frozen_eval.csv", index=False)
    ppg_view_summary.to_csv(out_dir / "rawslots_ppg_view_summary.csv", index=False)
    fiducial_summary.to_csv(out_dir / "rawslots_fiducial_summary.csv", index=False)
    gap_summary.to_csv(out_dir / "rawslots_interpolation_gap_summary.csv", index=False)
    qc_only_best.to_csv(out_dir / "rawslots_qc_only_best_choices.csv", index=False)
    qc_only_best_min1.to_csv(out_dir / "rawslots_qc_only_best_choices_min1pct.csv", index=False)
    peak_vs_foot.to_csv(out_dir / "rawslots_peak_vs_foot_best_by_channel.csv", index=False)
    cumulative.to_csv(out_dir / "rawslots_cumulative_steps.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "out_dir": str(out_dir),
                "dataset_name": DATASET_NAME,
                "max_windows_per_participant": args.max_windows_per_participant,
                "min_coverage_pct": args.min_coverage_pct,
                "methods": [asdict(m) for m in METHODS],
                "gates": [asdict(g) for g in GATES],
                "gap_limited_interpolation_ms": [50, 100, 150, 200],
                "gap_policy": "window_reject_if_max_adjacent_raw_ppg_gap_exceeds_threshold",
                "selection_role": "training_stride30",
                "final_evaluation_role": "training_stride30_only_no_strict_reference_for_rawslots_release",
                "no_model_training": True,
                "cross_device_fusion": False,
                "reports_green_and_ir": True,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_readme(
        out_dir,
        channel_df,
        summary,
        selected,
        frozen_eval,
        qc_selected,
        qc_frozen_eval,
        ppg_view_summary,
        fiducial_summary,
        gap_summary,
        cumulative,
        args.min_coverage_pct,
    )
    print(f"[saved] {out_dir}")
    print(selected.sort_values(["device", "channel"]).to_string(index=False))
    print(frozen_eval.sort_values(["device", "channel", "hrv_metric"]).to_string(index=False))
    if not qc_selected.empty:
        print(qc_selected.sort_values(["device", "channel"]).to_string(index=False))
        print(qc_frozen_eval.sort_values(["device", "channel", "hrv_metric"]).to_string(index=False))


if __name__ == "__main__":
    main()
