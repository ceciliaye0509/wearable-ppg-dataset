"""
Rawslots detector-level peak vs foot/onset validation.

This script answers a narrower question than evaluate_rawslots_baseline_ablation.py:
given the same rawslot PPG view, does a detector's native foot/onset fiducial
outperform that detector's peak fiducial for PRV/HRV?
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import config  # noqa: E402
from algorithms import hrv, msptd, pwd, qppgfast  # noqa: E402
from evaluate_rawslots_baseline_ablation import (  # noqa: E402
    DATASET_NAME,
    _agreement_stats,
    _dataset_npz_files,
    _fiducial_metrics,
    _fill_missing_uniform,
    _finite_runs,
    _fast_ppg_sqi,
    _peak_to_foot_indices,
    _prepare_channel,
    _score_ibi_train,
)
from preprocess import bandpass_filter  # noqa: E402


DATASET_DIR = config.HEURISTIC_RESULT_ROOT / DATASET_NAME
OUT_DIR = config.HEURISTIC_RESULT_ROOT / "rawslots_detector_fiducial_light_v1"
METRICS = ("RMSSD", "SDNN")
DEFAULT_LIGHT_DETECTORS = ("scipy", "qppgfast")
DEFAULT_LIGHT_WINDOWS_PER_PARTICIPANT = 5


@dataclass(frozen=True)
class DetectorMethod:
    detector: str
    fiducial: str
    common_band: tuple[float, float] | None = (0.7, 3.5)
    ibi_correction: bool = True
    correction_threshold: float = 0.20

    @property
    def name(self) -> str:
        correction = "ibicorr02" if self.ibi_correction else "noibicorr"
        return f"{self.detector}_{self.fiducial}_{_band_label(self.common_band)}_{correction}"


@dataclass(frozen=True)
class DetectorPipeline:
    """Detector plus the common preprocessing applied before its native code."""

    detector: str
    common_band: tuple[float, float] | None

    @property
    def label(self) -> str:
        return _band_label(self.common_band)


@dataclass(frozen=True)
class Gate:
    name: str
    min_sqi: float
    min_valid_ibi: float
    max_ibi_cv: float
    max_rmssd_ms: float = 200.0
    max_sdnn_ms: float = 200.0


DETECTORS = ("scipy", "pwd", "msptd", "qppgfast")
FIDUCIALS = ("peak", "foot")
WAVEFORM_CONSENSUS_SEGMENTS = 5
WAVEFORM_CONSENSUS_MIN_VOTES = 4
GATES = (
    Gate("no_qc", 0.0, 0.0, 10.0, 1e9, 1e9),
    # Pre-registered one-factor QC grid. Each sweep holds the other two
    # detector-derived QC statistics at a wide base level, so the coverage /
    # MAE trade-off of SQI, valid-IBI ratio, and IBI CV remains interpretable.
    Gate("qc_base_sqi005_ibi050_cv040_rmssd200_sdnn200", 0.05, 0.50, 0.40),
    Gate("qc_sqi000_ibi050_cv040_rmssd200_sdnn200", 0.00, 0.50, 0.40),
    Gate("qc_sqi010_ibi050_cv040_rmssd200_sdnn200", 0.10, 0.50, 0.40),
    Gate("qc_sqi020_ibi050_cv040_rmssd200_sdnn200", 0.20, 0.50, 0.40),
    Gate("qc_sqi030_ibi050_cv040_rmssd200_sdnn200", 0.30, 0.50, 0.40),
    Gate("qc_sqi005_ibi030_cv040_rmssd200_sdnn200", 0.05, 0.30, 0.40),
    Gate("qc_sqi005_ibi060_cv040_rmssd200_sdnn200", 0.05, 0.60, 0.40),
    Gate("qc_sqi005_ibi070_cv040_rmssd200_sdnn200", 0.05, 0.70, 0.40),
    Gate("qc_sqi005_ibi085_cv040_rmssd200_sdnn200", 0.05, 0.85, 0.40),
    Gate("qc_sqi005_ibi050_cv025_rmssd200_sdnn200", 0.05, 0.50, 0.25),
    Gate("qc_sqi005_ibi050_cv030_rmssd200_sdnn200", 0.05, 0.50, 0.30),
    Gate("qc_sqi005_ibi050_cv035_rmssd200_sdnn200", 0.05, 0.50, 0.35),
    Gate("qc_sqi005_ibi050_cv050_rmssd200_sdnn200", 0.05, 0.50, 0.50),
)


def _fmt(value: object, digits: int = 2) -> str:
    try:
        f = float(value)
    except Exception:
        return ""
    if not np.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


def _band_label(band: tuple[float, float] | None) -> str:
    if band is None:
        return "bp_off"
    return f"bp{int(round(band[0] * 100)):03d}_{int(round(band[1] * 100)):03d}"


def _bandpass_or_original(signal: np.ndarray, fs: float, band: tuple[float, float] | None) -> np.ndarray:
    x = _fill_missing_uniform(np.asarray(signal, dtype=np.float64))
    if band is None or x.size < 50 or not np.isfinite(x).all():
        return x
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bandpass_filter(x, band[0], band[1], fs)
    except Exception:
        return x


def _detect_scipy(signal: np.ndarray, fs: float, polarity: int) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if polarity < 0:
        x = -x
    x = x - np.nanmean(x)
    std = float(np.nanstd(x))
    if not np.isfinite(std) or std < 1e-12:
        return np.empty(0, dtype=np.int64)
    min_dist = max(1, int(round(0.30 * fs)))
    peaks, _ = find_peaks(x, distance=min_dist, prominence=0.25 * std)
    return peaks.astype(np.int64)


def _call_detector(detector: str, signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    if detector == "scipy":
        peaks = _detect_scipy(signal, fs, polarity=1)
        feet = _peak_to_foot_indices(signal, peaks, fs)
        return peaks, feet
    if detector == "pwd":
        return pwd._pwd_beat_detector(signal, fs)
    if detector == "msptd":
        return msptd._msptd_beat_detector(signal, fs)
    if detector == "qppgfast":
        # qppgfast normalizes invalid sentinel values in place. Polarity
        # candidates must therefore receive independent signal buffers.
        return qppgfast._qppgfast_beat_detector(np.array(signal, dtype=np.float64, copy=True), fs)
    raise KeyError(detector)


def _detect_oriented(detector: str, oriented: np.ndarray, times_ms: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Run one detector after the waveform orientation has been fixed."""
    if np.isfinite(oriented).all():
        peaks, feet = _call_detector(detector, oriented, fs)
    else:
        peak_chunks: list[np.ndarray] = []
        foot_chunks: list[np.ndarray] = []
        for start, end in _finite_runs(oriented, min_len=max(50, int(round(1.5 * fs)))):
            local_peaks, local_feet = _call_detector(detector, oriented[start:end], fs)
            if local_peaks.size:
                peak_chunks.append(local_peaks + start)
            if local_feet.size:
                foot_chunks.append(local_feet + start)
        peaks = np.concatenate(peak_chunks).astype(np.int64) if peak_chunks else np.empty(0, dtype=np.int64)
        feet = np.concatenate(foot_chunks).astype(np.int64) if foot_chunks else np.empty(0, dtype=np.int64)
    return {
        "peak": peaks[(peaks >= 0) & (peaks < times_ms.size)],
        "foot": feet[(feet >= 0) & (feet < times_ms.size)],
    }


def _waveform_polarity_score(signal: np.ndarray) -> float:
    """Score whether ``signal`` has the expected upward PPG pulse morphology.

    This intentionally uses only waveform samples, before and independently of
    any detector/fiducial. Positive PPG pulses are expected to be relatively
    sharp and positively skewed: a stronger positive derivative than negative
    derivative, plus a narrow upward tail. Negating the waveform reverses both
    terms and therefore reverses the decision.
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 50:
        return 0.0
    lo, hi = np.quantile(x, [0.01, 0.99])
    x = np.clip(x, lo, hi)
    centered = x - np.mean(x)
    scale = float(np.std(centered))
    if not np.isfinite(scale) or scale < 1e-12:
        return 0.0
    skewness = float(np.mean((centered / scale) ** 3))
    derivative = np.diff(x)
    if derivative.size < 2:
        return skewness
    rise = max(0.0, float(np.quantile(derivative, 0.95)))
    fall = max(0.0, abs(float(np.quantile(derivative, 0.05))))
    slope_asymmetry = (rise - fall) / (rise + fall + 1e-12)
    return skewness + slope_asymmetry


def _waveform_consensus_polarity(signal: np.ndarray) -> tuple[str, float, int, float]:
    """Choose waveform polarity only when 4 of 5 time segments agree.

    The five equal-duration segments and the 4/5 rule are fixed before any
    detector, fiducial, IBI, or ECG values are read.  A tie or an unstable
    waveform becomes ``ambiguous`` instead of being forced into a direction.
    """
    x = np.asarray(signal, dtype=np.float64)
    segments = [part for part in np.array_split(x, WAVEFORM_CONSENSUS_SEGMENTS) if part.size >= 50]
    scores = np.asarray([_waveform_polarity_score(part) for part in segments], dtype=np.float64)
    usable = scores[np.isfinite(scores) & (scores != 0.0)]
    if usable.size < WAVEFORM_CONSENSUS_SEGMENTS:
        return "ambiguous", np.nan, 0, 0.0
    positive_votes = int(np.sum(usable > 0.0))
    negative_votes = int(np.sum(usable < 0.0))
    if positive_votes >= WAVEFORM_CONSENSUS_MIN_VOTES:
        return "positive", float(np.median(usable)), positive_votes, float(positive_votes / usable.size)
    if negative_votes >= WAVEFORM_CONSENSUS_MIN_VOTES:
        return "negative", float(np.median(usable)), negative_votes, float(negative_votes / usable.size)
    return "ambiguous", float(np.median(usable)), max(positive_votes, negative_votes), float(max(positive_votes, negative_votes) / usable.size)


def _choose_polarity_detector(
    detector: str,
    signal: np.ndarray,
    times_ms: np.ndarray,
    fs: float,
    *,
    common_band: tuple[float, float] | None,
    polarity_mode: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, str, float, int, float]:
    filt = _bandpass_or_original(signal, fs, common_band)
    if polarity_mode in {"fixed_positive", "fixed_negative"}:
        label = "positive" if polarity_mode == "fixed_positive" else "negative"
        oriented = filt if label == "positive" else -filt
        return _detect_oriented(detector, oriented, times_ms, fs), oriented, label, np.nan, 0, np.nan

    if polarity_mode == "waveform_morphology":
        score = _waveform_polarity_score(filt)
        if score >= 0.0:
            oriented, label = filt, "positive"
        else:
            oriented, label = -filt, "negative"
        return _detect_oriented(detector, oriented, times_ms, fs), oriented, label, score, 1, 1.0

    if polarity_mode == "waveform_consensus_v1":
        label, score, votes, consensus = _waveform_consensus_polarity(filt)
        if label == "ambiguous":
            empty = {fiducial: np.empty(0, dtype=np.int64) for fiducial in FIDUCIALS}
            return empty, filt, label, score, votes, consensus
        oriented = filt if label == "positive" else -filt
        return _detect_oriented(detector, oriented, times_ms, fs), oriented, label, score, votes, consensus

    score_fiducial = "foot" if polarity_mode == "foot_train" else "peak"
    candidates: list[tuple[float, dict[str, np.ndarray], np.ndarray, str]] = []
    for polarity, label in ((1, "positive"), (-1, "negative")):
        oriented = filt if polarity > 0 else -filt
        fiducials = _detect_oriented(detector, oriented, times_ms, fs)
        score = _score_ibi_train(fiducials[score_fiducial], times_ms)
        candidates.append((score, fiducials, oriented, label))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2], candidates[0][3], candidates[0][0], 0, np.nan


def _parse_detectors(value: str | None) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        return DETECTORS
    out = tuple(x.strip().lower() for x in value.split(",") if x.strip())
    unknown = sorted(set(out) - set(DETECTORS))
    if unknown:
        raise ValueError(f"unknown detector(s): {unknown}; known={DETECTORS}")
    return out


def _strict_interp100_channel(
    arrays: dict[str, np.ndarray],
    wi: int,
    di: int,
    ci: int,
    *,
    max_gap_ms: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Build a uniform grid without extrapolating beyond raw PPG support.

    A window is rejected when an internal raw-sample gap, or either boundary
    support gap, exceeds ``max_gap_ms``. The returned grid is the supported
    part of the 100 Hz window, so every detector sees a genuinely uniform
    sequence and no value is extrapolated at the boundaries.
    """
    grid_t = np.asarray(arrays["ppg_grid_timestamp_ms"][wi], dtype=np.float64)
    values = np.asarray(arrays["ppg_rawslot_values"][wi, di, ci], dtype=np.float64)
    timestamps = np.asarray(arrays["ppg_rawslot_timestamp_ms"][wi, di, ci], dtype=np.float64)
    mask = (
        np.asarray(arrays["ppg_rawslot_mask"][wi, di, ci], dtype=bool)
        & np.isfinite(values)
        & np.isfinite(timestamps)
    )
    raw_ratio = float(arrays["ppg_rawslot_valid_sample_ratio"][wi, di, ci])
    if int(mask.sum()) < 2 or grid_t.size < 50:
        return np.empty(0), np.empty(0), 100.0, raw_ratio, float("inf")

    raw_t = timestamps[mask]
    raw_x = values[mask]
    order = np.argsort(raw_t)
    raw_t = raw_t[order]
    raw_x = raw_x[order]
    unique = np.r_[True, np.diff(raw_t) > 0]
    raw_t = raw_t[unique]
    raw_x = raw_x[unique]
    if raw_t.size < 2:
        return np.empty(0), np.empty(0), 100.0, raw_ratio, float("inf")

    internal_gaps = np.diff(raw_t)
    leading_gap = max(0.0, float(raw_t[0] - grid_t[0]))
    trailing_gap = max(0.0, float(grid_t[-1] - raw_t[-1]))
    max_support_gap = float(max(np.max(internal_gaps), leading_gap, trailing_gap))
    if not np.isfinite(max_support_gap) or max_support_gap > max_gap_ms:
        return np.empty(0), np.empty(0), 100.0, raw_ratio, max_support_gap

    inside = (grid_t >= raw_t[0]) & (grid_t <= raw_t[-1])
    uniform_t = grid_t[inside]
    if uniform_t.size < 50:
        return np.empty(0), np.empty(0), 100.0, raw_ratio, max_support_gap
    uniform_x = np.interp(uniform_t, raw_t, raw_x)
    return uniform_x, uniform_t, 100.0, raw_ratio, max_support_gap


def _parse_bandpass_candidates(value: str) -> tuple[tuple[float, float] | None, ...]:
    """Parse a compact, reproducible bandpass grid such as ``off,0.7-3.5``."""
    bands: list[tuple[float, float] | None] = []
    for raw in value.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token in {"off", "none"}:
            bands.append(None)
            continue
        try:
            low_text, high_text = token.split("-", maxsplit=1)
            low, high = float(low_text), float(high_text)
        except ValueError as exc:
            raise ValueError(f"invalid bandpass candidate {raw!r}; use off or low-high, e.g. 0.7-3.5") from exc
        if not (0.0 < low < high):
            raise ValueError(f"invalid bandpass candidate {raw!r}")
        bands.append((low, high))
    if not bands:
        raise ValueError("at least one bandpass candidate is required")
    if len(set(bands)) != len(bands):
        raise ValueError("bandpass candidates must be unique")
    return tuple(bands)


def _parse_participants(value: str | None) -> tuple[str, ...] | None:
    if value is None or not value.strip():
        return None
    participants = tuple(x.strip().upper() for x in value.split(",") if x.strip())
    if not participants:
        return None
    if len(set(participants)) != len(participants):
        raise ValueError("participants must be unique")
    return participants


def _parse_devices(value: str | None) -> tuple[str, ...] | None:
    if value is None or not value.strip():
        return None
    devices = tuple(x.strip() for x in value.split(",") if x.strip())
    if not devices:
        raise ValueError("devices must not be empty")
    if len(set(devices)) != len(devices):
        raise ValueError("devices must be unique")
    return devices


def _participant_from_path(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1].upper()


def _detector_pipelines(
    detectors: tuple[str, ...], bands: tuple[tuple[float, float] | None, ...]
) -> tuple[DetectorPipeline, ...]:
    return tuple(DetectorPipeline(detector, band) for detector in detectors for band in bands)


def _load_channel_metrics_for_path(
    path: Path,
    window_indices: np.ndarray,
    pipelines: tuple[DetectorPipeline, ...],
    *,
    input_mode: str,
    max_gap_ms: float,
    correction_states: tuple[bool, ...],
    polarity_mode: str,
    selected_devices: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
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
        print(f"[detector-fiducial] {participant} eval_windows={len(window_indices)}")
        for wi in window_indices:
            for di, device in enumerate(devices):
                if selected_devices is not None and device not in selected_devices:
                    continue
                motion = float(np.asarray(z["accel_motion_mean_mag"])[wi, di]) if "accel_motion_mean_mag" in z.files else np.nan
                for ci, channel in enumerate(channels):
                    if input_mode == "strict_interp100":
                        signal, times_ms, fs, sample_ratio, max_raw_gap_ms = _strict_interp100_channel(
                            arrays, int(wi), di, ci, max_gap_ms=max_gap_ms
                        )
                    else:
                        signal, times_ms, fs, sample_ratio, max_raw_gap_ms = _prepare_channel(
                            arrays, int(wi), di, ci, "rawslot"
                        )
                    base = {
                        "dataset": DATASET_NAME,
                        "role": "training_stride30",
                        "participant": participant,
                        "window_index": int(wi),
                        "device": device,
                        "channel": channel,
                        "ppg_view": input_mode,
                        "accel_motion_mean_mag": motion,
                        "ppg_valid_sample_ratio": sample_ratio,
                        "ppg_max_raw_gap_ms": max_raw_gap_ms,
                        "ecg_rmssd_ms": float(arrays["ecg_rmssd_corrected_ms"][wi]),
                        "ecg_sdnn_ms": float(arrays["ecg_sdnn_corrected_ms"][wi]),
                        "ecg_rmssd_uncorrected_ms": float(arrays["ecg_rmssd_uncorrected_ms"][wi]),
                        "ecg_sdnn_uncorrected_ms": float(arrays["ecg_sdnn_uncorrected_ms"][wi]),
                        "ecg_valid_ibi_ratio": float(arrays["ecg_valid_ibi_ratio"][wi]),
                        "ecg_ibi_correction_ratio": float(arrays["ecg_ibi_correction_ratio"][wi]),
                    }
                    if signal.size < 50 or times_ms.size != signal.size or int(np.isfinite(signal).sum()) < 50:
                        for pipeline in pipelines:
                            for fiducial in FIDUCIALS:
                                for ibi_correction in correction_states:
                                    method = DetectorMethod(
                                        pipeline.detector,
                                        fiducial,
                                        common_band=pipeline.common_band,
                                        ibi_correction=ibi_correction,
                                    )
                                    rows.append({
                                        **base,
                                        "peak_method": method.name,
                                        "detector": pipeline.detector,
                                        "fiducial": fiducial,
                                        "common_bandpass": pipeline.label,
                                        "ibi_correction": ibi_correction,
                                        "polarity": "invalid",
                                        "polarity_mode": polarity_mode,
                                        "polarity_decision_score": np.nan,
                                        "polarity_consensus_votes": 0,
                                        "polarity_consensus_fraction": np.nan,
                                        **_fiducial_metrics(np.empty(0, dtype=np.int64), signal, times_ms, method),
                                    })
                        continue
                    for pipeline in pipelines:
                        fiducials, oriented, polarity, polarity_score, consensus_votes, consensus_fraction = _choose_polarity_detector(
                            pipeline.detector,
                            signal,
                            times_ms,
                            fs,
                            common_band=pipeline.common_band,
                            polarity_mode=polarity_mode,
                        )
                        for fiducial in FIDUCIALS:
                            for ibi_correction in correction_states:
                                method = DetectorMethod(
                                    pipeline.detector,
                                    fiducial,
                                    common_band=pipeline.common_band,
                                    ibi_correction=ibi_correction,
                                )
                                fid = fiducials[fiducial]
                                # Keep the spectral SQI input fixed across the
                                # bandpass switch. Otherwise bandpass-on would
                                # pass the same spectral QC by construction.
                                sqi = _fast_ppg_sqi(signal, times_ms, fid)
                                rows.append({
                                    **base,
                                    "peak_method": method.name,
                                    "detector": pipeline.detector,
                                    "fiducial": fiducial,
                                    "common_bandpass": pipeline.label,
                                    "ibi_correction": ibi_correction,
                                    "polarity": polarity,
                                    "polarity_mode": polarity_mode,
                                    "polarity_decision_score": polarity_score,
                                    "polarity_consensus_votes": consensus_votes,
                                    "polarity_consensus_fraction": consensus_fraction,
                                    "fiducial_ibi_score": _score_ibi_train(fid, times_ms),
                                    **_fiducial_metrics(fid, oriented, times_ms, method, sqi=sqi),
                                })
    return pd.DataFrame(rows)


def _window_index_chunks(path: Path, max_windows_per_participant: int | None, chunks_per_participant: int) -> list[np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        n_windows = int(np.asarray(z["ecg_rmssd_corrected_ms"]).shape[0])
    if max_windows_per_participant is not None and n_windows > max_windows_per_participant:
        indices = np.unique(np.linspace(0, n_windows - 1, max_windows_per_participant, dtype=int))
    else:
        indices = np.arange(n_windows, dtype=int)
    return [chunk.astype(int) for chunk in np.array_split(indices, max(1, chunks_per_participant)) if chunk.size]


def _load_channel_metrics_parallel(
    dataset_dir: Path,
    max_windows_per_participant: int | None,
    *,
    pipelines: tuple[DetectorPipeline, ...],
    input_mode: str,
    max_gap_ms: float,
    correction_states: tuple[bool, ...],
    participants: tuple[str, ...] | None,
    polarity_mode: str,
    selected_devices: tuple[str, ...] | None,
    jobs: int,
    chunks_per_participant: int,
) -> pd.DataFrame:
    tasks: list[tuple[Path, np.ndarray]] = []
    paths = _dataset_npz_files(dataset_dir)
    if participants is not None:
        requested = set(participants)
        found = {_participant_from_path(path) for path in paths}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"participants not found in dataset: {missing}")
        paths = [path for path in paths if _participant_from_path(path) in requested]
    for path in paths:
        for chunk in _window_index_chunks(path, max_windows_per_participant, chunks_per_participant):
            tasks.append((path, chunk))
    if jobs <= 1:
        frames = [
            _load_channel_metrics_for_path(
                path,
                chunk,
                pipelines,
                input_mode=input_mode,
                max_gap_ms=max_gap_ms,
                correction_states=correction_states,
                polarity_mode=polarity_mode,
                selected_devices=selected_devices,
            )
            for path, chunk in tasks
        ]
    else:
        frames = []
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = [
                ex.submit(
                    _load_channel_metrics_for_path,
                    path,
                    chunk,
                    pipelines,
                    input_mode=input_mode,
                    max_gap_ms=max_gap_ms,
                    correction_states=correction_states,
                    polarity_mode=polarity_mode,
                    selected_devices=selected_devices,
                )
                for path, chunk in tasks
            ]
            for i, fut in enumerate(as_completed(futures), start=1):
                frames.append(fut.result())
                print(f"[detector-fiducial] finished chunk {i}/{len(futures)}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _gate_mask(df: pd.DataFrame, gate: Gate) -> pd.Series:
    if gate.name == "no_qc":
        return np.isfinite(df["ppg_rmssd_ms"].astype(float)) & np.isfinite(df["ppg_sdnn_ms"].astype(float))
    ibi_cv = df["ppg_ibi_cv"].astype(float)
    rmssd = df["ppg_rmssd_ms"].astype(float)
    sdnn = df["ppg_sdnn_ms"].astype(float)
    return (
        (df["ppg_valid_sample_ratio"].astype(float) >= 0.90)
        & (df["ppg_sqi"].astype(float) >= gate.min_sqi)
        & (df["ppg_valid_ibi_ratio"].astype(float) >= gate.min_valid_ibi)
        & np.isfinite(ibi_cv)
        & (ibi_cv <= gate.max_ibi_cv)
        & np.isfinite(rmssd)
        & (rmssd <= gate.max_rmssd_ms)
        & np.isfinite(sdnn)
        & (sdnn <= gate.max_sdnn_ms)
    )


def _denominators(df: pd.DataFrame) -> dict[tuple[str, str, str, str], int]:
    return {
        (str(dataset), str(role), str(device), str(channel)): int(group[["participant", "window_index"]].drop_duplicates().shape[0])
        for (dataset, role, device, channel), group in df.groupby(["dataset", "role", "device", "channel"], dropna=False)
    }


def _summarize(pred: pd.DataFrame, group_cols: list[str], denominators: dict[tuple[str, str, str, str], int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in pred.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        denom = denominators[(str(base["dataset"]), str(base["role"]), str(base["device"]), str(base["channel"]))]
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


def _training_summary(channel_df: pd.DataFrame, denominators: dict[tuple[str, str, str, str], int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for gate in GATES:
        gated = channel_df[_gate_mask(channel_df, gate)].copy()
        if gated.empty:
            continue
        gated["gate"] = gate.name
        frames.append(
            _summarize(
                gated,
                [
                    "dataset",
                    "role",
                    "device",
                    "channel",
                    "detector",
                    "fiducial",
                    "common_bandpass",
                    "ibi_correction",
                    "peak_method",
                    "gate",
                ],
                denominators,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _macro_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rmssd = summary[(summary["hrv_metric"] == "RMSSD") & np.isfinite(summary["MAE"].to_numpy(float))].copy()
    if rmssd.empty:
        return rmssd
    return (
        rmssd.groupby(["detector", "fiducial", "common_bandpass", "ibi_correction", "gate"], dropna=False)
        .agg(
            channels=("MAE", "size"),
            n_valid=("n_valid", "sum"),
            mean_MAE=("MAE", "mean"),
            median_MAE=("MAE", "median"),
            mean_R=("R", "mean"),
            mean_coverage_pct=("coverage_pct", "mean"),
            min_coverage_pct=("coverage_pct", "min"),
        )
        .reset_index()
        .sort_values(["gate", "mean_MAE", "mean_coverage_pct"], ascending=[True, True, False])
    )


def _paired_peak_foot(summary: pd.DataFrame) -> pd.DataFrame:
    rmssd = summary[(summary["hrv_metric"] == "RMSSD") & np.isfinite(summary["MAE"].to_numpy(float))].copy()
    rows: list[dict[str, object]] = []
    for keys, group in rmssd.groupby(
        ["device", "channel", "detector", "common_bandpass", "ibi_correction", "gate"], dropna=False
    ):
        peak = group[group["fiducial"].astype(str) == "peak"]
        foot = group[group["fiducial"].astype(str) == "foot"]
        if peak.empty or foot.empty:
            continue
        p = peak.iloc[0]
        f = foot.iloc[0]
        rows.append({
            "device": keys[0],
            "channel": keys[1],
            "detector": keys[2],
            "common_bandpass": keys[3],
            "ibi_correction": keys[4],
            "gate": keys[5],
            "peak_MAE": float(p["MAE"]),
            "foot_MAE": float(f["MAE"]),
            "foot_minus_peak_MAE": float(f["MAE"]) - float(p["MAE"]),
            "peak_R": float(p["R"]),
            "foot_R": float(f["R"]),
            "foot_minus_peak_R": float(f["R"]) - float(p["R"]),
            "peak_coverage_pct": float(p["coverage_pct"]),
            "foot_coverage_pct": float(f["coverage_pct"]),
            "foot_minus_peak_coverage_pct": float(f["coverage_pct"]) - float(p["coverage_pct"]),
            "winner_by_MAE": "foot" if float(f["MAE"]) < float(p["MAE"]) else "peak",
        })
    return (
        pd.DataFrame(rows).sort_values(
            ["gate", "detector", "common_bandpass", "ibi_correction", "device", "channel"]
        )
        if rows
        else pd.DataFrame()
    )


def _write_report(
    out_dir: Path,
    macro: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    input_mode: str,
    max_gap_ms: float,
    pipelines: tuple[DetectorPipeline, ...],
    correction_states: tuple[bool, ...],
    polarity_mode: str,
) -> None:
    detectors = ", ".join(f"`{x}`" for x in sorted(macro["detector"].astype(str).unique())) if not macro.empty else ""
    bandpass_states = ", ".join(dict.fromkeys(p.label for p in pipelines))
    correction_text = ", ".join("on (0.20)" if value else "off" for value in correction_states)
    lines: list[str] = [
        "# Rawslots Detector 级 Peak vs Foot/Onset 验证",
        "",
        f"- 数据：`{DATASET_NAME}`。",
        f"- 输入：`{input_mode}`；严格插值模式要求内部与两端 raw PPG support gap 均不超过 `{max_gap_ms:.0f} ms`，且不做边界外推。",
        f"- 本次 common bandpass 候选：`{bandpass_states}`；PWD 始终保留其原生 Bessel low-pass。",
        f"- 本次 IBI correction：{correction_text}。QC gate 对所有候选相同，只用于报告，不参与按算法挑选阈值。",
        f"- 极性选择：`{polarity_mode}`。`waveform_morphology` 仅用 detector 前波形的偏度与上升/回落沿不对称性选择方向；`waveform_consensus_v1` 额外要求固定的 5 个等时段中至少 4 段同方向，否则标记 ambiguous；两者均不使用 peak、foot 或 IBI。",
        "- SQI 的频谱部分固定由 common bandpass 前的严格插值输入计算，避免 bandpass-on 因定义本身获得 QC 优势。",
        f"- detector：{detectors}。",
        "- 比较方式：同一个 detector 内分别用 `peak` 和原生 `foot/onset` 计算 PRV，再和 ECG RMSSD/SDNN 对齐比较。",
        "- `coverage_pct = n_valid / n_total * 100%`，这里的 `n_total` 是对应 device/channel 的总窗口数。",
        "",
        "## RMSSD 宏平均",
        "",
    ]
    if macro.empty:
        lines.append("_没有可汇总的结果行。_")
    else:
        show = macro.copy()
        lines.append("| detector | fiducial | common bandpass | IBI correction | gate | 通道数 | 平均 coverage % | 最低 coverage % | MAE | R |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|")
        for _, row in show.iterrows():
            lines.append(
                f"| `{row['detector']}` | `{row['fiducial']}` | `{row['common_bandpass']}` | "
                f"`{bool(row['ibi_correction'])}` | `{row['gate']}` | "
                f"{int(row['channels'])} | {_fmt(row['mean_coverage_pct'])} | {_fmt(row['min_coverage_pct'])} | "
                f"{_fmt(row['mean_MAE'])} | {_fmt(row['mean_R'], 3)} |"
            )
    lines.extend([
        "",
        "## Peak vs Foot 配对比较",
        "",
        "负的 `foot_minus_peak_MAE` 表示 foot 更好；正数表示 peak 更好。",
        "",
    ])
    if paired.empty:
        lines.append("_没有可配对的 peak/foot 结果。_")
    else:
        winners = (
            paired.groupby(["detector", "common_bandpass", "ibi_correction", "gate", "winner_by_MAE"], dropna=False)
            .size()
            .reset_index(name="n_channels")
            .sort_values(["gate", "detector", "common_bandpass", "ibi_correction", "winner_by_MAE"])
        )
        lines.append("| detector | common bandpass | IBI correction | gate | MAE 更好者 | 通道数 |")
        lines.append("|---|---|---|---|---|---:|")
        for _, row in winners.iterrows():
            lines.append(
                f"| `{row['detector']}` | `{row['common_bandpass']}` | `{bool(row['ibi_correction'])}` | "
                f"`{row['gate']}` | `{row['winner_by_MAE']}` | {int(row['n_channels'])} |"
            )
    lines.extend([
        "",
        "## 文件说明",
        "",
        "| 文件 | 含义 |",
        "|---|---|",
        "| `rawslots_detector_fiducial_channel_metrics.csv` | 每窗口、每设备、每通道、每 detector/fiducial 的 PRV 指标 |",
        "| `rawslots_detector_fiducial_summary.csv` | 逐 device/channel 的 RMSSD/SDNN 结果 |",
        "| `rawslots_detector_fiducial_macro_summary.csv` | 6 通道宏平均 |",
        "| `rawslots_detector_peak_vs_foot_paired.csv` | 同一 detector 内 peak vs foot 的逐通道配对差值 |",
    ])
    out_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate detector-native foot/onset vs peak on rawslots.")
    ap.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--max-windows-per-participant", type=int, default=DEFAULT_LIGHT_WINDOWS_PER_PARTICIPANT)
    ap.add_argument(
        "--all-windows",
        action="store_true",
        help="Evaluate every available window for each selected participant. Requires --allow-full.",
    )
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--chunks-per-participant", type=int, default=1)
    ap.add_argument("--detectors", type=str, default=",".join(DEFAULT_LIGHT_DETECTORS))
    ap.add_argument(
        "--participants",
        type=str,
        default=None,
        help="Comma-separated participant IDs. Omit to use every participant in the dataset.",
    )
    ap.add_argument(
        "--input-mode",
        choices=("rawslot", "strict_interp100"),
        default="rawslot",
        help="rawslot keeps the legacy compacted sequence; strict_interp100 makes a uniform grid with a gap limit.",
    )
    ap.add_argument("--max-gap-ms", type=float, default=100.0)
    ap.add_argument(
        "--polarity-mode",
        choices=("peak_train", "foot_train", "fixed_positive", "fixed_negative", "waveform_morphology", "waveform_consensus_v1"),
        default="peak_train",
        help="peak_train/foot_train choose a common direction from that fiducial's IBI quality; fixed modes retain or negate all PPG; waveform modes choose polarity before detection, and consensus_v1 rejects unstable directions.",
    )
    ap.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Optional comma-separated device names, for example Earring. Omit to evaluate all devices.",
    )
    ap.add_argument(
        "--bandpass-candidates",
        type=str,
        default=None,
        help="Comma-separated common bandpass candidates: off,0.5-4.0,0.7-3.5,0.5-8.0.",
    )
    ap.add_argument(
        "--fair-screen",
        action="store_true",
        help="Compatibility shortcut: run bandpass off/0.7-3.5 and IBI correction off/on.",
    )
    ap.add_argument("--allow-full", action="store_true", help="Allow full-cohort detector runs. Use with care on 16GB machines.")
    ap.add_argument("--reuse-channel-metrics", action="store_true")
    args = ap.parse_args()
    detectors = _parse_detectors(args.detectors)
    if args.bandpass_candidates is not None and args.fair_screen:
        raise SystemExit("--bandpass-candidates and --fair-screen cannot be used together")
    bandpass_spec = args.bandpass_candidates
    if bandpass_spec is None:
        bandpass_spec = "off,0.7-3.5" if args.fair_screen else "0.7-3.5"
    bands = _parse_bandpass_candidates(bandpass_spec)
    pipelines = _detector_pipelines(detectors, bands)
    correction_states = (False, True) if args.fair_screen else (True,)
    participants = _parse_participants(args.participants)
    selected_devices = _parse_devices(args.devices)
    if args.all_windows:
        args.max_windows_per_participant = None
    if args.max_windows_per_participant is None and not args.allow_full:
        raise SystemExit("--all-windows requires --allow-full.")
    slow = {"pwd", "msptd", "qppgfast"}
    if args.max_windows_per_participant is None and slow.intersection(detectors) and args.jobs > 1 and not args.allow_full:
        raise SystemExit("Full slow-detector multiprocessing requires --allow-full.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "rawslots_detector_fiducial_channel_metrics.csv"
    if args.reuse_channel_metrics and metrics_path.is_file():
        channel_df = pd.read_csv(metrics_path)
    else:
        channel_df = _load_channel_metrics_parallel(
            args.dataset_dir,
            args.max_windows_per_participant,
            pipelines=pipelines,
            input_mode=args.input_mode,
            max_gap_ms=args.max_gap_ms,
            correction_states=correction_states,
            participants=participants,
            polarity_mode=args.polarity_mode,
            selected_devices=selected_devices,
            jobs=args.jobs,
            chunks_per_participant=args.chunks_per_participant,
        )
        channel_df.to_csv(metrics_path, index=False)

    denominators = _denominators(channel_df)
    summary = _training_summary(channel_df, denominators)
    macro = _macro_summary(summary)
    paired = _paired_peak_foot(summary)

    summary.to_csv(args.out_dir / "rawslots_detector_fiducial_summary.csv", index=False)
    macro.to_csv(args.out_dir / "rawslots_detector_fiducial_macro_summary.csv", index=False)
    paired.to_csv(args.out_dir / "rawslots_detector_peak_vs_foot_paired.csv", index=False)
    args.out_dir.joinpath("run_config.json").write_text(
        json.dumps(
            {
                "dataset": DATASET_NAME,
                "ppg_view": args.input_mode,
                "strict_max_gap_ms": args.max_gap_ms if args.input_mode == "strict_interp100" else None,
                "polarity_mode": args.polarity_mode,
                "common_bandpass_candidates_hz": [list(band) if band is not None else None for band in bands],
                "detectors": list(detectors),
                "pipelines": [
                    {"detector": pipeline.detector, "common_band_hz": list(pipeline.common_band) if pipeline.common_band else None}
                    for pipeline in pipelines
                ],
                "fiducials": list(FIDUCIALS),
                "ibi_correction_states": list(correction_states),
                "gates": [gate.__dict__ for gate in GATES],
                "participants": list(participants) if participants is not None else None,
                "devices": list(selected_devices) if selected_devices is not None else None,
                "max_windows_per_participant": args.max_windows_per_participant,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(
        args.out_dir,
        macro,
        paired,
        input_mode=args.input_mode,
        max_gap_ms=args.max_gap_ms,
        pipelines=pipelines,
        correction_states=correction_states,
        polarity_mode=args.polarity_mode,
    )
    print(f"[SAVED] {args.out_dir}")


if __name__ == "__main__":
    main()
