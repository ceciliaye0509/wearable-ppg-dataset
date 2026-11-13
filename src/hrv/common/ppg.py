from __future__ import annotations
from pathlib import Path
import math
import numpy as np
from scipy.signal import butter,filtfilt,find_peaks
import warnings
from . import rr as hrv
from . import qppgfast


def bandpass_filter(x: np.ndarray, low_hz: float, high_hz: float, fs: float, order: int = 2) -> np.ndarray:
    nyq = fs / 2
    low = max(0.01, low_hz / nyq)
    high = min(0.99, high_hz / nyq)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x.astype(np.float64))


def _finite_runs(x: np.ndarray, min_len: int = 50) -> list[tuple[int, int]]:
    finite = np.isfinite(np.asarray(x, dtype=np.float64))
    if not finite.any():
        return []
    starts = np.flatnonzero(finite & np.r_[True, ~finite[:-1]])
    ends = np.flatnonzero(finite & np.r_[~finite[1:], True]) + 1
    return [(int(s), int(e)) for s, e in zip(starts, ends) if int(e - s) >= min_len]

def _score_ibi_train(fiducials: np.ndarray, times_ms: np.ndarray) -> float:
    """Score one fiducial IBI train for polarity selection."""
    fiducials = np.asarray(fiducials, dtype=np.int64)
    if fiducials.size < 3:
        return -1.0
    ibi = np.diff(np.asarray(times_ms, dtype=np.float64)[fiducials])
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

FIDUCIALS=("peak","foot")
WAVEFORM_CONSENSUS_SEGMENTS=5
WAVEFORM_CONSENSUS_MIN_VOTES=4


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

def _call_detector(detector,signal,fs):
    if detector == "qppgfast":
        return qppgfast._qppgfast_beat_detector(np.array(signal,dtype=np.float64,copy=True),fs)
    if detector == "scipy":
        peaks=_detect_scipy(signal,fs,1)
        return peaks,_peak_to_foot_indices(signal,peaks,fs)
    raise ValueError(detector)

def detect(signal,times,device,detector="qppgfast"):
    fid,*_=_choose_polarity_detector(detector,signal,times,100.,common_band=(.7,3.5),polarity_mode="peak_train")
    return times[fid["peak" if detector=="scipy" or device=="Earring" else "foot"]]

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
