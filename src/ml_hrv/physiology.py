"""Beat decoding and adjacency-preserving physiological HRV estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrectedRR:
    rr_ms: np.ndarray
    correction_ratio: float
    n_merged_extra_beats: int
    n_split_missed_beats: int
    n_broken_intervals: int


@dataclass(frozen=True)
class BeatHRVResult:
    rmssd_ms: float
    sdnn_ms: float
    confidence: float
    reject_reason: str
    beat_times_ms: np.ndarray
    corrected_rr: CorrectedRR


def _rolling_median(values: np.ndarray, index: int, radius: int = 5) -> float:
    start, stop = max(0, index - radius), min(len(values), index + radius + 1)
    local = values[start:stop]
    local = local[np.isfinite(local) & (local >= 300.0) & (local <= 2000.0)]
    return float(np.median(local)) if len(local) else 800.0


def correct_rr_adjacency(beat_times_ms: np.ndarray) -> CorrectedRR:
    """Correct missed/extra beats without fabricating RMSSD adjacency.

    Uncorrectable intervals become NaN separators.  Consequently a difference
    is never taken between RR intervals that were separated by a discarded
    interval—the exact bug in the legacy ``rr = rr[valid]`` implementation.
    """
    beats = np.unique(np.sort(np.asarray(beat_times_ms, dtype=np.float64)))
    raw = np.diff(beats)
    if len(raw) == 0:
        return CorrectedRR(np.array([], dtype=float), 1.0, 0, 0, 0)
    corrected: list[float] = []
    merged = split = broken = changed = 0
    index = 0
    while index < len(raw):
        interval = float(raw[index])
        median = _rolling_median(raw, index)
        # Two short adjacent intervals usually indicate one extra detected beat.
        if (
            index + 1 < len(raw)
            and interval < 0.62 * median
            and raw[index + 1] < 0.75 * median
            and 0.72 * median <= interval + raw[index + 1] <= 1.32 * median
        ):
            corrected.append(interval + float(raw[index + 1]))
            index += 2
            merged += 1
            changed += 2
            continue
        # A long interval close to an integer multiple suggests missed beats.
        multiple = int(np.clip(round(interval / max(median, 1.0)), 2, 4))
        piece = interval / multiple
        if interval > 1.55 * median and 300.0 <= piece <= 2000.0:
            corrected.extend([piece] * multiple)
            index += 1
            split += multiple - 1
            changed += 1
            continue
        if not 300.0 <= interval <= 2000.0:
            corrected.append(float("nan"))
            broken += 1
            changed += 1
        else:
            corrected.append(interval)
        index += 1
    return CorrectedRR(
        np.asarray(corrected, dtype=np.float64),
        changed / max(len(raw), 1),
        merged,
        split,
        broken,
    )


def hrv_from_corrected_rr(corrected: CorrectedRR) -> tuple[float, float]:
    rr = corrected.rr_ms
    valid = np.isfinite(rr)
    finite_rr = rr[valid]
    if len(finite_rr) < 3:
        return float("nan"), float("nan")
    sdnn = float(np.std(finite_rr, ddof=1))
    # Both sides must be finite; NaN separators prevent false adjacency.
    pair_valid = np.isfinite(rr[:-1]) & np.isfinite(rr[1:])
    successive = np.diff(rr)[pair_valid]
    rmssd = float(np.sqrt(np.mean(np.square(successive)))) if len(successive) >= 2 else float("nan")
    return rmssd, sdnn


def decode_beat_times(
    probability: np.ndarray,
    offset_ms: np.ndarray | None = None,
    fs_hz: float = 100.0,
    threshold: float = 0.35,
    min_rr_ms: float = 300.0,
) -> np.ndarray:
    probability = np.asarray(probability, dtype=float).reshape(-1)
    candidates = np.flatnonzero(
        (probability[1:-1] >= probability[:-2])
        & (probability[1:-1] > probability[2:])
        & (probability[1:-1] >= threshold)
    ) + 1
    min_distance = max(1, round(min_rr_ms * fs_hz / 1000.0))
    # Greedy non-maximum suppression prioritizes the most confident pulse.
    selected: list[int] = []
    for candidate in candidates[np.argsort(probability[candidates])[::-1]]:
        if all(abs(int(candidate) - existing) >= min_distance for existing in selected):
            selected.append(int(candidate))
    selected = sorted(selected)
    times = np.asarray(selected, dtype=float) * 1000.0 / fs_hz
    if offset_ms is not None and selected:
        times += np.asarray(offset_ms, dtype=float).reshape(-1)[selected]
    return times


def physiological_hrv(
    probability: np.ndarray,
    offset_ms: np.ndarray | None,
    sqi: np.ndarray | float,
    coverage: np.ndarray | float,
    fs_hz: float = 100.0,
    threshold: float = 0.35,
) -> BeatHRVResult:
    beats = decode_beat_times(probability, offset_ms, fs_hz, threshold)
    corrected = correct_rr_adjacency(beats)
    rmssd, sdnn = hrv_from_corrected_rr(corrected)
    sqi_mean = float(np.mean(sqi))
    coverage_mean = float(np.mean(coverage))
    duration_minutes = max(len(np.asarray(probability).reshape(-1)) / fs_hz / 60.0, 1e-6)
    expected_min_beats = 35.0 * duration_minutes
    count_factor = min(1.0, len(beats) / max(expected_min_beats, 1.0))
    confidence = float(np.clip(sqi_mean * coverage_mean * (1.0 - corrected.correction_ratio) * count_factor, 0, 1))
    reasons: list[str] = []
    if len(beats) < 4:
        reasons.append("insufficient_beats")
    if coverage_mean < 0.90:
        reasons.append("low_ppg_coverage")
    if sqi_mean < 0.30:
        reasons.append("low_sqi")
    if corrected.correction_ratio > 0.20:
        reasons.append("excessive_rr_correction")
    if not np.isfinite([rmssd, sdnn]).all():
        reasons.append("nonfinite_hrv")
    return BeatHRVResult(
        rmssd,
        sdnn,
        confidence,
        "|".join(dict.fromkeys(reasons)),
        beats,
        corrected,
    )


def reference_hrv_from_rpeaks(rpeaks_ms: np.ndarray) -> tuple[float, float]:
    """Auxiliary short-window ECG target; not the primary five-minute label."""
    return hrv_from_corrected_rr(correct_rr_adjacency(np.asarray(rpeaks_ms, dtype=float)))
