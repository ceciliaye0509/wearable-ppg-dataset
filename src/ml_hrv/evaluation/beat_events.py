"""One-to-one ECG-referenced event metrics for the decoded PPG beat path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BeatEventMetrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    timing_mae_ms: float
    count_error: int


def match_beat_events(
    predicted_ppg_ms: np.ndarray,
    reference_ecg_ms: np.ndarray,
    pulse_delay_ms: float,
    tolerance_ms: float = 100.0,
) -> BeatEventMetrics:
    """Match ordered events after applying the model's ECG→PPG delay."""
    predicted = np.sort(np.asarray(predicted_ppg_ms, dtype=float))
    reference = np.sort(np.asarray(reference_ecg_ms, dtype=float) + float(pulse_delay_ms))
    predicted = predicted[np.isfinite(predicted)]
    reference = reference[np.isfinite(reference)]
    pred_index = ref_index = 0
    errors = []
    fp = fn = 0
    while pred_index < len(predicted) and ref_index < len(reference):
        delta = predicted[pred_index] - reference[ref_index]
        if abs(delta) <= tolerance_ms:
            errors.append(abs(float(delta)))
            pred_index += 1
            ref_index += 1
        elif delta < -tolerance_ms:
            fp += 1
            pred_index += 1
        else:
            fn += 1
            ref_index += 1
    fp += len(predicted) - pred_index
    fn += len(reference) - ref_index
    tp = len(errors)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BeatEventMetrics(
        tp, fp, fn, precision, recall, f1,
        float(np.mean(errors)) if errors else float("nan"),
        int(len(predicted) - len(reference)),
    )
