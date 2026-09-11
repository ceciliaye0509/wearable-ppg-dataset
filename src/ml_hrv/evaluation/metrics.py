"""Metrics required for continuous-HRV claims and confidence auditing."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np


def _finite_pair(target: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(target) & np.isfinite(prediction)
    return target[valid], prediction[valid]


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    target, prediction = _finite_pair(np.asarray(target, float), np.asarray(prediction, float))
    if len(target) == 0:
        return {key: float("nan") for key in ("mae", "rmse", "bias", "loa_low", "loa_high", "r", "r2", "ccc")} | {"n": 0}
    error = prediction - target
    sd_error = float(np.std(error, ddof=1)) if len(error) > 1 else float("nan")
    r = float(np.corrcoef(target, prediction)[0, 1]) if len(target) > 1 and target.std() > 0 and prediction.std() > 0 else float("nan")
    denominator = float(np.square(target - target.mean()).sum())
    target_var = float(np.var(target))
    prediction_var = float(np.var(prediction))
    covariance = float(np.mean((target - target.mean()) * (prediction - prediction.mean())))
    ccc_denominator = target_var + prediction_var + float(target.mean() - prediction.mean()) ** 2
    return {
        "n": int(len(target)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "loa_low": float(error.mean() - 1.96 * sd_error),
        "loa_high": float(error.mean() + 1.96 * sd_error),
        "r": r,
        "r2": float(1.0 - np.square(error).sum() / denominator) if denominator > 0 else float("nan"),
        "ccc": 2.0 * covariance / ccc_denominator if ccc_denominator > 0 else float("nan"),
    }


def _metric_pair(rows: list[dict[str, object]], metric: str, source: str) -> dict[str, float | int]:
    target = np.asarray([x[f"target_{metric}_ms"] for x in rows], dtype=float)
    prediction = np.asarray([x[f"{source}_{metric}_ms"] for x in rows], dtype=float)
    return regression_metrics(target, prediction)


def _group(rows: list[dict[str, object]], key: Callable[[dict[str, object]], str]) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return dict(groups)


def _two_metric_summary(rows: list[dict[str, object]], source: str) -> dict[str, object]:
    return {metric: _metric_pair(rows, metric, source) for metric in ("rmssd", "sdnn")}


def _participant_macro(rows: list[dict[str, object]], source: str) -> dict[str, dict[str, float]]:
    participants = _group(rows, lambda x: str(x["participant"]))
    result: dict[str, dict[str, float]] = {}
    for metric in ("rmssd", "sdnn"):
        fold_values = [_metric_pair(group, metric, source) for group in participants.values()]
        result[metric] = {}
        for name in ("mae", "rmse", "bias", "r", "r2", "ccc"):
            values = np.asarray([float(x[name]) for x in fold_values])
            result[metric][name] = float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else float("nan")
        result[metric]["participants"] = float(len(participants))
    return result


def _participant_bootstrap_mae(
    rows: list[dict[str, object]], source: str, repeats: int = 2000, seed: int = 20260911
) -> dict[str, dict[str, float]]:
    groups = _group(rows, lambda x: str(x["participant"]))
    participants = sorted(groups)
    rng = np.random.default_rng(seed)
    output: dict[str, dict[str, float]] = {}
    for metric in ("rmssd", "sdnn"):
        values = np.asarray([float(_metric_pair(groups[p], metric, source)["mae"]) for p in participants])
        draws = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(1)
        output[metric] = {
            "mae_ms": float(values.mean()),
            "ci95_low_ms": float(np.quantile(draws, 0.025)),
            "ci95_high_ms": float(np.quantile(draws, 0.975)),
            "bootstrap_unit": "participant",
        }
    return output


def _selective_risk(rows: list[dict[str, object]], metric: str = "rmssd") -> list[dict[str, float]]:
    usable = [x for x in rows if np.isfinite(float(x["confidence"]))]
    usable.sort(key=lambda x: float(x["confidence"]), reverse=True)
    result = []
    for retained in (1.0, 0.9, 0.8, 0.5):
        count = max(1, round(len(usable) * retained))
        subset = usable[:count]
        mae = _metric_pair(subset, metric, "prediction")["mae"]
        result.append({"retained_fraction": retained, "n": float(count), "mae_ms": float(mae)})
    return result


def _uncertainty_audit(rows: list[dict[str, object]], metric: str) -> dict[str, float]:
    confidence = np.asarray([float(x["confidence"]) for x in rows], dtype=float)
    target = np.asarray([float(x[f"target_{metric}_ms"]) for x in rows])
    prediction = np.asarray([float(x[f"prediction_{metric}_ms"]) for x in rows])
    error = np.abs(prediction - target)
    valid = np.isfinite(confidence) & np.isfinite(error)
    confidence, error = confidence[valid], error[valid]
    if len(error) == 0:
        return {"high_error_threshold_ms": float("nan"), "overconfident_high_error_rate": float("nan")}
    threshold = float(np.quantile(error, 0.90))
    high_error = error >= threshold
    return {
        "high_error_threshold_ms": threshold,
        "overconfident_high_error_rate": float(np.mean(confidence[high_error] >= 0.80)),
        "mean_confidence_high_error": float(np.mean(confidence[high_error])),
        "mean_confidence_other": float(np.mean(confidence[~high_error])) if (~high_error).any() else float("nan"),
    }


def _multi_device_upper_bound(rows: list[dict[str, object]]) -> dict[str, object]:
    """Precision-fuse matched devices only as a non-deployment upper bound."""
    grouped = _group(rows, lambda x: f"{x['participant']}:{x['window_index']}")
    fused_rows: list[dict[str, object]] = []
    for group in grouped.values():
        if len({str(x["device"]) for x in group}) < 2:
            continue
        output: dict[str, object] = {}
        for metric in ("rmssd", "sdnn"):
            predictions = np.asarray([float(x[f"direct_{metric}_ms"]) for x in group])
            std = np.asarray([float(x[f"direct_{metric}_std_ms"]) for x in group])
            precision = 1.0 / np.square(np.clip(std, 1.0, None))
            output[f"prediction_{metric}_ms"] = float(np.sum(predictions * precision) / np.sum(precision))
            output[f"target_{metric}_ms"] = float(group[0][f"target_{metric}_ms"])
        fused_rows.append(output)
    return _two_metric_summary(fused_rows, "prediction") if fused_rows else {}


def _beat_event_summary(rows: list[dict[str, object]]) -> dict[str, float | int]:
    usable = [x for x in rows if np.isfinite(float(x.get("beat_event_f1", float("nan"))))]
    if not usable:
        return {}
    tp = sum(int(x["beat_event_tp"]) for x in usable)
    fp = sum(int(x["beat_event_fp"]) for x in usable)
    fn = sum(int(x["beat_event_fn"]) for x in usable)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    timing_numerator = sum(
        float(x["beat_timing_mae_ms"]) * int(x["beat_event_tp"])
        for x in usable if np.isfinite(float(x["beat_timing_mae_ms"]))
    )
    timing_denominator = sum(
        int(x["beat_event_tp"])
        for x in usable if np.isfinite(float(x["beat_timing_mae_ms"]))
    )
    return {
        "windows": len(usable), "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "timing_mae_ms": timing_numerator / timing_denominator if timing_denominator else float("nan"),
        "count_mae": float(np.mean([abs(int(x["beat_count_error"])) for x in usable])),
        "matching_tolerance_ms": 100.0,
        "reference_alignment": "ECG R peak + model-predicted pulse delay",
    }


def summarize_predictions(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("No prediction rows")
    sources = ["prediction", "direct"]
    if any(np.isfinite(float(x["beat_rmssd_ms"])) for x in rows):
        sources.append("beat")
    summary: dict[str, object] = {
        "n_rows": len(rows),
        "sources": {source: _two_metric_summary(rows, source) for source in sources},
        "participant_macro": {source: _participant_macro(rows, source) for source in sources},
        "participant_bootstrap_mae": {
            source: _participant_bootstrap_mae(rows, source) for source in sources
        },
    }
    by_device = _group(rows, lambda x: str(x["device"]))
    summary["by_device"] = {
        device: {source: _two_metric_summary(group, source) for source in sources}
        for device, group in by_device.items()
    }
    summary["by_device_participant_macro"] = {
        device: {source: _participant_macro(group, source) for source in sources}
        for device, group in by_device.items()
    }

    motion = np.asarray([float(x["accel_motion_mean_mag"]) for x in rows])
    cuts = np.quantile(motion[np.isfinite(motion)], [0.25, 0.75]) if np.isfinite(motion).any() else [0, 0]
    def motion_key(row):
        value = float(row["accel_motion_mean_mag"])
        return "low" if value <= cuts[0] else ("high" if value >= cuts[1] else "middle")
    by_motion = _group(rows, motion_key)
    summary["by_motion"] = {name: _two_metric_summary(group, "prediction") for name, group in by_motion.items()}

    def coverage_key(row):
        value = float(row["coverage"])
        if value < 0.95:
            return "[0.90,0.95)"
        if value < 0.98:
            return "[0.95,0.98)"
        return "[0.98,1.00]"
    by_coverage = _group(rows, coverage_key)
    summary["by_coverage"] = {
        name: _two_metric_summary(group, "prediction") for name, group in by_coverage.items()
    }
    summary["device_by_coverage"] = {
        coverage_name: {
            device: _two_metric_summary(device_rows, "prediction")
            for device, device_rows in _group(coverage_rows, lambda x: str(x["device"])).items()
        }
        for coverage_name, coverage_rows in by_coverage.items()
    }
    summary["selective_risk"] = {
        metric: _selective_risk(rows, metric) for metric in ("rmssd", "sdnn")
    }
    summary["acceptance_rate"] = float(np.mean([bool(x["accepted"]) for x in rows]))
    summary["uncertainty_audit"] = {
        metric: _uncertainty_audit(rows, metric) for metric in ("rmssd", "sdnn")
    }
    summary["multi_device_fusion_upper_bound"] = _multi_device_upper_bound(rows)
    summary["beat_event_metrics"] = _beat_event_summary(rows)
    return summary
