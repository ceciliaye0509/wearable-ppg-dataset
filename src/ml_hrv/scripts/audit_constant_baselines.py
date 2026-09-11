"""Compare frozen SegNet predictions with train-only constant predictors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ml_hrv.data.splits import load_folds
from ml_hrv.evaluation.metrics import regression_metrics
from ml_hrv.evaluation.reporting import read_prediction_rows


def participant_macro(rows, prediction_key: str, target_key: str) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["participant"])].append(row)
    metrics = []
    for group in grouped.values():
        metrics.append(
            regression_metrics(
                np.asarray([x[target_key] for x in group], float),
                np.asarray([x[prediction_key] for x in group], float),
            )
        )
    output = {"participants": len(metrics)}
    for key in ("mae", "rmse", "bias", "r", "r2", "ccc"):
        values = np.asarray([x[key] for x in metrics], float)
        output[key] = float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else None
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    folds = load_folds(args.run_dir / "folds.json")
    all_rows = read_prediction_rows(args.run_dir / "combined" / "window_predictions.csv")
    by_fold = defaultdict(list)
    for row in all_rows:
        by_fold[str(row["fold"])].append(row)
    result = {"protocol": "constants fitted on unique train windows only; scored on frozen outer-test rows", "folds": {}}
    pooled = defaultdict(list)
    for fold in folds:
        train_targets = []
        for participant in fold.train:
            with np.load(args.cache_dir / participant / "metadata.npz", allow_pickle=True) as z:
                keep = np.asarray(z["ecg_label_qc_pass"], bool)
                values = np.column_stack(
                    (z["ecg_rmssd_corrected_ms"], z["ecg_sdnn_corrected_ms"])
                ).astype(float)
                keep &= np.isfinite(values).all(1)
                train_targets.append(values[keep])
        target = np.concatenate(train_targets)
        constants = {
            "arithmetic_mean": np.mean(target, axis=0),
            "median": np.median(target, axis=0),
            "geometric_mean": np.exp(np.mean(np.log(np.clip(target, 1e-3, None)), axis=0)),
        }
        fold_result = {"train_unique_windows": len(target), "constants_ms": {}, "outer_test_participant_macro": {}}
        for method, values in constants.items():
            fold_result["constants_ms"][method] = {"rmssd": float(values[0]), "sdnn": float(values[1])}
            rows = []
            for original in by_fold[fold.name]:
                row = dict(original)
                row[f"{method}_rmssd_ms"] = float(values[0])
                row[f"{method}_sdnn_ms"] = float(values[1])
                rows.append(row)
                pooled[method].append(row)
            fold_result["outer_test_participant_macro"][method] = {
                metric: participant_macro(rows, f"{method}_{metric}_ms", f"target_{metric}_ms")
                for metric in ("rmssd", "sdnn")
            }
        result["folds"][fold.name] = fold_result
    result["combined_outer_test_participant_macro"] = {
        method: {
            metric: participant_macro(rows, f"{method}_{metric}_ms", f"target_{metric}_ms")
            for metric in ("rmssd", "sdnn")
        }
        for method, rows in pooled.items()
    }
    output = args.output or args.run_dir / "constant_baseline_audit.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["combined_outer_test_participant_macro"], indent=2))
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
