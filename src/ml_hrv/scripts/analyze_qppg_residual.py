"""Read-only audit of qPPG base estimates versus residual-model predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ml_hrv.evaluation.metrics import regression_metrics


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _summary(rows, source: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for metric in ("rmssd", "sdnn"):
        target = np.asarray([float(row["target_%s_ms" % metric]) for row in rows])
        prediction = np.asarray([float(row["%s_%s_ms" % (source, metric)]) for row in rows])
        values = regression_metrics(target, prediction)
        values["target_std_ms"] = float(np.std(target, ddof=1)) if len(target) > 1 else float("nan")
        values["prediction_std_ms"] = float(np.std(prediction, ddof=1)) if len(prediction) > 1 else float("nan")
        result[metric] = values
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare qPPG base and qPPG+neural residual predictions by qPPG validity."
    )
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    with Path(args.prediction_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "qppg_valid", "qppg_base_rmssd_ms", "qppg_base_sdnn_ms",
        "prediction_rmssd_ms", "prediction_sdnn_ms", "target_rmssd_ms", "target_sdnn_ms",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("prediction CSV is not a qPPG residual evaluation output")
    groups = {
        "all_windows": rows,
        "qppg_valid": [row for row in rows if _as_bool(row["qppg_valid"])],
        "qppg_invalid": [row for row in rows if not _as_bool(row["qppg_valid"])],
    }
    output = {
        "prediction_csv": str(Path(args.prediction_csv)),
        "interpretation": {
            "qppg_base": "Frozen qPPGFast estimate; invalid rows use the train-only qPPG median fallback.",
            "prediction": "qPPG base plus the neural residual correction.",
        },
        "groups": {
            name: {"n": len(group), "qppg_base": _summary(group, "qppg_base"), "residual_model": _summary(group, "prediction")}
            for name, group in groups.items()
        },
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for name, group in output["groups"].items():
        print("%s n=%d" % (name, group["n"]))
        for metric in ("rmssd", "sdnn"):
            base = group["qppg_base"][metric]
            model = group["residual_model"][metric]
            print(
                "  %s base_mae=%.3f residual_mae=%.3f base_r=%.3f residual_r=%.3f"
                % (metric, base["mae"], model["mae"], base["r"], model["r"])
            )
    print("wrote", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
