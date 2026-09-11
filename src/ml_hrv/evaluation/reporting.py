"""Write CSV predictions, machine-readable metrics, and a compact Markdown report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .metrics import summarize_predictions


TEXT_FIELDS = {
    "fold", "participant", "device", "label_source", "beat_reject_reason",
    "reject_reason", "prediction_source",
}
BOOL_FIELDS = {"accepted"}
INT_FIELDS = {
    "window_index", "beat_count", "beat_event_tp", "beat_event_fp",
    "beat_event_fn", "beat_count_error",
}


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def read_prediction_rows(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows: list[dict[str, object]] = list(csv.DictReader(stream))
    for row in rows:
        for key, value in list(row.items()):
            if key in TEXT_FIELDS:
                continue
            if key in BOOL_FIELDS:
                row[key] = str(value).lower() == "true"
            elif key in INT_FIELDS:
                row[key] = int(float(value))
            else:
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
    return rows


def write_run_artifacts(
    output_dir: str | Path,
    rows: list[dict[str, object]],
    run_metadata: dict[str, object],
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "window_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = summarize_predictions(rows)
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
    metadata_path = output / "run_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2, default=_json_default), encoding="utf-8")

    overall = metrics["participant_macro"]["prediction"]
    lines = [
        "# Continuous raw-slot HRV run summary",
        "",
        f"- Rows: {metrics['n_rows']}",
        f"- Accepted: {metrics['acceptance_rate']:.1%}",
        "- Primary input: `ppg_rawslot_values` + mask + timestamp jitter (no interpolated PPG)",
        "- Target order: RMSSD, SDNN; fields are the corrected ECG labels",
        "",
        "## Participant-macro primary result",
        "",
        "| Metric | MAE (ms) | RMSE (ms) | r | R² |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in ("rmssd", "sdnn"):
        value = overall[metric]
        lines.append(
            f"| {metric.upper()} | {value['mae']:.3f} | {value['rmse']:.3f} | {value['r']:.3f} | {value['r2']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Per-device, motion-stratified, coverage-matched, and selective-risk results are in `metrics.json`.",
            "The no-PPG-coverage-prefilter challenge set is intentionally not generated in phase one.",
            "",
        ]
    )
    summary_path = output / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "predictions": predictions_path,
        "metrics": metrics_path,
        "metadata": metadata_path,
        "summary": summary_path,
    }
