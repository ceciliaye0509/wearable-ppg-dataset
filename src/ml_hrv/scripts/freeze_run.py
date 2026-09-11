"""Create a content-hash manifest for a completed cross-validation run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ml_hrv.evaluation.reporting import read_prediction_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-folds", type=int, default=4)
    args = parser.parse_args()
    fold_predictions = sorted(args.run_dir.glob("fold_*/window_predictions.csv"))
    if len(fold_predictions) != args.expected_folds:
        parser.error(f"expected {args.expected_folds} completed folds, found {len(fold_predictions)}")
    combined = args.run_dir / "combined" / "window_predictions.csv"
    combined_rows = read_prediction_rows(combined)
    fold_rows = sum(len(read_prediction_rows(path)) for path in fold_predictions)
    if fold_rows != len(combined_rows):
        raise ValueError(f"combined row count {len(combined_rows)} != fold total {fold_rows}")
    paths = [args.config, args.run_dir / "folds.json", combined]
    paths.extend(sorted((args.run_dir / "checkpoints").glob("*.pt")))
    paths.extend(sorted(args.run_dir.glob("fold_*/metrics.json")))
    paths.extend(sorted(args.run_dir.glob("fold_*/window_predictions.csv")))
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "expected_folds": args.expected_folds,
        "combined_rows": len(combined_rows),
        "target_order": ["rmssd_ms", "sdnn_ms"],
        "primary_input": "ppg_rawslot_values",
        "files": {
            str(path.resolve()): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in paths
        },
    }
    output = args.run_dir / "FROZEN.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
