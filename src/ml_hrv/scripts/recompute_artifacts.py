"""Rebuild metrics/summary from saved predictions, optionally restoring motion metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml_hrv import DEVICE_NAMES
from ml_hrv.evaluation.reporting import read_prediction_rows, write_run_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    rows = read_prediction_rows(args.predictions_csv)
    if args.cache_dir is not None:
        metadata: dict[str, np.ndarray] = {}
        for participant in {str(row["participant"]) for row in rows}:
            with np.load(args.cache_dir / participant / "metadata.npz", allow_pickle=True) as data:
                metadata[participant] = np.asarray(data["accel_motion_mean_mag"], dtype=float)
        for row in rows:
            participant = str(row["participant"])
            window = int(row["window_index"])
            device = DEVICE_NAMES.index(str(row["device"]))
            row["accel_motion_mean_mag"] = float(metadata[participant][window, device])
    output = args.output_dir or args.predictions_csv.parent
    metadata_path = output / "run_metadata.json"
    run_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    run_metadata["artifacts_recomputed_from"] = str(args.predictions_csv.resolve())
    run_metadata["motion_backfilled_from"] = str(args.cache_dir.resolve()) if args.cache_dir else None
    paths = write_run_artifacts(
        output,
        rows,
        run_metadata,
    )
    print(json.dumps({key: str(path.resolve()) for key, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
