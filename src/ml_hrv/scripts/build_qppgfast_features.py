"""Build frozen qPPGFast features for one raw-slot HRV development fold.

The output is PPG-only.  It deliberately contains no ECG target fields so it
can be merged into a residual-correction model without label leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml_hrv.config import ExperimentConfig
from ml_hrv.data.splits import load_folds


FROZEN_QPPGFAST = {
    "detector": "qppgfast",
    "fiducial": "peak",
    "input": "strict interpolation to 100 Hz; max raw support gap 100 ms; no boundary extrapolation",
    "common_bandpass_hz": [0.7, 3.5],
    "ibi_correction": True,
    "ibi_correction_threshold": 0.20,
    "polarity_mode": "peak_train",
    "device": "Earring",
    "channel": "green",
}


def _raw_paths(source_dir: Path, participants: set[str]) -> list[Path]:
    paths = sorted(source_dir.glob("*.npz"))
    found = {path.stem.rsplit("_", 1)[-1].upper() for path in paths}
    missing = sorted(participants - found)
    if missing:
        raise FileNotFoundError(f"raw-slot NPZ files missing for: {missing}")
    return [path for path in paths if path.stem.rsplit("_", 1)[-1].upper() in participants]


def _load_frozen_rows(path: Path) -> pd.DataFrame:
    heuristic_root = Path(__file__).resolve().parents[2] / "heuristic_baselines"
    if str(heuristic_root) not in sys.path:
        sys.path.insert(0, str(heuristic_root))
    from evaluate_rawslots_detector_fiducials import DetectorPipeline, _load_channel_metrics_for_path

    with np.load(path, allow_pickle=True) as payload:
        count = int(np.asarray(payload["ecg_rmssd_corrected_ms"]).shape[0])
    rows = _load_channel_metrics_for_path(
        path,
        np.arange(count, dtype=int),
        (DetectorPipeline("qppgfast", (0.7, 3.5)),),
        input_mode="strict_interp100",
        max_gap_ms=100.0,
        correction_states=(True,),
        polarity_mode="peak_train",
        selected_devices=("Earring",),
    )
    rows = rows[
        (rows["detector"] == "qppgfast")
        & (rows["fiducial"] == "peak")
        & (rows["device"] == "Earring")
        & (rows["channel"] == "green")
    ].copy()
    keep = [
        "participant", "window_index", "device", "channel",
        "ppg_rmssd_ms", "ppg_sdnn_ms", "ppg_mean_ibi_ms", "ppg_hr_bpm",
        "n_fiducials", "ppg_valid_ibi_ratio", "ppg_ibi_cv",
        "ppg_ibi_correction_ratio", "ppg_sqi", "ppg_valid_sample_ratio",
        "ppg_max_raw_gap_ms", "accel_motion_mean_mag", "polarity",
    ]
    result = rows[keep].copy()
    result.rename(columns={
        "ppg_rmssd_ms": "qppg_rmssd_ms",
        "ppg_sdnn_ms": "qppg_sdnn_ms",
        "ppg_mean_ibi_ms": "qppg_mean_ibi_ms",
        "ppg_hr_bpm": "qppg_hr_bpm",
        "n_fiducials": "qppg_peak_count",
        "ppg_valid_ibi_ratio": "qppg_valid_ibi_ratio",
        "ppg_ibi_cv": "qppg_ibi_cv",
        "ppg_ibi_correction_ratio": "qppg_ibi_correction_ratio",
        "ppg_sqi": "qppg_sqi",
        "ppg_valid_sample_ratio": "qppg_valid_sample_ratio",
        "ppg_max_raw_gap_ms": "qppg_max_raw_gap_ms",
        "polarity": "qppg_polarity",
    }, inplace=True)
    result["qppg_valid"] = np.isfinite(result["qppg_rmssd_ms"]) & np.isfinite(result["qppg_sdnn_ms"])
    return result


def _summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "qppg_valid_rows": int(frame["qppg_valid"].sum()),
        "qppg_valid_fraction": float(frame["qppg_valid"].mean()) if len(frame) else float("nan"),
        "unique_keys": int(frame[["participant", "window_index", "device", "channel"]].drop_duplicates().shape[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build frozen qPPGFast features for one fold.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--fold-name", default="fold_0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    if tuple(config.data.devices) != ("Earring",) or tuple(config.data.ppg_channels) != ("green",):
        raise ValueError("This frozen adapter is limited to the Earring green comparison contract")
    folds = {fold.name: fold for fold in load_folds(Path(args.run_dir) / "folds.json")}
    fold = folds.get(args.fold_name)
    if fold is None:
        raise KeyError(f"No {args.fold_name} in {args.run_dir}/folds.json")

    split_by_participant = {participant: "train" for participant in fold.train}
    split_by_participant.update({participant: "val" for participant in fold.val})
    frames = []
    for path in _raw_paths(Path(config.data.source_dir), set(split_by_participant)):
        participant = path.stem.rsplit("_", 1)[-1].upper()
        frame = _load_frozen_rows(path)
        frame["split"] = split_by_participant[participant]
        frames.append(frame)
        print(f"[qppgfast-features] {participant} rows={len(frame)} valid={int(frame['qppg_valid'].sum())}", flush=True)
    features = pd.concat(frames, ignore_index=True)
    keys = ["participant", "window_index", "device", "channel"]
    if features.duplicated(keys).any():
        raise RuntimeError("qPPGFast output has duplicate window keys")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features.to_csv(output / "qppgfast_features.csv", index=False)
    train = features[features["split"] == "train"].copy()
    val = features[features["split"] == "val"].copy()
    train.to_csv(output / "qppgfast_features_train.csv", index=False)
    val.to_csv(output / "qppgfast_features_val.csv", index=False)
    audit = {
        "contract": {"name": "comparison_rawslot100_v1", **FROZEN_QPPGFAST},
        "fold": fold.to_dict(),
        "all": _summary(features),
        "train": _summary(train),
        "val": _summary(val),
        "source_dir": str(Path(config.data.source_dir)),
        "label_leakage": "none: output fields are computed from PPG, timestamps, and accelerometer metadata only",
    }
    (output / "qppgfast_feature_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
