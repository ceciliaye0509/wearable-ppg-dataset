"""Evaluate a saved raw-slot checkpoint only on its frozen inner validation split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml_hrv.config import DataConfig, EvaluationConfig, ExperimentConfig, ModelConfig, TrainConfig
from ml_hrv.data.dataset import RawslotWindowDataset, collate_rawslot
from ml_hrv.data.splits import Fold, load_folds
from ml_hrv.evaluation.evaluator import evaluate_model
from ml_hrv.evaluation.metrics import regression_metrics
from ml_hrv.models.pipeline import RawContinuousHRVModel
from ml_hrv.training.checkpointing import load_checkpoint
from ml_hrv.training.losses import TargetScaler


def _config(raw):
    return ExperimentConfig(
        data=DataConfig(**raw["data"]), model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]), evaluation=EvaluationConfig(**raw["evaluation"]),
    )


def _dataset(config, participants):
    return RawslotWindowDataset(
        config.data.cache_dir, participants, config.data.devices, config.data.segment_seconds,
        config.data.window_seconds, config.data.accel_mode, config.data.qc_only,
        config.data.cache_open_participants, ppg_channels=config.data.ppg_channels,
    )


def _metric(rows, metric, prediction):
    target = np.asarray([row["target_%s_ms" % metric] for row in rows], dtype=float)
    result = regression_metrics(target, np.asarray(prediction, dtype=float))
    result["target_std_ms"] = float(np.std(target, ddof=1))
    result["prediction_std_ms"] = float(np.std(prediction, ddof=1))
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate only a checkpoint's inner validation split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    payload = load_checkpoint(args.checkpoint, "cpu")
    config = _config(payload["config"])
    config.validate()
    saved_fold = Fold(payload["fold"]["name"], tuple(payload["fold"]["train"]), tuple(payload["fold"]["val"]), tuple(payload["fold"]["test"]))
    frozen = {fold.name: fold for fold in load_folds(Path(args.run_dir) / "folds.json")}
    if frozen.get(saved_fold.name) != saved_fold:
        raise RuntimeError("checkpoint split differs from run-dir folds.json")
    train_data, val_data = _dataset(config, saved_fold.train), _dataset(config, saved_fold.val)
    scaler = TargetScaler.from_state_dict(payload["target_scaler"])
    recomputed = TargetScaler.fit(train_data.target_array())
    if not (np.allclose(scaler.mean, recomputed.mean) and np.allclose(scaler.std, recomputed.std)):
        raise RuntimeError("checkpoint target scaler is not train-only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.train.cpu_threads:
        torch.set_num_threads(config.train.cpu_threads)
    loader = DataLoader(val_data, batch_size=config.train.windows_per_batch * 3, shuffle=False, num_workers=0, collate_fn=collate_rawslot)
    model = RawContinuousHRVModel(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    rows = evaluate_model(model, loader, scaler, config, device, saved_fold.name, beat_trained=False)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "window_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    train_targets = train_data.target_array()
    metrics = {}
    for index, metric in enumerate(("rmssd", "sdnn")):
        baseline = float(np.median(train_targets[:, index]))
        metrics[metric] = {
            "checkpoint": _metric(rows, metric, [row["prediction_%s_ms" % metric] for row in rows]),
            "train_median_baseline": _metric(rows, metric, np.full(len(rows), baseline)),
            "train_median_ms": baseline,
        }
    report = {
        "checkpoint": str(Path(args.checkpoint)), "checkpoint_epoch": int(payload["epoch"]),
        "best_validation_loss": float(payload["best_validation_loss"]), "fold": saved_fold.to_dict(),
        "n_validation_rows": len(rows), "train_only_scaler_match": True, "metrics": metrics,
    }
    (output / "inner_validation_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
