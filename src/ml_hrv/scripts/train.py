"""Train participant-disjoint raw-slot HRV folds and save every test window."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml_hrv.config import ExperimentConfig
from ml_hrv.data.dataset import RawslotWindowDataset, collate_rawslot
from ml_hrv.data.samplers import ParticipantDeviceBatchSampler
from ml_hrv.data.schema import discover_participant_files
from ml_hrv.data.splits import make_group_folds, save_folds
from ml_hrv.evaluation.evaluator import evaluate_model
from ml_hrv.evaluation.reporting import read_prediction_rows, write_run_artifacts
from ml_hrv.models.pipeline import RawContinuousHRVModel
from ml_hrv.training.checkpointing import load_checkpoint
from ml_hrv.training.losses import TargetScaler
from ml_hrv.training.trainer import Trainer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataset(config: ExperimentConfig, participants, trailing_seconds: int | None = None):
    return RawslotWindowDataset(
        config.data.cache_dir,
        participants,
        config.data.devices,
        config.data.segment_seconds,
        trailing_seconds or config.data.window_seconds,
        config.data.accel_mode,
        config.data.qc_only,
        config.data.cache_open_participants,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "configs" / "phase1.json"))
    parser.add_argument("--output-dir", default="artifacts/ml_hrv/phase1")
    parser.add_argument("--fold-index", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--short-window-seconds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batches-per-epoch", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument(
        "--skip-test-evaluation",
        action="store_true",
        help="Development-only: train/select on inner validation without loading outer test data.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional path template, e.g. artifacts/direct/checkpoints/{fold}_direct.pt",
    )
    parser.add_argument("--freeze-encoder-epochs", type=int, default=None)
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batches_per_epoch is not None:
        config.train.batches_per_epoch = args.batches_per_epoch
    if args.max_validation_batches is not None:
        config.train.max_validation_batches = args.max_validation_batches
    if args.max_test_batches is not None:
        config.train.max_test_batches = args.max_test_batches
    if args.freeze_encoder_epochs is not None:
        config.train.freeze_encoder_epochs = args.freeze_encoder_epochs
    if args.short_window_seconds is not None:
        if args.short_window_seconds not in config.evaluation.short_window_seconds:
            parser.error("short window must be one of the configured auxiliary durations")
    if config.train.stage == "beat_pretrain" and args.short_window_seconds not in {30, 60}:
        parser.error("beat_pretrain requires --short-window-seconds 30 or 60")
    config.validate()
    seed_everything(config.train.seed)
    if config.train.cpu_threads > 0:
        torch.set_num_threads(config.train.cpu_threads)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("mps" if args.device == "auto" and torch.backends.mps.is_available() else ("cpu" if args.device == "auto" else args.device))
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    participants = list(discover_participant_files(config.data.source_dir))
    folds = make_group_folds(participants, 4, config.train.seed)
    save_folds(folds, output / "folds.json")
    if args.fold_index is not None:
        folds = [folds[args.fold_index]]

    all_rows: list[dict[str, object]] = []
    trailing = args.short_window_seconds or config.data.window_seconds
    evaluate_outer_test = config.train.stage != "beat_pretrain" and not args.skip_test_evaluation
    for fold in folds:
        train_data = make_dataset(config, fold.train, trailing)
        val_data = make_dataset(config, fold.val, trailing)
        test_data = make_dataset(config, fold.test, trailing) if evaluate_outer_test else None
        scaler = TargetScaler.fit(train_data.target_array())
        train_sampler = ParticipantDeviceBatchSampler(
            train_data.sample_refs,
            config.train.windows_per_batch,
            config.train.batches_per_epoch,
            config.train.seed,
        )
        train_loader = DataLoader(
            train_data,
            batch_sampler=train_sampler,
            num_workers=config.train.num_workers,
            persistent_workers=config.train.persistent_workers,
            collate_fn=collate_rawslot,
        )
        validation_loader = DataLoader(
            val_data,
            batch_size=config.train.windows_per_batch * 3,
            shuffle=False,
            num_workers=config.train.num_workers,
            persistent_workers=config.train.persistent_workers,
            collate_fn=collate_rawslot,
        )
        test_loader = None if test_data is None else DataLoader(
            test_data,
            batch_size=config.train.windows_per_batch * 3,
            shuffle=False,
            num_workers=config.train.num_workers,
            persistent_workers=config.train.persistent_workers,
            collate_fn=collate_rawslot,
        )
        model = RawContinuousHRVModel(config)
        initialization_checkpoint = None
        if args.init_checkpoint:
            initialization_checkpoint = Path(args.init_checkpoint.format(fold=fold.name))
            initialized = load_checkpoint(initialization_checkpoint, device)
            initialized_fold = initialized.get("fold", {})
            for split_name in ("train", "val", "test"):
                if tuple(map(str, initialized_fold.get(split_name, ()))) != tuple(
                    map(str, getattr(fold, split_name))
                ):
                    raise ValueError(
                        f"initialization checkpoint {initialization_checkpoint} has a different {split_name} split"
                    )
            model.load_state_dict(initialized["model_state_dict"])
        trainer = Trainer(model, config, scaler, device)
        checkpoint_path = trainer.fit(train_loader, validation_loader, fold, output)
        checkpoint = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        # Beat pretraining is selected only on inner validation event loss.  It
        # deliberately does not decode or inspect the outer test participants.
        rows = [] if not evaluate_outer_test else evaluate_model(
            model,
            test_loader,
            scaler,
            config,
            device,
            fold.name,
            beat_trained=config.train.stage in {"beat", "joint"},
            max_batches=config.train.max_test_batches,
        )
        run_metadata = {
            "config": config.to_dict(),
            "fold": fold.to_dict(),
            "checkpoint": str(checkpoint_path.resolve()),
            "device": str(device),
            "trailing_seconds": trailing,
            "label_scope": rows[0].get("label_source", "see dataset contract") if rows else "none",
            "initialization_checkpoint": str(initialization_checkpoint.resolve())
            if initialization_checkpoint
            else None,
        }
        if not evaluate_outer_test:
            fold_output = output / fold.name
            fold_output.mkdir(parents=True, exist_ok=True)
            run_metadata["label_scope"] = (
                "ECG beat events only; outer test not loaded or decoded"
                if config.train.stage == "beat_pretrain"
                else "development run; outer test not loaded or evaluated"
            )
            (fold_output / "run_metadata.json").write_text(
                json.dumps(run_metadata, indent=2), encoding="utf-8"
            )
            (fold_output / "summary.md").write_text(
                f"# {'Beat pretraining' if config.train.stage == 'beat_pretrain' else 'Inner-validation development run'}\n\n"
                "Outer-test participants were not loaded or evaluated.\n",
                encoding="utf-8",
            )
        else:
            write_run_artifacts(output / fold.name, rows, run_metadata)
        all_rows.extend(rows)
    # A user may run one fold at a time on CPU. Rebuild combined artifacts from
    # every completed fold already present in this output directory.
    completed_rows: list[dict[str, object]] = []
    prediction_files = (
        sorted(output.glob("fold_*/window_predictions.csv"))
        if evaluate_outer_test
        else ()
    )
    for predictions in prediction_files:
        completed_rows.extend(read_prediction_rows(predictions))
    if completed_rows:
        write_run_artifacts(
            output / "combined",
            completed_rows,
            {
                "config": config.to_dict(),
                "completed_folds": sorted({str(row["fold"]) for row in completed_rows}),
                "trailing_seconds": trailing,
            },
        )
    print(
        json.dumps(
            {"output": str(output.resolve()), "new_rows": len(all_rows), "combined_rows": len(completed_rows)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
