"""Fold trainer with early stopping and complete checkpoints."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch

from ml_hrv.config import ExperimentConfig
from ml_hrv.data.splits import Fold

from .checkpointing import save_checkpoint
from .losses import PipelineLoss, TargetScaler


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: ExperimentConfig,
        scaler: TargetScaler,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.scaler = scaler
        self.device = device
        self.loss = PipelineLoss(config, scaler)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay
        )

    def _epoch(self, loader, training: bool, max_batches: int = 0) -> dict[str, float]:
        self.model.train(training)
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for raw_batch in loader:
                batch = _move_batch(raw_batch, self.device)
                outputs = self.model(
                    batch["values"], batch["mask"], batch["jitter"], batch["device_id"], batch["accel"],
                    compute_beat=self.config.train.stage in {"beat_pretrain", "beat", "joint"},
                )
                losses = self.loss(outputs, batch)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip_norm)
                    self.optimizer.step()
                for key, value in losses.items():
                    totals[key] += float(value.detach())
                batches += 1
                if max_batches and batches >= max_batches:
                    break
        if batches == 0:
            raise ValueError("DataLoader produced no batches")
        return {key: value / batches for key, value in totals.items()}

    def fit(self, train_loader, val_loader, fold: Fold, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        checkpoint = output_dir / "checkpoints" / f"{fold.name}_{self.config.train.stage}.pt"
        history: list[dict[str, float]] = []
        best = float("inf")
        stale = 0
        for epoch in range(1, self.config.train.epochs + 1):
            encoder_trainable = epoch > self.config.train.freeze_encoder_epochs
            for parameter in self.model.encoder.parameters():
                parameter.requires_grad_(encoder_trainable)
            train_metrics = self._epoch(train_loader, True)
            val_metrics = self._epoch(
                val_loader, False, max_batches=self.config.train.max_validation_batches
            )
            row = {"epoch": float(epoch)}
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(row)
            print(
                f"{fold.name} epoch={epoch:03d} "
                f"train={train_metrics['total']:.5f} val={val_metrics['total']:.5f}",
                flush=True,
            )
            if val_metrics["total"] < best:
                best = val_metrics["total"]
                stale = 0
                save_checkpoint(
                    checkpoint,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_validation_loss=best,
                    config=self.config.to_dict(),
                    fold=fold.to_dict(),
                    target_scaler=self.scaler.state_dict(),
                    history=history,
                )
            else:
                stale += 1
                if stale >= self.config.train.patience:
                    break
        return checkpoint
