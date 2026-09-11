"""Atomic, self-describing checkpoints for reproducible fold evaluation."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch
import numpy as np


CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation_loss: float,
    config: dict[str, Any],
    fold: dict[str, Any],
    target_scaler: dict[str, Any],
    history: list[dict[str, float]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "fold": fold,
        "target_scaler": target_scaler,
        "target_order": ["rmssd_ms", "sdnn_ms"],
        "primary_input": "ppg_rawslot_values",
        "history": history,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {payload.get('checkpoint_version')}")
    if payload.get("target_order") != ["rmssd_ms", "sdnn_ms"]:
        raise ValueError("Checkpoint target order is not RMSSD, SDNN")
    if payload.get("primary_input") != "ppg_rawslot_values":
        raise ValueError("Checkpoint was not trained on raw-slot PPG")
    return payload
