"""Fast synthetic end-to-end check; does not stage or train the large dataset."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from ml_hrv.config import ExperimentConfig
from ml_hrv.models.pipeline import RawContinuousHRVModel
from ml_hrv.physiology import correct_rr_adjacency, hrv_from_corrected_rr
from ml_hrv.training.checkpointing import load_checkpoint, save_checkpoint
from ml_hrv.training.losses import PipelineLoss, TargetScaler


def main() -> int:
    config = ExperimentConfig()
    config.data.fs_hz = 20.0
    config.data.window_seconds = 20
    config.data.segment_seconds = 10
    config.data.update_seconds = 5
    config.model.width = 8
    config.model.token_dim = 16
    config.model.tcn_layers = 2
    config.train.stage = "joint"
    config.train.consistency_weight = 0.1
    model = RawContinuousHRVModel(config)
    batch, segments, length = 3, 2, 200
    values = torch.randn(batch, segments, 2, length)
    mask = torch.rand(batch, segments, 2, length) > 0.05
    jitter = torch.randn_like(values) * 0.1
    accel = torch.rand(batch, 1)
    outputs = model(values, mask, jitter, torch.arange(3), accel)
    assert outputs["direct_loc"].shape == (batch, 2)
    assert outputs["beat_heatmap_logits"].shape == (batch, segments, length)

    scaler = TargetScaler.fit(np.array([[20, 40], [30, 50], [40, 60]], dtype=float))
    peak_lists = [torch.tensor([1000.0, 2000.0, 3000.0, 4000.0])] * batch
    losses = PipelineLoss(config, scaler)(
        outputs,
        {
            "target_ms": torch.tensor([[25.0, 45.0]] * batch),
            "group_id": torch.zeros(batch, dtype=torch.long),
            "rpeaks_ms": peak_lists,
        },
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])

    # 400 + 600 ms should merge into one 1000 ms RR without joining across gaps.
    corrected = correct_rr_adjacency(np.array([0, 1000, 1400, 2000, 3000, 4000], dtype=float))
    rmssd, sdnn = hrv_from_corrected_rr(corrected)
    assert np.isfinite([rmssd, sdnn]).all()

    optimizer = torch.optim.AdamW(model.parameters())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.pt"
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            epoch=1,
            best_validation_loss=float(losses["total"].detach()),
            config=config.to_dict(),
            fold={"name": "smoke", "train": ["P1"], "val": ["P2"], "test": ["P3"]},
            target_scaler=scaler.state_dict(),
            history=[],
        )
        load_checkpoint(path)
    print("ml_hrv synthetic smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
