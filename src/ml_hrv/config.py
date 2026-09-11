"""Typed experiment configuration with conservative phase-one defaults."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    source_dir: str = "src/heuristic_baselines/outputs/synced_3device_rawaligned_training_v1_stride30_rawslots"
    cache_dir: str = "artifacts/ml_hrv/rawslot_cache"
    fs_hz: float = 100.0
    window_seconds: int = 300
    update_seconds: int = 30
    segment_seconds: int = 10
    devices: tuple[str, ...] = ("Earring", "Ring", "Watch")
    accel_mode: str = "scalar"  # none | scalar; raw_xyz is a later ablation.
    qc_only: bool = True
    cache_open_participants: int = 4


@dataclass
class ModelConfig:
    direct_architecture: str = "causal_tcn"  # causal_tcn | segnet_mean
    use_timestamp_jitter: bool = True
    width: int = 24
    token_dim: int = 96
    tcn_layers: int = 4
    dropout: float = 0.15
    device_embedding_dim: int = 8
    delay_min_ms: float = 100.0
    delay_max_ms: float = 700.0
    fusion_enabled: bool = False


@dataclass
class TrainConfig:
    seed: int = 20260911
    stage: str = "direct"  # direct | beat_pretrain | beat | joint
    epochs: int = 60
    patience: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    windows_per_batch: int = 4
    batches_per_epoch: int = 256
    num_workers: int = 0
    persistent_workers: bool = False
    cpu_threads: int = 0  # 0 keeps the PyTorch process default.
    freeze_encoder_epochs: int = 0
    consistency_weight: float = 0.10
    beat_weight: float = 1.0
    offset_weight: float = 0.20
    sqi_weight: float = 0.10
    grad_clip_norm: float = 5.0
    max_validation_batches: int = 0  # 0 = complete split; nonzero is smoke/debug only.
    max_test_batches: int = 0


@dataclass
class EvaluationConfig:
    short_window_seconds: tuple[int, ...] = (30, 60, 120)
    coverage_bins: tuple[float, ...] = (0.90, 0.95, 0.98, 1.01)
    reject_confidence_below: float = 0.20
    complementarity_min_gain_ms: float = 0.5


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.data.window_seconds % self.data.segment_seconds:
            raise ValueError("window_seconds must be divisible by segment_seconds")
        if self.data.update_seconds <= 0:
            raise ValueError("update_seconds must be positive")
        if self.data.accel_mode not in {"none", "scalar"}:
            raise ValueError("phase one supports accel_mode='none' or 'scalar'")
        if self.train.stage not in {"direct", "beat_pretrain", "beat", "joint"}:
            raise ValueError("stage must be direct, beat_pretrain, beat, or joint")
        if self.train.num_workers == 0 and self.train.persistent_workers:
            raise ValueError("persistent_workers requires num_workers > 0")
        if self.model.fusion_enabled and self.train.stage != "joint":
            raise ValueError("fusion can only be enabled after joint/complementarity evaluation")
        if self.model.direct_architecture not in {"causal_tcn", "segnet_mean"}:
            raise ValueError("direct_architecture must be causal_tcn or segnet_mean")
        if self.model.direct_architecture == "segnet_mean" and self.train.stage != "direct":
            raise ValueError("segnet_mean is the direct-only reproduction baseline")
