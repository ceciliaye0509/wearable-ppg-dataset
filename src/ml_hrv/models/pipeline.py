"""The deployable single-device raw PPG continuous HRV network."""

from __future__ import annotations

import torch
from torch import nn

from ml_hrv.config import ExperimentConfig
from ml_hrv.data.qppg import QPPG_FEATURE_COLUMNS

from .beat import BeatSequenceHead
from .direct import DirectHRVHead, SegNetMeanHead
from .encoder import MaskAwareSharedEncoder, PartnerSegNetEncoder


class RawContinuousHRVModel(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        model = config.model
        signal_channels = len(config.data.ppg_channels)
        qppg_feature_dim = len(QPPG_FEATURE_COLUMNS) if model.qppg_residual_enabled else 0
        if model.direct_architecture == "segnet_mean":
            self.encoder = PartnerSegNetEncoder(model.token_dim, signal_channels)
            self.direct = SegNetMeanHead(model.token_dim, model.dropout, qppg_feature_dim)
        else:
            self.encoder = MaskAwareSharedEncoder(
                model.width,
                model.token_dim,
                model.dropout,
                model.use_timestamp_jitter,
                signal_channels,
            )
            self.direct = DirectHRVHead(
                model.token_dim,
                model.tcn_layers,
                model.dropout,
                model.device_embedding_dim,
                use_accel=config.data.accel_mode == "scalar",
                qppg_feature_dim=qppg_feature_dim,
            )
        self.beat = BeatSequenceHead(
            model.width,
            model.token_dim,
            model.delay_min_ms,
            model.delay_max_ms,
            sample_period_ms=1000.0 / config.data.fs_hz,
        )

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        jitter: torch.Tensor,
        device_id: torch.Tensor,
        accel: torch.Tensor | None = None,
        qppg_features: torch.Tensor | None = None,
        qppg_base_normalized: torch.Tensor | None = None,
        compute_beat: bool = True,
    ) -> dict[str, torch.Tensor]:
        tokens, local, coverage = self.encoder(values, mask, jitter)
        direct = self.direct(tokens, device_id, accel, qppg_features)
        direct_loc = direct["loc"]
        if self.config.model.qppg_residual_enabled:
            if qppg_base_normalized is None:
                raise ValueError("qPPG residual model requires a normalized qPPG base prediction")
            direct_loc = qppg_base_normalized + direct_loc
        outputs = {
            "direct_loc": direct_loc,
            "direct_delta": direct["loc"],
            "direct_logvar": direct["logvar"],
            "segment_coverage": coverage,
        }
        if compute_beat:
            if local is None:
                raise RuntimeError("Partner SegNet baseline does not expose beat-resolution features")
            beat = self.beat(local, direct["context"])
            outputs.update(
                beat_heatmap_logits=beat["heatmap_logits"],
                beat_offset_ms=beat["offset_ms"],
                beat_sqi_logits=beat["sqi_logits"],
                pulse_delay_ms=beat["delay_ms"],
            )
        return outputs
