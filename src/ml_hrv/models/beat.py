"""Beat heatmap, sub-sample offset, SQI, and pulse-delay outputs."""

from __future__ import annotations

import torch
from torch import nn


class BeatSequenceHead(nn.Module):
    def __init__(
        self,
        local_width: int = 24,
        token_dim: int = 96,
        delay_min_ms: float = 100.0,
        delay_max_ms: float = 700.0,
        sample_period_ms: float = 10.0,
    ) -> None:
        super().__init__()
        self.delay_min_ms = delay_min_ms
        self.delay_range_ms = delay_max_ms - delay_min_ms
        self.sample_period_ms = sample_period_ms
        self.context = nn.Linear(token_dim, local_width)
        self.local_head = nn.Sequential(
            nn.Conv1d(local_width, local_width, 7, padding=3),
            nn.SiLU(),
            nn.Conv1d(local_width, 2, 1),
        )
        self.sqi_head = nn.Linear(token_dim, 1)
        self.delay_head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))

    def forward(self, local: torch.Tensor, context: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, segments, width, length = local.shape
        conditioned = local + self.context(context).unsqueeze(-1)
        raw = self.local_head(conditioned.reshape(batch * segments, width, length))
        raw = raw.reshape(batch, segments, 2, length)
        heatmap_logits = raw[:, :, 0]
        # The offset is relative to the nominal slot; it cannot exceed half a sample.
        offset_ms = torch.tanh(raw[:, :, 1]) * (self.sample_period_ms / 2.0)
        sqi_logits = self.sqi_head(context).squeeze(-1)
        pooled = context[:, -1]
        delay_ms = self.delay_min_ms + self.delay_range_ms * torch.sigmoid(self.delay_head(pooled).squeeze(-1))
        return {
            "heatmap_logits": heatmap_logits,
            "offset_ms": offset_ms,
            "sqi_logits": sqi_logits,
            "delay_ms": delay_ms,
        }
