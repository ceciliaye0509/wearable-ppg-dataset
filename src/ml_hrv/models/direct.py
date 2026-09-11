"""Causal segment-token direct HRV regression branch."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.norm = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(F.pad(x, (self.left_padding, 0)))
        y = self.dropout(F.silu(self.norm(y)))
        return x + y


class DirectHRVHead(nn.Module):
    """30 segment tokens -> heteroscedastic log(RMSSD), log(SDNN)."""

    def __init__(
        self,
        token_dim: int = 96,
        tcn_layers: int = 4,
        dropout: float = 0.15,
        device_embedding_dim: int = 8,
        use_accel: bool = True,
    ) -> None:
        super().__init__()
        self.device_embedding = nn.Embedding(3, device_embedding_dim)
        self.accel = nn.Sequential(nn.Linear(1, device_embedding_dim), nn.SiLU()) if use_accel else None
        condition_dim = device_embedding_dim + (device_embedding_dim if use_accel else 0)
        self.condition = nn.Linear(condition_dim, token_dim)
        self.tcn = nn.Sequential(
            *[CausalResidualBlock(token_dim, 2**layer, dropout) for layer in range(tcn_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, 4),
        )

    def forward(
        self, tokens: torch.Tensor, device_id: torch.Tensor, accel: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        condition = [self.device_embedding(device_id)]
        if self.accel is not None:
            if accel is None:
                raise ValueError("accel scalar is required by this model")
            # log1p limits the influence of rare high-motion windows.
            condition.append(self.accel(torch.log1p(accel.clamp_min(0.0))))
        conditioned = tokens + self.condition(torch.cat(condition, dim=-1)).unsqueeze(1)
        context = self.tcn(conditioned.transpose(1, 2)).transpose(1, 2)
        params = self.head(context[:, -1])
        return {
            "loc": params[:, :2],
            "logvar": params[:, 2:].clamp(-7.0, 5.0),
            "context": context,
        }


class SegNetMeanHead(nn.Module):
    """Partner SegNet's mean segment aggregation with corrected output semantics."""

    def __init__(self, token_dim: int = 96, dropout: float = 0.15) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Dropout(dropout),
            nn.Linear(token_dim, 2),
        )

    def forward(
        self, tokens: torch.Tensor, device_id: torch.Tensor, accel: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        del device_id, accel
        loc = self.head(tokens.mean(1))
        return {
            "loc": loc,
            "logvar": torch.zeros_like(loc),
            "context": tokens,
        }
