"""Shared segment encoder for selected raw PPG slots, masks, and timestamp jitter."""

from __future__ import annotations

import torch
from torch import nn


def masked_channel_normalize(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Robust-ish masked normalization used for synthetic/direct model calls.

    The staged dataset already applies median/MAD normalization.  This second
    masked centering is intentionally light and makes the model safe when called
    by deployment code with raw standardized slots.
    """
    maskf = mask.to(values.dtype)
    count = maskf.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (values * maskf).sum(dim=-1, keepdim=True) / count
    variance = ((values - mean).square() * maskf).sum(dim=-1, keepdim=True) / count
    return ((values - mean) / variance.clamp_min(1e-4).sqrt()).clamp(-12.0, 12.0) * maskf


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        pad = 3 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 7, padding=pad, dilation=dilation),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class MaskAwareSharedEncoder(nn.Module):
    """Encode each 10-second segment while retaining a beat-resolution map."""

    def __init__(
        self,
        width: int = 24,
        token_dim: int = 96,
        dropout: float = 0.1,
        include_jitter: bool = True,
        signal_channels: int = 2,
    ) -> None:
        super().__init__()
        if signal_channels < 1:
            raise ValueError("signal_channels must be positive")
        self.include_jitter = include_jitter
        self.signal_channels = signal_channels
        # Selected PPG channels, their masks, and optionally their jitter.
        input_channels = signal_channels * (3 if include_jitter else 2)
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, width, 9, padding=4),
            nn.GroupNorm(4, width),
            nn.SiLU(),
        )
        self.local = nn.Sequential(
            ResidualBlock(width, 1, dropout),
            ResidualBlock(width, 2, dropout),
            ResidualBlock(width, 4, dropout),
        )
        self.token = nn.Sequential(
            nn.Conv1d(width, width * 2, 9, stride=4, padding=4),
            nn.GroupNorm(4, width * 2),
            nn.SiLU(),
            nn.Conv1d(width * 2, token_dim, 9, stride=4, padding=4),
            nn.GroupNorm(8, token_dim),
            nn.SiLU(),
        )

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor, jitter: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim != 4 or values.shape[2] != self.signal_channels:
            raise ValueError(
                f"values must have shape (batch, segments, {self.signal_channels}, samples)"
            )
        if mask.shape != values.shape or jitter.shape != values.shape:
            raise ValueError("mask and jitter must match values")
        batch, segments, channels, length = values.shape
        normalized = masked_channel_normalize(values, mask)
        maskf = mask.to(values.dtype)
        parts = (normalized, maskf, jitter * maskf) if self.include_jitter else (normalized, maskf)
        x = torch.cat(parts, dim=2)
        x = x.reshape(batch * segments, channels * len(parts), length)
        sample_valid = maskf.mean(dim=2).reshape(batch * segments, 1, length)
        local = self.local(self.stem(x)) * (sample_valid > 0).to(x.dtype)
        compressed = self.token(local)
        weights = torch.nn.functional.adaptive_avg_pool1d(sample_valid, compressed.shape[-1])
        token = (compressed * weights).sum(-1) / weights.sum(-1).clamp_min(1e-4)
        token = token.reshape(batch, segments, -1)
        local = local.reshape(batch, segments, local.shape[1], length)
        coverage = maskf.mean(dim=(2, 3))
        return token, local, coverage


class PartnerSegNetEncoder(nn.Module):
    """Faithful shared 10-second CNN from the partner SegNet, with raw masks."""

    def __init__(self, token_dim: int = 64, signal_channels: int = 2) -> None:
        super().__init__()
        if signal_channels < 1:
            raise ValueError("signal_channels must be positive")
        self.signal_channels = signal_channels
        # The original has three stride-2 Conv/BatchNorm/ReLU blocks.  Four
        # inputs are selected PPG channels and their masks; no interpolation.
        self.net = nn.Sequential(
            nn.Conv1d(signal_channels * 2, 32, 7, 2, 3), nn.BatchNorm1d(32), nn.ReLU(True),
            nn.Conv1d(32, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Conv1d(64, token_dim, 7, 2, 3), nn.BatchNorm1d(token_dim), nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor, jitter: torch.Tensor
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        del jitter
        batch, segments, channels, length = values.shape
        if channels != self.signal_channels:
            raise ValueError(f"Expected {self.signal_channels} PPG channels, got {channels}")
        normalized = masked_channel_normalize(values, mask)
        maskf = mask.to(values.dtype)
        x = torch.cat((normalized, maskf), dim=2).reshape(batch * segments, channels * 2, length)
        token = self.net(x).squeeze(-1).reshape(batch, segments, -1)
        coverage = maskf.mean(dim=(2, 3))
        return token, None, coverage
