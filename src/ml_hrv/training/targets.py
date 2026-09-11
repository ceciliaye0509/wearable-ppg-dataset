"""Dense ECG reference targets used by delay-marginalized pulse training."""

from __future__ import annotations

import torch


def build_ecg_beat_targets(
    rpeaks_ms: list[torch.Tensor],
    n_samples: int,
    fs_hz: float,
    sigma_ms: float = 20.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(rpeaks_ms)
    heatmap = torch.zeros(batch, n_samples, dtype=torch.float32, device=device)
    offsets = torch.zeros_like(heatmap)
    sigma_samples = max(sigma_ms * fs_hz / 1000.0, 0.75)
    radius = max(1, round(3 * sigma_samples))
    for row, peaks in enumerate(rpeaks_ms):
        for time_ms in peaks.detach().cpu().tolist():
            sample_float = float(time_ms) * fs_hz / 1000.0
            center = round(sample_float)
            if center < 0 or center >= n_samples:
                continue
            start, stop = max(0, center - radius), min(n_samples, center + radius + 1)
            positions = torch.arange(start, stop, device=device, dtype=torch.float32)
            bump = torch.exp(-0.5 * ((positions - sample_float) / sigma_samples).square())
            current = heatmap[row, start:stop]
            heatmap[row, start:stop] = torch.maximum(current, bump)
            offsets[row, center] = (sample_float - center) * 1000.0 / fs_hz
    return heatmap, offsets


def shift_with_zeros(values: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Shift each row right by a non-negative, sample-specific delay."""
    if values.ndim != 2 or shifts.ndim != 1 or len(values) != len(shifts):
        raise ValueError("values must be [batch,time] and shifts must be [batch]")
    positions = torch.arange(values.shape[-1], device=values.device)
    source = positions[None, :] - shifts.to(device=values.device, dtype=torch.long)[:, None]
    valid = (source >= 0) & (source < values.shape[-1])
    gathered = values.gather(1, source.clamp(0, values.shape[-1] - 1))
    return gathered * valid.to(values.dtype)


def shift_candidates_with_zeros(values: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Create every shared delay candidate as one batched gather.

    Returns ``[batch, candidate, time]`` and replaces the former Python loop
    over 100--700 ms candidates in the beat alignment hot path.
    """
    if values.ndim != 2 or shifts.ndim != 1:
        raise ValueError("values must be [batch,time] and shifts must be [candidate]")
    positions = torch.arange(values.shape[-1], device=values.device)
    source = positions[None, :] - shifts.to(device=values.device, dtype=torch.long)[:, None]
    valid = (source >= 0) & (source < values.shape[-1])
    source = source.clamp(0, values.shape[-1] - 1)
    gathered = values[:, None, :].expand(-1, len(shifts), -1).gather(
        2, source[None, :, :].expand(len(values), -1, -1)
    )
    return gathered * valid[None, :, :].to(values.dtype)
