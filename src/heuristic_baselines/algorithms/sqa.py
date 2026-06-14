"""PPG Signal Quality Assessment (SQA).

Computes a composite Signal Quality Index (SQI) for each PPG window
to enable quality-gated HRV analysis.  Windows with SQI below a
configurable threshold are flagged as unreliable.

Sub-indices:
  1. Template-matching SQI (Li & Clifford 2012): median per-beat
     correlation with the ensemble-median pulse template.
  2. Spectral-purity SQI: fraction of power in the cardiac band
     (0.7–3.5 Hz) relative to total power.

References:
  - Li & Clifford (2012), Physiological Measurement.
  - Elgendi (2016), DOI: 10.3390/bioengineering3040021.
"""
from __future__ import annotations

import numpy as np


def ppg_sqi(
    ppg: np.ndarray,
    fs: float,
    peaks: np.ndarray,
    *,
    template_weight: float = 0.6,
    spectral_weight: float = 0.4,
) -> float:
    """Composite SQI in [0, 1].  Higher is better.

    Parameters
    ----------
    ppg : 1-D signal (already bandpassed / detrended).
    fs  : Sampling rate (Hz).
    peaks : Integer peak indices from the detector.
    template_weight, spectral_weight : Relative weights (must sum to 1).
    """
    if peaks.size < 5:
        return 0.0

    t_sqi = _template_sqi(ppg, peaks)
    s_sqi = _spectral_sqi(ppg, fs)
    return template_weight * t_sqi + spectral_weight * s_sqi


# ------------------------------------------------------------------
# Template-matching SQI
# ------------------------------------------------------------------
def _template_sqi(ppg: np.ndarray, peaks: np.ndarray) -> float:
    """Median correlation of individual beats with the ensemble template."""
    ipi = np.diff(peaks)
    if ipi.size < 3:
        return 0.0
    pulse_len = int(np.median(ipi))
    if pulse_len < 5:
        return 0.0

    # Extract fixed-length beat segments centred on each peak.
    half = pulse_len // 4
    templates: list[np.ndarray] = []
    for p in peaks:
        start = int(p) - half
        end = start + pulse_len
        if start < 0 or end > len(ppg):
            continue
        seg = ppg[start:end].astype(np.float64)
        # Normalise to zero-mean, unit-variance.
        std = float(np.std(seg))
        if std < 1e-12:
            continue
        templates.append((seg - np.mean(seg)) / std)

    if len(templates) < 3:
        return 0.0

    median_tmpl = np.median(templates, axis=0)
    std_tmpl = float(np.std(median_tmpl))
    if std_tmpl < 1e-12:
        return 0.0
    median_tmpl = (median_tmpl - np.mean(median_tmpl)) / std_tmpl

    corrs = []
    for t in templates:
        r = float(np.corrcoef(t, median_tmpl)[0, 1])
        if np.isfinite(r):
            corrs.append(r)
    if not corrs:
        return 0.0
    return float(np.clip(np.median(corrs), 0.0, 1.0))


# ------------------------------------------------------------------
# Spectral-purity SQI
# ------------------------------------------------------------------
def _spectral_sqi(ppg: np.ndarray, fs: float) -> float:
    """Fraction of signal power inside the cardiac frequency band."""
    fft_vals = np.abs(np.fft.rfft(ppg.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(ppg), 1.0 / fs)
    power = fft_vals ** 2
    cardiac = (freqs >= 0.7) & (freqs <= 3.5)
    total = float(np.sum(power))
    if total < 1e-12:
        return 0.0
    return float(np.clip(np.sum(power[cardiac]) / total, 0.0, 1.0))
