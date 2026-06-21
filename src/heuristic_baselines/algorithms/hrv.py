"""
HRV / PRV metrics from a windowed PPG signal, computed with NeuroKit2.

Beat detection uses NeuroKit2's ``elgendi`` PPG pipeline (same detector as
``neurokit.py``), and the HRV indices themselves come straight from NeuroKit2's
official estimators ``nk.hrv_time`` / ``nk.hrv_frequency`` / ``nk.hrv_nonlinear``,
so every column maps 1:1 to the NeuroKit docs and is citable as such.

Output column names are kept in NeuroKit's native ``HRV_*`` form on purpose
(e.g. ``HRV_RMSSD``, ``HRV_SDNN``, ``HRV_LFHF``) for traceability. Two extras are
added: ``n_peaks`` (beats detected) and ``hr_mean`` (60000 / HRV_MeanNN).

If NeuroKit2 is not installed, a SciPy time-domain fallback computes a reduced
set (MeanNN/SDNN/RMSSD/SDSD/pNN50/pNN20/CVNN); frequency/nonlinear are NaN.

TERMINOLOGY: HRV from PPG peaks is pulse rate variability (PRV), an approximation
of ECG-derived HRV.

WINDOW LENGTH: time-domain (esp. RMSSD) is reliable from ~1-2 min; frequency-
domain LF/HF needs >=2-5 min. With the planned 5-min windows, every window is a
standard short-term HRV recording (Task Force 1996), so per-window output below
is the proper analysis unit.
"""
from __future__ import annotations

import warnings

import numpy as np

try:
    import neurokit2 as nk
except ImportError:  # pragma: no cover - optional dependency
    nk = None  # type: ignore[assignment]

from scipy.signal import find_peaks  # noqa: E402

# Physiological inter-beat-interval gate (ms), matching neurokit.py.
IBI_MIN_MS = 300.0
IBI_MAX_MS = 2000.0

# Physiological RMSSD upper bound for PPG — values above this indicate
# false peak detection (motion artifacts producing inflated IBI variability).
RMSSD_MAX_MS = 200.0
# IBI coefficient-of-variation upper bound: normal sinus rhythm at rest
# rarely exceeds 0.15; 0.20 is a generous limit for ambulatory recordings.
IBI_CV_MAX = 0.20

# Curated, reportable subset of NeuroKit HRV columns (native HRV_* names).
TIME_COLS: tuple[str, ...] = (
    "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD",
    "HRV_pNN50", "HRV_pNN20", "HRV_CVNN", "HRV_MedianNN",
)
FREQ_COLS: tuple[str, ...] = (
    "HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_LFn", "HRV_HFn", "HRV_TP",
)
NONLINEAR_COLS: tuple[str, ...] = ("HRV_SD1", "HRV_SD2", "HRV_SD1SD2")


def hrv_columns(freq: bool = True, nonlinear: bool = True) -> list[str]:
    """Ordered output columns for the chosen metric families."""
    cols = ["n_peaks", "hr_mean", *TIME_COLS]
    if freq:
        cols += list(FREQ_COLS)
    if nonlinear:
        cols += list(NONLINEAR_COLS)
    return cols


QC_COLS: tuple[str, ...] = (
    "sqi",
    "valid_ibi_ratio",
    "ibi_cv",
    "ibi_correction_ratio",
    "ppg_qc_reason",
)


def _consensus_peaks(
    peaks_a: np.ndarray, peaks_b: np.ndarray, tol_samples: int = 5
) -> np.ndarray:
    """Keep only peaks agreed upon by two detectors within *tol_samples*.

    For each peak in *peaks_a*, if *peaks_b* has a peak within ±tol
    samples, keep the *peaks_a* position (assumed more precise).

    Reference: Charlton et al. (2022) recommends multi-detector fusion
    for improved robustness.
    """
    if peaks_a.size == 0:
        return peaks_b.copy()
    if peaks_b.size == 0:
        return peaks_a.copy()
    consensus = []
    for p in peaks_a:
        dists = np.abs(peaks_b.astype(np.int64) - int(p))
        if dists.min() <= tol_samples:
            consensus.append(int(p))
    return np.array(consensus, dtype=np.int64) if consensus else peaks_a.copy()


def detect_ppg_peaks(ppg: np.ndarray, fs: float) -> np.ndarray:
    """Return systolic peak sample indices for one PPG window."""
    x = np.asarray(ppg, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size < 50:
        return np.empty(0, dtype=np.int64)

    if nk is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xc = nk.ppg_clean(x, sampling_rate=fs, method="elgendi")
                _, info = nk.ppg_peaks(
                    xc, sampling_rate=fs, method="elgendi", correct_artifacts=True
                )
            return np.asarray(info.get("PPG_Peaks", []), dtype=np.int64).ravel()
        except Exception:
            pass  # fall through to scipy

    # SciPy fallback: assumes x is detrended/bandpassed (preprocess.py).
    x = x - np.mean(x)
    std = float(np.std(x))
    if std == 0.0:
        return np.empty(0, dtype=np.int64)
    min_dist = max(1, int(round(0.273 * fs)))  # 220 bpm ceiling
    peaks, _ = find_peaks(x, distance=min_dist, prominence=0.3 * std)
    return peaks.astype(np.int64)


def _refine_peaks_parabolic(signal: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Cubic (n=3) polynomial interpolation for sub-sample peak precision.

    Fits a cubic polynomial through 5 neighbouring samples
    [p-2, p-1, p, p+1, p+2] and finds the maximum analytically.
    Falls back to quadratic (3-point) for edge peaks.

    CinC2025-187 (Valencio et al.) showed n=3 outperforms n=2 for
    PPG peak refinement on Samsung Galaxy Ring data.

    Reference: Fioravanti et al. (2023); Valencio et al. (CinC 2025).
    """
    refined = np.empty(len(peaks), dtype=np.float64)
    for i, p in enumerate(peaks):
        # 5-point cubic interpolation (n=3).
        if p >= 2 and p <= len(signal) - 3:
            xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            ys = np.array([
                float(signal[p - 2]), float(signal[p - 1]),
                float(signal[p]),
                float(signal[p + 1]), float(signal[p + 2]),
            ])
            coeffs = np.polyfit(xs, ys, 3)  # [a, b, c, d] for ax³+bx²+cx+d
            a, b, c, _d = coeffs
            # f'(x) = 3ax² + 2bx + c = 0
            disc = 4.0 * b * b - 12.0 * a * c
            if abs(a) > 1e-15 and disc >= 0:
                sqrt_disc = np.sqrt(disc)
                x1 = (-2.0 * b + sqrt_disc) / (6.0 * a)
                x2 = (-2.0 * b - sqrt_disc) / (6.0 * a)
                # Pick the root closest to 0 (the original peak) that
                # is a maximum (f''(x) = 6ax + 2b < 0).
                candidates = []
                for xc in [x1, x2]:
                    if abs(xc) <= 2.0 and (6.0 * a * xc + 2.0 * b) < 0:
                        candidates.append(xc)
                if candidates:
                    best = min(candidates, key=abs)
                    refined[i] = p + best
                    continue
            # Cubic solve failed — fall through to quadratic.

        # 3-point quadratic fallback.
        if p <= 0 or p >= len(signal) - 1:
            refined[i] = float(p)
            continue
        y0, y1, y2 = float(signal[p - 1]), float(signal[p]), float(signal[p + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) < 1e-12:
            refined[i] = float(p)
        else:
            refined[i] = p + 0.5 * (y0 - y2) / denom
    return refined


def _correct_ibi_artifacts(
    ibi_ms: np.ndarray,
    threshold: float = 0.20,
) -> np.ndarray:
    """Simplified Lipponen & Tarvainen (2019) IBI artifact correction.

    Detects IBIs that deviate more than *threshold* (fraction) from a
    local sliding median and replaces them with the median value.

    Reference: Lipponen & Tarvainen (2019), "A robust algorithm for
    heart rate variability time series artefact correction using novel
    beat classification", DOI: 10.1080/03091902.2019.1640306.
    """
    if ibi_ms.size < 5:
        return ibi_ms.copy()
    from scipy.ndimage import median_filter

    corrected = ibi_ms.copy()
    med = median_filter(ibi_ms, size=11, mode="reflect")
    outlier = np.abs(corrected - med) > threshold * med
    corrected[outlier] = med[outlier]
    return corrected


def _correct_ibi_artifacts_with_ratio(
    ibi_ms: np.ndarray,
    threshold: float = 0.20,
) -> tuple[np.ndarray, float]:
    """Return corrected IBI values plus the fraction replaced."""
    if ibi_ms.size < 5:
        return ibi_ms.copy(), 0.0
    from scipy.ndimage import median_filter

    corrected = ibi_ms.copy()
    med = median_filter(ibi_ms, size=11, mode="reflect")
    outlier = np.abs(corrected - med) > threshold * med
    corrected[outlier] = med[outlier]
    return corrected, float(np.mean(outlier))


def _override_time_domain_subsample(
    out: dict[str, float],
    peaks_float: np.ndarray,
    fs: float,
) -> None:
    """Recompute RMSSD / SDNN / MeanNN from sub-sample peak positions.

    NeuroKit internally uses integer peak indices (10 ms granularity at
    100 Hz).  This function replaces those coarse estimates with values
    derived from parabolic-interpolated float positions, after applying
    IBI artifact correction (Lipponen & Tarvainen 2019).
    """
    if peaks_float.size < 3:
        return
    ibi = np.diff(peaks_float) / fs * 1000.0          # ms, float precision
    nn = ibi[(ibi >= IBI_MIN_MS) & (ibi <= IBI_MAX_MS)]
    if nn.size < 3:
        return
    nn, corr_ratio = _correct_ibi_artifacts_with_ratio(nn)
    diff_nn = np.diff(nn)
    mean_nn = float(np.mean(nn))
    sdnn = float(np.std(nn, ddof=1))
    rmssd = float(np.sqrt(np.mean(diff_nn ** 2)))
    out["HRV_MeanNN"] = mean_nn
    out["hr_mean"] = 60000.0 / mean_nn if mean_nn > 0 else float("nan")
    out["HRV_SDNN"] = sdnn
    out["HRV_RMSSD"] = rmssd
    out["HRV_SDSD"] = float(np.std(diff_nn, ddof=1))
    out["HRV_pNN50"] = 100.0 * float(np.mean(np.abs(diff_nn) > 50.0))
    out["HRV_pNN20"] = 100.0 * float(np.mean(np.abs(diff_nn) > 20.0))
    out["HRV_CVNN"] = sdnn / mean_nn if mean_nn > 0 else float("nan")
    out["HRV_MedianNN"] = float(np.median(nn))
    # Override SD1/SD2 using Poincaré identities from corrected IBI so that
    # quantization error from integer peak positions does not propagate.
    sd1 = rmssd / (2.0 ** 0.5)
    sd2_sq = max(0.0, 2.0 * sdnn ** 2 - sd1 ** 2)
    out["HRV_SD1"] = sd1
    out["HRV_SD2"] = sd2_sq ** 0.5
    out["ibi_correction_ratio"] = corr_ratio


# ---------------------------------------------------------------------------
# HRV metrics
# ---------------------------------------------------------------------------
def _nan_metrics(freq: bool, nonlinear: bool) -> dict[str, float]:
    d = {c: float("nan") for c in hrv_columns(freq, nonlinear)}
    d["n_peaks"] = 0.0
    return d


def _scipy_time_domain(peaks: np.ndarray, fs: float) -> dict[str, float]:
    """Fallback time-domain HRV when NeuroKit2 is unavailable."""
    out = _nan_metrics(freq=False, nonlinear=False)
    out["n_peaks"] = float(peaks.size)
    if peaks.size < 3:
        return out
    ibi = np.diff(peaks) / float(fs) * 1000.0
    nn = ibi[(ibi >= IBI_MIN_MS) & (ibi <= IBI_MAX_MS)]
    if nn.size < 3:
        return out
    diff = np.diff(nn)
    mean_nn = float(np.mean(nn))
    sdnn = float(np.std(nn, ddof=1))
    out["HRV_MeanNN"] = mean_nn
    out["hr_mean"] = 60000.0 / mean_nn if mean_nn > 0 else float("nan")
    out["HRV_SDNN"] = sdnn
    out["HRV_RMSSD"] = float(np.sqrt(np.mean(diff**2)))
    out["HRV_SDSD"] = float(np.std(diff, ddof=1))
    out["HRV_pNN50"] = 100.0 * float(np.mean(np.abs(diff) > 50.0))
    out["HRV_pNN20"] = 100.0 * float(np.mean(np.abs(diff) > 20.0))
    out["HRV_CVNN"] = sdnn / mean_nn if mean_nn > 0 else float("nan")
    out["HRV_MedianNN"] = float(np.median(nn))
    return out


def hrv_metrics(
    peaks: np.ndarray, fs: float, *, freq: bool = True, nonlinear: bool = True
) -> dict[str, float]:
    """Peak indices -> curated HRV dict via NeuroKit2 (SciPy fallback)."""
    peaks = np.asarray(peaks, dtype=np.int64).ravel()
    out = _nan_metrics(freq, nonlinear)
    out["n_peaks"] = float(peaks.size)
    if peaks.size < 3:
        return out

    if nk is None:
        return _scipy_time_domain(peaks, fs)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            td = nk.hrv_time(peaks, sampling_rate=fs)
            for c in TIME_COLS:
                if c in td.columns:
                    out[c] = float(td[c].iloc[0])
            mean_nn = out.get("HRV_MeanNN", float("nan"))
            out["hr_mean"] = 60000.0 / mean_nn if mean_nn and mean_nn > 0 else float("nan")
            if freq:
                fd = nk.hrv_frequency(peaks, sampling_rate=fs)
                for c in FREQ_COLS:
                    if c in fd.columns:
                        out[c] = float(fd[c].iloc[0])
            if nonlinear:
                nl = nk.hrv_nonlinear(peaks, sampling_rate=fs)
                for c in NONLINEAR_COLS:
                    if c in nl.columns:
                        out[c] = float(nl[c].iloc[0])
    except Exception:
        return _scipy_time_domain(peaks, fs)
    return out


def hrv_from_ppg(
    ppg: np.ndarray,
    fs: float,
    *,
    freq: bool = True,
    nonlinear: bool = True,
    sqi_threshold: float = 0.4,
    ibi_validity_threshold: float = 0.80,
    ibi_cv_threshold: float = IBI_CV_MAX,
    rmssd_max_ms: float = RMSSD_MAX_MS,
) -> dict[str, float]:
    """One-window convenience wrapper: PPG -> curated HRV dict.

    Pipeline:
      1. Peak detection (NeuroKit elgendi).
      2. IBI validity ratio gate — if fewer than *ibi_validity_threshold*
         of detected IBIs fall within [300, 2000] ms, the window is
         marked unreliable (all HRV metrics set to NaN).
         Reference: PMC11644394, Sensors 2024 — recommends ~80%.
      3. Signal Quality Assessment (SQA) — secondary gate.
      4. NeuroKit HRV (integer peaks) for freq / nonlinear.
      5. Sub-sample parabolic interpolation + IBI artifact correction
         to override time-domain metrics with higher precision values.

    Set env BASELINE=1 to skip all improvements (steps 2-5) for
    comparison against the optimized pipeline.
    """
    import os
    baseline_mode = os.environ.get("BASELINE", "") == "1"

    from .sqa import ppg_sqi

    ppg_clean = np.asarray(ppg, dtype=np.float64)
    ppg_clean = ppg_clean[~np.isnan(ppg_clean)]
    peaks = detect_ppg_peaks(ppg_clean, fs)

    # --- Baseline mode: skip all improvements ---
    if baseline_mode:
        out = hrv_metrics(peaks, fs, freq=freq, nonlinear=nonlinear)
        out["sqi"] = 0.0
        out["valid_ibi_ratio"] = 0.0
        out["ibi_cv"] = float("nan")
        out["ibi_correction_ratio"] = 0.0
        out["ppg_qc_reason"] = "baseline_mode"
        return out

    # --- IBI validity ratio gate (PMC11644394) ---
    valid_ibi_ratio = 0.0
    ibi_cv = float("nan")
    if peaks.size >= 2:
        ibi_raw = np.diff(peaks) / fs * 1000.0
        nn_raw = ibi_raw[(ibi_raw >= IBI_MIN_MS) & (ibi_raw <= IBI_MAX_MS)]
        n_valid = int(len(nn_raw))
        valid_ibi_ratio = n_valid / len(ibi_raw)
        if nn_raw.size >= 2:
            mean_nn = float(np.mean(nn_raw))
            ibi_cv = float(np.std(nn_raw, ddof=1) / mean_nn) if mean_nn > 0 else float("nan")
    if valid_ibi_ratio < ibi_validity_threshold:
        out = _nan_metrics(freq, nonlinear)
        out["sqi"] = 0.0
        out["valid_ibi_ratio"] = valid_ibi_ratio
        out["ibi_cv"] = ibi_cv
        out["ibi_correction_ratio"] = float("nan")
        out["ppg_qc_reason"] = "low_valid_ibi_ratio"
        return out

    # --- IBI CV gate: high CV indicates motion-induced false peaks ---
    if np.isfinite(ibi_cv) and ibi_cv > ibi_cv_threshold:
        out = _nan_metrics(freq, nonlinear)
        out["sqi"] = 0.0
        out["valid_ibi_ratio"] = valid_ibi_ratio
        out["ibi_cv"] = ibi_cv
        out["ibi_correction_ratio"] = float("nan")
        out["ppg_qc_reason"] = "high_ibi_cv"
        return out

    # --- Signal quality gate ---
    sqi = ppg_sqi(ppg_clean, fs, peaks)
    if sqi < sqi_threshold:
        out = _nan_metrics(freq, nonlinear)
        out["sqi"] = sqi
        out["valid_ibi_ratio"] = valid_ibi_ratio
        out["ibi_cv"] = ibi_cv
        out["ibi_correction_ratio"] = float("nan")
        out["ppg_qc_reason"] = "low_sqi"
        return out

    out = hrv_metrics(peaks, fs, freq=freq, nonlinear=nonlinear)
    out["sqi"] = sqi
    out["valid_ibi_ratio"] = valid_ibi_ratio
    out["ibi_cv"] = ibi_cv
    out["ibi_correction_ratio"] = 0.0
    out["ppg_qc_reason"] = "ok"

    # Sub-sample refinement for time-domain metrics.
    if peaks.size >= 3 and ppg_clean.size > 0:
        peaks_f = _refine_peaks_parabolic(ppg_clean, peaks)
        _override_time_domain_subsample(out, peaks_f, fs)

    # --- RMSSD upper bound: values above threshold indicate false-peak inflation ---
    if np.isfinite(out.get("HRV_RMSSD", float("nan"))) and out["HRV_RMSSD"] > rmssd_max_ms:
        for k in _nan_metrics(freq, nonlinear):
            out[k] = float("nan")
        out["sqi"] = sqi
        out["valid_ibi_ratio"] = valid_ibi_ratio
        out["ibi_cv"] = ibi_cv
        out["ppg_qc_reason"] = "high_rmssd"

    return out
