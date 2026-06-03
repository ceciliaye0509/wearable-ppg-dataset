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


# ---------------------------------------------------------------------------
# Peak detection (NeuroKit elgendi; SciPy fallback)
# ---------------------------------------------------------------------------
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
    ppg: np.ndarray, fs: float, *, freq: bool = True, nonlinear: bool = True
) -> dict[str, float]:
    """One-window convenience wrapper: PPG -> curated HRV dict."""
    peaks = detect_ppg_peaks(ppg, fs)
    return hrv_metrics(peaks, fs, freq=freq, nonlinear=nonlinear)
