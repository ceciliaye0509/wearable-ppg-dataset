"""
Configuration for ``heuristic_baselines`` (PPG preprocess + heuristic HR models).

HuggingFace window NPZ is read from the **sibling** dataset tree (fixed layout)::

    <parent>/
      <this-repo>/      # clone name may vary, e.g. wearable-ppg-dataset
                         # .../src/heuristic_baselines/ = PACKAGE_ROOT
      Multisite-PPG/    # HuggingFace ``snowballlab/Multisite-PPG`` (``local_dir`` default)
          ppg_windowed_data/<Px>/...                    (``HEURISTIC_DATA_SOURCE="full"``)
          sample_data/ppg_windowed_data/<Px>/...         (``HEURISTIC_DATA_SOURCE="sample"``)

Outputs (CSV, ``*_preprocess.npz``) go under this package: ``outputs/<Px>/``.

Run from this directory::

    python runner.py
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
# Repository root (parent of ``src/``); folder name does not matter.
REPO_ROOT = PACKAGE_ROOT.parent.parent
# Sibling of the code repo: Multisite-PPG/
HEURISTIC_HF_SUBMISSION_ROOT: Path = (
    REPO_ROOT.parent / "Multisite-PPG"
).resolve()
# Sibling of the code repo: hf_upload/ (contains 5-min windowed data for HRV)
HEURISTIC_HF_UPLOAD_ROOT: Path = (
    REPO_ROOT.parent / "hf_upload"
).resolve()

# =============================================================================
# Manual settings (edit here only)
# =============================================================================
# Which windowed tree under ``HEURISTIC_HF_SUBMISSION_ROOT`` to use.
# "full" -> .../ppg_windowed_data/<Px>/...
# "sample" -> .../sample_data/ppg_windowed_data/<Px>/...
HEURISTIC_DATA_SOURCE: str = "5min"

HEURISTIC_PIPELINE_PARTICIPANTS: list[str] = ["P7", "P3"]
HEURISTIC_DEVICE_ROLES: tuple[str, ...] = ("Earring", "Ring", "Necklace", "Watch")

HEURISTIC_RESULT_ROOT: Path = PACKAGE_ROOT / "outputs"

HEURISTIC_RUN_PREPROCESS: bool = True

# Single channel: one-element tuple, e.g. ("ppg_green","ppg_ir") — trailing comma required.
HEURISTIC_PPG_CHANNELS: tuple[str, ...] = ("ppg_green", "ppg_ir")

HEURISTIC_ALGORITHMS: tuple[str, ...] = ("neurokit",)

KNOWN_HEURISTIC_ALGORITHMS: frozenset[str] = frozenset(
    ("pwd", "msptd", "fft", "autocorr", "heartpy", "neurokit", "qppgfast")
)

# =============================================================================
# HRV / PRV settings (used by hrv_runner.py only)
# =============================================================================
# hrv_runner.py reuses HEURISTIC_PIPELINE_PARTICIPANTS / DEVICE_ROLES /
# PPG_CHANNELS / RUN_PREPROCESS above, and computes HRV per window with
# NeuroKit2 (nk.hrv_time / nk.hrv_frequency / nk.hrv_nonlinear).
#
# These windows are produced at 5 min (300 s) each, which is the standard
# short-term HRV recording length (Task Force, 1996). At 5 min, time-domain
# (RMSSD, SDNN), frequency-domain (LF/HF), and Poincaré (SD1/SD2) are all valid.
HRV_WINDOW_SEC: float = 300.0
# Compute frequency-domain (LF/HF/LFn/HFn/TP). Valid at >=2-5 min; turn OFF if
# you ever run this on short (<60 s) windows.
HRV_COMPUTE_FREQ: bool = True
# Compute nonlinear Poincaré indices (SD1/SD2/SD1SD2).
HRV_COMPUTE_NONLINEAR: bool = True


def _resolve_heuristic_windows_root() -> Path:
    key = HEURISTIC_DATA_SOURCE.strip().lower()
    base = HEURISTIC_HF_SUBMISSION_ROOT
    if key == "full":
        return (base / "ppg_windowed_data").resolve()
    if key == "sample":
        return (base / "sample_data" / "ppg_windowed_data").resolve()
    if key == "5min":
        return (HEURISTIC_HF_UPLOAD_ROOT / "5min_windowed").resolve()
    raise ValueError(
        f'HEURISTIC_DATA_SOURCE must be "full", "sample", or "5min", got {HEURISTIC_DATA_SOURCE!r}'
    )


HEURISTIC_WINDOWS_ROOT: Path = _resolve_heuristic_windows_root()
