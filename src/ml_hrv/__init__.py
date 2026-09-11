"""Raw-slot continuous HRV modelling package.

The public contract deliberately names RMSSD before SDNN everywhere.  This
prevents the silent target-order inversions that occurred in the legacy HRV
extension under :mod:`model_baselines`.
"""

from .config import ExperimentConfig

TARGET_NAMES = ("rmssd_ms", "sdnn_ms")
TARGET_FIELDS = ("ecg_rmssd_corrected_ms", "ecg_sdnn_corrected_ms")
DEVICE_NAMES = ("Earring", "Ring", "Watch")

__all__ = ["DEVICE_NAMES", "ExperimentConfig", "TARGET_FIELDS", "TARGET_NAMES"]
