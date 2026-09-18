"""PPG-only qPPGFast feature joins and train-only scaling.

The qPPGFast CSV is an intermediate PPG-derived artifact.  This module never
reads ECG labels: its only purpose is to make a precomputed physiological
estimate available as a *base* prediction plus quality covariates.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


QPPG_FEATURE_COLUMNS = (
    "qppg_valid",
    "qppg_peak_count",
    "qppg_valid_ibi_ratio",
    "qppg_ibi_cv",
    "qppg_ibi_correction_ratio",
    "qppg_sqi",
    "qppg_valid_sample_ratio",
    "qppg_max_raw_gap_ms",
    "accel_motion_mean_mag",
)


def _as_float(row: dict[str, str], name: str) -> float:
    try:
        return float(row.get(name, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(row: dict[str, str], name: str) -> float:
    return float(str(row.get(name, "")).strip().lower() in {"1", "true", "yes"})


def _feature_vector(row: dict[str, str]) -> np.ndarray:
    """Convert quality fields to a stable numeric, PPG-only covariate vector."""
    values = np.asarray([
        _as_bool(row, "qppg_valid"),
        np.log1p(max(_as_float(row, "qppg_peak_count"), 0.0)),
        _as_float(row, "qppg_valid_ibi_ratio"),
        _as_float(row, "qppg_ibi_cv"),
        _as_float(row, "qppg_ibi_correction_ratio"),
        _as_float(row, "qppg_sqi"),
        _as_float(row, "qppg_valid_sample_ratio"),
        np.log1p(max(_as_float(row, "qppg_max_raw_gap_ms"), 0.0)),
        np.log1p(max(_as_float(row, "accel_motion_mean_mag"), 0.0)),
    ], dtype=np.float32)
    return values


def _key(participant: str, window_index: int, device: str, channel: str) -> tuple[str, int, str, str]:
    return str(participant), int(window_index), str(device), str(channel)


def load_qppg_rows(path: str | Path) -> dict[tuple[str, int, str, str], dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Missing qPPG feature CSV: %s" % path)
    result: dict[tuple[str, int, str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"participant", "window_index", "device", "channel", "qppg_rmssd_ms", "qppg_sdnn_ms", "qppg_valid"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("qPPG feature CSV is missing required columns")
        for row in reader:
            row_key = _key(row["participant"], int(row["window_index"]), row["device"], row["channel"])
            if row_key in result:
                raise ValueError("duplicate qPPG feature key: %r" % (row_key,))
            result[row_key] = row
    return result


@dataclass
class QPPGFeatureScaler:
    """Median-impute and standardize PPG-only covariates from the train split."""

    impute: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    base_median_ms: np.ndarray

    @classmethod
    def fit(cls, rows: Iterable[dict[str, str]]) -> "QPPGFeatureScaler":
        rows = list(rows)
        if not rows:
            raise ValueError("cannot fit qPPG feature scaler on no rows")
        features = np.asarray([_feature_vector(row) for row in rows], dtype=np.float32)
        impute = np.nanmedian(features, axis=0)
        # A whole-invalid quality column remains an explicit zero after scaling.
        impute = np.where(np.isfinite(impute), impute, 0.0)
        filled = np.where(np.isfinite(features), features, impute[None, :])
        mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        std = np.where(std > 1e-6, std, 1.0)
        base = np.asarray([
            [_as_float(row, "qppg_rmssd_ms"), _as_float(row, "qppg_sdnn_ms")]
            for row in rows
        ], dtype=np.float32)
        base[~np.isfinite(base)] = np.nan
        base_median = np.nanmedian(base, axis=0)
        if not np.isfinite(base_median).all():
            raise ValueError("train qPPG CSV contains no finite base estimate for an HRV target")
        return cls(impute.astype(np.float32), mean.astype(np.float32), std.astype(np.float32), base_median.astype(np.float32))

    def transform(self, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
        features = _feature_vector(row)
        features = np.where(np.isfinite(features), features, self.impute)
        base = np.asarray([
            _as_float(row, "qppg_rmssd_ms"), _as_float(row, "qppg_sdnn_ms")
        ], dtype=np.float32)
        base = np.where(np.isfinite(base) & (base > 0.0), base, self.base_median_ms)
        return ((features - self.mean) / self.std).astype(np.float32), base.astype(np.float32)

    def state_dict(self) -> dict[str, object]:
        return {
            "feature_columns": list(QPPG_FEATURE_COLUMNS),
            "impute": self.impute.tolist(), "mean": self.mean.tolist(),
            "std": self.std.tolist(), "base_median_ms": self.base_median_ms.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "QPPGFeatureScaler":
        if tuple(state.get("feature_columns", ())) != QPPG_FEATURE_COLUMNS:
            raise ValueError("checkpoint qPPG feature schema differs from current code")
        return cls(
            np.asarray(state["impute"], dtype=np.float32),
            np.asarray(state["mean"], dtype=np.float32),
            np.asarray(state["std"], dtype=np.float32),
            np.asarray(state["base_median_ms"], dtype=np.float32),
        )
