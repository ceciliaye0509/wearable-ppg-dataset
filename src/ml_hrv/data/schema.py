"""Strict schema checks for the raw-aligned HRV release.

The validator inspects NPY headers inside each NPZ.  It therefore verifies the
large raw-slot arrays without decompressing multiple gigabytes into RAM.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npformat


REQUIRED_FIELDS = {
    "participant",
    "devices",
    "channels",
    "target_fs",
    "target_len",
    "window_sec",
    "stride_sec",
    "ppg_window_t0_ms",
    "ppg_window_t1_ms",
    "ppg_grid_timestamp_ms",
    "ppg_rawslot_values",
    "ppg_rawslot_mask",
    "ppg_rawslot_timestamp_ms",
    "ppg_rawslot_valid_sample_ratio",
    "accel_mean_mag",
    "accel_motion_mean_mag",
    "ecg_r_peak_times_rel_ms",
    "ecg_rmssd_corrected_ms",
    "ecg_sdnn_corrected_ms",
    "ecg_label_qc_pass",
}

FORBIDDEN_PRIMARY_FIELDS = {"ppg_resampled", "ppg_resampled_values"}
_PID_RE = re.compile(r"_(P\d+)\.npz$")


def discover_participant_files(data_dir: str | Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(Path(data_dir).glob("*_P*.npz")):
        match = _PID_RE.search(path.name)
        if match:
            files[match.group(1)] = path.resolve()
    if not files:
        raise FileNotFoundError(f"No participant NPZ files found under {data_dir}")
    return files


def _headers(path: Path) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    out: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as stream:
                major, minor = npformat.read_magic(stream)
                if (major, minor) == (1, 0):
                    shape, _, dtype = npformat.read_array_header_1_0(stream)
                else:
                    shape, _, dtype = npformat.read_array_header_2_0(stream)
            out[Path(member).stem] = (shape, dtype)
    return out


def validate_npz_schema(path: str | Path) -> dict[str, object]:
    path = Path(path)
    headers = _headers(path)
    missing = sorted(REQUIRED_FIELDS - headers.keys())
    if missing:
        raise ValueError(f"{path.name}: missing required fields {missing}")

    values_shape = headers["ppg_rawslot_values"][0]
    if len(values_shape) != 4 or values_shape[1:3] != (3, 2):
        raise ValueError(f"{path.name}: rawslot values must be (N,3,2,T), got {values_shape}")
    for field in ("ppg_rawslot_mask", "ppg_rawslot_timestamp_ms"):
        if headers[field][0] != values_shape:
            raise ValueError(f"{path.name}: {field} shape does not match rawslot values")
    n_windows, _, _, n_samples = values_shape
    if headers["ppg_grid_timestamp_ms"][0] != (n_windows, n_samples):
        raise ValueError(f"{path.name}: grid timestamps do not match rawslot values")
    for field in ("ecg_rmssd_corrected_ms", "ecg_sdnn_corrected_ms", "ecg_label_qc_pass"):
        if headers[field][0] != (n_windows,):
            raise ValueError(f"{path.name}: {field} must have one value per window")

    with np.load(path, allow_pickle=True) as data:
        fs = float(data["target_fs"])
        seconds = int(data["window_sec"])
        participant = str(data["participant"])
        devices = tuple(map(str, data["devices"].tolist()))
        channels = tuple(map(str, data["channels"].tolist()))
    if n_samples != round(fs * seconds):
        raise ValueError(f"{path.name}: target_len is inconsistent with fs/window duration")
    if channels != ("ppg_green", "ppg_ir"):
        raise ValueError(f"{path.name}: expected green+IR channels, got {channels}")
    if devices != ("Earring", "Ring", "Watch"):
        raise ValueError(f"{path.name}: expected Earring/Ring/Watch order, got {devices}")
    return {
        "path": str(path.resolve()),
        "participant": participant,
        "n_windows": n_windows,
        "n_samples": n_samples,
        "fs_hz": fs,
        "window_seconds": seconds,
        "devices": devices,
        "channels": channels,
        "primary_input": "ppg_rawslot_values",
        "primary_targets": ["ecg_rmssd_corrected_ms", "ecg_sdnn_corrected_ms"],
    }
