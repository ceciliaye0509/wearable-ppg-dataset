"""One-time lossless conversion from compressed NPZ to memory-mappable arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np

from .schema import discover_participant_files, validate_npz_schema


ROBUST_STATS_VERSION = 1


def robust_stats_filename(trailing_seconds: int) -> str:
    return f"robust_stats_{int(trailing_seconds)}s.npy"


def write_robust_stats_from_cache(
    participant_dir: str | Path,
    trailing_seconds: int,
    *,
    overwrite: bool = False,
    chunk_windows: int = 4,
) -> Path:
    """Precompute window-local median/MAD/std without touching waveform values.

    Stats have shape ``[window, device, channel, (median, MAD, std)]``.  A
    separate file is required for every trailing duration, so a 60 s beat
    pretraining crop can never inherit a statistic computed on the full 5 min.
    """
    root = Path(participant_dir)
    output = root / robust_stats_filename(trailing_seconds)
    sidecar = output.with_suffix(".json")
    if output.exists() and sidecar.exists() and not overwrite:
        return output
    values = np.load(root / "values.npy", mmap_mode="r")
    mask = np.load(root / "mask.npy", mmap_mode="r")
    with np.load(root / "metadata.npz", allow_pickle=True) as z:
        fs_hz = float(z["target_fs"])
        full_seconds = int(z["window_sec"])
    if not 0 < trailing_seconds <= full_seconds:
        raise ValueError(f"trailing_seconds must be in [1, {full_seconds}]")
    trailing_samples = round(fs_hz * trailing_seconds)
    stats = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=(*values.shape[:3], 3)
    )
    start_sample = values.shape[-1] - trailing_samples
    for start in range(0, len(values), chunk_windows):
        stop = min(len(values), start + chunk_windows)
        raw = np.asarray(values[start:stop, ..., start_sample:], dtype=np.float32)
        valid = np.asarray(mask[start:stop, ..., start_sample:], dtype=np.bool_) & np.isfinite(raw)
        observed = np.where(valid, raw, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(observed, axis=-1).astype(np.float32)
            mad = np.nanmedian(np.abs(observed - median[..., None]), axis=-1).astype(np.float32)
            std = np.nanstd(observed, axis=-1).astype(np.float32)
        stats[start:stop, ..., 0] = median
        stats[start:stop, ..., 1] = mad
        stats[start:stop, ..., 2] = std
    stats.flush()
    del stats
    sidecar.write_text(
        json.dumps(
            {
                "version": ROBUST_STATS_VERSION,
                "scope": "participant/window/device/channel/trailing-crop only",
                "columns": ["median", "mad", "std"],
                "trailing_seconds": trailing_seconds,
                "trailing_samples": trailing_samples,
                "normalization": "clip((x-median)/max(1.4826*mad,0.1*std,1e-4),-12,12)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def _write_array(path: Path, array: np.ndarray, dtype: np.dtype | None = None) -> None:
    dtype = np.dtype(dtype or array.dtype)
    target = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=array.shape)
    for start in range(0, len(array), 16):
        target[start : start + 16] = array[start : start + 16]
    target.flush()
    del target


def stage_participant(source: str | Path, cache_dir: str | Path, overwrite: bool = False) -> dict[str, object]:
    source = Path(source)
    info = validate_npz_schema(source)
    participant = str(info["participant"])
    output = Path(cache_dir) / participant
    completion = output / "complete.json"
    if completion.exists() and not overwrite:
        return json.loads(completion.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)

    with np.load(source, allow_pickle=True) as data:
        values = np.asarray(data["ppg_rawslot_values"], dtype=np.float32)
        _write_array(output / "values.npy", values)
        del values

        mask = np.asarray(data["ppg_rawslot_mask"], dtype=np.bool_)
        _write_array(output / "mask.npy", mask)

        grid = np.asarray(data["ppg_grid_timestamp_ms"], dtype=np.float64)
        timestamps = np.asarray(data["ppg_rawslot_timestamp_ms"], dtype=np.float64)
        sample_period_ms = 1000.0 / float(data["target_fs"])
        jitter = np.zeros(timestamps.shape, dtype=np.float16)
        for start in range(0, len(timestamps), 8):
            stop = min(len(timestamps), start + 8)
            delta = (timestamps[start:stop] - grid[start:stop, None, None, :]) / sample_period_ms
            delta = np.clip(delta, -2.0, 2.0)
            jitter[start:stop] = np.where(mask[start:stop] & np.isfinite(delta), delta, 0.0)
        _write_array(output / "jitter.npy", jitter)
        del timestamps, jitter, grid, mask

        metadata_fields = (
            "participant", "devices", "channels", "target_fs", "target_len",
            "window_sec", "stride_sec", "ppg_window_t0_ms", "ppg_window_t1_ms",
            "ppg_rawslot_valid_sample_ratio", "accel_mean_mag",
            "accel_motion_mean_mag", "ecg_r_peak_times_rel_ms",
            "ecg_rmssd_corrected_ms", "ecg_sdnn_corrected_ms",
            "ecg_label_qc_pass", "ecg_label_qc_reason", "ecg_qrs_sqi",
            "ecg_ibi_correction_ratio",
        )
        metadata = {key: data[key] for key in metadata_fields if key in data.files}
        np.savez(output / "metadata.npz", **metadata)

    # This is derived only from each individual window.  It is an I/O/CPU
    # cache, not a fitted dataset statistic, and therefore cannot leak across
    # participants or folds.
    write_robust_stats_from_cache(output, int(info["window_seconds"]), overwrite=True)

    record = {**info, "source": str(source.resolve()), "cache": str(output.resolve())}
    completion.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def stage_dataset(
    source_dir: str | Path,
    cache_dir: str | Path,
    participants: Iterable[str] | None = None,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    files = discover_participant_files(source_dir)
    selected = set(participants or files)
    records: list[dict[str, object]] = []
    for pid in files:
        if pid not in selected:
            continue
        record = stage_participant(files[pid], cache_dir, overwrite)
        records.append(record)
        print(f"staged {pid}: {record['cache']}", flush=True)
    manifest = Path(cache_dir) / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records
