"""Memory-mapped one-device-per-sample raw-slot dataset."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from ml_hrv import DEVICE_NAMES
from ml_hrv.physiology import reference_hrv_from_rpeaks

from .staging import robust_stats_filename


@dataclass(frozen=True)
class SampleRef:
    participant: str
    window_index: int
    device_index: int
    group_id: int


class RawslotWindowDataset(Dataset):
    """Returns a single device's green+IR values, masks, and jitter.

    Input values are never interpolated.  ``values`` are copied verbatim from
    ``ppg_rawslot_values``; missing slots are represented separately by
    ``mask`` and set to zero only after robust normalization.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        participants: Iterable[str],
        devices: Iterable[str] = DEVICE_NAMES,
        segment_seconds: int = 10,
        trailing_seconds: int = 300,
        accel_mode: str = "scalar",
        qc_only: bool = True,
        cache_open_participants: int = 4,
        use_precomputed_stats: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.participants = tuple(participants)
        self.device_names = tuple(devices)
        self.device_indices = tuple(DEVICE_NAMES.index(x) for x in self.device_names)
        self.segment_seconds = segment_seconds
        self.trailing_seconds = trailing_seconds
        self.accel_mode = accel_mode
        self.cache_open_participants = cache_open_participants
        self.use_precomputed_stats = use_precomputed_stats
        if accel_mode not in {"none", "scalar"}:
            raise ValueError("Phase-one dataset supports accel_mode none/scalar")

        self._metadata: dict[str, dict[str, np.ndarray]] = {}
        self._arrays: OrderedDict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]
        ] = OrderedDict()
        self.refs: list[SampleRef] = []
        group_id = 0
        for participant in self.participants:
            root = self.cache_dir / participant
            for filename in ("values.npy", "mask.npy", "jitter.npy", "metadata.npz", "complete.json"):
                if not (root / filename).exists():
                    raise FileNotFoundError(
                        f"Missing {root / filename}; run python -m ml_hrv.scripts.prepare_cache first"
                    )
            with np.load(root / "metadata.npz", allow_pickle=True) as z:
                meta = {key: z[key] for key in z.files}
            self._metadata[participant] = meta
            fs = float(meta["target_fs"])
            full_seconds = int(meta["window_sec"])
            if trailing_seconds > full_seconds or trailing_seconds % segment_seconds:
                raise ValueError("trailing_seconds must divide into whole segments within the 5-minute window")
            keep = np.ones(len(meta["ecg_label_qc_pass"]), dtype=bool)
            if qc_only:
                keep &= np.asarray(meta["ecg_label_qc_pass"], dtype=bool)
            keep &= np.isfinite(meta["ecg_rmssd_corrected_ms"])
            keep &= np.isfinite(meta["ecg_sdnn_corrected_ms"])
            for window_index in np.flatnonzero(keep):
                if trailing_seconds < int(meta["window_sec"]):
                    crop_ms = (int(meta["window_sec"]) - trailing_seconds) * 1000.0
                    peaks = np.asarray(meta["ecg_r_peak_times_rel_ms"][window_index], dtype=float)
                    if np.count_nonzero(peaks >= crop_ms) < 4:
                        continue
                for device_index in self.device_indices:
                    self.refs.append(SampleRef(participant, int(window_index), device_index, group_id))
                group_id += 1
            self.fs_hz = fs

        self.samples_per_segment = round(self.fs_hz * self.segment_seconds)
        self.n_segments = self.trailing_seconds // self.segment_seconds
        self.trailing_samples = round(self.fs_hz * self.trailing_seconds)

    def __len__(self) -> int:
        return len(self.refs)

    @property
    def sample_refs(self) -> tuple[SampleRef, ...]:
        return tuple(self.refs)

    def target_array(self, unique_windows: bool = True) -> np.ndarray:
        """Read targets from metadata only; this never pages PPG waveforms in."""
        targets: list[tuple[float, float]] = []
        seen: set[int] = set()
        for ref in self.refs:
            if unique_windows and ref.group_id in seen:
                continue
            seen.add(ref.group_id)
            meta = self._metadata[ref.participant]
            if self.trailing_seconds == int(meta["window_sec"]):
                targets.append(
                    (
                        float(meta["ecg_rmssd_corrected_ms"][ref.window_index]),
                        float(meta["ecg_sdnn_corrected_ms"][ref.window_index]),
                    )
                )
            else:
                crop_ms = (int(meta["window_sec"]) - self.trailing_seconds) * 1000.0
                peaks = np.asarray(meta["ecg_r_peak_times_rel_ms"][ref.window_index], dtype=float)
                peaks = peaks[peaks >= crop_ms] - crop_ms
                targets.append(reference_hrv_from_rpeaks(peaks))
        result = np.asarray(targets, dtype=np.float32)
        return result[np.isfinite(result).all(1)]

    def _open_arrays(
        self, participant: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        if participant in self._arrays:
            arrays = self._arrays.pop(participant)
            self._arrays[participant] = arrays
            return arrays
        root = self.cache_dir / participant
        stats_path = root / robust_stats_filename(self.trailing_seconds)
        arrays = (
            np.load(root / "values.npy", mmap_mode="r"),
            np.load(root / "mask.npy", mmap_mode="r"),
            np.load(root / "jitter.npy", mmap_mode="r"),
            np.load(stats_path, mmap_mode="r")
            if self.use_precomputed_stats
            and stats_path.exists()
            and stats_path.with_suffix(".json").exists()
            else None,
        )
        self._arrays[participant] = arrays
        while len(self._arrays) > self.cache_open_participants:
            self._arrays.popitem(last=False)
        return arrays

    @staticmethod
    def _robust_normalize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.zeros_like(values, dtype=np.float32)
        for channel in range(values.shape[0]):
            valid = mask[channel] & np.isfinite(values[channel])
            if not valid.any():
                continue
            observed = values[channel, valid].astype(np.float32)
            median = float(np.median(observed))
            mad = float(np.median(np.abs(observed - median)))
            scale = max(1.4826 * mad, float(np.std(observed)) * 0.1, 1e-4)
            out[channel, valid] = np.clip((observed - median) / scale, -12.0, 12.0)
        return out

    @staticmethod
    def _robust_normalize_from_stats(
        values: np.ndarray, mask: np.ndarray, stats: np.ndarray
    ) -> np.ndarray:
        """Apply cached window-local statistics to both channels at once."""
        valid = mask & np.isfinite(values)
        median = np.asarray(stats[:, 0], dtype=np.float32)[:, None]
        mad = np.asarray(stats[:, 1], dtype=np.float32)[:, None]
        std = np.asarray(stats[:, 2], dtype=np.float32)[:, None]
        scale = np.maximum(np.maximum(1.4826 * mad, 0.1 * std), 1e-4)
        normalized = np.clip((values - median) / scale, -12.0, 12.0)
        return np.where(valid & np.isfinite(normalized), normalized, 0.0).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, object]:
        ref = self.refs[index]
        values_all, mask_all, jitter_all, stats_all = self._open_arrays(ref.participant)
        start = values_all.shape[-1] - self.trailing_samples
        raw = np.asarray(values_all[ref.window_index, ref.device_index, :, start:], dtype=np.float32)
        mask = np.asarray(mask_all[ref.window_index, ref.device_index, :, start:], dtype=np.bool_)
        jitter = np.asarray(jitter_all[ref.window_index, ref.device_index, :, start:], dtype=np.float32)
        if stats_all is None:
            values = self._robust_normalize(raw, mask)
        else:
            stats = np.asarray(stats_all[ref.window_index, ref.device_index], dtype=np.float32)
            values = self._robust_normalize_from_stats(raw, mask, stats)
        shape = (2, self.n_segments, self.samples_per_segment)
        values = values.reshape(shape).transpose(1, 0, 2).copy()
        mask = mask.reshape(shape).transpose(1, 0, 2).copy()
        jitter = jitter.reshape(shape).transpose(1, 0, 2).copy()

        meta = self._metadata[ref.participant]
        coverage = float(np.mean(mask))
        motion_scalar = float(meta["accel_motion_mean_mag"][ref.window_index, ref.device_index])
        accel = 0.0
        if self.accel_mode == "scalar":
            accel = motion_scalar
        rpeaks = np.asarray(meta["ecg_r_peak_times_rel_ms"][ref.window_index], dtype=np.float32)
        full_seconds = int(meta["window_sec"])
        crop_ms = (full_seconds - self.trailing_seconds) * 1000.0
        rpeaks = rpeaks[(rpeaks >= crop_ms) & (rpeaks < int(meta["window_sec"]) * 1000.0)] - crop_ms
        if self.trailing_seconds == full_seconds:
            target = np.array(
                [
                    meta["ecg_rmssd_corrected_ms"][ref.window_index],
                    meta["ecg_sdnn_corrected_ms"][ref.window_index],
                ],
                dtype=np.float32,
            )
            label_source = "ecg_corrected_5min_fields"
        else:
            # Auxiliary low-latency result only.  The release has no timestamped
            # corrected RR list, so the trailing ECG peaks are corrected with the
            # same adjacency-preserving routine used by the physiological branch.
            target = np.asarray(reference_hrv_from_rpeaks(rpeaks), dtype=np.float32)
            label_source = "ecg_rpeaks_short_adjacency_corrected_auxiliary"
        return {
            "values": torch.from_numpy(values),
            "mask": torch.from_numpy(mask),
            "jitter": torch.from_numpy(jitter),
            "accel": torch.tensor([accel], dtype=torch.float32),
            "motion_scalar": torch.tensor([motion_scalar], dtype=torch.float32),
            "device_id": torch.tensor(ref.device_index, dtype=torch.long),
            "target_ms": torch.from_numpy(target),
            "rpeaks_ms": torch.from_numpy(rpeaks),
            "participant": ref.participant,
            "window_index": ref.window_index,
            "window_start_ms": float(meta["ppg_window_t0_ms"][ref.window_index]),
            "device": DEVICE_NAMES[ref.device_index],
            "group_id": ref.group_id,
            "coverage": coverage,
            "label_source": label_source,
        }


def collate_rawslot(batch: list[dict[str, object]]) -> dict[str, object]:
    """Collate variable-length ECG peak lists without padding ambiguity."""
    tensor_keys = (
        "values", "mask", "jitter", "accel", "motion_scalar", "device_id", "target_ms",
    )
    result: dict[str, object] = {key: torch.stack([x[key] for x in batch]) for key in tensor_keys}
    for key in (
        "rpeaks_ms", "participant", "window_index", "window_start_ms", "device",
        "group_id", "coverage", "label_source",
    ):
        result[key] = [x[key] for x in batch]
    result["group_id"] = torch.tensor(result["group_id"], dtype=torch.long)
    result["coverage"] = torch.tensor(result["coverage"], dtype=torch.float32)
    return result
