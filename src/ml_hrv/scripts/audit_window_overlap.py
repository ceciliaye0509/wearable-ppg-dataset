"""Audit whether 30 s-stride windows can share a canonical session store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml_hrv.data.schema import discover_participant_files


def equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b, equal_nan=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--timestamp-participant", default="P5")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    participants = sorted(
        (x for x in args.cache_dir.iterdir() if x.is_dir() and (x / "complete.json").exists()),
        key=lambda x: int(x.name.lstrip("P")),
    )
    totals = {
        "windows": 0, "contiguous_pairs": 0, "runs": 0,
        "current_slot_windows": 0, "canonical_slot_windows": 0,
        "value_pair_mismatches": 0, "mask_pair_mismatches": 0, "jitter_pair_mismatches": 0,
    }
    by_participant = {}
    for root in participants:
        with np.load(root / "metadata.npz", allow_pickle=True) as z:
            t0 = np.asarray(z["ppg_window_t0_ms"], float)
            fs = float(z["target_fs"])
            window_seconds = int(z["window_sec"])
            stride_seconds = int(z["stride_sec"])
        values = np.load(root / "values.npy", mmap_mode="r")
        mask = np.load(root / "mask.npy", mmap_mode="r")
        jitter = np.load(root / "jitter.npy", mmap_mode="r")
        stride = round(fs * stride_seconds)
        window = round(fs * window_seconds)
        contiguous = np.isclose(np.diff(t0), stride_seconds * 1000.0, rtol=0.0, atol=0.5)
        pair_indices = np.flatnonzero(contiguous)
        mismatches = {"value": 0, "mask": 0, "jitter": 0}
        for left in pair_indices:
            right = left + 1
            if not equal(values[left, ..., stride:], values[right, ..., : window - stride]):
                mismatches["value"] += 1
            if not equal(mask[left, ..., stride:], mask[right, ..., : window - stride]):
                mismatches["mask"] += 1
            if not equal(jitter[left, ..., stride:], jitter[right, ..., : window - stride]):
                mismatches["jitter"] += 1
        run_lengths = []
        start = 0
        for index, is_contiguous in enumerate(contiguous):
            if not is_contiguous:
                run_lengths.append(index + 1 - start)
                start = index + 1
        run_lengths.append(len(t0) - start)
        canonical = sum(window + (length - 1) * stride for length in run_lengths)
        current = len(t0) * window
        row = {
            "windows": len(t0), "contiguous_pairs": len(pair_indices), "runs": len(run_lengths),
            "current_slots_per_device_channel": current,
            "canonical_slots_per_device_channel": canonical,
            "theoretical_reduction_fraction": 1.0 - canonical / current,
            "mismatches": mismatches,
        }
        by_participant[root.name] = row
        totals["windows"] += len(t0)
        totals["contiguous_pairs"] += len(pair_indices)
        totals["runs"] += len(run_lengths)
        totals["current_slot_windows"] += current
        totals["canonical_slot_windows"] += canonical
        for field in mismatches:
            totals[f"{field}_pair_mismatches"] += mismatches[field]

    timestamp_audit = None
    if args.source_dir is not None:
        source = discover_participant_files(args.source_dir)[args.timestamp_participant]
        root = args.cache_dir / args.timestamp_participant
        with np.load(root / "metadata.npz", allow_pickle=True) as z:
            t0 = np.asarray(z["ppg_window_t0_ms"], float)
            stride_seconds = int(z["stride_sec"])
            fs = float(z["target_fs"])
        with np.load(source, allow_pickle=True) as z:
            timestamps = np.asarray(z["ppg_rawslot_timestamp_ms"], dtype=np.float64)
        stride = round(fs * stride_seconds)
        contiguous = np.flatnonzero(
            np.isclose(np.diff(t0), stride_seconds * 1000.0, rtol=0.0, atol=0.5)
        )
        timestamp_mismatches = sum(
            not equal(timestamps[i, ..., stride:], timestamps[i + 1, ..., :-stride])
            for i in contiguous
        )
        timestamp_audit = {
            "participant": args.timestamp_participant,
            "pairs": len(contiguous),
            "raw_timestamp_pair_mismatches": int(timestamp_mismatches),
        }

    reduction = 1.0 - totals["canonical_slot_windows"] / totals["current_slot_windows"]
    result = {
        "protocol": "adjacent windows are contiguous when t0 increases exactly one stride",
        "totals": {**totals, "theoretical_reduction_fraction": reduction},
        "raw_timestamp_source_audit": timestamp_audit,
        "participants": by_participant,
        "decision": "canonical store is permitted only if all value/mask/jitter and source timestamp mismatch counts are zero",
    }
    output = args.output or args.cache_dir.parent / "benchmarks" / "window_overlap_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"totals": result["totals"], "timestamp": timestamp_audit}, indent=2))
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
