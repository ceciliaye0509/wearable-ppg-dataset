"""Build a separate session-canonical cache prototype; never mutates v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b, equal_nan=True))


def contiguous_runs(t0: np.ndarray, stride_ms: float) -> list[tuple[int, int]]:
    breaks = np.flatnonzero(~np.isclose(np.diff(t0), stride_ms, rtol=0.0, atol=0.5)) + 1
    edges = np.concatenate(([0], breaks, [len(t0)]))
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--participants", nargs="+", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for participant in args.participants:
        source = args.cache_dir / participant
        output = args.output_dir / participant
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing prototype {output}")
        output.mkdir(parents=True)
        with np.load(source / "metadata.npz", allow_pickle=True) as z:
            t0 = np.asarray(z["ppg_window_t0_ms"], float)
            fs = float(z["target_fs"])
            window_seconds = int(z["window_sec"])
            stride_seconds = int(z["stride_sec"])
        window = round(fs * window_seconds)
        stride = round(fs * stride_seconds)
        runs = contiguous_runs(t0, stride_seconds * 1000.0)
        total_samples = sum(window + (stop - start - 1) * stride for start, stop in runs)
        inputs = {
            "values": np.load(source / "values.npy", mmap_mode="r"),
            "mask": np.load(source / "mask.npy", mmap_mode="r"),
            "jitter": np.load(source / "jitter.npy", mmap_mode="r"),
        }
        outputs = {
            name: np.lib.format.open_memmap(
                output / f"{name}.npy", mode="w+", dtype=array.dtype,
                shape=(*array.shape[1:3], total_samples),
            )
            for name, array in inputs.items()
        }
        offsets = np.zeros(len(t0), dtype=np.int64)
        cursor = 0
        for start, stop in runs:
            for name in inputs:
                outputs[name][..., cursor : cursor + window] = inputs[name][start]
            offsets[start] = cursor
            for index in range(start + 1, stop):
                offsets[index] = cursor + (index - start) * stride
                tail_start = offsets[index] + window - stride
                for name in inputs:
                    outputs[name][..., tail_start : tail_start + stride] = inputs[name][index, ..., -stride:]
            cursor += window + (stop - start - 1) * stride
        for array in outputs.values():
            array.flush()
        np.save(output / "window_offsets.npy", offsets)
        mismatches = {name: 0 for name in inputs}
        for index, offset in enumerate(offsets):
            for name in inputs:
                if not equal(inputs[name][index], outputs[name][..., offset : offset + window]):
                    mismatches[name] += 1
        manifest = {
            "version": "canonical_raw_v2_prototype",
            "participant": participant,
            "source_cache": str(source.resolve()),
            "windows": len(t0),
            "runs": len(runs),
            "window_samples": window,
            "stride_samples": stride,
            "current_samples_per_device_channel": len(t0) * window,
            "canonical_samples_per_device_channel": total_samples,
            "reduction_fraction": 1.0 - total_samples / (len(t0) * window),
            "window_reconstruction_mismatches": mismatches,
            "status": "equivalent" if not any(mismatches.values()) else "failed",
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
