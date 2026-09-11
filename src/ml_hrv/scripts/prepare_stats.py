"""Backfill leakage-free robust-normalization statistics into an mmap cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml_hrv.data.staging import write_robust_stats_from_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir")
    parser.add_argument("--seconds", type=int, nargs="+", default=[300])
    parser.add_argument("--participants", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-windows", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.cache_dir)
    participants = args.participants or sorted(
        path.name for path in root.iterdir() if path.is_dir() and (path / "complete.json").exists()
    )
    for participant in participants:
        for seconds in args.seconds:
            path = write_robust_stats_from_cache(
                root / participant,
                seconds,
                overwrite=args.overwrite,
                chunk_windows=args.chunk_windows,
            )
            print(f"stats {participant} {seconds}s: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
