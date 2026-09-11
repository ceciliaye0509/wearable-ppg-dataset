"""Stage compressed raw-slot NPZ files as memory-mapped training arrays."""

from __future__ import annotations

import argparse

from ml_hrv.data.staging import stage_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("cache_dir")
    parser.add_argument("--participants", nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    records = stage_dataset(args.data_dir, args.cache_dir, args.participants, args.overwrite)
    print(f"complete: {len(records)} participant caches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
