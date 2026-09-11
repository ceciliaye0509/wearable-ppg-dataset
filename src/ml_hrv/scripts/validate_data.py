"""Validate raw-slot fields and emit a compact dataset manifest."""

from __future__ import annotations

import argparse
import json

from ml_hrv.data.schema import discover_participant_files, validate_npz_schema


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()
    records = [validate_npz_schema(path) for path in discover_participant_files(args.data_dir).values()]
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
