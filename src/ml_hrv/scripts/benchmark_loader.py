"""Reproducible macOS mmap DataLoader benchmark for the causal/beat pipeline."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml_hrv.config import ExperimentConfig
from ml_hrv.data.dataset import RawslotWindowDataset, collate_rawslot
from ml_hrv.data.samplers import ParticipantDeviceBatchSampler
from ml_hrv.data.schema import discover_participant_files
from ml_hrv.data.splits import make_group_folds


@dataclass(frozen=True)
class Candidate:
    name: str
    workers: int
    persistent: bool
    cpu_threads: int
    windows_per_batch: int
    cached_stats: bool = True


CANDIDATES = (
    Candidate("w0_t1_b4_cached", 0, False, 1, 4),
    Candidate("w0_t4_b4_cached", 0, False, 4, 4),
    Candidate("w2_np_t4_b4_cached", 2, False, 4, 4),
    Candidate("w2_p_t4_b4_cached", 2, True, 4, 4),
    Candidate("w4_p_t4_b4_cached", 4, True, 4, 4),
    Candidate("w0_t4_b2_cached", 0, False, 4, 2),
    Candidate("w0_t4_b8_cached", 0, False, 4, 8),
    Candidate("w0_t4_b4_legacy", 0, False, 4, 4, False),
)


def make_dataset(config: ExperimentConfig, participants, cached_stats: bool) -> RawslotWindowDataset:
    return RawslotWindowDataset(
        config.data.cache_dir,
        participants,
        config.data.devices,
        config.data.segment_seconds,
        config.data.window_seconds,
        config.data.accel_mode,
        config.data.qc_only,
        config.data.cache_open_participants,
        use_precomputed_stats=cached_stats,
    )


def time_loader(loader: DataLoader, warmup: int, measured: int) -> dict[str, float]:
    iterator = iter(loader)
    for _ in range(warmup):
        next(iterator)
    start = time.perf_counter()
    samples = 0
    for _ in range(measured):
        batch = next(iterator)
        samples += int(batch["values"].shape[0])
    seconds = time.perf_counter() - start
    return {
        "batches": measured,
        "samples": samples,
        "seconds": seconds,
        "batches_per_second": measured / seconds,
        "samples_per_second": samples / seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "configs" / "phase1.json"))
    parser.add_argument("--output", default="artifacts/ml_hrv/benchmarks/loader_cpu.json")
    parser.add_argument("--train-batches", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=30)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument(
        "--candidates", nargs="*", default=None,
        help="Optional candidate names; useful where multiprocessing is sandbox-restricted.",
    )
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    participants = list(discover_participant_files(config.data.source_dir))
    fold = make_group_folds(participants, 4, config.train.seed)[0]
    results = []
    selected = [x for x in CANDIDATES if args.candidates is None or x.name in args.candidates]
    unknown = set(args.candidates or ()) - {x.name for x in CANDIDATES}
    if unknown:
        parser.error(f"unknown candidates: {sorted(unknown)}")
    for candidate in selected:
        torch.set_num_threads(candidate.cpu_threads)
        train_data = make_dataset(config, fold.train, candidate.cached_stats)
        val_data = make_dataset(config, fold.val, candidate.cached_stats)
        sampler = ParticipantDeviceBatchSampler(
            train_data.sample_refs,
            candidate.windows_per_batch,
            args.train_batches + args.warmup_batches,
            config.train.seed,
        )
        common = {
            "num_workers": candidate.workers,
            "persistent_workers": candidate.persistent if candidate.workers else False,
            "collate_fn": collate_rawslot,
        }
        train_loader = DataLoader(train_data, batch_sampler=sampler, **common)
        val_loader = DataLoader(
            val_data,
            batch_size=candidate.windows_per_batch * len(config.data.devices),
            shuffle=False,
            **common,
        )
        train_result = time_loader(train_loader, args.warmup_batches, args.train_batches)
        val_result = time_loader(val_loader, args.warmup_batches, args.val_batches)
        row = {**asdict(candidate), "train": train_result, "validation_sequential": val_result}
        results.append(row)
        print(json.dumps(row), flush=True)
        del train_loader, val_loader, train_data, val_data
    payload = {
        "protocol": {
            "fold": fold.name,
            "train_batches": args.train_batches,
            "val_batches": args.val_batches,
            "warmup_batches": args.warmup_batches,
            "validation_order": "participant/window/device sequential",
            "scope": "DataLoader+normalization+collation only; no model compute",
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "results": results,
        "not_run": [x.name for x in CANDIDATES if x not in selected],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
