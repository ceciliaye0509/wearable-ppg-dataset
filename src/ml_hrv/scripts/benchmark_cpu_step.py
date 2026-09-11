"""Benchmark eager FP32 causal-direct training steps on the current CPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml_hrv.config import ExperimentConfig
from ml_hrv.data.dataset import RawslotWindowDataset, collate_rawslot
from ml_hrv.data.samplers import ParticipantDeviceBatchSampler
from ml_hrv.data.schema import discover_participant_files
from ml_hrv.data.splits import make_group_folds
from ml_hrv.models.pipeline import RawContinuousHRVModel


CANDIDATES = ((1, 4), (2, 4), (4, 4), (8, 4), (4, 2), (4, 8))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "configs" / "phase1.json"))
    parser.add_argument("--output", default="artifacts/ml_hrv/benchmarks/cpu_step.json")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    participants = list(discover_participant_files(config.data.source_dir))
    fold = make_group_folds(participants, 4, config.train.seed)[0]
    dataset = RawslotWindowDataset(
        config.data.cache_dir, fold.train, config.data.devices,
        config.data.segment_seconds, config.data.window_seconds,
        config.data.accel_mode, config.data.qc_only, config.data.cache_open_participants,
    )
    results = []
    for threads, windows_per_batch in CANDIDATES:
        torch.set_num_threads(threads)
        sampler = ParticipantDeviceBatchSampler(
            dataset.sample_refs, windows_per_batch, 1, config.train.seed
        )
        batch = next(iter(DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_rawslot)))
        torch.manual_seed(config.train.seed)
        model = RawContinuousHRVModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        durations = []
        for step in range(args.warmup + args.steps):
            start = time.perf_counter()
            output = model(
                batch["values"], batch["mask"], batch["jitter"],
                batch["device_id"], batch["accel"], compute_beat=False,
            )
            loss = output["direct_loc"].square().mean() + output["direct_logvar"].square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
            optimizer.step()
            if step >= args.warmup:
                durations.append(time.perf_counter() - start)
        seconds = sum(durations)
        samples = int(batch["values"].shape[0]) * args.steps
        row = {
            "cpu_threads": threads,
            "windows_per_batch": windows_per_batch,
            "device_samples_per_batch": int(batch["values"].shape[0]),
            "steps": args.steps,
            "seconds": seconds,
            "seconds_per_step": seconds / args.steps,
            "samples_per_second": samples / seconds,
            "precision": "FP32",
            "compile": False,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        del model, optimizer, batch
    payload = {
        "scope": "eager FP32 causal-direct forward+backward+AdamW; fixed fold-0 sampled batch; loader excluded",
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
