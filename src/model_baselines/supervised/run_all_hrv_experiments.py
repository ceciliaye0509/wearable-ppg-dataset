#!/usr/bin/env python3
"""Run direct-HRV and peak/RR HRV experiments for all wearable devices.

Place this file in ``src/model_baselines/supervised/`` next to
``main_supervised_baseline.py`` and ``hrv_ext.py``.

Examples
--------
Smoke test::

    python3 run_all_hrv_experiments.py \
        --data-dir ~/hf_hrv_data/synced_3device_rawaligned_training_v1_stride30_rawslots \
        --mode smoke

Full experiment::

    python3 run_all_hrv_experiments.py \
        --data-dir ~/hf_hrv_data/synced_3device_rawaligned_training_v1_stride30_rawslots \
        --mode full

The script runs experiments sequentially on one GPU and writes:

* ``experiment_outputs/logs/<method>_<device>.txt``: complete run logs.
* ``experiment_outputs/hrv_all_experiments_summary.txt``: combined results.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEVICES = ("earring", "ring", "watch")
METHODS = ("direct", "peak")

METRIC_RE = re.compile(
    r"^\s*(ecg_(?:sdnn|rmssd)_corrected_ms)\s+"
    r"R2\s+([-+\w.]+)\+/-([-+\w.]+)\s+"
    r"r\s+([-+\w.]+)\+/-([-+\w.]+)\s+"
    r"MAE\s+([-+\w.]+)\+/-([-+\w.]+)\s+ms\s*$",
    re.MULTILINE,
)
PEAK_F1_RE = re.compile(r"Peak detection:.*?F1=([-+\d.eE]+)")
PEAK_COVERAGE_RE = re.compile(r"HRV coverage:.*?\(([-+\d.eE]+)%\)")
PEAK_CORRECTION_RE = re.compile(
    r"Mean RR correction ratio:\s*([-+\d.eE]+)"
)


@dataclass(frozen=True)
class Experiment:
    method: str
    device: str

    @property
    def tag(self) -> str:
        return f"{self.method}_{self.device}"


def parse_csv_choices(raw: str, allowed: Iterable[str], name: str) -> list[str]:
    allowed_set = set(allowed)
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [item for item in values if item not in allowed_set]
    if not values or invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid {name}: {invalid or raw}. Allowed: {sorted(allowed_set)}"
        )
    return values


def finite_number(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def mean(values: list[float]) -> float:
    usable = [x for x in values if x == x]
    return sum(usable) / len(usable) if usable else float("nan")


def build_command(args: argparse.Namespace, exp: Experiment) -> list[str]:
    smoke = args.mode == "smoke"

    if exp.method == "direct":
        command = [
            sys.executable,
            "-u",
            str(args.main_script),
            "--task",
            "hrv_seg",
            "--device_name",
            exp.device,
            "--agg",
            args.agg,
            "--batch_size",
            str(args.direct_batch_size),
            "--n_epoch",
            str(3 if smoke else args.direct_epochs),
            "--patience",
            str(2 if smoke else args.direct_patience),
            "--lr",
            str(args.direct_lr),
        ]
    else:
        command = [
            sys.executable,
            "-u",
            str(args.main_script),
            "--task",
            "peak",
            "--device_name",
            exp.device,
            "--batch_size",
            str(args.peak_batch_size),
            "--n_epoch",
            str(3 if smoke else args.peak_epochs),
            "--patience",
            str(2 if smoke else args.peak_patience),
            "--lr",
            str(args.peak_lr),
        ]

    command.extend(
        [
            "--data_dir",
            str(args.data_dir),
            "--cuda",
            str(args.cuda),
            "--resample_hz",
            str(args.resample_hz),
        ]
    )
    if smoke:
        command.extend(["--limit", str(args.smoke_limit)])
    return command


def stream_run(
    command: list[str],
    log_path: Path,
    cwd: Path,
    env: dict[str, str],
) -> int:
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_file.write("COMMAND: " + " ".join(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return process.wait()


def parse_log(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics: dict[str, dict[str, float]] = {}
    for match in METRIC_RE.finditer(text):
        metric = match.group(1)
        metrics[metric] = {
            "r2_mean": finite_number(match.group(2)),
            "r2_std": finite_number(match.group(3)),
            "r_mean": finite_number(match.group(4)),
            "r_std": finite_number(match.group(5)),
            "mae_mean": finite_number(match.group(6)),
            "mae_std": finite_number(match.group(7)),
        }

    f1_values = [float(x) for x in PEAK_F1_RE.findall(text)]
    coverage_values = [float(x) for x in PEAK_COVERAGE_RE.findall(text)]
    correction_values = [float(x) for x in PEAK_CORRECTION_RE.findall(text)]
    return {
        "metrics": metrics,
        "peak_f1_mean": mean(f1_values),
        "peak_coverage_pct_mean": mean(coverage_values),
        "rr_correction_ratio_mean": mean(correction_values),
    }


def fmt(value: float, digits: int = 3) -> str:
    return "nan" if value != value else f"{value:.{digits}f}"


def write_summary(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, object]],
) -> None:
    lines = [
        "HRV ALL-DEVICE EXPERIMENT SUMMARY",
        "=" * 100,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Mode: {args.mode}",
        f"Data: {args.data_dir}",
        f"Devices: {', '.join(args.devices)}",
        f"Methods: {', '.join(args.methods)}",
        f"Resample: {args.resample_hz:g} Hz",
        f"Direct aggregation: {args.agg}",
        "",
        "Method definitions:",
        "  direct = hrv_seg: PPG segments -> aggregate features -> SDNN/RMSSD",
        "  peak   = PPG -> peak probabilities -> RR correction -> SDNN/RMSSD",
        "",
        (
            "method   device    status  metric                    "
            "R2 mean+/-std       r mean+/-std        MAE mean+/-std (ms)"
        ),
        "-" * 100,
    ]

    for row in rows:
        method = str(row["method"])
        device = str(row["device"])
        status = str(row["status"])
        parsed = row.get("parsed", {})
        metrics = parsed.get("metrics", {}) if isinstance(parsed, dict) else {}

        if not metrics:
            lines.append(
                f"{method:<8} {device:<9} {status:<7} "
                f"{'NO FINAL METRICS':<25} {'-':<19} {'-':<19} {'-'}"
            )
            continue

        for metric_name in (
            "ecg_sdnn_corrected_ms",
            "ecg_rmssd_corrected_ms",
        ):
            metric = metrics.get(metric_name)
            if metric is None:
                continue
            lines.append(
                f"{method:<8} {device:<9} {status:<7} {metric_name:<25} "
                f"{fmt(metric['r2_mean']):>7}+/-{fmt(metric['r2_std']):<7} "
                f"{fmt(metric['r_mean']):>7}+/-{fmt(metric['r_std']):<7} "
                f"{fmt(metric['mae_mean'], 2):>7}+/-{fmt(metric['mae_std'], 2):<7}"
            )

    lines.extend(["", "PEAK/RR DIAGNOSTICS", "-" * 100])
    peak_rows = [row for row in rows if row["method"] == "peak"]
    if peak_rows:
        lines.append(
            "device    mean peak F1    mean HRV coverage (%)    mean RR correction ratio"
        )
        for row in peak_rows:
            parsed = row.get("parsed", {})
            if not isinstance(parsed, dict):
                parsed = {}
            lines.append(
                f"{str(row['device']):<9} "
                f"{fmt(float(parsed.get('peak_f1_mean', float('nan')))):>12} "
                f"{fmt(float(parsed.get('peak_coverage_pct_mean', float('nan'))), 1):>25} "
                f"{fmt(float(parsed.get('rr_correction_ratio_mean', float('nan')))):>27}"
            )
    else:
        lines.append("Peak method was not requested.")

    lines.extend(["", "LOG FILES", "-" * 100])
    for row in rows:
        lines.append(
            f"{row['method']}/{row['device']}: {row['log_path']} "
            f"(exit code {row['return_code']})"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser(script_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all-device direct-HRV and peak/RR experiments sequentially."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--devices", default=",".join(DEVICES))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--output-dir", type=Path, default=script_dir / "experiment_outputs")
    parser.add_argument("--main-script", type=Path, default=script_dir / "main_supervised_baseline.py")
    parser.add_argument("--resample-hz", type=float, default=25.0)
    parser.add_argument("--agg", choices=("mean", "lstm"), default="mean")
    parser.add_argument("--smoke-limit", type=int, default=50)

    parser.add_argument("--direct-epochs", type=int, default=50)
    parser.add_argument("--direct-patience", type=int, default=8)
    parser.add_argument("--direct-batch-size", type=int, default=32)
    parser.add_argument("--direct-lr", type=float, default=5e-4)

    parser.add_argument("--peak-epochs", type=int, default=30)
    parser.add_argument("--peak-patience", type=int, default=5)
    parser.add_argument("--peak-batch-size", type=int, default=8)
    parser.add_argument("--peak-lr", type=float, default=5e-4)
    return parser


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = build_parser(script_dir)
    args = parser.parse_args()
    args.devices = parse_csv_choices(args.devices, DEVICES, "devices")
    args.methods = parse_csv_choices(args.methods, METHODS, "methods")
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.main_script = args.main_script.expanduser().resolve()

    if not args.data_dir.is_dir():
        parser.error(f"Data directory not found: {args.data_dir}")
    if not args.main_script.is_file():
        parser.error(f"Main script not found: {args.main_script}")

    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "hrv_all_experiments_summary.txt"

    model_baselines_dir = script_dir.parent
    env = os.environ.copy()
    pythonpath = [str(model_baselines_dir), str(script_dir)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    experiments = [
        Experiment(method=method, device=device)
        for method in args.methods
        for device in args.devices
    ]
    rows: list[dict[str, object]] = []

    print(f"Running {len(experiments)} experiments sequentially on cuda:{args.cuda}")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output_dir}")

    for index, exp in enumerate(experiments, start=1):
        print("\n" + "=" * 80)
        print(f"[{index}/{len(experiments)}] {exp.method} / {exp.device}")
        print("=" * 80)

        command = build_command(args, exp)
        log_path = log_dir / f"{exp.tag}.txt"
        return_code = stream_run(command, log_path, script_dir, env)
        parsed = parse_log(log_path)
        # Some revisions of main_supervised_baseline.py raise an error only in
        # their legacy per-run summary writer after already printing complete
        # HRV metrics. Preserve those usable results and expose the nonzero exit
        # code in the LOG FILES section instead of discarding the experiment.
        status = "OK" if parsed["metrics"] else "FAILED"
        rows.append(
            {
                "method": exp.method,
                "device": exp.device,
                "status": status,
                "return_code": return_code,
                "log_path": log_path,
                "parsed": parsed,
            }
        )
        write_summary(summary_path, args, rows)
        print(f"[{status}] log: {log_path}")
        print(f"Updated summary: {summary_path}")

    failures = [row for row in rows if row["status"] != "OK"]
    print("\n" + "=" * 80)
    print(f"Completed {len(rows) - len(failures)}/{len(rows)} experiments successfully")
    print(f"Combined summary: {summary_path}")
    if failures:
        print("Failed experiments:")
        for row in failures:
            print(f"  {row['method']}/{row['device']} -> {row['log_path']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
