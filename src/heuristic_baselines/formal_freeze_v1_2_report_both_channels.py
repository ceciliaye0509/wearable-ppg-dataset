"""
Formal v1.2 baseline optimizer that reports both PPG channels.

Scientific boundary:
  - freeze parameters using only `training_stride30`;
  - evaluate on `strict_reference` only after freezing;
  - no subject-level held-out split in this run;
  - no cross-device fusion;
  - no best-channel selection between green and IR.

The optimized unit is device x channel. Each device/channel pair may use its own
peak method, bandpass/correction parameters, and QC gate, but both channels are
reported after freezing.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import config  # noqa: E402
import formal_freeze_v1_2_device_channel_params as base  # noqa: E402


POLICY = "fixed_device_channel_params_report_both_channels"


def _select_per_device_channel(training_summary: pd.DataFrame, min_coverage_pct: float) -> pd.DataFrame:
    rmssd = training_summary[
        (training_summary["hrv_metric"] == "RMSSD")
        & np.isfinite(training_summary["MAE"].to_numpy(float))
    ].copy()
    if rmssd.empty:
        return rmssd

    eligible = rmssd[rmssd["coverage_pct"].astype(float) >= min_coverage_pct].copy()
    if eligible.empty:
        eligible = rmssd.copy()
        eligible["passes_min_channel_coverage"] = False
    else:
        eligible["passes_min_channel_coverage"] = True

    selected = (
        eligible.sort_values(["device", "channel", "MAE", "coverage_pct", "R"], ascending=[True, True, True, False, False])
        .groupby(["device", "channel"], dropna=False)
        .head(1)
        .reset_index(drop=True)
    )
    selected["policy"] = POLICY
    return selected


def _apply_frozen(df: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, row in selected.iterrows():
        gate = base._gate_by_name(str(row["gate"]))
        mask = (
            (df["device"].astype(str) == str(row["device"]))
            & (df["channel"].astype(str) == str(row["channel"]))
            & (df["peak_method"].astype(str) == str(row["peak_method"]))
            & base._gate_mask(df, gate)
        )
        piece = df[mask].copy()
        piece["gate"] = gate.name
        piece["policy"] = POLICY
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _top_candidates(training_summary: pd.DataFrame, min_coverage_pct: float) -> pd.DataFrame:
    top = training_summary[
        (training_summary["hrv_metric"] == "RMSSD")
        & np.isfinite(training_summary["MAE"].to_numpy(float))
        & (training_summary["coverage_pct"].astype(float) >= min_coverage_pct)
    ].copy()
    if top.empty:
        top = training_summary[
            (training_summary["hrv_metric"] == "RMSSD")
            & np.isfinite(training_summary["MAE"].to_numpy(float))
        ].copy()
    return (
        top.sort_values(["device", "channel", "MAE", "coverage_pct"], ascending=[True, True, True, False])
        .groupby(["device", "channel"], dropna=False)
        .head(5)
        .reset_index(drop=True)
    )


def _write_readme(
    out_dir: Path,
    channel_df: pd.DataFrame,
    selected: pd.DataFrame,
    frozen_eval: pd.DataFrame,
    top: pd.DataFrame,
    min_coverage_pct: float,
) -> None:
    lines = [
        "# Formal v1.2 Baseline: Report Both Channels",
        "",
        "This is an experiment record, not manuscript prose.",
        "",
        "## Scientific Boundary",
        "",
        "- Selection uses only `training_stride30`.",
        "- `strict_reference` is used only after the rules are frozen.",
        "- No subject-level held-out split is used in this run.",
        "- No model training, no cross-device fusion, and no ECG-label per-window selection.",
        "- The shared method family is recompute PPG peaks -> derive IBI/QC -> compute HRV -> apply frozen QC.",
        "- The optimized unit is `device x channel`; green and IR are both reported, not reduced to a single best channel.",
        "",
        "## Inputs",
        "",
        f"- Channel metric rows after de-duplication: `{len(channel_df)}`",
        f"- Peak methods: `{channel_df['peak_method'].nunique()}`",
        f"- Gate grid size: `{len(base.GATES)}`",
        f"- Minimum per-channel training coverage target: `{min_coverage_pct:.2f}%`",
        "",
        "## Frozen Device x Channel Choices",
        "",
        "| Device | Channel | Peak method | Gate | Training RMSSD MAE | Training R | Training coverage |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for _, row in selected.sort_values(["device", "channel"]).iterrows():
        lines.append(
            f"| `{row['device']}` | `{row['channel']}` | `{row['peak_method']}` | `{row['gate']}` | "
            f"{base._fmt(row['MAE'])} ms | {base._fmt(row['R'], 3)} | {base._fmt(row['coverage_pct'])}% |"
        )

    lines.extend([
        "",
        "## Frozen Evaluation",
        "",
        "| Role | Device | Channel | Metric | Valid windows | Coverage | MAE | RMSE | R | Bias |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in frozen_eval.sort_values(["role", "device", "channel", "hrv_metric"]).iterrows():
        lines.append(
            f"| {row['role']} | `{row['device']}` | `{row['channel']}` | {row['hrv_metric']} | {int(row['n_valid'])} | "
            f"{base._fmt(row['coverage_pct'])}% | {base._fmt(row['MAE'])} ms | {base._fmt(row['RMSE'])} ms | "
            f"{base._fmt(row['R'], 3)} | {base._fmt(row['bias'])} ms |"
        )

    lines.extend([
        "",
        "## Training Top Candidates",
        "",
        "| Device | Channel | Peak method | Gate | Coverage | MAE | R |",
        "|---|---|---|---|---:|---:|---:|",
    ])
    for _, row in top.iterrows():
        lines.append(
            f"| `{row['device']}` | `{row['channel']}` | `{row['peak_method']}` | `{row['gate']}` | "
            f"{base._fmt(row['coverage_pct'])}% | {base._fmt(row['MAE'])} ms | {base._fmt(row['R'], 3)} |"
        )

    lines.extend([
        "",
        "## Output Files",
        "",
        "| File | Meaning |",
        "|---|---|",
        "| `v1_2_both_channels_training_summary.csv` | Training-only channel/peak/gate candidate summary |",
        "| `v1_2_both_channels_frozen_choices.csv` | Frozen device x channel parameter choices |",
        "| `v1_2_both_channels_frozen_eval.csv` | Frozen evaluation for both channels on training and strict_reference |",
        "| `summary.json` | Machine-readable run metadata |",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze v1.2 baseline and report both green/IR channels.")
    parser.add_argument("--channel-metrics-csv", action="append", default=[])
    parser.add_argument(
        "--out-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "formal_v1_2_report_both_channels"),
    )
    parser.add_argument("--min-channel-coverage-pct", type=float, default=20.0)
    args = parser.parse_args()

    if not args.channel_metrics_csv:
        args.channel_metrics_csv = [
            str(
                config.HEURISTIC_RESULT_ROOT
                / "all_participants_devicewise_baseline_v1_1_plus_neurokit"
                / "v1_1_plus_neurokit_channel_metrics.csv"
            )
        ]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_df = base._load_channel_metrics(args.channel_metrics_csv)
    denominators = base._denominators(channel_df)
    training_df = channel_df[channel_df["role"] == "training_stride30"].copy()
    training_summary = base._candidate_training_summary(training_df, denominators)
    selected = _select_per_device_channel(training_summary, args.min_channel_coverage_pct)
    frozen_pred = _apply_frozen(channel_df, selected)
    frozen_eval = base._summarize_predictions(
        frozen_pred,
        ["dataset", "role", "device", "channel", "policy"],
        denominators,
    )
    top = _top_candidates(training_summary, args.min_channel_coverage_pct)

    training_summary.to_csv(out_dir / "v1_2_both_channels_training_summary.csv", index=False)
    selected.to_csv(out_dir / "v1_2_both_channels_frozen_choices.csv", index=False)
    frozen_eval.to_csv(out_dir / "v1_2_both_channels_frozen_eval.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "channel_metrics_csv": [str(Path(p).resolve()) for p in args.channel_metrics_csv],
                "out_dir": str(out_dir),
                "policy": POLICY,
                "min_channel_coverage_pct": args.min_channel_coverage_pct,
                "selection_role": "training_stride30",
                "final_evaluation_role": "strict_reference",
                "subject_level_held_out_split": False,
                "cross_device_fusion": False,
                "best_channel_selection": False,
                "reports_green_and_ir": True,
                "gate_grid": [asdict(g) for g in base.GATES],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_readme(out_dir, channel_df, selected, frozen_eval, top, args.min_channel_coverage_pct)

    print(f"[saved] {out_dir}")
    print(selected.sort_values(["device", "channel"]).to_string(index=False))
    print(frozen_eval.to_string(index=False))


if __name__ == "__main__":
    main()
