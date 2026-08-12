"""
Evaluate v1.2 frozen device/channel baseline under training motion thresholds.

This script reads existing full-cohort channel metrics, applies the frozen
device x channel v1.2 choices, filters `training_stride30` windows by a
common-window intersection where every reported device x channel satisfies
`accel_motion_mean_mag < threshold`, and writes threshold-specific CSV/MD
reports plus an overall sensitivity summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import config  # noqa: E402
import formal_freeze_v1_2_device_channel_params as v12_base  # noqa: E402


METRICS = ("RMSSD", "SDNN")


def _fmt(value: object, digits: int = 2) -> str:
    try:
        f = float(value)
    except Exception:
        return ""
    if not np.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


def _threshold_label(value: object) -> str:
    return f"{float(value):.1f}"


def _impact_note(device: str, channel: str) -> str:
    if device == "Earring" and channel == "ppg_green":
        return "common low-motion 子集下误差最低；阈值放宽后样本更充分但 MAE 上升。"
    if device == "Earring" and channel == "ppg_ir":
        return "common low-motion 子集下接近 green；阈值放宽后仍是 Earring 的主要 RMSSD 通道。"
    if device == "Ring" and channel == "ppg_green":
        return "在 common-motion 横向比较中保持 Ring 最佳；严格阈值样本很少。"
    if device == "Ring" and channel == "ppg_ir":
        return "coverage 和 RMSSD agreement 均弱于 Ring green；严格阈值下 R 不稳定。"
    if device == "Watch" and channel == "ppg_green":
        return "Watch 的可用通道；common-motion 下 coverage 仍明显低于 Earring/Ring green。"
    if device == "Watch" and channel == "ppg_ir":
        return "即使使用 common low-motion 子集，IR 仍明显失败。"
    return "用于比较阈值放宽时窗口数、coverage、MAE 和 R 的变化。"


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _agreement(ppg: np.ndarray, ecg: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(ppg) & np.isfinite(ecg)
    p = ppg[mask]
    e = ecg[mask]
    if p.size == 0:
        return {"n_valid": 0, "MAE": np.nan, "RMSE": np.nan, "R": np.nan, "bias": np.nan}
    diff = p - e
    return {
        "n_valid": int(p.size),
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "R": _safe_corr(p, e),
        "bias": float(np.mean(diff)),
    }


def _load_frozen_predictions(metrics: pd.DataFrame, choices: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, row in choices.iterrows():
        gate = v12_base._gate_by_name(str(row["gate"]))
        mask = (
            (metrics["device"].astype(str) == str(row["device"]))
            & (metrics["channel"].astype(str) == str(row["channel"]))
            & (metrics["peak_method"].astype(str) == str(row["peak_method"]))
        )
        subset = metrics[mask].copy()
        subset["frozen_gate"] = gate.name
        subset["frozen_gate_pass"] = v12_base._gate_mask(subset, gate)
        pieces.append(subset)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _common_motion_subset(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    motion_pass = pred[pred["accel_motion_mean_mag"].astype(float) < threshold].copy()
    required_pairs = pred[["device", "channel"]].drop_duplicates().shape[0]
    common_counts = motion_pass.groupby(["participant", "window_index"], dropna=False)[["device", "channel"]].apply(
        lambda x: x.drop_duplicates().shape[0]
    )
    common_keys = common_counts[common_counts == required_pairs].index
    if len(common_keys) == 0:
        return pred.iloc[0:0].copy()
    common_key_df = pd.DataFrame(list(common_keys), columns=["participant", "window_index"])
    return pred.merge(common_key_df, on=["participant", "window_index"], how="inner").copy()


def _summarize(
    pred: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    participant_rows: list[dict[str, object]] = []

    motion_subset = _common_motion_subset(pred, threshold)
    valid_pred = motion_subset[motion_subset["frozen_gate_pass"]].copy()

    for keys, total_group in motion_subset.groupby(["device", "channel"], dropna=False):
        device, channel = keys
        valid_group = valid_pred[(valid_pred["device"] == device) & (valid_pred["channel"] == channel)]
        n_motion = int(total_group[["participant", "window_index"]].drop_duplicates().shape[0])
        for metric in METRICS:
            stats = _agreement(
                valid_group[f"ppg_{metric.lower()}_ms"].to_numpy(float),
                valid_group[f"ecg_{metric.lower()}_ms"].to_numpy(float),
            )
            rows.append(
                {
                    "motion_threshold": threshold,
                    "motion_filter": f"common-window all device/channel accel_motion_mean_mag < {_threshold_label(threshold)}",
                    "role": "training_stride30",
                    "device": device,
                    "channel": channel,
                    "hrv_metric": metric,
                    "motion_windows": n_motion,
                    **stats,
                    "coverage_within_motion_pct": float(100.0 * stats["n_valid"] / n_motion) if n_motion else np.nan,
                }
            )

    for keys, total_group in motion_subset.groupby(["participant", "device", "channel"], dropna=False):
        participant, device, channel = keys
        valid_group = valid_pred[
            (valid_pred["participant"] == participant)
            & (valid_pred["device"] == device)
            & (valid_pred["channel"] == channel)
        ]
        n_motion = int(total_group["window_index"].nunique())
        for metric in METRICS:
            stats = _agreement(
                valid_group[f"ppg_{metric.lower()}_ms"].to_numpy(float),
                valid_group[f"ecg_{metric.lower()}_ms"].to_numpy(float),
            )
            participant_rows.append(
                {
                    "motion_threshold": threshold,
                    "motion_filter": f"common-window all device/channel accel_motion_mean_mag < {_threshold_label(threshold)}",
                    "role": "training_stride30",
                    "participant": participant,
                    "device": device,
                    "channel": channel,
                    "hrv_metric": metric,
                    "motion_windows": n_motion,
                    **stats,
                    "coverage_within_motion_pct": float(100.0 * stats["n_valid"] / n_motion) if n_motion else np.nan,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(participant_rows)


def _write_threshold_md(out_path: Path, threshold: float, summary: pd.DataFrame, participant: pd.DataFrame) -> None:
    threshold_text = _threshold_label(threshold)
    rmssd = summary[summary["hrv_metric"] == "RMSSD"].sort_values(["MAE", "coverage_within_motion_pct"], ascending=[True, False])
    sdnn = summary[summary["hrv_metric"] == "SDNN"][["device", "channel", "MAE", "R"]].rename(
        columns={"MAE": "SDNN_MAE", "R": "SDNN_R"}
    )
    joined = rmssd.merge(sdnn, on=["device", "channel"], how="left")

    part_rmssd = participant[participant["hrv_metric"] == "RMSSD"].copy()
    variability = (
        part_rmssd.groupby(["device", "channel"], dropna=False)
        .agg(
            participant_R_median=("R", "median"),
            participant_R_q25=("R", lambda x: x.quantile(0.25)),
            participant_R_q75=("R", lambda x: x.quantile(0.75)),
            participant_coverage_median=("coverage_within_motion_pct", "median"),
            participant_coverage_q25=("coverage_within_motion_pct", lambda x: x.quantile(0.25)),
            participant_coverage_q75=("coverage_within_motion_pct", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )

    lines = [
        f"# v1.2 Baseline Motion Threshold < {threshold_text}",
        "",
        "本报告只使用 `training_stride30`，并先取同一批 `participant + window_index` common-window 交集。",
        "",
        f"- Motion filter: all reported `device x channel` rows satisfy `accel_motion_mean_mag < {threshold_text}`.",
        "- Baseline: frozen v1.2 device x channel parameters, both green and IR reported.",
        "- Denominator: the same common-motion windows for every device/channel at this threshold.",
        "- MAE/R are computed on each device/channel's QC-valid predictions inside this common-motion subset.",
        "",
        "## Aggregate Results",
        "",
        "| Rank | Device | Channel | Motion windows | Valid preds | Coverage within motion | RMSSD MAE | RMSSD R | SDNN MAE | SDNN R |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, (_, row) in enumerate(joined.iterrows(), start=1):
        lines.append(
            f"| {i} | `{row['device']}` | `{row['channel']}` | {int(row['motion_windows'])} | {int(row['n_valid'])} | "
            f"{_fmt(row['coverage_within_motion_pct'])}% | {_fmt(row['MAE'])} ms | {_fmt(row['R'], 3)} | "
            f"{_fmt(row['SDNN_MAE'])} ms | {_fmt(row['SDNN_R'], 3)} |"
        )

    lines.extend([
        "",
        "## Participant R/Coverage Variability",
        "",
        "| Device | Channel | Participant R median [IQR] | Participant coverage median [IQR] |",
        "|---|---|---:|---:|",
    ])
    for _, row in variability.sort_values(["device", "channel"]).iterrows():
        lines.append(
            f"| `{row['device']}` | `{row['channel']}` | "
            f"{_fmt(row['participant_R_median'], 3)} [{_fmt(row['participant_R_q25'], 3)}, {_fmt(row['participant_R_q75'], 3)}] | "
            f"{_fmt(row['participant_coverage_median'])}% [{_fmt(row['participant_coverage_q25'])}, {_fmt(row['participant_coverage_q75'])}] |"
        )

    lines.extend([
        "",
        "## Output",
        "",
        f"- CSV: `{out_path.with_suffix('.csv').name}`",
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_overall_md(out_dir: Path, all_summary: pd.DataFrame) -> None:
    rmssd = all_summary[all_summary["hrv_metric"] == "RMSSD"].copy()
    sdnn = all_summary[all_summary["hrv_metric"] == "SDNN"][["motion_threshold", "device", "channel", "MAE", "R"]].rename(
        columns={"MAE": "SDNN_MAE", "R": "SDNN_R"}
    )
    joined = rmssd.merge(sdnn, on=["motion_threshold", "device", "channel"], how="left")
    joined = joined.sort_values(["motion_threshold", "MAE", "coverage_within_motion_pct"], ascending=[True, True, False])

    best_rows = (
        joined.sort_values(["motion_threshold", "device", "MAE"], ascending=[True, True, True])
        .groupby(["motion_threshold", "device"], dropna=False)
        .head(1)
        .reset_index(drop=True)
    )
    thresholds = sorted(joined["motion_threshold"].dropna().unique())
    strict_t = thresholds[0]
    loose_t = thresholds[-1]
    strict_label = _threshold_label(strict_t)
    loose_label = _threshold_label(loose_t)
    impact_rows: list[dict[str, object]] = []
    for (device, channel), group in rmssd.groupby(["device", "channel"], dropna=False):
        group = group.sort_values("motion_threshold")
        first = group[group["motion_threshold"] == strict_t].iloc[0]
        last = group[group["motion_threshold"] == loose_t].iloc[0]
        impact_rows.append(
            {
                "device": device,
                "channel": channel,
                "strict_motion_windows": int(first["motion_windows"]),
                "loose_motion_windows": int(last["motion_windows"]),
                "strict_coverage": float(first["coverage_within_motion_pct"]),
                "loose_coverage": float(last["coverage_within_motion_pct"]),
                "strict_MAE": float(first["MAE"]),
                "loose_MAE": float(last["MAE"]),
                "strict_R": float(first["R"]),
                "loose_R": float(last["R"]),
            }
        )

    lines = [
        "# v1.2 Motion Threshold Sensitivity on training_stride30",
        "",
        "本汇总使用当前 v1.2 最佳 baseline：每个 `device x channel` 使用冻结参数，并且 green/IR 都汇报。",
        "",
        "Motion threshold 横向比较使用同一批窗口交集：每个 threshold 先取所有汇报的 `device x channel` 都满足 `accel_motion_mean_mag < threshold` 的 `participant + window_index`，再在这批 common-motion windows 内计算各通道 QC-valid coverage、MAE 和 R。",
        "",
        "## Threshold Impact Comparison",
        "",
        "下表比较 motion threshold 从严格到宽松时，对当前 baseline 的影响。Coverage 使用同一批 common-motion windows 作分母。",
        "",
        f"| Device | Channel | Motion windows `<{strict_label} -> <{loose_label}` | Coverage `<{strict_label} -> <{loose_label}` | RMSSD MAE `<{strict_label} -> <{loose_label}` | RMSSD R `<{strict_label} -> <{loose_label}` | 主要影响 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in sorted(impact_rows, key=lambda x: (str(x["device"]), str(x["channel"]))):
        lines.append(
            f"| `{row['device']}` | `{row['channel']}` | "
            f"{row['strict_motion_windows']} -> {row['loose_motion_windows']} | "
            f"{_fmt(row['strict_coverage'])}% -> {_fmt(row['loose_coverage'])}% | "
            f"{_fmt(row['strict_MAE'])} -> {_fmt(row['loose_MAE'])} ms | "
            f"{_fmt(row['strict_R'], 3)} -> {_fmt(row['loose_R'], 3)} | "
            f"{_impact_note(str(row['device']), str(row['channel']))} |"
        )

    lines.extend([
        "",
        "核心结论：",
        "",
        "- `<0.1` 是最严格的 common low-motion 子集，窗口数很少，代表性有限。",
        "- `<0.2` 是更平衡的 common low-motion sensitivity：窗口数明显多于 `<0.1`。",
        "- `<0.5` 和 `<1.0` 更接近 common low-to-moderate motion/full training 行为，窗口数多，但 Ring/Watch 的 coverage within motion 仍明显低于 Earring。",
        "- 对当前 v1.2 baseline，common low-motion 子集会改变各通道 RMSSD 排序；Watch IR 始终不适合作为 HRV baseline。",
        "",
        "## Aggregate RMSSD/SDNN Results",
        "",
        "| Threshold | Device | Channel | Motion windows | Valid preds | Coverage within motion | RMSSD MAE | RMSSD R | SDNN MAE | SDNN R |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in joined.iterrows():
        lines.append(
            f"| <{_threshold_label(row['motion_threshold'])} | `{row['device']}` | `{row['channel']}` | {int(row['motion_windows'])} | {int(row['n_valid'])} | "
            f"{_fmt(row['coverage_within_motion_pct'])}% | {_fmt(row['MAE'])} ms | {_fmt(row['R'], 3)} | "
            f"{_fmt(row['SDNN_MAE'])} ms | {_fmt(row['SDNN_R'], 3)} |"
        )

    lines.extend([
        "",
        "## Best Channel per Device at Each Motion Threshold",
        "",
        "这里只用于解释 motion sensitivity；正式报告仍保留两个通道。",
        "",
        "| Threshold | Device | Best channel by RMSSD MAE | RMSSD MAE | RMSSD R | Coverage within motion |",
        "|---:|---|---|---:|---:|---:|",
    ])
    for _, row in best_rows.iterrows():
        lines.append(
            f"| <{_threshold_label(row['motion_threshold'])} | `{row['device']}` | `{row['channel']}` | "
            f"{_fmt(row['MAE'])} ms | {_fmt(row['R'], 3)} | {_fmt(row['coverage_within_motion_pct'])}% |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- 更严格的 common motion threshold 通常提升部分通道的 RMSSD R、降低 MAE，但会牺牲大量 motion windows。",
        "- Earring green/IR 在低 motion 子集下保留较多窗口，说明 Earring 的 motion 分布相对更温和；但 RMSSD agreement 仍通常是 IR 更好，SDNN 通常是 green 更好。",
        "- Ring green 在所有阈值下都是 Ring 的最佳通道；Ring IR 保留汇报，但 MAE/R/coverage 都弱于 green。",
        "- Watch green 是最受 motion threshold 影响的可用通道之一；严格阈值下样本量很小，因此不能单独代表全训练集表现。",
        "- Watch IR 在所有阈值下仍明显失败；motion filtering 不能把 Watch IR 修成可用 HRV baseline。",
        "- Coverage within motion 使用 common-motion 子集作为分母，不能直接和完整 training_stride30 overall coverage 混为一谈。",
        "",
        "## Files",
        "",
        "- `v1_2_motion_threshold_sensitivity_training_stride30.csv`",
        "- `v1_2_motion_threshold_sensitivity_participant_training_stride30.csv`",
        "- `v1_2_motion_lt_0p1_training_stride30.md`",
        "- `v1_2_motion_lt_0p2_training_stride30.md`",
        "- `v1_2_motion_lt_0p5_training_stride30.md`",
        "- `v1_2_motion_lt_1p0_training_stride30.md`",
    ])
    (out_dir / "v1_2_motion_threshold_sensitivity_training_stride30.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v1.2 training_stride30 motion threshold sensitivity.")
    parser.add_argument(
        "--channel-metrics-csv",
        default=str(
            config.HEURISTIC_RESULT_ROOT
            / "all_participants_devicewise_baseline_v1_1_plus_neurokit"
            / "v1_1_plus_neurokit_channel_metrics.csv"
        ),
    )
    parser.add_argument(
        "--frozen-choices-csv",
        default=str(
            config.HEURISTIC_RESULT_ROOT
            / "formal_v1_2_report_both_channels"
            / "v1_2_both_channels_frozen_choices.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            (config.HEURISTIC_BASE_DIR if hasattr(config, "HEURISTIC_BASE_DIR") else config.PACKAGE_ROOT)
            / "archive"
            / "legacy_current_best_baseline"
            / "motion_tests"
        ),
    )
    parser.add_argument("--thresholds", default="0.1,0.2,0.5,1.0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(Path(args.channel_metrics_csv).resolve())
    metrics = metrics[metrics["role"] == "training_stride30"].copy()
    choices = pd.read_csv(Path(args.frozen_choices_csv).resolve())
    pred = _load_frozen_predictions(metrics, choices)

    all_summary: list[pd.DataFrame] = []
    all_participant: list[pd.DataFrame] = []
    for threshold in [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]:
        summary, participant = _summarize(pred, threshold)
        tag = str(threshold).replace(".", "p")
        csv_path = out_dir / f"v1_2_motion_lt_{tag}_training_stride30.csv"
        md_path = out_dir / f"v1_2_motion_lt_{tag}_training_stride30.md"
        summary.to_csv(csv_path, index=False)
        _write_threshold_md(md_path, threshold, summary, participant)
        all_summary.append(summary)
        all_participant.append(participant)

    summary_df = pd.concat(all_summary, ignore_index=True)
    participant_df = pd.concat(all_participant, ignore_index=True)
    summary_df.to_csv(out_dir / "v1_2_motion_threshold_sensitivity_training_stride30.csv", index=False)
    participant_df.to_csv(out_dir / "v1_2_motion_threshold_sensitivity_participant_training_stride30.csv", index=False)
    _write_overall_md(out_dir, summary_df)
    (out_dir / "v1_2_motion_threshold_sensitivity_training_stride30_summary.json").write_text(
        json.dumps(
            {
                "channel_metrics_csv": str(Path(args.channel_metrics_csv).resolve()),
                "frozen_choices_csv": str(Path(args.frozen_choices_csv).resolve()),
                "thresholds": [float(x.strip()) for x in args.thresholds.split(",") if x.strip()],
                "role": "training_stride30",
                "motion_column": "accel_motion_mean_mag",
                "reports_both_channels": True,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[saved] {out_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
