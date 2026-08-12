"""Frozen rawslots baseline: qppgfast devicewise foot/peak versus SciPy peak.

This is the stable, intentionally non-tunable entry point for the rawslots
baseline.  The detector-validation script remains the place for experiments;
this module only reproduces the frozen configuration and its SciPy comparator.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_rawslots_baseline_ablation import _dataset_npz_files
from evaluate_rawslots_detector_fiducials import (
    DATASET_DIR,
    DetectorPipeline,
    _load_channel_metrics_for_path,
)
from append_rawslots_strict_interpolation_amount import _collect as _collect_interpolation_amount
from append_rawslots_strict_interpolation_amount import _summary as _interpolation_amount_summary


METHOD_NAME = "qppgfast_devicewise_v1"
SCIPY_COMPARATOR_NAME = "scipy_peak_v1"
DEVICE_FIDUCIALS = {"Earring": "peak", "Ring": "foot", "Watch": "foot"}
FROZEN_CONFIG = {
    "method": METHOD_NAME,
    "input": "strict interpolation to a 100 Hz grid; maximum raw support gap 100 ms; no boundary extrapolation",
    "max_raw_support_gap_ms": 100.0,
    "common_bandpass_hz": [0.7, 3.5],
    "ibi_correction": True,
    "ibi_correction_threshold": 0.20,
    "polarity_mode": "peak_train",
    "polarity_sensitivity_freeze": {
        "tested_protocol": "waveform_consensus_v1: 5 equal 60-s waveform-only segments; >=4/5 same morphology-score sign; ambiguous skips detector",
        "uses_fiducials_or_ibi": False,
        "decision": "not adopted; retain peak_train",
        "reason": "independent holdout coverage fell by 28.958 percentage points and R fell by 0.035 despite lower MAE",
        "evidence_output": "rawslots_qppgfast_waveform_consensus_freeze_v1",
    },
    "device_fiducials": DEVICE_FIDUCIALS,
    "post_detector_ppg_qc": "none",
    "ecg_reference": "dataset-provided corrected strict ECG reference",
}


def _select_frozen_methods(raw: pd.DataFrame) -> pd.DataFrame:
    """Select only the two pre-specified methods from common detector output."""
    scipy = raw[(raw["detector"] == "scipy") & (raw["fiducial"] == "peak")].copy()
    scipy["baseline_method"] = SCIPY_COMPARATOR_NAME

    qppg = raw[raw["detector"] == "qppgfast"].copy()
    expected = qppg["device"].map(DEVICE_FIDUCIALS)
    qppg = qppg[qppg["fiducial"] == expected].copy()
    qppg["baseline_method"] = METHOD_NAME

    selected = pd.concat([scipy, qppg], ignore_index=True)
    if selected.empty:
        raise RuntimeError("no frozen baseline rows were generated")
    return selected.sort_values(["baseline_method", "participant", "window_index", "device", "channel"]).reset_index(drop=True)


def _primary_channel_summary(selected: pd.DataFrame) -> pd.DataFrame:
    """Summarize the two production methods at participant/device/channel level."""
    rows: list[dict[str, object]] = []
    for keys, group in selected.groupby(["baseline_method", "participant", "device", "channel"], dropna=False):
        pred = group["ppg_rmssd_ms"].to_numpy(float)
        ref = group["ecg_rmssd_ms"].to_numpy(float)
        sdnn = group["ppg_sdnn_ms"].to_numpy(float)
        valid = np.isfinite(pred) & np.isfinite(ref) & np.isfinite(sdnn)
        pred, ref = pred[valid], ref[valid]
        n_total = int(group.shape[0])
        n_valid = int(pred.size)
        rows.append({
            "baseline_method": keys[0],
            "participant": keys[1],
            "device": keys[2],
            "channel": keys[3],
            "n_total": n_total,
            "n_valid": n_valid,
            "coverage_pct": 100.0 * n_valid / n_total if n_total else np.nan,
            "MAE": float(np.mean(np.abs(pred - ref))) if n_valid else np.nan,
            "R": float(np.corrcoef(pred, ref)[0, 1]) if n_valid >= 3 and np.std(pred) > 0 and np.std(ref) > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["baseline_method", "participant", "device", "channel"])


def _primary_cohort_summary(channel: pd.DataFrame) -> pd.DataFrame:
    """Each participant and each of its six device/channel results have equal weight."""
    participant = (
        channel.groupby(["baseline_method", "participant"], dropna=False)
        .agg(
            channels=("channel", "size"),
            mean_MAE=("MAE", "mean"),
            mean_R=("R", "mean"),
            mean_coverage_pct=("coverage_pct", "mean"),
            min_channel_coverage_pct=("coverage_pct", "min"),
        )
        .reset_index()
    )
    cohort = (
        participant.groupby("baseline_method", dropna=False)
        .agg(
            participants=("participant", "size"),
            mean_participant_MAE=("mean_MAE", "mean"),
            median_participant_MAE=("mean_MAE", "median"),
            mean_participant_R=("mean_R", "mean"),
            median_participant_R=("mean_R", "median"),
            mean_participant_coverage_pct=("mean_coverage_pct", "mean"),
            min_channel_coverage_pct=("min_channel_coverage_pct", "min"),
        )
        .reset_index()
        .sort_values("mean_participant_MAE")
    )
    return participant.sort_values(["baseline_method", "participant"]), cohort


def _acceptance_summary(channel: pd.DataFrame, participant: pd.DataFrame) -> pd.DataFrame:
    qppg = participant[participant["baseline_method"] == METHOD_NAME].set_index("participant")
    scipy = participant[participant["baseline_method"] == SCIPY_COMPARATOR_NAME].set_index("participant")
    joined = qppg.join(scipy, lsuffix="_qppg", rsuffix="_scipy", how="inner")
    coverage_gap = joined["mean_coverage_pct_qppg"] - joined["mean_coverage_pct_scipy"]
    mae_gap = joined["mean_MAE_qppg"] - joined["mean_MAE_scipy"]
    r_gap = joined["mean_R_qppg"] - joined["mean_R_scipy"]
    return pd.DataFrame([{
        "criterion": "qppgfast coverage no worse than SciPy by more than 1 percentage point",
        "result": bool((coverage_gap >= -1.0).all()),
        "value": float(coverage_gap.mean()),
        "detail": "mean participant coverage difference (qppgfast - SciPy), percentage points",
    }, {
        "criterion": "qppgfast mean participant MAE lower than SciPy",
        "result": bool(float(mae_gap.mean()) < 0.0),
        "value": float(mae_gap.mean()),
        "detail": "mean participant MAE difference (qppgfast - SciPy), ms",
    }, {
        "criterion": "qppgfast has lower MAE for a majority of participants",
        "result": bool(int((mae_gap < 0.0).sum()) > int(mae_gap.size / 2)),
        "value": int((mae_gap < 0.0).sum()),
        "detail": f"participants with lower MAE out of {mae_gap.size}",
    }, {
        "criterion": "qppgfast mean participant R higher than SciPy",
        "result": bool(float(r_gap.mean()) > 0.0),
        "value": float(r_gap.mean()),
        "detail": "mean participant R difference (qppgfast - SciPy)",
    }])


def _write_readme(out_dir: Path, cohort: pd.DataFrame, acceptance: pd.DataFrame, interpolation: pd.DataFrame) -> None:
    lines = [
        "# qppgfast_devicewise_v1 端到端复现",
        "",
        "这是 rawslots 的稳定 baseline 入口输出。它不做参数搜索；所有算法设置均由 `qppgfast_devicewise_v1.py` 固定。每个原始窗口同时产出 qppgfast-devicewise 与 SciPy-peak 的结果，便于公平对照。",
        "",
        "## 固定方法",
        "",
        "- `qppgfast_devicewise_v1`：Earring 使用 qppgfast peak；Ring/Watch 使用 qppgfast 原生 foot/onset。",
        "- `scipy_peak_v1`：同一严格输入、同一 bandpass、同一 IBI correction、同一极性选择的 SciPy peak 对照。",
        "- 两者均为 strict interpolation：仅在最大原始 support gap（含窗口首尾边界）不超过 100 ms 时，把原始采样线性插值到 100 Hz 网格；不会做边界外推。",
        "- 不添加 post-detector PPG QC；coverage 的分母是所有输入窗口，分子是 PPG RMSSD、SDNN 与 ECG RMSSD 均有限的窗口。",
        "",
        "## 全队列结果",
        "",
        "| method | 平均参与者 MAE (ms) | 平均参与者 R | 平均参与者 coverage (%) | 最低通道 coverage (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in cohort.iterrows():
        lines.append(
            f"| `{row['baseline_method']}` | {row['mean_participant_MAE']:.2f} | {row['mean_participant_R']:.3f} | "
            f"{row['mean_participant_coverage_pct']:.2f} | {row['min_channel_coverage_pct']:.2f} |"
        )
    lines.extend([
        "",
        "## 冻结验收",
        "",
        "| 验收条件 | 通过 | 数值 | 说明 |",
        "|---|---|---:|---|",
    ])
    for _, row in acceptance.iterrows():
        lines.append(f"| {row['criterion']} | {'是' if row['result'] else '否'} | {row['value']:.3f} | {row['detail']} |")
    lines.extend([
        "",
        "## 无 fiducial 极性敏感性冻结",
        "",
        "已预定义并测试 `waveform_consensus_v1`：把 5 分钟波形固定切为 5 个 60 秒段，只按形态 score（robust skewness + rise/fall derivative asymmetry）投票，至少 4/5 同号才选方向；否则为 ambiguous，不运行 detector。它不使用 ECG、peak、foot、IBI、SQI 或 HRV。独立 holdout 的 MAE 虽低 `1.93 ms`，但 coverage 低 `28.96` 个百分点、R 低 `0.035`，因此冻结决定为不采用，正式极性仍为 `peak_train`。完整证据见 `outputs/rawslots_qppgfast_waveform_consensus_freeze_v1/`。",
        "",
        "## 严格插值量",
        "",
        "`interpolated_grid_ratio_within_strict_input_pct` 的分母只包含已通过 100 ms strict input 的 grid 点；它回答“已接纳窗口中有多少精确 100 Hz grid 点由线性插值补得”。`strict_input_coverage_pct` 的分母是全部原始窗口；它回答“有多少窗口先通过了最大 gap 与边界 gap 均不超过 100 ms 的输入规则”。两者不能混为一个百分比。",
        "",
        "| device | channel | strict input coverage (%) | rawslot-observed within strict input (%) | interpolated within strict input (%) |",
        "|---|---|---:|---:|---:|",
    ])
    for _, row in interpolation.iterrows():
        lines.append(
            f"| {row['device']} | {row['channel']} | {row['strict_input_coverage_pct']:.2f} | "
            f"{row['rawslot_observed_grid_ratio_within_strict_input_pct']:.2f} | "
            f"{row['interpolated_grid_ratio_within_strict_input_pct']:.2f} |"
        )
    lines.extend([
        "",
        "## 文件",
        "",
        "| 文件 | 含义 |",
        "|---|---|",
        "| `frozen_channel_metrics.csv` | 两种正式方法的逐窗口、逐通道指标。 |",
        "| `frozen_channel_summary.csv` | 每位参与者 x 设备 x 通道结果。 |",
        "| `frozen_participant_summary.csv` | 每位参与者在六通道宏平均上的结果。 |",
        "| `frozen_cohort_summary.csv` | 16 位参与者等权的全队列汇总。 |",
        "| `frozen_acceptance_summary.csv` | 预先规定的端到端验收条件。 |",
        "| `strict_interpolation_amount_by_window.csv` | 每窗口/设备/通道的 strict 输入通过状态与 rawslot-observed、插值 grid 点数。 |",
        "| `strict_interpolation_amount_by_channel.csv` | 上述 strict 插值量按设备/通道汇总。 |",
        "| `frozen_config.json` | 不可调的冻结配置和实际运行范围。 |",
    ])
    out_dir.joinpath("README_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_qppgfast_devicewise_v1(dataset_dir: Path, out_dir: Path, participants: tuple[str, ...] | None = None) -> None:
    """Run the frozen qppgfast baseline and SciPy comparator on rawslots."""
    paths = _dataset_npz_files(dataset_dir)
    if not paths:
        raise FileNotFoundError(f"no rawslots .npz files found in {dataset_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if participants is not None:
        requested = set(participants)
        paths = [path for path in paths if path.stem.rsplit("_", 1)[-1].upper() in requested]
        found = {path.stem.rsplit("_", 1)[-1].upper() for path in paths}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"participants not found in dataset: {missing}")

    # Process one participant at a time. The raw NPZ tensors are large enough
    # that retaining every participant's candidate frame would waste several
    # GB on a 16-GB machine. Only the two frozen methods are kept after each
    # participant, so resident memory does not scale with cohort size.
    selected_frames: list[pd.DataFrame] = []
    pipelines = (DetectorPipeline("scipy", (0.7, 3.5)), DetectorPipeline("qppgfast", (0.7, 3.5)))
    for path in paths:
        with np.load(path, allow_pickle=True) as z:
            n_windows = int(np.asarray(z["ecg_rmssd_corrected_ms"]).shape[0])
        raw_participant = _load_channel_metrics_for_path(
            path,
            np.arange(n_windows, dtype=int),
            pipelines,
            input_mode="strict_interp100",
            max_gap_ms=100.0,
            correction_states=(True,),
            polarity_mode="peak_train",
        )
        selected_frames.append(_select_frozen_methods(raw_participant))
        del raw_participant
        gc.collect()

    selected = pd.concat(selected_frames, ignore_index=True)
    interpolation_by_window = _collect_interpolation_amount(dataset_dir, 100.0, participants)
    keys = ["participant", "window_index", "device", "channel"]
    selected = selected.merge(interpolation_by_window, on=keys, how="left", validate="many_to_one")
    if selected["strict_input_pass"].isna().any():
        raise RuntimeError("failed to attach strict interpolation amounts to every baseline row")
    interpolation_by_channel = _interpolation_amount_summary(interpolation_by_window)
    channel = _primary_channel_summary(selected)
    participant, cohort = _primary_cohort_summary(channel)
    acceptance = _acceptance_summary(channel, participant)
    selected.to_csv(out_dir / "frozen_channel_metrics.csv", index=False)
    channel.to_csv(out_dir / "frozen_channel_summary.csv", index=False)
    participant.to_csv(out_dir / "frozen_participant_summary.csv", index=False)
    cohort.to_csv(out_dir / "frozen_cohort_summary.csv", index=False)
    acceptance.to_csv(out_dir / "frozen_acceptance_summary.csv", index=False)
    interpolation_by_window.to_csv(out_dir / "strict_interpolation_amount_by_window.csv", index=False)
    interpolation_by_channel.to_csv(out_dir / "strict_interpolation_amount_by_channel.csv", index=False)
    config = {**FROZEN_CONFIG, "dataset_dir": str(dataset_dir), "participants": list(participants) if participants else "all"}
    (out_dir / "frozen_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    _write_readme(out_dir, cohort, acceptance, interpolation_by_channel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen rawslots qppgfast-devicewise baseline.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--participants", type=str, default=None, help="Optional comma-separated subset; method settings remain frozen.")
    args = parser.parse_args()
    participants = tuple(x.strip().upper() for x in args.participants.split(",") if x.strip()) if args.participants else None
    run_qppgfast_devicewise_v1(args.dataset_dir, args.out_dir, participants)


if __name__ == "__main__":
    main()
