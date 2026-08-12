"""
Evaluate current v1.2 frozen baseline on training_stride30 with no QC gate.

This is a no-QC ablation: keep the frozen device x channel peak-method choices,
but do not apply SQI, valid-IBI, correction-ratio, IBI-CV, correlation, or RMSSD
range quality gates. Coverage therefore measures the detector/HRV-computation
upper bound for the frozen methods.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_BASE_DIR = _THIS_DIR.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import config  # noqa: E402


METRICS = ("RMSSD", "SDNN")
POLICY = "v1_2_frozen_peak_methods_no_qc_training_stride30"


def _fmt(value: object, digits: int = 2) -> str:
    try:
        f = float(value)
    except Exception:
        return ""
    if not np.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


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


def _load_noqc_predictions(metrics: pd.DataFrame, choices: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, row in choices.iterrows():
        mask = (
            (metrics["role"].astype(str) == "training_stride30")
            & (metrics["device"].astype(str) == str(row["device"]))
            & (metrics["channel"].astype(str) == str(row["channel"]))
            & (metrics["peak_method"].astype(str) == str(row["peak_method"]))
        )
        piece = metrics[mask].copy()
        piece["frozen_gate_ignored"] = str(row["gate"])
        piece["policy"] = POLICY
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _summarize(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    participant_rows: list[dict[str, object]] = []

    for (device, channel), group in pred.groupby(["device", "channel"], dropna=False):
        n_total = int(group[["participant", "window_index"]].drop_duplicates().shape[0])
        base = group.iloc[0]
        for metric in METRICS:
            stats = _agreement(
                group[f"ppg_{metric.lower()}_ms"].to_numpy(float),
                group[f"ecg_{metric.lower()}_ms"].to_numpy(float),
            )
            rows.append(
                {
                    "dataset": base["dataset"],
                    "role": "training_stride30",
                    "device": device,
                    "channel": channel,
                    "peak_method": base["peak_method"],
                    "ignored_gate": base["frozen_gate_ignored"],
                    "policy": POLICY,
                    "hrv_metric": metric,
                    "n_total": n_total,
                    **stats,
                    "coverage_pct": float(100.0 * stats["n_valid"] / n_total) if n_total else np.nan,
                }
            )

    for (participant, device, channel), group in pred.groupby(["participant", "device", "channel"], dropna=False):
        n_total = int(group["window_index"].nunique())
        base = group.iloc[0]
        for metric in METRICS:
            stats = _agreement(
                group[f"ppg_{metric.lower()}_ms"].to_numpy(float),
                group[f"ecg_{metric.lower()}_ms"].to_numpy(float),
            )
            participant_rows.append(
                {
                    "dataset": base["dataset"],
                    "role": "training_stride30",
                    "participant": participant,
                    "device": device,
                    "channel": channel,
                    "peak_method": base["peak_method"],
                    "ignored_gate": base["frozen_gate_ignored"],
                    "policy": POLICY,
                    "hrv_metric": metric,
                    "n_total": n_total,
                    **stats,
                    "coverage_pct": float(100.0 * stats["n_valid"] / n_total) if n_total else np.nan,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(participant_rows)


def _join_rmssd_sdnn(summary: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rmssd = summary[summary["hrv_metric"] == "RMSSD"].copy()
    sdnn = summary[summary["hrv_metric"] == "SDNN"][["device", "channel", "MAE", "R"]].rename(
        columns={"MAE": f"{prefix}_SDNN_MAE", "R": f"{prefix}_SDNN_R"}
    )
    rename_cols = {
        "n_valid": f"{prefix}_n_valid",
        "coverage_pct": f"{prefix}_coverage_pct",
        "MAE": f"{prefix}_RMSSD_MAE",
        "RMSE": f"{prefix}_RMSSD_RMSE",
        "R": f"{prefix}_RMSSD_R",
        "bias": f"{prefix}_RMSSD_bias",
    }
    rmssd = rmssd.rename(columns=rename_cols)
    return rmssd.merge(sdnn, on=["device", "channel"], how="left")


def _write_readme(
    out_dir: Path,
    noqc_summary: pd.DataFrame,
    participant_summary: pd.DataFrame,
    qc_eval: pd.DataFrame,
    metrics_csv: Path,
    choices_csv: Path,
) -> None:
    noqc = _join_rmssd_sdnn(noqc_summary, "noQC")
    qc = _join_rmssd_sdnn(qc_eval[qc_eval["role"] == "training_stride30"].copy(), "QC")
    compare = noqc.merge(
        qc[["device", "channel", "QC_n_valid", "QC_coverage_pct", "QC_RMSSD_MAE", "QC_RMSSD_R", "QC_SDNN_MAE", "QC_SDNN_R"]],
        on=["device", "channel"],
        how="left",
    )
    compare["coverage_gain_pctpt"] = compare["noQC_coverage_pct"] - compare["QC_coverage_pct"]
    compare["rmssd_mae_change_ms"] = compare["noQC_RMSSD_MAE"] - compare["QC_RMSSD_MAE"]
    compare["rmssd_r_change"] = compare["noQC_RMSSD_R"] - compare["QC_RMSSD_R"]
    compare = compare.sort_values("noQC_RMSSD_MAE", na_position="last").reset_index(drop=True)
    compare.to_csv(out_dir / "v1_2_noQC_vs_QC_training_stride30_comparison.csv", index=False)

    rmssd_part = participant_summary[participant_summary["hrv_metric"] == "RMSSD"].copy()
    variability = (
        rmssd_part.groupby(["device", "channel"], dropna=False)
        .agg(
            participant_R_median=("R", "median"),
            participant_R_q25=("R", lambda x: x.quantile(0.25)),
            participant_R_q75=("R", lambda x: x.quantile(0.75)),
            participant_coverage_median=("coverage_pct", "median"),
            participant_coverage_min=("coverage_pct", "min"),
            participant_coverage_max=("coverage_pct", "max"),
        )
        .reset_index()
        .sort_values(["device", "channel"])
    )

    dataset_names = sorted(str(x) for x in noqc_summary["dataset"].dropna().unique())
    lines = [
        "# v1.2 noQC Training-Stride30 Ablation",
        "",
        "本目录汇总当前 v1.2 双通道 frozen baseline 在 `training_stride30` 上关闭所有 QC gate 后的结果。",
        "",
        "## Dataset / Inputs",
        "",
        f"- Role: `training_stride30`",
        f"- Dataset: `{dataset_names[0] if dataset_names else ''}`",
        f"- Channel metrics: `{metrics_csv}`",
        f"- Frozen choices: `{choices_csv}`",
        "- Baseline: keep frozen `device x channel` peak methods from v1.2; report both green and IR.",
        "- noQC definition: do not apply SQI, valid-IBI-ratio, IBI-correction-ratio, IBI-CV, correlation, or RMSSD-range gates.",
        "- Remaining validity requirement: PPG HRV and ECG HRV must both be finite, otherwise MAE/R cannot be computed.",
        "",
        "## Current QC Baseline",
        "",
        "当前 v1.2 QC baseline 对每个窗口应用 frozen `device x channel` gate。QC 包括：",
        "",
        "- `ppg_valid_sample_ratio >= 0.90`",
        "- `ppg_sqi >= frozen min_sqi`，当前 frozen 值按 `device x channel` 为 `0.30` 或 `0.50`",
        "- `ppg_valid_ibi_ratio >= frozen min_valid_ibi`，当前 frozen 值按 `device x channel` 为 `0.70 / 0.80 / 0.85 / 0.90`",
        "- `ppg_ibi_correction_ratio` 必须有限，且 `<= frozen max_correction`，当前 frozen 值按 `device x channel` 为 `0.20 / 0.30 / 0.35`",
        "- `ppg_ibi_cv` 必须有限，且 `<= frozen max_ibi_cv`，当前 frozen 值按 `device x channel` 为 `0.30 / 0.35`",
        "- `ppg_rmssd_ms` 必须有限，且 `<= 200 ms`",
        "- `ppg_sdnn_ms` 必须有限",
        "",
        "当前 frozen 参数展开：",
        "",
        "- Earring green: `min_sqi=0.30`, `min_valid_ibi=0.85`, `max_correction=0.20`, `max_ibi_cv=0.35`, `max_rmssd=200 ms`",
        "- Earring IR: `min_sqi=0.50`, `min_valid_ibi=0.90`, `max_correction=0.20`, `max_ibi_cv=0.30`, `max_rmssd=200 ms`",
        "- Ring green: `min_sqi=0.30`, `min_valid_ibi=0.80`, `max_correction=0.20`, `max_ibi_cv=0.30`, `max_rmssd=200 ms`",
        "- Ring IR: `min_sqi=0.50`, `min_valid_ibi=0.90`, `max_correction=0.30`, `max_ibi_cv=0.30`, `max_rmssd=200 ms`",
        "- Watch green: `min_sqi=0.30`, `min_valid_ibi=0.90`, `max_correction=0.30`, `max_ibi_cv=0.30`, `max_rmssd=200 ms`",
        "- Watch IR: `min_sqi=0.50`, `min_valid_ibi=0.70`, `max_correction=0.35`, `max_ibi_cv=0.35`, `max_rmssd=200 ms`",
        "",
        "各 `device x channel` frozen QC gate：",
        "",
        "| Device | Channel | Frozen gate | min SQI | min valid IBI | max correction | max IBI CV | max RMSSD |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        "| `Earring` | `ppg_green` | `gate_sqi030_ibi085_corr020_cv035_rmssd200` | 0.30 | 0.85 | 0.20 | 0.35 | 200 ms |",
        "| `Earring` | `ppg_ir` | `gate_sqi050_ibi090_corr020_cv030_rmssd200` | 0.50 | 0.90 | 0.20 | 0.30 | 200 ms |",
        "| `Ring` | `ppg_green` | `gate_sqi030_ibi080_corr020_cv030_rmssd200` | 0.30 | 0.80 | 0.20 | 0.30 | 200 ms |",
        "| `Ring` | `ppg_ir` | `gate_sqi050_ibi090_corr030_cv030_rmssd200` | 0.50 | 0.90 | 0.30 | 0.30 | 200 ms |",
        "| `Watch` | `ppg_green` | `gate_sqi030_ibi090_corr030_cv030_rmssd200` | 0.30 | 0.90 | 0.30 | 0.30 | 200 ms |",
        "| `Watch` | `ppg_ir` | `gate_sqi050_ibi070_corr035_cv035_rmssd200` | 0.50 | 0.70 | 0.35 | 0.35 | 200 ms |",
        "",
        "## Coverage Definition",
        "",
        "noQC ablation 的 coverage 口径：",
        "",
        "```text",
        "noQC coverage = n_finite_HRV / n_total",
        "",
        "n_finite_HRV = PPG HRV 和 ECG HRV 都是有限值的窗口数",
        "```",
        "",
        "noQC 不应用 SQI、valid IBI ratio、IBI correction ratio、IBI CV、correlation 或 RMSSD range gate；它衡量的是 frozen detector/bandpass 能产出有限 HRV 的上限。因此 noQC coverage 可以接近 100%，但如果 detector 找不到足够 peaks、IBI 序列不足、PPG HRV 是 NaN/inf，coverage 仍然不会等于 100%。",
        "",
        "这也是为什么不同 detector 会改变 coverage：不同 detector 会产生不同 peak 序列，进而改变 IBI 数量、IBI 稳定性、SQI、correction ratio 和最终 HRV 是否为有限值。coverage 上升不一定代表 HRV agreement 变好；它必须和 MAE、R、bias 一起解释。",
        "",
        "## noQC vs Current QC Baseline",
        "",
        "| Rank | Device | Channel | noQC valid / total | noQC coverage | noQC RMSSD MAE | noQC RMSSD R | QC coverage | QC RMSSD MAE | QC RMSSD R | Coverage gain | MAE change | R change | noQC SDNN MAE | noQC SDNN R |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, (_, row) in enumerate(compare.iterrows(), start=1):
        lines.append(
            f"| {i} | `{row['device']}` | `{row['channel']}` | "
            f"{int(row['noQC_n_valid'])} / {int(row['n_total'])} | "
            f"{_fmt(row['noQC_coverage_pct'])}% | {_fmt(row['noQC_RMSSD_MAE'])} ms | {_fmt(row['noQC_RMSSD_R'], 3)} | "
            f"{_fmt(row['QC_coverage_pct'])}% | {_fmt(row['QC_RMSSD_MAE'])} ms | {_fmt(row['QC_RMSSD_R'], 3)} | "
            f"{_fmt(row['coverage_gain_pctpt'])} pp | {_fmt(row['rmssd_mae_change_ms'])} ms | {_fmt(row['rmssd_r_change'], 3)} | "
            f"{_fmt(row['noQC_SDNN_MAE'])} ms | {_fmt(row['noQC_SDNN_R'], 3)} |"
        )

    lines.extend([
        "",
        "## Participant Variability",
        "",
        "| Device | Channel | Participant RMSSD R median [IQR] | Participant coverage median [min, max] |",
        "|---|---|---:|---:|",
    ])
    for _, row in variability.iterrows():
        lines.append(
            f"| `{row['device']}` | `{row['channel']}` | "
            f"{_fmt(row['participant_R_median'], 3)} [{_fmt(row['participant_R_q25'], 3)}, {_fmt(row['participant_R_q75'], 3)}] | "
            f"{_fmt(row['participant_coverage_median'])}% [{_fmt(row['participant_coverage_min'])}, {_fmt(row['participant_coverage_max'])}] |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- noQC coverage 是 frozen detector/bandpass 能产出有限 HRV 的上限，不等于真实可用质量。",
        "- 如果 noQC coverage 接近 100% 但 MAE/R 变差，说明原 QC gate 主要是在移除错误 peak/IBI 窗口，而不是 detector 算不出来。",
        "- 如果 noQC coverage 仍明显低于 100%，说明即使不做 QC，部分窗口也无法产生有限 HRV prediction。",
        "",
        "## Output Files",
        "",
        "- `v1_2_noQC_training_stride30_eval.csv`",
        "- `v1_2_noQC_training_stride30_participant_eval.csv`",
        "- `v1_2_noQC_vs_QC_training_stride30_comparison.csv`",
        "- `summary.json`",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = _THIS_DIR
    metrics_csv = (
        config.HEURISTIC_RESULT_ROOT
        / "all_participants_devicewise_baseline_v1_1_plus_neurokit"
        / "v1_1_plus_neurokit_channel_metrics.csv"
    )
    choices_csv = (
        config.HEURISTIC_RESULT_ROOT
        / "formal_v1_2_report_both_channels"
        / "v1_2_both_channels_frozen_choices.csv"
    )
    qc_eval_csv = (
        config.HEURISTIC_RESULT_ROOT
        / "formal_v1_2_report_both_channels"
        / "v1_2_both_channels_frozen_eval.csv"
    )

    metrics = pd.read_csv(metrics_csv, low_memory=False)
    choices = pd.read_csv(choices_csv)
    qc_eval = pd.read_csv(qc_eval_csv)
    pred = _load_noqc_predictions(metrics, choices)
    summary, participant = _summarize(pred)

    summary.to_csv(out_dir / "v1_2_noQC_training_stride30_eval.csv", index=False)
    participant.to_csv(out_dir / "v1_2_noQC_training_stride30_participant_eval.csv", index=False)
    _write_readme(out_dir, summary, participant, qc_eval, metrics_csv, choices_csv)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "policy": POLICY,
                "role": "training_stride30",
                "channel_metrics_csv": str(metrics_csv.resolve()),
                "frozen_choices_csv": str(choices_csv.resolve()),
                "qc_eval_csv": str(qc_eval_csv.resolve()),
                "quality_checks_applied": False,
                "reports_green_and_ir": True,
                "validity_requirement": "finite PPG and ECG HRV metric values",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[saved] {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
