"""Freeze the detector-independent waveform-polarity consensus decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEVICE_FIDUCIALS = {"Earring": "peak", "Ring": "foot", "Watch": "foot"}
PROTOCOL = {
    "name": "waveform_consensus_v1",
    "uses": "bandpassed waveform samples only; no ECG, detector fiducials, IBI, SQI, or HRV metrics",
    "segment_count": 5,
    "segment_duration_sec": 60,
    "shape_score": "robust skewness + derivative rise/fall asymmetry",
    "accept_rule": "at least 4 of 5 nonzero segment score signs agree",
    "ambiguous_handling": "do not invoke detector; PPG PRV remains invalid",
    "device_fiducials": DEVICE_FIDUCIALS,
    "bandpass_hz": [0.7, 3.5],
    "strict_input": "100 Hz; internal and boundary raw support gap <= 100 ms; no extrapolation",
    "ibi_correction_threshold": 0.20,
}


def _devicewise(df: pd.DataFrame) -> pd.DataFrame:
    qppg = df[df["detector"].eq("qppgfast")].copy()
    required = qppg["device"].map(DEVICE_FIDUCIALS)
    return qppg[qppg["fiducial"].eq(required)].copy()


def _stats(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(["device", "channel"], dropna=False):
        pred = group["ppg_rmssd_ms"].to_numpy(float)
        ref = group["ecg_rmssd_ms"].to_numpy(float)
        sdnn = group["ppg_sdnn_ms"].to_numpy(float)
        valid = np.isfinite(pred) & np.isfinite(ref) & np.isfinite(sdnn)
        p, e = pred[valid], ref[valid]
        n_total, n_valid = int(group.shape[0]), int(p.size)
        rows.append({
            "device": keys[0], "channel": keys[1], "n_total": n_total, "n_valid": n_valid,
            "coverage_pct": 100.0 * n_valid / n_total if n_total else np.nan,
            "MAE_ms": float(np.mean(np.abs(p - e))) if n_valid else np.nan,
            "R": float(np.corrcoef(p, e)[0, 1]) if n_valid >= 3 and np.std(p) > 0 and np.std(e) > 0 else np.nan,
        })
    channel = pd.DataFrame(rows)
    macro = {
        "channels": int(channel.shape[0]),
        "n_total": int(channel["n_total"].sum()),
        "n_valid": int(channel["n_valid"].sum()),
        "mean_coverage_pct": float(channel["coverage_pct"].mean()),
        "min_coverage_pct": float(channel["coverage_pct"].min()),
        "mean_MAE_ms": float(channel["MAE_ms"].mean()),
        "mean_R": float(channel["R"].mean()),
    }
    return channel, macro


def _polarity_audit(df: pd.DataFrame) -> pd.DataFrame:
    # Polarity is shared by peak/foot rows. Retain one row per input window.
    one = df[df["fiducial"].eq("peak")].copy()
    rows: list[dict[str, object]] = []
    for keys, group in one.groupby(["device", "channel"], dropna=False):
        n = int(group.shape[0])
        invalid = int(group["polarity"].eq("invalid").sum())
        ambiguous = int(group["polarity"].eq("ambiguous").sum())
        accepted = int(group["polarity"].isin(["positive", "negative"]).sum())
        strict = n - invalid
        rows.append({
            "device": keys[0], "channel": keys[1], "n_total": n,
            "n_strict_input": strict, "n_accepted_orientation": accepted,
            "n_ambiguous": ambiguous,
            "strict_input_coverage_pct": 100.0 * strict / n if n else np.nan,
            "accepted_among_strict_input_pct": 100.0 * accepted / strict if strict else np.nan,
            "ambiguous_among_strict_input_pct": 100.0 * ambiguous / strict if strict else np.nan,
            "n_positive": int(group["polarity"].eq("positive").sum()),
            "n_negative": int(group["polarity"].eq("negative").sum()),
        })
    order_device = {"Earring": 0, "Ring": 1, "Watch": 2}
    order_channel = {"ppg_green": 0, "ppg_ir": 1}
    return pd.DataFrame(rows).sort_values(
        ["device", "channel"], key=lambda s: s.map(order_device if s.name == "device" else order_channel)
    ).reset_index(drop=True)


def _protocol_aggregate(audit: pd.DataFrame) -> dict[str, float | int]:
    strict = int(audit["n_strict_input"].sum())
    accepted = int(audit["n_accepted_orientation"].sum())
    ambiguous = int(audit["n_ambiguous"].sum())
    return {
        "n_total": int(audit["n_total"].sum()), "n_strict_input": strict,
        "n_accepted_orientation": accepted, "n_ambiguous": ambiguous,
        "accepted_among_strict_input_pct": 100.0 * accepted / strict if strict else np.nan,
        "ambiguous_among_strict_input_pct": 100.0 * ambiguous / strict if strict else np.nan,
    }


def _write_report(out_dir: Path, table: pd.DataFrame, audits: dict[str, pd.DataFrame], acceptance: pd.DataFrame) -> None:
    lines = [
        "# 无 fiducial 波形方向规则：预定义流程与冻结决定",
        "",
        "## 固定协议",
        "",
        "- 在 strict interpolation、`0.7-3.5 Hz` bandpass 后，把每个 5 分钟波形等分为 5 个 60 秒段。",
        "- 每段仅计算 `robust skewness + derivative rise/fall asymmetry`。不读取 ECG、detector、peak、foot、IBI、SQI 或 HRV。",
        "- 至少 4/5 段形态 score 同号才选该极性；其余为 `ambiguous`，不运行 detector。",
        "- Earring peak、Ring/Watch foot/onset 的 devicewise fiducial 规则已在本协议前固定；本报告不重新选择它。",
        "",
        "## 两组结果",
        "",
        "| split | polarity method | coverage (%) | min coverage (%) | MAE (ms) | R |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in table.sort_values(["split", "mean_MAE_ms"]).iterrows():
        lines.append(
            f"| {row['split']} | `{row['method']}` | {row['mean_coverage_pct']:.2f} | {row['min_coverage_pct']:.2f} | "
            f"{row['mean_MAE_ms']:.2f} | {row['mean_R']:.3f} |"
        )
    for split, audit in audits.items():
        agg = _protocol_aggregate(audit)
        lines.extend([
            "",
            f"## {split}：方向一致性",
            "",
            f"strict 输入窗口中，`{agg['accepted_among_strict_input_pct']:.2f}%` 通过 4/5 一致性；"
            f"`{agg['ambiguous_among_strict_input_pct']:.2f}%` 被标为 ambiguous。",
            "",
            "| device | channel | strict input (%) | accepted among strict input (%) | ambiguous among strict input (%) |",
            "|---|---|---:|---:|---:|",
        ])
        for _, row in audit.iterrows():
            lines.append(
                f"| {row['device']} | {row['channel']} | {row['strict_input_coverage_pct']:.2f} | "
                f"{row['accepted_among_strict_input_pct']:.2f} | {row['ambiguous_among_strict_input_pct']:.2f} |"
            )
    lines.extend([
        "",
        "## 独立 holdout 验收与冻结",
        "",
        "| 条件 | 要求 | 结果 | 通过 |",
        "|---|---|---:|---|",
    ])
    for _, row in acceptance.iterrows():
        lines.append(f"| {row['criterion']} | {row['requirement']} | {row['value']:.3f} | {'是' if row['passed'] else '否'} |")
    lines.extend([
        "",
        "**冻结决定：不采用 `waveform_consensus_v1` 替代正式 `qppgfast_devicewise_v1` 的 `peak_train` 极性规则。**"
        "原因是预定义的 4/5 一致性门槛在独立 holdout 上显著降低 coverage；这不是通过事后放松阈值得到修复的对象。"
        "该协议与本次拒绝决定被冻结为可复现 sensitivity result。",
    ])
    out_dir.joinpath("FREEZE_REPORT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the frozen waveform-consensus polarity decision.")
    parser.add_argument("--peak-train-train", type=Path, required=True)
    parser.add_argument("--consensus-train", type=Path, required=True)
    parser.add_argument("--peak-train-holdout", type=Path, required=True)
    parser.add_argument("--consensus-holdout", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        ("train12", "peak_train"): args.peak_train_train,
        ("train12", "waveform_consensus_v1"): args.consensus_train,
        ("holdout4", "peak_train"): args.peak_train_holdout,
        ("holdout4", "waveform_consensus_v1"): args.consensus_holdout,
    }
    summary_rows: list[dict[str, object]] = []
    audits: dict[str, pd.DataFrame] = {}
    for (split, method), path in inputs.items():
        raw = pd.read_csv(path)
        selected = _devicewise(raw)
        channel, macro = _stats(selected)
        channel.assign(split=split, method=method).to_csv(args.out_dir / f"{split}_{method}_by_channel.csv", index=False)
        summary_rows.append({"split": split, "method": method, **macro})
        if method == "waveform_consensus_v1":
            audit = _polarity_audit(raw)
            audit.to_csv(args.out_dir / f"{split}_waveform_consensus_audit_by_channel.csv", index=False)
            audits[split] = audit
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "devicewise_summary.csv", index=False)
    holdout_peak = summary[(summary["split"] == "holdout4") & (summary["method"] == "peak_train")].iloc[0]
    holdout_consensus = summary[(summary["split"] == "holdout4") & (summary["method"] == "waveform_consensus_v1")].iloc[0]
    holdout_protocol = _protocol_aggregate(audits["holdout4"])
    acceptance = pd.DataFrame([
        {"criterion": "方向可靠性", "requirement": "accepted among strict input >= 80%", "value": holdout_protocol["accepted_among_strict_input_pct"], "passed": holdout_protocol["accepted_among_strict_input_pct"] >= 80.0},
        {"criterion": "coverage 保留", "requirement": "consensus - peak_train >= -1 percentage point", "value": holdout_consensus["mean_coverage_pct"] - holdout_peak["mean_coverage_pct"], "passed": holdout_consensus["mean_coverage_pct"] - holdout_peak["mean_coverage_pct"] >= -1.0},
        {"criterion": "MAE 不退步", "requirement": "consensus - peak_train <= 0 ms", "value": holdout_consensus["mean_MAE_ms"] - holdout_peak["mean_MAE_ms"], "passed": holdout_consensus["mean_MAE_ms"] - holdout_peak["mean_MAE_ms"] <= 0.0},
        {"criterion": "R 不退步", "requirement": "consensus - peak_train >= 0", "value": holdout_consensus["mean_R"] - holdout_peak["mean_R"], "passed": holdout_consensus["mean_R"] - holdout_peak["mean_R"] >= 0.0},
    ])
    acceptance.to_csv(args.out_dir / "holdout_acceptance.csv", index=False)
    config = {**PROTOCOL, "decision": "not_adopted", "retained_official_polarity_mode": "peak_train"}
    (args.out_dir / "frozen_waveform_consensus_protocol.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    _write_report(args.out_dir, summary, audits, acceptance)


if __name__ == "__main__":
    main()
