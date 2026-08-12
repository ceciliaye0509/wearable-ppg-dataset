"""Summarize fixed detector candidates by held-out participant.

The candidates are frozen before this script reads the full-cohort metrics.
It therefore reports participant-level robustness, not a new tuning round.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEVICE_FIDUCIALS = {"Earring": "peak", "Ring": "foot", "Watch": "foot"}


def _stats(group: pd.DataFrame) -> dict[str, float | int]:
    pred = group["ppg_rmssd_ms"].to_numpy(float)
    ref = group["ecg_rmssd_ms"].to_numpy(float)
    valid = np.isfinite(pred) & np.isfinite(ref) & np.isfinite(group["ppg_sdnn_ms"].to_numpy(float))
    pred, ref = pred[valid], ref[valid]
    if pred.size == 0:
        return {"n_total": int(group.shape[0]), "n_valid": 0, "coverage_pct": 0.0, "MAE": np.nan, "R": np.nan}
    return {
        "n_total": int(group.shape[0]),
        "n_valid": int(pred.size),
        "coverage_pct": float(100.0 * pred.size / group.shape[0]),
        "MAE": float(np.mean(np.abs(pred - ref))),
        "R": float(np.corrcoef(pred, ref)[0, 1]) if pred.size >= 3 and np.std(pred) > 0 and np.std(ref) > 0 else np.nan,
    }


def _candidate_rows(metrics: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if candidate == "scipy_peak":
        return metrics[(metrics["detector"] == "scipy") & (metrics["fiducial"] == "peak")]
    if candidate == "qppgfast_peak":
        return metrics[(metrics["detector"] == "qppgfast") & (metrics["fiducial"] == "peak")]
    if candidate == "qppgfast_foot":
        return metrics[(metrics["detector"] == "qppgfast") & (metrics["fiducial"] == "foot")]
    if candidate == "qppgfast_devicewise":
        qppg = metrics[metrics["detector"] == "qppgfast"].copy()
        wanted = qppg["device"].map(DEVICE_FIDUCIALS)
        return qppg[qppg["fiducial"] == wanted]
    raise KeyError(candidate)


def _channel_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for candidate in ("scipy_peak", "qppgfast_peak", "qppgfast_foot", "qppgfast_devicewise"):
        selected = _candidate_rows(metrics, candidate)
        rows: list[dict[str, object]] = []
        for keys, group in selected.groupby(["participant", "device", "channel"], dropna=False):
            rows.append({"candidate": candidate, "participant": keys[0], "device": keys[1], "channel": keys[2], **_stats(group)})
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def _participant_summary(channel: pd.DataFrame) -> pd.DataFrame:
    return (
        channel.groupby(["candidate", "participant"], dropna=False)
        .agg(
            channels=("channel", "size"),
            n_total=("n_total", "sum"),
            n_valid=("n_valid", "sum"),
            mean_MAE=("MAE", "mean"),
            mean_R=("R", "mean"),
            mean_coverage_pct=("coverage_pct", "mean"),
            min_coverage_pct=("coverage_pct", "min"),
        )
        .reset_index()
        .sort_values(["candidate", "participant"])
    )


def _cohort_summary(participant: pd.DataFrame) -> pd.DataFrame:
    return (
        participant.groupby("candidate", dropna=False)
        .agg(
            participants=("participant", "size"),
            mean_participant_MAE=("mean_MAE", "mean"),
            median_participant_MAE=("mean_MAE", "median"),
            mean_participant_R=("mean_R", "mean"),
            median_participant_R=("mean_R", "median"),
            mean_participant_coverage_pct=("mean_coverage_pct", "mean"),
            min_channel_coverage_pct=("min_coverage_pct", "min"),
        )
        .reset_index()
        .sort_values("mean_participant_MAE")
    )


def _paired(participant: pd.DataFrame) -> pd.DataFrame:
    baseline = participant[participant["candidate"] == "scipy_peak"].set_index("participant")
    rows: list[dict[str, object]] = []
    for candidate in ("qppgfast_peak", "qppgfast_foot", "qppgfast_devicewise"):
        other = participant[participant["candidate"] == candidate].set_index("participant")
        joined = other.join(baseline, lsuffix="_candidate", rsuffix="_scipy", how="inner")
        for participant_id, row in joined.iterrows():
            rows.append({
                "candidate": candidate,
                "participant": participant_id,
                "candidate_MAE": row["mean_MAE_candidate"],
                "scipy_MAE": row["mean_MAE_scipy"],
                "candidate_minus_scipy_MAE": row["mean_MAE_candidate"] - row["mean_MAE_scipy"],
                "candidate_R": row["mean_R_candidate"],
                "scipy_R": row["mean_R_scipy"],
                "candidate_minus_scipy_R": row["mean_R_candidate"] - row["mean_R_scipy"],
                "candidate_coverage_pct": row["mean_coverage_pct_candidate"],
                "scipy_coverage_pct": row["mean_coverage_pct_scipy"],
            })
    return pd.DataFrame(rows).sort_values(["candidate", "participant"])


def _common_window_vs_scipy(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every qppgfast candidate with SciPy on exactly shared valid windows."""
    keys = ["participant", "window_index", "device", "channel"]
    scipy = _candidate_rows(metrics, "scipy_peak").copy()
    scipy = scipy[keys + ["ppg_rmssd_ms", "ppg_sdnn_ms", "ecg_rmssd_ms"]].rename(
        columns={"ppg_rmssd_ms": "scipy_rmssd", "ppg_sdnn_ms": "scipy_sdnn", "ecg_rmssd_ms": "ecg_rmssd"}
    )
    channel_rows: list[dict[str, object]] = []
    for candidate in ("qppgfast_peak", "qppgfast_foot", "qppgfast_devicewise"):
        selected = _candidate_rows(metrics, candidate).copy()
        selected = selected[keys + ["ppg_rmssd_ms", "ppg_sdnn_ms"]].rename(
            columns={"ppg_rmssd_ms": "candidate_rmssd", "ppg_sdnn_ms": "candidate_sdnn"}
        )
        joined = selected.merge(scipy, on=keys, how="inner", validate="one_to_one")
        valid = (
            np.isfinite(joined["candidate_rmssd"].to_numpy(float))
            & np.isfinite(joined["candidate_sdnn"].to_numpy(float))
            & np.isfinite(joined["scipy_rmssd"].to_numpy(float))
            & np.isfinite(joined["scipy_sdnn"].to_numpy(float))
            & np.isfinite(joined["ecg_rmssd"].to_numpy(float))
        )
        joined["common_valid"] = valid
        for group_keys, group in joined.groupby(["participant", "device", "channel"], dropna=False):
            common = group[group["common_valid"]]
            total = int(group.shape[0])
            if common.empty:
                candidate_mae = scipy_mae = candidate_r = scipy_r = np.nan
            else:
                ref = common["ecg_rmssd"].to_numpy(float)
                candidate_pred = common["candidate_rmssd"].to_numpy(float)
                scipy_pred = common["scipy_rmssd"].to_numpy(float)
                candidate_mae = float(np.mean(np.abs(candidate_pred - ref)))
                scipy_mae = float(np.mean(np.abs(scipy_pred - ref)))
                candidate_r = float(np.corrcoef(candidate_pred, ref)[0, 1]) if ref.size >= 3 and np.std(candidate_pred) > 0 and np.std(ref) > 0 else np.nan
                scipy_r = float(np.corrcoef(scipy_pred, ref)[0, 1]) if ref.size >= 3 and np.std(scipy_pred) > 0 and np.std(ref) > 0 else np.nan
            channel_rows.append({
                "candidate": candidate,
                "participant": group_keys[0],
                "device": group_keys[1],
                "channel": group_keys[2],
                "n_total": total,
                "n_common_valid": int(common.shape[0]),
                "common_coverage_pct": float(100.0 * common.shape[0] / total) if total else np.nan,
                "candidate_MAE": candidate_mae,
                "scipy_MAE": scipy_mae,
                "candidate_minus_scipy_MAE": candidate_mae - scipy_mae,
                "candidate_R": candidate_r,
                "scipy_R": scipy_r,
                "candidate_minus_scipy_R": candidate_r - scipy_r,
            })
    channel = pd.DataFrame(channel_rows)
    participant = (
        channel.groupby(["candidate", "participant"], dropna=False)
        .agg(
            channels=("channel", "size"),
            mean_common_coverage_pct=("common_coverage_pct", "mean"),
            candidate_MAE=("candidate_MAE", "mean"),
            scipy_MAE=("scipy_MAE", "mean"),
            candidate_minus_scipy_MAE=("candidate_minus_scipy_MAE", "mean"),
            candidate_R=("candidate_R", "mean"),
            scipy_R=("scipy_R", "mean"),
            candidate_minus_scipy_R=("candidate_minus_scipy_R", "mean"),
        )
        .reset_index()
        .sort_values(["candidate", "participant"])
    )
    return channel, participant


def _leave_one_participant_selection(participant: pd.DataFrame) -> pd.DataFrame:
    """Select among pre-fixed candidates using the other 15 participants only."""
    candidates = ("scipy_peak", "qppgfast_peak", "qppgfast_foot", "qppgfast_devicewise")
    rows: list[dict[str, object]] = []
    for held_out in sorted(participant["participant"].unique()):
        train = participant[participant["participant"] != held_out]
        ranking = (
            train[train["candidate"].isin(candidates)]
            .groupby("candidate", dropna=False)
            .agg(train_mean_MAE=("mean_MAE", "mean"), train_mean_R=("mean_R", "mean"), train_mean_coverage_pct=("mean_coverage_pct", "mean"))
            .reset_index()
            .sort_values(["train_mean_MAE", "train_mean_R", "train_mean_coverage_pct"], ascending=[True, False, False])
        )
        selected = ranking.iloc[0]
        test = participant[(participant["participant"] == held_out) & (participant["candidate"] == selected["candidate"])].iloc[0]
        scipy = participant[(participant["participant"] == held_out) & (participant["candidate"] == "scipy_peak")].iloc[0]
        rows.append({
            "held_out_participant": held_out,
            "selected_candidate": selected["candidate"],
            "train_participants": int(train["participant"].nunique()),
            "train_mean_MAE": selected["train_mean_MAE"],
            "train_mean_R": selected["train_mean_R"],
            "test_selected_MAE": test["mean_MAE"],
            "test_scipy_MAE": scipy["mean_MAE"],
            "test_selected_minus_scipy_MAE": test["mean_MAE"] - scipy["mean_MAE"],
            "test_selected_R": test["mean_R"],
            "test_scipy_R": scipy["mean_R"],
            "test_selected_minus_scipy_R": test["mean_R"] - scipy["mean_R"],
            "test_selected_coverage_pct": test["mean_coverage_pct"],
            "test_scipy_coverage_pct": scipy["mean_coverage_pct"],
        })
    return pd.DataFrame(rows).sort_values("held_out_participant")


def _write_report(
    out_dir: Path,
    cohort: pd.DataFrame,
    paired: pd.DataFrame,
    common_participant: pd.DataFrame,
    lopo_selection: pd.DataFrame,
) -> None:
    lines = [
        "# 固定候选的参与者级稳健性与留一选择验证",
        "",
        "## 解释",
        "",
        "这不是新一轮调参：四条候选在读取本次全体窗口结果前已固定。每位参与者先在 3 设备 x 2 通道上算 MAE、R、coverage 的宏平均，再在参与者之间汇总，因此每人权重相同。报告同时提供全 16 人稳健性汇总，以及在四条固定候选之间执行的参与者级留一选择。",
        "",
        "候选是：SciPy-peak；qppgfast 全 peak；qppgfast 全 foot；qppgfast 设备化（Earring peak，Ring/Watch foot）。所有候选均使用 strict interpolation（100 Hz、最大原始 support gap 100 ms、无边界外推）、common bandpass `0.7-3.5 Hz`、IBI correction `0.20`，且不加 post-detector PPG QC。ECG reference 为数据集原有 strict ECG reference。",
        "",
        "注意：qppgfast 设备化规则曾用 P1/P3/P5/P6/P7/P9/P10/P11/P12/P15/P18/P20 选择，因此 P4/P8/P13/P19 是真正独立的冻结集；全 16 人表是固定候选跨参与者的稳健性汇总，不能重新称为独立调参验证。",
        "",
        "## 全 16 人汇总",
        "",
        "| candidate | 参与者数 | 平均参与者 MAE (ms) | 中位参与者 MAE (ms) | 平均参与者 R | 平均参与者 coverage (%) | 最低通道 coverage (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in cohort.iterrows():
        lines.append(
            f"| `{row['candidate']}` | {int(row['participants'])} | {row['mean_participant_MAE']:.2f} | "
            f"{row['median_participant_MAE']:.2f} | {row['mean_participant_R']:.3f} | "
            f"{row['mean_participant_coverage_pct']:.2f} | {row['min_channel_coverage_pct']:.2f} |"
        )
    lines.extend([
        "",
        "## 相对 SciPy-peak 的参与者胜负",
        "",
        "负的 MAE 差值表示候选优于 SciPy-peak。`R` 与 coverage 在候选和 SciPy 的共同有效窗口情况需结合逐通道表解释。",
        "",
        "| candidate | MAE 更好参与者数 / 16 | 平均 MAE 差 (ms) | 中位 MAE 差 (ms) | 平均 R 差 |",
        "|---|---:|---:|---:|---:|",
    ])
    for candidate, group in paired.groupby("candidate", sort=True):
        diff = group["candidate_minus_scipy_MAE"].to_numpy(float)
        r_diff = group["candidate_minus_scipy_R"].to_numpy(float)
        lines.append(f"| `{candidate}` | {int((diff < 0).sum())} / {diff.size} | {np.mean(diff):.2f} | {np.median(diff):.2f} | {np.nanmean(r_diff):.3f} |")
    lines.extend([
        "",
        "## 固定候选的参与者级留一选择",
        "",
        "每一折留出 1 位参与者，在余下 15 人上仅按平均 MAE（并以 R、coverage 破平）从四条已冻结候选中选 1 条；没有重新调 detector、bandpass、correction 或 QC。随后只在被留出的参与者上与 SciPy-peak 比较。",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 留一折数 | {lopo_selection.shape[0]} |",
        f"| 选择 qppgfast_devicewise 的折数 | {int((lopo_selection['selected_candidate'] == 'qppgfast_devicewise').sum())} |",
        f"| 留出参与者上 MAE 优于 SciPy 的折数 | {int((lopo_selection['test_selected_minus_scipy_MAE'] < 0).sum())} / {lopo_selection.shape[0]} |",
        f"| 平均留出 MAE 差 (ms) | {lopo_selection['test_selected_minus_scipy_MAE'].mean():.2f} |",
        f"| 平均留出 R 差 | {lopo_selection['test_selected_minus_scipy_R'].mean():.3f} |",
        f"| 平均留出 coverage 差 (%) | {(lopo_selection['test_selected_coverage_pct'] - lopo_selection['test_scipy_coverage_pct']).mean():.2f} |",
        "",
        "## 共同有效窗口的配对比较",
        "",
        "此表对每位参与者、每个 device/channel 先取 qppgfast 候选与 SciPy-peak 都能生成 RMSSD/SDNN 的窗口，再在这批完全相同的窗口上计算 MAE 与 R；因此差异不可能来自谁保留了更多窗口。之后仍是先做 6 通道宏平均、再做 16 人汇总。",
        "",
        "| candidate | MAE 更好参与者数 / 16 | 平均共同 coverage (%) | 平均 MAE 差 (ms) | 中位 MAE 差 (ms) | 平均 R 差 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for candidate, group in common_participant.groupby("candidate", sort=True):
        diff = group["candidate_minus_scipy_MAE"].to_numpy(float)
        lines.append(
            f"| `{candidate}` | {int((diff < 0).sum())} / {diff.size} | "
            f"{group['mean_common_coverage_pct'].mean():.2f} | {np.nanmean(diff):.2f} | "
            f"{np.nanmedian(diff):.2f} | {np.nanmean(group['candidate_minus_scipy_R'].to_numpy(float)):.3f} |"
        )
    lines.extend([
        "",
        "## 文件",
        "",
        "| 文件 | 含义 |",
        "|---|---|",
        "| `lopo_fixed_channel_summary.csv` | 每参与者、设备、通道、候选的 RMSSD MAE/R/coverage |",
        "| `lopo_fixed_participant_summary.csv` | 每参与者的 6 通道宏平均 |",
        "| `lopo_fixed_cohort_summary.csv` | 全 16 人宏平均汇总 |",
        "| `lopo_fixed_vs_scipy_paired.csv` | 每位参与者相对 SciPy-peak 的配对差值 |",
        "| `lopo_fixed_leave_one_participant_selection.csv` | 每折用其余 15 人选择候选、并在留出参与者上测试的结果 |",
        "| `lopo_fixed_common_window_channel_vs_scipy.csv` | 共同有效窗口上的逐参与者、逐通道配对结果 |",
        "| `lopo_fixed_common_window_participant_vs_scipy.csv` | 共同有效窗口上的参与者级配对结果 |",
    ])
    out_dir.joinpath("LOPO_FIXED_REPORT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize fixed detector candidates at participant level.")
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    metrics = pd.read_csv(args.metrics)
    metrics = metrics[(metrics["common_bandpass"] == "bp070_350") & (metrics["ibi_correction"] == True)].copy()
    if metrics.empty:
        raise SystemExit("no bp070_350 + correction-on rows found")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    channel = _channel_summary(metrics)
    participant = _participant_summary(channel)
    cohort = _cohort_summary(participant)
    paired = _paired(participant)
    common_channel, common_participant = _common_window_vs_scipy(metrics)
    lopo_selection = _leave_one_participant_selection(participant)
    channel.to_csv(args.out_dir / "lopo_fixed_channel_summary.csv", index=False)
    participant.to_csv(args.out_dir / "lopo_fixed_participant_summary.csv", index=False)
    cohort.to_csv(args.out_dir / "lopo_fixed_cohort_summary.csv", index=False)
    paired.to_csv(args.out_dir / "lopo_fixed_vs_scipy_paired.csv", index=False)
    common_channel.to_csv(args.out_dir / "lopo_fixed_common_window_channel_vs_scipy.csv", index=False)
    common_participant.to_csv(args.out_dir / "lopo_fixed_common_window_participant_vs_scipy.csv", index=False)
    lopo_selection.to_csv(args.out_dir / "lopo_fixed_leave_one_participant_selection.csv", index=False)
    _write_report(args.out_dir, cohort, paired, common_participant, lopo_selection)
    print(f"[SAVED] {args.out_dir}")


if __name__ == "__main__":
    main()
