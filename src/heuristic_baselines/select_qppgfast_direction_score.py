"""Select and validate qppgfast peak/foot direction-score pipelines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEVICE_INCUMBENT = {"Earring": "peak", "Ring": "foot", "Watch": "foot"}
SCORE_MODES = ("peak_train", "foot_train")
FIDUCIALS = ("peak", "foot")
MAX_COVERAGE_LOSS_PP = 1.0
MAE_TIE_MS = 1.0


def _valid(group: pd.DataFrame) -> pd.Series:
    return (
        np.isfinite(group["ppg_rmssd_ms"].astype(float))
        & np.isfinite(group["ppg_sdnn_ms"].astype(float))
        & np.isfinite(group["ecg_rmssd_ms"].astype(float))
        & np.isfinite(group["ecg_sdnn_ms"].astype(float))
    )


def _correlation(pred: pd.Series, ref: pd.Series) -> float:
    if len(pred) < 3 or float(pred.std()) == 0.0 or float(ref.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(pred, ref)[0, 1])


def _load(path: Path, expected_mode: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[
        (df["detector"] == "qppgfast")
        & (df["common_bandpass"] == "bp070_350")
        & (df["ibi_correction"].astype(bool))
        & (df["polarity_mode"] == expected_mode)
    ].copy()
    if df.empty:
        raise RuntimeError(f"no qppgfast {expected_mode} rows in {path}")
    df["metric_valid"] = _valid(df)
    return df


def _participant_channel(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["polarity_mode", "fiducial", "participant", "device", "channel"]
    for keys, group in raw.groupby(group_cols, dropna=False):
        valid = group[group["metric_valid"]]
        rows.append(
            {
                "polarity_mode": keys[0],
                "fiducial": keys[1],
                "participant": keys[2],
                "device": keys[3],
                "channel": keys[4],
                "n_total": int(len(group)),
                "n_valid": int(len(valid)),
                "coverage_over_total_pct": 100.0 * float(group["metric_valid"].mean()),
                "MAE_ms": float(np.mean(np.abs(valid["ppg_rmssd_ms"] - valid["ecg_rmssd_ms"]))) if len(valid) else float("nan"),
                "R": _correlation(valid["ppg_rmssd_ms"], valid["ecg_rmssd_ms"]),
            }
        )
    return pd.DataFrame(rows)


def _device_summary(channel: pd.DataFrame) -> pd.DataFrame:
    return (
        channel.groupby(["polarity_mode", "fiducial", "device"], dropna=False)
        .agg(
            participant_channels=("MAE_ms", "size"),
            mean_coverage_over_total_pct=("coverage_over_total_pct", "mean"),
            min_coverage_over_total_pct=("coverage_over_total_pct", "min"),
            mean_MAE_ms=("MAE_ms", "mean"),
            mean_R=("R", "mean"),
        )
        .reset_index()
        .sort_values(["device", "mean_MAE_ms", "mean_R"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def _select(device_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for device, incumbent_fiducial in DEVICE_INCUMBENT.items():
        candidates = device_summary[device_summary["device"] == device].copy()
        incumbent = candidates[
            (candidates["polarity_mode"] == "peak_train") & (candidates["fiducial"] == incumbent_fiducial)
        ]
        if len(incumbent) != 1:
            raise RuntimeError(f"missing incumbent for {device}")
        incumbent_row = incumbent.iloc[0]
        coverage_floor = float(incumbent_row["mean_coverage_over_total_pct"]) - MAX_COVERAGE_LOSS_PP
        candidates["coverage_floor_pct"] = coverage_floor
        candidates["coverage_eligible"] = candidates["mean_coverage_over_total_pct"] >= coverage_floor
        eligible = candidates[candidates["coverage_eligible"]].copy()
        if eligible.empty:
            raise RuntimeError(f"no coverage-eligible candidate for {device}")
        best_mae = float(eligible["mean_MAE_ms"].min())
        eligible["mae_within_tie_ms"] = eligible["mean_MAE_ms"] <= best_mae + MAE_TIE_MS
        finalists = eligible[eligible["mae_within_tie_ms"]].copy()
        finalists["_r_sort"] = finalists["mean_R"].fillna(-np.inf)
        winner = finalists.sort_values(
            ["_r_sort", "mean_coverage_over_total_pct", "polarity_mode", "fiducial"],
            ascending=[False, False, True, True],
        ).iloc[0]
        candidates["selected"] = (
            (candidates["polarity_mode"] == winner["polarity_mode"])
            & (candidates["fiducial"] == winner["fiducial"])
        )
        candidates["incumbent_fiducial"] = incumbent_fiducial
        candidates["incumbent_coverage_pct"] = float(incumbent_row["mean_coverage_over_total_pct"])
        rows.extend(candidates.drop(columns="_r_sort", errors="ignore").to_dict("records"))
    return pd.DataFrame(rows).sort_values(["device", "selected", "mean_MAE_ms"], ascending=[True, False, True]).reset_index(drop=True)


def _apply_selection(channel: pd.DataFrame, selection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    winners = selection[selection["selected"]][["device", "polarity_mode", "fiducial"]]
    chosen = channel.merge(winners, on=["device", "polarity_mode", "fiducial"], how="inner", validate="many_to_one")
    per_device = _device_summary(chosen)
    overall = pd.DataFrame(
        [
            {
                "participant_channels": int(len(chosen)),
                "mean_coverage_over_total_pct": float(chosen["coverage_over_total_pct"].mean()),
                "min_coverage_over_total_pct": float(chosen["coverage_over_total_pct"].min()),
                "mean_MAE_ms": float(chosen["MAE_ms"].mean()),
                "mean_R": float(chosen["R"].mean()),
            }
        ]
    )
    return per_device, overall


def _write_readme(
    out_dir: Path,
    train_selection: pd.DataFrame,
    holdout_selected: pd.DataFrame,
    holdout_incumbent: pd.DataFrame,
    holdout_selected_device: pd.DataFrame,
    holdout_incumbent_device: pd.DataFrame,
) -> None:
    lines = [
        "# qppgfast peak/foot 方向 score 选择",
        "",
        "## 预先固定规则",
        "",
        "- 候选：`peak_train`/`foot_train` 两种共同方向选择，各自配 `peak`/`foot` 输出，共四个可部署 pipeline。",
        "- 训练选择按设备汇总两通道：coverage 不得低于 `peak_train + 现行设备 fiducial` 超过 1 个百分点；其余先选最低 MAE；距最低 MAE 不超过 1 ms 的候选再按较高 R、较高 coverage 排序。",
        "- holdout 不参与选择，只按训练期选定的设备规则验收。MAE/R 均先按 participant/channel 计算，再等权平均；coverage 是每个 participant/channel 的 `n_valid / n_total` 后等权平均。",
        "",
        "## 训练期选择结果",
        "",
        "| 设备 | 选定方向 score | 选定输出 | 训练 MAE (ms) | 训练 R | 训练 coverage (%) |",
        "|---|---|---|---:|---:|---:|",
    ]
    selected = train_selection[train_selection["selected"]]
    for _, row in selected.iterrows():
        lines.append(
            f"| {row['device']} | `{row['polarity_mode']}` | `{row['fiducial']}` | "
            f"{row['mean_MAE_ms']:.2f} | {row['mean_R']:.3f} | {row['mean_coverage_over_total_pct']:.2f} |"
        )
    lines.extend([
        "",
        "## 独立 Holdout",
        "",
        "| 条件 | MAE (ms) | R | coverage (%) |",
        "|---|---:|---:|---:|",
    ])
    for label, frame in (("训练选定规则", holdout_selected), ("现行 peak_train devicewise 对照（重新评估）", holdout_incumbent)):
        row = frame.iloc[0]
        lines.append(
            f"| {label} | {row['mean_MAE_ms']:.2f} | {row['mean_R']:.3f} | {row['mean_coverage_over_total_pct']:.2f} |"
        )
    lines.extend([
        "",
        "| 设备 | 训练选定 MAE | 现行对照 MAE | MAE 差（选定-对照） | 训练选定 R | 现行对照 R |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    selected_by_device = holdout_selected_device.set_index("device")
    incumbent_by_device = holdout_incumbent_device.set_index("device")
    for device in DEVICE_INCUMBENT:
        selected_row = selected_by_device.loc[device]
        incumbent_row = incumbent_by_device.loc[device]
        mae_gap = float(selected_row["mean_MAE_ms"] - incumbent_row["mean_MAE_ms"])
        lines.append(
            f"| {device} | {selected_row['mean_MAE_ms']:.2f} | {incumbent_row['mean_MAE_ms']:.2f} | "
            f"{mae_gap:+.2f} | {selected_row['mean_R']:.3f} | {incumbent_row['mean_R']:.3f} |"
        )
    selected = holdout_selected.iloc[0]
    incumbent = holdout_incumbent.iloc[0]
    mae_gap = float(selected["mean_MAE_ms"] - incumbent["mean_MAE_ms"])
    lines.extend([
        "",
        "## 冻结结论",
        "",
    ])
    if mae_gap < 0.0:
        lines.append("训练选定规则在独立 holdout 的平均 MAE 更低；仍需结合逐设备结果后才可考虑替代冻结规则。")
    else:
        lines.append(
            f"不冻结训练选定规则：独立 holdout 的平均 MAE 比现行 `peak_train` devicewise 对照高 `{mae_gap:.2f} ms`。"
        )
    (out_dir / "README_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select qppgfast direction-score/fiducial pipelines on train and validate on holdout.")
    parser.add_argument("--train-peak", type=Path, required=True)
    parser.add_argument("--train-foot", type=Path, required=True)
    parser.add_argument("--holdout-peak", type=Path, required=True)
    parser.add_argument("--holdout-foot", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    train_raw = pd.concat([_load(args.train_peak, "peak_train"), _load(args.train_foot, "foot_train")], ignore_index=True)
    holdout_raw = pd.concat([_load(args.holdout_peak, "peak_train"), _load(args.holdout_foot, "foot_train")], ignore_index=True)
    train_channel = _participant_channel(train_raw)
    holdout_channel = _participant_channel(holdout_raw)
    train_device = _device_summary(train_channel)
    selection = _select(train_device)
    holdout_selected_device, holdout_selected_overall = _apply_selection(holdout_channel, selection)

    incumbent = pd.DataFrame(
        [{"device": device, "polarity_mode": "peak_train", "fiducial": fiducial} for device, fiducial in DEVICE_INCUMBENT.items()]
    )
    incumbent_selection = incumbent.merge(selection[["device", "polarity_mode", "fiducial", "selected"]], on=["device", "polarity_mode", "fiducial"], how="left")
    incumbent_selection["selected"] = True
    holdout_incumbent_device, holdout_incumbent_overall = _apply_selection(holdout_channel, incumbent_selection)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_channel.to_csv(args.out_dir / "train_participant_channel_candidates.csv", index=False)
    train_device.to_csv(args.out_dir / "train_device_candidates.csv", index=False)
    selection.to_csv(args.out_dir / "train_device_selection.csv", index=False)
    holdout_channel.to_csv(args.out_dir / "holdout_participant_channel_candidates.csv", index=False)
    holdout_selected_device.to_csv(args.out_dir / "holdout_selected_device_summary.csv", index=False)
    holdout_selected_overall.to_csv(args.out_dir / "holdout_selected_overall_summary.csv", index=False)
    holdout_incumbent_device.to_csv(args.out_dir / "holdout_incumbent_device_summary.csv", index=False)
    holdout_incumbent_overall.to_csv(args.out_dir / "holdout_incumbent_overall_summary.csv", index=False)
    (args.out_dir / "selection_config.json").write_text(
        json.dumps(
            {
                "score_modes": list(SCORE_MODES),
                "fiducials": list(FIDUCIALS),
                "coverage_max_loss_pp": MAX_COVERAGE_LOSS_PP,
                "mae_tie_ms": MAE_TIE_MS,
                "incumbent": DEVICE_INCUMBENT,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_readme(
        args.out_dir,
        selection,
        holdout_selected_overall,
        holdout_incumbent_overall,
        holdout_selected_device,
        holdout_incumbent_device,
    )
    print(f"[SAVED] {args.out_dir}")


if __name__ == "__main__":
    main()
