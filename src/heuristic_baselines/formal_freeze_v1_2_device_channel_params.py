"""
Formal v1.2 baseline optimizer with device/channel-specific parameters.

This script keeps one shared evaluation method family: recompute PPG peaks/IBI/HRV
from `ppg_resampled`, apply transparent QC gates, freeze on `training_stride30`,
then evaluate the frozen rule on `strict_reference`.

v1.2 intentionally relaxes the earlier primary-unified boundary: each device, or
each device channel, may use its own peak-method/bandpass/correction/gate choice.
It still does not train a model, does not use ECG labels per window, and does not
perform cross-device fusion.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import config  # noqa: E402
import evaluate_devicewise_recomputed_peaks_v1_1 as eval_v11  # noqa: E402


METRICS = ("RMSSD", "SDNN")
KEY_COLS = ["dataset", "role", "participant", "window_index", "device"]
CHANNEL_KEY_COLS = KEY_COLS + ["channel"]
METHOD_KEY_COLS = CHANNEL_KEY_COLS + ["peak_method"]


@dataclass(frozen=True)
class Gate:
    name: str
    min_sqi: float
    min_valid_ibi: float
    max_correction: float
    max_ibi_cv: float
    max_rmssd_ms: float = 200.0


def _tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "")


def _default_gates() -> tuple[Gate, ...]:
    gates: list[Gate] = []
    for sqi in (0.30, 0.35, 0.40, 0.45, 0.50):
        for valid_ibi in (0.70, 0.75, 0.80, 0.85, 0.90):
            for corr in (0.20, 0.25, 0.30, 0.35):
                for cv in (0.25, 0.30, 0.35, 0.40):
                    gates.append(
                        Gate(
                            f"gate_sqi{_tag(sqi)}_ibi{_tag(valid_ibi)}_corr{_tag(corr)}_cv{_tag(cv)}_rmssd200",
                            sqi,
                            valid_ibi,
                            corr,
                            cv,
                        )
                    )
    return tuple(gates)


GATES = _default_gates()


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


def _agreement_stats(ppg: np.ndarray, ecg: np.ndarray) -> dict[str, float]:
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


def _gate_mask(df: pd.DataFrame, gate: Gate) -> pd.Series:
    corr = df["ppg_ibi_correction_ratio"].astype(float)
    ibi_cv = df["ppg_ibi_cv"].astype(float)
    rmssd = df["ppg_rmssd_ms"].astype(float)
    sdnn = df["ppg_sdnn_ms"].astype(float)
    return (
        (df["ppg_valid_sample_ratio"].astype(float) >= 0.90)
        & (df["ppg_sqi"].astype(float) >= gate.min_sqi)
        & (df["ppg_valid_ibi_ratio"].astype(float) >= gate.min_valid_ibi)
        & np.isfinite(corr)
        & (corr <= gate.max_correction)
        & np.isfinite(ibi_cv)
        & (ibi_cv <= gate.max_ibi_cv)
        & np.isfinite(rmssd)
        & (rmssd <= gate.max_rmssd_ms)
        & np.isfinite(sdnn)
    )


def _load_channel_metrics(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(Path(path).resolve()) for path in paths]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    before = len(df)
    df = df.drop_duplicates(METHOD_KEY_COLS, keep="first").reset_index(drop=True)
    if len(df) != before:
        print(f"[dedupe] dropped {before - len(df)} duplicated channel-method rows")
    return df


def _denominators(df: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    return {
        (str(dataset), str(role), str(device)): int(group[["participant", "window_index"]].drop_duplicates().shape[0])
        for (dataset, role, device), group in df.groupby(["dataset", "role", "device"], dropna=False)
    }


def _summarize_predictions(
    pred: pd.DataFrame,
    group_cols: list[str],
    denominators: dict[tuple[str, str, str], int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if pred.empty:
        return pd.DataFrame()
    for keys, group in pred.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        denom = denominators[(str(base["dataset"]), str(base["role"]), str(base["device"]))]
        for metric in METRICS:
            ppg = group[f"ppg_{metric.lower()}_ms"].to_numpy(float)
            ecg = group[f"ecg_{metric.lower()}_ms"].to_numpy(float)
            stats = _agreement_stats(ppg, ecg)
            rows.append(
                {
                    **base,
                    "hrv_metric": metric,
                    "n_total": int(denom),
                    **stats,
                    "coverage_pct": float(100.0 * stats["n_valid"] / denom) if denom else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _candidate_training_summary(
    training_df: pd.DataFrame,
    denominators: dict[tuple[str, str, str], int],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for gate in GATES:
        gated = training_df[_gate_mask(training_df, gate)].copy()
        if gated.empty:
            continue
        gated["gate"] = gate.name
        frames.append(
            _summarize_predictions(
                gated,
                ["dataset", "role", "device", "channel", "peak_method", "gate"],
                denominators,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _select_device_channel_params(
    training_summary: pd.DataFrame,
    min_coverage_pct: float,
) -> pd.DataFrame:
    rmssd = training_summary[
        (training_summary["hrv_metric"] == "RMSSD")
        & np.isfinite(training_summary["MAE"].to_numpy(float))
        & (training_summary["coverage_pct"].astype(float) >= min_coverage_pct)
    ].copy()
    if rmssd.empty:
        return rmssd
    selected = (
        rmssd.sort_values(["device", "MAE", "coverage_pct", "R"], ascending=[True, True, False, False])
        .groupby("device", dropna=False)
        .head(1)
        .reset_index(drop=True)
    )
    selected["policy"] = "fixed_device_channel_params"
    return selected


def _select_channel_specific_params(
    training_summary: pd.DataFrame,
    min_channel_coverage_pct: float,
    top_k_per_channel: int,
) -> pd.DataFrame:
    rmssd = training_summary[
        (training_summary["hrv_metric"] == "RMSSD")
        & np.isfinite(training_summary["MAE"].to_numpy(float))
        & (training_summary["coverage_pct"].astype(float) >= min_channel_coverage_pct)
    ].copy()
    if rmssd.empty:
        return rmssd
    ranked = (
        rmssd.sort_values(["device", "channel", "MAE", "coverage_pct", "R"], ascending=[True, True, True, False, False])
        .groupby(["device", "channel"], dropna=False)
        .head(top_k_per_channel)
        .reset_index(drop=True)
    )
    ranked["policy"] = "channel_specific_params_sqi_best"
    return ranked


def _apply_fixed_device_policy(df: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    pieces: list[pd.DataFrame] = []
    for _, row in selected.iterrows():
        gate = _gate_by_name(str(row["gate"]))
        mask = (
            (df["device"].astype(str) == str(row["device"]))
            & (df["channel"].astype(str) == str(row["channel"]))
            & (df["peak_method"].astype(str) == str(row["peak_method"]))
            & _gate_mask(df, gate)
        )
        piece = df[mask].copy()
        piece["gate"] = gate.name
        piece["policy"] = "fixed_device_channel_params"
        piece["frozen_method"] = (
            piece["device"].astype(str)
            + "__"
            + piece["channel"].astype(str)
            + "__"
            + piece["peak_method"].astype(str)
            + "__"
            + gate.name
        )
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _apply_channel_specific_policy(df: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    pieces: list[pd.DataFrame] = []
    for _, row in selected.iterrows():
        gate = _gate_by_name(str(row["gate"]))
        mask = (
            (df["device"].astype(str) == str(row["device"]))
            & (df["channel"].astype(str) == str(row["channel"]))
            & (df["peak_method"].astype(str) == str(row["peak_method"]))
            & _gate_mask(df, gate)
        )
        piece = df[mask].copy()
        piece["gate"] = gate.name
        piece["policy"] = "channel_specific_params_sqi_best"
        piece["frozen_method"] = (
            piece["device"].astype(str)
            + "__"
            + piece["channel"].astype(str)
            + "__"
            + piece["peak_method"].astype(str)
            + "__"
            + gate.name
        )
        pieces.append(piece)
    candidates = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if candidates.empty:
        return candidates
    sort_cols = KEY_COLS + ["ppg_sqi", "ppg_valid_ibi_ratio", "ppg_ibi_correction_ratio"]
    ordered = candidates.sort_values(sort_cols, ascending=[True, True, True, True, True, False, False, True])
    return ordered.groupby(KEY_COLS, dropna=False).head(1).copy()


def _gate_by_name(name: str) -> Gate:
    for gate in GATES:
        if gate.name == name:
            return gate
    raise KeyError(name)


def _choose_best_policy(
    fixed_eval: pd.DataFrame,
    channel_eval: pd.DataFrame,
    min_coverage_pct: float,
) -> str:
    rows: list[dict[str, object]] = []
    for policy, eval_df in (
        ("fixed_device_channel_params", fixed_eval),
        ("channel_specific_params_sqi_best", channel_eval),
    ):
        rmssd = eval_df[
            (eval_df["role"] == "training_stride30")
            & (eval_df["hrv_metric"] == "RMSSD")
            & np.isfinite(eval_df["MAE"].to_numpy(float))
            & (eval_df["coverage_pct"].astype(float) >= min_coverage_pct)
        ].copy()
        if rmssd.empty:
            continue
        rows.append(
            {
                "policy": policy,
                "n_devices": rmssd["device"].nunique(),
                "mean_MAE": float(rmssd["MAE"].mean()),
                "mean_R": float(rmssd["R"].mean()),
                "mean_coverage_pct": float(rmssd["coverage_pct"].mean()),
                "min_coverage_pct": float(rmssd["coverage_pct"].min()),
            }
        )
    if not rows:
        return ""
    summary = pd.DataFrame(rows)
    return str(summary.sort_values(["mean_MAE", "min_coverage_pct"], ascending=[True, False]).iloc[0]["policy"])


def _write_readme(
    out_dir: Path,
    channel_df: pd.DataFrame,
    selected_fixed: pd.DataFrame,
    selected_channel: pd.DataFrame,
    chosen_policy: str,
    frozen_eval: pd.DataFrame,
    policy_comparison: pd.DataFrame,
    top_candidates: pd.DataFrame,
) -> None:
    lines = [
        "# Formal v1.2 Device/Channel Parameter Baseline",
        "",
        "This is an experiment record, not manuscript prose.",
        "",
        "## Boundary",
        "",
        "- Selection uses only `training_stride30`.",
        "- `strict_reference` is used only after freezing.",
        "- No subject-level held-out split is used here.",
        "- No model training, no cross-device fusion, and no ECG-label per-window selection.",
        "- All candidates use the same transparent recompute-peaks -> IBI/QC -> HRV method family.",
        "- v1.2 allows device/channel-specific peak parameters and QC gates.",
        "",
        "## Inputs",
        "",
        f"- Channel metric rows after de-duplication: `{len(channel_df)}`",
        f"- Peak methods: `{channel_df['peak_method'].nunique()}`",
        f"- Gate grid size: `{len(GATES)}`",
        "",
        "## Chosen Policy",
        "",
        f"- Policy: `{chosen_policy}`" if chosen_policy else "- No policy satisfied the coverage constraint.",
        "",
    ]
    selected = selected_channel if chosen_policy == "channel_specific_params_sqi_best" else selected_fixed
    if not selected.empty:
        lines.extend([
            "## Frozen Choices",
            "",
            "| Device | Channel | Peak method | Gate | Training RMSSD MAE | Training R | Training coverage |",
            "|---|---|---|---|---:|---:|---:|",
        ])
        for _, row in selected.sort_values(["device", "channel"]).iterrows():
            lines.append(
                f"| `{row['device']}` | `{row['channel']}` | `{row['peak_method']}` | `{row['gate']}` | "
                f"{_fmt(row['MAE'])} ms | {_fmt(row['R'], 3)} | {_fmt(row['coverage_pct'])}% |"
            )
        lines.append("")

    if not policy_comparison.empty:
        lines.extend([
            "## Policy Comparison on Training",
            "",
            "| Policy | Mean RMSSD MAE | Mean R | Mean coverage | Min coverage |",
            "|---|---:|---:|---:|---:|",
        ])
        for _, row in policy_comparison.iterrows():
            lines.append(
                f"| `{row['policy']}` | {_fmt(row['mean_MAE'])} ms | {_fmt(row['mean_R'], 3)} | "
                f"{_fmt(row['mean_coverage_pct'])}% | {_fmt(row['min_coverage_pct'])}% |"
            )
        lines.append("")

    lines.extend([
        "## Frozen Evaluation",
        "",
        "| Role | Device | Metric | Valid windows | Coverage | MAE | RMSE | R | Bias |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in frozen_eval.sort_values(["role", "device", "hrv_metric"]).iterrows():
        lines.append(
            f"| {row['role']} | `{row['device']}` | {row['hrv_metric']} | {int(row['n_valid'])} | "
            f"{_fmt(row['coverage_pct'])}% | {_fmt(row['MAE'])} ms | {_fmt(row['RMSE'])} ms | "
            f"{_fmt(row['R'], 3)} | {_fmt(row['bias'])} ms |"
        )

    lines.extend([
        "",
        "## Training Top Candidates",
        "",
        "| Device | Channel | Peak method | Gate | Coverage | MAE | R |",
        "|---|---|---|---|---:|---:|---:|",
    ])
    for _, row in top_candidates.iterrows():
        lines.append(
            f"| `{row['device']}` | `{row['channel']}` | `{row['peak_method']}` | `{row['gate']}` | "
            f"{_fmt(row['coverage_pct'])}% | {_fmt(row['MAE'])} ms | {_fmt(row['R'], 3)} |"
        )

    lines.extend([
        "",
        "## Output Files",
        "",
        "| File | Meaning |",
        "|---|---|",
        "| `v1_2_training_channel_param_summary.csv` | Training-only summaries for channel/peak/gate candidates |",
        "| `v1_2_fixed_device_channel_choices.csv` | Best fixed device-channel parameter choices |",
        "| `v1_2_channel_specific_choices.csv` | Best channel-specific parameter choices |",
        "| `v1_2_policy_comparison.csv` | Policy-level training comparison |",
        "| `v1_2_frozen_eval.csv` | Frozen policy evaluation on training and strict_reference |",
        "| `summary.json` | Machine-readable run metadata |",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze v1.2 baseline with device/channel-specific parameters.")
    parser.add_argument(
        "--channel-metrics-csv",
        action="append",
        default=[],
        help="Input channel metrics CSV. May be provided more than once; duplicate channel-method rows are removed.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "formal_v1_2_device_channel_params"),
    )
    parser.add_argument("--min-device-coverage-pct", type=float, default=20.0)
    parser.add_argument("--min-channel-coverage-pct", type=float, default=10.0)
    parser.add_argument("--top-k-per-channel", type=int, default=1)
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

    channel_df = _load_channel_metrics(args.channel_metrics_csv)
    denominators = _denominators(channel_df)
    training_df = channel_df[channel_df["role"] == "training_stride30"].copy()

    training_summary = _candidate_training_summary(training_df, denominators)
    selected_fixed = _select_device_channel_params(training_summary, args.min_device_coverage_pct)
    selected_channel = _select_channel_specific_params(training_summary, args.min_channel_coverage_pct, args.top_k_per_channel)

    fixed_pred = _apply_fixed_device_policy(channel_df, selected_fixed)
    fixed_eval = _summarize_predictions(fixed_pred, ["dataset", "role", "device", "policy"], denominators)
    channel_pred = _apply_channel_specific_policy(channel_df, selected_channel)
    channel_eval = _summarize_predictions(channel_pred, ["dataset", "role", "device", "policy"], denominators)

    policy_rows: list[dict[str, object]] = []
    for policy, eval_df in (
        ("fixed_device_channel_params", fixed_eval),
        ("channel_specific_params_sqi_best", channel_eval),
    ):
        rmssd = eval_df[(eval_df["role"] == "training_stride30") & (eval_df["hrv_metric"] == "RMSSD")].copy()
        if rmssd.empty:
            continue
        policy_rows.append(
            {
                "policy": policy,
                "n_devices": int(rmssd["device"].nunique()),
                "mean_MAE": float(rmssd["MAE"].mean()),
                "mean_R": float(rmssd["R"].mean()),
                "mean_coverage_pct": float(rmssd["coverage_pct"].mean()),
                "min_coverage_pct": float(rmssd["coverage_pct"].min()),
            }
        )
    policy_comparison = pd.DataFrame(policy_rows).sort_values(["mean_MAE", "min_coverage_pct"], ascending=[True, False])
    chosen_policy = _choose_best_policy(fixed_eval, channel_eval, args.min_device_coverage_pct)
    frozen_eval = channel_eval if chosen_policy == "channel_specific_params_sqi_best" else fixed_eval
    chosen_rows = selected_channel if chosen_policy == "channel_specific_params_sqi_best" else selected_fixed

    top_candidates = (
        training_summary[
            (training_summary["hrv_metric"] == "RMSSD")
            & np.isfinite(training_summary["MAE"].to_numpy(float))
            & (training_summary["coverage_pct"].astype(float) >= args.min_device_coverage_pct)
        ]
        .sort_values(["device", "MAE", "coverage_pct"], ascending=[True, True, False])
        .groupby("device", dropna=False)
        .head(8)
        .reset_index(drop=True)
    )

    training_summary.to_csv(out_dir / "v1_2_training_channel_param_summary.csv", index=False)
    selected_fixed.to_csv(out_dir / "v1_2_fixed_device_channel_choices.csv", index=False)
    selected_channel.to_csv(out_dir / "v1_2_channel_specific_choices.csv", index=False)
    policy_comparison.to_csv(out_dir / "v1_2_policy_comparison.csv", index=False)
    frozen_eval.to_csv(out_dir / "v1_2_frozen_eval.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "channel_metrics_csv": [str(Path(p).resolve()) for p in args.channel_metrics_csv],
                "out_dir": str(out_dir),
                "min_device_coverage_pct": args.min_device_coverage_pct,
                "min_channel_coverage_pct": args.min_channel_coverage_pct,
                "top_k_per_channel": args.top_k_per_channel,
                "selection_role": "training_stride30",
                "final_evaluation_role": "strict_reference",
                "subject_level_held_out_split": False,
                "cross_device_fusion": False,
                "uses_saved_ppg_metadata": False,
                "gate_grid": [asdict(g) for g in GATES],
                "chosen_policy": chosen_policy,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_readme(
        out_dir,
        channel_df,
        selected_fixed,
        chosen_rows,
        chosen_policy,
        frozen_eval,
        policy_comparison,
        top_candidates,
    )

    print(f"[saved] {out_dir}")
    print(policy_comparison.to_string(index=False))
    print(f"[chosen] {chosen_policy}")
    print(frozen_eval.to_string(index=False))


if __name__ == "__main__":
    main()
