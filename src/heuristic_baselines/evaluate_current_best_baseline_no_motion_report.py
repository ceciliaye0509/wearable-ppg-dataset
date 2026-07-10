"""Generate participant/device report for the current best no-motion baseline."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PKG_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

OUTPUT_ROOT = PKG_ROOT / "outputs"

NO_MOTION_THRESHOLD = 0.1

DATASETS = [
    {
        "label": "strict_reference",
        "name": "synced_3device_rawaligned_strict_reference_100hz_ecgibi100_sample90_ecgsample100",
        "dir": OUTPUT_ROOT / "synced_3device_rawaligned_strict_reference_100hz_ecgibi100_sample90_ecgsample100",
        "interpretation": "non-overlapping 5 min windows, filtered to per-device no-motion windows",
    },
    {
        "label": "training_v1_stride30",
        "name": "synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100_stride30",
        "dir": OUTPUT_ROOT / "synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100_stride30",
        "interpretation": "30 s stride rolling windows, filtered to per-device no-motion windows",
    },
]

METRICS = ("RMSSD", "SDNN")


def _threshold_token(threshold: float) -> str:
    return str(threshold).replace(".", "p").replace("-", "neg")


def _report_paths(threshold: float, output_dir: Path | None = None) -> tuple[Path, Path]:
    root = output_dir or PKG_ROOT
    if np.isclose(threshold, 0.1):
        stem = "current_best_baseline_no_motion_strict_vs_training_stride30_by_participant_device"
    else:
        stem = f"current_best_baseline_motion_lt_{_threshold_token(threshold)}_strict_vs_training_stride30_by_participant_device"
    return root / f"{stem}.md", root / f"{stem}.csv"


def _participant_sort_key(pid: object) -> int:
    s = str(pid)
    return int(s[1:]) if s.startswith("P") and s[1:].isdigit() else 999999


def _safe_corr(x: list[float], y: list[float]) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(xx[mask], yy[mask])[0, 1])


def _agreement(ppg: list[float], ecg: list[float]) -> dict[str, float]:
    p = np.asarray(ppg, dtype=float)
    e = np.asarray(ecg, dtype=float)
    mask = np.isfinite(p) & np.isfinite(e)
    p = p[mask]
    e = e[mask]
    if p.size == 0:
        return {"n_valid": 0, "MAE": np.nan, "RMSE": np.nan, "R": np.nan, "bias": np.nan}
    diff = p - e
    return {
        "n_valid": int(p.size),
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(np.mean(diff * diff))),
        "R": _safe_corr(p.tolist(), e.tolist()),
        "bias": float(np.mean(diff)),
    }


def _process_npz(task: tuple[str, str, str]) -> list[dict[str, object]]:
    dataset_label, dataset_name, path_str = task
    from heuristic_baselines.algorithms.hrv import hrv_from_ppg

    rows: list[dict[str, object]] = []
    path = Path(path_str)
    with np.load(path, allow_pickle=True) as z:
        participant = str(np.asarray(z["participant"]).item())
        devices = [str(x) for x in np.asarray(z["devices"]).tolist()]
        channels = [str(x) for x in np.asarray(z["channels"]).tolist()]
        cfg = json.loads(str(np.asarray(z["config_json"]).item()))
        fs = float(cfg.get("target_fs", 100.0))
        ppg = z["ppg_resampled"]
        ecg_rmssd = np.asarray(z["ecg_rmssd_ms"], dtype=float)
        ecg_sdnn = np.asarray(z["ecg_sdnn_ms"], dtype=float)
        accel_raw = np.asarray(z["accel_mean_mag"], dtype=float)
        accel_motion = np.asarray(
            z["accel_motion_mean_mag"] if "accel_motion_mean_mag" in z.files else z["accel_mean_mag"],
            dtype=float,
        )
        total_windows = int(ppg.shape[0])

        for di, device in enumerate(devices):
            no_motion_mask = np.isfinite(accel_motion[:, di]) & (accel_motion[:, di] < NO_MOTION_THRESHOLD)
            no_motion_idx = np.where(no_motion_mask)[0]
            n_no_motion = int(no_motion_idx.size)
            pred = {metric: [] for metric in METRICS}
            ref = {metric: [] for metric in METRICS}
            selected_counts: Counter[str] = Counter()
            reason_counts: Counter[str] = Counter()
            sqi_vals: list[float] = []
            valid_ibi_vals: list[float] = []
            corr_vals: list[float] = []
            n_selected = 0

            for wi in no_motion_idx:
                candidates: list[dict[str, float | str]] = []
                for ci, channel in enumerate(channels):
                    out = hrv_from_ppg(ppg[wi, di, ci], fs, freq=False, nonlinear=False)
                    rmssd = float(out.get("HRV_RMSSD", np.nan))
                    sdnn = float(out.get("HRV_SDNN", np.nan))
                    sqi = float(out.get("sqi", np.nan))
                    valid_ibi = float(out.get("valid_ibi_ratio", np.nan))
                    corr = float(out.get("ibi_correction_ratio", np.nan))
                    reason = str(out.get("ppg_qc_reason", ""))
                    if np.isfinite(rmssd) and np.isfinite(sdnn):
                        candidates.append(
                            {
                                "channel": channel,
                                "rmssd": rmssd,
                                "sdnn": sdnn,
                                "sqi": sqi,
                                "valid_ibi": valid_ibi,
                                "corr": corr,
                                "reason": reason,
                            }
                        )
                    else:
                        reason_counts[reason or "invalid"] += 1

                if not candidates:
                    continue

                def sort_key(c: dict[str, float | str]) -> tuple[float, float, float, str]:
                    corr = float(c["corr"]) if np.isfinite(float(c["corr"])) else 999.0
                    sqi = float(c["sqi"]) if np.isfinite(float(c["sqi"])) else -1.0
                    valid_ibi = float(c["valid_ibi"]) if np.isfinite(float(c["valid_ibi"])) else -1.0
                    return (-sqi, -valid_ibi, corr, str(c["channel"]))

                best = sorted(candidates, key=sort_key)[0]
                n_selected += 1
                selected_counts[str(best["channel"])] += 1
                reason_counts[str(best["reason"]) or "ok"] += 1
                if np.isfinite(float(best["sqi"])):
                    sqi_vals.append(float(best["sqi"]))
                if np.isfinite(float(best["valid_ibi"])):
                    valid_ibi_vals.append(float(best["valid_ibi"]))
                if np.isfinite(float(best["corr"])):
                    corr_vals.append(float(best["corr"]))
                pred["RMSSD"].append(float(best["rmssd"]))
                ref["RMSSD"].append(float(ecg_rmssd[wi]))
                pred["SDNN"].append(float(best["sdnn"]))
                ref["SDNN"].append(float(ecg_sdnn[wi]))

            base = {
                "dataset": dataset_name,
                "dataset_label": dataset_label,
                "participant": participant,
                "device": device,
                "method": "no_motion_device_best_sqi_green_or_ir__hrv_from_ppg",
                "no_motion_definition": f"accel_motion_mean_mag < {NO_MOTION_THRESHOLD}",
                "total_windows": total_windows,
                "n_no_motion_windows": n_no_motion,
                "no_motion_pct": 100.0 * n_no_motion / total_windows if total_windows else np.nan,
                "n_selected_windows": n_selected,
                "coverage_pct_within_no_motion": 100.0 * n_selected / n_no_motion if n_no_motion else np.nan,
                "coverage_pct_of_all_windows": 100.0 * n_selected / total_windows if total_windows else np.nan,
                "mean_accel_mean_mag_all_windows": float(np.nanmean(accel_raw[:, di])) if total_windows else np.nan,
                "mean_accel_motion_mean_mag_all_windows": float(np.nanmean(accel_motion[:, di])) if total_windows else np.nan,
                "mean_accel_mean_mag_no_motion": float(np.nanmean(accel_raw[no_motion_idx, di])) if n_no_motion else np.nan,
                "mean_accel_motion_mean_mag_no_motion": float(np.nanmean(accel_motion[no_motion_idx, di])) if n_no_motion else np.nan,
                "selected_ppg_green": int(selected_counts.get("ppg_green", 0)),
                "selected_ppg_ir": int(selected_counts.get("ppg_ir", 0)),
                "mean_selected_sqi": float(np.nanmean(sqi_vals)) if sqi_vals else np.nan,
                "mean_selected_valid_ibi_ratio": float(np.nanmean(valid_ibi_vals)) if valid_ibi_vals else np.nan,
                "mean_selected_ibi_correction_ratio": float(np.nanmean(corr_vals)) if corr_vals else np.nan,
                "top_qc_reasons": "; ".join(f"{k}:{v}" for k, v in reason_counts.most_common(4)),
            }
            for metric in METRICS:
                rows.append({**base, "hrv_metric": metric, **_agreement(pred[metric], ref[metric])})
    return rows


def _fmt(value: object, digits: int = 2) -> str:
    try:
        f = float(value)
    except Exception:
        return ""
    if not np.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


def _md_table(df: pd.DataFrame, cols: list[str], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals: list[str] = []
        for col in cols:
            v = row[col]
            if col in {
                "MAE",
                "RMSE",
                "bias",
                "mean_accel_mean_mag_no_motion",
                "mean_accel_motion_mean_mag_no_motion",
                "RMSSD_MAE_mean_by_participant",
                "mae_full",
                "mae_candidate",
                "mean_rmssd_mae",
            }:
                vals.append(_fmt(v, 2))
            elif col in {"R", "RMSSD_R_median_by_participant", "r_full_median", "r_candidate_median", "median_rmssd_r"}:
                vals.append(_fmt(v, 3))
            elif col.endswith("pct") or col.startswith("coverage_pct") or col == "no_motion_pct" or col.endswith("_pct"):
                vals.append(_fmt(v, 1))
            elif col.startswith("n_") or col.startswith("selected_") or col == "total_windows" or col in {"paired_mae_rows", "candidate_lower_mae", "paired_r_rows", "candidate_higher_r", "coverage_rows", "candidate_higher_coverage", "subset_windows", "valid_predictions"}:
                vals.append(str(int(v)) if pd.notna(v) else "")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _sort_participant_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(_pid_sort=df["participant"].map(_participant_sort_key))
        .sort_values(["dataset_label", "_pid_sort", "device"])
        .drop(columns=["_pid_sort"])
    )


def _comparison_candidates(current_threshold: float, current_csv: Path, output_dir: Path | None = None) -> dict[str, Path]:
    root = output_dir or current_csv.parent
    candidates = {
        "full": root / "current_best_baseline_strict_vs_training_stride30_by_participant_device.csv",
        "<0.1": _report_paths(0.1, root)[1],
        "<0.2": _report_paths(0.2, root)[1],
    }
    temp_02 = Path("/private/tmp/hrv_no_motion_threshold_0p2.csv")
    if not candidates["<0.2"].is_file() and temp_02.is_file():
        candidates["<0.2"] = temp_02
    label = f"<{current_threshold:g}"
    candidates[label] = current_csv
    return candidates


def _coverage_col(df: pd.DataFrame) -> str:
    if "coverage_pct_within_no_motion" in df.columns:
        return "coverage_pct_within_no_motion"
    return "coverage_pct"


def _selected_denominator_col(df: pd.DataFrame) -> str:
    if "n_no_motion_windows" in df.columns:
        return "n_no_motion_windows"
    return "n_windows"


def _paired_comparison(reference: pd.DataFrame, candidate: pd.DataFrame, reference_label: str, candidate_label: str) -> pd.DataFrame:
    keys = ["dataset_label", "participant", "device", "hrv_metric"]
    ref_cov = _coverage_col(reference)
    cand_cov = _coverage_col(candidate)
    merged = reference[keys + ["MAE", "R", ref_cov]].merge(
        candidate[keys + ["MAE", "R", cand_cov]],
        on=keys,
        suffixes=("_ref", "_cand"),
    )
    ref_cov_col = f"{ref_cov}_ref" if f"{ref_cov}_ref" in merged.columns else ref_cov
    cand_cov_col = f"{cand_cov}_cand" if f"{cand_cov}_cand" in merged.columns else cand_cov
    rows = []
    for metric in METRICS:
        m = merged[merged["hrv_metric"] == metric]
        mae = m[np.isfinite(m["MAE_ref"]) & np.isfinite(m["MAE_cand"])]
        r = m[np.isfinite(m["R_ref"]) & np.isfinite(m["R_cand"])]
        rows.append(
            {
                "comparison": f"{candidate_label} vs {reference_label}",
                "hrv_metric": metric,
                "paired_mae_rows": len(mae),
                "candidate_lower_mae": int((mae["MAE_cand"] < mae["MAE_ref"]).sum()),
                "paired_r_rows": len(r),
                "candidate_higher_r": int((r["R_cand"] > r["R_ref"]).sum()),
                "coverage_rows": len(m),
                "candidate_higher_coverage": int((m[cand_cov_col] > m[ref_cov_col]).sum()),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_against_full(full: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        full_metric = full[full["hrv_metric"] == metric]
        cand_metric = candidate[candidate["hrv_metric"] == metric]
        for dataset in ("strict_reference", "training_v1_stride30"):
            for device in ("Earring", "Ring", "Watch"):
                f = full_metric[(full_metric["dataset_label"] == dataset) & (full_metric["device"] == device)]
                c = cand_metric[(cand_metric["dataset_label"] == dataset) & (cand_metric["device"] == device)]
                if f.empty or c.empty:
                    continue
                full_denom = f["n_windows"].sum()
                cand_denom = c[_selected_denominator_col(c)].sum()
                rows.append(
                    {
                        "hrv_metric": metric,
                        "dataset_label": dataset,
                        "device": device,
                        "mae_full": float(np.nanmean(f["MAE"])),
                        "mae_candidate": float(np.nanmean(c["MAE"])),
                        "r_full_median": float(np.nanmedian(f["R"])),
                        "r_candidate_median": float(np.nanmedian(c["R"])),
                        "coverage_full_pct": 100.0 * f["n_selected_windows"].sum() / full_denom if full_denom else np.nan,
                        "coverage_candidate_pct": 100.0 * c["n_selected_windows"].sum() / cand_denom if cand_denom else np.nan,
                        "candidate_subset_pct": 100.0 * cand_denom / c["total_windows"].sum() if "total_windows" in c else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _threshold_summary(candidates: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for label, path in candidates.items():
        if label == "full" or not path.is_file():
            continue
        df = pd.read_csv(path)
        rmssd = df[df["hrv_metric"] == "RMSSD"]
        for dataset in ("strict_reference", "training_v1_stride30"):
            one = rmssd[rmssd["dataset_label"] == dataset]
            if one.empty:
                continue
            denom = one[_selected_denominator_col(one)].sum()
            rows.append(
                {
                    "threshold": label,
                    "dataset_label": dataset,
                    "subset_windows": int(denom),
                    "valid_predictions": int(one["n_selected_windows"].sum()),
                    "coverage_pct": 100.0 * one["n_selected_windows"].sum() / denom if denom else np.nan,
                    "mean_rmssd_mae": float(np.nanmean(one["MAE"])),
                    "median_rmssd_r": float(np.nanmedian(one["R"])),
                }
            )
    return pd.DataFrame(rows)


def _comparison_section(df: pd.DataFrame, current_threshold: float, current_csv: Path, output_dir: Path | None = None) -> list[str]:
    candidates = _comparison_candidates(current_threshold, current_csv, output_dir)
    full_path = candidates["full"]
    if not full_path.is_file():
        return ["", "## 比较分析", "", "- 未找到完整窗口报告 CSV，因此跳过 full/no-motion 比较。"]

    full = pd.read_csv(full_path)
    current_label = f"<{current_threshold:g}"
    sections = ["", "## 与 full / <0.1 / <0.2 的比较", ""]
    sections.append(
        f"当前报告的 motion threshold 是 `{current_label}`。这里把它和 full report、严格 no-motion `<0.1`、以及已计算的 `<0.2` 结果做 sensitivity comparison。"
    )

    paired_frames = [_paired_comparison(full, df, "full", current_label)]
    for label in ("<0.1", "<0.2"):
        path = candidates.get(label)
        if path and path.is_file() and path != current_csv:
            paired_frames.append(_paired_comparison(pd.read_csv(path), df, label, current_label))
    paired = pd.concat(paired_frames, ignore_index=True)
    sections.extend(
        [
            "",
            "### 有效配对行统计",
            "",
            _md_table(
                paired,
                [
                    "comparison",
                    "hrv_metric",
                    "paired_mae_rows",
                    "candidate_lower_mae",
                    "paired_r_rows",
                    "candidate_higher_r",
                    "coverage_rows",
                    "candidate_higher_coverage",
                ],
                [
                    "Comparison",
                    "Metric",
                    "Paired MAE rows",
                    "Candidate lower MAE",
                    "Paired R rows",
                    "Candidate higher R",
                    "Coverage rows",
                    "Candidate higher coverage",
                ],
            ),
        ]
    )

    threshold_overview = _threshold_summary(candidates)
    if not threshold_overview.empty:
        sections.extend(
            [
                "",
                "### Threshold 总览（RMSSD）",
                "",
                _md_table(
                    threshold_overview.sort_values(["dataset_label", "threshold"]),
                    ["threshold", "dataset_label", "subset_windows", "valid_predictions", "coverage_pct", "mean_rmssd_mae", "median_rmssd_r"],
                    ["Threshold", "Dataset", "Subset windows", "Valid preds", "Coverage %", "Mean RMSSD MAE", "Median RMSSD R"],
                ),
            ]
        )

    agg = _aggregate_against_full(full, df)
    if not agg.empty:
        for metric in METRICS:
            metric_agg = agg[agg["hrv_metric"] == metric]
            sections.extend(
                [
                    "",
                    f"### Dataset x Device 汇总比较（{metric}, {current_label} vs full）",
                    "",
                    _md_table(
                        metric_agg,
                        [
                            "dataset_label",
                            "device",
                            "mae_full",
                            "mae_candidate",
                            "r_full_median",
                            "r_candidate_median",
                            "coverage_full_pct",
                            "coverage_candidate_pct",
                            "candidate_subset_pct",
                        ],
                        [
                            "Dataset",
                            "Device",
                            "MAE full",
                            f"MAE {current_label}",
                            "R full",
                            f"R {current_label}",
                            "Coverage full",
                            f"Coverage {current_label}",
                            f"Subset % {current_label}",
                        ],
                    ),
                ]
            )

    sections.extend(
        [
            "",
            "解释：threshold 越宽，subset windows 通常越多；但它逐渐从 strict no-motion 变成 low-motion / low-to-moderate-motion sensitivity analysis。Coverage 的分母是 threshold 子集内部窗口数，不能直接等同于完整报告的 overall coverage。",
        ]
    )
    return sections


def main() -> None:
    global NO_MOTION_THRESHOLD

    parser = argparse.ArgumentParser(description="Evaluate current best baseline under a configurable motion threshold.")
    parser.add_argument("--motion-threshold", type=float, default=NO_MOTION_THRESHOLD)
    parser.add_argument("--reuse-csv", action="store_true", help="Skip PPG recomputation and regenerate the Markdown report from the existing CSV.")
    parser.add_argument("--reuse-csv-source", default=None, help="Optional CSV to copy/use before regenerating the Markdown report.")
    parser.add_argument("--output-dir", default=str(PKG_ROOT), help="Directory for the generated MD/CSV reports.")
    args = parser.parse_args()
    NO_MOTION_THRESHOLD = float(args.motion_threshold)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_md, report_csv = _report_paths(NO_MOTION_THRESHOLD, output_dir)

    start = time.time()
    if args.reuse_csv:
        if args.reuse_csv_source:
            source = Path(args.reuse_csv_source).resolve()
            if not source.is_file():
                raise SystemExit(f"--reuse-csv-source not found: {source}")
            df = pd.read_csv(source)
            df.to_csv(report_csv, index=False)
            print(f"[reuse] copied {source} -> {report_csv}")
        if not report_csv.is_file():
            raise SystemExit(f"--reuse-csv requested but not found: {report_csv}")
        print(f"[reuse] loading {report_csv}")
        df = pd.read_csv(report_csv)
    else:
        tasks: list[tuple[str, str, str]] = []
        for spec in DATASETS:
            files = sorted(
                spec["dir"].glob(f"{spec['name']}_P*.npz"),
                key=lambda p: _participant_sort_key(p.stem.split("_")[-1]),
            )
            if not files:
                raise SystemExit(f"No NPZ files found for {spec['dir']}")
            tasks.extend((spec["label"], spec["name"], str(path)) for path in files)

        print(f"[eval no-motion] tasks={len(tasks)}")
        all_rows: list[dict[str, object]] = []
        max_workers = min(4, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_npz, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                rows = future.result()
                all_rows.extend(rows)
                sample = rows[0] if rows else {}
                print(
                    f"[eval no-motion] done {i}/{len(tasks)} "
                    f"{sample.get('dataset_label', '')} {sample.get('participant', '')} rows={len(rows)}",
                    flush=True,
                )

        df = pd.DataFrame(all_rows)
        df = _sort_participant_frame(df).reset_index(drop=True)
        df.to_csv(report_csv, index=False)

    rmssd = df[df["hrv_metric"] == "RMSSD"].copy()
    sdnn = df[df["hrv_metric"] == "SDNN"].copy()
    dataset_device = (
        rmssd.groupby(["dataset_label", "device"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n_participants": g["participant"].nunique(),
                    "total_windows": g["total_windows"].sum(),
                    "n_no_motion_windows": g["n_no_motion_windows"].sum(),
                    "n_selected_windows": g["n_selected_windows"].sum(),
                    "no_motion_pct": 100.0 * g["n_no_motion_windows"].sum() / g["total_windows"].sum(),
                    "coverage_pct_within_no_motion": 100.0 * g["n_selected_windows"].sum() / g["n_no_motion_windows"].sum()
                    if g["n_no_motion_windows"].sum()
                    else np.nan,
                    "RMSSD_MAE_mean_by_participant": g["MAE"].mean(),
                    "RMSSD_R_median_by_participant": g["R"].median(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values(["dataset_label", "device"])
    )

    lines = [
        "# Current Best Heuristic Baseline: No-Motion Strict Reference vs Training Stride30",
        "",
        "本报告只在指定 motion threshold 子集上重新计算当前最佳 raw-aligned baseline。",
        "",
        "## Motion Threshold 定义",
        "",
        f"- 按每个 window、每个 device 独立判断：`accel_motion_mean_mag < {NO_MOTION_THRESHOLD}`。",
        "- `accel_motion_mean_mag` 是数据集内保存的去重力 motion magnitude 均值。",
        "- 过滤发生在 device 层面：同一个 window 对 Earring 可能是 no-motion，对 Ring/Watch 不一定是 no-motion。",
        "",
        "## Baseline 定义",
        "",
        "- 方法：`no_motion_device_best_sqi_green_or_ir__hrv_from_ppg`。",
        "- 对 no-motion 子集中的每个 window、每个 device 独立评估 `ppg_green` 与 `ppg_ir`。",
        "- 使用 `heuristic_baselines.algorithms.hrv.hrv_from_ppg` 从 PPG 重新计算 PRV/HRV。",
        "- 若 green 和 IR 都有效，则仅用 PPG 自身信息选择 SQI 更高的一路；不使用 ECG label 选通道。",
        "",
        "## 数据集",
        "",
        "| Dataset | Total windows | No-motion windows | Interpretation |",
        "|---|---:|---:|---|",
    ]
    for spec in DATASETS:
        one = rmssd[rmssd["dataset_label"] == spec["label"]]
        total_windows = int(one[["participant", "device", "total_windows"]].drop_duplicates()["total_windows"].sum())
        no_motion_windows = int(one[["participant", "device", "n_no_motion_windows"]].drop_duplicates()["n_no_motion_windows"].sum())
        lines.append(f"| `{spec['label']}` | {total_windows} | {no_motion_windows} | {spec['interpretation']} |")

    lines.extend(
        [
            "",
            "## Dataset x Device 汇总（RMSSD）",
            "",
            _md_table(
                dataset_device,
                [
                    "dataset_label",
                    "device",
                    "n_participants",
                    "total_windows",
                    "n_no_motion_windows",
                    "n_selected_windows",
                    "no_motion_pct",
                    "coverage_pct_within_no_motion",
                    "RMSSD_MAE_mean_by_participant",
                    "RMSSD_R_median_by_participant",
                ],
                [
                    "Dataset",
                    "Device",
                    "Participants",
                    "Total windows",
                    "No-motion windows",
                    "Valid preds",
                    "No-motion %",
                    "Coverage within no-motion %",
                    "Mean participant RMSSD MAE",
                    "Median participant R",
                ],
            ),
            "",
            "## 每个参与者 x 每个设备结果（RMSSD）",
            "",
            _md_table(
                _sort_participant_frame(rmssd),
                [
                    "dataset_label",
                    "participant",
                    "device",
                    "total_windows",
                    "n_no_motion_windows",
                    "n_selected_windows",
                    "no_motion_pct",
                    "coverage_pct_within_no_motion",
                    "MAE",
                    "RMSE",
                    "R",
                    "bias",
                    "mean_accel_mean_mag_no_motion",
                    "mean_accel_motion_mean_mag_no_motion",
                    "selected_ppg_green",
                    "selected_ppg_ir",
                ],
                [
                    "Dataset",
                    "Participant",
                    "Device",
                    "Total windows",
                    "No-motion windows",
                    "Valid preds",
                    "No-motion %",
                    "Coverage within no-motion %",
                    "MAE ms",
                    "RMSE ms",
                    "R",
                    "Bias ms",
                    "Raw accel no-motion",
                    "Motion accel no-motion",
                    "Green selected",
                    "IR selected",
                ],
            ),
            "",
            "## 每个参与者 x 每个设备结果（SDNN）",
            "",
            _md_table(
                _sort_participant_frame(sdnn),
                [
                    "dataset_label",
                    "participant",
                    "device",
                    "total_windows",
                    "n_no_motion_windows",
                    "n_selected_windows",
                    "no_motion_pct",
                    "coverage_pct_within_no_motion",
                    "MAE",
                    "RMSE",
                    "R",
                    "bias",
                ],
                [
                    "Dataset",
                    "Participant",
                    "Device",
                    "Total windows",
                    "No-motion windows",
                    "Valid preds",
                    "No-motion %",
                    "Coverage within no-motion %",
                    "MAE ms",
                    "RMSE ms",
                    "R",
                    "Bias ms",
                ],
            ),
            *_comparison_section(df, NO_MOTION_THRESHOLD, report_csv, output_dir),
            "",
            "## 重要解析",
            "",
            "- 这个报告回答的是“在数据集内低运动窗口中，当前 PPG heuristic baseline 表现如何”。",
            "- no-motion 是 per-device 判断，不是 per-window 全设备统一判断，因此同一个参与者的不同设备 no-motion 窗口数可能差很多。",
            "- `No-motion %` 很低时，MAE/R 可能不稳定，因为用于评估的窗口太少。",
            "- `Coverage within no-motion %` 表示 no-motion 子集中有多少窗口能从 green/IR 至少一路得到有效 PRV 估计。",
            "- `R` 至少需要 3 个有效预测点才会计算；少于 3 个时 CSV/MD 中会留空。",
            "- stride30 数据高度重叠，适合观察开发/训练视角，不应当按独立测试样本解释。",
            "",
            "## 输出文件",
            "",
            f"- Markdown report: `{report_md.name}`",
            f"- Machine-readable table: `{report_csv.name}`",
            "",
            f"Generated in {time.time() - start:.1f} seconds.",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {report_md}")
    print(f"[saved] {report_csv}")


if __name__ == "__main__":
    main()
