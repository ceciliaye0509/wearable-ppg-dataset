"""Generate participant/device report for the current best raw-aligned baseline."""
from __future__ import annotations

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
REPORT_MD = PKG_ROOT / "current_best_baseline_strict_vs_training_stride30_by_participant_device.md"
REPORT_CSV = PKG_ROOT / "current_best_baseline_strict_vs_training_stride30_by_participant_device.csv"

DATASETS = [
    {
        "label": "strict_reference",
        "name": "synced_3device_rawaligned_strict_reference_100hz_ecgibi100_sample90_ecgsample100",
        "dir": OUTPUT_ROOT / "synced_3device_rawaligned_strict_reference_100hz_ecgibi100_sample90_ecgsample100",
        "interpretation": "non-overlapping 5 min windows, primary reference-style evaluation",
    },
    {
        "label": "training_v1_stride30",
        "name": "synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100_stride30",
        "dir": OUTPUT_ROOT / "synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100_stride30",
        "interpretation": "30 s stride rolling windows, heavily overlapping development/training view",
    },
]

METRICS = ("RMSSD", "SDNN")


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
        n_windows = int(ppg.shape[0])

        for di, device in enumerate(devices):
            pred = {metric: [] for metric in METRICS}
            ref = {metric: [] for metric in METRICS}
            selected_counts: Counter[str] = Counter()
            reason_counts: Counter[str] = Counter()
            sqi_vals: list[float] = []
            valid_ibi_vals: list[float] = []
            corr_vals: list[float] = []
            n_selected = 0
            n_no_motion = int(np.sum(accel_motion[:, di] < 0.1))

            for wi in range(n_windows):
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
                "method": "device_best_sqi_green_or_ir__hrv_from_ppg",
                "n_windows": n_windows,
                "n_selected_windows": n_selected,
                "coverage_pct": 100.0 * n_selected / n_windows if n_windows else np.nan,
                "mean_accel_mean_mag": float(np.nanmean(accel_raw[:, di])) if n_windows else np.nan,
                "mean_accel_motion_mean_mag": float(np.nanmean(accel_motion[:, di])) if n_windows else np.nan,
                "no_motion_pct_accel_lt_0p1": 100.0 * n_no_motion / n_windows if n_windows else np.nan,
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
            if col in {"MAE", "RMSE", "bias", "mean_accel_mean_mag", "mean_accel_motion_mean_mag", "RMSSD_MAE_mean_by_participant"}:
                vals.append(_fmt(v, 2))
            elif col in {"R", "RMSSD_R_median_by_participant"}:
                vals.append(_fmt(v, 3))
            elif col.endswith("pct") or col == "coverage_pct" or col == "no_motion_pct_accel_lt_0p1":
                vals.append(_fmt(v, 1))
            elif col.startswith("n_") or col.startswith("selected_"):
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


def main() -> None:
    start = time.time()
    tasks: list[tuple[str, str, str]] = []
    for spec in DATASETS:
        files = sorted(
            spec["dir"].glob(f"{spec['name']}_P*.npz"),
            key=lambda p: _participant_sort_key(p.stem.split("_")[-1]),
        )
        if not files:
            raise SystemExit(f"No NPZ files found for {spec['dir']}")
        tasks.extend((spec["label"], spec["name"], str(path)) for path in files)

    print(f"[eval] tasks={len(tasks)}")
    all_rows: list[dict[str, object]] = []
    max_workers = min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_npz, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            all_rows.extend(rows)
            sample = rows[0] if rows else {}
            print(
                f"[eval] done {i}/{len(tasks)} "
                f"{sample.get('dataset_label', '')} {sample.get('participant', '')} rows={len(rows)}",
                flush=True,
            )

    df = pd.DataFrame(all_rows)
    df = _sort_participant_frame(df).reset_index(drop=True)
    df.to_csv(REPORT_CSV, index=False)

    rmssd = df[df["hrv_metric"] == "RMSSD"].copy()
    sdnn = df[df["hrv_metric"] == "SDNN"].copy()
    dataset_device = (
        rmssd.groupby(["dataset_label", "device"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n_participants": g["participant"].nunique(),
                    "n_windows": g["n_windows"].sum(),
                    "n_selected_windows": g["n_selected_windows"].sum(),
                    "coverage_pct": 100.0 * g["n_selected_windows"].sum() / g["n_windows"].sum(),
                    "RMSSD_MAE_mean_by_participant": g["MAE"].mean(),
                    "RMSSD_R_median_by_participant": g["R"].median(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values(["dataset_label", "device"])
    )
    top_strict = (
        rmssd[rmssd["dataset_label"] == "strict_reference"]
        .sort_values(["MAE", "coverage_pct"], ascending=[True, False])
        .head(10)
    )
    worst_strict = (
        rmssd[rmssd["dataset_label"] == "strict_reference"]
        .sort_values(["MAE", "coverage_pct"], ascending=[False, True])
        .head(10)
    )

    lines = [
        "# Current Best Heuristic Baseline: Strict Reference vs Training Stride30",
        "",
        "本报告使用当前 raw-aligned NPZ 重新计算 baseline，不依赖数据集里保存的 PPG-derived metadata。",
        "",
        "## Baseline 定义",
        "",
        "- 方法：`device_best_sqi_green_or_ir__hrv_from_ppg`。",
        "- 每个 window、每个 device 独立评估 `ppg_green` 与 `ppg_ir`。",
        "- 每一路 PPG 使用 `heuristic_baselines.algorithms.hrv.hrv_from_ppg` 计算 PRV/HRV，包含 PPG peak detection、IBI validity、IBI CV、SQI、RMSSD 上限和亚采样 peak refinement。",
        "- 若 green 和 IR 都有效，则只用 PPG 自身信息选择 SQI 更高的一路；不跨设备 fusion，也不使用 ECG label 选通道。",
        "- 运动解释默认使用数据集里的 `accel_motion_mean_mag`，`<0.1` 作为 no-motion 参考阈值；baseline 选择本身没有用 ECG label。",
        "",
        "## 数据集",
        "",
        "| Dataset | Windows | Interpretation |",
        "|---|---:|---|",
    ]
    for spec in DATASETS:
        n_windows = int(
            rmssd.loc[rmssd["dataset_label"] == spec["label"], ["participant", "n_windows"]]
            .drop_duplicates()["n_windows"]
            .sum()
        )
        lines.append(f"| `{spec['label']}` | {n_windows} | {spec['interpretation']} |")

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
                    "n_windows",
                    "n_selected_windows",
                    "coverage_pct",
                    "RMSSD_MAE_mean_by_participant",
                    "RMSSD_R_median_by_participant",
                ],
                [
                    "Dataset",
                    "Device",
                    "Participants",
                    "Windows",
                    "Valid preds",
                    "Coverage %",
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
                    "n_windows",
                    "n_selected_windows",
                    "coverage_pct",
                    "MAE",
                    "RMSE",
                    "R",
                    "bias",
                    "mean_accel_mean_mag",
                    "mean_accel_motion_mean_mag",
                    "no_motion_pct_accel_lt_0p1",
                    "selected_ppg_green",
                    "selected_ppg_ir",
                ],
                [
                    "Dataset",
                    "Participant",
                    "Device",
                    "Windows",
                    "Valid preds",
                    "Coverage %",
                    "MAE ms",
                    "RMSE ms",
                    "R",
                    "Bias ms",
                    "Raw mean accel",
                    "Motion mean accel",
                    "No-motion %",
                    "Green selected",
                    "IR selected",
                ],
            ),
            "",
            "## 每个参与者 x 每个设备结果（SDNN）",
            "",
            _md_table(
                _sort_participant_frame(sdnn),
                ["dataset_label", "participant", "device", "n_windows", "n_selected_windows", "coverage_pct", "MAE", "RMSE", "R", "bias"],
                ["Dataset", "Participant", "Device", "Windows", "Valid preds", "Coverage %", "MAE ms", "RMSE ms", "R", "Bias ms"],
            ),
            "",
            "## 重要解析",
            "",
            "- `strict_reference` 是更适合报告 generalization / evaluation 的表，因为 5 分钟窗口不重叠。",
            "- `training_v1_stride30` 的窗口高度重叠，同一个生理片段会出现多次；它适合训练/开发稳定性观察，不应被当作独立测试样本数。",
            "- Coverage 表示每个 device 的窗口中，有多少窗口能从 green/IR 至少一路得到有效 PRV 估计。Coverage 低通常意味着该 device/channel 在这些窗口里的 peak detection 或 QC 不稳定。",
            "- Bias 是 `PPG-derived - ECG-derived`。负 bias 表示 baseline 系统性低估 ECG HRV。",
            "- `Raw mean accel` 保留原始 magnitude 便于追溯；`Motion mean accel` 和 `No-motion %` 用于解释运动状态。如果某参与者/设备 motion 高且 no-motion 比例低，PPG baseline 表现差更可能是运动伪差导致。",
            "- Green/IR selected counts 展示 device 内部 best-SQI 选择更偏向哪一路；这有助于判断某个设备是否主要依赖某个 wavelength。",
            "",
            "## Strict Reference: RMSSD 最好与最困难的参与者设备",
            "",
            "### 最好 10 组（按 RMSSD MAE）",
            "",
            _md_table(
                top_strict,
                ["participant", "device", "n_windows", "n_selected_windows", "coverage_pct", "MAE", "RMSE", "R", "bias"],
                ["Participant", "Device", "Windows", "Valid preds", "Coverage %", "MAE ms", "RMSE ms", "R", "Bias ms"],
            ),
            "",
            "### 最困难 10 组（按 RMSSD MAE）",
            "",
            _md_table(
                worst_strict,
                ["participant", "device", "n_windows", "n_selected_windows", "coverage_pct", "MAE", "RMSE", "R", "bias"],
                ["Participant", "Device", "Windows", "Valid preds", "Coverage %", "MAE ms", "RMSE ms", "R", "Bias ms"],
            ),
            "",
            "## 输出文件",
            "",
            f"- Markdown report: `{REPORT_MD.name}`",
            f"- Machine-readable table: `{REPORT_CSV.name}`",
            "",
            f"Generated in {time.time() - start:.1f} seconds.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {REPORT_MD}")
    print(f"[saved] {REPORT_CSV}")


if __name__ == "__main__":
    main()
