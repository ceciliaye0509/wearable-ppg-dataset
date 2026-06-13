"""
HRV summary tables + scatter plots across participants and devices.

Reads all ``hrv_*.csv`` and ``ppg_vs_ecg_*_summary.csv`` files, produces:
  - Cross-participant summary table (markdown + CSV)
  - Scatter plots: PPG RMSSD/SDNN vs ECG RMSSD/SDNN per device

Run from this package directory::

    python summarize_hrv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
import config  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available — skipping plots")

from io_utils import normalize_participant_id, participant_device_id  # noqa: E402

DEVICES = list(config.HEURISTIC_DEVICE_ROLES)
CHANNELS = list(config.HEURISTIC_PPG_CHANNELS)
PARTICIPANTS = [normalize_participant_id(p) for p in config.HEURISTIC_PIPELINE_PARTICIPANTS]
RESULT_ROOT = config.HEURISTIC_RESULT_ROOT

DEVICE_COLORS = {"Earring": "#2196F3", "Ring": "#4CAF50", "Necklace": "#FF9800", "Watch": "#9C27B0"}
DEVICE_MARKERS = {"Earring": "o", "Ring": "s", "Necklace": "^", "Watch": "D"}


def collect_hrv_data(channel: str = "ppg_ir") -> pd.DataFrame:
    """Collect all hrv_*.csv into one DataFrame."""
    frames = []
    for pid in PARTICIPANTS:
        result_dir = RESULT_ROOT / pid
        for dev in DEVICES:
            dev_id = participant_device_id(pid, dev)
            csv_path = result_dir / f"hrv_{dev_id}_{channel}.csv"
            if csv_path.is_file():
                df = pd.read_csv(csv_path)
                df["participant"] = pid
                df["device"] = dev
                df["channel"] = channel
                frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def collect_comparison_data(channel: str = "ppg_ir") -> pd.DataFrame:
    """Collect all ppg_vs_ecg_*.csv into one DataFrame."""
    frames = []
    for pid in PARTICIPANTS:
        result_dir = RESULT_ROOT / pid
        for dev in DEVICES:
            dev_id = participant_device_id(pid, dev)
            csv_path = result_dir / f"ppg_vs_ecg_{dev_id}_{channel}.csv"
            if csv_path.is_file():
                df = pd.read_csv(csv_path)
                df["participant"] = pid
                df["device"] = dev
                df["channel"] = channel
                frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def collect_agreement_summaries(channel: str = "ppg_ir") -> pd.DataFrame:
    """Collect all ppg_vs_ecg_*_summary.csv."""
    frames = []
    for pid in PARTICIPANTS:
        result_dir = RESULT_ROOT / pid
        for dev in DEVICES:
            dev_id = participant_device_id(pid, dev)
            csv_path = result_dir / f"ppg_vs_ecg_{dev_id}_{channel}_summary.csv"
            if csv_path.is_file():
                df = pd.read_csv(csv_path)
                df["participant"] = pid
                df["device"] = dev
                frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def print_summary_table(channel: str = "ppg_ir") -> str:
    """Print a markdown summary table of RMSSD and SDNN per participant/device."""
    hrv_df = collect_hrv_data(channel)
    if hrv_df.empty:
        print("[WARN] No HRV data found")
        return ""

    lines = [f"## HRV Summary (channel: {channel})\n"]
    for metric in ["HRV_RMSSD", "HRV_SDNN"]:
        lines.append(f"\n### {metric}\n")
        header = f"| Participant | {'  |  '.join(DEVICES)} |"
        sep = f"|---|{'---|' * len(DEVICES)}"
        lines.append(header)
        lines.append(sep)
        for pid in PARTICIPANTS:
            row_parts = [pid]
            for dev in DEVICES:
                mask = (hrv_df["participant"] == pid) & (hrv_df["device"] == dev)
                vals = hrv_df.loc[mask, metric].dropna()
                if len(vals) > 0:
                    row_parts.append(f"{vals.mean():.2f} ± {vals.std():.2f} (n={len(vals)})")
                else:
                    row_parts.append("—")
            lines.append("| " + " | ".join(row_parts) + " |")
        lines.append("")

    # Add agreement summary
    agr_df = collect_agreement_summaries(channel)
    if not agr_df.empty:
        lines.append("\n## PPG vs ECG Agreement\n")
        for metric in ["HRV_RMSSD", "HRV_SDNN"]:
            lines.append(f"\n### {metric}\n")
            header = "| Participant | Device | r | MAE | Bias | LoA Lo | LoA Hi |"
            sep = "|---|---|---|---|---|---|---|"
            lines.append(header)
            lines.append(sep)
            for pid in PARTICIPANTS:
                for dev in DEVICES:
                    mask = (agr_df["participant"] == pid) & (agr_df["device"] == dev) & (agr_df["metric"] == metric)
                    rows = agr_df.loc[mask]
                    if not rows.empty:
                        r = rows.iloc[0]
                        lines.append(
                            f"| {pid} | {dev} | {r['r']:.4f} | {r['mae']:.2f} | "
                            f"{r['bias']:.2f} | {r['loa_lo']:.2f} | {r['loa_hi']:.2f} |"
                        )

    text = "\n".join(lines)
    summary_path = RESULT_ROOT / f"hrv_summary_{channel}.md"
    summary_path.write_text(text, encoding="utf-8")
    print(f"\n[SAVED] {summary_path}")
    print(text)
    return text


def plot_scatter(channel: str = "ppg_ir") -> None:
    """Scatter plot: PPG HRV vs ECG HRV for RMSSD and SDNN."""
    if not HAS_MPL:
        return

    comp_df = collect_comparison_data(channel)
    if comp_df.empty:
        print("[WARN] No comparison data for scatter plots")
        return

    for metric in ["HRV_RMSSD", "HRV_SDNN"]:
        # Remote eval: PPG columns use metric name directly, ECG uses ecg_ prefix
        ppg_col = metric
        ecg_col = f"ecg_{metric}"
        if ppg_col not in comp_df.columns or ecg_col not in comp_df.columns:
            continue

        n_participants = len(PARTICIPANTS)
        fig, axes = plt.subplots(1, n_participants, figsize=(5 * n_participants, 5),
                                  squeeze=False, sharey=False)

        for j, pid in enumerate(PARTICIPANTS):
            ax = axes[0, j]
            pid_data = comp_df[comp_df["participant"] == pid]

            all_vals = []
            for dev in DEVICES:
                dev_data = pid_data[pid_data["device"] == dev]
                mask = dev_data[ppg_col].notna() & dev_data[ecg_col].notna()
                x = dev_data.loc[mask, ecg_col].values
                y = dev_data.loc[mask, ppg_col].values
                if len(x) > 0:
                    ax.scatter(x, y, c=DEVICE_COLORS.get(dev, "gray"),
                              marker=DEVICE_MARKERS.get(dev, "o"),
                              alpha=0.3, s=10, label=f"{dev} (n={len(x)})")
                    all_vals.extend(x)
                    all_vals.extend(y)

            if all_vals:
                lo = max(0, np.nanpercentile(all_vals, 1) * 0.8)
                hi = np.nanpercentile(all_vals, 99) * 1.2
                ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=0.8, label="y=x")
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)

            ax.set_xlabel(f"ECG {metric} (ms)")
            ax.set_ylabel(f"PPG {metric} (ms)")
            ax.set_title(pid)
            ax.legend(fontsize=7, loc="upper left")
            ax.set_aspect("equal", adjustable="datalim")

        fig.suptitle(f"PPG vs ECG: {metric} ({channel})", fontsize=14)
        fig.tight_layout()
        out_path = RESULT_ROOT / f"scatter_{metric}_{channel}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[SAVED] {out_path}")


def main() -> None:
    for ch in CHANNELS:
        print(f"\n{'#'*60}\n# Channel: {ch}\n{'#'*60}")
        print_summary_table(ch)
        plot_scatter(ch)
    print("\n[summarize] Done.")


if __name__ == "__main__":
    main()
