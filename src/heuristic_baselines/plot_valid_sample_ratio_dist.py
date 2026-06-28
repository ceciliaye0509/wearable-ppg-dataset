"""
Plot the distribution of ppg_valid_sample_ratio from the rawaligned dataset,
broken down by device and overall, to help decide if the 0.50 threshold is
appropriate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEVICES = ("apple_watch", "garmin", "polar", "samsung")
CHANNELS = ("ppg_green", "ppg_ir")

def main() -> None:
    data_dir = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "outputs/synced_4device_rawaligned_strict_reference"
    )
    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {data_dir}")
        return

    # Collect ratios: overall and per-device
    all_ratios = []                                # flat across everything
    per_device: dict[str, list[np.ndarray]] = {d: [] for d in DEVICES}

    for f in npz_files:
        with np.load(f, allow_pickle=True) as z:
            r = z["ppg_valid_sample_ratio"]        # shape (N, 4, 2)
            all_ratios.append(r.ravel())
            for di, dev in enumerate(DEVICES):
                per_device[dev].append(r[:, di, :].ravel())

    all_ratios = np.concatenate(all_ratios)
    for dev in DEVICES:
        per_device[dev] = np.concatenate(per_device[dev])

    # ---------- Figure 1: overall histogram ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(all_ratios, bins=100, edgecolor="k", alpha=0.8, color="#4c72b0")
    ax.axvline(0.50, color="red", ls="--", lw=1.5, label="threshold = 0.50")
    ax.set_xlabel("ppg_valid_sample_ratio")
    ax.set_ylabel("Count (all device×channel)")
    ax.set_title("Overall Distribution")
    ax.legend()

    # Annotate stats
    below = (all_ratios < 0.50).sum()
    total = all_ratios.size
    txt = (
        f"total = {total}\n"
        f"< 0.50: {below} ({100*below/total:.1f}%)\n"
        f"median = {np.median(all_ratios):.3f}\n"
        f"P5 = {np.percentile(all_ratios, 5):.3f}\n"
        f"P25 = {np.percentile(all_ratios, 25):.3f}"
    )
    ax.text(0.03, 0.95, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

    # ---------- Figure 2: per-device CDF ----------
    ax = axes[1]
    thresholds_to_mark = [0.50, 0.70, 0.85, 0.95]
    for dev in DEVICES:
        vals = np.sort(per_device[dev])
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, label=dev, lw=1.5)
    for thr in thresholds_to_mark:
        ax.axvline(thr, color="gray", ls=":", lw=0.8, alpha=0.6)
        ax.text(thr, 1.02, f"{thr}", ha="center", fontsize=8, color="gray",
                transform=ax.get_xaxis_transform())
    ax.set_xlabel("ppg_valid_sample_ratio")
    ax.set_ylabel("CDF")
    ax.set_title("Per-Device CDF")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = data_dir / "ppg_valid_sample_ratio_distribution.png"
    fig.savefig(out_path, dpi=150)
    print(f"[SAVED] {out_path}")
    plt.close()

    # ---------- Print summary table ----------
    print("\n--- Per-device summary ---")
    print(f"{'device':<15} {'N':>6} {'median':>8} {'P5':>8} {'P25':>8} {'<0.50':>8} {'<0.70':>8} {'<0.85':>8} {'<0.95':>8}")
    for dev in DEVICES:
        v = per_device[dev]
        n = v.size
        print(
            f"{dev:<15} {n:>6} {np.median(v):>8.3f} {np.percentile(v,5):>8.3f} "
            f"{np.percentile(v,25):>8.3f} {(v<0.50).sum():>8} {(v<0.70).sum():>8} "
            f"{(v<0.85).sum():>8} {(v<0.95).sum():>8}"
        )


if __name__ == "__main__":
    main()
