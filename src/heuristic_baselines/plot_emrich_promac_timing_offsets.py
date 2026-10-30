"""Plot every material human-vs-Emrich+ProMAC timing offset on holdout ECG."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HYBRID = "emrich2023_plus_promac_raw_apex_fill"
EMRICH = "emrich2023_raw_apex_refined"
PROMAC = "promac_raw_apex_refined"


def _candidate_times(matches: pd.DataFrame, method: str, clip_id: str) -> np.ndarray:
    values = matches.loc[
        matches.candidate_method.eq(method) & matches.clip_id.eq(clip_id) & matches.candidate_unix_ms.notna(),
        "candidate_unix_ms",
    ].to_numpy(float)
    return np.unique(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold-ms", type=float, default=20.0)
    parser.add_argument("--half-window-ms", type=float, default=320.0)
    args = parser.parse_args()

    matches = pd.read_csv(args.matches, dtype={"clip_id": str, "participant": str})
    manifest = pd.read_csv(args.manifest, dtype={"review_id": str, "participant": str})
    events = matches[(matches.candidate_method == HYBRID) & (matches.match_status == "matched")].copy()
    events["abs_error_ms"] = events.candidate_minus_reference_ms.abs()
    events = events[events.abs_error_ms > args.threshold_ms].sort_values(
        ["participant", "clip_id", "reference_unix_ms"]
    ).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out_dir / "material_timing_offset_events.csv", index=False)

    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    panels_per_page = 9
    for offset in range(0, len(events), panels_per_page):
        page = events.iloc[offset:offset + panels_per_page]
        fig, axes = plt.subplots(3, 3, figsize=(18, 11))
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.86, hspace=0.55, wspace=0.16)
        axes = axes.ravel()
        for axis, (_, row) in zip(axes, page.iterrows(), strict=False):
            review_id = str(row.clip_id).rsplit("_C", 1)[0]
            source = manifest.loc[manifest.review_id.eq(review_id)].iloc[0]
            raw_path = str(source.raw_ecg_path)
            if raw_path not in cache:
                with np.load(raw_path, allow_pickle=True) as z:
                    x, t = np.asarray(z["ecg_uv"], dtype=float), np.asarray(z["time_ms"], dtype=float)
                keep = np.isfinite(x) & np.isfinite(t)
                cache[raw_path] = x[keep], t[keep]
            x, t = cache[raw_path]
            reference = float(row.reference_unix_ms)
            left, right = reference - args.half_window_ms, reference + args.half_window_ms
            lo, hi = np.searchsorted(t, left, side="left"), np.searchsorted(t, right, side="right")
            local_t, local_x = t[lo:hi], x[lo:hi]
            axis.plot((local_t - reference) / 1000.0, local_x, color="#273746", lw=1.35)
            marker_specs = (
                (np.array([reference]), "#2471a3", "o", "human-confirmed"),
                (_candidate_times(matches, HYBRID, str(row.clip_id)), "#d68910", "^", "Emrich + ProMAC"),
                (_candidate_times(matches, EMRICH, str(row.clip_id)), "#1e8449", "x", "Emrich"),
                (_candidate_times(matches, PROMAC, str(row.clip_id)), "#7d3c98", "s", "ProMAC"),
            )
            for values, color, marker, label in marker_specs:
                values = values[(values >= left) & (values <= right)]
                if values.size:
                    y = np.interp(values, local_t, local_x)
                    axis.scatter((values - reference) / 1000.0, y, s=46, color=color, marker=marker, linewidth=1.4, zorder=3, label=label)
            delta = float(row.candidate_minus_reference_ms)
            axis.axvline(delta / 1000.0, color="#d68910", lw=1.5, alpha=0.75)
            axis.axvline(0, color="#2471a3", lw=1.5, alpha=0.75)
            axis.set_title(f"{row.clip_id} · candidate − human = {delta:+.1f} ms", fontsize=10)
            axis.set_xlabel("seconds from human-confirmed apex")
            axis.set_ylabel("raw ECG (µV)")
            axis.grid(alpha=0.22)
        for axis in axes[len(page):]:
            axis.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=4, frameon=False)
        fig.suptitle(f"Emrich + ProMAC: all |timing error| > {args.threshold_ms:g} ms", fontsize=16, y=0.985)
        page_number = offset // panels_per_page + 1
        fig.savefig(args.out_dir / f"emrich_promac_timing_offsets_page_{page_number:02d}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {len(events)} timing-offset events to {args.out_dir}")


if __name__ == "__main__":
    main()
