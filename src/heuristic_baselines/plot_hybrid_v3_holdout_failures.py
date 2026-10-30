"""Plot every missed/extra event for the deployed hybrid-v3 peak strategy."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_assisted_human_confirmed_rpeaks import parse_clip_ranges


METHOD = "deployment_5s_raw_morphology_plus_pan_raw_apex_fill"


def _clip_bounds(window: pd.Series, clip_id: str) -> tuple[float, float]:
    number = int(clip_id.rsplit("_C", 1)[1]) - 1
    left_s, right_s = parse_clip_ranges(str(window.detail_excerpt_seconds))[number]
    start = float(window.window_start_unix_ms)
    return start + left_s * 1000.0, start + right_s * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--half-window-ms", type=float, default=900.0)
    args = parser.parse_args()

    matches = pd.read_csv(args.matches, dtype={"clip_id": str, "participant": str})
    annotations = pd.read_csv(args.annotations, dtype={"review_id": str, "participant": str})
    manifest = pd.read_csv(args.manifest, dtype={"review_id": str, "participant": str})
    annotations["manual_rpeak_abs_unix_ms"] = pd.to_numeric(annotations.manual_rpeak_abs_unix_ms, errors="coerce")
    failures = matches[(matches.candidate_method == METHOD) & matches.match_status.isin(["missed_reference", "extra_candidate"])].copy()
    failures["event_unix_ms"] = failures.reference_unix_ms.fillna(failures.candidate_unix_ms)
    failures = failures.sort_values(["participant", "clip_id", "event_unix_ms", "match_status"]).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failures.to_csv(args.out_dir / "failure_events.csv", index=False)

    raw_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    panels_per_page = 9
    for page_start in range(0, len(failures), panels_per_page):
        page = failures.iloc[page_start:page_start + panels_per_page]
        fig, axes = plt.subplots(3, 3, figsize=(18, 11))
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.86, hspace=0.55, wspace=0.16)
        axes = axes.ravel()
        for axis, (_, failure) in zip(axes, page.iterrows(), strict=False):
            clip_id = str(failure.clip_id)
            review_id = clip_id.rsplit("_C", 1)[0]
            window = manifest.loc[manifest.review_id.eq(review_id)].iloc[0]
            raw_path = str(window.raw_ecg_path)
            if raw_path not in raw_cache:
                with np.load(raw_path, allow_pickle=True) as source:
                    x = np.asarray(source["ecg_uv"], dtype=float)
                    t = np.asarray(source["time_ms"], dtype=float)
                keep = np.isfinite(x) & np.isfinite(t)
                raw_cache[raw_path] = x[keep], t[keep]
            x, t = raw_cache[raw_path]
            event = float(failure.event_unix_ms)
            left, right = event - args.half_window_ms, event + args.half_window_ms
            lo, hi = np.searchsorted(t, left, side="left"), np.searchsorted(t, right, side="right")
            local_t, local_x = t[lo:hi], x[lo:hi]
            clip_left, clip_right = _clip_bounds(window, clip_id)
            reference = annotations.loc[
                annotations.review_id.eq(review_id)
                & annotations.manual_rpeak_abs_unix_ms.between(clip_left, clip_right),
                "manual_rpeak_abs_unix_ms",
            ].dropna().to_numpy(float)
            candidates = matches.loc[
                matches.candidate_method.eq(METHOD) & matches.clip_id.eq(clip_id) & matches.candidate_unix_ms.notna(),
                "candidate_unix_ms",
            ].to_numpy(float)
            axis.plot((local_t - event) / 1000.0, local_x, color="#273746", lw=1.25)
            for values, color, marker, label in (
                (reference, "#2471a3", "o", "human-confirmed"),
                (candidates, "#d68910", "^", "v3 candidate"),
            ):
                values = values[(values >= left) & (values <= right)]
                if values.size:
                    y = np.interp(values, local_t, local_x)
                    axis.scatter((values - event) / 1000.0, y, s=36, marker=marker, color=color, edgecolor="white", linewidth=0.6, zorder=3, label=label)
            is_miss = str(failure.match_status) == "missed_reference"
            accent = "#c0392b" if is_miss else "#e67e22"
            axis.axvline(0, color=accent, lw=2, alpha=0.85)
            title = "MISS: manual peak absent" if is_miss else "EXTRA: no manual peak"
            axis.set_title(f"{clip_id} · {title}", fontsize=10, color=accent)
            axis.set_xlabel("seconds from failure event")
            axis.set_ylabel("raw ECG (µV)")
            axis.grid(alpha=0.22)
        for axis in axes[len(page):]:
            axis.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
        fig.suptitle("Hybrid v3: every holdout missed or extra QRS event", fontsize=16, y=0.985)
        page_number = page_start // panels_per_page + 1
        fig.savefig(args.out_dir / f"hybrid_v3_failure_events_page_{page_number:02d}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {len(failures)} failure events to {args.out_dir}")


if __name__ == "__main__":
    main()
