"""Plot where the selected local Emrich+ProMAC candidate trades off against v3."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


V3 = "deployment_5s_raw_morphology_plus_pan_raw_apex_fill"
BEST = "emrich2023_local_qrs_extreme_plus_promac_local_qrs_extreme_fill"


def _events(matches: pd.DataFrame, set_name: str) -> pd.DataFrame:
    table = matches.copy()
    table["reference_key"] = table.reference_unix_ms.round(3)
    v3 = table[table.candidate_method.eq(V3)]
    best = table[table.candidate_method.eq(BEST)]
    if set_name == "first":
        left = v3[v3.match_status.eq("matched")][["clip_id", "participant", "reference_key", "reference_unix_ms", "candidate_unix_ms"]]
        right = best[best.match_status.eq("missed_reference")][["clip_id", "reference_key"]]
        out = left.merge(right, on=["clip_id", "reference_key"], how="inner")
        out["case"] = "v3 matches; best misses"
    else:
        left = v3[v3.match_status.eq("missed_reference")][["clip_id", "participant", "reference_key", "reference_unix_ms"]]
        right = best[best.match_status.eq("matched")][["clip_id", "reference_key", "candidate_unix_ms"]]
        out = left.merge(right, on=["clip_id", "reference_key"], how="inner")
        out["case"] = "v3 misses; best matches"
    out["set"] = set_name
    return out


def _all_candidates(matches: pd.DataFrame, method: str, clip_id: str) -> np.ndarray:
    return np.unique(matches.loc[
        matches.candidate_method.eq(method) & matches.clip_id.eq(clip_id) & matches.candidate_unix_ms.notna(),
        "candidate_unix_ms",
    ].to_numpy(float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("first", "second"):
        parser.add_argument(f"--{prefix}-matches", type=Path, required=True)
        parser.add_argument(f"--{prefix}-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--half-window-ms", type=float, default=900.0)
    args = parser.parse_args()
    payload = {}
    events = []
    for name in ("first", "second"):
        matches = pd.read_csv(getattr(args, f"{name}_matches"), dtype={"clip_id": str, "participant": str})
        manifest = pd.read_csv(getattr(args, f"{name}_manifest"), dtype={"review_id": str, "participant": str})
        payload[name] = (matches, manifest)
        events.append(_events(matches, name))
    events = pd.concat(events, ignore_index=True).sort_values(["set", "participant", "clip_id", "reference_unix_ms"]).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out_dir / "v3_vs_best_tradeoff_events.csv", index=False)

    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    per_page = 8
    for offset in range(0, len(events), per_page):
        page = events.iloc[offset:offset + per_page]
        fig, axes = plt.subplots(4, 2, figsize=(15, 15))
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.06, top=0.89, hspace=0.55, wspace=0.16)
        axes = axes.ravel()
        for axis, (_, row) in zip(axes, page.iterrows(), strict=False):
            matches, manifest = payload[str(row.set)]
            review_id = str(row.clip_id).rsplit("_C", 1)[0]
            window = manifest.loc[manifest.review_id.eq(review_id)].iloc[0]
            raw_path = str(window.raw_ecg_path)
            if raw_path not in cache:
                with np.load(raw_path, allow_pickle=True) as z:
                    x, t = np.asarray(z["ecg_uv"], dtype=float), np.asarray(z["time_ms"], dtype=float)
                good = np.isfinite(x) & np.isfinite(t)
                cache[raw_path] = x[good], t[good]
            x, t = cache[raw_path]
            reference = float(row.reference_unix_ms)
            left, right = reference - args.half_window_ms, reference + args.half_window_ms
            lo, hi = np.searchsorted(t, left, side="left"), np.searchsorted(t, right, side="right")
            local_t, local_x = t[lo:hi], x[lo:hi]
            axis.plot((local_t - reference) / 1000.0, local_x, color="#273746", lw=1.25)
            markers = (
                (np.array([reference]), "#2471a3", "o", "human-confirmed"),
                (_all_candidates(matches, V3, str(row.clip_id)), "#d68910", "^", "v3"),
                (_all_candidates(matches, BEST, str(row.clip_id)), "#1e8449", "s", "best"),
            )
            for values, color, marker, label in markers:
                values = values[(values >= left) & (values <= right)]
                if values.size:
                    axis.scatter((values - reference) / 1000.0, np.interp(values, local_t, local_x), color=color, marker=marker, s=40, edgecolor="white", linewidth=0.5, zorder=3, label=label)
            accent = "#c0392b" if row.set == "first" else "#1e8449"
            axis.axvline(0, color=accent, lw=2, alpha=.8)
            axis.set_title(f"{row.set}: {row.clip_id} · {row.case}", fontsize=10, color=accent)
            axis.set_xlabel("seconds from human-confirmed R apex")
            axis.set_ylabel("raw ECG (µV)")
            axis.grid(alpha=.22)
        for axis in axes[len(page):]:
            axis.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,.94), ncol=3, frameon=False)
        fig.suptitle("v3 versus selected Emrich + ProMAC local candidate: disagreement events", fontsize=16, y=.985)
        fig.savefig(args.out_dir / f"v3_vs_best_tradeoffs_page_{offset // per_page + 1:02d}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {len(events)} tradeoff events to {args.out_dir}")


if __name__ == "__main__":
    main()
