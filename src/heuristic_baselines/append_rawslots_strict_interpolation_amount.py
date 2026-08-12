"""Append auditable strict-interpolation amounts to frozen rawslots results."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_rawslots_baseline_ablation import _dataset_npz_files


def _one_channel_stats(arrays: dict[str, np.ndarray], wi: int, di: int, ci: int, max_gap_ms: float) -> dict[str, object]:
    grid = np.asarray(arrays["ppg_grid_timestamp_ms"][wi], dtype=np.float64)
    values = np.asarray(arrays["ppg_rawslot_values"][wi, di, ci], dtype=np.float64)
    times = np.asarray(arrays["ppg_rawslot_timestamp_ms"][wi, di, ci], dtype=np.float64)
    rawslot_mask = np.asarray(arrays["ppg_rawslot_mask"][wi, di, ci], dtype=bool)
    raw = rawslot_mask & np.isfinite(values) & np.isfinite(times)
    n_total_grid = int(grid.size)
    if int(raw.sum()) < 2 or n_total_grid < 50:
        return {
            "strict_input_pass": False, "strict_grid_points": 0, "rawslot_observed_grid_points": 0,
            "interpolated_grid_points": 0, "rawslot_observed_grid_ratio": np.nan,
            "interpolated_grid_ratio": np.nan, "strict_grid_coverage_over_full_window_pct": 0.0,
            "max_raw_support_gap_ms": np.inf,
        }

    raw_t = times[raw]
    order = np.argsort(raw_t)
    raw_t = raw_t[order]
    raw_t = raw_t[np.r_[True, np.diff(raw_t) > 0]]
    if raw_t.size < 2:
        return {
            "strict_input_pass": False, "strict_grid_points": 0, "rawslot_observed_grid_points": 0,
            "interpolated_grid_points": 0, "rawslot_observed_grid_ratio": np.nan,
            "interpolated_grid_ratio": np.nan, "strict_grid_coverage_over_full_window_pct": 0.0,
            "max_raw_support_gap_ms": np.inf,
        }

    max_support = float(max(np.max(np.diff(raw_t)), max(0.0, raw_t[0] - grid[0]), max(0.0, grid[-1] - raw_t[-1])))
    inside = (grid >= raw_t[0]) & (grid <= raw_t[-1])
    n_strict_grid = int(inside.sum())
    passed = bool(np.isfinite(max_support) and max_support <= max_gap_ms and n_strict_grid >= 50)
    if not passed:
        return {
            "strict_input_pass": False, "strict_grid_points": 0, "rawslot_observed_grid_points": 0,
            "interpolated_grid_points": 0, "rawslot_observed_grid_ratio": np.nan,
            "interpolated_grid_ratio": np.nan, "strict_grid_coverage_over_full_window_pct": 0.0,
            "max_raw_support_gap_ms": max_support,
        }

    # A "rawslot-observed" grid point has a raw sample assigned within the
    # producer's 5-ms slot tolerance. The complement is what the strict linear
    # interpolation supplies at the exact 100-Hz grid time.
    n_observed = int((rawslot_mask & inside).sum())
    n_interpolated = int(n_strict_grid - n_observed)
    return {
        "strict_input_pass": True,
        "strict_grid_points": n_strict_grid,
        "rawslot_observed_grid_points": n_observed,
        "interpolated_grid_points": n_interpolated,
        "rawslot_observed_grid_ratio": float(n_observed / n_strict_grid),
        "interpolated_grid_ratio": float(n_interpolated / n_strict_grid),
        "strict_grid_coverage_over_full_window_pct": float(100.0 * n_strict_grid / n_total_grid),
        "max_raw_support_gap_ms": max_support,
    }


def _collect(dataset_dir: Path, max_gap_ms: float, participants: tuple[str, ...] | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in _dataset_npz_files(dataset_dir):
        participant_id = path.stem.rsplit("_", 1)[-1].upper()
        if participants is not None and participant_id not in set(participants):
            continue
        with np.load(path, allow_pickle=True) as z:
            participant = str(np.asarray(z["participant"]).item())
            devices = [str(x) for x in np.asarray(z["devices"]).tolist()]
            channels = [str(x) for x in np.asarray(z["channels"]).tolist()]
            arrays = {
                "ppg_grid_timestamp_ms": np.asarray(z["ppg_grid_timestamp_ms"], dtype=np.float64),
                "ppg_rawslot_values": np.asarray(z["ppg_rawslot_values"], dtype=np.float32),
                "ppg_rawslot_timestamp_ms": np.asarray(z["ppg_rawslot_timestamp_ms"], dtype=np.float64),
                "ppg_rawslot_mask": np.asarray(z["ppg_rawslot_mask"], dtype=bool),
            }
            n_windows = arrays["ppg_grid_timestamp_ms"].shape[0]
            for wi in range(n_windows):
                for di, device in enumerate(devices):
                    for ci, channel in enumerate(channels):
                        rows.append({
                            "participant": participant,
                            "window_index": wi,
                            "device": device,
                            "channel": channel,
                            **_one_channel_stats(arrays, wi, di, ci, max_gap_ms),
                        })
        print(f"[strict-interpolation-amount] {participant} windows={n_windows}")
    return pd.DataFrame(rows)


def _summary(stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in stats.groupby(["device", "channel"], dropna=False):
        passed = group[group["strict_input_pass"]]
        total_grid = int(group.shape[0] * 30000)
        strict_grid = int(passed["strict_grid_points"].sum())
        observed = int(passed["rawslot_observed_grid_points"].sum())
        interpolated = int(passed["interpolated_grid_points"].sum())
        rows.append({
            "device": keys[0], "channel": keys[1], "n_total_windows": int(group.shape[0]),
            "n_strict_input_pass": int(passed.shape[0]),
            "strict_input_coverage_pct": 100.0 * passed.shape[0] / group.shape[0],
            "strict_grid_points": strict_grid,
            "rawslot_observed_grid_points": observed,
            "interpolated_grid_points": interpolated,
            "rawslot_observed_grid_ratio_within_strict_input_pct": 100.0 * observed / strict_grid if strict_grid else np.nan,
            "interpolated_grid_ratio_within_strict_input_pct": 100.0 * interpolated / strict_grid if strict_grid else np.nan,
            "strict_grid_points_over_all_candidate_grid_pct": 100.0 * strict_grid / total_grid if total_grid else np.nan,
        })
    order = {"Earring": 0, "Ring": 1, "Watch": 2}
    channels = {"ppg_green": 0, "ppg_ir": 1}
    return pd.DataFrame(rows).sort_values(["device", "channel"], key=lambda s: s.map(order if s.name == "device" else channels)).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append strict interpolation amounts to frozen rawslots metrics.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-gap-ms", type=float, default=100.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = _collect(args.dataset_dir, args.max_gap_ms)
    keys = ["participant", "window_index", "device", "channel"]
    metrics = pd.read_csv(args.metrics_csv)
    merged = metrics.drop(columns=[c for c in stats.columns if c in metrics.columns and c not in keys]).merge(stats, on=keys, how="left", validate="many_to_one")
    if merged["strict_input_pass"].isna().any():
        raise RuntimeError("failed to join interpolation statistics to every metric row")
    merged.to_csv(args.metrics_csv, index=False)
    stats.to_csv(args.out_dir / "strict_interpolation_amount_by_window.csv", index=False)
    _summary(stats).to_csv(args.out_dir / "strict_interpolation_amount_by_channel.csv", index=False)


if __name__ == "__main__":
    main()
