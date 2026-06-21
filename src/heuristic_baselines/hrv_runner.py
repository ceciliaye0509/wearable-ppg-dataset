"""
HRV / PRV batch runner (NeuroKit2).

Mirrors ``runner.py`` but, instead of one HR per window, computes a full set of
NeuroKit2 HRV indices per window. Reuses ``config.py``, ``io_utils.py`` and the
cached ``*_preprocess.npz`` from ``runner.py`` / ``preprocess.py``.

With the planned 5-min windows, each window is a standard short-term HRV
recording, so one row per window is the proper analysis unit. Output per
device+channel under ``outputs/<Px>/``::

    hrv_<dev>_<channel>.csv   # one row per window: time + frequency + Poincaré

Run from this package directory::

    python hrv_runner.py
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

from algorithms import hrv  # noqa: E402
from io_utils import (  # noqa: E402
    merged_windows_npz,
    normalize_participant_id,
    participant_device_id,
    ppg_from_npz,
)
from preprocess import write_preprocess_npz  # noqa: E402

HRV_COMPUTE_FREQ: bool = bool(getattr(config, "HRV_COMPUTE_FREQ", True))
HRV_COMPUTE_NONLINEAR: bool = bool(getattr(config, "HRV_COMPUTE_NONLINEAR", True))
MOTION_QC_ENABLED: bool = bool(getattr(config, "MOTION_QC_ENABLED", True))
MOTION_SEG_SEC: float = float(getattr(config, "MOTION_SEG_SEC", 10.0))
MOTION_PERCENTILE: float = float(getattr(config, "MOTION_PERCENTILE", 75.0))
MOTION_MAX_FRACTION: float = float(getattr(config, "MOTION_MAX_FRACTION", 0.50))


# Step 9: Earring skips bandpass — signal is clean, bandpass hurts precision.
_SKIP_BANDPASS_ROLES = {"Earring"}


def _motion_fraction_per_window(
    data: dict,
    *,
    seg_sec: float = MOTION_SEG_SEC,
    percentile: float = MOTION_PERCENTILE,
) -> tuple[np.ndarray, float]:
    """Per-window fraction of high-motion 10s chunks from accelerometer."""
    if not all(k in data for k in ("accel_x", "accel_y", "accel_z")):
        n = len(np.asarray(data.get("t0_ms", [])))
        return np.full(n, np.nan), float("nan")

    ax = np.asarray(data["accel_x"], dtype=np.float64)
    ay = np.asarray(data["accel_y"], dtype=np.float64)
    az = np.asarray(data["accel_z"], dtype=np.float64)
    fs = float(np.asarray(data["ppg_fs"]).item()) if "ppg_fs" in data else 100.0
    seg_n = max(1, int(round(seg_sec * fs)))

    all_stds: list[float] = []
    per_win: list[np.ndarray] = []
    for i in range(ax.shape[0]):
        mag = np.sqrt(ax[i] ** 2 + ay[i] ** 2 + az[i] ** 2)
        n_segs = len(mag) // seg_n
        if n_segs == 0:
            stds = np.array([float(np.std(mag))])
        else:
            stds = np.array([
                float(np.std(mag[j * seg_n:(j + 1) * seg_n]))
                for j in range(n_segs)
            ])
        per_win.append(stds)
        all_stds.extend(stds.tolist())

    if not all_stds:
        return np.zeros(ax.shape[0]), float("nan")

    threshold = float(np.percentile(all_stds, percentile))
    fractions = np.array([float(np.mean(stds > threshold)) for stds in per_win])
    return fractions, threshold


def _invalidate_hrv_metrics(row: dict, *, reason: str) -> None:
    """Mark HRV unavailable while preserving QC diagnostics."""
    for col in hrv.hrv_columns(HRV_COMPUTE_FREQ, HRV_COMPUTE_NONLINEAR):
        if col != "n_peaks":
            row[col] = float("nan")
    existing = str(row.get("ppg_qc_reason", "ok"))
    row["ppg_qc_reason"] = reason if existing == "ok" else f"{existing};{reason}"


def _load_windows(raw_npz: Path, result_dir: Path, run_preprocess: bool, role: str = "") -> dict:
    import os
    baseline_mode = os.environ.get("BASELINE", "") == "1"
    # Step 9: skip bandpass for Earring (unless in baseline mode)
    skip_bp = (not baseline_mode) and (role in _SKIP_BANDPASS_ROLES)
    pre_npz = result_dir / f"{raw_npz.stem}_preprocess.npz"
    if run_preprocess and not skip_bp:
        if not pre_npz.is_file():
            print(f"  [preprocess] writing {pre_npz.name} -> {result_dir}")
            write_preprocess_npz(raw_npz, pre_npz)
        load_path = pre_npz
    else:
        if skip_bp:
            print(f"  [Step 9] Skipping bandpass for {role} — using raw signal")
        load_path = raw_npz
    with np.load(load_path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def run_one_device_channel(
    *, raw_npz: Path, result_dir: Path, device_id: str, ppg_channel: str,
    run_preprocess: bool, role: str = ""
) -> None:
    data = _load_windows(raw_npz, result_dir, run_preprocess, role=role)
    t0_ms = np.asarray(data.get("t0_ms", []), dtype=np.float64)
    hr_gt = np.asarray(data.get("hr_gt", []), dtype=np.float64)
    fs = float(np.asarray(data["ppg_fs"]).item()) if "ppg_fs" in data else 100.0
    ppg = np.asarray(ppg_from_npz(data, ppg_channel), dtype=np.float64)
    n = int(ppg.shape[0])
    result_dir.mkdir(parents=True, exist_ok=True)
    motion_fraction, motion_threshold = _motion_fraction_per_window(data)

    rows = []
    for i in range(n):
        m = hrv.hrv_from_ppg(
            ppg[i], fs, freq=HRV_COMPUTE_FREQ, nonlinear=HRV_COMPUTE_NONLINEAR
        )
        row = {"t0_ms": float(t0_ms[i]) if i < len(t0_ms) else np.nan}
        if i < len(hr_gt):
            row["hr_gt"] = float(hr_gt[i])
        row.update(m)
        mf = float(motion_fraction[i]) if i < len(motion_fraction) else float("nan")
        row["motion_fraction"] = mf
        row["motion_threshold"] = motion_threshold
        row["motion_gate"] = bool(
            MOTION_QC_ENABLED
            and np.isfinite(mf)
            and mf >= MOTION_MAX_FRACTION
        )
        if row["motion_gate"]:
            _invalidate_hrv_metrics(row, reason="motion_artifact")
        rows.append(row)

    cols = ["t0_ms"]
    if len(hr_gt):
        cols.append("hr_gt")
    cols += hrv.hrv_columns(HRV_COMPUTE_FREQ, HRV_COMPUTE_NONLINEAR)
    cols += list(hrv.QC_COLS)
    cols += ["motion_fraction", "motion_threshold", "motion_gate"]
    df = pd.DataFrame(rows, columns=cols)

    out_path = result_dir / f"hrv_{device_id}_{ppg_channel}.csv"
    df.to_csv(out_path, index=False)
    valid = int(np.sum(df["HRV_RMSSD"].notna())) if "HRV_RMSSD" in df else 0
    print(f"  [SAVED] {out_path.name}  (windows with valid RMSSD: {valid}/{n})")


def main() -> None:
    root = config.HEURISTIC_WINDOWS_ROOT
    participants = [normalize_participant_id(p) for p in config.HEURISTIC_PIPELINE_PARTICIPANTS]
    roles = list(config.HEURISTIC_DEVICE_ROLES)
    _ch_cfg = config.HEURISTIC_PPG_CHANNELS
    channels = [_ch_cfg] if isinstance(_ch_cfg, str) else list(_ch_cfg)
    result_root = config.HEURISTIC_RESULT_ROOT

    if not root.is_dir():
        print(f"[ERROR] Window NPZ root does not exist: {root}")
        print('  Set HEURISTIC_DATA_SOURCE to "full" or "sample".')
        sys.exit(1)
    if hrv.nk is None:
        print("[hrv] WARNING: neurokit2 not installed -> SciPy time-domain fallback "
              "(no frequency/nonlinear). Install with: pip install neurokit2")

    print(f"[hrv] data_source={config.HEURISTIC_DATA_SOURCE!r} windows_npz_root={root}")
    print(f"[hrv] participants={participants} roles={roles} channels={channels}")
    print(
        f"[hrv] preprocess={config.HEURISTIC_RUN_PREPROCESS} "
        f"freq={HRV_COMPUTE_FREQ} nonlinear={HRV_COMPUTE_NONLINEAR} "
        f"backend={'neurokit2' if hrv.nk is not None else 'scipy-fallback'}"
    )

    for pid in participants:
        result_dir = (result_root / pid).resolve()
        result_dir.mkdir(parents=True, exist_ok=True)
        for role in roles:
            raw = merged_windows_npz(root, pid, role)
            dev_id = participant_device_id(pid, role)
            if not raw.is_file():
                print(f"\n[SKIP] missing {raw}")
                continue
            print(f"\n{'='*60}\n  {dev_id}\n  npz={raw.name}\n  out -> {result_dir}\n{'='*60}")
            for ch in channels:
                if ch not in ("ppg_ir", "ppg_green"):
                    print(f"  [SKIP] unknown channel {ch}")
                    continue
                print(f"\n  --- channel {ch} ---")
                run_one_device_channel(
                    raw_npz=raw,
                    result_dir=result_dir,
                    device_id=dev_id,
                    ppg_channel=ch,
                    run_preprocess=config.HEURISTIC_RUN_PREPROCESS,
                    role=role,
                )

    print("\n[hrv] Done.")


if __name__ == "__main__":
    main()
