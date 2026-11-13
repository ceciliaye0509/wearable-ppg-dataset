from __future__ import annotations
from pathlib import Path
import math
import numpy as np
from . import settings as config


def _raw_path(raw_root: Path, pid: str, name: str) -> Path:
    return raw_root / pid / f"{pid}_{name}_raw.npz"

def _load_ppg_raw(raw_root: Path, pid: str, devices: list[str]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for dev in devices:
        path = _raw_path(raw_root, pid, dev)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as z:
            out[dev] = {k: np.asarray(z[k]) for k in z.files}
    return out

def _load_ecg_raw(raw_root: Path, pid: str) -> dict[str, np.ndarray]:
    path = raw_root / pid / f"{pid}_polar_ecg_raw.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}

def _participant_ids(raw_root: Path, raw: str | None) -> list[str]:
    if raw:
        return [config.normalize_participant_id(p) for p in raw.split(",") if p.strip()]
    return sorted((p.name for p in raw_root.iterdir() if p.is_dir()), key=lambda x: int(x[1:]))
