"""Participant-disjoint outer folds and deterministic validation groups."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Fold:
    name: str
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def make_group_folds(
    participants: list[str] | tuple[str, ...], n_folds: int = 4, seed: int = 20260911
) -> list[Fold]:
    """Make outer participant folds; windows never cross train/val/test.

    Validation is a rotating subset of the remaining participants, so scaler,
    early stopping, and threshold selection never see the outer test group.
    """
    participants = sorted(set(participants), key=lambda x: int(x.lstrip("P")))
    if n_folds < 3 or len(participants) < n_folds + 2:
        raise ValueError("Need at least n_folds+2 participants and >=3 folds")
    rng = np.random.default_rng(seed)
    shuffled = [str(value) for value in np.asarray(participants)[rng.permutation(len(participants))]]
    test_groups = [tuple(map(str, x)) for x in np.array_split(shuffled, n_folds)]
    folds: list[Fold] = []
    for index, test in enumerate(test_groups):
        remaining = [p for p in shuffled if p not in test]
        val_count = max(1, round(0.20 * len(remaining)))
        offset = (index * val_count) % len(remaining)
        rotated = remaining[offset:] + remaining[:offset]
        val = tuple(rotated[:val_count])
        train = tuple(p for p in remaining if p not in val)
        folds.append(Fold(f"fold_{index}", train, val, test))
    return folds


def save_folds(folds: list[Fold], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([f.to_dict() for f in folds], indent=2), encoding="utf-8")


def load_folds(path: str | Path) -> list[Fold]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Fold(x["name"], tuple(x["train"]), tuple(x["val"]), tuple(x["test"])) for x in raw]
