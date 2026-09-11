"""Participant-balanced batches containing matched single-device examples."""

from __future__ import annotations

from collections import defaultdict
import random

from torch.utils.data import Sampler

from .dataset import SampleRef


class ParticipantDeviceBatchSampler(Sampler[list[int]]):
    """Sample windows uniformly by participant, then expand to three devices.

    Every dataset item remains one device.  Keeping the matched Earring/Ring/
    Watch items in the same batch makes multi-view consistency measurable while
    avoiding an unrealistic multi-device deployment input.
    """

    def __init__(
        self,
        refs: tuple[SampleRef, ...] | list[SampleRef],
        windows_per_batch: int,
        batches_per_epoch: int,
        seed: int = 20260911,
    ) -> None:
        self.windows_per_batch = windows_per_batch
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, ref in enumerate(refs):
            grouped[(ref.participant, ref.group_id)].append(index)
        self.by_participant: dict[str, list[list[int]]] = defaultdict(list)
        for (participant, _), indices in grouped.items():
            self.by_participant[participant].append(sorted(indices))
        self.participants = sorted(self.by_participant)
        if not self.participants:
            raise ValueError("Cannot build sampler for an empty dataset")
        self._epoch = 0

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            for _ in range(self.windows_per_batch):
                participant = rng.choice(self.participants)
                batch.extend(rng.choice(self.by_participant[participant]))
            rng.shuffle(batch)
            yield batch
