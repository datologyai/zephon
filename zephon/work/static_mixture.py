# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Mixture-aware work source with shard-respecting traversal."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping

from zephon.core.constants import SampleId
from zephon.io.dataset import Dataset
from zephon.work.base import WorkChunk, WorkSource
from zephon.work.mixture import MixtureSpec

_GOLDEN_RATIO_64 = 0x9E3779B97F4A7C15  # Used to decorrelate derived seeds.


@dataclass(frozen=True)
class _DatasetKnobs:
    seed: int
    shuffle_shards: bool
    shuffle_within_shard: bool
    shuffle_block_size: int | None


class _DatasetCursor:
    """Iterator over a dataset respecting shard + block shuffles."""

    def __init__(
        self,
        dataset_id: int,
        shard_index: Mapping[int, int],
        knobs: _DatasetKnobs,
    ) -> None:
        shard_ids = list(shard_index.keys())
        # Shuffle the shards
        if knobs.shuffle_shards and len(shard_ids) > 1:
            random.Random(knobs.seed).shuffle(shard_ids)

        sequence: list[SampleId] = []
        for position, shard_id in enumerate(shard_ids):
            count = int(shard_index[shard_id])
            offsets = list(range(count))
            if knobs.shuffle_within_shard and count > 1:  # Shuffle within the shards
                shard_seed = (knobs.seed << 32) ^ ((position << 16) + shard_id)
                random.Random(shard_seed).shuffle(offsets)
            # Construct global sequence
            sequence.extend((dataset_id, shard_id, offset) for offset in offsets)

        block_size = knobs.shuffle_block_size
        # Mosaic-style block-based shuffle on top to create cross-shard shuffles
        if block_size is not None and block_size > 0 and len(sequence) > 1:
            block_size = max(1, int(block_size))
            rng = random.Random(knobs.seed ^ _GOLDEN_RATIO_64)
            # TODO: consider numpy.random.Generator.shuffle for large blocks if this
            # becomes a hotspot; NumPy performs the shuffle in C and can be faster.
            for start in range(0, len(sequence), block_size):
                end = min(start + block_size, len(sequence))
                block = sequence[start:end]
                rng.shuffle(block)
                sequence[start:end] = block

        self._order = sequence
        self._position = 0
        self.remaining = len(sequence)

    def next_many(self, limit: int) -> list[SampleId]:
        if limit <= 0 or self._position >= len(self._order):
            return []
        end = min(self._position + limit, len(self._order))
        chunk = self._order[self._position : end]
        self._position = end
        self.remaining -= len(chunk)
        return chunk


class StaticMixtureWorkSource(WorkSource):
    """Emit SampleId triples from one or more datasets according to a mixture.

    This source orchestrates ordering only. It builds per-dataset cursors from
    the provided ``Dataset`` descriptors and applies shuffling at three levels:
    - shard order within a dataset (``shuffle_shards``)
    - sample order within each shard (``shuffle_within_shard``)
    - optional block-based cross-shard shuffle (``shuffle_block_size``)

    No IO is performed here; fetching is handled by FetchOp which constructs an
    internal shard store using the dataset descriptors exposed via
    ``datasets_by_id``.

    The emitted identifiers have the shape ``(dataset_id, shard_id, sample_idx)``
    where ``dataset_id`` is the position of the dataset in the constructor list.
    """

    def __init__(
        self,
        datasets: list[Dataset],
        mixture: Mapping[str, float],
        *,
        chunk_size: int = 1024,
        seed: int = 0,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = False,
        shuffle_block_size: int | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not datasets:
            raise ValueError("At least one dataset must be provided")
        self._datasets = list(datasets)
        dataset_names = [ds.name for ds in self._datasets]
        mixture_spec = MixtureSpec(mixture)
        mixture_spec.validate_for(dataset_names)
        normalized = mixture_spec.normalized_for(dataset_names)

        self._dataset_ids: dict[str, int] = {}
        self._datasets_by_id: dict[int, Dataset] = {}
        self._cursors: dict[str, _DatasetCursor] = {}
        self._weights: dict[str, float] = {}
        knobs = _DatasetKnobs(
            seed=seed,
            shuffle_shards=shuffle_shards,
            shuffle_within_shard=shuffle_within_shard,
            shuffle_block_size=shuffle_block_size,
        )

        for dataset_id, ds in enumerate(self._datasets):
            self._dataset_ids[ds.name] = dataset_id
            self._datasets_by_id[dataset_id] = ds

            cursor = _DatasetCursor(dataset_id, ds.shard_index, knobs)

            # Should not really happen but can sanity check anyways
            if cursor.remaining <= 0:
                continue
            weight = normalized.get(ds.name)
            if weight is None or weight <= 0.0:
                continue

            self._cursors[ds.name] = cursor
            self._weights[ds.name] = weight

        if not self._weights:
            raise ValueError("No datasets with positive weight and non-empty cursors")

        self._current: dict[str, float] = dict.fromkeys(self._weights, 0.0)
        self._order: list[str] = [
            ds.name for ds in self._datasets if ds.name in self._weights
        ]
        self._total_weight = sum(self._weights.values())
        self._chunk_size = chunk_size
        self._seed = seed
        self._remaining = sum(cursor.remaining for cursor in self._cursors.values())

    @property
    def datasets_by_id(self) -> Mapping[int, Dataset]:
        """Mapping of dataset_id to its Dataset descriptor.

        The engine forwards this mapping to FetchOp so it can construct a
        dataset-aware shard store internally. The descriptors themselves do not
        perform IO.
        """
        return dict(self._datasets_by_id)

    @property
    def dataset_ids(self) -> Mapping[str, int]:
        """Mapping of dataset name to the stable dataset_id used in SampleIds."""
        return dict(self._dataset_ids)

    def _prune_exhausted(self) -> None:
        removed = False
        for name in list(self._weights.keys()):
            if self._cursors[name].remaining <= 0:
                self._weights.pop(name, None)
                self._current.pop(name, None)
                removed = True
        if removed:
            self._order = [name for name in self._order if name in self._weights]
            self._total_weight = sum(self._weights.values())

    def _choose_component(
        self,
    ) -> str | None:
        # TODO(MaxiBoether): Refactor this together with next_chunk.
        if not self._weights:
            return None
        import math

        chosen_name: str | None = None
        chosen_value = -math.inf
        for name in self._order:
            if name not in self._weights:
                continue
            value = self._current[name] + self._weights[name]
            self._current[name] = value
            if value > chosen_value:
                chosen_name = name
                chosen_value = value
        if chosen_name is None:
            return None
        self._current[chosen_name] -= self._total_weight
        return chosen_name

    # TODO(MaxiBoether): next step together with elastic deterrminism is to have next_chunk_for() instead (canonical node ID, worker id).
    # withot a server, this probably literally just computes the same things everywhere and discards, with potentially shared memory optimizations as in mosaic on the same mnode.
    # For ADO for example, this would literally just forward to next_chunk. To guarantee elastic determinism it is task of the nodes that emulate canoncial nodes to fetch 2 chunks in advance, then we should even in ADO fundamentally get elastic determinism (I think)
    def next_chunk(self) -> WorkChunk | None:
        # TODO(MaxiBoether): This is not correct. Right now we construct a chunk that contains items only from a single dataset.
        # we should instead fill it up mixture-correct.
        self._prune_exhausted()
        name = self._choose_component()
        if name is None:
            return None
        cursor = self._cursors[name]
        items = cursor.next_many(self._chunk_size)
        if not items:
            self._prune_exhausted()
            name = self._choose_component()
            if name is None:
                return None
            cursor = self._cursors[name]
            items = cursor.next_many(self._chunk_size)
            if not items:
                return None
        self._remaining -= len(items)
        return WorkChunk(
            components={name: items},
            seed=self._seed,
            explicit_mixture={name: 1.0},
        )

    def checkpoint(self) -> bytes:
        # TODO: capture cursor positions and scheduler state when persistence matters.
        return b""

    def restore(self, state: bytes) -> None:
        # TODO: restore checkpointed state once persistence format is defined.
        return None

    def supports_indexing(self) -> bool:
        return False

    def __len__(self) -> int:
        return self._remaining  # should len really be dynamic?

    def sample_id_at(self, index: int) -> SampleId:
        raise NotImplementedError("StaticMixtureWorkSource is not indexable")
