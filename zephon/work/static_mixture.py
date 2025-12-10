# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Mixture-aware work source with shard-respecting traversal."""

import math
import random
import warnings
from dataclasses import dataclass
from typing import Any, Mapping

from zephon.core.constants import SampleId
from zephon.io.dataset import Dataset
from zephon.utils import disable_gc
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
    """Iterator over a single dataset with shard/offset/block shuffles.

    What gets shuffled (and in which order):
    - Shards: if ``shuffle_shards`` is True, the list of shard IDs is
      permuted using ``seed``.
    - Within-shard offsets: if ``shuffle_within_shard`` is True, sample
      offsets inside each shard are permuted with a per-shard seed derived
      from ``seed``, the shard's position, and ``shard_id``.
    - Block mosaic: if ``shuffle_block_size`` is provided (> 0), the final
      per-dataset sequence is split into contiguous blocks of that size and
      each block is shuffled independently. This creates cross-shard
      permutations while remaining cache-friendly.

    Important: This cursor only handles shuffling within a single dataset.
    Cross-dataset mixing/merging happens later when a ``WorkChunk`` is read
    by the engine (see ``WorkChunk.iter_samples``). In other words, each
    dataset contributes its own internally shuffled sequence; the reader then
    interleaves sequences from multiple datasets according to the chosen
    mixture read mode and weights.

    Determinism and seeding:
    - Shard order uses ``seed`` directly.
    - Within-shard order uses ``shard_seed = (seed << 32) ^ ((position << 16)
      + shard_id)`` so different shards and positions decorrelate.
    - Block shuffles use ``seed ^ _GOLDEN_RATIO_64`` to decorrelate from the
      previous steps while remaining deterministic for a given seed.

    Sketch (one dataset):
      Without any shuffles (default order):
        [ S0: 0 1 2 | S1: 0 1 2 | S2: 0 1 2 ]

      shuffle_shards=True (shuffled shard order; offsets stay in-order):
        [ S2: 0 1 2 | S0: 0 1 2 | S1: 0 1 2 ]

      shuffle_within_shard=True (offsets permuted per shard; shard order as above):
        [ S2: 2 0 1 | S0: 1 2 0 | S1: 0 2 1 ]

      shuffle_block_size=4 (block-wise mosaic on the final flattened sequence):
        Flatten the above per-shard view to sample ids (Sx,offset):
        seq : [ (S2,2) (S2,0) (S2,1) (S0,1) | (S0,2) (S0,0) (S1,0) (S1,2) | (S1,1) ]
        perm: [ (S2,0) (S0,1) (S2,2) (S2,1) | (S1,2) (S0,2) (S1,0) (S0,0) | (S1,1) ]
              (each block is shuffled independently)

    Cross-dataset view (handled when reading a chunk, not here):
      Given per-dataset sequences, the engine merges them when iterating a
      ``WorkChunk`` using either weighted round-robin or weighted random.
      This is where samples from different datasets interleave.
    """

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
        with disable_gc():  # see https://github.com/python/cpython/issues/142531
            for position, shard_id in enumerate(shard_ids):
                count = int(shard_index[shard_id])
                offsets = list(range(count))
                if (
                    knobs.shuffle_within_shard and count > 1
                ):  # Shuffle within the shards
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
        mixture: MixtureSpec | Mapping[str, float],
        chunk_size: int = 1024,
        seed: int = 0,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = False,
        shuffle_block_size: int | None = None,
        exhausted_policy: str = "stop",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not datasets:
            raise ValueError("At least one dataset must be provided")
        super().__init__()

        self._datasets: list[Dataset] = list(datasets)
        dataset_names = [ds.name for ds in self._datasets]

        mixture_spec = MixtureSpec(mixture) if isinstance(mixture, Mapping) else mixture
        mixture_spec.validate_for(dataset_names)

        self._dataset_ids: dict[str, int] = {}
        self._datasets_by_id: dict[int, Dataset] = {}
        self._cursors: dict[str, _DatasetCursor] = {}
        self._weights: dict[str, float] = dict(
            mixture_spec.normalized
        )  # create an internal copy
        self._global_chunk_index = 0

        self._knobs = _DatasetKnobs(
            seed=seed,
            shuffle_shards=shuffle_shards,
            shuffle_within_shard=shuffle_within_shard,
            shuffle_block_size=shuffle_block_size,
        )

        for dataset_id, ds in enumerate(self._datasets):
            self._dataset_ids[ds.name] = dataset_id
            self._datasets_by_id[dataset_id] = ds
            cursor = _DatasetCursor(dataset_id, ds.shard_index, self._knobs)

            if cursor.remaining <= 0:
                # Should not happen due to validate_for call
                print(f"Unexpected: cursor for ds {dataset_id} has 0 remaining items.")
                self._weights.pop(ds.name)
                continue
            self._cursors[ds.name] = cursor

        if not self._weights:
            raise ValueError("No datasets with positive weight and non-empty cursors")

        self._chunk_size = chunk_size
        self._component_order: list[str] = [
            ds.name for ds in self._datasets if ds.name in self._weights
        ]
        if not self._component_order:
            raise ValueError("No active mixture components available")
        if self._chunk_size < len(self._component_order):
            raise ValueError(
                "chunk_size must be at least the number of mixture components"
            )
        self._chunk_quota = self._compute_chunk_quota()
        self._seed = seed
        self._remaining = sum(cursor.remaining for cursor in self._cursors.values())
        self.total_samples = len(self)

        valid_policies = {"stop", "redistribute", "repeat"}
        if exhausted_policy not in valid_policies:
            raise ValueError(
                "exhausted_policy must be one of 'stop', 'redistribute', 'repeat'"
            )
        self._exhausted_policy = exhausted_policy

    def clone_for_lane(self, lane_id: int, canonical_replicas: int) -> WorkSource:
        """Lightweight clone that avoids deepcopying large cursor buffers.

        The default ``WorkSource`` implementation performs a full ``deepcopy``,
        which replicates every per-dataset order list. Those lists can be very
        large and immutable, so we instead share the order buffers and copy only
        the mutable cursor state.
        """
        if self._cloned:
            raise RuntimeError(
                "State Error: clone_for_lane should only be called on user-defined WorkSource instances."
            )

        clone = object.__new__(type(self))
        WorkSource.__init__(clone)

        clone._datasets = list(self._datasets)
        clone._dataset_ids = dict(self._dataset_ids)
        clone._datasets_by_id = dict(self._datasets_by_id)
        clone._weights = dict(self._weights)
        clone._component_order = list(self._component_order)
        clone._chunk_size = self._chunk_size
        clone._chunk_quota = dict(self._chunk_quota)
        clone._seed = self._seed
        clone._global_chunk_index = self._global_chunk_index
        clone._exhausted_policy = self._exhausted_policy
        clone._knobs = self._knobs

        # Create fresh cursors that share the immutable order buffer but have
        # independent positions/remaining counts.
        clone._cursors = {}
        for name, cur in self._cursors.items():
            new_cur = _DatasetCursor.__new__(_DatasetCursor)
            new_cur._order = cur._order
            new_cur._position = cur._position
            new_cur.remaining = cur.remaining
            clone._cursors[name] = new_cur

        clone._remaining = self._remaining
        clone.total_samples = self.total_samples

        clone._cloned = True
        clone._bind_lane(lane_id, canonical_replicas)
        return clone

    def chunk_size_hint(self) -> int | None:
        return self._chunk_size

    @property
    def datasets_by_id(self) -> Mapping[int, Dataset]:
        """Mapping of dataset_id to its Dataset descriptor.

        The engine forwards this mapping to FetchOp so it can construct a
        dataset-aware shard store internally. The descriptors themselves do not
        perform IO.
        """
        # We expose a copy to avoid external objects interfering with our internal state.
        return dict(self._datasets_by_id)

    @property
    def dataset_ids(self) -> Mapping[str, int]:
        """Mapping of dataset name to the stable dataset_id used in SampleIds."""
        # We expose a copy to avoid external objects interfering with our internal state.
        return dict(self._dataset_ids)

    def _compute_chunk_quota(self) -> dict[str, int]:
        component_count = len(self._component_order)
        base_slots = component_count
        remaining_slots = self._chunk_size - base_slots
        quota: dict[str, int] = dict.fromkeys(self._component_order, 1)
        warn_components: list[str] = []
        entries: list[tuple[float, int, str]] = []

        for position, name in enumerate(self._component_order):
            weight = self._weights[name]
            ideal = self._chunk_size * weight
            if ideal < 1.0:
                warn_components.append(name)
            extras = max(0.0, ideal - 1.0)
            entries.append((extras, position, name))

        # Allocate integer extras greedily in descending extras order.
        remaining = remaining_slots
        integer_order = sorted(entries, key=lambda item: (-item[0], item[1]))
        remainders: list[tuple[float, int, str]] = []
        for extras, position, name in integer_order:
            if remaining <= 0:
                fractional = extras - math.floor(extras)
                remainders.append((fractional, position, name))
                continue
            desired = int(math.floor(extras))
            assign = min(desired, remaining)
            quota[name] += assign
            remaining -= assign
            fractional = extras - math.floor(extras)
            remainders.append((fractional, position, name))

        # Distribute leftover slots using largest remainder.
        if remaining > 0:
            remainders.sort(key=lambda item: (-item[0], item[1]))
            for _, _, name in remainders:
                if remaining <= 0:
                    break
                quota[name] += 1
                remaining -= 1

        if remaining != 0:
            raise RuntimeError("Per-chunk quota assignment failed to exhaust slots")

        if warn_components:
            joined = ", ".join(warn_components)
            warnings.warn(
                (
                    "Mixture components with ideal < 1 sample per chunk will "
                    f"receive 1 sample: {joined}."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        assigned = sum(quota.values())
        if assigned != self._chunk_size:
            raise RuntimeError("Per-chunk quota does not match chunk_size")

        return quota

    def next_chunk(self) -> WorkChunk | None:
        """Return the next chunk for a given canonical lane and worker.

        This implementation follows a compute-everywhere-then-discard strategy:
        - Enumerate the global chunk stream deterministically using the existing
          chunking logic.
        - Assign each global chunk index ``g`` to a lane via ``g % canonical_replicas``.
        - Discard non-matching chunks locally.
        """
        assert self._lane is not None, (
            "Please assign lane for WorkSource before requesting chunk."
        )
        assert self._canon is not None, (
            "Please assign lane for WorkSource before requesting chunk."
        )
        while True:
            chunk = self._next_chunk()
            if chunk is None:
                return None
            g = self._global_chunk_index
            self._global_chunk_index += 1
            chunk_lane = g % self._canon  # for which lane is this chunk?
            if chunk_lane != (self._lane % self._canon):  # are we asking for this lane?
                continue

            return chunk

    def _next_chunk(self) -> WorkChunk | None:
        if self._exhausted_policy == "redistribute":
            raise NotImplementedError("Exhausted policy 'redistribute' not implemented")
        if self._exhausted_policy == "repeat":
            raise NotImplementedError("Exhausted policy 'repeat' not implemented")

        # Check if all components still have sufficient samples available to generate a full chunk.
        for name in self._component_order:
            quota = self._chunk_quota[name]
            cursor = self._cursors[name]
            if cursor.remaining < quota:
                return None

        components: dict[str, list[SampleId]] = {}
        for name in self._component_order:
            quota = self._chunk_quota[name]
            if quota <= 0:
                continue
            cursor = self._cursors[name]
            samples = cursor.next_many(quota)
            if len(samples) != quota:
                raise RuntimeError(
                    f"Cursor for component '{name}' returned {len(samples)}"
                    + f" samples, expected {quota}"
                )
            components[name] = samples

        if not components:
            return None

        emitted = sum(len(ids) for ids in components.values())
        self._remaining -= emitted
        self.total_samples = len(self)

        return WorkChunk(
            components=components,
            seed=self._seed,
        )

    def __len__(self) -> int:
        """Returns the number of available _samples_ across all chunks that can be yielded."""
        if hasattr(self, "exhausted_policy"):
            if self._exhausted_policy == "redistribute":
                raise NotImplementedError(
                    "Exhausted policy 'redistribute' not implemented"
                )
            if self._exhausted_policy == "repeat":
                raise NotImplementedError("Exhausted policy 'repeat' not implemented")

        if not self._chunk_quota:
            return 0
        chunk_capacity = self._chunk_size
        chunks_possible = math.inf
        for name in self._component_order:
            quota = self._chunk_quota[name]
            if quota <= 0:
                continue
            cursor = self._cursors[name]
            available_chunks = cursor.remaining // quota
            chunks_possible = min(chunks_possible, available_chunks)
            if chunks_possible == 0:
                return 0
        if chunks_possible is math.inf:
            return 0
        return int(chunks_possible) * chunk_capacity

    def sample_id_at(self, index: int) -> SampleId:
        raise NotImplementedError("StaticMixtureWorkSource is not yet indexable")

    def supports_indexing(self) -> bool:
        return False  # StaticMixtureWorkSource is not yet indexable

    def state_dict(self) -> dict[str, Any]:
        base = super().state_dict()
        return base | {
            "version": 1,
            "seed": int(self._seed),
            "chunk_size": int(self._chunk_size),
            "knobs": {
                "shuffle_shards": self._knobs.shuffle_shards,
                "shuffle_within_shard": self._knobs.shuffle_within_shard,
                "shuffle_block_size": self._knobs.shuffle_block_size,
            },
            "global_chunk_index": int(self._global_chunk_index),
            "weights": dict(self._weights),
            "component_order": list(self._component_order),
            "dataset_ids": dict(self._dataset_ids),
            "cursor_positions": {
                name: int(cur._position) for name, cur in self._cursors.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)

        if int(state.get("version", 0)) != 1:
            raise RuntimeError("Unsupported StaticMixtureWorkSource checkpoint version")

        self._seed = int(state["seed"])
        self._chunk_size = int(state["chunk_size"])
        knobs = _DatasetKnobs(
            seed=self._seed,
            shuffle_shards=bool(state["knobs"]["shuffle_shards"]),
            shuffle_within_shard=bool(state["knobs"]["shuffle_within_shard"]),
            shuffle_block_size=state["knobs"]["shuffle_block_size"],
        )
        # Persist restored knobs for future state_dict() calls
        self._knobs = knobs

        # Rebuild cursors deterministically and set positions
        self._cursors.clear()
        self._dataset_ids.clear()
        self._datasets_by_id.clear()

        for dataset_id, ds in enumerate(self._datasets):
            self._dataset_ids[ds.name] = dataset_id
            self._datasets_by_id[dataset_id] = ds
            cur = _DatasetCursor(dataset_id, ds.shard_index, knobs)
            self._cursors[ds.name] = cur

        # Restore positions
        pos = state["cursor_positions"]
        for name, cur in self._cursors.items():
            p = int(pos.get(name, 0))
            cur._position = max(0, min(p, len(cur._order)))
            cur.remaining = len(cur._order) - cur._position

        self._weights = dict(state["weights"])
        self._component_order = list(state["component_order"])
        self._global_chunk_index = int(state["global_chunk_index"])

        # Recompute per-chunk quota to reflect restored chunk_size/weights/components
        self._chunk_quota = self._compute_chunk_quota()

        # Recompute remaining/total samples with the new cursor positions
        self._remaining = sum(cur.remaining for cur in self._cursors.values())
        self.total_samples = len(self)
