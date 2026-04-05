# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Mixture-aware work source with shard-respecting traversal."""

import math
import random
import warnings
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

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

    Memory model:
    Instead of materializing a single ``(total_samples, 3)`` NumPy array,
    this cursor streams samples shard-by-shard, lazily generating per-shard
    offset permutations.  Memory usage is O(max_shard_size) rather than
    O(total_samples).  When block shuffle is active, a block-sized buffer
    adds O(block_size) on top.

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

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_order_reference(
        dataset_id: int,
        shard_index: Mapping[int, int],
        knobs: _DatasetKnobs,
    ) -> np.ndarray:
        """Build the full order array for validation/testing.

        This is the original materializing implementation kept as a reference
        oracle.  The runtime path uses lazy shard-at-a-time iteration instead.
        """
        shard_ids = list(shard_index.keys())
        if knobs.shuffle_shards and len(shard_ids) > 1:
            random.Random(knobs.seed).shuffle(shard_ids)

        total_samples = sum(int(shard_index[sid]) for sid in shard_ids)
        order = np.empty((total_samples, 3), dtype=np.int32)

        pos = 0
        for position, shard_id in enumerate(shard_ids):
            count = int(shard_index[shard_id])
            order[pos : pos + count, 0] = dataset_id
            order[pos : pos + count, 1] = shard_id
            offsets = np.arange(count, dtype=np.int32)
            if knobs.shuffle_within_shard and count > 1:
                shard_seed = (knobs.seed << 32) ^ ((position << 16) + shard_id)
                np.random.default_rng(shard_seed).shuffle(offsets)
            order[pos : pos + count, 2] = offsets
            pos += count

        block_size = knobs.shuffle_block_size
        if block_size is not None and block_size > 0 and len(order) > 1:
            block_size = max(1, int(block_size))
            rng = np.random.default_rng(knobs.seed ^ _GOLDEN_RATIO_64)
            for start in range(0, len(order), block_size):
                end = min(start + block_size, len(order))
                rng.shuffle(order[start:end])

        return order

    @staticmethod
    def _compute_shard_order(
        shard_index: Mapping[int, int], knobs: _DatasetKnobs
    ) -> tuple[int, ...]:
        """Return shard IDs in traversal order (shuffled if requested)."""
        shard_ids = list(shard_index.keys())
        if knobs.shuffle_shards and len(shard_ids) > 1:
            random.Random(knobs.seed).shuffle(shard_ids)
        return tuple(shard_ids)

    @staticmethod
    def _make_block_rng(knobs: _DatasetKnobs) -> np.random.Generator:
        """Create the block-shuffle RNG for the given knobs."""
        return np.random.default_rng(knobs.seed ^ _GOLDEN_RATIO_64)

    # ------------------------------------------------------------------
    # Construction / epoch management
    # ------------------------------------------------------------------

    def __init__(
        self,
        dataset_id: int,
        shard_index: Mapping[int, int],
        knobs: _DatasetKnobs,
    ) -> None:
        self._dataset_id = dataset_id
        self._shard_index = shard_index
        self._epoch = 0
        # Declare all instance variables for Pyright; _init_cursor_state sets values.
        self._knobs: _DatasetKnobs = knobs
        self._shard_order: tuple[int, ...] = ()
        self._shard_sizes: tuple[int, ...] = ()
        self._total_samples: int = 0
        self._shard_cumsum: np.ndarray = np.array([], dtype=np.int64)
        self._current_shard_idx: int = 0
        self._current_shard_pos: int = 0
        self._current_offsets: np.ndarray | None = None
        self._has_block_shuffle: bool = False
        self._block_buffer: np.ndarray | None = None
        self._block_buffer_pos: int = 0
        self._blocks_consumed: int = 0
        self._block_rng: np.random.Generator | None = None
        self._block_rng_state: Mapping[str, Any] | None = None
        self._pre_fill_rng_state: Mapping[str, Any] | None = None
        self._pre_fill_block_count: int = 0
        self._position: int = 0
        self.remaining: int = 0
        self._init_cursor_state(knobs)

    def _init_cursor_state(self, knobs: _DatasetKnobs) -> None:
        """(Re-)initialise all mutable cursor state for the given knobs."""
        self._knobs = knobs

        # Shard-level metadata (small — one entry per shard).
        self._shard_order = self._compute_shard_order(self._shard_index, knobs)
        self._shard_sizes = tuple(
            int(self._shard_index[sid]) for sid in self._shard_order
        )
        self._total_samples = sum(self._shard_sizes)
        self._shard_cumsum = (
            np.cumsum(self._shard_sizes, dtype=np.int64)
            if self._shard_sizes
            else np.array([], dtype=np.int64)
        )

        # Shard-level cursor.
        self._current_shard_idx = 0
        self._current_shard_pos = 0
        self._current_offsets = None

        # Block shuffle state.
        block_size = knobs.shuffle_block_size
        self._has_block_shuffle = block_size is not None and block_size > 0
        self._block_buffer = None
        self._block_buffer_pos = 0
        self._blocks_consumed = 0
        if self._has_block_shuffle:
            self._block_rng = self._make_block_rng(knobs)
            self._block_rng_state = self._block_rng.bit_generator.state
            self._pre_fill_rng_state = self._block_rng_state
            self._pre_fill_block_count = 0
        else:
            self._block_rng = None
            self._block_rng_state = None
            self._pre_fill_rng_state = None
            self._pre_fill_block_count = 0

        # Public cursor state.
        self._position = 0
        self.remaining = self._total_samples

    def _seek_epoch(
        self, epoch: int, *, reshuffle: bool, base_knobs: _DatasetKnobs
    ) -> None:
        """Jump directly to the given epoch, rebuilding cursor state."""
        self._epoch = epoch
        if epoch > 0 and reshuffle:
            knobs = _DatasetKnobs(
                seed=base_knobs.seed + epoch * _GOLDEN_RATIO_64,
                shuffle_shards=base_knobs.shuffle_shards,
                shuffle_within_shard=base_knobs.shuffle_within_shard,
                shuffle_block_size=base_knobs.shuffle_block_size,
            )
        else:
            knobs = base_knobs
        self._init_cursor_state(knobs)

    def reset(self, *, reshuffle: bool, base_knobs: _DatasetKnobs) -> None:
        """Reset the cursor to position 0, starting a new epoch.

        If *reshuffle* is True, rebuilds the traversal order with an
        epoch-derived seed so each epoch sees a different ordering.
        """
        self._seek_epoch(self._epoch + 1, reshuffle=reshuffle, base_knobs=base_knobs)

    # ------------------------------------------------------------------
    # Lazy shard-level iteration helpers
    # ------------------------------------------------------------------

    def _ensure_shuffled_shard_offsets(self) -> None:
        """Lazily generate shuffled offsets for the current shard.

        Only call when ``shuffle_within_shard`` is True — the offsets are
        always permuted.
        """
        if self._current_offsets is not None:
            return
        if self._current_shard_idx >= len(self._shard_order):
            return
        shard_id = self._shard_order[self._current_shard_idx]
        count = self._shard_sizes[self._current_shard_idx]
        offsets = np.arange(count, dtype=np.int32)
        if count > 1:
            position = self._current_shard_idx
            shard_seed = (self._knobs.seed << 32) ^ ((position << 16) + shard_id)
            np.random.default_rng(shard_seed).shuffle(offsets)
        self._current_offsets = offsets

    def _pull_from_shards_array(self, n: int) -> np.ndarray:
        """Advance the shard cursor by up to *n* samples, returning an (M, 3) array."""
        segments: list[np.ndarray] = []
        left = n
        ds_id = self._dataset_id
        while left > 0 and self._current_shard_idx < len(self._shard_order):
            shard_id = self._shard_order[self._current_shard_idx]
            shard_size = self._shard_sizes[self._current_shard_idx]
            available = shard_size - self._current_shard_pos
            take = min(left, available)

            seg = np.empty((take, 3), dtype=np.int32)
            seg[:, 0] = ds_id
            seg[:, 1] = shard_id
            if self._knobs.shuffle_within_shard:
                self._ensure_shuffled_shard_offsets()
                assert self._current_offsets is not None
                seg[:, 2] = self._current_offsets[
                    self._current_shard_pos : self._current_shard_pos + take
                ]
            else:
                seg[:, 2] = np.arange(
                    self._current_shard_pos,
                    self._current_shard_pos + take,
                    dtype=np.int32,
                )

            segments.append(seg)
            self._current_shard_pos += take
            left -= take

            if self._current_shard_pos >= shard_size:
                self._current_shard_idx += 1
                self._current_shard_pos = 0
                self._current_offsets = None

        if not segments:
            return np.empty((0, 3), dtype=np.int32)
        if len(segments) == 1:
            return segments[0]
        return np.concatenate(segments)

    @property
    def _remaining_in_shards(self) -> int:
        """Samples not yet pulled from the shard cursor."""
        if self._current_shard_idx >= len(self._shard_order):
            return 0
        pulled = (
            int(self._shard_cumsum[self._current_shard_idx - 1])
            if self._current_shard_idx > 0
            else 0
        ) + self._current_shard_pos
        return self._total_samples - pulled

    # ------------------------------------------------------------------
    # Block shuffle helpers
    # ------------------------------------------------------------------

    # How many samples to batch into a single block buffer.  Larger values
    # amortize the per-fill overhead; the memory cost is modest (~12 bytes
    # per sample).  The value is rounded up to the next multiple of
    # block_size during filling.
    _BLOCK_BUFFER_TARGET = 1 << 16  # 65 536 samples ≈ 768 KiB

    def _fill_block_buffer(self) -> None:
        """Pull many blocks from shards, shuffle each in-place, and buffer."""
        block_size = max(1, int(self._knobs.shuffle_block_size))  # type: ignore[arg-type]
        remaining_in_shards = self._remaining_in_shards
        # Round target up to a whole number of blocks.
        target = min(self._BLOCK_BUFFER_TARGET, remaining_in_shards)
        target = max(target, block_size)  # at least one block
        n_blocks = (target + block_size - 1) // block_size
        target = n_blocks * block_size

        arr = self._pull_from_shards_array(target)
        if len(arr) == 0:
            self._block_buffer = None
            return

        # Apply per-block shuffles.  We shuffle a 1D permutation index and
        # apply via fancy indexing rather than shuffling 2D row-views, because
        # NumPy's multi-dimensional shuffle path has significant per-call
        # overhead.  The 1D path advances the RNG identically.
        if self._block_rng is not None:
            # Snapshot RNG state before this fill — used for fast checkpoint
            # restore (partial replay bounded by buffer size).
            self._pre_fill_rng_state = self._block_rng.bit_generator.state
            self._pre_fill_block_count = self._blocks_consumed
            for start in range(0, len(arr), block_size):
                end = min(start + block_size, len(arr))
                n = end - start
                perm = np.arange(n, dtype=np.intp)
                self._block_rng.shuffle(perm)
                arr[start:end] = arr[start:end][perm]
            self._block_rng_state = self._block_rng.bit_generator.state

        actual_blocks = (len(arr) + block_size - 1) // block_size
        self._block_buffer = arr
        self._block_buffer_pos = 0
        self._blocks_consumed += actual_blocks

    # ------------------------------------------------------------------
    # Public iteration
    # ------------------------------------------------------------------

    def next_many(self, limit: int) -> list[SampleId]:
        if limit <= 0 or self._position >= self._total_samples:
            return []
        if self._has_block_shuffle:
            return self._next_many_block_shuffle(limit)
        return self._next_many_sequential(limit)

    def _next_many_sequential(self, limit: int) -> list[SampleId]:
        actual = min(limit, self._total_samples - self._position)
        arr = self._pull_from_shards_array(actual)
        self._position += len(arr)
        self.remaining -= len(arr)
        return [tuple(row) for row in arr.tolist()]

    def _next_many_block_shuffle(self, limit: int) -> list[SampleId]:
        actual = min(limit, self._total_samples - self._position)
        segments: list[np.ndarray] = []
        collected = 0
        left = actual
        while left > 0:
            if self._block_buffer is None or self._block_buffer_pos >= len(
                self._block_buffer
            ):
                self._fill_block_buffer()
                if self._block_buffer is None:
                    break

            available = len(self._block_buffer) - self._block_buffer_pos
            take = min(left, available)
            segments.append(
                self._block_buffer[
                    self._block_buffer_pos : self._block_buffer_pos + take
                ]
            )
            self._block_buffer_pos += take
            collected += take
            left -= take

        self._position += collected
        self.remaining -= collected
        if not segments:
            return []
        if len(segments) == 1:
            return [tuple(row) for row in segments[0].tolist()]
        combined = np.concatenate(segments)
        return [tuple(row) for row in combined.tolist()]

    # ------------------------------------------------------------------
    # Seeking (for checkpoint restore and cloning)
    # ------------------------------------------------------------------

    def _seek_shard_cursor(self, p: int) -> None:
        """Position the shard-level cursor at global offset *p*."""
        if p <= 0 or self._total_samples == 0:
            self._current_shard_idx = 0
            self._current_shard_pos = 0
            self._current_offsets = None
            return
        if p >= self._total_samples:
            self._current_shard_idx = len(self._shard_order)
            self._current_shard_pos = 0
            self._current_offsets = None
            return
        idx = int(np.searchsorted(self._shard_cumsum, p, side="right"))
        prev_cum = int(self._shard_cumsum[idx - 1]) if idx > 0 else 0
        self._current_shard_idx = idx
        self._current_shard_pos = p - prev_cum
        self._current_offsets = None

    def _seek_to_position(
        self,
        p: int,
        block_rng_snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        """Seek the cursor to global position *p* (used by checkpoint restore).

        If *block_rng_snapshot* is provided (from a checkpoint), it contains
        the RNG state captured before the last buffer fill and the block count
        at that point.  The seek replays only the blocks between that snapshot
        and *block_idx* — bounded by the buffer size rather than O(block_idx).
        """
        p = max(0, min(p, self._total_samples))
        self._position = p
        self.remaining = self._total_samples - p

        if not self._has_block_shuffle:
            self._seek_shard_cursor(p)
            return

        block_size = max(1, int(self._knobs.shuffle_block_size))  # type: ignore[arg-type]

        if p == 0:
            self._seek_shard_cursor(0)
            self._block_rng = self._make_block_rng(self._knobs)
            self._block_rng_state = self._block_rng.bit_generator.state
            self._pre_fill_rng_state = self._block_rng_state
            self._pre_fill_block_count = 0
            self._block_buffer = None
            self._block_buffer_pos = 0
            self._blocks_consumed = 0
            return

        if p >= self._total_samples:
            self._seek_shard_cursor(self._total_samples)
            self._block_buffer = None
            self._block_buffer_pos = 0
            self._block_rng = None
            self._block_rng_state = None
            self._pre_fill_rng_state = None
            return

        block_idx = p // block_size
        pos_in_block = p % block_size
        block_start = block_idx * block_size

        # Position the shard cursor at the start of the target block.
        self._seek_shard_cursor(block_start)

        # Restore the block RNG to the state just before block_idx.
        replay_from = 0
        if (
            block_rng_snapshot is not None
            and int(block_rng_snapshot["block_count"]) <= block_idx
        ):
            # Partial replay from the saved snapshot — at most one buffer's
            # worth of blocks.
            replay_from = int(block_rng_snapshot["block_count"])
            self._block_rng = np.random.default_rng(0)
            self._block_rng.bit_generator.state = block_rng_snapshot["rng_state"]
        else:
            # Full replay from scratch — fallback for old checkpoints.
            self._block_rng = self._make_block_rng(self._knobs)

        for i in range(replay_from, block_idx):
            b_start = i * block_size
            b_end = min(b_start + block_size, self._total_samples)
            dummy = np.arange(b_end - b_start, dtype=np.intp)
            self._block_rng.shuffle(dummy)

        self._blocks_consumed = block_idx

        # Fill the block buffer starting from the target block.
        self._fill_block_buffer()
        self._block_buffer_pos = pos_in_block

    # ------------------------------------------------------------------
    # Cloning
    # ------------------------------------------------------------------

    def _clone(self) -> "_DatasetCursor":
        """Create an independent copy that shares immutable epoch-level data.

        Mutable state (position, shard cursor, block buffer, RNG) is copied
        so the clone can advance independently.
        """
        c = _DatasetCursor.__new__(_DatasetCursor)
        # Immutable / epoch-level (shared references).
        c._dataset_id = self._dataset_id
        c._shard_index = self._shard_index
        c._knobs = self._knobs
        c._shard_order = self._shard_order
        c._shard_sizes = self._shard_sizes
        c._shard_cumsum = self._shard_cumsum
        c._total_samples = self._total_samples
        c._has_block_shuffle = self._has_block_shuffle
        # Mutable scalar state (copy by value).
        c._epoch = self._epoch
        c._position = self._position
        c.remaining = self.remaining
        c._current_shard_idx = self._current_shard_idx
        c._current_shard_pos = self._current_shard_pos
        c._blocks_consumed = self._blocks_consumed
        c._pre_fill_rng_state = self._pre_fill_rng_state
        c._pre_fill_block_count = self._pre_fill_block_count
        # Lazy per-shard offsets (immutable once generated, safe to share).
        c._current_offsets = self._current_offsets
        # Block buffer (copy if active so the clone can advance independently).
        c._block_buffer = (
            self._block_buffer.copy() if self._block_buffer is not None else None
        )
        c._block_buffer_pos = self._block_buffer_pos
        # Restore block RNG from cached state.
        if self._block_rng_state is not None:
            c._block_rng = np.random.default_rng(0)
            c._block_rng.bit_generator.state = self._block_rng_state
            c._block_rng_state = self._block_rng_state
        else:
            c._block_rng = None
            c._block_rng_state = None
        return c

    # ------------------------------------------------------------------
    # Checkpoint / restore
    # ------------------------------------------------------------------

    def checkpoint_state(self) -> dict[str, Any]:
        """Return cursor state needed for deterministic continuation."""
        state: dict[str, Any] = {
            "position": int(self._position),
            "epoch": int(self._epoch),
        }
        if self._pre_fill_rng_state is not None:
            state["block_rng_snapshot"] = {
                "rng_state": self._pre_fill_rng_state,
                "block_count": self._pre_fill_block_count,
            }
        return state

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        reshuffle: bool,
        base_knobs: _DatasetKnobs,
    ) -> None:
        """Restore cursor state from a serialized checkpoint payload."""
        epoch = int(state.get("epoch", 0))
        self._seek_epoch(epoch, reshuffle=reshuffle, base_knobs=base_knobs)
        position = int(state.get("position", 0))
        snapshot = state.get("block_rng_snapshot")
        self._seek_to_position(position, block_rng_snapshot=snapshot)


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
        chunk_size: int = 16384,
        seed: int = 0,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = False,
        shuffle_block_size: int | None = None,
        exhausted_policy: str = "stop",
        reshuffle_on_repeat: bool = True,
        max_repeats: int | None = None,
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
        valid_policies = {"stop", "redistribute", "repeat"}
        if exhausted_policy not in valid_policies:
            raise ValueError(
                "exhausted_policy must be one of 'stop', 'redistribute', 'repeat'"
            )
        self._exhausted_policy = exhausted_policy
        self._reshuffle_on_repeat = reshuffle_on_repeat
        self._max_repeats = max_repeats

        if max_repeats is not None and exhausted_policy != "repeat":
            warnings.warn(
                f"max_repeats={max_repeats} has no effect with "
                f"exhausted_policy='{exhausted_policy}'",
                RuntimeWarning,
                stacklevel=2,
            )

        # Guard: with repeat policy, each dataset must have at least as many
        # samples as its per-chunk quota, otherwise reset() would loop forever.
        if exhausted_policy == "repeat":
            for name in self._component_order:
                quota = self._chunk_quota[name]
                cursor = self._cursors[name]
                if cursor._total_samples < quota:
                    raise ValueError(
                        f"Dataset '{name}' has {cursor._total_samples} samples but "
                        f"repeat policy requires at least {quota} per chunk "
                        f"(chunk_size={chunk_size}). "
                        f"Increase dataset size or decrease chunk_size."
                    )

        self.total_samples: int | float = (
            float("inf") if self._exhausted_policy == "repeat" else len(self)
        )

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
        clone._reshuffle_on_repeat = self._reshuffle_on_repeat
        clone._max_repeats = self._max_repeats
        clone._knobs = self._knobs

        clone._cursors = {name: cur._clone() for name, cur in self._cursors.items()}

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
            min_weight = min(self._weights[n] for n in warn_components)
            min_chunk_size = math.ceil(1.0 / min_weight)
            min_proportion = 1.0 / self._chunk_size
            raise ValueError(
                f"Mixture components have ideal < 1 sample per chunk: {joined}. "
                f"Either increase chunk_size to at least {min_chunk_size} to "
                f"support the current MixtureSpec, or ensure every component "
                f"has a minimum proportion of at least {min_proportion:.4g} "
                f"(= 1/chunk_size) in the MixtureSpec."
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

        # Check exhaustion per component and either stop or reset.
        for name in self._component_order:
            quota = self._chunk_quota[name]
            cursor = self._cursors[name]
            if cursor.remaining < quota:
                if self._exhausted_policy == "repeat":
                    if (
                        self._max_repeats is not None
                        and cursor._epoch >= self._max_repeats
                    ):
                        return None  # hit repeat cap
                    cursor.reset(
                        reshuffle=self._reshuffle_on_repeat,
                        base_knobs=self._knobs,
                    )
                else:
                    return None  # "stop" policy

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

        if self._exhausted_policy != "repeat":
            self.total_samples = len(self)

        return WorkChunk(
            components=components,
            seed=self._seed,
        )

    def __len__(self) -> int:
        """Returns the number of available _samples_ across all chunks that can be yielded."""
        if hasattr(self, "_exhausted_policy"):
            if self._exhausted_policy == "redistribute":
                raise NotImplementedError(
                    "Exhausted policy 'redistribute' not implemented"
                )
            if self._exhausted_policy == "repeat":
                raise TypeError(
                    "len() is not defined for an infinite work source "
                    "(exhausted_policy='repeat')"
                )

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
        cursor_states = {
            name: cur.checkpoint_state() for name, cur in self._cursors.items()
        }
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
            "cursor_states": cursor_states,
            "cursor_positions": {
                name: int(cur_state["position"])
                for name, cur_state in cursor_states.items()
            },
            "cursor_epochs": {
                name: int(cur_state["epoch"])
                for name, cur_state in cursor_states.items()
            },
            "exhausted_policy": self._exhausted_policy,
            "reshuffle_on_repeat": self._reshuffle_on_repeat,
            "max_repeats": self._max_repeats,
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

        # Restore repeat-related settings (defaults match "stop" policy for v1 compat)
        ckpt_reshuffle = state.get("reshuffle_on_repeat", self._reshuffle_on_repeat)
        ckpt_max_repeats = state.get("max_repeats", self._max_repeats)

        if ckpt_reshuffle != self._reshuffle_on_repeat:
            raise RuntimeError(
                f"Checkpoint has reshuffle_on_repeat={ckpt_reshuffle} but "
                f"the current instance was constructed with "
                f"reshuffle_on_repeat={self._reshuffle_on_repeat}. "
                f"These must match for deterministic continuation."
            )
        if ckpt_max_repeats != self._max_repeats:
            raise RuntimeError(
                f"Checkpoint has max_repeats={ckpt_max_repeats} but "
                f"the current instance was constructed with "
                f"max_repeats={self._max_repeats}. "
                f"These must match for deterministic continuation."
            )
        self._reshuffle_on_repeat = ckpt_reshuffle
        self._max_repeats = ckpt_max_repeats
        cursor_states: dict[str, Any] = state.get("cursor_states", {})
        cursor_epochs: dict[str, int] = state.get("cursor_epochs", {})

        # Verify dataset_ids matching
        ckpt_dataset_ids_raw = state.get("dataset_ids")
        if not isinstance(ckpt_dataset_ids_raw, dict):
            raise RuntimeError("Checkpoint missing or invalid dataset_ids mapping.")
        ckpt_dataset_ids = {
            str(name): int(dataset_id)
            for name, dataset_id in ckpt_dataset_ids_raw.items()
        }
        current_dataset_ids = {
            ds.name: dataset_id for dataset_id, ds in enumerate(self._datasets)
        }
        if ckpt_dataset_ids != current_dataset_ids:
            raise RuntimeError(
                "Checkpoint dataset_ids do not match current dataset ordering. "
                f"checkpoint={ckpt_dataset_ids}, current={current_dataset_ids}"
            )

        # Rebuild cursors deterministically and set positions
        self._cursors.clear()
        self._dataset_ids.clear()
        self._datasets_by_id.clear()

        for dataset_id, ds in enumerate(self._datasets):
            self._dataset_ids[ds.name] = dataset_id
            self._datasets_by_id[dataset_id] = ds
            cur = _DatasetCursor(dataset_id, ds.shard_index, knobs)
            if ds.name in cursor_states:
                cur.restore_checkpoint_state(
                    cursor_states[ds.name],
                    reshuffle=self._reshuffle_on_repeat,
                    base_knobs=knobs,
                )
            else:
                # Legacy cursor state checkpoint
                cur.restore_checkpoint_state(
                    {
                        "epoch": int(cursor_epochs.get(ds.name, 0)),
                        "position": int(state["cursor_positions"].get(ds.name, 0)),
                    },
                    reshuffle=self._reshuffle_on_repeat,
                    base_knobs=knobs,
                )
            self._cursors[ds.name] = cur

        self._weights = dict(state["weights"])
        self._component_order = list(state["component_order"])
        self._global_chunk_index = int(state["global_chunk_index"])

        # Recompute per-chunk quota to reflect restored chunk_size/weights/components
        self._chunk_quota = self._compute_chunk_quota()

        self.total_samples = (
            float("inf") if self._exhausted_policy == "repeat" else len(self)
        )
