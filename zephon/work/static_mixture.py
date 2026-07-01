# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Mixture-aware work source with shard-respecting traversal."""

import math
import random
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal

import numpy as np

from zephon.core.checkpoint import (
    CursorStateV1,
    StaticMixtureStateV4,
)
from zephon.core.constants import SampleId
from zephon.io.dataset import Dataset
from zephon.work.base import WorkChunk, WorkSource
from zephon.work.mixture import MixtureSpec

_GOLDEN_RATIO_64 = 0x9E3779B97F4A7C15  # Used to decorrelate derived seeds.

#: Multiplier applied to the largest shard when ``shuffle_block_size="auto"``.
_AUTO_BLOCK_SIZE_FACTOR = 8

#: Accepted shapes for ``shuffle_block_size``. See :func:`_resolve_block_size`.
ShuffleBlockSpec = int | Literal["auto", "global"] | None


class _Sentinel(Enum):
    UNSET = auto()


_STOP_AFTER_PASSES_UNSET = _Sentinel.UNSET


def _resolve_block_size(
    spec: ShuffleBlockSpec, total_samples: int, max_shard: int
) -> int | None:
    """Resolve ``shuffle_block_size`` to a concrete value, clamped to total.

    * ``None`` → ``None`` (cross-shard shuffle disabled)
    * ``"auto"`` → ``_AUTO_BLOCK_SIZE_FACTOR * max_shard``
    * ``"global"`` → ``total_samples``
    * positive ``int`` → the value itself; ``bool`` is rejected

    The clamp to ``total_samples`` lets callers rely on
    ``block_size <= total_samples``.
    """
    if spec is None:
        return None
    if spec == "auto":
        resolved = _AUTO_BLOCK_SIZE_FACTOR * max_shard
    elif spec == "global":
        resolved = total_samples
    elif type(spec) is int:  # exact int — bool is an int subclass
        if spec <= 0:
            raise ValueError(
                f"shuffle_block_size must be positive when given as int, got {spec}"
            )
        resolved = spec
    else:
        raise ValueError(
            "shuffle_block_size must be None, a positive int, 'auto', or 'global'; "
            f"got {spec!r}"
        )
    return min(resolved, total_samples)


def _normalize_per_dataset(
    value: Any,
    dataset_names: list[str],
    *,
    default: Any,
    name: str,
) -> dict[str, Any]:
    """Normalize a scalar-or-mapping policy knob into a per-dataset dict.

    A scalar value is broadcast to every dataset.  A mapping is validated to
    contain only known dataset names; missing keys are filled with ``default``.
    """
    if isinstance(value, Mapping):
        unknown = set(value) - set(dataset_names)
        if unknown:
            raise ValueError(
                f"{name} contains unknown dataset names: {sorted(unknown)}. "
                f"Known datasets: {dataset_names}"
            )
        return {ds: value.get(ds, default) for ds in dataset_names}
    return dict.fromkeys(dataset_names, value)


@dataclass(frozen=True)
class _DatasetKnobs:
    seed: int
    shuffle_shards: bool
    shuffle_within_shard: bool
    shuffle_block_size: int | None


@dataclass(frozen=True, slots=True)
class _AllocationConfig:
    """Immutable parameters shared by all allocation strategies.

    ``exhausted_policy``, ``reshuffle_on_repeat``, and ``max_repeats`` are
    keyed by dataset name so that different datasets in the same mixture can
    use different policies (e.g. one dataset repeats forever as padding while
    another drives termination when exhausted).

    ``stop_after_passes`` (when set) is a global termination floor: every dataset
    repeats and the stream ends once the slowest has completed that many
    passes. Mutually exclusive with per-dataset ``"stop"`` and ``max_repeats``.
    """

    component_order: tuple[str, ...]
    weights: dict[str, float]
    chunk_size: int
    exhausted_policy: Mapping[str, str]
    reshuffle_on_repeat: Mapping[str, bool]
    max_repeats: Mapping[str, int | None]
    stop_after_passes: int | None = None


def _is_unbounded(cfg: _AllocationConfig) -> bool:
    """True when nothing can ever force termination.

    Only a stream where *every* dataset repeats uncapped runs forever; a single
    ``"stop"`` or capped-``"repeat"`` dataset bounds it, as does ``stop_after_passes``.
    """
    if cfg.stop_after_passes is not None:
        return False
    return all(
        cfg.exhausted_policy[name] == "repeat" and cfg.max_repeats[name] is None
        for name in cfg.component_order
    )


def _effective_remaining_samples(
    cfg: _AllocationConfig, name: str, cursor: "_DatasetCursor"
) -> float | None:
    """Samples ``name`` can still yield before termination, or None if endless.

    A capped repeat dataset yields its current epoch's remainder plus one full
    pass for every epoch left under the cap (``reset`` fires while
    ``_epoch < max_repeats``, so epochs ``_epoch+1 .. max_repeats`` still run).
    """
    policy = cfg.exhausted_policy[name]
    ds_max = cfg.max_repeats[name]
    if policy == "repeat":
        if ds_max is None:
            return None
        return cursor.remaining + max(0, ds_max - cursor._epoch) * cursor._total_samples
    return cursor.remaining


def _stop_after_passes_reached(
    cfg: _AllocationConfig,
    cursors: Mapping[str, "_DatasetCursor"],
    exhausting: str | None = None,
) -> bool:
    """True when every dataset has completed at least ``stop_after_passes`` passes.

    A dataset has finished its current pass when its cursor sits at
    ``remaining == 0`` (consumed cleanly) or it is ``exhausting`` (about to be
    reset because it can't fill this chunk's quota); either counts as
    ``_epoch + 1``, others are mid-pass at ``_epoch``. Checking ``remaining == 0``
    — not only ``exhausting`` — is what catches a sparse dataset that finishes on
    a chunk boundary without tripping ``remaining < quota``.
    """
    assert cfg.stop_after_passes is not None
    for name in cfg.component_order:
        cursor = cursors[name]
        done_current = name == exhausting or cursor.remaining == 0
        completed = cursor._epoch + (1 if done_current else 0)
        if completed < cfg.stop_after_passes:
            return False
    return True


def _stop_after_passes_remaining_samples(
    cfg: _AllocationConfig, cursors: Mapping[str, "_DatasetCursor"]
) -> int:
    """Samples left until the slowest dataset completes ``stop_after_passes`` passes.

    Cursors advance together, so the bound is the ``max`` over datasets of each
    one's remaining-to-floor (one already past the floor contributes 0) — the
    opposite of the ``min`` that governs first-exhaustion. Approximate like the
    sibling estimators; ``compute_quotas`` is the exact stop.
    """
    assert cfg.stop_after_passes is not None
    chunks_needed = 0.0
    for name in cfg.component_order:
        avg_quota = cfg.weights[name] * cfg.chunk_size
        if avg_quota <= 0:
            continue
        cursor = cursors[name]
        if cursor._epoch >= cfg.stop_after_passes:
            continue
        remaining_to_floor = (
            cursor.remaining
            + (cfg.stop_after_passes - cursor._epoch - 1) * cursor._total_samples
        )
        chunks_needed = max(chunks_needed, remaining_to_floor / avg_quota)
    return int(chunks_needed) * cfg.chunk_size


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
    def _order_permutation(n: int, knobs: _DatasetKnobs) -> list[int] | None:
        """Shard-traversal permutation of ``range(n)``, or ``None`` when unshuffled.

        The shuffle is content-independent, so permuting indices then gathering
        matches permuting the shard ids directly — preserving checkpoint order
        (given ascending shard ids, as every ``discover()`` emits).
        """
        if not (knobs.shuffle_shards and n > 1):
            return None
        order_idx = list(range(n))
        random.Random(knobs.seed).shuffle(order_idx)
        return order_idx

    @staticmethod
    def _build_order_reference(
        dataset_id: int,
        shard_ids: np.ndarray,
        shard_sizes: np.ndarray,
        knobs: _DatasetKnobs,
    ) -> np.ndarray:
        """Build the full order array for validation/testing.

        This is the original materializing implementation kept as a reference
        oracle.  The runtime path uses lazy shard-at-a-time iteration instead.
        """
        shard_ids = np.asarray(shard_ids, dtype=np.int64)
        shard_sizes = np.asarray(shard_sizes, dtype=np.int64)
        perm = _DatasetCursor._order_permutation(shard_ids.size, knobs)
        order_idx = range(shard_ids.size) if perm is None else perm

        total_samples = int(shard_sizes.sum())
        order = np.empty((total_samples, 3), dtype=np.int32)

        pos = 0
        for position, sidx in enumerate(order_idx):
            shard_id = int(shard_ids[sidx])  # plain int: feeds the seed math below
            count = shard_sizes[sidx]
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
    def _make_block_rng(knobs: _DatasetKnobs) -> np.random.Generator:
        """Create the block-shuffle RNG for the given knobs."""
        return np.random.default_rng(knobs.seed ^ _GOLDEN_RATIO_64)

    # ------------------------------------------------------------------
    # Construction / epoch management
    # ------------------------------------------------------------------

    def __init__(
        self,
        dataset_id: int,
        shard_ids: np.ndarray,
        shard_sizes: np.ndarray,
        knobs: _DatasetKnobs,
    ) -> None:
        self._dataset_id = dataset_id
        # numpy int64 (not Python ints) keeps the WorkSource pickle into the
        # MTP/DataLoader child small; must stay immutable (shared by lane clones).
        self._ids: np.ndarray = np.asarray(shard_ids, dtype=np.int64)
        self._sizes: np.ndarray = np.asarray(shard_sizes, dtype=np.int64)
        self._epoch = 0
        # ``_base_knobs`` is the immutable epoch-0 config; ``_knobs`` is the
        # per-epoch view (seed shifts when reshuffle_on_repeat is true).
        self._base_knobs: _DatasetKnobs = knobs
        # Declare all instance variables for Pyright; _init_cursor_state sets values.
        self._knobs: _DatasetKnobs = knobs
        self._shard_order: np.ndarray = np.array([], dtype=np.int64)
        self._shard_sizes: np.ndarray = np.array([], dtype=np.int64)
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

        # ``None`` => natural order; ``slice(None)`` gathers the whole array.
        perm = self._order_permutation(self._ids.size, knobs)
        idx: slice | np.ndarray = (
            slice(None) if perm is None else np.asarray(perm, dtype=np.intp)
        )
        self._shard_order = self._ids[idx]
        self._shard_sizes = self._sizes[idx]
        self._total_samples = int(self._shard_sizes.sum())
        self._shard_cumsum = np.cumsum(self._shard_sizes, dtype=np.int64)

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

    def _seek_epoch(self, epoch: int, *, reshuffle: bool) -> None:
        """Jump directly to the given epoch, rebuilding cursor state."""
        self._epoch = epoch
        base = self._base_knobs
        if epoch > 0 and reshuffle:
            knobs = _DatasetKnobs(
                seed=base.seed + epoch * _GOLDEN_RATIO_64,
                shuffle_shards=base.shuffle_shards,
                shuffle_within_shard=base.shuffle_within_shard,
                shuffle_block_size=base.shuffle_block_size,
            )
        else:
            knobs = base
        self._init_cursor_state(knobs)

    def reset(self, *, reshuffle: bool) -> None:
        """Reset the cursor to position 0, starting a new epoch.

        If *reshuffle* is True, rebuilds the traversal order with an
        epoch-derived seed so each epoch sees a different ordering.
        """
        self._seek_epoch(self._epoch + 1, reshuffle=reshuffle)

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
        # Plain int: a numpy int64 here overflows the (seed << 32) ^ ... seed math.
        shard_id = int(self._shard_order[self._current_shard_idx])
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

    def next_one(self) -> SampleId | None:
        """Consume one sample without the ``next_many(1)`` numpy/list overhead."""
        if self._position >= self._total_samples:
            return None
        if self._has_block_shuffle:
            if self._block_buffer is not None and self._block_buffer_pos < len(
                self._block_buffer
            ):
                row = self._block_buffer[self._block_buffer_pos]
                self._block_buffer_pos += 1
                self._position += 1
                self.remaining -= 1
                return tuple(row.tolist())
            got = self.next_many(1)
            return got[0] if got else None

        # Mirror _pull_from_shards_array(take=1) without building the array wrapper.
        # Skip empty shards; the _position guard guarantees a real one remains.
        while (
            self._current_shard_idx < len(self._shard_order)
            and int(self._shard_sizes[self._current_shard_idx]) == 0
        ):
            self._current_shard_idx += 1
            self._current_offsets = None
        assert self._current_shard_idx < len(self._shard_order)
        shard_id = int(self._shard_order[self._current_shard_idx])
        if self._knobs.shuffle_within_shard:
            self._ensure_shuffled_shard_offsets()
            assert self._current_offsets is not None
            offset = int(self._current_offsets[self._current_shard_pos])
        else:
            offset = self._current_shard_pos
        self._current_shard_pos += 1
        if self._current_shard_pos >= int(self._shard_sizes[self._current_shard_idx]):
            self._current_shard_idx += 1
            self._current_shard_pos = 0
            self._current_offsets = None
        self._position += 1
        self.remaining -= 1
        return (self._dataset_id, shard_id, offset)

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
        c._ids = self._ids
        c._sizes = self._sizes
        c._base_knobs = self._base_knobs
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
        snapshot = None
        if self._pre_fill_rng_state is not None:
            snapshot = {
                "rng_state": self._pre_fill_rng_state,
                "block_count": self._pre_fill_block_count,
            }
        state = CursorStateV1(
            position=int(self._position),
            epoch=int(self._epoch),
            block_rng_snapshot=snapshot,
        )
        return state.to_dict(strip_none=True)

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        reshuffle: bool,
    ) -> None:
        """Restore cursor state from a serialized checkpoint payload."""
        cur = CursorStateV1.load(state)
        self._seek_epoch(cur.epoch, reshuffle=reshuffle)
        self._seek_to_position(cur.position, block_rng_snapshot=cur.block_rng_snapshot)


# ---------------------------------------------------------------------------
# Allocation strategies
# ---------------------------------------------------------------------------


class AllocationStrategy(ABC):
    """Encapsulates per-chunk sample production, length estimation, and checkpoint state.

    Concrete subclasses own mode-specific mutable state (e.g. fractional
    accumulators or fixed quotas) and the exhaustion-handling loop.
    """

    def __init__(self, config: _AllocationConfig) -> None:
        self._config = config

    @abstractmethod
    def produce(
        self, cursors: dict[str, _DatasetCursor]
    ) -> dict[str, list[SampleId]] | None:
        """Materialize one chunk's samples, or None if exhausted.

        Implementations own exhaustion rollback. Zero-sample components are omitted.
        """

    @abstractmethod
    def estimate_remaining_samples(self, cursors: dict[str, _DatasetCursor]) -> int:
        """Estimate the number of samples still available (for ``__len__``)."""

    @abstractmethod
    def clone(self) -> "AllocationStrategy":
        """Return an independent copy with deep-copied mutable state."""

    @abstractmethod
    def checkpoint_state(self) -> dict[str, Any]:
        """Return strategy-specific fields to merge into the checkpoint."""

    def _check_no_redistribute(self) -> None:
        for name in self._config.component_order:
            if self._config.exhausted_policy[name] == "redistribute":
                raise NotImplementedError(
                    "Exhausted policy 'redistribute' not implemented"
                )


class QuotaAllocationStrategy(AllocationStrategy):
    """Shared ``produce`` for strategies that first compute integer quotas.

    Subclasses own quota policy; this class centralizes cursor pulls, zero
    quotas, and underfill handling.
    """

    @abstractmethod
    def compute_quotas(
        self, cursors: dict[str, _DatasetCursor]
    ) -> dict[str, int] | None:
        """Return per-component quotas for one chunk, or None if exhausted."""

    def produce(
        self, cursors: dict[str, _DatasetCursor]
    ) -> dict[str, list[SampleId]] | None:
        quotas = self.compute_quotas(cursors)
        if quotas is None:
            return None

        components: dict[str, list[SampleId]] = {}
        for name in self._config.component_order:
            quota = quotas[name]
            if quota <= 0:
                continue
            cursor = cursors[name]
            samples = cursor.next_many(quota)
            if len(samples) != quota:
                raise RuntimeError(
                    f"Cursor for component '{name}' returned {len(samples)}"
                    f" samples, expected {quota}"
                )
            components[name] = samples
        return components


class AccumulatorStrategy(QuotaAllocationStrategy):
    """Bresenham-style fractional-accumulator allocation (default).

    Over many chunks the running average converges to exact requested
    mixture weights with no ``1/chunk_size`` granularity limitation.
    Individual chunks may be sparse (a dataset may contribute 0 samples).
    """

    def __init__(
        self,
        config: _AllocationConfig,
        accumulators: dict[str, float] | None = None,
    ) -> None:
        super().__init__(config)
        self._accumulators: dict[str, float] = (
            dict(accumulators)
            if accumulators is not None
            else dict.fromkeys(config.component_order, 0.0)
        )

    # -- quota computation ------------------------------------------------

    def _advance_accumulators(self) -> dict[str, int]:
        """Advance accumulators and return per-component quotas for this chunk.

        Uses Bresenham-style fractional accumulation:

        1. Add ``weight * chunk_size`` to each accumulator.
        2. Take ``int(accum)`` (truncation toward zero) as the base quota;
           subtract it so the accumulator holds only the remainder.
        3. Apply a largest-remainder correction so ``sum(quotas) == chunk_size``
           exactly, and **debit/credit each correction back into the
           accumulator** so that future chunks account for the actual
           (corrected) allocation.

        Accumulators may temporarily go negative after a +1 correction; they
        recover naturally as ``weight * chunk_size`` is added each chunk.
        Negative accumulators sort last in the deficit correction, preventing
        back-to-back over-allocation.
        """
        cfg = self._config
        quotas: dict[str, int] = {}
        total = 0

        for name in cfg.component_order:
            self._accumulators[name] += cfg.weights[name] * cfg.chunk_size
            q = int(self._accumulators[name])  # truncation toward zero
            self._accumulators[name] -= q
            quotas[name] = q
            total += q

        deficit = cfg.chunk_size - total

        if deficit > 0:
            # Give extra slots to components with largest fractional remainder.
            order = sorted(
                enumerate(cfg.component_order),
                key=lambda pair: (-self._accumulators[pair[1]], pair[0]),
            )
            for _, name in order:
                if deficit <= 0:
                    break
                quotas[name] += 1
                deficit -= 1

                # debit each correction back into the accumulator
                self._accumulators[name] -= 1.0

        elif deficit < 0:
            # Remove slots from components with smallest fractional remainder.
            order = sorted(
                enumerate(cfg.component_order),
                key=lambda pair: (self._accumulators[pair[1]], pair[0]),
            )
            for _, name in order:
                if deficit >= 0:
                    break
                if quotas[name] > 0:
                    quotas[name] -= 1
                    deficit += 1

                    # credit each correction back into the accumulator
                    self._accumulators[name] += 1.0

        if sum(quotas.values()) != cfg.chunk_size:
            raise RuntimeError(
                "Accumulator quota correction failed to hit chunk_size "
                f"(got {sum(quotas.values())}, expected {cfg.chunk_size})"
            )

        return quotas

    def compute_quotas(
        self, cursors: dict[str, _DatasetCursor]
    ) -> dict[str, int] | None:
        """Accumulator-based chunk production (default)."""
        cfg = self._config
        self._check_no_redistribute()

        # Compute quotas, then verify cursors can fulfill them.
        # On exhaustion with repeat policy: rollback accumulators, reset
        # the exhausted cursor, and retry.
        while True:
            saved = dict(self._accumulators)
            quotas = self._advance_accumulators()

            needs_retry = False
            for name in cfg.component_order:
                quota = quotas[name]
                cursor = cursors[name]
                if cursor.remaining < quota:
                    if cfg.exhausted_policy[name] == "repeat":
                        ds_max = cfg.max_repeats[name]
                        # Roll back this chunk's accumulator advance before any
                        # terminal None: otherwise a checkpoint taken after the
                        # engine prefetched it stores debited-but-unfulfilled
                        # accumulators, and the restored source emits phantom
                        # chunks before re-exhausting. The terminal None must be
                        # side-effect-free.
                        if ds_max is not None and cursor._epoch >= ds_max:
                            self._accumulators = saved
                            return None
                        if (
                            cfg.stop_after_passes is not None
                            and _stop_after_passes_reached(cfg, cursors, name)
                        ):
                            self._accumulators = saved
                            return None
                        cursor.reset(reshuffle=cfg.reshuffle_on_repeat[name])
                        self._accumulators = saved
                        needs_retry = True
                        break
                    else:
                        self._accumulators = saved  # see terminal note above
                        return None  # "stop" policy

            if needs_retry:
                continue
            # Even when every cursor can fill this chunk, stop if the floor is
            # already met: a sparse component can finish its pass cleanly (then
            # draw quota 0) without ever tripping ``remaining < quota`` above,
            # which would otherwise overshoot the floor. Roll back so the
            # terminal None stays side-effect-free (see the rollback note above).
            if cfg.stop_after_passes is not None and _stop_after_passes_reached(
                cfg, cursors
            ):
                self._accumulators = saved
                return None
            return quotas

    # -- length estimation ------------------------------------------------

    def estimate_remaining_samples(self, cursors: dict[str, _DatasetCursor]) -> int:
        """Estimate of remaining samples using average per-chunk quota.

        Uses ``weight * chunk_size`` (the long-run average quota) as the
        divisor for each component. This may slightly overestimate because a
        component could receive a larger-than-average quota in the very next
        chunk, but it is close to correct and consistent with accumulator
        convergence. Actual termination is governed by ``compute_quotas``.

        Repeating datasets with no ``max_repeats`` cap contribute infinitely
        and are skipped from the ``min``.  Repeating datasets with a finite
        cap are projected forward by the remaining epochs.  Under ``stop_after_passes``
        termination is governed by the slowest dataset (a ``max``), so that
        case is delegated to :func:`_stop_after_passes_remaining_samples`.
        """
        cfg = self._config
        if cfg.stop_after_passes is not None:
            return _stop_after_passes_remaining_samples(cfg, cursors)
        chunks_possible: float = math.inf
        for name in cfg.component_order:
            avg_quota = cfg.weights[name] * cfg.chunk_size
            if avg_quota <= 0:
                continue
            effective_remaining = _effective_remaining_samples(cfg, name, cursors[name])
            if effective_remaining is None:
                continue  # unbounded contribution
            available = effective_remaining / avg_quota
            chunks_possible = min(chunks_possible, available)
            if chunks_possible <= 0:
                return 0
        if chunks_possible is math.inf:
            return 0
        return int(chunks_possible) * cfg.chunk_size

    # -- clone / checkpoint -----------------------------------------------

    def clone(self) -> "AccumulatorStrategy":
        return AccumulatorStrategy(
            config=self._config,
            accumulators=self._accumulators,
        )

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "allocation_mode": "accumulator",
            "accumulators": {name: float(v) for name, v in self._accumulators.items()},
        }


class LegacyFixedStrategy(QuotaAllocationStrategy):
    """Fixed per-chunk quota allocation (pre-accumulator checkpoints only).

    Each component gets at least 1 sample per chunk; the remainder is
    distributed via largest-remainder allocation.  Quotas are computed
    once at construction and reused for every chunk.
    """

    def __init__(self, config: _AllocationConfig) -> None:
        super().__init__(config)
        self._chunk_quota = self._compute_chunk_quota()

    # -- fixed-quota computation ------------------------------------------

    def _compute_chunk_quota(self) -> dict[str, int]:
        """Largest-remainder allocation for legacy fixed-quota mode."""
        cfg = self._config
        component_count = len(cfg.component_order)
        base_slots = component_count
        remaining_slots = cfg.chunk_size - base_slots
        quota: dict[str, int] = dict.fromkeys(cfg.component_order, 1)
        warn_components: list[str] = []
        entries: list[tuple[float, int, str]] = []

        for position, name in enumerate(cfg.component_order):
            weight = cfg.weights[name]
            ideal = cfg.chunk_size * weight
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
            min_weight = min(cfg.weights[n] for n in warn_components)
            min_chunk_size = math.ceil(1.0 / min_weight)
            min_proportion = 1.0 / cfg.chunk_size
            raise ValueError(
                f"Mixture components have ideal < 1 sample per chunk: {joined}. "
                f"Either increase chunk_size to at least {min_chunk_size} to "
                f"support the current MixtureSpec, or ensure every component "
                f"has a minimum proportion of at least {min_proportion:.4g} "
                f"(= 1/chunk_size) in the MixtureSpec."
            )

        assigned = sum(quota.values())
        if assigned != cfg.chunk_size:
            raise RuntimeError("Per-chunk quota does not match chunk_size")

        return quota

    # -- quota computation ------------------------------------------------

    def compute_quotas(
        self, cursors: dict[str, _DatasetCursor]
    ) -> dict[str, int] | None:
        """Legacy fixed-quota chunk production (pre-accumulator checkpoints)."""
        cfg = self._config
        self._check_no_redistribute()

        # Check exhaustion per component and either stop or reset.
        for name in cfg.component_order:
            quota = self._chunk_quota[name]
            cursor = cursors[name]
            if cursor.remaining < quota:
                if cfg.exhausted_policy[name] == "repeat":
                    ds_max = cfg.max_repeats[name]
                    if ds_max is not None and cursor._epoch >= ds_max:
                        return None  # hit repeat cap
                    cursor.reset(reshuffle=cfg.reshuffle_on_repeat[name])
                else:
                    return None  # "stop" policy

        return dict(self._chunk_quota)

    # -- length estimation ------------------------------------------------

    def estimate_remaining_samples(self, cursors: dict[str, _DatasetCursor]) -> int:
        # Legacy mode is reached only from pre-strategy checkpoints, which always
        # migrate to stop_after_passes=None, so the floor never applies here.
        if not self._chunk_quota:
            return 0
        cfg = self._config
        chunk_capacity = cfg.chunk_size
        chunks_possible: float = math.inf
        for name in cfg.component_order:
            quota = self._chunk_quota[name]
            if quota <= 0:
                continue
            effective_remaining = _effective_remaining_samples(cfg, name, cursors[name])
            if effective_remaining is None:
                continue
            available_chunks = effective_remaining // quota
            chunks_possible = min(chunks_possible, available_chunks)
            if chunks_possible == 0:
                return 0
        if chunks_possible is math.inf:
            return 0
        return int(chunks_possible) * chunk_capacity

    # -- clone / checkpoint -----------------------------------------------

    def clone(self) -> "LegacyFixedStrategy":
        s = LegacyFixedStrategy.__new__(LegacyFixedStrategy)
        AllocationStrategy.__init__(s, self._config)
        s._chunk_quota = dict(self._chunk_quota)
        return s

    def checkpoint_state(self) -> dict[str, Any]:
        return {"allocation_mode": "legacy_fixed"}


# ---------------------------------------------------------------------------
# Main work source
# ---------------------------------------------------------------------------


class StaticMixtureWorkSource(WorkSource):
    """Emit SampleId triples from one or more datasets according to a mixture.

    By default this source uses Bresenham-style fractional accumulators to
    distribute samples across datasets. Over many chunks the running average
    converges to the exact requested mixture weights with no
    ``1/chunk_size`` granularity limitation. Individual chunks may have
    sparse allocations (a dataset may contribute 0 samples in a given chunk).

    When loading a checkpoint that was created by the legacy fixed-quota
    allocator (pre-accumulator), the source automatically falls back to the
    original allocation strategy for deterministic continuation.

    This source orchestrates ordering only. It builds per-dataset cursors from
    the provided ``Dataset`` descriptors and applies shuffling at three levels:
    - shard order within a dataset (``shuffle_shards``)
    - sample order within each shard (``shuffle_within_shard``)
    - optional block-based cross-shard shuffle (``shuffle_block_size``)

    ``shuffle_block_size`` accepts:
    - ``None`` (default) — cross-shard block shuffle disabled
    - a positive ``int`` — explicit block size
    - ``"auto"`` — ``8 * max_shard`` across all datasets in the mix
    - ``"global"`` — that dataset's total sample count; the block buffer is
      O(total_samples)

    Resolved per-cursor values are clamped to the dataset's total.

    No IO is performed here; fetching is handled by FetchOp which constructs an
    internal shard store using the dataset descriptors exposed via
    ``datasets_by_id``.

    The emitted identifiers have the shape ``(dataset_id, shard_id, sample_idx)``
    where ``dataset_id`` is the position of the dataset in the constructor list.

    ``exhausted_policy``, ``reshuffle_on_repeat``, and ``max_repeats`` each
    accept either a scalar (broadcast to every dataset) or a per-dataset
    mapping, so different datasets in the mixture can use different policies
    (e.g. one repeats forever as padding while another drives termination when
    exhausted).

    ``stop_after_passes`` is the minimum number of full passes per dataset: every
    dataset repeats and the run stops once all have been traversed that many
    times. The slowest dataset is seen exactly this many times and triggers the
    stop; faster datasets loop more in the meantime to hold the mixing ratio. It
    is the **default** (``stop_after_passes=1`` — see each dataset once, then
    stop) when no exhaustion config is passed; giving ``exhausted_policy`` or
    ``max_repeats`` opts into the explicit per-dataset API and turns it off. An
    explicit ``stop_after_passes`` requires every dataset to repeat, so it rejects
    (``ValueError``) a non-``"repeat"`` ``exhausted_policy`` or any ``max_repeats``.

    ``lane_assignment`` selects chunk->lane routing. ``"permute"`` (default)
    uses a seeded per-block permutation that breaks cadence-sharding resonance
    (see :meth:`_lane_for_chunk`); ``"modulo"`` is plain
    ``g % canonical_replicas`` and reproduces pre-fix routing.
    """

    def __init__(
        self,
        datasets: list[Dataset],
        mixture: MixtureSpec | Mapping[str, float],
        chunk_size: int = 16384,
        seed: int = 0,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = False,
        shuffle_block_size: ShuffleBlockSpec = None,
        exhausted_policy: str | Mapping[str, str] | None = None,
        reshuffle_on_repeat: bool | Mapping[str, bool] | _Sentinel = _Sentinel.UNSET,
        max_repeats: int | None | Mapping[str, int | None] = None,
        stop_after_passes: int | None | _Sentinel = _STOP_AFTER_PASSES_UNSET,
        lane_assignment: Literal["modulo", "permute"] = "permute",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not datasets:
            raise ValueError("At least one dataset must be provided")
        if lane_assignment not in ("modulo", "permute"):
            raise ValueError(
                "lane_assignment must be 'modulo' or 'permute', "
                f"got {lane_assignment!r}"
            )
        super().__init__()

        # Remember which exhaustion-policy args the caller set explicitly: a
        # resume rejects a caller that contradicts the checkpoint, while args
        # left at their defaults still inherit the frozen policy even if the
        # default has since moved (see load_state_dict).
        self._explicit_policy_fields: frozenset[str] = frozenset(
            name
            for name, provided in (
                ("exhausted_policy", exhausted_policy is not None),
                ("reshuffle_on_repeat", reshuffle_on_repeat is not _Sentinel.UNSET),
                ("max_repeats", max_repeats is not None),
                (
                    "stop_after_passes",
                    stop_after_passes is not _STOP_AFTER_PASSES_UNSET,
                ),
            )
            if provided
        )

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
        self._lane_assignment: str = lane_assignment
        # _perm_* memoize the current block's permutation (reshuffled once per block).
        self._perm_block: int = -1
        self._perm_order: list[int] = []

        self._shuffle_block_size_spec: ShuffleBlockSpec = shuffle_block_size

        # max_shard is only consumed by the "auto" path; default to 0 so the
        # value is well-defined but unused for the other specs.
        max_shard = 0
        if shuffle_block_size == "auto":
            per_dataset_max = [ds.max_count() for ds in self._datasets]
            max_shard = max(per_dataset_max, default=0)
            if max_shard <= 0:
                raise ValueError(
                    "shuffle_block_size='auto' requires at least one non-empty "
                    "shard across the provided datasets"
                )

        self._knobs_by_name: dict[str, _DatasetKnobs] = {}
        for dataset_id, ds in enumerate(self._datasets):
            self._dataset_ids[ds.name] = dataset_id
            self._datasets_by_id[dataset_id] = ds

            total = ds.total()
            knobs = _DatasetKnobs(
                seed=seed,
                shuffle_shards=shuffle_shards,
                shuffle_within_shard=shuffle_within_shard,
                shuffle_block_size=_resolve_block_size(
                    shuffle_block_size, total, max_shard
                ),
            )
            self._knobs_by_name[ds.name] = knobs
            cursor = _DatasetCursor(dataset_id, ds.ids(), ds.counts(), knobs)

            if cursor.remaining <= 0:
                # Should not happen due to validate_for call
                print(f"Unexpected: cursor for ds {dataset_id} has 0 remaining items.")
                self._weights.pop(ds.name)
                self._knobs_by_name.pop(ds.name)
                continue
            self._cursors[ds.name] = cursor

        if not self._weights:
            raise ValueError("No datasets with positive weight and non-empty cursors")

        component_order = tuple(
            ds.name for ds in self._datasets if ds.name in self._weights
        )
        if not component_order:
            raise ValueError("No active mixture components available")

        self._seed = seed
        active_names = list(component_order)

        # Resolve the stop_after_passes default: unset (caller passed nothing) becomes
        # stop_after_passes=1 — epoch the slowest dataset once, repeat the rest —
        # unless the caller opted into the explicit stop/cap API, which is
        # mutually exclusive with stop_after_passes.
        if stop_after_passes is _STOP_AFTER_PASSES_UNSET:
            stop_after_passes = (
                1 if exhausted_policy is None and max_repeats is None else None
            )

        # Validate stop_after_passes and its interactions up-front so a contradictory
        # config fails loudly instead of silently overriding the caller.
        if stop_after_passes is not None:
            if isinstance(stop_after_passes, bool) or not isinstance(
                stop_after_passes, int
            ):
                raise ValueError(
                    f"stop_after_passes must be a positive int or None, got {stop_after_passes!r}"
                )
            if stop_after_passes < 1:
                raise ValueError(
                    f"stop_after_passes must be >= 1, got {stop_after_passes}"
                )

        # An unspecified policy resolves to "stop" normally, but to "repeat"
        # under stop_after_passes, which requires every dataset to repeat.
        policy_default = "repeat" if stop_after_passes is not None else "stop"
        if exhausted_policy is None:
            exhausted_policy = policy_default
        exhausted_policy_map = _normalize_per_dataset(
            exhausted_policy,
            active_names,
            default=policy_default,
            name="exhausted_policy",
        )
        valid_policies = {"stop", "redistribute", "repeat"}
        for ds_name, policy in exhausted_policy_map.items():
            if policy not in valid_policies:
                raise ValueError(
                    f"exhausted_policy['{ds_name}']={policy!r} must be one of "
                    f"{sorted(valid_policies)}"
                )

        if stop_after_passes is not None:
            conflicting = sorted(
                ds for ds, pol in exhausted_policy_map.items() if pol != "repeat"
            )
            if conflicting:
                raise ValueError(
                    f"stop_after_passes={stop_after_passes} requires every dataset to repeat "
                    f"toward the global floor, but exhausted_policy sets "
                    f"{conflicting} to a non-'repeat' policy. stop_after_passes governs "
                    f"termination globally (stop once every dataset has completed "
                    f"{stop_after_passes} pass(es)); drop the explicit exhausted_policy "
                    f"override or drop stop_after_passes."
                )
            if max_repeats is not None:
                raise ValueError(
                    f"stop_after_passes={stop_after_passes} cannot be combined with "
                    f"max_repeats={max_repeats!r}: stop_after_passes is a global lower "
                    f"bound on passes while max_repeats is a per-dataset upper "
                    f"cap that could force the stream to end before the floor is "
                    f"reached. Use one or the other."
                )

        if reshuffle_on_repeat is _Sentinel.UNSET:
            reshuffle_on_repeat = True
        reshuffle_on_repeat_map = _normalize_per_dataset(
            reshuffle_on_repeat,
            active_names,
            default=True,
            name="reshuffle_on_repeat",
        )

        max_repeats_map = _normalize_per_dataset(
            max_repeats,
            active_names,
            default=None,
            name="max_repeats",
        )

        for ds_name, mr in max_repeats_map.items():
            if mr is not None and exhausted_policy_map[ds_name] != "repeat":
                warnings.warn(
                    f"max_repeats['{ds_name}']={mr} has no effect with "
                    f"exhausted_policy['{ds_name}']="
                    f"'{exhausted_policy_map[ds_name]}'",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self._alloc_config = _AllocationConfig(
            component_order=component_order,
            weights=self._weights,
            chunk_size=chunk_size,
            exhausted_policy=exhausted_policy_map,
            reshuffle_on_repeat=reshuffle_on_repeat_map,
            max_repeats=max_repeats_map,
            stop_after_passes=stop_after_passes,
        )

        self._validate_repeat_capacity()

        self._strategy: AllocationStrategy = AccumulatorStrategy(
            config=self._alloc_config,
        )

        self.total_samples: int | float = (
            float("inf") if _is_unbounded(self._alloc_config) else len(self)
        )

    def _validate_repeat_capacity(self) -> None:
        """Reject a repeat-policy dataset too small to fill one chunk's quota.

        The accumulator can carry up to ~1.0 of fractional remainder, so the
        worst-case single-chunk quota is ceil(weight * chunk_size).  Without
        this check the retry loop in _next_chunk_accumulator would spin
        forever (rollback → same accumulators → same impossible quota).
        """
        cfg = self._alloc_config
        for name in cfg.component_order:
            if cfg.exhausted_policy[name] != "repeat":
                continue
            min_required = math.ceil(self._weights[name] * cfg.chunk_size)
            cursor = self._cursors[name]
            if cursor._total_samples < min_required:
                raise ValueError(
                    f"Dataset '{name}' has {cursor._total_samples} samples "
                    f"but repeat policy requires at least {min_required} "
                    f"(ceil(weight={self._weights[name]:.4g} * "
                    f"chunk_size={cfg.chunk_size})). "
                    f"Increase dataset size or decrease chunk_size."
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
        clone._alloc_config = self._alloc_config  # frozen, safe to share
        clone._explicit_policy_fields = self._explicit_policy_fields
        clone._seed = self._seed
        clone._global_chunk_index = self._global_chunk_index
        clone._lane_assignment = self._lane_assignment
        clone._perm_block = -1
        clone._perm_order = []
        clone._shuffle_block_size_spec = self._shuffle_block_size_spec
        clone._knobs_by_name = dict(self._knobs_by_name)

        clone._strategy = self._strategy.clone()

        clone._cursors = {name: cur._clone() for name, cur in self._cursors.items()}

        clone.total_samples = self.total_samples

        clone._cloned = True
        clone._bind_lane(lane_id, canonical_replicas)
        return clone

    def chunk_size_hint(self) -> int | None:
        return self._alloc_config.chunk_size

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

    # ------------------------------------------------------------------
    # Chunk production
    # ------------------------------------------------------------------

    def next_chunk(self) -> WorkChunk | None:
        """Return the next chunk for a given canonical lane and worker.

        This implementation follows a compute-everywhere-then-discard strategy:
        - Enumerate the global chunk stream deterministically using the existing
          chunking logic.
        - Assign each global chunk index ``g`` to a lane via :meth:`_lane_for_chunk`.
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
            if self._lane_for_chunk(g) != (self._lane % self._canon):
                continue
            return chunk

    def _lane_for_chunk(self, g: int) -> int:
        """Map global chunk index ``g`` to a canonical lane.

        Plain ``g % canon`` (``lane_assignment="modulo"``) is vulnerable to
        cadence-sharding resonance: the SWRR allocator emits a rare component
        roughly every ``1 / weight`` chunks, and when that period shares a
        factor with ``canon`` the component lands on only ``gcd(period, canon)``
        of the lanes. The starved lanes can stall a downstream gate (e.g. a
        strict ``ensure_mixture``) that waits for that component.

        ``"permute"`` (the default) routes each block of ``canon`` consecutive
        chunks through a seeded permutation of the lanes: every block still
        hands each lane exactly one chunk (load balance and exactly-once
        preserved), the map is a pure function of ``(seed, g)`` so replay is
        unaffected, and a periodic emitter now sprays lanes ~uniformly. The seed
        composes integers rather than hashing a tuple because ``hash()`` is not
        stable across processes (PYTHONHASHSEED).
        """
        assert self._canon is not None
        if self._lane_assignment != "permute":
            return g % self._canon
        block, slot = divmod(g, self._canon)
        if block != self._perm_block:
            order = list(range(self._canon))
            random.Random(self._seed + block * _GOLDEN_RATIO_64).shuffle(order)
            self._perm_block = block
            self._perm_order = order
        return self._perm_order[slot]

    def _next_chunk(self) -> WorkChunk | None:
        components = self._strategy.produce(self._cursors)
        if not components:
            return None

        if not _is_unbounded(self._alloc_config):
            self.total_samples = len(self)

        return WorkChunk(
            components=components,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Returns the number of available samples across all chunks that can be yielded."""
        if hasattr(self, "_alloc_config"):
            cfg = self._alloc_config
            for name in cfg.component_order:
                if cfg.exhausted_policy[name] == "redistribute":
                    raise NotImplementedError(
                        "Exhausted policy 'redistribute' not implemented"
                    )
            if _is_unbounded(cfg):
                raise TypeError(
                    "len() is not defined for an infinite work source "
                    "(at least one dataset has exhausted_policy='repeat' "
                    "with max_repeats=None)"
                )

        return self._strategy.estimate_remaining_samples(self._cursors)

    # ------------------------------------------------------------------
    # Indexing (not supported)
    # ------------------------------------------------------------------

    def sample_id_at(self, index: int) -> SampleId:
        raise NotImplementedError("StaticMixtureWorkSource is not yet indexable")

    def supports_indexing(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Checkpoint / restore
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        base = super().state_dict()
        cursor_states = {
            name: cur.checkpoint_state() for name, cur in self._cursors.items()
        }
        cfg = self._alloc_config
        strategy_state = self._strategy.checkpoint_state()
        accumulators = strategy_state.get("accumulators")
        # seed / shuffle_shards / shuffle_within_shard are identical across
        # the per-dataset knobs by construction; pick any one as a representative.
        sample_knobs = next(iter(self._knobs_by_name.values()))
        cursor_block_sizes = {
            name: self._knobs_by_name[name].shuffle_block_size for name in self._cursors
        }
        state = StaticMixtureStateV4(
            lane_id=base["lane_id"],
            canonical_replicas=base["canonical_replicas"],
            chunk_size_hint=base["chunk_size_hint"],
            seed=int(self._seed),
            chunk_size=int(cfg.chunk_size),
            knobs={
                "shuffle_shards": sample_knobs.shuffle_shards,
                "shuffle_within_shard": sample_knobs.shuffle_within_shard,
            },
            shuffle_block_size_spec=self._shuffle_block_size_spec,
            cursor_block_sizes=cursor_block_sizes,
            global_chunk_index=int(self._global_chunk_index),
            weights=dict(cfg.weights),
            component_order=list(cfg.component_order),
            dataset_ids=dict(self._dataset_ids),
            cursor_states=cursor_states,
            cursor_positions={
                name: int(cur_state["position"])
                for name, cur_state in cursor_states.items()
            },
            cursor_epochs={
                name: int(cur_state["epoch"])
                for name, cur_state in cursor_states.items()
            },
            exhausted_policy=dict(cfg.exhausted_policy),
            reshuffle_on_repeat=dict(cfg.reshuffle_on_repeat),
            max_repeats=dict(cfg.max_repeats),
            stop_after_passes=cfg.stop_after_passes,
            lane_assignment=self._lane_assignment,
            accumulators=accumulators,
            allocation_mode=strategy_state["allocation_mode"],
        )
        return state.to_dict()

    def _reject_policy_override(self, ckpt: StaticMixtureStateV4) -> None:
        """Fail a resume whose explicit policy args contradict the checkpoint.

        A resume restores the policy frozen in the checkpoint, so an explicit
        constructor arg that disagrees would be silently dropped. Only args the
        caller set explicitly are compared, so a default that has since moved
        still resumes cleanly.
        """
        seeded = self._alloc_config
        candidates = (
            ("exhausted_policy", seeded.exhausted_policy, ckpt.exhausted_policy),
            (
                "reshuffle_on_repeat",
                seeded.reshuffle_on_repeat,
                ckpt.reshuffle_on_repeat,
            ),
            ("max_repeats", seeded.max_repeats, ckpt.max_repeats),
            ("stop_after_passes", seeded.stop_after_passes, ckpt.stop_after_passes),
        )
        conflicts = [
            f"{name}: constructor={got!r} != checkpoint={want!r}"
            for name, got, want in candidates
            if name in self._explicit_policy_fields and got != want
        ]
        if conflicts:
            raise RuntimeError(
                "Constructor exhaustion-policy args contradict the checkpoint, "
                "which a resume cannot honor (the frozen policy is restored "
                "instead). Drop the conflicting args or resume from a matching "
                "config: " + "; ".join(conflicts)
            )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ckpt = StaticMixtureStateV4.load(state)

        self._verify_base_state(int(ckpt.lane_id), int(ckpt.canonical_replicas))

        is_accumulator_ckpt = ckpt.allocation_mode == "accumulator"

        self._seed = ckpt.seed
        # Replay the routing the checkpoint was produced with; pre-fix states
        # have no field and the v3 -> v4 migration fills "modulo".
        self._lane_assignment = ckpt.lane_assignment
        self._perm_block = -1
        self._perm_order = []
        self._shuffle_block_size_spec = ckpt.shuffle_block_size_spec

        # A different dataset set/order can't be reconciled — fail loudly.
        if not ckpt.dataset_ids:
            raise RuntimeError("Checkpoint missing or invalid dataset_ids mapping.")
        ckpt_dataset_ids = {
            str(name): int(dataset_id) for name, dataset_id in ckpt.dataset_ids.items()
        }
        current_dataset_ids = {
            ds.name: dataset_id for dataset_id, ds in enumerate(self._datasets)
        }
        if ckpt_dataset_ids != current_dataset_ids:
            raise RuntimeError(
                "Checkpoint dataset_ids do not match current dataset ordering. "
                f"checkpoint={ckpt_dataset_ids}, current={current_dataset_ids}"
            )

        # The exhaustion-policy block (exhausted_policy / reshuffle_on_repeat /
        # max_repeats / stop_after_passes) is taken from the checkpoint below,
        # just like seed / chunk_size / weights: a resumed run keeps the policy
        # it was frozen with, so a default that has since moved (e.g. the
        # stop_after_passes default) resumes cleanly and old checkpoints keep
        # old behavior. The constructor's policy args only seed fresh runs, so
        # an explicit arg that contradicts the checkpoint is rejected rather
        # than silently dropped.
        self._reject_policy_override(ckpt)

        # Rebuild cursors deterministically and set positions
        self._cursors.clear()
        self._dataset_ids.clear()
        self._datasets_by_id.clear()
        self._knobs_by_name = {}

        if not ckpt.cursor_states and is_accumulator_ckpt:
            raise RuntimeError("Accumulator-mode checkpoints require cursor_states")

        shared_seed = self._seed
        shared_shuffle_shards = bool(ckpt.knobs["shuffle_shards"])
        shared_shuffle_within_shard = bool(ckpt.knobs["shuffle_within_shard"])

        for dataset_id, ds in enumerate(self._datasets):
            self._dataset_ids[ds.name] = dataset_id
            self._datasets_by_id[dataset_id] = ds

            ds_knobs = _DatasetKnobs(
                seed=shared_seed,
                shuffle_shards=shared_shuffle_shards,
                shuffle_within_shard=shared_shuffle_within_shard,
                shuffle_block_size=ckpt.cursor_block_sizes[ds.name],
            )
            self._knobs_by_name[ds.name] = ds_knobs
            cur = _DatasetCursor(dataset_id, ds.ids(), ds.counts(), ds_knobs)
            reshuffle_for_ds = ckpt.reshuffle_on_repeat[ds.name]
            if ds.name in ckpt.cursor_states:
                cur.restore_checkpoint_state(
                    ckpt.cursor_states[ds.name],
                    reshuffle=reshuffle_for_ds,
                )
            else:
                # Pre-strategy v1 checkpoint had only cursor_positions/_epochs.
                cur.restore_checkpoint_state(
                    {
                        "epoch": int(ckpt.cursor_epochs.get(ds.name, 0)),
                        "position": int(ckpt.cursor_positions.get(ds.name, 0)),
                    },
                    reshuffle=reshuffle_for_ds,
                )
            self._cursors[ds.name] = cur

        self._global_chunk_index = int(ckpt.global_chunk_index)

        self._alloc_config = _AllocationConfig(
            component_order=tuple(ckpt.component_order),
            weights=dict(ckpt.weights),
            chunk_size=int(ckpt.chunk_size),
            exhausted_policy=ckpt.exhausted_policy,
            reshuffle_on_repeat=ckpt.reshuffle_on_repeat,
            max_repeats=ckpt.max_repeats,
            stop_after_passes=ckpt.stop_after_passes,
        )
        self._weights = dict(ckpt.weights)

        if is_accumulator_ckpt:
            if ckpt.accumulators is None:
                raise RuntimeError(
                    "Accumulator-mode checkpoint missing accumulators field"
                )
            self._validate_repeat_capacity()
            self._strategy = AccumulatorStrategy(
                config=self._alloc_config,
                accumulators={str(k): float(v) for k, v in ckpt.accumulators.items()},
            )
        else:
            self._strategy = LegacyFixedStrategy(config=self._alloc_config)

        self.total_samples = (
            float("inf") if _is_unbounded(self._alloc_config) else len(self)
        )
