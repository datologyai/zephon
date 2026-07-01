# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Deterministic shuffle buffer operator."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import repeat
from random import Random
from typing import Any, Literal, TypeAlias, TypeVar

import numpy as np

from zephon.core.accumulators import Accumulator, CountingAccumulator, ReadyBatch
from zephon.core.constants import (
    ContributorRef,
    LaneId,
    SampleMeta,
    SampleRecord,
    lane_of,
)
from zephon.core.op_base import DefaultSetup
from zephon.core.traits import OpTraits
from zephon.utils.seeding import batch_seed

DEFAULT_BUFFER_SIZE = 8192
ShuffleAlgorithm: TypeAlias = Literal["block", "streaming", "block_warmup"]
_MASK_64 = (1 << 64) - 1
_MASK_32 = 0xFFFFFFFF
_FLUSH_SALT = 0xD1B54A32D192ED03

T = TypeVar("T", bound=SampleRecord)
_BaseOffsetKey: TypeAlias = tuple[int, int, int]

# Hash constants shared by scalar and vector paths.
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB
_GOLDEN = 0x9E3779B97F4A7C15
_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193

# np.uint64 mirrors for vectorized arithmetic; wraparound matches scalar masking.
_NP_MIX_A = np.uint64(_MIX_A)
_NP_MIX_B = np.uint64(_MIX_B)
_NP_GOLDEN = np.uint64(_GOLDEN)
_NP_S30 = np.uint64(30)
_NP_S27 = np.uint64(27)
_NP_S31 = np.uint64(31)

# Above this many evictions in one call, the vectorized avalanche beats the
# per-element scalar path; below it the numpy array overhead dominates.
_VEC_MIN = 32


@dataclass(slots=True)
class _BaseOffsetState:
    buffered: int = 0
    closer_seen: bool = False


@dataclass(slots=True)
class _StreamingLaneState:
    buffer: list[SampleRecord] = field(default_factory=list)
    emitted_since_reset: int = 0
    base_offsets: dict[_BaseOffsetKey, _BaseOffsetState] = field(default_factory=dict)


def _mix64(value: int) -> int:
    """Return a stable 64-bit avalanche of ``value`` (scalar reference)."""
    value &= _MASK_64
    value ^= value >> 30
    value = (value * _MIX_A) & _MASK_64
    value ^= value >> 27
    value = (value * _MIX_B) & _MASK_64
    value ^= value >> 31
    return value & _MASK_64


def _fnv_fold(h: int, value: int) -> int:
    """Fold both 32-bit halves into a 32-bit FNV-1a accumulator."""
    value = int(value) & _MASK_64
    h = ((h ^ (value & _MASK_32)) * _FNV_PRIME) & _MASK_32
    h = ((h ^ (value >> 32)) * _FNV_PRIME) & _MASK_32
    return h


def _seed_from_cursor(meta: SampleMeta) -> int:
    """Cheap deterministic 32-bit salt from cursor identity."""
    h = _fnv_fold(_FNV_OFFSET, meta.chunk_id)
    h = _fnv_fold(h, meta.chunk_offset)
    for item in meta.lineage:
        h = _fnv_fold(h, item)
    for item in meta.sample_id:
        h = _fnv_fold(h, item)
    return h


def _combine64(acc: int, value: int) -> int:
    return _mix64(acc ^ _mix64(value + _GOLDEN))


def _stable_index_scalar(
    seed_mixed: int, lane: int, salt: int, counter: int, size: int
) -> int:
    acc = _combine64(seed_mixed, lane)
    acc = _combine64(acc, salt)
    acc = _combine64(acc, counter)
    acc = _combine64(acc, size)
    return acc % size


def _mix64_vec(values: np.ndarray) -> np.ndarray:
    v = values ^ (values >> _NP_S30)
    v = v * _NP_MIX_A
    v = v ^ (v >> _NP_S27)
    v = v * _NP_MIX_B
    v = v ^ (v >> _NP_S31)
    return v


def _combine_vec(acc: np.ndarray, value: np.ndarray) -> np.ndarray:
    return _mix64_vec(acc ^ _mix64_vec(value + _NP_GOLDEN))


def _stable_index_vec(
    *,
    seed_mixed: int,
    lanes: Sequence[int] | int,
    salts: Sequence[int] | int,
    counters: Sequence[int],
    sizes: Sequence[int],
) -> list[int]:
    """Choose deterministic pseudo-random indices in ``[0, size)`` per element.

    Pure function of ``(seed, lane, salt, counter, size)`` for each element, so
    output is independent of how inputs are chunked across ``push_many`` calls.
    Scalar ``lanes``/``salts`` broadcast across the batch.
    """
    sizes_a = np.asarray(sizes, dtype=np.uint64)
    # uint64 wraparound is the intended modular arithmetic (matches the scalar
    # ``& _MASK_64``); numpy flags it as an overflow on broadcast scalars.
    with np.errstate(over="ignore"):
        acc = np.full(sizes_a.shape, np.uint64(seed_mixed), dtype=np.uint64)
        acc = _combine_vec(acc, np.asarray(lanes, dtype=np.uint64))
        acc = _combine_vec(acc, np.asarray(salts, dtype=np.uint64))
        acc = _combine_vec(acc, np.asarray(counters, dtype=np.uint64))
        acc = _combine_vec(acc, sizes_a)
        return (acc % sizes_a).tolist()


def _victim_indices(
    *,
    seed_mixed: int,
    lanes: Sequence[int] | int,
    salts: Sequence[int] | int,
    counters: Sequence[int],
    sizes: Sequence[int],
) -> list[int]:
    """Compute victim indices, vectorizing only when it pays off.

    ``lanes``/``salts`` may be scalars (constant across the call, e.g. a flush
    drain) — numpy broadcasts them, and the scalar path expands them lazily.
    """
    n = len(sizes)
    if n >= _VEC_MIN:
        return _stable_index_vec(
            seed_mixed=seed_mixed,
            lanes=lanes,
            salts=salts,
            counters=counters,
            sizes=sizes,
        )
    lane_seq = lanes if not isinstance(lanes, int) else repeat(lanes)
    salt_seq = salts if not isinstance(salts, int) else repeat(salts)
    return [
        _stable_index_scalar(seed_mixed, lane, salt, counter, size)
        for lane, salt, counter, size in zip(lane_seq, salt_seq, counters, sizes)
    ]


def _streaming_warmup_min_buffer(buffer_size: int) -> int:
    """Initial reservoir window before streaming shuffle starts emitting.

    Small but non-trivial (128 for the default 8192 buffer); the reservoir then
    grows one record per emission until it reaches ``buffer_size``.
    """
    if buffer_size <= 1:
        return 0
    return max(1, min(128, buffer_size // 16))


def _redistribute_closers_in_place(records: list[T]) -> None:
    """Shift ``is_last_child`` to the last occurrence per base offset after a reorder.

    Internal helper for reorder-only operators: assumes ``records`` is a permutation of
    the inputs (no drops or inserts). Operators that drop a closing child must emit a
    tombstone instead of relying on this helper.
    """
    base_last_idx: dict[tuple[int, int], int] = {}
    base_has_closer: set[tuple[int, int]] = set()
    refs_by_record: list[tuple[ContributorRef, ...] | None] = []
    seen_duplicate = False

    # First pass: find the last index for every base offset and whether a closer existed.
    for idx, rec in enumerate(records):
        meta = rec.meta
        if meta.contributors:
            refs = meta.contributors
            refs_by_record.append(refs)
            for ref in refs:
                key = ref.cursor.base_offset
                if key in base_last_idx:
                    seen_duplicate = True
                base_last_idx[key] = idx
                if ref.is_last_child:
                    base_has_closer.add(key)
        else:
            refs_by_record.append(None)
            key = meta.cursor.base_offset
            if key in base_last_idx:
                seen_duplicate = True
            base_last_idx[key] = idx
            base_has_closer.add(key)  # default contributor is closing

    if not seen_duplicate:
        return

    # Second pass: rewrite contributors only where the closer flag needs to move.
    for idx, rec in enumerate(records):
        meta = rec.meta
        refs = refs_by_record[idx]

        if refs is None:
            key = meta.cursor.base_offset
            should_close = key in base_has_closer and base_last_idx[key] == idx
            if not should_close:
                rec.meta = meta.with_contributors(
                    (ContributorRef(cursor=meta.cursor, is_last_child=False),)
                )
            continue

        changed = False
        new_refs: list[ContributorRef] = []
        for ref in refs:
            key = ref.cursor.base_offset
            should_close = key in base_has_closer and base_last_idx[key] == idx
            if ref.is_last_child == should_close:
                new_refs.append(ref)
            else:
                changed = True
                new_refs.append(
                    ContributorRef(cursor=ref.cursor, is_last_child=should_close)
                )

        if changed:
            rec.meta = meta.with_contributors(tuple(new_refs))


class StreamingShuffleAccumulator(Accumulator[SampleRecord]):
    """Per-lane deterministic streaming shuffle accumulator.

    Emits during refill: after a small retained window, each arrival evicts one
    deterministic victim. Victim selection is a pure function of seed, lane,
    cursor-derived salt, emit counter, and reservoir size.
    """

    def __init__(self, buffer_size: int, seed: int = 0) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self._buffer_size = int(buffer_size)
        self._seed = int(seed)
        self._seed_mixed = _mix64(self._seed)
        self._warmup_min_buffer = _streaming_warmup_min_buffer(self._buffer_size)
        self._lanes: dict[LaneId, _StreamingLaneState] = {}

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        """Return True if buffered records remain (in ``lane_id`` if given)."""
        if lane_id is None:
            return any(state.buffer for state in self._lanes.values())
        state = self._lanes.get(lane_id)
        return bool(state and state.buffer)

    def push_many(
        self, elems: Sequence[SampleRecord]
    ) -> list[ReadyBatch[SampleRecord]]:
        """Accumulate records and emit deterministic streaming-shuffle records."""
        if not elems:
            return []

        buffer_size = self._buffer_size
        warmup = self._warmup_min_buffer

        # Pass 1: emit schedule.  ``target`` grows monotonically with the emit
        # counter, so at most one eviction happens per arrival.
        emit_arrival: list[int] = []
        lanes: list[int] = []
        salts: list[int] = []
        counters: list[int] = []
        sizes: list[int] = []
        sim_len: dict[int, int] = {}
        sim_cnt: dict[int, int] = {}
        for i, elem in enumerate(elems):
            lane_id = lane_of(elem)
            blen = sim_len.get(lane_id)
            if blen is None:
                state = self._state_for_lane(lane_id)
                blen = len(state.buffer)
                cnt = state.emitted_since_reset
            else:
                cnt = sim_cnt[lane_id]
            blen += 1
            target = 0 if buffer_size == 1 else min(buffer_size, warmup + cnt)
            if blen > target:
                emit_arrival.append(i)
                lanes.append(lane_id)
                salts.append(_seed_from_cursor(elem.meta))
                counters.append(cnt)
                sizes.append(blen)
                cnt += 1
                blen -= 1
            sim_len[lane_id] = blen
            sim_cnt[lane_id] = cnt

        # Pass 2: victim indices for every scheduled eviction.
        indices = (
            _victim_indices(
                seed_mixed=self._seed_mixed,
                lanes=lanes,
                salts=salts,
                counters=counters,
                sizes=sizes,
            )
            if emit_arrival
            else []
        )

        # Pass 3: replay the append/swap-pop interleaving deterministically.
        emitted: list[SampleRecord] = []
        emit_ptr = 0
        n_emits = len(emit_arrival)
        for i, elem in enumerate(elems):
            lane_id = lane_of(elem)
            state = self._lanes[lane_id]
            self._register_record(state, lane_id, elem)
            state.buffer.append(elem)
            if emit_ptr < n_emits and emit_arrival[emit_ptr] == i:
                record = _swap_pop(state.buffer, indices[emit_ptr])
                emit_ptr += 1
                state.emitted_since_reset += 1
                emitted.append(self._rewrite_closers_for_emit(state, lane_id, record))

        return [(emitted, 0)] if emitted else []

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[SampleRecord]]:
        """Drain buffered records; ``lane_id`` limits the reset to one lane."""
        keys = [lane_id] if lane_id is not None else sorted(self._lanes)
        ready: list[ReadyBatch[SampleRecord]] = []

        for key in keys:
            state = self._lanes.get(key)
            if state is not None and state.buffer:
                # Fixed flush salt, rising emit counter, shrinking reservoir size.
                # Lane and salt are constant, so they broadcast (no per-row list).
                length = len(state.buffer)
                base = state.emitted_since_reset
                indices = _victim_indices(
                    seed_mixed=self._seed_mixed,
                    lanes=key,
                    salts=_FLUSH_SALT,
                    counters=range(base, base + length),
                    sizes=range(length, 0, -1),
                )
                emitted: list[SampleRecord] = []
                for index in indices:
                    record = _swap_pop(state.buffer, index)
                    emitted.append(self._rewrite_closers_for_emit(state, key, record))
                ready.append((emitted, 0))
            self._lanes.pop(key, None)

        return ready

    def _state_for_lane(self, lane_id: LaneId) -> _StreamingLaneState:
        state = self._lanes.get(lane_id)
        if state is None:
            state = _StreamingLaneState()
            self._lanes[lane_id] = state
        return state

    def _register_record(
        self,
        state: _StreamingLaneState,
        lane_id: LaneId,
        record: SampleRecord,
    ) -> None:
        # 1:1 records (no contributors) own a unique base offset and always
        # close it regardless of shuffle position, so they need no tracking.
        if not record.meta.contributors:
            return
        for ref in record.meta.contributors:
            key = _base_offset_key(lane_id, ref)
            base_state = state.base_offsets.get(key)
            if base_state is None:
                base_state = _BaseOffsetState()
                state.base_offsets[key] = base_state
            base_state.buffered += 1
            if ref.is_last_child:
                base_state.closer_seen = True

    def _rewrite_closers_for_emit(
        self,
        state: _StreamingLaneState,
        lane_id: LaneId,
        record: SampleRecord,
    ) -> SampleRecord:
        meta = record.meta
        refs = meta.contributors
        if not refs:
            return record  # untracked 1:1 record: already its own sole closer

        changed = False
        new_refs: list[ContributorRef] = []
        keys_to_drop: list[_BaseOffsetKey] = []

        for ref in refs:
            key = _base_offset_key(lane_id, ref)
            base_state = state.base_offsets.get(key)
            if base_state is None:
                raise RuntimeError(
                    "Streaming shuffle emitted a record whose base offset "
                    + "was not registered"
                )

            base_state.buffered -= 1
            should_close = base_state.closer_seen and base_state.buffered == 0
            if ref.is_last_child == should_close:
                new_refs.append(ref)
            else:
                changed = True
                new_refs.append(
                    ContributorRef(cursor=ref.cursor, is_last_child=should_close)
                )

            if base_state.buffered == 0:
                keys_to_drop.append(key)

        for key in keys_to_drop:
            state.base_offsets.pop(key, None)

        if changed:
            record.meta = meta.with_contributors(tuple(new_refs))
        return record


def _swap_pop(buffer: list[SampleRecord], index: int) -> SampleRecord:
    """Remove and return ``buffer[index]`` in O(1) by swapping in the tail."""
    record = buffer[index]
    last = buffer.pop()
    if index < len(buffer):
        buffer[index] = last
    return record


def _base_offset_key(lane_id: LaneId, ref: ContributorRef) -> _BaseOffsetKey:
    # chunk fields may be numpy-sourced; normalize to plain int for uniform keys.
    chunk_id, chunk_offset = ref.cursor.base_offset
    return (lane_id, int(chunk_id), int(chunk_offset))


class WarmupBlockAccumulator(Accumulator[SampleRecord]):
    """Per-lane tumbling-block shuffle with a geometric warmup ramp.

    Counts lane-pure blocks on the pump thread; ``process_many`` shuffles them
    on workers. Each lane starts with a small block and grows by ``growth`` up
    to ``buffer_size`` after every reset.
    """

    def __init__(self, buffer_size: int, *, growth: float = 1.5) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if growth <= 1.0:
            raise ValueError("growth must be > 1.0")
        self._buffer_size = int(buffer_size)
        self._growth = float(growth)
        self._initial_target = min(
            self._buffer_size, max(1, _streaming_warmup_min_buffer(self._buffer_size))
        )
        self._buffers: dict[LaneId, list[SampleRecord]] = {}
        self._targets: dict[LaneId, int] = {}

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        """Return True if a partial block remains (in ``lane_id`` if given)."""
        if lane_id is None:
            return any(self._buffers.values())
        return bool(self._buffers.get(lane_id))

    def push_many(
        self, elems: Sequence[SampleRecord]
    ) -> list[ReadyBatch[SampleRecord]]:
        """Group records into per-lane blocks, emitting each filled block."""
        ready: list[ReadyBatch[SampleRecord]] = []
        bufs = self._buffers
        targets = self._targets
        buffer_size = self._buffer_size
        growth = self._growth
        initial = self._initial_target
        for elem in elems:
            lane_id = lane_of(elem)
            buf = bufs.get(lane_id)
            if buf is None:
                buf = []
                bufs[lane_id] = buf
                targets[lane_id] = initial
            buf.append(elem)
            if len(buf) >= targets[lane_id]:
                ready.append((buf, 0))
                bufs[lane_id] = []
                grown = math.ceil(targets[lane_id] * growth)
                targets[lane_id] = min(buffer_size, max(targets[lane_id] + 1, grown))
        return ready

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[SampleRecord]]:
        """Emit partial blocks; ``lane_id`` restarts only that lane's ramp."""
        keys = [lane_id] if lane_id is not None else sorted(self._buffers)
        ready: list[ReadyBatch[SampleRecord]] = []
        for key in keys:
            buf = self._buffers.get(key)
            if buf:
                ready.append((buf, 0))
            self._buffers.pop(key, None)
            self._targets.pop(key, None)
        return ready


class ShuffleBuffer(DefaultSetup):
    """Deterministically shuffle records with block or streaming buffering.

    ``streaming`` uses a per-lane reservoir accumulator. ``block`` preserves the
    legacy fixed-size block shuffle. ``block_warmup`` keeps block shuffling but
    grows each lane's block size after reset.
    """

    def __init__(
        self,
        buffer_size: int | None = None,
        seed: int = 0,
        *,
        algorithm: ShuffleAlgorithm = "streaming",
        warmup_growth: float = 1.5,
    ) -> None:
        if buffer_size is None:
            buffer_size = DEFAULT_BUFFER_SIZE
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if algorithm not in ("block", "streaming", "block_warmup"):
            raise ValueError(
                "algorithm must be 'block', 'streaming', or 'block_warmup'"
            )
        if warmup_growth <= 1.0:
            raise ValueError("warmup_growth must be > 1.0")
        DefaultSetup.__init__(self)
        self.buffer_size: int = int(buffer_size)
        self.seed = int(seed)
        self.algorithm: ShuffleAlgorithm = algorithm
        self.warmup_growth: float = float(warmup_growth)

    def traits(self) -> OpTraits:
        # Stateless; deterministic even when multiple worker instances are present.
        # Keep suggested parallelism at 1 to avoid oversubscribing by default.
        return OpTraits(indexable=False, preserves_cursor_order=False, parallelism=1)

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        if self.algorithm == "streaming":
            return StreamingShuffleAccumulator(
                buffer_size=self.buffer_size,
                seed=self.seed,
            )
        if self.algorithm == "block_warmup":
            return WarmupBlockAccumulator(
                buffer_size=self.buffer_size, growth=self.warmup_growth
            )
        return CountingAccumulator[SampleRecord](max_batch=self.buffer_size)

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        return self.process_many([elem])

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        if not elems:
            return []
        if self.algorithm == "streaming":
            # Reservoir already produced the final ordered batch in the
            # accumulator; this stage is a pass-through.
            return elems
        # block / block_warmup: shuffle each lane-pure block on the worker.
        rng = Random(batch_seed(self.seed, elems))
        rng.shuffle(elems)
        _redistribute_closers_in_place(elems)
        return elems


__all__ = [
    "ShuffleAlgorithm",
    "ShuffleBuffer",
    "StreamingShuffleAccumulator",
    "WarmupBlockAccumulator",
]
