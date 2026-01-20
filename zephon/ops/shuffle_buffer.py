# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Deterministic shuffle buffer operator."""

from random import Random
from typing import TypeVar

from zephon.core.accumulators import Accumulator, CountingAccumulator
from zephon.core.constants import ContributorRef, SampleRecord
from zephon.core.op_base import DefaultSetup, OpContext
from zephon.core.traits import OpTraits
from zephon.utils.seeding import batch_seed

T = TypeVar("T", bound=SampleRecord)


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


class ShuffleBuffer(DefaultSetup):
    """Deterministically shuffle runner-sized micro-batches.

    This operator is deliberately *stateless*: it shuffles each incoming batch
    (as grouped by the runner according to ``buffer_size``) using a seed that is
    derived from the configured seed and the batch contents. Because the runner
    handles buffering on a single pump thread, the batch boundaries are
    deterministic across degrees of parallelism; keeping the operator stateless
    avoids sharding RNG state across workers.
    """

    def __init__(self, buffer_size: int, seed: int = 0) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        DefaultSetup.__init__(self)
        self.buffer_size = int(buffer_size)
        self.seed = int(seed)

    def setup(
        self,
        ctx: OpContext,
        stage_index: int,
        stage_name: str,
        op_index: int,
        collect_stats: bool,
    ) -> None:
        super().setup(ctx, stage_index, stage_name, op_index, collect_stats)

    def traits(self) -> OpTraits:
        # Stateless; deterministic even when multiple worker instances are present.
        # Keep suggested parallelism at 1 to avoid oversubscribing by default.
        return OpTraits(indexable=False, preserves_cursor_order=False, parallelism=1)

    def accumulator(self, *, deterministic: bool) -> Accumulator[SampleRecord]:
        # Use count-only accumulator (no time-based flushing) to ensure
        # deterministic batch boundaries for shuffling.
        return CountingAccumulator[SampleRecord](max_batch=self.buffer_size)

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        return self.process_many([elem])

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        if not elems:
            return []
        rng = Random(batch_seed(self.seed, elems))
        rng.shuffle(elems)
        _redistribute_closers_in_place(elems)
        return elems


__all__ = ["ShuffleBuffer"]
