# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Integration test illustrating checkpoint loss under cross-chunk reordering."""

from typing import Sequence

import pytest

from zephon import Pipeline as PublicPipeline
from zephon.io import Dataset, InMemoryShard
from zephon.ops.accumulators import Accumulator, ReadyBatch
from zephon.ops.base import BaseOp
from zephon.ops.traits import OpTraits
from zephon.types import SampleRecord
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


class _DeferringAccumulator(Accumulator[SampleRecord]):
    """Accumulator that defers the first element until flush, reordering chunk_ids."""

    def __init__(self) -> None:
        self._first: SampleRecord | None = None

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        return self._first is not None

    def push_many(
        self, elems: Sequence[SampleRecord]
    ) -> list[ReadyBatch[SampleRecord]]:
        ready: list[ReadyBatch[SampleRecord]] = []
        for elem in elems:
            if self._first is None:
                # Buffer the first element
                self._first = elem
            else:
                # Emit subsequent elements immediately
                ready.append(([elem], 0))
        return ready

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[SampleRecord]]:
        if self._first is None:
            return []
        elem = self._first
        self._first = None
        return [([elem], 0)]


class _DeferFirstOp(BaseOp):
    """Buffer the first element via accumulator; emit it only at flush, reordering chunk_ids."""

    def __init__(self) -> None:
        super().__init__()

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=True,
            preserves_cursor_order=False,
            parallelism=1,
            batch_shape_sensitive=False,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict
    ) -> Accumulator[SampleRecord]:
        return _DeferringAccumulator()

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        # The accumulator handles deferral; operator just passes through
        return [elem]

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        # The accumulator handles deferral; operator just passes through
        return list(elems)


def _make_pipe(sample_count: int, mtp_mode: bool = False) -> PublicPipeline:
    rows = [{"text": f"s{i}"} for i in range(sample_count)]
    ds = Dataset.from_dict("demo", {0: InMemoryShard(rows)})
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=1,
        seed=0,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipe = PublicPipeline(work).decode_text()
    # Inject the reordering operator directly.
    node = pipe._graph.add(
        "defer_first",
        _DeferFirstOp(),
        pipe._tail,
        placement="local",
        parallelism=1,
    )
    pipe._tail = node
    pipe = pipe.options(
        deterministic=True,
        default_stage_prefetch=16,
        prefetch_batches=0,
        mtp_mode=mtp_mode,
    )
    return pipe


@pytest.mark.parametrize("mtp_mode", [False, True], ids=["inline", "mtp"])
def test_checkpoint_breaks_on_cross_chunk_reorder(mtp_mode: bool) -> None:
    """Reordering newer chunk ahead of older causes the older to disappear on resume."""
    total = 4

    # Baseline: full run without checkpoint
    baseline_pipe = _make_pipe(total, mtp_mode=mtp_mode)
    baseline_texts = list(iter(baseline_pipe))

    # Run again, checkpoint immediately after first emitted record (chunk 1), then resume.
    pipe1 = _make_pipe(total, mtp_mode=mtp_mode)
    it1 = iter(pipe1)
    first = next(it1)  # emits chunk_id=1, leaving chunk_id=0 buffered
    ckpt = pipe1.checkpoint()
    it1.close()

    # Resume from checkpoint
    pipe2 = _make_pipe(total, mtp_mode=mtp_mode)
    pipe2.restore(ckpt)
    resumed_texts = list(iter(pipe2))

    observed = [first] + resumed_texts

    # The buffered first element (chunk 0) is lost on resume; observed differs from baseline.
    assert observed == baseline_texts
