# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Integration test illustrating checkpoint loss under cross-chunk reordering."""

import pytest

from zephon.api import Pipeline as PublicPipeline
from zephon.core.constants import SampleRecord
from zephon.core.op_base import DefaultFinalize, DefaultSetup, Op
from zephon.core.traits import Buffering, OpTraits
from zephon.io import Dataset, InMemoryShard
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


class _DeferFirstOp(
    DefaultSetup, DefaultFinalize[SampleRecord], Op[SampleRecord, SampleRecord]
):
    """Buffer the first element; emit it only at finalize, reordering chunk_ids."""

    def __init__(self) -> None:
        DefaultSetup.__init__(self)
        self._buffer: SampleRecord | None = None

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=True,
            preserves_cursor_order=False,
            parallelism=1,
            batch_shape_sensitive=False,
        )

    def buffering(self) -> Buffering | None:
        return None

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        if self._buffer is None:
            self._buffer = elem
            return []
        return [elem]

    def finalize(self) -> list[SampleRecord]:
        if self._buffer is None:
            return []
        buf, self._buffer = self._buffer, None
        return [buf]


def _make_pipe(sample_count: int) -> PublicPipeline:
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
        default_stage_prefetch=0,
        prefetch_batches=0,
    )
    return pipe


def test_checkpoint_breaks_on_cross_chunk_reorder() -> None:
    """Reordering newer chunk ahead of older causes the older to disappear on resume."""
    total = 4

    # Baseline: full run without checkpoint
    baseline_pipe = _make_pipe(total)
    baseline_texts = list(iter(baseline_pipe))

    # Run again, checkpoint immediately after first emitted record (chunk 1), then resume.
    pipe1 = _make_pipe(total)
    pipe1._ensure()
    assert pipe1._engine is not None
    eng1 = pipe1._engine
    it1 = iter(pipe1)
    first = next(it1)  # emits chunk_id=1, leaving chunk_id=0 buffered
    ckpt = eng1.state_dict()
    eng1.close()

    # Resume from checkpoint
    pipe2 = _make_pipe(total)
    pipe2._ensure()
    assert pipe2._engine is not None
    pipe2._engine.load_state_dict(ckpt, replay=True)
    resumed_texts = list(iter(pipe2))

    observed = [first] + resumed_texts

    # The buffered first element (chunk 0) is lost on resume; observed differs from baseline.
    assert observed == baseline_texts
