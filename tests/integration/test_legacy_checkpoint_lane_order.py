# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Exact packed continuation from the engine-v1 state written by b23."""

import warnings
from collections.abc import Generator, Iterable
from typing import cast

import pytest

from zephon import (
    Dataset,
    InMemoryShard,
    Pipeline,
    SampleBatch,
    StaticMixtureWorkSource,
)

pytestmark = pytest.mark.integration


def _pipeline() -> Pipeline:
    ds = Dataset.from_dict(
        name="compat",
        shards={
            0: InMemoryShard(
                [
                    {"input_ids": [i * 1000 + j for j in range(11 + i % 45)]}
                    for i in range(160)
                ]
            )
        },
    )
    work = StaticMixtureWorkSource(
        [ds],
        {"compat": 1.0},
        chunk_size=16,
        seed=42,
        shuffle_shards=True,
        shuffle_within_shard=True,
        shuffle_block_size=32,
    )
    return (
        Pipeline(work)
        .fetch_parallelism(1)
        .pack_flat(
            max_length=33,
            algorithm="best_fit_wrap",
            emit_positions=True,
            pad_token_id=3,
        )
        .batch(2, drop_last=True)
        .options(
            canonical_replicas=4,
            deterministic=True,
            runner="inline",
            prefetch_batches=0,
        )
    )


def _signature(batch: SampleBatch) -> tuple:
    tokens = []
    for record in batch.records:
        assert isinstance(record.payload, dict)
        ids = cast(Iterable[int], record.payload["input_ids"])
        tokens.append(tuple(int(t) for t in ids))
    return tuple(batch.lane_ids), tuple(tokens)


@pytest.mark.parametrize("cut", [8, 9, 10, 11])
def test_resume_b23_packed_checkpoint_preserves_batch_order(cut: int) -> None:
    baseline = [_signature(batch) for batch in _pipeline()]
    pipe = _pipeline()
    iterator = cast(Generator[SampleBatch, None, None], iter(pipe))
    try:
        prefix = [_signature(next(iterator)) for _ in range(cut)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            legacy = pipe.checkpoint()
    finally:
        iterator.close()

    # Reproduce b23's persisted state: engine v1 and WorkChunk v2, with the
    # RR pointer overwritten from source progress instead of delivered batches.
    legacy["version"] = 1
    progress = legacy["progress"]
    next_lane = min(
        range(4),
        key=lambda lane: (
            progress[lane]["chunk_id"],
            progress[lane]["offset"],
            lane,
        ),
    )
    legacy["rr_next_idx"] = {"0:0/1:0,1,2,3": next_lane}
    for chunks in legacy["inflight"].values():
        for chunk in chunks.values():
            chunk["version"] = 2
            chunk.pop("source_exhausted")
    if cut == 8:
        assert legacy["lane_emitted"] == {0: 2, 1: 2, 2: 2, 3: 2}
        assert next_lane == 1  # Next output must be lane 0, despite its progress.

    restored = _pipeline()
    restored.restore(legacy)
    assert prefix + [_signature(batch) for batch in restored] == baseline
