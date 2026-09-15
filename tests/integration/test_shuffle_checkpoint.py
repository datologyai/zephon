# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Literal

import pytest

from tests.integration.test_elastic_continuation import (
    consume_until,
    make_dataset,
)
from zephon import Pipeline as PublicPipeline
from zephon.types import SampleRecord
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def _pipe(
    buffer_size: int,
    seed: int,
    sample_count: int = 128,
    mtp_mode: bool = False,
    algorithm: Literal["streaming", "block", "block_warmup"] = "streaming",
) -> PublicPipeline:
    ds = make_dataset("alpha", sample_count)
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=8,
        seed=7,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    return (
        PublicPipeline(work)
        .decode_text()
        .shuffle(buffer_size=buffer_size, seed=seed, algorithm=algorithm)
        .tokenize(
            tokenizer_id="__fallback__",
            field="text",
            parallelism=2,
            preserve_upstream_payload=True,
        )
        .options(mtp_mode=mtp_mode)
    )


@pytest.mark.parametrize("mtp_mode", [False, True], ids=["inline", "mtp"])
@pytest.mark.parametrize("algorithm", ["streaming", "block", "block_warmup"])
def test_shuffle_checkpoint_resume_matches_baseline(
    mtp_mode: bool, algorithm: Literal["streaming", "block", "block_warmup"]
) -> None:
    buffer_size = 16
    seed = 1234
    baseline, _ = consume_until(
        _pipe(buffer_size, seed, mtp_mode=mtp_mode, algorithm=algorithm)
    )

    cut = len(baseline) // 2 + 5
    p1 = _pipe(buffer_size, seed, mtp_mode=mtp_mode, algorithm=algorithm)
    prefix, ckpt = consume_until(p1, flat_limit=cut)
    assert ckpt is not None
    assert prefix == baseline[:cut]

    p2 = _pipe(buffer_size, seed, mtp_mode=mtp_mode, algorithm=algorithm)
    p2.restore(ckpt)
    suffix, _ = consume_until(p2)

    assert prefix + suffix == baseline


@pytest.mark.parametrize("algorithm", ["streaming", "block", "block_warmup"])
def test_shuffle_reorders_and_is_lossless_e2e(
    algorithm: Literal["streaming", "block", "block_warmup"],
) -> None:
    """Each algorithm reorders without losing samples."""
    sample_count = 128
    out, _ = consume_until(
        _pipe(buffer_size=16, seed=1234, sample_count=sample_count, algorithm=algorithm)
    )
    natural = [f"alpha-{i}" for i in range(sample_count)]
    assert sorted(out) == sorted(natural), "shuffle must be a lossless permutation"
    assert out != natural, "shuffle must actually reorder the stream"


@pytest.mark.parametrize(
    "runner,mtp_mode",
    [("inline", False), ("threads", False), ("process", False), ("threads", True)],
)
@pytest.mark.parametrize("prefetch_batches", [0, 8])
def test_shuffle_resume_retains_lanes_without_a_flush_boundary(
    runner: str, mtp_mode: bool, prefetch_batches: int
) -> None:
    """One lane's flush must not evict another lane's open shuffle epoch."""
    # 22 chunks: lane 0 reaches its first boundary at 8 chunks, while lanes
    # 1 and 2 exhaust after 7. Their full history must survive checkpoints
    # taken while lane 0 is still delivering its tail.
    ds = make_dataset("alpha", 242)

    def make_pipe() -> PublicPipeline:
        work = StaticMixtureWorkSource(
            [ds],
            {ds.name: 1.0},
            chunk_size=11,
            shuffle_shards=False,
            lane_assignment="modulo",
        )
        return (
            PublicPipeline(work)
            .shuffle(buffer_size=15, seed=17)
            .options(
                runner=runner,
                mtp_mode=mtp_mode,
                canonical_replicas=3,
                max_workers=2,
                default_stage_prefetch=0,
                prefetch_batches=prefetch_batches,
                flush_every_k_chunks=8,
            )
        )

    def per_lane(items: list[SampleRecord]) -> dict[int, list[str]]:
        lanes: dict[int, list[str]] = defaultdict(list)
        for item in items:
            lanes[item.meta.lane_id].append(item.payload["text"])
        return dict(lanes)

    baseline = list(make_pipe())
    assert len(baseline) == 242
    assert len({item.payload["text"] for item in baseline}) == 242

    pipe = make_pipe()
    iterator = iter(pipe)
    try:
        prefix = [next(iterator) for _ in range(232)]
        checkpoint = pipe.checkpoint()
    finally:
        iterator.close()

    restored = make_pipe()
    restored.restore(checkpoint)
    combined = prefix + list(restored)
    # Compare each lane's exact sequence: cross-lane scheduling is independent
    # of whether replay preserves all records and their shuffle order.
    assert per_lane(combined) == per_lane(baseline)
