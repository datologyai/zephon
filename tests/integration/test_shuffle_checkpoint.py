# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.integration.test_elastic_continuation import (
    consume_until,
    make_dataset,
)
from zephon.api import Pipeline as PublicPipeline
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def _pipe(buffer_size: int, seed: int, sample_count: int = 128) -> PublicPipeline:
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
        .shuffle(buffer_size=buffer_size, seed=seed)
        .tokenize(tokenizer_id="__fallback__", parallelism=2)
    )


def test_shuffle_checkpoint_resume_matches_baseline() -> None:
    buffer_size = 16
    seed = 1234
    baseline, _ = consume_until(_pipe(buffer_size, seed))

    cut = len(baseline) // 2 + 5
    p1 = _pipe(buffer_size, seed)
    prefix, ckpt = consume_until(p1, flat_limit=cut)
    assert ckpt is not None
    assert prefix == baseline[:cut]

    p2 = _pipe(buffer_size, seed)
    p2._ensure()
    assert p2._engine is not None
    p2._engine.load_state_dict(ckpt, replay=True)
    suffix, _ = consume_until(p2)

    assert prefix + suffix == baseline
