# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import pytest

from tests.integration.test_elastic_continuation import (
    consume_until,
    make_dataset,
)
from zephon.api import Pipeline as PublicPipeline
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
