# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint bytes retain replay state through shuffled, packed training batches."""

from typing import Any

import pytest

from zephon import Dataset, InMemoryShard, Pipeline, StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def _pipeline() -> Pipeline:
    dataset = Dataset.from_dict(
        "tokens", {0: InMemoryShard([{"tokens": [i + 1] * 3} for i in range(96)])}
    )
    work = StaticMixtureWorkSource(
        [dataset],
        {"tokens": 1.0},
        chunk_size=8,
        seed=7,
        shuffle_within_shard=True,
        shuffle_block_size=16,
    )
    return (
        Pipeline(work)
        .shuffle(buffer_size=9, seed=11)
        .pack_flat(max_length=13, algorithm="first_fit", num_bins=4, pad_token_id=0)
        .batch(microbatch_size=2)
        .options(mtp_mode=False)
    )


def _drain(pipe: Pipeline) -> list[dict[str, Any]]:
    return [batch.to_training(dtype=None, return_labels=True) for batch in pipe]


def test_checkpoint_bytes_resume_matches_uninterrupted_training() -> None:
    baseline = _drain(_pipeline())
    source = _pipeline()
    iterator = iter(source)
    try:
        prefix = [
            next(iterator).to_training(dtype=None, return_labels=True) for _ in range(2)
        ]
        blob = source.checkpoint(as_bytes=True)
    finally:
        iterator.close()

    restored = _pipeline()
    restored.restore(blob)
    assert any(restored.checkpoint()["inflight"].values())
    assert prefix + _drain(restored) == baseline
