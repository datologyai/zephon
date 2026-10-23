# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from zephon import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.work import StaticMixtureWorkSource


def _pipeline(**options: Any) -> Pipeline:
    work = StaticMixtureWorkSource(
        [
            Dataset.from_dict(name, {0: InMemoryShard(list(range(100)))})
            for name in ("alpha", "beta")
        ],
        {"alpha": 0.5, "beta": 0.5},
        chunk_size=4,
        exhausted_policy="repeat",
    )
    return Pipeline(work).batch(8).options(mtp_mode=False, **options)


@pytest.mark.parametrize("global_batch_size", [0, -8, True, 8.0])
def test_engine_rejects_invalid_global_batch_size(global_batch_size: Any) -> None:
    pipe = _pipeline(canonical_replicas=2, global_batch_size=global_batch_size)
    with pytest.raises(ValueError, match="global_batch_size must be a positive"):
        next(iter(pipe))


def test_global_batch_size_checks_lane_windows_for_any_pipeline() -> None:
    pipe = _pipeline(canonical_replicas=128, global_batch_size=512)
    with pytest.raises(
        ValueError, match=r"Use canonical_replicas=64, or an output batch size of 4"
    ):
        next(iter(pipe))


def test_matching_global_batch_size_runs() -> None:
    iterator = iter(_pipeline(canonical_replicas=2, global_batch_size=16))
    try:
        assert len(next(iterator).records) == 8
    finally:
        iterator.close()
