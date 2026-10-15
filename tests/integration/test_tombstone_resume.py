# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from zephon import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.types import SampleBatch, SampleRecord
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def _identity(item: SampleRecord | SampleBatch) -> tuple[Any, ...]:
    records = item.records if isinstance(item, SampleBatch) else (item,)
    return tuple((record.meta.cursor, record.payload) for record in records)


def _drop_ids(payload: dict[str, Any]) -> dict[str, Any] | None:
    return None if payload["id"] in {2, 5} else payload


def _drop_short_child(payload: dict[str, Any]) -> dict[str, Any] | None:
    return payload if len(payload["input_ids"]) == 8 else None


@pytest.mark.parametrize(
    ("monotone", "batched", "runner", "mtp_mode"),
    [
        pytest.param(True, False, "threads", False, id="monotone-records"),
        pytest.param(True, True, "threads", False, id="monotone-batches"),
        pytest.param(False, False, "threads", False, id="shuffle-records"),
        pytest.param(False, True, "threads", False, id="shuffle-batches"),
        pytest.param(True, False, "process", False, id="process-runner"),
        pytest.param(False, False, "threads", True, id="mtp-ack"),
    ],
)
def test_resume_when_tombstones_overtake_buffered_payloads(
    runner: str,
    mtp_mode: bool,
    batched: bool,
    monotone: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The test tokenizer hashes words with Python's hash(); keep its token
    # IDs stable across the fresh subprocesses used by MTP and restore.
    monkeypatch.setenv("PYTHONHASHSEED", "0")

    def build() -> Pipeline:
        rows = [{"id": i, "text": "word " * 16 + "end"} for i in range(8)]
        work = StaticMixtureWorkSource(
            [Dataset.from_dict("d", {0: InMemoryShard(rows)})],
            {"d": 1.0},
            chunk_size=1,
            seed=3,
        )
        pipe = Pipeline(work)
        if monotone:
            # The map buffers real records while tombstones bypass it.
            # A downstream batch alone does not reproduce the original bug.
            pipe.map_transform(_drop_ids)
            pipe.map_transform(lambda p: p)
        else:
            # Each source produces children of length 8, 8, 1. Dropping the
            # closing child sets every completion bit before shuffle emits
            # its buffered siblings. There is no delivered cursor yet.
            pipe.tokenize(
                tokenizer_id="__fallback__",
                field="text",
                max_length=8,
                split_long_samples=True,
                special_tokens="none",
            )
            pipe.map_transform(_drop_short_child)
            pipe.shuffle(seed=7)
        if batched:
            pipe.batch(2)
        return pipe.options(
            runner=runner,
            mtp_mode=mtp_mode,
            canonical_replicas=1,
            max_workers=2,
            deterministic=True,
            default_stage_prefetch=0,
            prefetch_batches=0,
        )

    baseline = [_identity(item) for item in build()]
    assert sum(len(item) for item in baseline) == (6 if monotone else 16)
    # The first delivery is sufficient to expose both premature monotone
    # progress and eviction before any real cursor exists. Keep this focused
    # rather than sweeping all checkpoint positions and runner combinations.
    pipe = build()
    iterator = iter(pipe)
    try:
        prefix = [_identity(next(iterator))]
        checkpoint = pipe.checkpoint()
    finally:
        iterator.close()
    assert prefix == baseline[:1]

    resumed = build()
    resumed.restore(checkpoint)
    if not monotone and not batched and not mtp_mode:
        # Checkpoint again after replay has rebuilt completion state from
        # dropped prefix records. One inline case covers this extra transition.
        iterator = iter(resumed)
        try:
            prefix.append(_identity(next(iterator)))
            checkpoint = resumed.checkpoint()
        finally:
            iterator.close()
        resumed = build()
        resumed.restore(checkpoint)
    assert prefix + [_identity(item) for item in resumed] == baseline
