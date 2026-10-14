# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from zephon import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.ops import BaseOp, OpTraits
from zephon.ops.children import spawn_child
from zephon.types import SampleRecord
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("runner", "downstream", "mtp_mode", "dropped", "prefetch"),
    [
        (runner, downstream, mtp_mode, {2, 5}, 0)
        for runner in ("threads", "process")
        for downstream in ("map", "batch")
        for mtp_mode in (False, True)
    ]
    + [
        ("threads", "map", False, {0}, None),
        ("process", "batch", False, {0}, None),
        ("threads", "batch", True, {2, 3}, None),
        ("process", "map", True, {2, 3}, None),
    ],
)
def test_resume_after_mid_pipeline_drops(
    runner: str,
    downstream: str,
    mtp_mode: bool,
    dropped: set[int],
    prefetch: int | None,
) -> None:
    def build() -> Pipeline:
        work = StaticMixtureWorkSource(
            [Dataset.from_dict("d", {0: InMemoryShard([{"id": i} for i in range(8)])})],
            {"d": 1.0},
            chunk_size=2,
            seed=3,
        )
        pipe = Pipeline(work).map_transform(lambda p: None if p["id"] in dropped else p)
        if downstream == "batch":
            pipe.batch(2)
        else:
            pipe.map_transform(lambda p: p)
        if prefetch is not None:
            pipe.options(default_stage_prefetch=prefetch, prefetch_batches=prefetch)
        return pipe.options(
            runner=runner,
            mtp_mode=mtp_mode,
            max_workers=2,
            canonical_replicas=1,
            deterministic=True,
        )

    def identity(item):
        records = item.records if hasattr(item, "records") else [item]
        return tuple((r.meta.as_cursor_key(), r.payload["id"]) for r in records)

    baseline = [identity(item) for item in build()]
    expected_count = 8 - len(dropped)
    if downstream == "batch":
        expected_count -= expected_count % 2
    assert sum(len(item) for item in baseline) == expected_count
    for count in range(1, len(baseline) + 1):
        pipe = build()
        iterator = iter(pipe)
        try:
            prefix = [identity(next(iterator)) for _ in range(count)]
            checkpoint = pipe.checkpoint()
        finally:
            iterator.close()
        resumed = build()
        resumed.restore(checkpoint)
        if count == len(baseline):
            assert list(resumed) == []
            continue
        iterator = iter(resumed)
        try:
            prefix.append(identity(next(iterator)))
            checkpoint = resumed.checkpoint()
        finally:
            iterator.close()
        chained = build()
        chained.restore(checkpoint)
        assert prefix + [identity(item) for item in chained] == baseline, count


@pytest.mark.parametrize("prefetch", [0, None])
def test_monotone_drop_bookkeeping_is_bounded(prefetch: int | None) -> None:
    rows = [{"id": i} for i in range(2000)]
    work = StaticMixtureWorkSource(
        [Dataset.from_dict("d", {0: InMemoryShard(rows)})],
        {"d": 1.0},
        chunk_size=20,
        seed=3,
    )
    pipe = (
        Pipeline(work)
        .map_transform(lambda p: None if p["id"] % 20 == 1 else p)
        .map_transform(lambda p: p)
        .options(runner="threads", max_workers=2, canonical_replicas=1)
    )
    if prefetch is not None:
        pipe.options(default_stage_prefetch=prefetch, prefetch_batches=prefetch)
    count = 0
    iterator = iter(pipe)
    try:
        for _ in iterator:
            count += 1
            engine = pipe._engine
            assert engine is not None
            inflight = engine.inflight_chunks_per_lane[0]
            assert engine._offset_done[0].keys() <= inflight.keys()
            assert engine._offset_done_count[0].keys() == engine._offset_done[0].keys()
    finally:
        iterator.close()
    assert count == 1900


class TwoTextLengths(BaseOp):
    def traits(self) -> OpTraits:
        return OpTraits(indexable=False, preserves_cursor_order=True)

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        return [
            SampleRecord(
                meta=spawn_child(record.meta, index, is_last_child=index == 1),
                payload={"text": "word " * length + "end"},
            )
            for record in elems
            for index, length in enumerate((16, 32))
        ]


@pytest.mark.parametrize("runner", ["threads", "process"])
@pytest.mark.parametrize("drop_final", [False, True])
def test_nested_token_splitting_resume_at_every_position(
    runner: str, drop_final: bool
) -> None:
    def build() -> Pipeline:
        work = StaticMixtureWorkSource(
            [Dataset.from_dict("d", {0: InMemoryShard([{}, {}])})],
            {"d": 1.0},
            chunk_size=2,
            seed=3,
        )
        pipe = (
            Pipeline(work)
            .add_op(TwoTextLengths())
            .tokenize(
                tokenizer_id="__fallback__",
                field="text",
                max_length=8,
                split_long_samples=True,
                special_tokens="none",
            )
        )
        if drop_final:
            pipe.map_transform(lambda p: p if len(p["input_ids"]) == 8 else None)
        return pipe.options(
            runner=runner, deterministic=True, canonical_replicas=1, max_workers=2
        )

    def identity(item: SampleRecord) -> tuple[Any, ...]:
        return item.meta.cursor, len(item.payload["input_ids"])

    records = list(build())
    baseline = [identity(item) for item in records]
    assert len(baseline) == (12 if drop_final else 16)
    if drop_final:
        assert all(len(r.payload["input_ids"]) == 8 for r in records)
    else:
        assert any(len(r.payload["input_ids"]) < 8 for r in records)
        closing = [
            r for r in records if any(ref.is_last_child for ref in r.meta.contributors)
        ]
        assert len(closing) == 2
        assert all(r.meta.lineage[0] == 1 for r in closing)
    for count in range(1, len(baseline) + 1):
        pipe = build()
        iterator = iter(pipe)
        try:
            prefix = [identity(next(iterator)) for _ in range(count)]
            checkpoint = pipe.checkpoint()
        finally:
            iterator.close()
        resumed = build()
        resumed.restore(checkpoint)
        assert prefix + [identity(item) for item in resumed] == baseline, count
