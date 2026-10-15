# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, cast

import pytest

from zephon import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.ops.children import spawn_child
from zephon.types import SampleCursor, SampleRecord
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def _split_text(records: list[SampleRecord]) -> list[SampleRecord]:
    return [
        SampleRecord(
            meta=spawn_child(record.meta, index, is_last_child=index == 1),
            payload={"text": "word " * length},
        )
        for record in records
        for index, length in enumerate((5, 9))
    ]


def _identity(record: SampleRecord) -> tuple[SampleCursor, tuple[int, ...]]:
    payload = cast(dict[str, list[int]], record.payload)
    return record.meta.cursor, tuple(payload["input_ids"])


@pytest.mark.parametrize("runner", ["threads", "process"])
def test_tokenize_nested_children_and_resume(
    runner: Literal["threads", "process"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fallback tokenizer hashes words; spawned workers need a shared seed
    # for payload comparisons across the baseline and resumed pipelines.
    monkeypatch.setenv("PYTHONHASHSEED", "0")

    def build() -> Pipeline:
        dataset = Dataset.from_dict("d", {0: InMemoryShard([{}, {}])})
        work = StaticMixtureWorkSource([dataset], {"d": 1.0}, chunk_size=1, seed=3)
        return (
            Pipeline(work)
            .add_op("split_text", process_many=_split_text, preserves_cursor_order=True)
            .tokenize(
                tokenizer_id="__fallback__",
                field="text",
                max_length=4,
                split_long_samples=True,
                special_tokens="none",
            )
            .options(
                runner=runner, deterministic=True, canonical_replicas=1, max_workers=2
            )
        )

    records = list(build())
    assert [record.meta.lineage for record in records] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
    ] * 2
    assert [
        (record.meta.chunk_id, record.meta.lineage)
        for record in records
        if any(ref.is_last_child for ref in record.meta.contributors)
    ] == [(0, (1, 2)), (1, (1, 2))]

    baseline = [_identity(record) for record in records]
    # Cut within a non-final parent, within a final parent, and after advancing
    # to the next source chunk so the earlier chunk has been evicted.
    for count in (1, 3, 6):
        pipe = build()
        iterator = iter(pipe)
        try:
            prefix = [_identity(next(iterator)) for _ in range(count)]
            checkpoint = pipe.checkpoint()
        finally:
            iterator.close()
        resumed = build()
        resumed.restore(checkpoint)
        assert prefix + [_identity(record) for record in resumed] == baseline, count
