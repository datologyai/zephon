from __future__ import annotations

from typing import Any

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon.api import Pipeline as PublicPipeline
from zephon.core.constants import SampleBatch, SampleRecord


def _payload_dict(record: SampleRecord) -> dict[str, Any]:
    payload = record.payload
    assert isinstance(payload, dict)
    return payload


def _consume_all_batches(pipe: PublicPipeline) -> list[SampleBatch]:
    out: list[SampleBatch] = []
    it = iter(pipe)
    try:
        for item in it:
            assert isinstance(item, SampleBatch)
            out.append(item)
    finally:
        it.close()
    return out


def test_pipeline_iter_yields_records_without_batch() -> None:
    ds = make_inmem_dataset(
        "tiny",
        [
            {"text": b"Hello\r\nWorld"},
            {"text": b"Bye\rNow"},
        ],
    )
    ws = FakeIndexableWorkSource(ds, chunk_size=2)
    pipe = (
        PublicPipeline(ws)
        .decode_text(lowercase=True)
        .options(
            deterministic=True,
            max_workers=1,
            default_stage_prefetch=0,
            prefetch_batches=0,
        )
    )

    it = iter(pipe)
    try:
        first = next(it)
    finally:
        # close engine via iterator finalizer
        it.close()

    assert isinstance(first, SampleRecord)
    assert _payload_dict(first).get("text") == "hello\nworld"


def test_pipeline_iter_with_batch_and_drop_last_behavior() -> None:
    rows = [{"text": f"s{i}"} for i in range(5)]
    ds = make_inmem_dataset("tiny", rows)
    ws = FakeIndexableWorkSource(ds, chunk_size=3)

    # drop_last=False should emit 3 batches for 5 records at microbatch_size=2: (2,2,1)
    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .batch(microbatch_size=2, drop_last=False)
        .options(deterministic=True, max_workers=1, default_stage_prefetch=0)
    )
    got = []
    it = iter(pipe)
    try:
        for item in it:
            got.append(item)
    finally:
        it.close()

    assert all(isinstance(x, SampleBatch) for x in got)
    sizes = [len(b) for b in got]
    assert sizes[:2] == [2, 2]
    assert sizes[-1] == 1


def test_pipeline_iter_with_batch_drop_last_true_discards_tail() -> None:
    rows = [{"text": f"s{i}"} for i in range(5)]
    ds = make_inmem_dataset("tiny", rows)
    ws = FakeIndexableWorkSource(ds, chunk_size=4)

    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .batch(microbatch_size=2, drop_last=True)
        .options(deterministic=True, max_workers=1, default_stage_prefetch=0)
    )

    batches = _consume_all_batches(pipe)
    lengths = [len(b) for b in batches]
    assert lengths == [2, 2]
