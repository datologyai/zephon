from __future__ import annotations

import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon._internal.graph import Plan
from zephon.pipeline import Pipeline
from zephon.types import (
    ContributorRef,
    SampleBatch,
    SampleCursor,
    SampleMeta,
    SampleRecord,
)


class _StubEngine:
    def __init__(self) -> None:
        self.calls: list[
            tuple[int, list[ContributorRef], SampleCursor | None, bool]
        ] = []
        self.monotone_calls: list[tuple[int, int, int, SampleCursor | None]] = []
        self.delivery_calls: list[int] = []
        self.inflight_chunks_per_lane: dict[int, dict[int, object]] = {}

    def notify(
        self,
        lane_id: int,
        entries: list[ContributorRef],
        record_cursor: SampleCursor | None = None,
        *,
        advance_replay: bool = True,
    ) -> bool:
        self.calls.append((lane_id, list(entries), record_cursor, advance_replay))
        return True

    def notify_monotone(
        self,
        lane_id: int,
        max_chunk_id: int,
        add_k: int,
        max_cursor: SampleCursor | None,
    ) -> bool:
        self.monotone_calls.append((lane_id, max_chunk_id, add_k, max_cursor))
        return True

    def record_delivery(self, lane_id: int) -> None:
        self.delivery_calls.append(lane_id)


def test_yield_while_notifying_type_checks_and_forwards() -> None:
    pipe = object.__new__(Pipeline)  # bypass __init__
    stub = _StubEngine()
    object.__setattr__(pipe, "_engine", stub)
    plan = Plan(stages=[], explain="", indexable=True, preserves_cursor_order=False)
    object.__setattr__(pipe, "_plan", plan)

    # Craft a SampleRecord and a SampleBatch for the same lane
    rec = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 1), lane_id=2, chunk_id=5),
        payload={"text": "x"},
    )
    batch = SampleBatch(
        records=(
            SampleRecord(
                meta=SampleMeta(sample_id=(0, 0, 2), lane_id=2, chunk_id=6), payload={}
            ),
            SampleRecord(
                meta=SampleMeta(sample_id=(0, 0, 3), lane_id=2, chunk_id=6), payload={}
            ),
        )
    )

    out = list(Pipeline._yield_while_notifying(pipe, [rec, batch]))
    # Pass-through of both elements
    assert out[0] is rec
    assert out[1] is batch

    # Engine.notify was invoked with lane and chunk derived from elements
    assert stub.calls[0][0] == 2
    assert stub.calls[1][0] == 2
    assert stub.calls[0][2] == rec.meta.cursor
    assert stub.calls[1][2] == batch.records[-1].meta.cursor
    assert [e.cursor for e in stub.calls[0][1]] == [rec.meta.cursor]
    assert [e.cursor for e in stub.calls[1][1]] == [
        r.meta.cursor for r in batch.records
    ]


def test_yield_while_notifying_unsupported_type_raises() -> None:
    pipe = object.__new__(Pipeline)
    object.__setattr__(pipe, "_engine", _StubEngine())
    plan = Plan(stages=[], explain="", indexable=True, preserves_cursor_order=False)
    object.__setattr__(pipe, "_plan", plan)
    with pytest.raises(TypeError):
        _ = list(Pipeline._yield_while_notifying(pipe, [123]))


def test_pipeline_batch_requires_positive_microbatch() -> None:
    ds = make_inmem_dataset("tiny", [{"text": "x"}])
    ws = FakeIndexableWorkSource(ds, chunk_size=1)
    pipe = Pipeline(ws)
    with pytest.raises(ValueError):
        pipe.batch(0)


@pytest.mark.parametrize("monotone", [False, True])
def test_yield_while_notifying_drops_tombstones(monotone: bool) -> None:
    pipe = object.__new__(Pipeline)
    stub = _StubEngine()
    object.__setattr__(pipe, "_engine", stub)
    plan = Plan(stages=[], explain="", indexable=True, preserves_cursor_order=monotone)
    object.__setattr__(pipe, "_plan", plan)

    tomb_meta = SampleMeta(
        sample_id=(0, 0, 1),
        lane_id=0,
        chunk_id=0,
    ).with_tombstone(True)
    tomb = SampleRecord(meta=tomb_meta, payload={})
    kept = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 2), lane_id=0, chunk_id=0), payload={}
    )

    out = list(Pipeline._yield_while_notifying(pipe, [tomb, kept]))
    assert out == [kept]
    assert len(stub.calls) == (1 if monotone else 2)
    assert stub.calls[0] == (0, list(tomb.meta.contribution_refs()), None, False)
    assert stub.delivery_calls == [0]
    if monotone:
        assert stub.monotone_calls == [(0, 0, 1, kept.meta.cursor)]
    else:
        assert stub.calls[1][2:] == (kept.meta.cursor, True)
