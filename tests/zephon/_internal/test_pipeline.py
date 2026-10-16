"""Unit tests for pipeline options and output notification."""

from collections.abc import Iterator
from unittest.mock import Mock

import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon._internal.graph import Plan
from zephon._internal.pipeline import Pipeline
from zephon.types import SampleMeta, SampleRecord


def test_removed_payload_memory_options_fail_before_updating_pipeline() -> None:
    pipe = Pipeline(FakeIndexableWorkSource(make_inmem_dataset("tiny", [{"x": 1}])))
    pipe.options(shm_min_item_bytes=456, shm_coalesce=True)
    for old, new, value in (
        ("shm_min_size", "shm_min_item_bytes", 123),
        ("coalesce_tensors", "shm_coalesce", False),
    ):
        with pytest.raises(TypeError, match=f"{old}.*Use '{new}'"):
            pipe.options(max_workers=99, **{old: value, new: value})
    assert pipe._options.max_workers is None
    assert pipe._options.shm_min_item_bytes == 456
    assert pipe._options.shm_coalesce


@pytest.mark.parametrize("use_monotone", [True, False])
def test_controls_do_not_notify_delivery_or_escape_the_tail(use_monotone: bool) -> None:
    # Exercise only the tail wrapper: no source, graph compilation or engine.
    pipe = object.__new__(Pipeline)
    engine = Mock()
    pipe._engine = engine
    pipe._plan = Plan(
        stages=[], explain="", indexable=False, preserves_cursor_order=use_monotone
    )
    controls = [
        SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0, tags=tags),
            payload=None,
        )
        for tags in ({"_source_exhausted": 0}, {"_flush_sentinel": True})
    ]
    dropped = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 3), lane_id=0, chunk_id=3).with_tombstone(),
        payload=None,
    )
    record = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"value": "still buffered when the controls passed"},
    )

    def source() -> Iterator[SampleRecord]:
        yield from controls
        assert engine.mock_calls == []  # Dummy cursors cannot complete offsets.
        yield dropped
        engine.record_delivery.assert_not_called()
        engine.notify.assert_not_called()
        engine.notify_monotone.assert_not_called()
        if use_monotone:
            engine.notify_completion.assert_not_called()
        else:
            engine.notify_completion.assert_called_once_with(
                0, dropped.meta.contribution_refs()
            )
        yield record

    iterator = pipe._yield_while_notifying(source())
    try:
        assert next(iterator) is record
        engine.record_delivery.assert_called_once_with(0)
        if use_monotone:
            engine.notify_monotone.assert_called_once_with(0, 0, 1, record.meta.cursor)
            engine.notify.assert_not_called()
        else:
            engine.notify.assert_called_once_with(
                0, record.meta.contribution_refs(), record_cursor=record.meta.cursor
            )
            engine.notify_monotone.assert_not_called()
    finally:
        # Leave end-of-stream cleanup to integration tests.
        iterator.close()
