# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for zephon._internal.notify — notification extraction and dispatch."""

from unittest.mock import MagicMock

import pytest

from zephon._internal.notify import (
    _apply_notify_args,
    _ContributorNotify,
    _extract_notify_args,
    _MonotoneNotify,
    _notify_item,
    is_tombstone,
)
from zephon.types import (
    ContributorRef,
    SampleBatch,
    SampleCursor,
    SampleMeta,
    SampleRecord,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(
    lane: int = 0,
    chunk: int = 0,
    offset: int = 0,
    *,
    tombstone: bool = False,
) -> SampleMeta:
    sid = (0, 0, offset)
    m = SampleMeta(sample_id=sid, lane_id=lane, chunk_id=chunk, chunk_offset=offset)
    if tombstone:
        m = m.with_tombstone(True)
    return m


def _record(lane: int = 0, chunk: int = 0, offset: int = 0, **kw) -> SampleRecord:
    return SampleRecord(meta=_meta(lane, chunk, offset, **kw), payload={"x": offset})


def _batch(*records: SampleRecord) -> SampleBatch:
    return SampleBatch(records=tuple(records))


# ---------------------------------------------------------------------------
# _extract_notify_args — SampleRecord
# ---------------------------------------------------------------------------


class TestExtractRecord:
    def test_monotone_single_record(self):
        r = _record(lane=2, chunk=5, offset=3)
        result = _extract_notify_args(r, use_monotone=True)
        assert isinstance(result, _MonotoneNotify)
        assert result.lane_id == 2
        assert result.max_chunk_id == 5
        assert result.add_k == 1
        assert result.max_cursor == r.meta.cursor

    def test_contributor_single_record(self):
        r = _record(lane=1, chunk=3, offset=7)
        result = _extract_notify_args(r, use_monotone=False)
        assert isinstance(result, _ContributorNotify)
        assert result.lane_id == 1
        assert result.record_cursor == r.meta.cursor
        assert len(result.entries) == 1
        assert result.entries[0].cursor == r.meta.cursor
        assert result.entries[0].is_last_child is True


# ---------------------------------------------------------------------------
# _extract_notify_args — SampleBatch
# ---------------------------------------------------------------------------


class TestExtractBatch:
    def test_monotone_same_chunk(self):
        r0 = _record(lane=0, chunk=2, offset=0)
        r1 = _record(lane=0, chunk=2, offset=1)
        result = _extract_notify_args(_batch(r0, r1), use_monotone=True)
        assert isinstance(result, _MonotoneNotify)
        assert result.lane_id == 0
        assert result.max_chunk_id == 2
        assert result.add_k == 2
        assert result.max_cursor == r1.meta.cursor

    def test_monotone_different_chunks(self):
        r0 = _record(lane=0, chunk=1, offset=0)
        r1 = _record(lane=0, chunk=3, offset=0)
        result = _extract_notify_args(_batch(r0, r1), use_monotone=True)
        assert isinstance(result, _MonotoneNotify)
        assert result.max_chunk_id == 3
        assert result.add_k == 1
        assert result.max_cursor == r1.meta.cursor

    def test_contributor_batch(self):
        r0 = _record(lane=0, chunk=0, offset=0)
        r1 = _record(lane=0, chunk=0, offset=1)
        result = _extract_notify_args(_batch(r0, r1), use_monotone=False)
        assert isinstance(result, _ContributorNotify)
        assert result.lane_id == 0
        # record_cursor is the last record's cursor
        assert result.record_cursor == r1.meta.cursor
        # Both records contribute one ref each
        assert len(result.entries) == 2

    def test_empty_batch_raises(self):
        with pytest.raises(TypeError, match="at least one record"):
            _extract_notify_args(SampleBatch(records=()), use_monotone=True)


# ---------------------------------------------------------------------------
# _extract_notify_args — unsupported type
# ---------------------------------------------------------------------------


def test_unsupported_type_raises():
    with pytest.raises(TypeError, match="Unsupported element type"):
        _extract_notify_args(42, use_monotone=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _apply_notify_args
# ---------------------------------------------------------------------------


class TestApply:
    def test_dispatches_monotone(self):
        engine = MagicMock()
        cursor = SampleCursor(chunk_id=1, chunk_offset=0, sample_id=(0, 0, 0))
        args = _MonotoneNotify(lane_id=0, max_chunk_id=1, add_k=3, max_cursor=cursor)
        _apply_notify_args(engine, args)
        engine.notify_monotone.assert_called_once_with(0, 1, 3, cursor)

    def test_dispatches_contributor(self):
        engine = MagicMock()
        cursor = SampleCursor(chunk_id=0, chunk_offset=5, sample_id=(0, 0, 5))
        refs = [ContributorRef(cursor, True)]
        args = _ContributorNotify(lane_id=2, entries=refs, record_cursor=cursor)
        _apply_notify_args(engine, args)
        engine.notify.assert_called_once_with(2, refs, record_cursor=cursor)


# ---------------------------------------------------------------------------
# _notify_item (smoke test)
# ---------------------------------------------------------------------------


def test_notify_item_smoke():
    engine = MagicMock()
    r = _record(lane=0, chunk=0, offset=0)
    _notify_item(engine, r, use_monotone=True)
    engine.notify_monotone.assert_called_once()


# ---------------------------------------------------------------------------
# is_tombstone
# ---------------------------------------------------------------------------


class TestIsTombstone:
    def test_normal_record(self):
        assert is_tombstone(_record()) is False

    def test_tombstone_record(self):
        assert is_tombstone(_record(tombstone=True)) is True

    def test_batch_is_not_tombstone(self):
        assert is_tombstone(_batch(_record())) is False


# ---------------------------------------------------------------------------
# delivered flag — per-lane delivery counting at the tail
# ---------------------------------------------------------------------------


class TestDeliveredFlag:
    @pytest.mark.parametrize("use_monotone", [True, False])
    def test_real_record_is_delivered(self, use_monotone: bool):
        result = _extract_notify_args(_record(lane=1), use_monotone=use_monotone)
        assert result.delivered is True

    @pytest.mark.parametrize("use_monotone", [True, False])
    def test_tombstone_record_is_not_delivered(self, use_monotone: bool):
        result = _extract_notify_args(
            _record(lane=1, tombstone=True), use_monotone=use_monotone
        )
        assert result.delivered is False

    @pytest.mark.parametrize("use_monotone", [True, False])
    def test_batch_is_delivered(self, use_monotone: bool):
        result = _extract_notify_args(
            _batch(_record(lane=0, offset=0), _record(lane=0, offset=1)),
            use_monotone=use_monotone,
        )
        assert result.delivered is True

    def test_apply_records_delivery_for_delivered_items(self):
        engine = MagicMock()
        r = _record(lane=3)
        _notify_item(engine, r, use_monotone=True)
        engine.record_delivery.assert_called_once_with(3)

    def test_apply_skips_delivery_for_tombstones(self):
        engine = MagicMock()
        r = _record(lane=3, tombstone=True)
        _notify_item(engine, r, use_monotone=True)
        engine.record_delivery.assert_not_called()
        # The tombstone is still notified for eviction bookkeeping.
        engine.notify_monotone.assert_called_once()

    def test_apply_records_delivery_for_batches(self):
        engine = MagicMock()
        b = _batch(_record(lane=2, offset=0), _record(lane=2, offset=1))
        _notify_item(engine, b, use_monotone=False)
        engine.record_delivery.assert_called_once_with(2)
        engine.notify.assert_called_once()
