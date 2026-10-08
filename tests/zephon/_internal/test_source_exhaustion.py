# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Source-exhaustion markers are woven immediately after the final source sample."""

from zephon._internal.engine import Engine
from zephon._internal.notify import should_notify
from zephon._internal.stream import EngineSample
from zephon.types import SampleRecord
from zephon.work.base import WorkChunk


class _StubEngine:
    """Just enough of Engine for ``_iter_exhausted_chunk``."""

    _make_source_exhausted_sentinel = staticmethod(
        Engine._make_source_exhausted_sentinel
    )

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def _get_component_id(self, name: str) -> int:
        return self._ids.setdefault(name, len(self._ids))


def _weave(
    chunk: WorkChunk, lane_id: int = 0, cid: int = 7
) -> tuple[_StubEngine, list[EngineSample | SampleRecord]]:
    stub = _StubEngine()
    items = list(Engine._iter_exhausted_chunk(stub, lane_id, cid, chunk))
    return stub, items


def _sentinels(items: list[EngineSample | SampleRecord]) -> list[SampleRecord]:
    return [i for i in items if isinstance(i, SampleRecord)]


def test_no_stamp_yields_plain_samples() -> None:
    chunk = WorkChunk(components={"a": [(0, 0, i) for i in range(3)]}, seed=1)
    _, items = _weave(chunk)
    assert len(items) == 3
    assert not _sentinels(items)


def test_absent_announcements_have_stable_order() -> None:
    chunk = WorkChunk(components={"data": [(0, 0, 0)]}, source_exhausted=("z", "a"))
    stub, items = _weave(chunk)
    assert list(stub._ids) == ["a", "z", "data"]
    assert len(_sentinels(items)) == 2
    assert not isinstance(items[-1], SampleRecord)


def test_sentinel_immediately_after_last_sample_of_dying_component() -> None:
    chunk = WorkChunk(
        components={
            "a": [(0, 0, i) for i in range(4)],
            "b": [(1, 0, i) for i in range(2)],
        },
        seed=1,
        source_exhausted=("b",),
    )
    stub, items = _weave(chunk, lane_id=3)
    sentinels = _sentinels(items)
    assert len(sentinels) == 1
    sentinel = sentinels[0]
    assert sentinel.meta.is_source_exhausted
    assert sentinel.meta.lane_id == 3
    assert sentinel.meta.source_exhausted_component_id == stub._ids["b"]
    assert not should_notify(sentinel)

    b_id = stub._ids["b"]
    positions = [
        idx
        for idx, item in enumerate(items)
        if not isinstance(item, SampleRecord) and item[4] == b_id
    ]
    sentinel_pos = items.index(sentinel)
    # Every b sample precedes the sentinel, and it directly follows the last.
    assert positions and sentinel_pos == positions[-1] + 1
    assert sentinel_pos < len(items) - 1  # Other data remains in this chunk.
    restored = WorkChunk.from_state(chunk.state_dict())
    assert _weave(restored, lane_id=3)[1] == items
    # Sample count and order are untouched by the weaving.
    assert len([i for i in items if not isinstance(i, SampleRecord)]) == 6


def test_sentinel_before_chunk_when_component_absent() -> None:
    """Lanes that don't own the death chunk announce before their next chunk."""
    chunk = WorkChunk(
        components={"a": [(0, 0, i) for i in range(3)]},
        seed=1,
        source_exhausted=("b",),
    )
    _, items = _weave(chunk)
    assert isinstance(items[0], SampleRecord) and items[0].meta.is_source_exhausted
    assert not any(isinstance(i, SampleRecord) for i in items[1:])


def test_multiple_deaths_in_one_chunk() -> None:
    chunk = WorkChunk(
        components={
            "a": [(0, 0, i) for i in range(4)],
            "b": [(1, 0, 0)],
        },
        seed=1,
        source_exhausted=("b", "c"),
    )
    stub, items = _weave(chunk)
    sentinels = _sentinels(items)
    assert len(sentinels) == 2
    # Absent component "c" announces up front; "b" after its last sample.
    assert items[0] is sentinels[0]
    assert sentinels[0].meta.source_exhausted_component_id == stub._ids["c"]
    b_positions = [
        idx
        for idx, item in enumerate(items)
        if not isinstance(item, SampleRecord) and item[4] == stub._ids["b"]
    ]
    assert items.index(sentinels[1]) == b_positions[-1] + 1


def test_make_source_exhausted_sentinel_is_excluded_control_signal() -> None:
    """The EOD sentinel carries a dummy cursor, is tagged so the runner can
    dispatch it, and is excluded from engine notification (so it can never
    overwrite a lane's replay cursor)."""
    from zephon._internal.notify import should_notify

    sentinel = Engine._make_source_exhausted_sentinel(lane_id=4, component_id=2)
    assert isinstance(sentinel, SampleRecord)
    assert sentinel.meta.is_source_exhausted
    assert sentinel.meta.is_sentinel
    assert sentinel.meta.lane_id == 4
    assert sentinel.meta.source_exhausted_component_id == 2
    # Dummy cursor: chunk 0 / offset 0, never a real position.
    assert sentinel.meta.chunk_id == 0
    assert sentinel.meta.chunk_offset == 0
    assert not should_notify(sentinel)
