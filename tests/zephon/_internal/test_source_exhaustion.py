# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Source-exhaustion notifications are emitted immediately after the final source sample."""

from zephon._internal.engine import Engine
from zephon._internal.notify import should_notify, source_exhausted_component
from zephon._internal.stream import EngineSample
from zephon.types import SampleRecord
from zephon.work.base import WorkChunk


class _StubEngine:
    """Just enough of Engine for ``_iter_chunk``."""

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
    items = list(Engine._iter_chunk(stub, lane_id, cid, chunk, set()))
    return stub, items


def _sentinels(items: list[EngineSample | SampleRecord]) -> list[SampleRecord]:
    return [i for i in items if isinstance(i, SampleRecord)]


def test_no_stamp_yields_plain_samples() -> None:
    chunk = WorkChunk(components={"a": [(0, 0, i) for i in range(3)]}, seed=1)
    _, items = _weave(chunk)
    assert len(items) == 3
    assert not _sentinels(items)


def test_notification_is_once_per_stream_and_restored_from_cumulative_stamp() -> None:
    stub = _StubEngine()
    announced: set[str] = set()
    first = WorkChunk(components={"a": [(0, 0, 0)]}, source_exhausted=("a",))
    later = WorkChunk(components={"b": [(1, 0, 0)]}, source_exhausted=("a",))
    initial = list(Engine._iter_chunk(stub, 0, 0, first, announced))
    assert len(_sentinels(initial)) == 1
    for cid in (1, 2, 3):
        assert not _sentinels(list(Engine._iter_chunk(stub, 0, cid, later, announced)))
    restored = WorkChunk.from_state(later.state_dict())
    replay = list(Engine._iter_chunk(stub, 0, 2, restored, set()))
    assert isinstance(replay[0], SampleRecord)
    assert source_exhausted_component(replay[0]) == stub._ids["a"]
    assert replay[1:] == list(Engine._iter_chunk(stub, 0, 2, later, announced))


def test_absent_announcements_have_stable_order() -> None:
    chunk = WorkChunk(components={"data": [(0, 0, 0)]}, source_exhausted=("z", "a"))
    stub, items = _weave(chunk)
    assert list(stub._ids) == ["a", "z", "data"]
    assert len(_sentinels(items)) == 2
    assert not isinstance(items[-1], SampleRecord)


def test_sentinel_immediately_after_last_sample_of_exhausted_component() -> None:
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
    assert source_exhausted_component(sentinel) is not None
    assert sentinel.meta.lane_id == 3
    assert source_exhausted_component(sentinel) == stub._ids["b"]
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
    """Lanes that don't own the exhaustion chunk announce before their next chunk."""
    chunk = WorkChunk(
        components={"a": [(0, 0, i) for i in range(3)]},
        seed=1,
        source_exhausted=("b",),
    )
    _, items = _weave(chunk)
    assert (
        isinstance(items[0], SampleRecord)
        and source_exhausted_component(items[0]) is not None
    )
    assert not any(isinstance(i, SampleRecord) for i in items[1:])


def test_multiple_exhaustions_in_one_chunk() -> None:
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
    assert source_exhausted_component(sentinels[0]) == stub._ids["c"]
    b_positions = [
        idx
        for idx, item in enumerate(items)
        if not isinstance(item, SampleRecord) and item[4] == stub._ids["b"]
    ]
    assert items.index(sentinels[1]) == b_positions[-1] + 1


def test_make_source_exhausted_sentinel_is_excluded_control_signal() -> None:
    """The source-exhaustion notification carries a dummy cursor, is tagged so the runner can
    dispatch it, and is excluded from engine notification (so it can never
    overwrite a lane's replay cursor)."""
    sentinel = Engine._make_source_exhausted_sentinel(lane_id=4, component_id=2)
    assert isinstance(sentinel, SampleRecord)
    assert source_exhausted_component(sentinel) is not None
    assert sentinel.meta.is_sentinel
    assert sentinel.meta.lane_id == 4
    assert source_exhausted_component(sentinel) == 2
    # Dummy cursor: chunk 0 / offset 0, never a real position.
    assert sentinel.meta.chunk_id == 0
    assert sentinel.meta.chunk_offset == 0
    assert not should_notify(sentinel)
