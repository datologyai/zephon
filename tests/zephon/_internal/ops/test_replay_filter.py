from zephon._internal.ops.replay_filter import ReplayFilter
from zephon._internal.replay import ReplayConfigService
from zephon.ops.base import OpContext
from zephon.types import SampleCursor, SampleMeta, SampleRecord


def _only_tombstones(records: list[SampleRecord]) -> bool:
    """Check that every record in the list is a tombstone."""
    return len(records) > 0 and all(r.meta.tombstone for r in records)


def _noop(*args, **kwargs) -> None:  # pragma: no cover - trivial helper
    return None


def _record(
    sample_idx: int, lane: int = 0, lineage: tuple[int, ...] = ()
) -> SampleRecord:
    meta = SampleMeta(
        sample_id=(0, 0, sample_idx), lane_id=lane, chunk_id=0, chunk_offset=sample_idx
    ).with_lineage(lineage)
    return SampleRecord(meta=meta, payload={})


def _filter(service: ReplayConfigService) -> ReplayFilter:
    op = ReplayFilter()
    ctx = OpContext(
        {
            "replay_state_service": service,
            "record_node_metrics": _noop,
        }
    )
    op.setup(ctx)
    return op


def test_replay_filter_is_noop_when_no_snapshot() -> None:
    service = ReplayConfigService()
    op = _filter(service)
    records = [_record(0), _record(1)]
    out = op.process_many(records)
    assert out == records
    assert service.snapshot() == {}


def test_replay_filter_drops_until_past_checkpoint_cursor() -> None:
    cursor = SampleCursor(chunk_id=0, chunk_offset=4, sample_id=(0, 0, 4))
    service = ReplayConfigService()
    service.set_snapshot({0: cursor})
    op = _filter(service)

    assert _only_tombstones(op.process_one(_record(3)))
    # Target itself is also dropped (tombstones emitted), then pass-through resumes.
    assert _only_tombstones(op.process_one(_record(4)))
    accept = _record(5)
    assert op.process_one(accept) == [accept]
    assert service.snapshot()[0] == cursor


def test_replay_filter_allows_non_monotone_suffix() -> None:
    cursor = SampleCursor(chunk_id=0, chunk_offset=2, sample_id=(0, 0, 2))
    service = ReplayConfigService()
    service.set_snapshot({0: cursor})
    op = _filter(service)

    # Drop records until the exact target is seen (tombstones emitted).
    assert _only_tombstones(op.process_one(_record(3)))
    assert _only_tombstones(op.process_one(_record(2)))

    # After the flip, even "earlier" cursors must pass through.
    late = _record(1)
    assert op.process_one(late) == [late]
