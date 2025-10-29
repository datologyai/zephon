from zephon.core.constants import SampleCursor, SampleMeta, SampleRecord
from zephon.core.op_base import OpContext
from zephon.core.replay import ReplayConfigService
from zephon.ops.replay_filter import ReplayFilter


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
    op.setup(ctx, stage_index=0, stage_name="stage0", op_index=0, collect_stats=False)
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

    # Duplicate cursor should be dropped and must not advance the service.
    drop = _record(4)
    assert op.process_one(drop) == []
    assert service.snapshot()[0] == cursor

    # Next record advances past the checkpoint and reenables pass-through.
    accept = _record(5)
    assert op.process_one(accept) == [accept]
    assert service.snapshot()[0] == cursor

    # Subsequent records flow through while still updating progress.
    tail = _record(6)
    assert op.process_one(tail) == [tail]
    assert service.snapshot()[0] == cursor
