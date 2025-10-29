import pytest

from zephon.core.constants import SampleBatch, SampleMeta, SampleRecord
from zephon.core.op_base import OpContext
from zephon.ops.assert_lineage import AssertLineageOrder


def _noop(*args, **kwargs) -> None:  # pragma: no cover - trivial helper
    return None


def _setup(op: AssertLineageOrder) -> AssertLineageOrder:
    op.setup(
        OpContext({"record_node_metrics": _noop}),
        stage_index=0,
        stage_name="stage0",
        op_index=0,
        collect_stats=False,
    )
    return op


def _rec(lineage, lane=0, sample=0):
    meta = SampleMeta(sample_id=(0, 0, sample), lane_id=lane, chunk_id=0).with_lineage(
        lineage
    )
    return SampleRecord(meta=meta, payload={})


def test_assert_lineage_passes_for_monotonic_records() -> None:
    op = _setup(AssertLineageOrder())
    records = [
        _rec(()),
        _rec((0,)),
        _rec((0, 0)),
        _rec((1,)),
    ]
    assert op.process_many(records) == records


def test_assert_lineage_passes_for_batches() -> None:
    op = _setup(AssertLineageOrder())
    batch = SampleBatch(
        records=tuple([_rec((0,), lane=1, sample=idx) for idx in range(3)])
    )
    out = op.process_one(batch)
    assert out == [batch]


def test_assert_lineage_raises_on_non_monotonic_records() -> None:
    op = _setup(AssertLineageOrder())
    records = [_rec((1,)), _rec((0,))]
    with pytest.raises(AssertionError):
        op.process_many(records)


def test_assert_lineage_raises_on_mixed_lane_batches() -> None:
    op = _setup(AssertLineageOrder())
    bad_batch = SampleBatch(records=(_rec((0,), lane=0), _rec((1,), lane=1)))
    with pytest.raises(AssertionError):
        op.process_one(bad_batch)
