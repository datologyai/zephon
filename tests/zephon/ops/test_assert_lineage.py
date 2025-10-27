import pytest

from zephon.core.constants import SampleBatch, SampleMeta, SampleRecord
from zephon.core.op_base import OpContext
from zephon.ops.assert_lineage import AssertLineageOrder


def _rec(lineage, lane=0, sample=0):
    meta = SampleMeta(sample_id=(0, 0, sample), lane_id=lane, chunk_id=0).with_lineage(
        lineage
    )
    return SampleRecord(meta=meta, payload={})


def test_assert_lineage_passes_for_monotonic_records() -> None:
    op = AssertLineageOrder()
    op.setup(OpContext({}))
    records = [
        _rec(()),
        _rec((0,)),
        _rec((0, 0)),
        _rec((1,)),
    ]
    assert op.process_many(records) == records


def test_assert_lineage_passes_for_batches() -> None:
    op = AssertLineageOrder()
    op.setup(OpContext({}))
    batch = SampleBatch(
        records=tuple([_rec((0,), lane=1, sample=idx) for idx in range(3)])
    )
    out = op.process_one(batch)
    assert out == [batch]


def test_assert_lineage_raises_on_non_monotonic_records() -> None:
    op = AssertLineageOrder()
    op.setup(OpContext({}))
    records = [_rec((1,)), _rec((0,))]
    with pytest.raises(AssertionError):
        op.process_many(records)


def test_assert_lineage_raises_on_mixed_lane_batches() -> None:
    op = AssertLineageOrder()
    op.setup(OpContext({}))
    bad_batch = SampleBatch(records=(_rec((0,), lane=0), _rec((1,), lane=1)))
    with pytest.raises(AssertionError):
        op.process_one(bad_batch)
