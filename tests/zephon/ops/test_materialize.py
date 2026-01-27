# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from zephon.core.constants import SampleBatch, SampleMeta, SampleRecord
from zephon.ops.materialize import Materialize


def _rec(local_id: int) -> SampleRecord:
    meta = SampleMeta(sample_id=(0, 0, local_id), lane_id=0, chunk_id=0)
    return SampleRecord(meta=meta, payload={"v": local_id})


def test_materialize_identity_for_records_and_batches() -> None:
    op = Materialize()
    r = _rec(1)
    out = op.process_one(r)
    assert out == [r]

    b = SampleBatch(records=(r, _rec(2)))
    out2 = op.process_one(b)
    assert out2 == [b]


def test_materialize_process_many() -> None:
    op = Materialize()
    records = [_rec(i) for i in range(3)]
    out = op.process_many(records)
    assert out == records


def test_materialize_traits_and_accumulator() -> None:
    from zephon.core.accumulators import PassthroughAccumulator

    op = Materialize()
    t = op.traits()
    assert t.indexable is False

    # PassthroughAccumulator has no configuration - keep isinstance as regression guard
    acc = op.accumulator(deterministic=False, ctx={})
    assert isinstance(acc, PassthroughAccumulator)
