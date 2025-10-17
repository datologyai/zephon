# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from zephon.core.constants import SampleMeta, SampleRecord
from zephon.ops.batch import Batch


def _rec(i: int, *, lane: int = 0, chunk: int = 0, **payload: Any) -> SampleRecord:
    meta = SampleMeta(sample_id=(0, 0, i), lane_id=lane, chunk_id=chunk)
    return SampleRecord(meta=meta, payload={"value": i, **payload})


def test_batch_invalid_microbatch_size_raises() -> None:
    with pytest.raises(ValueError):
        Batch(0)
    with pytest.raises(ValueError):
        Batch(-2)


def test_batch_single_lane_drop_last_true() -> None:
    op = Batch(2, drop_last=True)
    out = []
    out += op.process_one(_rec(0))
    # first not enough
    assert out == []
    out += op.process_one(_rec(1))
    # now a full batch of 2
    assert len(out) == 1 and len(out[0]) == 2
    # add one extra (remainder)
    out += op.process_one(_rec(2))
    assert len(out) == 1  # no immediate flush
    # finalize drops remainder when drop_last=True
    tail = op.finalize()
    assert tail == []


def test_batch_single_lane_drop_last_false() -> None:
    op = Batch(3, drop_last=False)
    # feed 5 -> one batch of 3, remainder 2 emitted on finalize
    out = op.process_many([_rec(i) for i in range(5)])
    assert len(out) == 1 and len(out[0]) == 3
    tail = op.finalize()
    assert len(tail) == 1 and len(tail[0]) == 2


def test_batch_multi_lane_interleaving_lane_purity() -> None:
    op = Batch(2, drop_last=False)
    # Interleave lanes 0 and 1
    items = [_rec(0, lane=0), _rec(1, lane=1), _rec(2, lane=0), _rec(3, lane=1)]
    out = op.process_many(items)
    # Expect two batches, each lane-pure
    assert len(out) == 2
    assert set(out[0].lane_ids) == {0}
    assert set(out[1].lane_ids) == {1}
    # No remainder
    assert op.finalize() == []


def test_batch_process_many_equivalence_to_one_by_one() -> None:
    op_a = Batch(2, drop_last=False)
    op_b = Batch(2, drop_last=False)
    items = [_rec(i) for i in range(4)]
    seq_out = []
    for it in items:
        seq_out += op_a.process_one(it)
    seq_out += op_a.finalize()
    bulk_out = op_b.process_many(items) + op_b.finalize()
    assert [b.ids for b in seq_out] == [b.ids for b in bulk_out]


def test_batch_traits_and_buffering_none() -> None:
    op = Batch(2)
    traits = op.traits()
    assert traits.indexable is False
    assert traits.batch_shape_sensitive is False
    assert op.buffering() is None
