from zephon.core.children import spawn_child
from zephon.core.constants import SampleMeta, SampleRecord
from zephon.core.op_base import OpContext
from zephon.ops.shuffle_buffer import ShuffleBuffer


def _records(n: int) -> list[SampleRecord]:
    return [
        SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, i), lane_id=0, chunk_id=0, chunk_offset=i),
            payload={"text": str(i)},
        )
        for i in range(n)
    ]


def test_shuffle_buffer_is_deterministic() -> None:
    recs1 = _records(10)
    recs2 = _records(10)
    original_order = [r.meta.cursor for r in recs1]
    op1 = ShuffleBuffer(buffer_size=3, seed=123)
    op2 = ShuffleBuffer(buffer_size=3, seed=123)
    ctx = OpContext({"record_node_metrics": lambda *args, **kwargs: None})
    op1.setup(ctx, 0, "s", 0, False)
    op2.setup(ctx, 0, "s", 0, False)

    out1 = op1.process_many(recs1)
    out2 = op2.process_many(recs2)

    assert [r.meta.cursor for r in out1] == [r.meta.cursor for r in out2]
    assert [r.meta.cursor for r in out1] != original_order


def test_shuffle_buffer_flushes_tail() -> None:
    recs = _records(2)
    op = ShuffleBuffer(buffer_size=4, seed=7)
    ctx = OpContext({"record_node_metrics": lambda *args, **kwargs: None})
    op.setup(ctx, 0, "s", 0, False)

    # buffer smaller than window → no immediate output from process_many
    shuffled = op.process_many(recs)
    assert sorted(r.meta.chunk_offset for r in shuffled) == [0, 1]


def test_shuffle_buffer_keeps_closer_after_non_closer_for_same_base() -> None:
    base = SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0, chunk_offset=5)
    first = spawn_child(base, 0, is_last_child=False)
    last = spawn_child(base, 1, is_last_child=True)
    recs = [SampleRecord(meta=first, payload={}), SampleRecord(meta=last, payload={})]

    op = ShuffleBuffer(buffer_size=2, seed=99)
    ctx = OpContext({"record_node_metrics": lambda *args, **kwargs: None})
    op.setup(ctx, 0, "s", 0, False)

    out = op.process_many(recs)
    # The closing contributor must be on the last occurrence for the base offset.
    last_idx = max(
        idx
        for idx, rec in enumerate(out)
        if rec.meta.chunk_id == base.chunk_id
        and rec.meta.chunk_offset == base.chunk_offset
    )
    for idx, rec in enumerate(out):
        closes = any(ref.is_last_child for ref in rec.meta.contribution_refs())
        if idx == last_idx:
            assert closes
        else:
            assert not closes
