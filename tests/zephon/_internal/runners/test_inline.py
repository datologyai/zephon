import threading
from pathlib import Path
from typing import Iterable

import pytest

from tests.zephon._internal.runners._helpers import (
    _assert_close_waits_for_operator_execution,
    _ctx_services,
    _extract_values,
    _mk_record,
    _mk_records,
    _probe_stage,
    _resource_stage,
    _ResourceOp,
)
from zephon._internal.graph import Node, Stage
from zephon._internal.ops.batch import Batch
from zephon._internal.ops.delay import DelayById
from zephon._internal.runners.inline import InlineStageRunner
from zephon.observability.config import ExecutionTrackingMode
from zephon.types import SampleMeta, SampleRecord


def _collect(runner: InlineStageRunner, data: Iterable[int]) -> list[int]:
    records = _mk_records(data)
    out_records = list(runner.run(iter(records)))
    return _extract_values(out_records)


def test_inline_runner_preserves_input_order() -> None:
    op = DelayById(max_delay_ms=2.0)
    node = Node(name="delay", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")

    runner = InlineStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )
    data = list(range(50))
    assert _collect(runner, data) == data


def test_inline_prefetch_iterator_close_is_clean() -> None:
    op = DelayById(max_delay_ms=0.0)
    node = Node(name="delay", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")
    runner = InlineStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        prefetch_capacity=4,
        stage_output_mode="stream_items",
    )

    iterator = runner.run(iter(_mk_records(range(20))))
    got: list[SampleRecord] = []
    for _ in range(3):
        got.append(next(iterator))
    assert _extract_values(got) == [0, 1, 2]
    if hasattr(iterator, "close"):
        iterator.close()  # type: ignore[call-arg]


def test_inline_close_hard() -> None:
    op = DelayById(max_delay_ms=0.0)
    node = Node(name="delay", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")
    runner = InlineStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )
    list(runner.run(iter(_mk_records(range(5)))))
    runner.close(hard=True)


def test_inline_passthrough_stage_forwards_stream() -> None:
    stage = Stage(name="empty", nodes=[], placement="auto", break_reason="test")
    runner = InlineStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )
    data = _mk_records(range(6))
    out = list(runner.run(iter(data)))
    assert out == data


def test_inline_sentinel_bypass_accumulator_and_process_many() -> None:
    """Sentinels bypass accumulator and process_many at the runner level.

    The Batch operator with microbatch_size=3 buffers regular records in its
    accumulator and wraps them into SampleBatch via process_many.  Sentinels
    (tombstones) must not be buffered or wrapped — they should pass through
    unchanged.
    """
    op = Batch(3)
    node = Node(name="batch", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")

    runner = InlineStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )

    tomb_meta = SampleMeta(sample_id=(0, 0, 99), lane_id=0, chunk_id=0).with_tombstone(
        True
    )
    tomb = SampleRecord(meta=tomb_meta, payload={})

    # Feed: 3 regular records, then a tombstone, then 3 more regular records.
    inputs: list[SampleRecord] = [_mk_record(i) for i in range(3)]
    inputs.append(tomb)
    inputs.extend(_mk_record(i) for i in range(3, 6))

    out = list(runner.run(iter(inputs)))

    # The tombstone must appear as a bare SampleRecord (not inside a SampleBatch).
    tombstones_out = [
        item for item in out if isinstance(item, SampleRecord) and item.meta.tombstone
    ]
    assert len(tombstones_out) == 1
    assert tombstones_out[0] is tomb

    # The 6 regular records should be batched into SampleBatches.
    from zephon.types import SampleBatch

    batches_out = [item for item in out if isinstance(item, SampleBatch)]
    total_regular = sum(len(b.records) for b in batches_out)
    assert total_regular == 6


def test_inline_runner_provides_stage_info_per_op() -> None:
    runner = InlineStageRunner(
        _probe_stage("first", "second"),
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_index=3,
        stage_output_mode="stream_items",
    )
    try:
        (out,) = list(runner.run(iter(_mk_records([7]))))
    finally:
        runner.close()
    assert out.payload["first"] == (3, "probe_stage", 0, False)
    assert out.payload["second"] == (3, "probe_stage", 1, False)


def test_inline_runner_collect_stats_flag_reaches_ops() -> None:
    runner = InlineStageRunner(
        _probe_stage("probe"),
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        tracking_mode=ExecutionTrackingMode.NODES,
        stage_output_mode="stream_items",
    )
    try:
        (out,) = list(runner.run(iter(_mk_records([7]))))
    finally:
        runner.close()
    assert out.payload["probe"] == (0, "probe_stage", 0, True)


@pytest.mark.parametrize("hard", [False, True])
def test_close_releases_operator_resources(tmp_path: Path, hard: bool) -> None:
    runner = InlineStageRunner(
        _resource_stage(_ResourceOp(tmp_path)),
        _ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )
    instances = list(runner.ops[0].instances)
    try:
        assert list(runner.run(_mk_records([1, 2]))) == _mk_records([1, 2])
    finally:
        runner.close(hard=hard)
    runner.close(hard=hard)

    assert instances and all(op.resource is None for op in instances)
    assert not list(tmp_path.glob("*.live"))
    assert len(list(tmp_path.glob("*.closed"))) == 1

    with pytest.raises(RuntimeError, match="Stage runner is closed"):
        runner.run_one(_mk_record(1))
    with pytest.raises(RuntimeError, match="Stage runner is closed"):
        list(runner.run(_mk_records([1])))


@pytest.mark.parametrize("hard", [False, True])
def test_close_waits_for_operator_execution(hard: bool) -> None:
    _assert_close_waits_for_operator_execution(InlineStageRunner, hard=hard)


@pytest.mark.parametrize("hard", [False, True])
def test_close_from_active_operator_defers_teardown(tmp_path: Path, hard: bool) -> None:
    class ClosingOp(_ResourceOp):
        def process_many(self, elems):
            runner.close(hard=hard)
            assert self.resource is not None
            return super().process_many(elems)

    runner = InlineStageRunner(
        _resource_stage(ClosingOp(tmp_path)),
        _ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )
    errors: list[BaseException] = []

    def consume() -> None:
        try:
            assert list(runner.run(_mk_records([1]))) == _mk_records([1])
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), "close must not wait on its own active operator"
    assert not errors
    assert not list(tmp_path.glob("*.live"))
    assert len(list(tmp_path.glob("*.closed"))) == 1
