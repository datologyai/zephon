import multiprocessing
from dataclasses import dataclass
from typing import Any

import pytest

from tests.zephon.runners._helpers import (
    _ctx_services,
    _extract_values,
    _mk_record,
    _mk_records,
)
from zephon.core.accumulators import (
    Accumulator,
    PassthroughAccumulator,
)
from zephon.core.constants import SampleRecord
from zephon.core.graph import Node, Stage
from zephon.core.op_base import DefaultSetup, Op, OpContext
from zephon.core.traits import OpTraits
from zephon.observability.config import ExecutionTrackingMode
from zephon.observability.stats import NodeMetricsDelta
from zephon.ops.delay import DelayById
from zephon.runners.process import ProcessStageRunner


def _collect(runner: ProcessStageRunner, data: list[int]) -> list[int]:
    records = _mk_records(data)
    out_records = list(runner.run(iter(records)))
    return _extract_values(out_records)


def test_process_runner_emits_in_input_order_when_deterministic() -> None:
    op = DelayById(max_delay_ms=1.5)
    node = Node(name="delay", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=4,
        deterministic=True,
        stage_output_mode="stream_items",
    )
    data = list(range(30))
    assert _collect(runner, data) == data


def test_process_run_one_returns_through_single_op_stage() -> None:
    op = DelayById(max_delay_ms=0.0)
    node = Node(name="delay", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=2,
        deterministic=True,
        stage_output_mode="stream_items",
    )
    record = _mk_record(7)
    out = runner.run_one(record)
    assert isinstance(out, SampleRecord)
    assert out == record


def test_process_prefetch_iterator_close_is_clean() -> None:
    op = DelayById(max_delay_ms=0.0)
    node = Node(name="delay", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")
    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=2,
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


def test_process_passthrough_stage_forwards_stream() -> None:
    stage = Stage(name="empty", nodes=[], placement="auto", break_reason="test")
    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=2,
        deterministic=True,
        stage_output_mode="stream_items",
    )
    data = _mk_records(range(6))
    out = list(runner.run(iter(data)))
    assert out == data


@dataclass
class _IdentityOp(DefaultSetup, Op[Any, Any]):
    name: str = "identity"

    def __post_init__(self) -> None:
        DefaultSetup.__init__(self)

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=True,
            preserves_cursor_order=True,
            parallelism=1,
            batch_shape_sensitive=False,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[Any]:
        return PassthroughAccumulator[Any]()

    def process_one(self, elem: Any) -> list[Any]:
        return [elem]

    def process_many(self, elems: list[Any]) -> list[Any]:
        return list(elems)


class _ValueMappingOp(DefaultSetup, Op[Any, Any]):
    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=True,
            preserves_cursor_order=True,
            parallelism=1,
            batch_shape_sensitive=False,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[Any]:
        return PassthroughAccumulator[Any]()

    def _rewrite(self, elem: SampleRecord) -> SampleRecord:
        payload = dict(elem.payload)
        payload["value"] = self._map(int(payload["value"]))
        return SampleRecord(meta=elem.meta, payload=payload)

    def _map(self, value: int) -> int:
        raise NotImplementedError

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        return [self._rewrite(elem)]

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        return [self._rewrite(elem) for elem in elems]


@dataclass
class _AddValueOp(_ValueMappingOp):
    delta: int = 0

    def __post_init__(self) -> None:
        DefaultSetup.__init__(self)

    def _map(self, value: int) -> int:
        return value + self.delta


@dataclass
class _MultiplyValueOp(_ValueMappingOp):
    factor: int = 1

    def __post_init__(self) -> None:
        DefaultSetup.__init__(self)

    def _map(self, value: int) -> int:
        return value * self.factor


@dataclass
class _ServiceOp(DefaultSetup, Op[Any, Any]):
    def __post_init__(self) -> None:
        DefaultSetup.__init__(self)
        self._hook: Any = None

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=True,
            preserves_cursor_order=True,
            parallelism=1,
            batch_shape_sensitive=False,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[Any]:
        return PassthroughAccumulator[Any]()

    def setup(
        self,
        ctx: OpContext,
        stage_index: int,
        stage_name: str,
        op_index: int,
        collect_stats: bool,
    ) -> None:
        super().setup(ctx, stage_index, stage_name, op_index, collect_stats)
        self._hook = ctx.get("custom_service")

    def process_one(self, elem: Any) -> list[Any]:
        if callable(self._hook):
            self._hook(elem.payload["value"])
        return [elem]


def test_process_runner_proxies_context_services() -> None:
    calls: list[tuple[str, int]] = []

    def _hook(value: int) -> None:
        calls.append((multiprocessing.current_process().name, value))

    op = _ServiceOp()
    node = Node(name="svc", op=op)
    stage = Stage(name="svc_stage", nodes=[node], placement="auto", break_reason="test")

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services({"custom_service": _hook}),
        max_workers=2,
        deterministic=True,
        stage_output_mode="stream_items",
    )

    data = list(range(5))
    out = _collect(runner, data)
    assert out == data
    assert calls
    assert all(name == "MainProcess" for name, _ in calls)
    assert sorted(val for _, val in calls) == data


def test_process_runner_emits_metrics_deltas_when_callback_provided() -> None:
    op = _IdentityOp()
    node = Node(name="identity", op=op)
    stage = Stage(name="stage0", nodes=[node], placement="auto", break_reason="test")

    captured: list[NodeMetricsDelta] = []

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services({"record_node_metrics": captured.append}),
        max_workers=2,
        deterministic=True,
        stage_index=3,
        tracking_mode=ExecutionTrackingMode.NODES,
        stage_output_mode="stream_items",
    )

    data = _mk_records(range(6))
    out = list(runner.run(iter(data)))
    assert out == data

    assert captured
    produced = sum(delta.produced_elements for delta in captured)
    consumed = sum(delta.consumed_elements for delta in captured)
    assert produced == len(data)
    assert consumed == len(data)
    assert all(delta.stage_index == 3 for delta in captured)
    assert all(delta.stage_name == "stage0" for delta in captured)


def test_process_runner_emits_microbatches_and_accepts_batch_input() -> None:
    op = _IdentityOp()
    node = Node(name="identity", op=op)
    stage = Stage(name="stage1", nodes=[node], placement="auto", break_reason="test")

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=2,
        deterministic=True,
        stage_output_mode="microbatches",
    )

    singles = _mk_records(range(2))
    batch = _mk_records(range(2, 5))
    upstream = iter([singles[0], singles[1], batch])
    out = list(runner.run(upstream))

    assert len(out) == 3
    assert all(isinstance(elem, list) for elem in out)
    assert _extract_values(out[0]) == [0]
    assert _extract_values(out[1]) == [1]
    assert _extract_values(out[2]) == [2, 3, 4]


def test_process_runner_stream_mode_flattens_microbatch_input() -> None:
    op = _IdentityOp()
    node = Node(name="identity", op=op)
    stage = Stage(name="stage2", nodes=[node], placement="auto", break_reason="test")

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=2,
        deterministic=True,
        stage_output_mode="stream_items",
    )

    microbatch = _mk_records(range(5))
    out = list(runner.run(iter([microbatch])))
    assert out == microbatch


def test_process_runner_direct_ipc_fast_path_transforms_stream() -> None:
    op = _AddValueOp(delta=5)
    node = Node(name="add", op=op)
    stage = Stage(name="direct", nodes=[node], placement="auto", break_reason="test")

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=2,
        deterministic=True,
        stage_output_mode="stream_items",
    )

    assert getattr(runner, "_single_op_direct_ipc") is True

    data = list(range(12))
    out = _collect(runner, data)
    assert out == [value + 5 for value in data]


def test_process_runner_non_fast_path_handles_multiple_ops() -> None:
    add = Node(name="add", op=_AddValueOp(delta=1))
    multiply = Node(name="mul", op=_MultiplyValueOp(factor=3), inputs=[add])
    stage = Stage(
        name="chain",
        nodes=[add, multiply],
        placement="auto",
        break_reason="test",
    )

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=3,
        deterministic=True,
        stage_output_mode="stream_items",
    )

    assert getattr(runner, "_single_op_direct_ipc") is False

    data = list(range(10))
    out = _collect(runner, data)
    assert out == [(value + 1) * 3 for value in data]


def test_process_runner_multi_op_draining_forwards_to_next_op() -> None:
    """Regression test for multi-op stage correctness under backpressure.

    There was a bug where _send_command's draining path passed next_queue=None,
    which would route results directly to stage output instead of the next operator.
    The bug is very unlikely to trigger (requires a tight race condition), but this
    test exercises multi-op stages with backpressure to catch it if it ever happens.
    """
    add1 = Node(name="add1", op=_AddValueOp(delta=1))
    add10 = Node(name="add10", op=_AddValueOp(delta=10), inputs=[add1])

    stage = Stage(
        name="chain2",
        nodes=[add1, add10],
        placement="auto",
        break_reason="test",
    )

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=4,
        deterministic=True,
        queue_capacity=1,
        stage_output_mode="stream_items",
    )

    assert getattr(runner, "_single_op_direct_ipc") is False

    data = list(range(100))
    out = _collect(runner, data)

    # Expected: v + 1 + 10 = v + 11
    expected = [v + 11 for v in data]
    assert out == expected


@dataclass
class _CrashOp(DefaultSetup, Op[Any, Any]):
    def __post_init__(self) -> None:
        DefaultSetup.__init__(self)

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=True,
            preserves_cursor_order=True,
            parallelism=1,
            batch_shape_sensitive=False,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[Any]:
        return PassthroughAccumulator[Any]()

    def process_many(self, elems: list[Any]) -> list[Any]:
        raise ValueError("boom inside worker")


def test_process_runner_bubbles_worker_exceptions() -> None:
    op = _CrashOp()
    node = Node(name="crash", op=op)
    stage = Stage(name="stage3", nodes=[node], placement="auto", break_reason="test")
    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )

    with pytest.raises(RuntimeError) as excinfo:
        list(runner.run(iter(_mk_records(range(3)))))
    text = str(excinfo.value)
    assert "ValueError" in text
    assert "boom inside worker" in text
    assert "process_many" in text


@dataclass
class _LambdaOp(DefaultSetup, Op[SampleRecord, SampleRecord]):
    """Operator that uses a lambda function internally.

    This mimics what PackSequences does with its length_fn field.
    """

    transform: Any = None  # Will be set to a lambda in __post_init__

    def __post_init__(self) -> None:
        DefaultSetup.__init__(self)
        # Use a lambda to transform the value, just like PackSequences uses lambda for length_fn
        self.transform = lambda x: x * 2

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=True,
            preserves_cursor_order=True,
            parallelism=1,
            batch_shape_sensitive=False,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return PassthroughAccumulator[SampleRecord]()

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        payload = dict(elem.payload)
        payload["value"] = self.transform(int(payload["value"]))
        return [SampleRecord(meta=elem.meta, payload=payload)]


def test_process_runner_serializes_operator_with_lambda() -> None:
    """Test that operators containing lambda functions serialize correctly.

    This verifies that cloudpickle is working - standard pickle would fail
    with 'Can't pickle <lambda>' error.
    """
    op = _LambdaOp()
    node = Node(name="lambda_op", op=op)
    stage = Stage(
        name="lambda_stage", nodes=[node], placement="auto", break_reason="test"
    )

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=2,
        deterministic=True,
        stage_output_mode="stream_items",
    )

    # If cloudpickle is working, this should succeed
    # If standard pickle is used, it would fail with PicklingError
    data = list(range(5))
    out = _collect(runner, data)
    assert out == [x * 2 for x in data]


@dataclass
class _ClosureOp(DefaultSetup, Op[SampleRecord, SampleRecord]):
    """Operator that uses a closure (lambda capturing outer variable)."""

    multiplier: int = 3

    def __post_init__(self) -> None:
        DefaultSetup.__init__(self)
        # Create a closure that captures self.multiplier
        self.transform = lambda x: x * self.multiplier

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=True,
            preserves_cursor_order=True,
            parallelism=1,
            batch_shape_sensitive=False,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return PassthroughAccumulator[SampleRecord]()

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        payload = dict(elem.payload)
        payload["value"] = self.transform(int(payload["value"]))
        return [SampleRecord(meta=elem.meta, payload=payload)]


def test_process_runner_serializes_operator_with_closure() -> None:
    """Test that operators with closures (lambdas capturing outer variables) work.

    Closures are even trickier than plain lambdas for standard pickle.
    """
    op = _ClosureOp(multiplier=5)
    node = Node(name="closure_op", op=op)
    stage = Stage(
        name="closure_stage", nodes=[node], placement="auto", break_reason="test"
    )

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=2,
        deterministic=True,
        stage_output_mode="stream_items",
    )

    data = list(range(4))
    out = _collect(runner, data)
    assert out == [x * 5 for x in data]


def test_process_runner_partial_iteration_shutdown_no_underflow() -> None:
    """Partial iteration followed by shutdown should not cause underflow.

    This creates conditions where shutdown races with result handling:
    1. Multiple workers are processing batches
    2. Shutdown is triggered while results are still in-flight
    3. Both pump threads and shutdown logic try to decrement

    The fix uses atomic try_decrement() and force_zero() to prevent TOCTOU races.
    """
    op = DelayById(max_delay_ms=50)
    node = Node(name="slow", op=op)
    stage = Stage(
        name="shutdown_race", nodes=[node], placement="auto", break_reason="test"
    )

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=4,
        deterministic=False,
        queue_capacity=8,
        prefetch_capacity=4,
        stage_output_mode="stream_items",
    )

    records = _mk_records(range(50))
    iterator = runner.run(iter(records))

    # Consume partial results, leaving many in-flight
    for _ in range(5):
        try:
            next(iterator)
        except StopIteration:
            break

    # Shutdown - should NOT raise "Inflight counter underflowed"
    runner.close()
