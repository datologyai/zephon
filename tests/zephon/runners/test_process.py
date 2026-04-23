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
    CountingAccumulator,
    PassthroughAccumulator,
)
from zephon.core.constants import SampleRecord, lane_of
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


torch = pytest.importorskip("torch")


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
class _TensorizeValueOp(DefaultSetup, Op[SampleRecord, SampleRecord]):
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
    ) -> Accumulator[SampleRecord]:
        return PassthroughAccumulator[SampleRecord]()

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        out: list[SampleRecord] = []
        for elem in elems:
            payload = dict(elem.payload)
            payload["tensor"] = torch.tensor([int(payload["value"])], dtype=torch.int64)
            out.append(SampleRecord(meta=elem.meta, payload=payload))
        return out


@dataclass
class _EncodeLargeBytesOp(DefaultSetup, Op[SampleRecord, SampleRecord]):
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
    ) -> Accumulator[SampleRecord]:
        return PassthroughAccumulator[SampleRecord]()

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        out: list[SampleRecord] = []
        for elem in elems:
            value = int(elem.payload["value"])
            payload = dict(elem.payload)
            payload["text"] = (f"value-{value}|".encode("utf-8")) * 1024
            out.append(SampleRecord(meta=elem.meta, payload=payload))
        return out


@dataclass
class _DecodeAndAnnotateBatchOp(DefaultSetup, Op[SampleRecord, SampleRecord]):
    max_batch: int = 3

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
    ) -> Accumulator[SampleRecord]:
        return CountingAccumulator[SampleRecord](
            max_batch=self.max_batch,
            max_latency_ms=None if deterministic else 1,
            key_fn=lane_of,
        )

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        batch_size = len(elems)
        out: list[SampleRecord] = []
        for elem in elems:
            payload = dict(elem.payload)
            payload["text"] = payload["text"].decode("utf-8")
            payload["batch_size"] = batch_size
            out.append(SampleRecord(meta=elem.meta, payload=payload))
        return out


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


def test_process_runner_coalesced_tensors_preserve_deterministic_order() -> None:
    tensorize = Node(name="tensorize", op=_TensorizeValueOp())
    delay = Node(name="delay", op=DelayById(max_delay_ms=1.5), inputs=[tensorize])
    stage = Stage(
        name="coalesced_tensors",
        nodes=[tensorize, delay],
        placement="auto",
        break_reason="test",
    )

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=4,
        deterministic=True,
        stage_output_mode="stream_items",
        coalesce_tensors=True,
    )

    data = list(range(30))
    out = list(runner.run(iter(_mk_records(data))))
    assert _extract_values(out) == data
    for expected, rec in zip(data, out, strict=True):
        assert int(rec.payload["tensor"][0]) == expected


@dataclass
class _BatchTensorizeOp(DefaultSetup, Op[SampleRecord, SampleRecord]):
    """Tensorize with batching so multiple records land in one microbatch."""

    max_batch: int = 8

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
    ) -> Accumulator[SampleRecord]:
        return CountingAccumulator[SampleRecord](
            max_batch=self.max_batch, max_latency_ms=None, key_fn=lane_of
        )

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        out: list[SampleRecord] = []
        for elem in elems:
            payload = dict(elem.payload)
            payload["tensor"] = torch.tensor([int(payload["value"])], dtype=torch.int64)
            out.append(SampleRecord(meta=elem.meta, payload=payload))
        return out


def test_process_runner_coalesced_tensors_are_zero_copy_views() -> None:
    """Restored tensors after IPC should be views into the same SHM storage."""
    tensorize = Node(name="tensorize", op=_BatchTensorizeOp(max_batch=8))
    stage = Stage(
        name="coalesced_zerocopy",
        nodes=[tensorize],
        placement="auto",
        break_reason="test",
    )

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
        coalesce_tensors=True,
    )

    # 8 records with max_batch=8 → one microbatch → one coalesced buffer
    data = list(range(8))
    out = list(runner.run(iter(_mk_records(data))))
    assert _extract_values(out) == data

    # All tensors from the same coalesced microbatch share one storage
    storages = {rec.payload["tensor"].untyped_storage().data_ptr() for rec in out}
    assert len(storages) == 1, (
        f"Expected all tensors to share one SHM storage, got {len(storages)}"
    )


def test_process_runner_coalesced_bytes_preserve_counting_accumulator_batches() -> None:
    encode = Node(name="encode", op=_EncodeLargeBytesOp())
    decode = Node(
        name="decode",
        op=_DecodeAndAnnotateBatchOp(max_batch=3),
        inputs=[encode],
    )
    stage = Stage(
        name="coalesced_bytes",
        nodes=[encode, decode],
        placement="auto",
        break_reason="test",
    )

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=4,
        deterministic=True,
        stage_output_mode="stream_items",
        coalesce_tensors=True,
    )

    out = list(runner.run(iter(_mk_records(range(7)))))
    assert _extract_values(out) == list(range(7))
    assert [rec.payload["batch_size"] for rec in out] == [3, 3, 3, 3, 3, 3, 1]
    for rec in out:
        assert rec.payload["text"].startswith(f"value-{rec.payload['value']}|")


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


def test_process_runner_close_hard() -> None:
    """close(hard=True) should tear down quickly without hanging."""
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
    data = list(range(20))
    assert _collect(runner, data) == data
    runner.close(hard=True)


def test_process_runner_close_hard_with_inflight() -> None:
    """Hard close during iteration should not hang or raise."""
    op = DelayById(max_delay_ms=50)
    node = Node(name="slow", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=2,
        deterministic=False,
        stage_output_mode="stream_items",
    )
    records = _mk_records(range(30))
    iterator = runner.run(iter(records))
    for _ in range(3):
        try:
            next(iterator)
        except StopIteration:
            break
    runner.close(hard=True)


# ---------------------------------------------------------------------------
# Sentinel bypass in process runner
# ---------------------------------------------------------------------------


def test_process_sentinel_bypass_accumulator_and_process_many() -> None:
    """Sentinels bypass worker processes and pass through unchanged.

    Uses DelayById (known to work with ProcessStageRunner).  A tombstone
    injected among regular records must emerge as a bare SampleRecord —
    it must not be sent to worker processes or modified in any way.
    """
    from zephon.core.constants import SampleMeta

    op = DelayById(max_delay_ms=0.0)
    node = Node(name="delay", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")

    runner = ProcessStageRunner(
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

    # The tombstone must appear as a bare SampleRecord.
    tombstones_out = [
        item for item in out if isinstance(item, SampleRecord) and item.meta.tombstone
    ]
    assert len(tombstones_out) == 1
    assert tombstones_out[0].meta.tombstone

    # All 6 regular records should pass through.
    regular_out = [
        item
        for item in out
        if isinstance(item, SampleRecord) and not item.meta.tombstone
    ]
    assert len(regular_out) == 6


# ---------------------------------------------------------------------------
# NamedQueue feeder error detection
# ---------------------------------------------------------------------------

from zephon.runners.queue import NamedQueue, QueueFeederError
from zephon.utils.shm import is_shm_error, shm_has_free_space


class TestFeederErrorDetection:
    """Tests for NamedQueue's feeder-error-to-sentinel mechanism."""

    def _make_queue(self, maxsize: int = 0) -> NamedQueue:
        ctx = multiprocessing.get_context("spawn")
        return NamedQueue("test", maxsize=maxsize, ctx=ctx)

    def test_sentinel_round_trip(self) -> None:
        """_on_queue_feeder_error should enqueue a sentinel that get() raises on."""
        q = self._make_queue(maxsize=10)
        try:
            # Start the feeder thread — it only starts on the first put().
            # In production, _on_queue_feeder_error is called BY the running
            # feeder thread; here we simulate it from the test thread.
            q.put("_start")
            assert q.get(timeout=5) == "_start"

            # Simulate a non-SHM serialization failure (e.g. unpicklable object).
            # SHM errors now enter the retry loop instead of creating a sentinel.
            try:
                raise TypeError("cannot pickle 'generator' object")
            except TypeError as e:
                q._on_queue_feeder_error(e, "dummy_obj")

            with pytest.raises(QueueFeederError, match="cannot pickle"):
                q.get(timeout=5)
        finally:
            q.close()

    def test_sentinel_contains_full_traceback(self) -> None:
        """The raised QueueFeederError should contain the full traceback."""
        q = self._make_queue(maxsize=10)
        try:
            q.put("_start")
            q.get(timeout=5)

            try:
                raise RuntimeError("shm_full_test")
            except RuntimeError as e:
                q._on_queue_feeder_error(e, "obj")

            with pytest.raises(QueueFeederError) as exc_info:
                q.get(timeout=5)

            msg = str(exc_info.value)
            assert "RuntimeError" in msg
            assert "shm_full_test" in msg
            assert "Traceback" in msg
        finally:
            q.close()

    def test_normal_items_pass_through(self) -> None:
        """get() should return normal items unchanged."""
        q = self._make_queue(maxsize=10)
        try:
            q.put("hello")
            q.put(42)
            assert q.get(timeout=5) == "hello"
            assert q.get(timeout=5) == 42
        finally:
            q.close()

    def test_sentinel_interleaved_with_normal_items(self) -> None:
        """Normal items before and after a sentinel should be returned normally."""
        q = self._make_queue(maxsize=10)
        try:
            q.put("before")

            try:
                raise RuntimeError("test_interleave")
            except RuntimeError as e:
                q._on_queue_feeder_error(e, "obj")

            q.put("after")

            assert q.get(timeout=5) == "before"
            with pytest.raises(QueueFeederError, match="test_interleave"):
                q.get(timeout=5)
            assert q.get(timeout=5) == "after"
        finally:
            q.close()

    def test_feeder_error_fallback_stderr(self, capsys: pytest.CaptureFixture) -> None:
        """When semaphore cannot be acquired, error should be printed to stderr."""
        # Create a queue with maxsize=1, fill it, so no semaphore slot is available.
        q = self._make_queue(maxsize=1)
        try:
            q.put("fill")  # fills the single slot

            # Monkey-patch _FEEDER_SHORT_TIMEOUT and _FEEDER_LONG_TIMEOUT
            # to avoid waiting 125s in a test.
            import zephon.runners.queue as queue_mod

            orig_short = queue_mod._FEEDER_SHORT_TIMEOUT
            orig_long = queue_mod._FEEDER_LONG_TIMEOUT
            queue_mod._FEEDER_SHORT_TIMEOUT = 0.01
            queue_mod._FEEDER_LONG_TIMEOUT = 0.01
            try:
                try:
                    raise RuntimeError("timeout_test")
                except RuntimeError as e:
                    q._on_queue_feeder_error(e, "obj")
            finally:
                queue_mod._FEEDER_SHORT_TIMEOUT = orig_short
                queue_mod._FEEDER_LONG_TIMEOUT = orig_long

            captured = capsys.readouterr()
            assert "CRITICAL" in captured.err
            assert "timeout_test" in captured.err
        finally:
            q.close()


# ---------------------------------------------------------------------------
# SHM backpressure retry
# ---------------------------------------------------------------------------

import errno
from multiprocessing.reduction import ForkingPickler
from unittest.mock import patch

import zephon.utils.shm as _shm_mod


@pytest.fixture()
def _fast_shm_retry():
    """Speed up SHM retry constants so tests don't sleep for seconds."""
    orig = (
        _shm_mod._SHM_RETRY_BASE_BACKOFF,
        _shm_mod._SHM_RETRY_MAX_BACKOFF,
        _shm_mod._SHM_RETRY_MAX_JITTER,
    )
    _shm_mod._SHM_RETRY_BASE_BACKOFF = 0.001
    _shm_mod._SHM_RETRY_MAX_BACKOFF = 0.01
    _shm_mod._SHM_RETRY_MAX_JITTER = 0
    yield
    (
        _shm_mod._SHM_RETRY_BASE_BACKOFF,
        _shm_mod._SHM_RETRY_MAX_BACKOFF,
        _shm_mod._SHM_RETRY_MAX_JITTER,
    ) = orig


class TestShmBackpressureRetry:
    """Tests for SHM-aware retry in _on_queue_feeder_error."""

    def _make_queue(self, maxsize: int = 0) -> NamedQueue:
        ctx = multiprocessing.get_context("spawn")
        return NamedQueue("test", maxsize=maxsize, ctx=ctx)

    # -- is_shm_error classification --

    def test_is_shm_error_enospc_oserror(self) -> None:
        assert is_shm_error(OSError(errno.ENOSPC, "No space left on device"))

    def test_is_shm_error_torch_runtime_error(self) -> None:
        err = RuntimeError(
            "unable to write to file </torch_xxx>: No space left on device (28)"
        )
        assert is_shm_error(err)

    def test_is_shm_error_chained_cause(self) -> None:
        inner = OSError(errno.ENOSPC, "No space left on device")
        outer = RuntimeError("pickle failed")
        outer.__cause__ = inner
        assert is_shm_error(outer)

    def test_is_shm_error_chained_context(self) -> None:
        inner = OSError(errno.ENOSPC, "No space left on device")
        outer = RuntimeError("pickle failed")
        outer.__context__ = inner
        assert is_shm_error(outer)

    def test_is_shm_error_non_shm_errors(self) -> None:
        assert not is_shm_error(TypeError("cannot pickle"))
        assert not is_shm_error(ValueError("bad value"))
        assert not is_shm_error(OSError(errno.EPERM, "permission denied"))
        assert not is_shm_error(RuntimeError("some other error"))

    # -- shm_has_free_space --

    def test_shm_has_free_space_returns_true_on_oserror(self) -> None:
        """On non-Linux (e.g. macOS) statvfs raises OSError — should not block."""
        with patch("zephon.utils.shm.os.statvfs", side_effect=OSError):
            assert shm_has_free_space() is True

    def test_shm_has_free_space_cgroup_limit(self) -> None:
        """When cgroup limit < tmpfs total, cgroup limit is the effective total."""
        import os as _os

        # Simulate 1000GB tmpfs with 50GB free, but cgroup limit of 200GB
        fake_st = _os.statvfs_result(
            (4096, 4096, 262144000, 13107200, 13107200, 0, 0, 0, 0, 255)
            # f_bsize=4096, f_frsize=4096, f_blocks=262144000 (=1000GB),
            # f_bfree=13107200 (=50GB), f_bavail=13107200
        )
        cgroup_limit = 200 * 1024**3  # 200GB

        with (
            patch("zephon.utils.shm.os.statvfs", return_value=fake_st),
            patch(
                "zephon.utils.shm.read_cgroup_memory_limit",
                return_value=cgroup_limit,
            ),
        ):
            # used = 1000GB - 50GB = 950GB, effective_total = 200GB
            # free = max(0, 200GB - 950GB) = 0  → 0% free → should return False
            assert shm_has_free_space(threshold=0.05) is False

    # -- Retry behavior --

    @pytest.mark.usefixtures("_fast_shm_retry")
    def test_shm_retry_requeues_item(self, capsys: pytest.CaptureFixture) -> None:
        """SHM error + space available → item re-queued, no sentinel."""
        q = self._make_queue(maxsize=10)
        try:
            q.put("_start")
            assert q.get(timeout=5) == "_start"

            with patch("zephon.utils.shm.shm_has_free_space", return_value=True):
                try:
                    raise RuntimeError(
                        "unable to write to file </torch_xxx>: "
                        "No space left on device (28)"
                    )
                except RuntimeError as e:
                    q._on_queue_feeder_error(e, "retry_obj")

            # The original object should be re-queued, NOT a sentinel
            item = q.get(timeout=5)
            assert item == "retry_obj"
        finally:
            q.close()

    @pytest.mark.usefixtures("_fast_shm_retry")
    def test_shm_retry_waits_for_space(self, capsys: pytest.CaptureFixture) -> None:
        """Retry should wait until SHM has free space."""
        q = self._make_queue(maxsize=10)
        try:
            q.put("_start")
            assert q.get(timeout=5) == "_start"

            # Return False 3 times, then True
            side_effects = [False, False, False, True]
            with patch(
                "zephon.utils.shm.shm_has_free_space",
                side_effect=side_effects,
            ):
                try:
                    raise OSError(errno.ENOSPC, "No space left on device")
                except OSError as e:
                    q._on_queue_feeder_error(e, "waited_obj")

            item = q.get(timeout=5)
            assert item == "waited_obj"

            captured = capsys.readouterr()
            assert "SHM pressure" in captured.err
        finally:
            q.close()

    @pytest.mark.usefixtures("_fast_shm_retry")
    def test_shm_retry_logs_warning(self, capsys: pytest.CaptureFixture) -> None:
        """First backoff should log a WARNING."""
        q = self._make_queue(maxsize=10)
        try:
            q.put("_start")
            assert q.get(timeout=5) == "_start"

            # First check finds no space (triggers backoff + warning),
            # second check finds space (re-queues).
            with patch(
                "zephon.utils.shm.shm_has_free_space",
                side_effect=[False, True],
            ):
                try:
                    raise RuntimeError("No space left on device")
                except RuntimeError as e:
                    q._on_queue_feeder_error(e, "obj")

            # Consume the re-queued item
            assert q.get(timeout=5) == "obj"

            captured = capsys.readouterr()
            assert "WARNING" in captured.err
            assert "SHM pressure" in captured.err
            assert "attempt 1" in captured.err
        finally:
            q.close()

    @pytest.mark.usefixtures("_fast_shm_retry")
    def test_shm_retry_success_log(self, capsys: pytest.CaptureFixture) -> None:
        """After multi-attempt retry, should log success with attempt count."""
        q = self._make_queue(maxsize=10)
        try:
            q.put("_start")
            assert q.get(timeout=5) == "_start"

            # Fail 2 times, then succeed
            with patch(
                "zephon.utils.shm.shm_has_free_space",
                side_effect=[False, False, True],
            ):
                try:
                    raise OSError(errno.ENOSPC, "No space left on device")
                except OSError as e:
                    q._on_queue_feeder_error(e, "obj")

            # Consume the re-queued item
            assert q.get(timeout=5) == "obj"

            captured = capsys.readouterr()
            assert "succeeded after" in captured.err
        finally:
            q.close()

    def test_non_shm_error_creates_sentinel(self) -> None:
        """Non-SHM errors should still create a sentinel (regression test)."""
        q = self._make_queue(maxsize=10)
        try:
            q.put("_start")
            assert q.get(timeout=5) == "_start"

            try:
                raise TypeError("cannot pickle 'generator' object")
            except TypeError as e:
                q._on_queue_feeder_error(e, "obj")

            with pytest.raises(QueueFeederError, match="cannot pickle"):
                q.get(timeout=5)
        finally:
            q.close()

    def test_shm_usage_in_sentinel_message(self) -> None:
        """Sentinel error message should include SHM usage stats."""
        q = self._make_queue(maxsize=10)
        try:
            q.put("_start")
            assert q.get(timeout=5) == "_start"

            try:
                raise TypeError("bad object")
            except TypeError as e:
                q._on_queue_feeder_error(e, "obj")

            with pytest.raises(QueueFeederError) as exc_info:
                q.get(timeout=5)
            assert "shm=" in str(exc_info.value)
        finally:
            q.close()


# ---------------------------------------------------------------------------
# SHM backpressure E2E (real _feed thread, simulated ENOSPC)
# ---------------------------------------------------------------------------


class _ShmPressureItem:
    """Test payload whose serialization raises ENOSPC for the first N attempts.

    Each instance carries a mutable ``_attempts`` counter.  Because
    ``_on_queue_feeder_error`` re-queues the *same object reference* via
    ``appendleft``, the counter survives across retries and the reducer
    eventually succeeds once ``_attempts > fail_count``.
    """

    def __init__(self, value: int, fail_count: int = 0):
        self.value = value
        self.fail_count = fail_count
        self._attempts = 0


def _shm_pressure_reduce(item: _ShmPressureItem) -> tuple:
    """ForkingPickler reducer — raises ENOSPC for the first *fail_count* attempts."""
    item._attempts += 1
    if item._attempts <= item.fail_count:
        raise OSError(errno.ENOSPC, "No space left on device")
    # Success — return a normal reduction tuple.
    return (_ShmPressureItem, (item.value, 0))


class TestShmBackpressureE2E:
    """E2E tests using the real Queue._feed thread with simulated SHM pressure.

    A custom ``ForkingPickler`` reducer is registered for ``_ShmPressureItem``
    so that serialization inside the ``_feed`` thread raises ``OSError(ENOSPC)``
    a configurable number of times before succeeding.  This exercises the full
    ``_feed`` → ``_on_queue_feeder_error`` → backoff → ``shm_has_free_space``
    → ``appendleft`` → re-serialize path without needing actual ``/dev/shm``
    pressure.
    """

    @pytest.fixture(autouse=True)
    def _setup_reducer(self, _fast_shm_retry):
        """Register custom reducer for SHM pressure simulation."""
        ForkingPickler.register(_ShmPressureItem, _shm_pressure_reduce)
        yield
        ForkingPickler._extra_reducers.pop(_ShmPressureItem, None)

    def _make_queue(self, maxsize: int = 0) -> NamedQueue:
        ctx = multiprocessing.get_context("spawn")
        return NamedQueue("test-shm-e2e", maxsize=maxsize, ctx=ctx)

    def test_occasional_pressure_delivers_all_items(self) -> None:
        """Mix of normal and pressure items — all delivered, none lost."""
        q = self._make_queue(maxsize=20)
        try:
            items = [
                _ShmPressureItem(0),  # ok
                _ShmPressureItem(1, fail_count=1),  # fails 1×
                _ShmPressureItem(2),  # ok
                _ShmPressureItem(3, fail_count=2),  # fails 2×
                _ShmPressureItem(4),  # ok
                _ShmPressureItem(5, fail_count=1),  # fails 1×
                _ShmPressureItem(6),  # ok
                _ShmPressureItem(7),  # ok
                _ShmPressureItem(8, fail_count=3),  # fails 3×
                _ShmPressureItem(9),  # ok
            ]

            with patch("zephon.utils.shm.shm_has_free_space", return_value=True):
                for item in items:
                    q.put(item)

                received = []
                for _ in range(len(items)):
                    result = q.get(timeout=10)
                    received.append(result.value)

            assert sorted(received) == list(range(10))
        finally:
            q.close()

    def test_pressure_items_preserve_order(self) -> None:
        """Failed items retry at the front of the buffer, preserving order."""
        q = self._make_queue(maxsize=20)
        try:
            items = [
                _ShmPressureItem(0),
                _ShmPressureItem(1, fail_count=1),
                _ShmPressureItem(2),
            ]

            with patch("zephon.utils.shm.shm_has_free_space", return_value=True):
                for item in items:
                    q.put(item)

                received = []
                for _ in range(len(items)):
                    result = q.get(timeout=10)
                    received.append(result.value)

            # appendleft re-queues the failed item before items behind it,
            # so ordering is preserved.
            assert received == [0, 1, 2]
        finally:
            q.close()

    def test_retry_fast_path_no_warning(self, capsys: pytest.CaptureFixture) -> None:
        """Momentary pressure that clears immediately produces no warning."""
        q = self._make_queue(maxsize=10)
        try:
            with patch("zephon.utils.shm.shm_has_free_space", return_value=True):
                q.put(_ShmPressureItem(42, fail_count=1))
                result = q.get(timeout=10)
                assert result.value == 42

            captured = capsys.readouterr()
            assert "WARNING" not in captured.err
        finally:
            q.close()

    def test_retry_logs_warnings(self, capsys: pytest.CaptureFixture) -> None:
        """Retried items produce SHM pressure warnings when space is tight."""
        q = self._make_queue(maxsize=10)
        try:
            # First check finds no space (triggers backoff + warning),
            # second check finds space (re-queues).
            with patch(
                "zephon.utils.shm.shm_has_free_space",
                side_effect=[False, True],
            ):
                q.put(_ShmPressureItem(42, fail_count=1))
                result = q.get(timeout=10)
                assert result.value == 42

            captured = capsys.readouterr()
            assert "SHM pressure" in captured.err
            assert "attempt 1" in captured.err
        finally:
            q.close()

    def test_retry_success_logged_after_space_wait(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """When shm_has_free_space returns False first, success is logged with count."""
        q = self._make_queue(maxsize=10)
        try:
            # False twice → True: _on_queue_feeder_error loops 3 times (attempt=3)
            with patch(
                "zephon.utils.shm.shm_has_free_space",
                side_effect=[False, False, True],
            ):
                q.put(_ShmPressureItem(42, fail_count=1))
                result = q.get(timeout=10)
                assert result.value == 42

            captured = capsys.readouterr()
            assert "SHM pressure" in captured.err
            assert "succeeded after" in captured.err
        finally:
            q.close()

    def test_retry_waits_for_free_space(self) -> None:
        """Retry blocks until shm_has_free_space returns True."""
        q = self._make_queue(maxsize=10)
        try:
            with patch(
                "zephon.utils.shm.shm_has_free_space",
                side_effect=[False, False, False, True],
            ):
                q.put(_ShmPressureItem(7, fail_count=1))
                result = q.get(timeout=10)
                assert result.value == 7
        finally:
            q.close()

    def test_no_sentinel_for_transient_pressure(self) -> None:
        """SHM errors must retry — never produce FeederError sentinels."""
        q = self._make_queue(maxsize=10)
        try:
            with patch("zephon.utils.shm.shm_has_free_space", return_value=True):
                q.put(_ShmPressureItem(1, fail_count=2))
                q.put(_ShmPressureItem(2))

                r1 = q.get(timeout=10)
                r2 = q.get(timeout=10)

            # Both are real items, not FeederError sentinels.
            assert isinstance(r1, _ShmPressureItem)
            assert isinstance(r2, _ShmPressureItem)
            assert {r1.value, r2.value} == {1, 2}
        finally:
            q.close()
