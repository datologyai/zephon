import multiprocessing
import multiprocessing.context
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

import pytest

# ``ZEPHON_WATCHDOG_POLL_S`` is tightened in
# ``tests/zephon/runners/conftest.py`` so it's applied before any
# test module triggers ``import zephon._internal.runners.process``.
from tests.zephon._internal.runners._helpers import (
    _ctx_services,
    _extract_values,
    _mk_record,
    _mk_records,
    _probe_stage,
)
from zephon._internal.graph import Node, Stage
from zephon._internal.ops.delay import DelayById
from zephon._internal.runners.concurrent import WorkerCrashed
from zephon._internal.runners.process import ProcessStageRunner
from zephon._internal.utils.shm_coalesce import PayloadMemoryPolicy
from zephon.observability.config import ExecutionTrackingMode
from zephon.observability.stats import NodeMetricsDelta
from zephon.ops.accumulators import (
    Accumulator,
    CountingAccumulator,
    PassthroughAccumulator,
)
from zephon.ops.base import BaseOp, OpContext
from zephon.ops.traits import OpTraits
from zephon.types import SampleRecord, StreamItem


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
class _IdentityOp(BaseOp):
    name: str = "identity"

    def __post_init__(self) -> None:
        super().__init__()

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


class _ValueMappingOp(BaseOp):
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
        super().__init__()

    def _map(self, value: int) -> int:
        return value + self.delta


@dataclass
class _MultiplyValueOp(_ValueMappingOp):
    factor: int = 1

    def __post_init__(self) -> None:
        super().__init__()

    def _map(self, value: int) -> int:
        return value * self.factor


@dataclass
class _TensorizeValueOp(BaseOp):
    def __post_init__(self) -> None:
        super().__init__()

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
class _EncodeLargeBytesOp(BaseOp):
    def __post_init__(self) -> None:
        super().__init__()

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
class _DecodeAndAnnotateBatchOp(BaseOp):
    max_batch: int = 3

    def __post_init__(self) -> None:
        super().__init__()

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
class _ServiceOp(BaseOp):
    def __post_init__(self) -> None:
        super().__init__()
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

    def setup(self, ctx: OpContext) -> None:
        super().setup(ctx)
        self._hook = ctx.get("custom_service")

    def process_one(self, elem: Any) -> list[Any]:
        if callable(self._hook):
            self._hook(elem.payload["value"])
        return [elem]

    def process_many(self, elems: list[Any]) -> list[Any]:
        return [out for e in elems for out in self.process_one(e)]


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
        memory_policy=PayloadMemoryPolicy(coalesce=True),
    )

    data = list(range(30))
    out = list(runner.run(iter(_mk_records(data))))
    assert _extract_values(out) == data
    for expected, rec in zip(data, out, strict=True):
        assert int(rec.payload["tensor"][0]) == expected


@dataclass
class _BatchTensorizeOp(BaseOp):
    """Tensorize with batching so multiple records land in one microbatch."""

    max_batch: int = 8

    def __post_init__(self) -> None:
        super().__init__()

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
            max_batch=self.max_batch, max_latency_ms=None
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
        memory_policy=PayloadMemoryPolicy(
            coalesce=True,
            min_item_bytes=0,
            min_new_allocation_bytes=0,
            min_forward_bytes=0,
        ),
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
        memory_policy=PayloadMemoryPolicy(coalesce=True),
    )

    out = list(runner.run(iter(_mk_records(range(7)))))
    assert _extract_values(out) == list(range(7))
    assert [rec.payload["batch_size"] for rec in out] == [3, 3, 3, 3, 3, 3, 1]
    for rec in out:
        assert rec.payload["text"].startswith(f"value-{rec.payload['value']}|")


@dataclass
class _CrashOp(BaseOp):
    def __post_init__(self) -> None:
        super().__init__()

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
class _LambdaOp(BaseOp):
    """Operator that uses a lambda function internally.

    This mimics what PackSequences does with its length_fn field.
    """

    transform: Any = None  # Will be set to a lambda in __post_init__

    def __post_init__(self) -> None:
        super().__init__()
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

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        return [out for e in elems for out in self.process_one(e)]


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
class _ClosureOp(BaseOp):
    """Operator that uses a closure (lambda capturing outer variable)."""

    multiplier: int = 3

    def __post_init__(self) -> None:
        super().__init__()
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

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        return [out for e in elems for out in self.process_one(e)]


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
    from zephon.types import SampleMeta

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

from zephon._internal.runners.queue import NamedQueue, QueueFeederError
from zephon._internal.utils.shm import is_shm_error, shm_has_free_space
from zephon.options import IpcTransport


@pytest.mark.parametrize("transport", ["socketpair", "pipe"])
class TestFeederErrorDetection:
    """Tests for NamedQueue's feeder-error-to-sentinel mechanism.

    Parametrized over both transports: the sentinel path must survive the
    default socketpair as well as the stock pipe."""

    def _make_queue(self, maxsize: int, transport: IpcTransport) -> NamedQueue:
        ctx = multiprocessing.get_context("spawn")
        return NamedQueue("test", maxsize=maxsize, ctx=ctx, transport=transport)

    def test_sentinel_round_trip(self, transport: IpcTransport) -> None:
        """_on_queue_feeder_error should enqueue a sentinel that get() raises on."""
        q = self._make_queue(maxsize=10, transport=transport)
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

    def test_sentinel_contains_full_traceback(self, transport: IpcTransport) -> None:
        """The raised QueueFeederError should contain the full traceback."""
        q = self._make_queue(maxsize=10, transport=transport)
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

    def test_normal_items_pass_through(self, transport: IpcTransport) -> None:
        """get() should return normal items unchanged."""
        q = self._make_queue(maxsize=10, transport=transport)
        try:
            q.put("hello")
            q.put(42)
            assert q.get(timeout=5) == "hello"
            assert q.get(timeout=5) == 42
        finally:
            q.close()

    def test_sentinel_interleaved_with_normal_items(
        self, transport: IpcTransport
    ) -> None:
        """Normal items before and after a sentinel should be returned normally."""
        q = self._make_queue(maxsize=10, transport=transport)
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

    def test_feeder_error_fallback_stderr(
        self, capsys: pytest.CaptureFixture, transport: IpcTransport
    ) -> None:
        """When semaphore cannot be acquired, error should be printed to stderr."""
        # Create a queue with maxsize=1, fill it, so no semaphore slot is available.
        q = self._make_queue(maxsize=1, transport=transport)
        try:
            q.put("fill")  # fills the single slot

            # Monkey-patch _FEEDER_SHORT_TIMEOUT and _FEEDER_LONG_TIMEOUT
            # to avoid waiting 125s in a test.
            import zephon._internal.runners.queue as queue_mod

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
import queue
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace
from unittest.mock import Mock, patch

import zephon._internal.utils.shm as _shm_mod
from zephon._internal.runners.process import _RemoteServiceProxy, _ServiceRequest


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
        with patch("zephon._internal.utils.shm.os.statvfs", side_effect=OSError):
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
            patch("zephon._internal.utils.shm.os.statvfs", return_value=fake_st),
            patch(
                "zephon._internal.utils.shm.read_cgroup_memory_limit",
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

            with patch(
                "zephon._internal.utils.shm.shm_has_free_space", return_value=True
            ):
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
                "zephon._internal.utils.shm.shm_has_free_space",
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
                "zephon._internal.utils.shm.shm_has_free_space",
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
                "zephon._internal.utils.shm.shm_has_free_space",
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

            with patch(
                "zephon._internal.utils.shm.shm_has_free_space", return_value=True
            ):
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

            with patch(
                "zephon._internal.utils.shm.shm_has_free_space", return_value=True
            ):
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
            with patch(
                "zephon._internal.utils.shm.shm_has_free_space", return_value=True
            ):
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
                "zephon._internal.utils.shm.shm_has_free_space",
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
                "zephon._internal.utils.shm.shm_has_free_space",
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
                "zephon._internal.utils.shm.shm_has_free_space",
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
            with patch(
                "zephon._internal.utils.shm.shm_has_free_space", return_value=True
            ):
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


# ---------------------------------------------------------------------------
# Resilient workers (watchdog-driven respawn on SIGSEGV)
# ---------------------------------------------------------------------------


class _CrashOnValue(BaseOp):
    """Worker-side op that segfaults when it sees any of *crash_values*.

    Uses a filesystem marker to communicate "I have already crashed" between
    a dying worker and its replacement, so ``mode="first_only"`` really
    means "crash exactly once per marker directory".
    """

    def __init__(
        self,
        crash_values: list[int],
        *,
        marker_dir: str,
        mode: str = "first_only",
    ) -> None:
        super().__init__()
        self._crash_values = list(crash_values)
        self._marker_dir = marker_dir
        self._mode = mode

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, preserves_cursor_order=True, parallelism=1)

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[StreamItem]:
        return CountingAccumulator[StreamItem](max_batch=1, max_latency_ms=None)

    def process_one(self, elem: StreamItem) -> list[StreamItem]:
        return self.process_many([elem])

    def process_many(self, elems: list[StreamItem]) -> list[StreamItem]:
        out: list[StreamItem] = []
        for item in elems:
            val: int | None = None
            if isinstance(item, SampleRecord):
                v = (
                    item.payload.get("value")
                    if isinstance(item.payload, dict)
                    else None
                )
                val = int(v) if isinstance(v, int) else None
            if val is not None and val in self._crash_values:
                if self._mode == "always":
                    _segfault_self()
                marker = Path(self._marker_dir) / f"crashed_{val}"
                if not marker.exists():
                    marker.touch()
                    _segfault_self()
            out.append(item)
        return out


class _CrashNTimes(BaseOp):
    """Crash the first N times the op sees *crash_value*; succeed thereafter.

    Uses filesystem markers (one per attempt) to count crashes across
    respawned workers — the worker dies so an in-process counter cannot
    survive.  Each crash writes ``attempt_<k>`` and the next worker
    reads how many markers exist to decide whether to crash again.
    """

    def __init__(self, crash_value: int, *, fail_count: int, marker_dir: str) -> None:
        super().__init__()
        self._crash_value = crash_value
        self._fail_count = fail_count
        self._marker_dir = marker_dir

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, preserves_cursor_order=True, parallelism=1)

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[StreamItem]:
        return CountingAccumulator[StreamItem](max_batch=1, max_latency_ms=None)

    def process_one(self, elem: StreamItem) -> list[StreamItem]:
        return self.process_many([elem])

    def process_many(self, elems: list[StreamItem]) -> list[StreamItem]:
        out: list[StreamItem] = []
        for item in elems:
            val: int | None = None
            if isinstance(item, SampleRecord):
                v = (
                    item.payload.get("value")
                    if isinstance(item.payload, dict)
                    else None
                )
                val = int(v) if isinstance(v, int) else None
            if val is not None and val == self._crash_value:
                existing = len(list(Path(self._marker_dir).glob("crash_*")))
                if existing < self._fail_count:
                    (Path(self._marker_dir) / f"crash_{existing}").touch()
                    _segfault_self()
            out.append(item)
        return out


def _segfault_self() -> None:  # pragma: no cover - terminates the worker
    # We're about to terminate this process via SIGSEGV.  Disable
    # corefile writing for *just this process* before signaling — the
    # kernel can spend hundreds of milliseconds writing a corefile,
    # which compounds across multiple intentional crashes per test.
    # Scope is the worker process about to die; the parent test
    # process and any other workers still get their default
    # ``RLIMIT_CORE``.  Real (unintentional) zephon crashes are
    # unaffected.
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass  # Windows or restricted env — fall through to crash anyway.
    os.kill(os.getpid(), signal.SIGSEGV)


def _fast_spawn_ctx() -> "multiprocessing.context.BaseContext | None":
    """Prefer a ``forkserver`` context so respawn cycles skip cold-import cost.

    ``forkserver`` starts a helper process once that pre-imports everything;
    each subsequent spawn is a near-instant fork from that helper (~100ms)
    instead of a cold Python startup (~20-30s on slow CI).  The zephon
    watchdog accepts both ``spawn`` and ``forkserver``; only ``fork`` gets
    downgraded (fork + watchdog thread deadlocks on ``Process.start()``).

    Returns ``None`` on platforms where ``forkserver`` is unavailable
    (Windows), which lets the caller fall through to the ``spawn`` default.
    """
    if "forkserver" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("forkserver")
    return None


def _mk_resilience_stage(op: Any, *, parallelism: int = 1) -> Stage:
    node = Node(name="crash", op=op, parallelism=parallelism)
    return Stage(name="s", nodes=[node], placement="auto", break_reason="test")


def _run_flat(runner: ProcessStageRunner, values: list[int]) -> list[int]:
    """Run values through runner and flatten any microbatch output."""
    records = _mk_records(values)
    out = list(runner.run(iter(records)))
    flat: list[SampleRecord] = []
    for item in out:
        if isinstance(item, list):
            flat.extend(rec for rec in item if isinstance(rec, SampleRecord))
        elif isinstance(item, SampleRecord):
            flat.append(item)
    return _extract_values(flat)


class TestResilientWorkers:
    """Watchdog-driven resubmit + respawn actor enabled by ``max_worker_retries > 0``.

    These tests intentionally trigger real SIGSEGV-style crashes inside
    worker processes — the whole point is that the main process does NOT
    die and the pipeline either recovers (transient) or fails with a
    structured error (exhaustion) or silently drops (non-det).  Tests
    self-skip when the platform can't exercise the crash path cleanly
    (e.g. missing ``torch``).
    """

    @pytest.mark.timeout(90)
    def test_transient_crash_recovers_under_max_worker_retries(
        self, tmp_path: Path
    ) -> None:
        op = _CrashOnValue([5], marker_dir=str(tmp_path), mode="first_only")
        runner = ProcessStageRunner(
            _mk_resilience_stage(op, parallelism=1),
            ctx_services=_ctx_services(),
            max_workers=1,
            deterministic=True,
            stage_output_mode="stream_items",
            max_worker_retries=3,
            mp_context=_fast_spawn_ctx(),
        )
        t0 = time.monotonic()
        out = _run_flat(runner, [5])
        elapsed = time.monotonic() - t0
        assert out == [5]
        # Loose bound — the real hang guard is pytest-timeout.  Local runs
        # finish in ~2s; CI under Python 3.14t free-threaded can take
        # ~10-15s per respawn due to slower imports + spawn.
        assert elapsed < 30.0, f"resilient run took too long: {elapsed:.1f}s"

    @pytest.mark.timeout(90)
    def test_persistent_crash_deterministic_escalates(self, tmp_path: Path) -> None:
        op = _CrashOnValue([7], marker_dir=str(tmp_path), mode="always")
        runner = ProcessStageRunner(
            _mk_resilience_stage(op, parallelism=1),
            ctx_services=_ctx_services(),
            max_workers=1,
            deterministic=True,
            stage_output_mode="stream_items",
            max_worker_retries=2,
            mp_context=_fast_spawn_ctx(),
        )
        t0 = time.monotonic()
        with pytest.raises(WorkerCrashed) as excinfo:
            _run_flat(runner, [7])
        elapsed = time.monotonic() - t0
        assert excinfo.value.info.exc_type == "MaxWorkerRetriesExceeded"
        assert "seq=" in excinfo.value.info.message
        # 2 retries → 3 sequential cold spawns + 3 watchdog poll waits
        # (5s each by default).  Local runs finish in ~5s; 3.12 cold CI
        # has been observed at ~55s.  Real hang guard is the outer
        # pytest-timeout(90); this elapsed bound just catches gross
        # regressions.
        assert elapsed < 80.0, f"exhaustion path took too long: {elapsed:.1f}s"

    @pytest.mark.timeout(90)
    def test_persistent_crash_nondet_drops_and_continues(self, tmp_path: Path) -> None:
        op = _CrashOnValue([42], marker_dir=str(tmp_path), mode="always")
        runner = ProcessStageRunner(
            _mk_resilience_stage(op, parallelism=1),
            ctx_services=_ctx_services(),
            max_workers=1,
            deterministic=False,
            stage_output_mode="stream_items",
            max_worker_retries=1,
            mp_context=_fast_spawn_ctx(),
        )
        t0 = time.monotonic()
        out = _run_flat(runner, [1, 42, 2, 3])
        elapsed = time.monotonic() - t0
        # Poisoned sample 42 dropped silently; good samples pass through.
        # With blind-resubmit, good seqs still in task_queue when the
        # worker died get re-dispatched too — the pump dedups via the
        # ``pending_commands.pop(seq) is None`` branch in _handle_result,
        # so extras are harmlessly ack-and-dropped.
        assert set(out) == {1, 2, 3}
        assert 42 not in out
        assert elapsed < 30.0, f"non-det drop path took too long: {elapsed:.1f}s"

    def test_fork_mp_context_downgrades_when_retries_enabled(self) -> None:
        """Fork + retries warns and silently disables resilience.

        We don't raise because ``max_worker_retries`` defaults to 3 and we
        don't want a default-on feature to break existing fork-based
        pipelines.  The caller gets a RuntimeWarning so the downgrade is
        visible.
        """
        if "fork" not in multiprocessing.get_all_start_methods():
            pytest.skip("fork start method unavailable on this platform")
        op = _CrashOnValue([], marker_dir=tempfile.gettempdir())
        stage = _mk_resilience_stage(op, parallelism=1)
        with pytest.warns(RuntimeWarning, match="disabling worker-resilience watchdog"):
            runner = ProcessStageRunner(
                stage,
                ctx_services=_ctx_services(),
                max_workers=1,
                max_worker_retries=1,
                mp_context=multiprocessing.get_context("fork"),
            )
        # Resilience silently became a no-op.
        assert runner._max_worker_retries == 0
        runner.close()

    def test_fork_mp_context_accepted_when_retries_disabled(self) -> None:
        if "fork" not in multiprocessing.get_all_start_methods():
            pytest.skip("fork start method unavailable on this platform")
        op = _CrashOnValue([], marker_dir=tempfile.gettempdir())
        stage = _mk_resilience_stage(op, parallelism=1)
        runner = ProcessStageRunner(
            stage,
            ctx_services=_ctx_services(),
            max_workers=1,
            max_worker_retries=0,
            mp_context=multiprocessing.get_context("fork"),
        )
        runner.close()

    def test_is_unexpected_death_classifies_signal_termination(self) -> None:
        """Negative exitcodes trigger recovery; clean exits and pre-reap None do not.

        Clean exits (0 or positive) and transient ``None`` exitcodes must
        be ignored by the watchdog, or a routine clean shutdown or a
        caught Python exception can trigger a spurious
        ``_handle_dead_worker`` that deadlocks on ``task_queue.put()``
        while holding ``_shutdown_lock``.

        The predicate is currently signal-agnostic (just ``code < 0``),
        so the specific signal values below are documentation / regression
        guard against a future narrowing that would skip some of them,
        not branch coverage.
        """

        class _FakeProc:
            def __init__(self, exitcode: int | None):
                self.exitcode = exitcode

        canonical_crash_signals = (
            -signal.SIGSEGV,
            -signal.SIGKILL,
            -signal.SIGABRT,
            -signal.SIGBUS,
            -signal.SIGTERM,
            -signal.SIGINT,
        )
        for code in canonical_crash_signals:
            assert ProcessStageRunner._is_unexpected_death(_FakeProc(code))  # type: ignore[arg-type]

        # Non-negative exit codes and the pre-reap transient must NOT trigger.
        for code in (0, 1, None):
            assert not ProcessStageRunner._is_unexpected_death(_FakeProc(code))  # type: ignore[arg-type]

    @pytest.mark.timeout(90)
    def test_det_mode_multiworker_backpressure_deadlock_recovers(
        self, tmp_path: Path
    ) -> None:
        """Regression for a production deadlock: in det mode with
        ``parallelism>1``, when a worker crashes on an early seq, bystander
        workers keep producing results into ``pending_results`` (held by
        backpressure permits) until they exhaust their permit budget and
        block at ``backpressure.acquire()``.  ``task_queue`` fills with
        post-culprit seqs, pump's ``_send_command`` starts the drain-retry
        loop but sees an empty ``result_queue`` (blocked workers can't put),
        and the pipeline hangs.

        The ``_RelaxSignal`` path breaks this: the watchdog posts the
        signal onto ``result_queue``, the pump's drain-retry loop picks
        it up, post-hoc releases the permits held by ``pending_results``
        entries, switches to arrival-release until the resubmits come
        back, and the emit cascade unblocks everyone once the respawned
        worker processes the culprit.

        The value ``0`` crashes the first worker to see it
        (``first_only``); with ``max_worker_retries=3`` the respawn retries
        and succeeds.  Without the fix this test hangs until pytest-timeout
        fires.
        """
        op = _CrashOnValue([0], marker_dir=str(tmp_path), mode="first_only")
        runner = ProcessStageRunner(
            _mk_resilience_stage(op, parallelism=2),
            ctx_services=_ctx_services(),
            max_workers=2,
            deterministic=True,
            stage_output_mode="stream_items",
            max_worker_retries=3,
            mp_context=_fast_spawn_ctx(),
        )
        t0 = time.monotonic()
        # 24 items = 3x the total permit budget (queue_capacity=4 per
        # worker x 2 workers = 8 permits, task_queue cap = 8), well past
        # any threshold where the deadlock could be masked.
        out = _run_flat(runner, list(range(24)))
        elapsed = time.monotonic() - t0
        assert sorted(out) == list(range(24)), f"unexpected output: {out}"
        assert out == list(range(24))  # det mode preserves input order
        assert elapsed < 45.0, f"recovery took too long: {elapsed:.1f}s"

    # ------- gap 7: multi-worker crash exercises /proc-syscall path -------

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason=(
            "Validates the /proc/PID/syscall live-writer-discrimination "
            "code path in zephon._internal.runners.watchdog. /proc only exists on "
            "Linux; macOS multi-worker recovery is a known limitation "
            "that this test can't exercise."
        ),
    )
    @pytest.mark.timeout(90)
    @pytest.mark.parametrize("ipc_transport", ["socketpair", "pipe"])
    def test_multiworker_crash_with_proc_syscall_recovery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ipc_transport: str
    ) -> None:
        """Smoke test: parallelism>1 + crash + ``/proc/syscall``-only
        recovery doesn't hang and produces correct output, over both
        IPC transports.

        Disables ``rotation`` (parallelism=1-only anyway) and
        ``timed_acquire`` so every worker death routes through the
        ``/proc/syscall`` step.  Note: this does *not* prove
        live-writer discrimination per se — for that, see the focused
        unit tests in
        ``tests/zephon/runners/test_watchdog.py``, which
        spawn real processes with known states and check the helpers
        directly.  This integration test confirms the chain is wired
        up correctly and that the parallelism>1 flow doesn't regress.

        ``ZEPHON_RECLAIM_DISABLE`` is read dynamically at every
        recovery call by ``_disabled_strategies()`` so this monkeypatch
        applies to every crash within the test, including those in
        respawned worker pools.
        """
        monkeypatch.setenv("ZEPHON_RECLAIM_DISABLE", "rotation,timed_acquire")
        for trial in range(3):
            trial_dir = tmp_path / f"trial_{trial}"
            trial_dir.mkdir()
            op = _CrashOnValue([0], marker_dir=str(trial_dir), mode="first_only")
            runner = ProcessStageRunner(
                _mk_resilience_stage(op, parallelism=2),
                ctx_services=_ctx_services(),
                max_workers=2,
                deterministic=True,
                stage_output_mode="stream_items",
                max_worker_retries=3,
                mp_context=_fast_spawn_ctx(),
                ipc_transport=ipc_transport,
            )
            # 24 items lets both workers do meaningful concurrent work
            # around the time of the crash.  Det mode means we'll catch
            # any duplicate / lost / corrupted output by the strict
            # equality check.
            out = _run_flat(runner, list(range(24)))
            assert out == list(range(24)), (
                f"trial {trial}: unexpected output (corruption?): {out}"
            )

    # ------- gap 3: empty pending_commands on idle worker death -------

    @pytest.mark.timeout(90)
    def test_idle_worker_death_respawns_without_pending_work(
        self, tmp_path: Path
    ) -> None:
        """Watchdog respawns a worker whose ``pending_commands`` is empty
        at the moment of death — regression guard for the empty-pending
        branch in ``_handle_dead_worker``.

        Construction: consume exactly as many ``next(iterator)`` calls as
        records (3/3), which drains ``pending_commands`` without letting
        the outer ``run()`` generator advance to ``StopIteration`` (and
        therefore without triggering ``_after_run`` → ``_shutdown_workers``).
        Workers stay alive, pending is empty, and the pre-kill
        ``is_alive()`` guard catches the narrow window where the worker
        has already begun a clean shutdown (the feeder pushes
        ``_Stop`` after the input iterator raises StopIteration; with 3
        records fully consumed, the worker is racing between that path
        and our SIGKILL).  When ``is_alive()`` is True, the SIGKILL wins
        and we genuinely exercise the empty-pending respawn path.
        """
        op = _CrashOnValue([], marker_dir=str(tmp_path))  # never crashes on its own
        runner = ProcessStageRunner(
            _mk_resilience_stage(op, parallelism=1),
            ctx_services=_ctx_services(),
            max_workers=1,
            deterministic=True,
            stage_output_mode="stream_items",
            max_worker_retries=3,
            mp_context=_fast_spawn_ctx(),
        )
        iterator = runner.run(iter(_mk_records([0, 1, 2])))
        try:
            for _ in range(3):
                next(iterator)

            state = runner.ops[0]
            # Pump pops pending_commands before emitting; allow a brief
            # settle window for any trailing bookkeeping after the yield.
            drain_deadline = time.monotonic() + 2.0
            while state.pending_commands and time.monotonic() < drain_deadline:
                time.sleep(0.05)
            assert not state.pending_commands, (
                f"pending_commands not empty post-drain: {list(state.pending_commands)}"
            )

            assert state.workers, "worker list should be populated mid-iteration"
            worker_proc = state.workers[0]
            old_pid = worker_proc.pid
            assert old_pid is not None
            assert worker_proc.is_alive(), "worker should be alive pre-kill"
            os.kill(old_pid, signal.SIGKILL)

            # Wait for the watchdog to respawn the worker.  Bound matches
            # the other resilience tests' elapsed-time tolerance — local
            # runs finish in ~1-2s; CI under Python 3.14t free-threaded
            # can take 10-15s per spawn due to slower imports.  The
            # watchdog thread mutates ``state.workers``; catch
            # ``IndexError`` defensively during the narrow swap window.
            respawn_deadline = time.monotonic() + 30.0
            respawned = False
            while time.monotonic() < respawn_deadline:
                try:
                    current = state.workers[0]
                except IndexError:
                    time.sleep(0.05)
                    continue
                if current.pid != old_pid and current.is_alive():
                    respawned = True
                    break
                time.sleep(0.1)
            assert respawned, "watchdog did not respawn the idle worker within 30s"
        finally:
            try:
                iterator.close()
            except Exception:
                pass
            runner.close()

    # ------- gap 4: retry counter boundary -------

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        "fail_count,max_retries,should_succeed",
        [
            (1, 1, True),  # 1 crash + 1 retry -> 2nd attempt passes (at budget)
            (2, 1, False),  # 2 crashes + 1 retry -> exhausts just over budget
            (2, 2, True),  # 2 crashes + 2 retries -> 3rd attempt passes (at budget)
            (3, 2, False),  # 3 crashes + 2 retries -> exhausts just over budget
        ],
    )
    def test_retry_counter_boundary(
        self,
        tmp_path: Path,
        fail_count: int,
        max_retries: int,
        should_succeed: bool,
    ) -> None:
        """At-budget crashes recover; just-over-budget crashes escalate.

        Contract: ``max_worker_retries=N`` allows up to N retries after
        the initial dispatch.  A seq that crashes K times succeeds iff
        K <= N, otherwise escalates to
        ``WorkerCrashed(MaxWorkerRetriesExceeded)``.
        """
        op = _CrashNTimes(
            crash_value=0, fail_count=fail_count, marker_dir=str(tmp_path)
        )
        runner = ProcessStageRunner(
            _mk_resilience_stage(op, parallelism=1),
            ctx_services=_ctx_services(),
            max_workers=1,
            deterministic=True,
            stage_output_mode="stream_items",
            max_worker_retries=max_retries,
            mp_context=_fast_spawn_ctx(),
        )
        if should_succeed:
            out = _run_flat(runner, [0])
            assert out == [0]
        else:
            with pytest.raises(WorkerCrashed) as excinfo:
                _run_flat(runner, [0])
            assert excinfo.value.info.exc_type == "MaxWorkerRetriesExceeded"

    # ------- gap 6: resilience with multi-op stage -------

    @pytest.mark.timeout(30)
    def test_multi_op_stage_recovers_from_worker_crash(self, tmp_path: Path) -> None:
        """Multi-op stages exercise a different dispatch path than single-op
        direct-IPC; verify transient crash recovery still works when
        ``_single_op_direct_ipc`` is False.

        Stage has two ops: a harmless ``_AddValueOp`` followed by a
        ``_CrashOnValue`` that segfaults once on ``value=15``.  The add-op
        maps inputs value -> value+10, the crash-op crashes on the first
        post-shift value of 15, and the respawned worker succeeds.
        """
        add = Node(name="add", op=_AddValueOp(delta=10))
        crash = Node(
            name="crash",
            op=_CrashOnValue([15], marker_dir=str(tmp_path), mode="first_only"),
            inputs=[add],
        )
        stage = Stage(
            name="multi_op_crash",
            nodes=[add, crash],
            placement="auto",
            break_reason="test",
        )

        runner = ProcessStageRunner(
            stage,
            ctx_services=_ctx_services(),
            max_workers=1,
            deterministic=True,
            stage_output_mode="stream_items",
            max_worker_retries=3,
            mp_context=_fast_spawn_ctx(),
        )
        # Chains of ops force the non-fast path.
        assert getattr(runner, "_single_op_direct_ipc") is False

        data = [1, 2, 3, 4, 5]
        out = _run_flat(runner, data)
        assert out == [v + 10 for v in data]


def test_process_worker_receives_stage_info() -> None:
    runner = ProcessStageRunner(
        _probe_stage("probe"),
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_index=2,
        stage_output_mode="stream_items",
    )
    (out,) = list(runner.run(iter(_mk_records([7]))))
    assert out.payload["probe"] == (2, "probe_stage", 0, False)


@pytest.mark.parametrize("failure", ["cancelled", "parent"])
def test_proxy_wait_exits_on_cancellation_or_parent_death(failure: str) -> None:
    request, response = queue.Queue(), queue.Queue()
    cancelled, pending = SimpleNamespace(value=0), SimpleNamespace(value=0)
    parent = Mock()
    parent.is_alive.return_value = True
    proxy = _RemoteServiceProxy(0, "metadata", request, response, cancelled, pending)

    def stop_wait(**_: Any) -> None:
        assert pending.value  # put() does not mean the feeder has finished.
        if failure == "parent":
            parent.is_alive.return_value = False
        else:
            cancelled.value = 1
        raise queue.Empty

    with patch("multiprocessing.parent_process", return_value=parent):
        with patch.object(response, "get", side_effect=stop_wait):
            with pytest.raises(RuntimeError, match="cancelled|parent process"):
                proxy("report")
        assert not pending.value
        response.put((True, "late reply"))
        with pytest.raises(RuntimeError, match="cancelled"):
            proxy("next report")
    assert request.qsize() == 1


def test_proxy_slow_response_has_no_wall_clock_deadline() -> None:
    request, response = queue.Queue(), queue.Queue()
    proxy = _RemoteServiceProxy(
        0,
        "metadata",
        request,
        response,
        SimpleNamespace(value=0),
        SimpleNamespace(value=0),
    )
    # An arbitrarily late reply remains valid. No startup/handler time budget.
    with patch.object(response, "get", side_effect=[queue.Empty, (True, "result")]):
        with patch("time.monotonic", side_effect=[0.0, 3600.0]):
            assert proxy("report") == "result"


@pytest.fixture
def control_runner() -> Any:
    runner = ProcessStageRunner(
        _probe_stage("control"),
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
    )
    try:
        yield runner
    finally:
        runner.close(hard=True)


def test_service_transport_failure_is_reported(
    control_runner: ProcessStageRunner,
) -> None:
    context = control_runner._create_context()
    original = OSError("broken channel")

    def broken_transport() -> None:
        raise original

    with patch.object(control_runner, "_active_context", context):
        with patch.object(
            control_runner, "_serve_requests", side_effect=broken_transport
        ):
            control_runner._service_loop()
    assert control_runner._service_cancelled.value
    assert context.stop_event.is_set()
    assert context.stage_out_queue.get_nowait() is context.stop_token
    assert context.error is original
    with pytest.raises(OSError, match="broken channel") as caught:
        raise context.error
    assert caught.value is original
    assert (
        traceback.extract_tb(caught.value.__traceback__)[-1].name == "broken_transport"
    )


def test_failed_start_removes_pre_registered_endpoint(
    control_runner: ProcessStageRunner,
) -> None:
    response = object()
    control_runner._service_responses[17] = response
    control_runner._service_pending[17] = SimpleNamespace(value=0)
    proc = Mock()
    proc.start.side_effect = OSError("start failed")
    with patch.object(control_runner, "_close_ipc_queue") as close:
        with pytest.raises(OSError, match="start failed"):
            control_runner._start_worker(proc, 17)
    assert not control_runner._service_responses and not control_runner._service_pending
    close.assert_called_once_with(response, hard=True)


def test_retired_worker_request_cannot_reply_to_replacement(
    control_runner: ProcessStageRunner,
) -> None:
    accepted = []
    requests, response = queue.Queue(), queue.Queue()
    requests.put(_ServiceRequest(4, "custom_service", ("old",), {}))
    requests.put(_ServiceRequest(5, "custom_service", ("new",), {}))
    requests.put(None)
    with patch.object(control_runner, "_service_queue", requests):
        with patch.object(control_runner, "_service_responses", {5: response}):
            with patch.dict(
                control_runner._ctx_services, custom_service=accepted.append
            ):
                control_runner._serve_requests()
    assert accepted == ["new"]
    assert response.get_nowait() == (True, None)
    assert response.empty()


def _start_inheriting_helper(pid_path: Path) -> None:
    """Start a helper that outlives the worker, like torch_shm_manager."""
    helper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], close_fds=False
    )
    pid_path.write_text(str(helper.pid))


def _kill_helper(pid_path: Path) -> None:
    if pid_path.exists() and (pid := pid_path.read_text()):
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _worker_with_inheriting_helper(pid_path: Path, close_on_exec: bool) -> None:
    """Optionally protect the spawned worker's fds before starting its helper."""
    from zephon._internal.runners.process import _set_worker_fds_close_on_exec

    if close_on_exec:
        _set_worker_fds_close_on_exec()
    _start_inheriting_helper(pid_path)


@pytest.mark.parametrize("close_on_exec", [True, False])
def test_worker_exit_is_visible_despite_helper_subprocess(
    tmp_path: Path, close_on_exec: bool
) -> None:
    """Only the unprotected control should leave the exit pipe open."""
    pid_path = tmp_path / "helper.pid"
    proc = multiprocessing.get_context("spawn").Process(
        target=_worker_with_inheriting_helper, args=(pid_path, close_on_exec)
    )
    proc.start()
    try:
        deadline = time.monotonic() + 30
        while not pid_path.exists() or not pid_path.read_text():
            assert time.monotonic() < deadline, "child never started its helper"
            time.sleep(0.05)
        proc.join(timeout=3.0)
        # Check the pipe itself: exitcode/is_alive can reap the worker even
        # when its helper keeps the pipe open and join has timed out.
        assert bool(wait([proc.sentinel], timeout=0)) is close_on_exec
        assert proc.exitcode == 0
    finally:
        _kill_helper(pid_path)
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        proc.close()


@pytest.mark.timeout(30)
def test_process_runner_worker_exit_is_visible_despite_helper_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise fd protection through real worker startup and operator setup."""
    pid_path = tmp_path / "helper.pid"

    class HelperOp(_IdentityOp):
        def setup(self, ctx: OpContext) -> None:
            super().setup(ctx)
            _start_inheriting_helper(pid_path)

    node = Node(name="helper", op=HelperOp())
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")
    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )
    # Keep a regression's timeout short; assertions below inspect the pipe,
    # rather than imposing a timing bound on worker imports and setup.
    monkeypatch.setattr("zephon._internal.runners.process._GRACEFUL_WORKER_JOIN", 3.0)
    try:
        with patch.object(runner, "_start_worker", wraps=runner._start_worker) as start:
            assert _collect(runner, [7]) == [7]
        start.assert_called_once()
        proc = start.call_args.args[0]
        os.kill(int(pid_path.read_text()), 0)  # The helper must still be alive.
        assert wait([proc.sentinel], timeout=0), (
            "helper retained the worker's exit pipe"
        )
        assert proc.exitcode == 0
    finally:
        _kill_helper(pid_path)
        runner.close()


class _CopySharedArray(BaseOp):
    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True)

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        from zephon._internal.utils.shm_coalesce import _shared_numpy_storage

        outputs = []
        for record in elems:
            array: Any = record.payload
            assert _shared_numpy_storage(array) is not None
            copied = array.copy()
            copied[0] += 1
            outputs.append(SampleRecord(meta=record.meta, payload=copied))
        return outputs


def test_process_numpy_reduction_forwards_zephon_buffers_between_ops() -> None:
    import pickle
    from multiprocessing.reduction import ForkingPickler

    from zephon._internal.stream import resolve_lazy_payloads
    from zephon._internal.utils.shm_coalesce import prepare_microbatch

    np = pytest.importorskip("numpy")
    record = _mk_record(0)
    record.payload = np.arange(32)[3:20:2]
    prepared = prepare_microbatch(
        [record],
        PayloadMemoryPolicy(
            min_item_bytes=0, min_new_allocation_bytes=0, min_forward_bytes=0
        ),
    )
    records = pickle.loads(ForkingPickler.dumps(prepared))
    resolve_lazy_payloads(records)
    [record] = records
    array = record.payload
    stage = Stage(
        "shared_numpy",
        [
            Node("first", _CopySharedArray(), parallelism=1),
            Node("second", _CopySharedArray(), parallelism=1),
        ],
        "auto",
        "test",
    )
    runner = ProcessStageRunner(
        stage,
        _ctx_services(),
        max_workers=2,
        deterministic=True,
        memory_policy=PayloadMemoryPolicy(
            min_item_bytes=0, min_new_allocation_bytes=0, min_forward_bytes=0
        ),
        stage_output_mode="stream_items",
    )
    try:
        [result] = list(runner.run([record]))
        assert isinstance(result, SampleRecord)
        assert record.payload is array
        expected = array.copy()
        expected[0] += 2
        np.testing.assert_array_equal(result.payload, expected)
    finally:
        runner.close()


class _InspectPayloadMemory(BaseOp):
    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, preserves_cursor_order=True, parallelism=1)

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        from zephon._internal.utils.shm_coalesce import _shared_numpy_storage, _ShmBytes

        for record in elems:
            payload = record.payload
            payload["observed"] = (
                payload["small_t"].is_shared(),
                _shared_numpy_storage(payload["small_n"]) is not None,
                payload["large_t"].is_shared(),
                _shared_numpy_storage(payload["large_n"]) is not None,
                isinstance(payload["large_b"], _ShmBytes),
                isinstance(payload["large_ba"], _ShmBytes),
            )
            for key in ("large_b", "large_ba"):
                assert bytes(payload[key]) == b"x" * 512
                assert isinstance(payload[key], _ShmBytes)
        return elems


def _unexpected_shm_preparation(*args, **kwargs):
    raise AssertionError("disabled SHM must bypass preparation")


class _InspectDisabledShm(BaseOp):
    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, preserves_cursor_order=True, parallelism=1)

    def setup(self, ctx: OpContext) -> None:
        from zephon._internal.runners import process

        process.prepare_microbatch = _unexpected_shm_preparation

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        import numpy as np

        for record in elems:
            assert type(record.payload["bytes"]) is bytes
            assert type(record.payload["mutable"]) is bytearray
            assert type(record.payload["array"]) is np.ndarray
            record.payload["mutable"][0] = 7
            record.payload["array"][0] = 7
        return elems


def test_disabled_shm_skips_sender_and_worker_preparation(monkeypatch) -> None:
    import numpy as np

    from zephon._internal.runners import process

    monkeypatch.setattr(process, "TransportMicrobatch", _unexpected_shm_preparation)
    stage = Stage(
        "disabled_shm", [Node("inspect", _InspectDisabledShm())], "auto", "test"
    )
    record = _mk_record(0)
    record.payload = {"bytes": b"abc", "mutable": bytearray(16), "array": np.arange(16)}
    runner = ProcessStageRunner(
        stage,
        _ctx_services(),
        max_workers=1,
        deterministic=True,
        memory_policy=None,
        stage_output_mode="stream_items",
    )
    try:
        [actual] = list(runner.run([record]))
        assert actual.payload["bytes"] == b"abc"
        assert actual.payload["mutable"][0] == actual.payload["array"][0] == 7
        assert record.payload["mutable"][0] == record.payload["array"][0] == 0
    finally:
        runner.close()


def test_process_memory_policy_applies_on_input_and_across_lazy_hops() -> None:
    import numpy as np

    record = _mk_record(0)
    record.payload = {
        "small_t": torch.arange(4),
        "small_n": np.arange(4),
        "large_t": torch.arange(64),
        "large_n": np.arange(64),
        "large_b": b"x" * 512,
        "large_ba": bytearray(b"x" * 512),
    }
    stage = Stage(
        "memory_policy",
        [
            Node("first", _InspectPayloadMemory(), parallelism=1),
            Node("second", _InspectPayloadMemory(), parallelism=1),
        ],
        "auto",
        "test",
    )
    runner = ProcessStageRunner(
        stage,
        _ctx_services(),
        max_workers=2,
        deterministic=True,
        stage_output_mode="stream_items",
        # Exercise both routes without depending on production tuning defaults.
        memory_policy=PayloadMemoryPolicy(
            min_item_bytes=64, min_new_allocation_bytes=512, min_forward_bytes=256
        ),
    )
    try:
        [actual] = list(runner.run([record]))
        assert actual.payload["observed"] == (False, False, True, True, True, True)
        assert not actual.payload["small_t"].is_shared()
        assert (
            actual.payload["small_t"].tolist()
            == actual.payload["small_n"].tolist()
            == list(range(4))
        )
        assert not record.payload["small_t"].is_shared()
    finally:
        runner.close()
