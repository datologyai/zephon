# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for RemoteStageRunner (ConcurrentStageRunner integration).

Mirrors the test coverage in test_threads.py and test_process.py for the
Ray-backed runner: deterministic ordering, stream/microbatch modes, metrics,
sentinel handling, error propagation, serialization, and shutdown safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

pytestmark = [pytest.mark.requires_ray, pytest.mark.usefixtures("ray_init")]

from tests.zephon.runners._helpers import (
    _collect,
    _ctx_services,
    _extract_values,
    _make_stage,
    _mk_record,
    _mk_records,
)
from zephon.core.accumulators import Accumulator, PassthroughAccumulator
from zephon.core.constants import SampleBatch, SampleMeta, SampleRecord
from zephon.core.graph import Node, Stage
from zephon.core.op_base import DefaultSetup, Op
from zephon.core.traits import OpTraits
from zephon.observability.config import ExecutionTrackingMode
from zephon.observability.stats import NodeMetricsDelta
from zephon.ops.batch import Batch

pytest.importorskip("ray")


def _make_runner(
    stage: Stage,
    *,
    max_workers: int = 2,
    deterministic: bool = True,
    queue_capacity: int = 4,
    prefetch_capacity: int = 0,
    stage_output_mode: str = "microbatches",
    stage_index: int = 0,
    tracking_mode: ExecutionTrackingMode = ExecutionTrackingMode.OFF,
    ctx: dict[str, Any] | None = None,
):
    from zephon.runners.ray import RemoteStageRunner

    return RemoteStageRunner(
        stage,
        ctx or _ctx_services(),
        max_workers=max_workers,
        deterministic=deterministic,
        queue_capacity=queue_capacity,
        prefetch_capacity=prefetch_capacity,
        stage_index=stage_index,
        tracking_mode=tracking_mode,
        stage_output_mode=stage_output_mode,
    )


# ---------------------------------------------------------------------------
# Helper ops (mirrors test_threads.py / test_process.py)
# ---------------------------------------------------------------------------


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


@dataclass
class _LambdaOp(DefaultSetup, Op[SampleRecord, SampleRecord]):
    transform: Any = None

    def __post_init__(self) -> None:
        DefaultSetup.__init__(self)
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
        out: list[SampleRecord] = []
        for elem in elems:
            out.extend(self.process_one(elem))
        return out


@dataclass
class _ClosureOp(DefaultSetup, Op[SampleRecord, SampleRecord]):
    multiplier: int = 3

    def __post_init__(self) -> None:
        DefaultSetup.__init__(self)
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
        out: list[SampleRecord] = []
        for elem in elems:
            out.extend(self.process_one(elem))
        return out


# ===================================================================
# Basic passthrough and ordering
# ===================================================================


class TestRemoteStageRunner:
    """Core RemoteStageRunner tests."""

    def test_basic_passthrough(self) -> None:
        """Simple data flows through and comes back."""
        stage = _make_stage(max_delay_ms=0.0, parallelism=2, placement="remote")
        runner = _make_runner(stage, stage_output_mode="stream_items")
        data = list(range(10))
        assert sorted(_collect(runner, data)) == data
        runner.close()

    def test_deterministic_order(self) -> None:
        """Deterministic mode preserves input order."""
        stage = _make_stage(parallelism=2, max_delay_ms=2.0, placement="remote")
        runner = _make_runner(
            stage, max_workers=4, deterministic=True, stage_output_mode="stream_items"
        )
        data = list(range(100))
        assert _collect(runner, data) == data

    def test_non_deterministic_all_items(self) -> None:
        """Non-deterministic mode processes all items (order may vary)."""
        stage = _make_stage(max_delay_ms=0.0, parallelism=2, placement="remote")
        runner = _make_runner(
            stage, deterministic=False, stage_output_mode="stream_items"
        )
        data = list(range(15))
        out = _collect(runner, data)
        assert sorted(out) == data
        runner.close()

    def test_multi_operator_stage(self) -> None:
        """Stage with multiple operators chains correctly."""
        stage = _make_stage(
            max_delay_ms=0.0, parallelism=2, num_ops=2, placement="remote"
        )
        runner = _make_runner(stage, stage_output_mode="stream_items")
        data = list(range(10))
        assert sorted(_collect(runner, data)) == data
        runner.close()

    def test_empty_input(self) -> None:
        """Empty input produces no output."""
        stage = _make_stage(max_delay_ms=0.0, parallelism=1, placement="remote")
        runner = _make_runner(stage, max_workers=1)
        results = list(runner.run(iter([])))
        assert results == []
        runner.close()


# ===================================================================
# run_one
# ===================================================================


class TestRunOne:
    def test_run_one_returns_through_single_op_stage(self) -> None:
        stage = _make_stage(max_delay_ms=0.0, parallelism=1, placement="remote")
        runner = _make_runner(stage, max_workers=1, stage_output_mode="stream_items")
        record = _mk_record(7)
        out = runner.run_one(record)
        assert isinstance(out, SampleRecord)
        assert out == record


# ===================================================================
# Stream and microbatch output modes
# ===================================================================


class TestOutputModes:
    def test_microbatches_and_batch_input(self) -> None:
        """Microbatch mode emits lists; batch input preserved."""
        op = _IdentityOp()
        node = Node(name="identity", op=op)
        stage = _make_stage(nodes=[node], placement="remote")
        runner = _make_runner(stage, stage_output_mode="microbatches")

        singles = _mk_records(range(2))
        batch = _mk_records(range(2, 5))
        upstream = iter([singles[0], singles[1], batch])
        out = list(runner.run(upstream))

        assert len(out) == 3
        assert all(isinstance(elem, list) for elem in out)
        assert _extract_values(out[0]) == [0]
        assert _extract_values(out[1]) == [1]
        assert _extract_values(out[2]) == [2, 3, 4]

    def test_stream_mode_flattens_microbatch_input(self) -> None:
        """Stream mode flattens microbatch input into individual records."""
        op = _IdentityOp()
        node = Node(name="identity", op=op)
        stage = _make_stage(nodes=[node], placement="remote")
        runner = _make_runner(stage, stage_output_mode="stream_items")

        microbatch = _mk_records(range(5))
        out = list(runner.run(iter([microbatch])))
        assert out == microbatch


# ===================================================================
# Passthrough (empty stage)
# ===================================================================


class TestPassthrough:
    def test_passthrough_stage_forwards_stream(self) -> None:
        """Empty stage (no ops) passes through stream elements."""
        stage = Stage(name="empty", nodes=[], placement="remote", break_reason="test")
        runner = _make_runner(stage, stage_output_mode="stream_items")
        data = _mk_records(range(10))
        out = list(runner.run(iter(data)))
        assert out == data


# ===================================================================
# Prefetch + early close
# ===================================================================


class TestPrefetch:
    def test_prefetch_iterator_close_is_clean(self) -> None:
        """With prefetch_capacity > 0, closing the iterator early is clean."""
        stage = _make_stage(max_delay_ms=0.0, parallelism=2, placement="remote")
        runner = _make_runner(
            stage,
            prefetch_capacity=4,
            stage_output_mode="stream_items",
        )

        it = runner.run(iter(_mk_records(range(100))))
        got: list[SampleRecord] = []
        for _ in range(5):
            got.append(next(it))
        assert _extract_values(got) == list(range(5))
        if hasattr(it, "close"):
            it.close()  # type: ignore[call-arg]


# ===================================================================
# Metrics / observability
# ===================================================================


class TestMetrics:
    def test_emits_metrics_deltas_when_callback_provided(self) -> None:
        """Metrics are collected and forwarded when tracking is enabled."""
        op = _IdentityOp()
        node = Node(name="identity", op=op)
        stage = _make_stage(nodes=[node], placement="remote")

        captured: list[NodeMetricsDelta] = []
        ctx = _ctx_services({"record_node_metrics": captured.append})

        runner = _make_runner(
            stage,
            stage_index=7,
            tracking_mode=ExecutionTrackingMode.NODES,
            stage_output_mode="stream_items",
            ctx=ctx,
        )

        data = _mk_records(range(6))
        out = list(runner.run(iter(data)))
        assert out == data

        assert captured, "expected at least one metrics delta"
        produced = sum(delta.produced_elements for delta in captured)
        consumed = sum(delta.consumed_elements for delta in captured)
        assert produced == len(data)
        assert consumed == len(data)
        assert all(delta.stage_index == 7 for delta in captured)
        assert all(delta.stage_name == "test_stage" for delta in captured)
        runner.close()


# ===================================================================
# Multi-operator chains with transforms
# ===================================================================


class TestMultiOpChains:
    def test_chained_transform_ops(self) -> None:
        """Multi-op stage with add + multiply chains correctly."""
        add = Node(name="add", op=_AddValueOp(delta=1))
        multiply = Node(name="mul", op=_MultiplyValueOp(factor=3), inputs=[add])
        stage = _make_stage(nodes=[add, multiply], placement="remote")

        runner = _make_runner(stage, max_workers=3, stage_output_mode="stream_items")

        data = list(range(10))
        out = _collect(runner, data)
        assert out == [(v + 1) * 3 for v in data]

    def test_multi_op_draining_forwards_to_next_op(self) -> None:
        """Multi-op with tight capacity routes results through all operators."""
        add1 = Node(name="add1", op=_AddValueOp(delta=1))
        add10 = Node(name="add10", op=_AddValueOp(delta=10), inputs=[add1])
        stage = _make_stage(nodes=[add1, add10], placement="remote")

        runner = _make_runner(
            stage,
            max_workers=4,
            queue_capacity=1,
            stage_output_mode="stream_items",
        )

        data = list(range(100))
        out = _collect(runner, data)
        expected = [v + 11 for v in data]
        assert out == expected


# ===================================================================
# Error propagation
# ===================================================================


class TestErrorPropagation:
    def test_bubbles_worker_exceptions(self) -> None:
        """Exceptions in actors are propagated to the caller."""
        op = _CrashOp()
        node = Node(name="crash", op=op)
        stage = _make_stage(nodes=[node], placement="remote")
        runner = _make_runner(stage, max_workers=1, stage_output_mode="stream_items")

        with pytest.raises(RuntimeError) as excinfo:
            list(runner.run(iter(_mk_records(range(3)))))
        text = str(excinfo.value)
        assert "ValueError" in text
        assert "boom inside worker" in text


# ===================================================================
# Serialization (cloudpickle)
# ===================================================================


class TestSerialization:
    def test_serializes_operator_with_lambda(self) -> None:
        """Operators containing lambdas serialize correctly via cloudpickle."""
        op = _LambdaOp()
        node = Node(name="lambda_op", op=op)
        stage = _make_stage(nodes=[node], placement="remote")

        runner = _make_runner(stage, stage_output_mode="stream_items")
        data = list(range(5))
        out = _collect(runner, data)
        assert out == [x * 2 for x in data]

    def test_serializes_operator_with_closure(self) -> None:
        """Operators with closures (lambdas capturing outer variables) work."""
        op = _ClosureOp(multiplier=5)
        node = Node(name="closure_op", op=op)
        stage = _make_stage(nodes=[node], placement="remote")

        runner = _make_runner(stage, stage_output_mode="stream_items")
        data = list(range(4))
        out = _collect(runner, data)
        assert out == [x * 5 for x in data]


# ===================================================================
# Sentinel (tombstone) handling
# ===================================================================


class TestSentinelHandling:
    def test_sentinel_bypass_accumulator_and_process_many(self) -> None:
        """Sentinels bypass actors and pass through unchanged.

        The Batch operator with microbatch_size=3 buffers regular records in
        its accumulator and wraps them into SampleBatch via process_many.
        Sentinels (tombstones) must not be buffered or wrapped — they should
        pass through unchanged.
        """
        op = Batch(3)
        node = Node(name="batch", op=op)
        stage = _make_stage(nodes=[node], placement="remote")

        runner = _make_runner(stage, max_workers=1, stage_output_mode="stream_items")

        tomb_meta = SampleMeta(
            sample_id=(0, 0, 99), lane_id=0, chunk_id=0
        ).with_tombstone(True)
        tomb = SampleRecord(meta=tomb_meta, payload={})

        inputs: list[SampleRecord] = [_mk_record(i) for i in range(3)]
        inputs.append(tomb)
        inputs.extend(_mk_record(i) for i in range(3, 6))

        out = list(runner.run(iter(inputs)))

        tombstones_out = [
            item
            for item in out
            if isinstance(item, SampleRecord) and item.meta.tombstone
        ]
        assert len(tombstones_out) == 1
        assert tombstones_out[0].meta.tombstone

        batches_out = [item for item in out if isinstance(item, SampleBatch)]
        total_regular = sum(len(b.records) for b in batches_out)
        assert total_regular == 6


# ===================================================================
# Shutdown and close safety
# ===================================================================


class TestShutdown:
    def test_close_without_run(self) -> None:
        """close() can be called without ever calling run()."""
        stage = _make_stage(max_delay_ms=0.0, parallelism=1, placement="remote")
        runner = _make_runner(stage, max_workers=1)
        runner.close()  # Should not raise

    def test_close_hard(self) -> None:
        """close(hard=True) tears down cleanly after a full run."""
        stage = _make_stage(max_delay_ms=0.0, parallelism=2, placement="remote")
        runner = _make_runner(stage, max_workers=4, stage_output_mode="stream_items")
        data = list(range(20))
        assert _collect(runner, data) == data
        runner.close(hard=True)

    def test_close_hard_with_inflight(self) -> None:
        """Hard close during iteration should not hang or raise."""
        stage = _make_stage(parallelism=2, max_delay_ms=5.0, placement="remote")
        runner = _make_runner(
            stage,
            max_workers=4,
            deterministic=False,
            stage_output_mode="stream_items",
        )
        iterator = runner.run(iter(_mk_records(range(50))))
        for _ in range(3):
            try:
                next(iterator)
            except StopIteration:
                break
        runner.close(hard=True)

    def test_partial_iteration_shutdown_no_underflow(self) -> None:
        """Partial iteration + shutdown must not cause inflight counter underflow."""
        stage = _make_stage(parallelism=2, max_delay_ms=50.0, placement="remote")
        runner = _make_runner(
            stage,
            max_workers=4,
            deterministic=False,
            queue_capacity=8,
            prefetch_capacity=4,
            stage_output_mode="stream_items",
        )

        records = _mk_records(range(50))
        iterator = runner.run(iter(records))
        for _ in range(5):
            try:
                next(iterator)
            except StopIteration:
                break
        runner.close()


# ===================================================================
# Runner parity tests
# ===================================================================


def _make_thread_runner(stage, **kwargs):
    from zephon.runners.threads import ThreadStageRunner

    return ThreadStageRunner(stage, _ctx_services(), **kwargs)


def _make_process_runner(stage, **kwargs):
    from zephon.runners.process import ProcessStageRunner

    return ProcessStageRunner(stage, _ctx_services(), **kwargs)


class TestThreadsProcessParity:
    """Threads vs process runner parity (no Ray dependency)."""

    def test_deterministic_output_parity(self) -> None:
        """Threads and process runners produce identical deterministic output."""
        data = list(range(20))
        stage = _make_stage(max_delay_ms=1.0, parallelism=4)
        common = dict(max_workers=4, deterministic=True, queue_capacity=4)

        threads_out = _collect(_make_thread_runner(stage, **common), data)
        process_out = _collect(_make_process_runner(stage, **common), data)

        assert threads_out == process_out == data


class TestAllRunnersParity:
    """Threads vs process vs ray runner parity."""

    def test_threads_process_ray_identical_output(self) -> None:
        """All three runners produce identical deterministic output."""
        data = list(range(20))
        stage = _make_stage(max_delay_ms=1.0, parallelism=4)
        common = dict(max_workers=4, deterministic=True, queue_capacity=4)

        threads_out = _collect(_make_thread_runner(stage, **common), data)
        process_out = _collect(_make_process_runner(stage, **common), data)
        ray_out = _collect(_make_runner(stage, **common), data)

        assert threads_out == process_out == ray_out == data

    @pytest.mark.parametrize("dop", [1, 2, 4])
    def test_parity_varying_dop(self, dop: int) -> None:
        """Parity holds at different parallelism levels."""
        data = list(range(30))
        stage = _make_stage(max_delay_ms=0.5, parallelism=dop)
        common = dict(max_workers=max(dop, 2), deterministic=True, queue_capacity=4)

        threads_out = _collect(_make_thread_runner(stage, **common), data)
        ray_out = _collect(_make_runner(stage, **common), data)

        assert threads_out == ray_out == data

    def test_parity_multi_operator_chain(self) -> None:
        """Parity with a 2-operator stage."""
        data = list(range(15))
        stage = _make_stage(max_delay_ms=0.5, parallelism=2, num_ops=2)
        common = dict(max_workers=2, deterministic=True, queue_capacity=4)

        threads_out = _collect(_make_thread_runner(stage, **common), data)
        ray_out = _collect(_make_runner(stage, **common), data)

        assert threads_out == ray_out == data

    def test_non_deterministic_parity_all_runners(self) -> None:
        """Non-deterministic mode: all items present, order may differ."""
        data = list(range(25))
        stage = _make_stage(max_delay_ms=0.5, parallelism=4)
        common = dict(max_workers=4, deterministic=False, queue_capacity=4)

        threads_out = _collect(_make_thread_runner(stage, **common), data)
        process_out = _collect(_make_process_runner(stage, **common), data)
        ray_out = _collect(_make_runner(stage, **common), data)

        assert sorted(threads_out) == sorted(process_out) == sorted(ray_out) == data

    def test_single_element(self) -> None:
        """Single-element input works across all runners."""
        data = [42]
        stage = _make_stage(max_delay_ms=0.0, parallelism=1)
        common = dict(max_workers=1, deterministic=True, queue_capacity=4)

        threads_out = _collect(_make_thread_runner(stage, **common), data)
        ray_out = _collect(_make_runner(stage, **common), data)

        assert threads_out == ray_out == data


class TestMultiTokenParity:
    """Verify multi-token doesn't break deterministic ordering."""

    def test_multi_token_deterministic_parity(self) -> None:
        """Identical output with queue_capacity=1 and queue_capacity=4."""
        data = list(range(30))
        stage = _make_stage(max_delay_ms=0.5, parallelism=4)
        common = dict(
            max_workers=4,
            deterministic=True,
            stage_output_mode="stream_items",
        )

        out_1 = _collect(_make_runner(stage, queue_capacity=1, **common), data)
        out_4 = _collect(_make_runner(stage, queue_capacity=4, **common), data)

        assert out_1 == out_4 == data
