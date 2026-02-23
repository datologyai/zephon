from dataclasses import dataclass
from typing import Any

from tests.zephon.runners._helpers import (
    _ctx_services,
    _extract_values,
    _mk_record,
    _mk_records,
)
from zephon.core.accumulators import Accumulator, PassthroughAccumulator
from zephon.core.constants import SampleRecord
from zephon.core.graph import Node, Stage
from zephon.core.op_base import DefaultSetup, Op
from zephon.core.traits import OpTraits
from zephon.observability.config import ExecutionTrackingMode
from zephon.observability.stats import NodeMetricsDelta
from zephon.ops.delay import DelayById
from zephon.runners.threads import ThreadStageRunner


def _collect(runner: ThreadStageRunner, data: list[int]) -> list[int]:
    records = _mk_records(data)
    out_records = list(runner.run(iter(records)))
    return _extract_values(out_records)


def test_runner_emits_in_input_order_when_deterministic() -> None:
    # Build a single-stage plan with a delay op that will reorder completions
    op = DelayById(max_delay_ms=2.0)
    node = Node(name="delay", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")

    data = list(range(100))

    # Deterministic: outputs must match the input order exactly
    det_runner = ThreadStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=8,
        deterministic=True,
        stage_output_mode="stream_items",
    )
    out_det = _collect(det_runner, data)
    assert out_det == data

    # Non-deterministic: should still be a permutation; may equal by chance
    nondet_runner = ThreadStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=8,
        deterministic=False,
        stage_output_mode="stream_items",
    )
    out_nondet = _collect(nondet_runner, data)
    assert sorted(out_nondet) == sorted(data)


def test_run_one_returns_through_single_op_stage() -> None:
    # Single-op stage: DelayById is identity on payloads; run_one should pass through
    op = DelayById(max_delay_ms=0.0)
    node = Node(name="delay", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")

    runner = ThreadStageRunner(
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


def test_set_parallelism_errors_and_adjustments() -> None:
    op = DelayById(max_delay_ms=0.0)
    node = Node(name="delay", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")
    runner = ThreadStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=4,
        deterministic=True,
        stage_output_mode="stream_items",
    )

    # Invalid op index raises
    try:
        runner.set_parallelism(-1, 2)
        assert False, "expected IndexError"
    except IndexError:
        pass
    try:
        runner.set_parallelism(99, 2)
        assert False, "expected IndexError"
    except IndexError:
        pass

    # Grow then shrink while idle should succeed
    runner.set_parallelism(0, 3)
    runner.set_parallelism(0, 1)


def test_prefetching_stage_iterator_close_is_clean() -> None:
    # With prefetch_capacity > 0 we wrap with buffered_iterable; closing early must be clean
    op = DelayById(max_delay_ms=0.0)
    node = Node(name="delay", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")
    runner = ThreadStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=2,
        deterministic=True,
        prefetch_capacity=4,
        stage_output_mode="stream_items",
    )

    it = runner.run(iter(_mk_records(range(100))))
    # pull a few then close early
    got_records: list[SampleRecord] = []
    for _ in range(5):
        got_records.append(next(it))
    assert _extract_values(got_records) == list(range(5))
    # Explicitly close iterator; should not raise or hang
    if hasattr(it, "close"):
        it.close()  # type: ignore[call-arg]


def test_passthrough_stage_forwards_stream() -> None:
    # Empty stage (no ops) must pass through stream elements
    stage = Stage(name="empty", nodes=[], placement="auto", break_reason="test")
    runner = ThreadStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=2,
        deterministic=True,
        stage_output_mode="stream_items",
    )
    data = _mk_records(range(10))
    out = list(runner.run(iter(data)))
    assert out == data


@dataclass
class _IdentityOp(DefaultSetup, Op[Any, Any]):
    """Simple identity operator used for observability tests."""

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


def test_thread_runner_emits_metrics_deltas_when_callback_provided() -> None:
    op = _IdentityOp()
    node = Node(name="identity", op=op)
    stage = Stage(name="stage0", nodes=[node], placement="auto", break_reason="test")

    captured: list[NodeMetricsDelta] = []

    runner = ThreadStageRunner(
        stage,
        ctx_services=_ctx_services({"record_node_metrics": captured.append}),
        max_workers=2,
        deterministic=True,
        stage_index=7,
        tracking_mode=ExecutionTrackingMode.NODES,
        stage_output_mode="stream_items",
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
    assert all(delta.stage_name == "stage0" for delta in captured)


def test_thread_runner_emits_microbatches_and_accepts_batch_input() -> None:
    op = _IdentityOp()
    node = Node(name="identity", op=op)
    stage = Stage(name="stage1", nodes=[node], placement="auto", break_reason="test")

    runner = ThreadStageRunner(
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


def test_thread_runner_close_hard() -> None:
    """close(hard=True) should tear down quickly without hanging."""
    op = DelayById(max_delay_ms=0.0)
    node = Node(name="delay", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")

    runner = ThreadStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=4,
        deterministic=True,
        stage_output_mode="stream_items",
    )
    data = list(range(20))
    assert _collect(runner, data) == data
    runner.close(hard=True)


def test_thread_runner_close_hard_with_inflight() -> None:
    """Hard close during iteration should not hang or raise."""
    op = DelayById(max_delay_ms=5.0)
    node = Node(name="slow", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")

    runner = ThreadStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=4,
        deterministic=False,
        stage_output_mode="stream_items",
    )
    iterator = runner.run(iter(_mk_records(range(50))))
    # Consume a few items to get workers busy
    for _ in range(3):
        try:
            next(iterator)
        except StopIteration:
            break
    runner.close(hard=True)


def test_thread_runner_stream_mode_flattens_microbatch_input() -> None:
    op = _IdentityOp()
    node = Node(name="identity", op=op)
    stage = Stage(name="stage2", nodes=[node], placement="auto", break_reason="test")

    runner = ThreadStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=2,
        deterministic=True,
        stage_output_mode="stream_items",
    )

    microbatch = _mk_records(range(5))
    out = list(runner.run(iter([microbatch])))
    assert out == microbatch
