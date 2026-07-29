import queue
import threading
from dataclasses import dataclass
from typing import Any

from tests.zephon._internal.runners._helpers import (
    _ctx_services,
    _extract_values,
    _mk_record,
    _mk_records,
)
from zephon._internal.graph import Node, Stage
from zephon._internal.op_base import DefaultSetup, Op
from zephon._internal.ops.batch import Batch
from zephon._internal.ops.delay import DelayById
from zephon._internal.ops.pack_sequences import PackSequences
from zephon._internal.runners.concurrent import RunnerResult
from zephon._internal.runners.threads import ThreadStageRunner
from zephon.observability.config import ExecutionTrackingMode
from zephon.observability.stats import NodeMetricsDelta
from zephon.ops.accumulators import Accumulator, PassthroughAccumulator
from zephon.ops.traits import OpTraits
from zephon.types import SampleBatch, SampleMeta, SampleRecord


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


# ---------------------------------------------------------------------------
# Sentinel bypass in thread runner
# ---------------------------------------------------------------------------


def test_thread_sentinel_bypass_accumulator_and_process_many() -> None:
    """Sentinels bypass accumulator and process_many in the thread runner.

    The Batch operator with microbatch_size=3 buffers regular records in its
    accumulator and wraps them into SampleBatch via process_many.  Sentinels
    (tombstones) must not be buffered or wrapped — they should pass through
    unchanged, just like in the inline runner.
    """
    op = Batch(3)
    node = Node(name="batch", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")

    runner = ThreadStageRunner(
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
    batches_out = [item for item in out if isinstance(item, SampleBatch)]
    total_regular = sum(len(b.records) for b in batches_out)
    assert total_regular == 6


# ---------------------------------------------------------------------------
# Sentinel deadlock on full result_queue
# ---------------------------------------------------------------------------


def test_thread_sentinel_does_not_deadlock_on_full_result_queue() -> None:
    """Sentinel scheduling must not block when the result_queue is full.

    Bug: ThreadStageRunner._schedule_batch puts sentinel results directly
    into result_queue via _put_result.  The pump thread is the only thread
    that drains result_queue (via _drain_results).  If the queue is already
    full (e.g. a worker completed between drain and sentinel scheduling),
    _put_result blocks → pump can never drain → self-deadlock.

    The fix stashes sentinel results in _local_results (a plain list) that
    _post_schedule_batch drains inline on the pump thread.
    """
    op = _IdentityOp()
    node = Node(name="identity", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")

    runner = ThreadStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        queue_capacity=1,
        stage_output_mode="stream_items",
    )

    state = runner.ops[0]
    context = runner._create_context()

    # Pre-fill result_queue to capacity.  This simulates a worker whose
    # done_callback put a result into result_queue between the pump's
    # _drain_results and the sentinel's _schedule_batch.
    dummy = RunnerResult(
        seq=0,
        payload=[],
        wait_ns=0,
        consumed_elements=0,
        consumed_bytes=0,
        queue_depth_snapshot=-1,
        proc_ns=0,
        collect_metrics=False,
    )
    state.result_queue.put(dummy)
    state.next_seq = 1  # dummy consumed seq 0

    tomb_meta = SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0).with_tombstone(
        True
    )
    sentinel = SampleRecord(meta=tomb_meta, payload={})

    # Schedule sentinel in a helper thread so we can detect blocking.
    completed = threading.Event()

    def schedule() -> None:
        runner._schedule_batch(state, [sentinel], wait_ns=0, context=context)
        completed.set()

    t = threading.Thread(target=schedule, daemon=True)
    t.start()
    t.join(timeout=2.0)

    deadlocked = not completed.is_set()

    # Cleanup: unblock the stuck thread by draining result_queue.
    if deadlocked:
        try:
            state.result_queue.get_nowait()
        except queue.Empty:
            pass
        t.join(timeout=1.0)

    assert not deadlocked, (
        "Sentinel scheduling deadlocked: _schedule_batch called _put_result "
        "on the pump thread while result_queue was full.  The pump thread is "
        "the only consumer of result_queue, so it blocked on itself."
    )


# ---------------------------------------------------------------------------
# Epoch floor protocol
# ---------------------------------------------------------------------------


def _rec_with_chunk(value: int, *, chunk_id: int) -> SampleRecord:
    meta = SampleMeta(
        sample_id=(0, 0, value),
        lane_id=0,
        chunk_id=chunk_id,
        chunk_offset=value,
    )
    return SampleRecord(meta=meta, payload={"value": value, "length": 1})


def test_thread_runner_epoch_floor() -> None:
    """ThreadStageRunner exposes the same epoch_floor() protocol as the base."""
    op = PackSequences(
        max_length=10, num_bins=2, length_fn=lambda r: r.payload["length"]
    )
    node = Node(name="pack", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")

    runner = ThreadStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )

    assert runner.epoch_floor() is None

    runner.ops[0].enqueue(
        [_rec_with_chunk(0, chunk_id=4), _rec_with_chunk(1, chunk_id=2)]
    )
    assert runner.epoch_floor() == 2

    runner.close()


# ---------------------------------------------------------------------------
# Sentinel deadlock on full result_queue
# ---------------------------------------------------------------------------


def _flush_sentinel(*, lane_id: int = 0, boundary_cid: int = 0) -> SampleRecord:
    """Create a flush sentinel record."""
    meta = SampleMeta(
        sample_id=(0, 0, 0),
        lane_id=lane_id,
        chunk_id=0,
        chunk_offset=0,
        tags={"_flush_sentinel": True, "_boundary_cid": boundary_cid},
    )
    return SampleRecord(meta=meta, payload={})


def test_thread_sentinel_does_not_deadlock_on_full_result_queue() -> None:
    """Sentinel scheduling must not block when the result_queue is full.

    Bug: ThreadStageRunner._schedule_batch puts sentinel results directly
    into result_queue via _put_result.  The pump thread is the only thread
    that drains result_queue (via _drain_results).  If the queue is already
    full (e.g. a worker completed between drain and sentinel scheduling),
    _put_result blocks → pump can never drain → self-deadlock.

    The process runner avoids this by stashing sentinel results in a local
    list (_local_results) handled inline by _post_schedule_batch.  The
    thread runner should use its equivalent mechanism (sync_result stash).
    """
    op = _IdentityOp()
    node = Node(name="identity", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")

    runner = ThreadStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        queue_capacity=1,
        stage_output_mode="stream_items",
    )

    state = runner.ops[0]
    context = runner._create_context()

    # Pre-fill result_queue to capacity.  This simulates a worker whose
    # done_callback put a result into result_queue between the pump's
    # _drain_results and the sentinel's _schedule_batch.
    dummy = RunnerResult(
        seq=0,
        payload=[],
        wait_ns=0,
        consumed_elements=0,
        consumed_bytes=0,
        queue_depth_snapshot=-1,
        proc_ns=0,
        collect_metrics=False,
    )
    state.result_queue.put(dummy)
    state.next_seq = 1  # dummy consumed seq 0

    sentinel = _flush_sentinel()

    # Schedule sentinel in a helper thread so we can detect blocking.
    completed = threading.Event()

    def schedule() -> None:
        runner._schedule_batch(state, [sentinel], wait_ns=0, context=context)
        completed.set()

    t = threading.Thread(target=schedule, daemon=True)
    t.start()
    t.join(timeout=2.0)

    deadlocked = not completed.is_set()

    # Cleanup: unblock the stuck thread by draining result_queue.
    if deadlocked:
        try:
            state.result_queue.get_nowait()
        except queue.Empty:
            pass
        t.join(timeout=1.0)

    assert not deadlocked, (
        "Sentinel scheduling deadlocked: _schedule_batch called _put_result "
        "on the pump thread while result_queue was full.  The pump thread is "
        "the only consumer of result_queue, so it blocked on itself."
    )
