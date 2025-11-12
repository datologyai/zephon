# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Threaded stage runner that drives operators with bounded queues.

Deterministic vs non-deterministic behaviour
-------------------------------------------

This runner supports two execution modes controlled by the `deterministic`
flag supplied at construction time:

- Non-deterministic (default):
  - Operators may buffer by count and by latency (``max_latency_ms``) as
    requested by their ``Buffering`` trait. Micro-batches are scheduled into a
    thread pool, and downstream receives results in completion order. This
    maximizes throughput/latency at the cost of non-stable ordering when tasks
    complete at different times.

- Deterministic:
  - Time-based flushes are enabled (every ``max_latency_ms``) unless the
    engine explicitly disables them (e.g., when any operator in the stage is
    batch-shape sensitive). In any case, count-based buffering is honoured.
  - Each scheduled micro-batch receives a monotonically increasing sequence
    number. Results are collected and re-ordered so that downstream observes
    the exact same order it would in a single-threaded execution regardless of
    completion timing. Within a micro-batch, element order is preserved.

The deterministic path is optimistic by construction: every scheduled
micro-batch is tagged with a sequence number, and the runner always enqueues an
explicit ``(seq, payload)`` pair to the result queue. Consumers of that queue
therefore do not need to defensively handle missing sequence numbers.
"""

import copy
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

from zephon.core.constants import EngineSample, SampleBatch, SampleRecord
from zephon.core.graph import Node, Stage
from zephon.core.op_base import Op, OpContext
from zephon.core.traits import Buffering
from zephon.observability.config import ExecutionTrackingMode
from zephon.observability.size_estimator import estimate_bytes
from zephon.observability.stats import NodeMetricsDelta
from zephon.observability.stopwatch import Stopwatch
from zephon.utils import buffered_iterable

_QItem = TypeVar("_QItem")


class _Stop:
    pass


StopToken = _Stop


class ThreadStageRunner:
    """Execute a stage locally with bounded queues between operators.

    The stage exposes a pull-driven iterator (`run`) while orchestrating push-
    based execution across the fused operators.  Each operator owns a small input
    queue, optional buffering rules, and a pool of worker instances.  When a
    micro-batch is ready it is pushed to the worker pool; results are pushed into
    the next operator's queue (or the final stage output queue).  Bounded queues
    provide backpressure even when operators expand or filter the stream, without
    requiring new stages.
    """

    # Input/output item types at the stage boundary.
    StreamIn: TypeAlias = EngineSample | SampleRecord | SampleBatch
    StreamOut: TypeAlias = SampleRecord | SampleBatch
    # When deterministic=False the result queue carries a plain list[StreamOut].
    # When deterministic=True it carries (seq:int, payload:list[StreamOut]).
    ResultItem: TypeAlias = Union[list[StreamOut], tuple[int, list[StreamOut]]]

    class _InflightCounter:
        """Track how many tasks are currently executing for an operator."""

        def __init__(self) -> None:
            self._count = 0
            self._cv = threading.Condition()

        def increment(self) -> None:
            with self._cv:
                self._count += 1

        def decrement(self) -> None:
            with self._cv:
                self._count -= 1
                if self._count < 0:
                    raise RuntimeError("Inflight counter underflowed")
                if self._count == 0:
                    self._cv.notify_all()

        def is_zero(self) -> bool:
            with self._cv:
                return self._count == 0

    @dataclass
    class _OperatorState:
        """State bundle for one operator inside the fused stage."""

        node: Node
        deterministic: bool
        ctx_proto: dict[str, Any]
        allow_latency_flush: bool = False
        stage_index: int = 0
        stage_name: str = ""
        collect_stats: bool = False
        op_index: int = 0  # position within the stage for metrics/metadata
        instances: list[Op[Any, Any]] = field(init=False, default_factory=list)
        parallelism: int = field(init=False)
        buffer_cfg: Buffering | None = field(init=False)
        buffer: list["ThreadStageRunner.StreamIn"] = field(
            init=False, default_factory=list
        )
        buffer_ts_ns: list[int] = field(
            init=False, default_factory=list
        )  # arrival timestamp per buffered element
        first_ts_ns: Optional[int] = field(init=False, default=None)
        # Deterministic sequencing state (per-operator)
        next_seq: int = field(init=False, default=0)
        emit_seq: int = field(init=False, default=0)
        pending_results: dict[int, list["ThreadStageRunner.StreamOut"]] = field(
            init=False, default_factory=dict
        )
        inflight: "ThreadStageRunner._InflightCounter" = field(init=False)
        input_queue: queue.Queue[Sequence["ThreadStageRunner.StreamIn"] | StopToken] = (
            field(init=False)
        )
        result_queue: queue.Queue["ThreadStageRunner.ResultItem"] = field(init=False)
        _instance_queue: queue.Queue[Op[Any, Any]] = field(init=False, repr=False)

        def __post_init__(self) -> None:
            traits = self.node.op.traits()
            self.parallelism = max(1, self.node.parallelism or traits.parallelism or 1)
            base_ctx = dict(self.ctx_proto)
            base_ctx.setdefault("deterministic", self.deterministic)

            self._instance_queue = queue.Queue()
            for _ in range(self.parallelism):
                instance = copy.deepcopy(self.node.op)
                ctx = OpContext(dict(base_ctx))
                instance.setup(
                    ctx,
                    self.stage_index,
                    self.stage_name,
                    self.op_index,
                    self.collect_stats,
                )
                self.instances.append(instance)
                self._instance_queue.put(instance)

            cfg = self.node.op.buffering()
            if cfg is None:
                self.buffer_cfg = None
            else:
                self.buffer_cfg = Buffering(
                    max_batch=cfg.max_batch,
                    max_latency_ms=(
                        cfg.max_latency_ms
                        if (not self.deterministic or self.allow_latency_flush)
                        else None
                    ),
                )

        def acquire_instance(self) -> Op[Any, Any]:
            return self._instance_queue.get()

        def release_instance(self, instance: Op[Any, Any]) -> None:
            self._instance_queue.put(instance)

        def reset_buffers(self) -> None:
            self.buffer = []
            self.buffer_ts_ns = []
            self.first_ts_ns = None

        def enqueue(
            self, elems: Sequence["ThreadStageRunner.StreamIn"], *, force: bool = False
        ) -> list[tuple[list["ThreadStageRunner.StreamIn"], int]]:
            ready: list[tuple[list["ThreadStageRunner.StreamIn"], int]] = []
            if not elems and not force:
                return ready
            if self.buffer_cfg is None:
                # Preserve the upstream microbatch as-is (no internal batching),
                # so process_many can vectorize when available.
                if elems:
                    batch = elems if isinstance(elems, list) else list(elems)
                    ready.append((batch, 0))
                # there is no internal buffer in the 'None' path, so nothing to flush on 'force'
                return ready

            now_ns = time.perf_counter_ns()
            for elem in elems:
                arrival_ns = time.perf_counter_ns()
                if not self.buffer:
                    self.first_ts_ns = arrival_ns
                self.buffer.append(elem)
                self.buffer_ts_ns.append(arrival_ns)
                if (
                    self.buffer_cfg.max_batch
                    and len(self.buffer) >= self.buffer_cfg.max_batch
                ):
                    ready.append(self._pop_batch(self.buffer_cfg.max_batch, now_ns))
                elif (
                    self.buffer_cfg.max_latency_ms is not None
                    and self.first_ts_ns is not None
                    and (now_ns - self.first_ts_ns) / 1_000_000
                    >= self.buffer_cfg.max_latency_ms
                ):
                    ready.append(self._drain_buffer(now_ns=now_ns))
                now_ns = time.perf_counter_ns()
            if force and self.buffer:
                ready.append(self._drain_buffer())
            return ready

        def _pop_batch(
            self, size: int, now_ns: int
        ) -> tuple[list["ThreadStageRunner.StreamIn"], int]:
            current_ns = now_ns
            chunk = self.buffer[:size]
            timestamps = self.buffer_ts_ns[:size]
            self.buffer = self.buffer[size:]
            self.buffer_ts_ns = self.buffer_ts_ns[size:]
            wait_ns = 0
            if timestamps:
                wait_ns = max(0, current_ns - timestamps[0])
            if not self.buffer:
                self.first_ts_ns = None
            else:
                self.first_ts_ns = self.buffer_ts_ns[0]
            return list(chunk), wait_ns

        def _drain_buffer(
            self, *, now_ns: Optional[int] = None
        ) -> tuple[list["ThreadStageRunner.StreamIn"], int]:
            current_ns = now_ns if now_ns is not None else time.perf_counter_ns()
            return self._pop_batch(len(self.buffer), current_ns)

        def finalize(self) -> list["ThreadStageRunner.StreamOut"]:
            tail: list["ThreadStageRunner.StreamOut"] = []
            for instance in self.instances:
                tail.extend(instance.finalize())
            return tail

        def adjust_parallelism(self, new_level: int) -> None:
            desired = max(1, new_level)
            if desired == self.parallelism:
                return
            if desired > self.parallelism:
                add = desired - self.parallelism
                for _ in range(add):
                    instance = copy.deepcopy(self.node.op)
                    ctx = OpContext(dict(self.ctx_proto))
                    instance.setup(
                        ctx,
                        self.stage_index,
                        self.stage_name,
                        self.op_index,
                        self.collect_stats,
                    )
                    self.instances.append(instance)
                    self._instance_queue.put(instance)
                self.parallelism = desired
                return

            remove = self.parallelism - desired
            removed: list[Op[Any, Any]] = []
            try:
                for _ in range(remove):
                    inst = self._instance_queue.get_nowait()
                    removed.append(inst)
            except queue.Empty as exc:
                for inst in removed:
                    self._instance_queue.put(inst)
                raise RuntimeError(
                    "Cannot shrink parallelism while operators are busy"
                ) from exc
            for inst in removed:
                self.instances.remove(inst)
            self.parallelism = desired

    @dataclass
    class _RunContext:
        stop_token: StopToken
        stage_out_queue: queue.Queue["ThreadStageRunner.StreamOut | StopToken"]
        stop_event: threading.Event
        pumps: list[threading.Thread] = field(default_factory=list)
        feeder: threading.Thread | None = None
        error: BaseException | None = None
        stop_sent: bool = False

    def __init__(
        self,
        stage: Stage,
        ctx_services: dict[str, Any],
        max_workers: int,
        *,
        prefetch_capacity: int = 0,
        queue_capacity: int = 4,
        deterministic: bool = False,
        allow_latency_flush_in_deterministic: bool = True,
        stage_index: int = 0,
        tracking_mode: ExecutionTrackingMode = ExecutionTrackingMode.OFF,
    ) -> None:
        self._stage = stage
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._prefetch_capacity = max(0, prefetch_capacity)
        self._queue_capacity = max(1, queue_capacity)
        self._stage_index = stage_index
        self._tracking_mode = tracking_mode
        self._record_node_metrics: Callable[[NodeMetricsDelta], None] = ctx_services[
            "record_node_metrics"
        ]
        self._metrics_meta: list[tuple[int, str, int, str]] = []
        self._context_lock = threading.Lock()
        self._active_context: Optional["ThreadStageRunner._RunContext"] = None
        self._node_sw = Stopwatch(self._tracking_mode.collects_nodes)

        base_services = dict(ctx_services)
        stage_name = stage.name or f"stage_{stage_index}"

        self.ops: list["ThreadStageRunner._OperatorState"] = []
        collect_op_stats = self._tracking_mode.collects_nodes  # caches result
        for op_index, node in enumerate(stage.nodes):
            state = ThreadStageRunner._OperatorState(
                node=node,
                deterministic=deterministic,
                ctx_proto=base_services,
                allow_latency_flush=allow_latency_flush_in_deterministic,
                stage_index=self._stage_index,
                stage_name=stage_name,
                op_index=op_index,
                collect_stats=collect_op_stats,
            )
            state.inflight = ThreadStageRunner._InflightCounter()
            # Small bounded input queue per operator.  It allows downstream ops to
            # apply backpressure without creating new stages and keeps memory usage
            # predictable even when operators change cardinality.
            state.input_queue = queue.Queue(maxsize=self._queue_capacity)
            # Worker callbacks push results into this queue.  We allow a few
            # micro-batches per worker instance before blocking so throughput stays
            # high but the queue can never grow without bound.
            result_capacity = max(1, self._queue_capacity * state.parallelism)
            state.result_queue = queue.Queue(maxsize=result_capacity)
            self.ops.append(state)
            self._metrics_meta.append(
                (self._stage_index, stage_name, state.op_index, state.node.name)
            )

    def _create_context(self) -> "ThreadStageRunner._RunContext":
        out_capacity = max(1, self._prefetch_capacity or self._queue_capacity)
        return ThreadStageRunner._RunContext(
            stop_token=StopToken(),
            stage_out_queue=queue.Queue(maxsize=out_capacity),
            stop_event=threading.Event(),
        )

    def _put_result(
        self,
        state: "ThreadStageRunner._OperatorState",
        result: list["ThreadStageRunner.StreamOut"],
        context: "ThreadStageRunner._RunContext",
        *,
        seq: Optional[int] = None,
    ) -> None:
        """Enqueue the operator output, regardless of how many elements it contains.

        ``result`` can be empty (filter) or contain more elements than the input
        micro-batch (fan-out).  We push the whole list as one unit; downstream pumps
        iterate element-by-element when emitting to their consumers.
        """
        # In deterministic mode we MUST enqueue even empty results so the
        # reordering gate can advance emit_seq. In non-deterministic mode
        # it's fine to drop empties.
        payload: ThreadStageRunner.ResultItem
        if state.deterministic:
            assert seq is not None, "Deterministic mode requires sequence numbers"
            payload = (seq, result)
        else:
            if not result:
                return

            payload = result

        while True:
            try:
                state.result_queue.put(payload, timeout=0.1)
                return
            except queue.Full:
                if context.stop_event.is_set():
                    # If shutdown has been requested we still retry until a consumer
                    # drains space; dropping here would lose data emitted before the
                    # stop signal propagated.
                    continue

    def _operator_loop(
        self, idx: int, context: "ThreadStageRunner._RunContext"
    ) -> None:
        state = self.ops[idx]
        next_queue: (
            queue.Queue[Sequence["ThreadStageRunner.StreamIn"] | StopToken] | None
        ) = self.ops[idx + 1].input_queue if idx + 1 < len(self.ops) else None

        state.reset_buffers()
        upstream_closed = False

        while True:
            self._drain_results(state, next_queue, context)

            if upstream_closed:
                # Once upstream is closed we keep draining until no tasks
                # remain. Only then do we forward finalize() output and
                # the stop token downstream.
                if (
                    state.inflight.is_zero()
                    and state.result_queue.empty()
                    and not state.buffer
                ):
                    tail = state.finalize()
                    if tail:
                        self._emit_downstream(tail, next_queue, context)
                    self._signal_downstream_stop(next_queue, context)
                    return
                try:
                    item = state.result_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if state.deterministic:
                    rseq, payload = cast(
                        tuple[int, list[ThreadStageRunner.StreamOut]], item
                    )
                    state.pending_results[int(rseq)] = payload
                    # Emit in order as far as possible
                    while state.emit_seq in state.pending_results:
                        ready_elems = state.pending_results.pop(state.emit_seq)
                        state.emit_seq += 1
                        self._emit_downstream(ready_elems, next_queue, context)
                else:
                    assert isinstance(item, list)
                    self._emit_downstream(item, next_queue, context)
                continue

            if context.stop_event.is_set():
                # Upstream is tearing down (either gracefully or because an error
                # was recorded).  Flush whatever we buffered so downstream stages
                # observe every element before we forward the stop signal.
                upstream_closed = True
                ready = state.enqueue([], force=True)
                for batch, wait_ns in ready:
                    self._schedule_batch(context, state, batch, wait_ns=wait_ns)
                continue

            try:
                item = state.input_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if isinstance(item, _Stop):
                upstream_closed = True
                ready = state.enqueue([], force=True)
            else:
                ready = state.enqueue(item, force=False)

            if ready:
                # Schedule at most one wave per inner loop; then drain.
                # We schedule 1 batch for every op instance (state.parallelism) before draining again.
                burst = max(1, state.parallelism)
                for i, (batch, wait_ns) in enumerate(ready):
                    self._schedule_batch(context, state, batch, wait_ns=wait_ns)
                    # Let done callbacks blocked on result_queue.put() make progress.
                    # Also keeps next operator fed so its input_queue doesn’t starve.
                    if (i + 1) % burst == 0:
                        self._drain_results(state, next_queue, context)

    def _drain_results(
        self,
        state: "ThreadStageRunner._OperatorState",
        next_queue: queue.Queue[Sequence["ThreadStageRunner.StreamIn"] | StopToken]
        | None,
        context: "ThreadStageRunner._RunContext",
    ) -> None:
        while True:
            try:
                item = state.result_queue.get_nowait()
            except queue.Empty:
                break
            if state.deterministic:
                rseq, payload = cast(
                    tuple[int, list[ThreadStageRunner.StreamOut]], item
                )
                state.pending_results[int(rseq)] = payload
                while state.emit_seq in state.pending_results:
                    ready_elems = state.pending_results.pop(state.emit_seq)
                    state.emit_seq += 1
                    self._emit_downstream(ready_elems, next_queue, context)
            else:
                self._emit_downstream(
                    cast(list[ThreadStageRunner.StreamOut], item), next_queue, context
                )

    def _emit_downstream(
        self,
        elements: list["ThreadStageRunner.StreamOut"],
        next_queue: queue.Queue[Sequence["ThreadStageRunner.StreamIn"] | StopToken]
        | None,
        context: "ThreadStageRunner._RunContext",
    ) -> None:
        if not elements:
            return
        # Preserve order: results are forwarded element-by-element so downstream
        # operators observe the same ordering they would have seen in a single-threaded
        # execution.

        # Between operators: ship the whole micro-batch as one queue item to amortize costs.
        # At the final boundary (next_queue is None), keep yielding element-by-element
        # so external consumers see the same stream of records/batches as before.
        if next_queue is not None:
            self._put_into_queue(next_queue, elements, context)
            return

        for elem in elements:
            self._put_into_queue(context.stage_out_queue, elem, context)

    def _schedule_batch(
        self,
        context: "ThreadStageRunner._RunContext",
        state: "ThreadStageRunner._OperatorState",
        batch: list["ThreadStageRunner.StreamIn"],
        *,
        wait_ns: int = 0,
    ) -> None:
        if not batch or context.stop_event.is_set():
            return
        instance = state.acquire_instance()

        seq: int | None = None
        if state.deterministic:
            seq = state.next_seq
            state.next_seq += 1

        consumed_elements = -1
        consumed_bytes = -1
        queue_depth_snapshot = -1
        metrics_meta = None
        collect_stats = self._tracking_mode.collects_nodes
        if collect_stats:
            consumed_elements = len(batch)
            consumed_bytes = estimate_bytes(batch)
            metrics_meta = self._metrics_meta[state.op_index]
            try:
                queue_depth_snapshot = state.input_queue.qsize()
            except NotImplementedError:
                queue_depth_snapshot = -1

        def work(
            items: list["ThreadStageRunner.StreamIn"],
        ) -> tuple[list["ThreadStageRunner.StreamOut"], int]:
            start_ns = self._node_sw.start()
            try:
                outputs = instance.process_many(items)
            except (NotImplementedError, AttributeError):
                out: list["ThreadStageRunner.StreamOut"] = []
                for element in items:
                    out.extend(instance.process_one(element))
                outputs = out
            return outputs, self._node_sw.elapsed(start_ns)

        state.inflight.increment()
        future = self._executor.submit(work, batch)

        def done_callback(
            fut: Future[tuple[list["ThreadStageRunner.StreamOut"], int]],
        ) -> None:
            """Executor completion hook for a single micro-batch.

            Operators are allowed to change cardinality: they may drop inputs
            (return ``[]``) or fan out (return more elements than they received).
            Whatever comes back is treated as the next payload and forwarded
            through the per-operator queues.
            """
            try:
                result, proc_ns = fut.result()
            except Exception as exc:  # noqa: BLE001
                self._record_error(context, exc)
                result = None
                proc_ns = 0
            finally:
                state.release_instance(instance)
                state.inflight.decrement()

            if result is None:
                return

            if collect_stats:
                stage_index, stage_name, op_index, op_name = metrics_meta  # pyright: ignore[reportGeneralTypeIssues] no assert none on hot path
                produced_elements = len(result)
                produced_bytes = estimate_bytes(result)
                delta = NodeMetricsDelta(
                    stage_index=stage_index,
                    op_index=op_index,
                    stage_name=stage_name,
                    name=op_name,
                    processed_ns=proc_ns,
                    produced_elements=produced_elements,
                    consumed_elements=consumed_elements,
                    produced_bytes=produced_bytes,
                    consumed_bytes=consumed_bytes,
                    wait_ns=wait_ns,
                    max_queue_depth=queue_depth_snapshot,
                    min_processing_ns=proc_ns,
                    max_processing_ns=proc_ns,
                )
                self._record_node_metrics(delta)

            self._put_result(state, result, context, seq=seq)

        future.add_done_callback(done_callback)

    def _signal_downstream_stop(
        self,
        next_queue: queue.Queue[Sequence["ThreadStageRunner.StreamIn"] | StopToken]
        | None,
        context: "ThreadStageRunner._RunContext",
    ) -> None:
        if next_queue is None:
            self._put_stage_stop(context)
        else:
            self._put_into_queue(next_queue, context.stop_token, context)

    def _put_into_queue(
        self,
        q: "queue.Queue[_QItem]",
        item: "_QItem",
        context: "ThreadStageRunner._RunContext",
    ) -> None:
        while True:
            try:
                q.put(item, timeout=0.1)
                return
            except queue.Full:
                if context.stop_event.is_set():
                    # During shutdown, avoid blocking producers on a full queue.
                    # The consumer has stopped reading (iterator closed) and set
                    # the stop_event, so further blocking puts would deadlock.
                    try:
                        q.put_nowait(item)
                    except queue.Full:
                        # Drop the item on shutdown to allow threads to exit cleanly.
                        return
                continue

    def _put_stage_stop(self, context: "ThreadStageRunner._RunContext") -> None:
        if context.stop_sent:
            return
        context.stop_sent = True
        self._put_into_queue(context.stage_out_queue, context.stop_token, context)

    def _record_error(
        self,
        context: "ThreadStageRunner._RunContext",
        exc: BaseException,
    ) -> None:
        if context.error is None:
            context.error = exc
            context.stop_event.set()
            self._put_stage_stop(context)

    def _start_feeder(
        self,
        upstream: Iterable["ThreadStageRunner.StreamIn"],
        context: "ThreadStageRunner._RunContext",
    ) -> None:
        if not self.ops:

            def passthrough() -> None:
                try:
                    for elem in upstream:
                        self._put_into_queue(
                            context.stage_out_queue,
                            cast(ThreadStageRunner.StreamOut, elem),
                            context,
                        )
                except BaseException as exc:  # noqa: BLE001
                    self._record_error(context, exc)
                finally:
                    self._put_stage_stop(context)

            feeder = threading.Thread(target=passthrough, daemon=True)
            feeder.start()
            context.feeder = feeder
            return

        first_state = self.ops[0]

        def feed() -> None:
            try:
                for elem in upstream:
                    if context.stop_event.is_set():
                        break
                    self._put_into_queue(first_state.input_queue, [elem], context)
            except BaseException as exc:  # noqa: BLE001
                self._record_error(context, exc)
            finally:
                self._put_into_queue(
                    first_state.input_queue, context.stop_token, context
                )

        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()
        context.feeder = feeder

    def _start_operator_threads(self, context: "ThreadStageRunner._RunContext") -> None:
        if not self.ops:
            return
        for idx in range(len(self.ops)):
            thread = threading.Thread(
                target=self._operator_loop,
                args=(idx, context),
                daemon=True,
                name=f"StageRunner-op{idx}",
            )
            thread.start()
            context.pumps.append(thread)

    def _join_threads(self, context: "ThreadStageRunner._RunContext") -> None:
        for thread in context.pumps:
            thread.join(timeout=1.0)
        if context.feeder is not None:
            context.feeder.join(timeout=1.0)

    def run(
        self, upstream: Iterable["ThreadStageRunner.StreamIn"]
    ) -> Iterator["ThreadStageRunner.StreamOut"]:
        context = self._create_context()
        with self._context_lock:
            if self._active_context is not None:
                raise RuntimeError("Stage runner already in use")
            self._active_context = context

        try:
            self._start_operator_threads(context)
            self._start_feeder(upstream, context)

            def iterator() -> Iterator["ThreadStageRunner.StreamOut"]:
                try:
                    while True:
                        try:
                            item = context.stage_out_queue.get(timeout=0.1)
                        except queue.Empty:
                            if context.stop_event.is_set():
                                break
                            continue
                        if isinstance(item, _Stop):
                            break
                        yield item
                    if context.error is not None:
                        raise context.error
                finally:
                    context.stop_event.set()
                    self._join_threads(context)

            stream = iterator()
            if self._prefetch_capacity > 0:

                def _on_stop() -> None:
                    context.stop_event.set()
                    self._put_stage_stop(context)

                stream = buffered_iterable(
                    stream,
                    self._prefetch_capacity,
                    on_stop=_on_stop,
                )
            yield from stream
        finally:
            with self._context_lock:
                self._active_context = None

    def run_one(
        self, elem: "ThreadStageRunner.StreamIn"
    ) -> "ThreadStageRunner.StreamOut":
        if not self.ops:
            # passthrough stage: elem must already be a StreamOut
            return cast("ThreadStageRunner.StreamOut", elem)
        value: Any = elem
        for state in self.ops:
            instance = state.instances[0]
            outputs = instance.process_one(value)
            if len(outputs) != 1:
                raise RuntimeError("Indexable path requires 1->1 ops through the stage")
            value = outputs[0]
        for state in self.ops:
            tail = state.instances[0].finalize()
            if tail:
                raise RuntimeError("Finalize emitted data in run_one path")
        return cast("ThreadStageRunner.StreamOut", value)

    def set_parallelism(self, op_index: int, new_parallelism: int) -> None:
        if op_index < 0 or op_index >= len(self.ops):
            raise IndexError("op_index out of range")
        target = max(1, min(new_parallelism, self._max_workers))
        self.ops[op_index].adjust_parallelism(target)

    def close(self) -> None:
        with self._context_lock:
            ctx = self._active_context

        if ctx is not None:
            # tell everyone to stop
            ctx.stop_event.set()
            self._put_stage_stop(ctx)
            self._join_threads(ctx)

        self._executor.shutdown(wait=True)
