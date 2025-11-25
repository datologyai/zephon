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

The deterministic path works by construction:

* The pump thread is the only producer of micro-batches. Right before it hands
  work to the thread pool it tags the batch with ``next_seq`` and increments it
  (see :meth:`_schedule_batch`).
* Worker callbacks always enqueue ``(seq, payload)`` pairs, even when the
  payload is empty because the operator buffered or filtered everything out.
  This guarantees the reordering gate never observes gaps.
* ``_drain_results`` advances ``emit_seq`` only when the next integer is
  present, so out-of-order completions merely accumulate in
  ``pending_results`` until their predecessors arrive.
* Within a micro-batch the runner preserves element order, so fan-out operators
  emit children in the same relative order a single-threaded execution would.

This scheme handles filtering, fan-out, and operator-local buffering as long as
operators are deterministic per invocation **and** any cross-invocation state is
confined to exactly one operator instance (i.e., the operator runs with
parallelism 1 or otherwise partitions its state explicitly). If the planner or
user scales such an operator to multiple instances, each instance would own a
disjoint buffer, leading to nondeterministic flush boundaries. Note the
difference between *runner-level* buffering (via ``Buffering`` traits) and
*operator-internal* buffering:

- Runner buffering happens on the pump thread before seq assignment, so the
  resulting micro-batches are deterministic. ``max_latency_ms`` may change the
  group size between runs, but ordering is unaffected. Stages with
  ``batch_shape_sensitive`` operators disable latency flushes in deterministic
  mode to avoid grouping-dependent behaviour.
- Operator buffering happens inside the worker instance that processed a given
  seq. When the operator finally flushes (e.g., ``Batch.finalize``), its output
  rides on the seq of that invocation, so downstream order remains the same as
  the single-threaded baseline.

Example (Batch):

``Batch`` keeps a per-lane buffer inside each operator instance. Because the
planner defaults its parallelism to 1, the stage creates a single instance,
meaning there is one authoritative buffer per lane. Inputs flow through the
runner in deterministic order, seq numbers enforce downstream ordering, and
every flush produces the same batches regardless of thread timing. If a user
overrides ``parallelism`` to a value > 1, the buffers split across instances
and batching becomes nondeterministic, which is why ``Batch`` is meant to run
single-threaded (or to rely on runner-level buffering instead).

In summary, an operator is compatible with the seq-based determinism if it:

1. Produces deterministic outputs for a deterministic input micro-batch.
2. Keeps state confined to a single operator instance (or explicitly runs at
   parallelism 1 when cross-element state is needed).
3. Derives child lineage deterministically when fan-out occurs.
4. Does not rely on wall-clock ordering between invocations.

Under those constraints, the sequence numbers guarantee the threaded execution
observes the exact same logical stream as a single-threaded run.
"""

import copy
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import (
    Any,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

from zephon.core.constants import (
    Microbatch,
    RunnerStageIn,
    RunnerStageOut,
    RunnerStreamIn,
    SampleBatch,
    SampleRecord,
    StreamItem,
)
from zephon.core.graph import Node, Stage
from zephon.core.op_base import Op, OpContext
from zephon.observability.config import ExecutionTrackingMode
from zephon.observability.size_estimator import estimate_bytes
from zephon.observability.stats import NodeMetricsDelta
from zephon.runners.base import BaseOperatorState, StageRunnerBase
from zephon.utils import buffered_iterable

_QItem = TypeVar("_QItem")


class _Stop:
    pass


StopToken = _Stop

# When deterministic=False the result queue carries a plain micro-batch payload.
# When deterministic=True it carries (seq:int, payload=Microbatch).
ResultItem: TypeAlias = Union[Microbatch, tuple[int, Microbatch]]


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
class _ThreadOperatorState(BaseOperatorState):
    queue_capacity: int = 1
    next_seq: int = field(init=False, default=0)
    emit_seq: int = field(init=False, default=0)
    pending_results: dict[int, Microbatch] = field(init=False, default_factory=dict)
    inflight: _InflightCounter = field(init=False)
    input_queue: queue.Queue[Sequence[RunnerStreamIn] | StopToken] = field(init=False)
    result_queue: queue.Queue[ResultItem] = field(init=False)
    _instance_queue: queue.Queue[Op[RunnerStreamIn, StreamItem]] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self._instance_queue = queue.Queue()
        for instance in self.instances:
            self._instance_queue.put(instance)

        self.inflight = _InflightCounter()
        self.input_queue = queue.Queue(maxsize=self.queue_capacity)
        result_capacity = max(1, self.queue_capacity * self.parallelism)
        self.result_queue = queue.Queue(maxsize=result_capacity)

    def acquire_instance(self) -> Op[RunnerStreamIn, StreamItem]:
        return self._instance_queue.get()

    def release_instance(self, instance: Op[RunnerStreamIn, StreamItem]) -> None:
        self._instance_queue.put(instance)

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
        removed: list[Op[RunnerStreamIn, StreamItem]] = []
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


class ThreadStageRunner(StageRunnerBase[_ThreadOperatorState]):
    """Execute a stage locally with bounded queues between operators.

    The stage exposes a pull-driven iterator (`run`) while orchestrating push-
    based execution across the fused operators.  Each operator owns a small input
    queue, optional buffering rules, and a pool of worker instances.  When a
    micro-batch is ready it is pushed to the worker pool; results are pushed into
    the next operator's queue (or the final stage output queue).  Bounded queues
    provide backpressure even when operators expand or filter the stream, without
    requiring new stages.
    """

    _OperatorState = _ThreadOperatorState

    @dataclass
    class _RunContext:
        stop_token: StopToken
        stage_out_queue: queue.Queue[RunnerStageOut | StopToken]
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
        stage_output_mode: Literal["microbatches", "stream_items"] = "microbatches",
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._queue_capacity = max(1, queue_capacity)
        self._context_lock = threading.Lock()
        self._active_context: Optional["ThreadStageRunner._RunContext"] = None
        super().__init__(
            stage,
            ctx_services,
            max_workers,
            prefetch_capacity=prefetch_capacity,
            deterministic=deterministic,
            allow_latency_flush_in_deterministic=allow_latency_flush_in_deterministic,
            stage_index=stage_index,
            tracking_mode=tracking_mode,
            stage_output_mode=stage_output_mode,
        )

    def _make_operator_state(
        self,
        *,
        node: Node,
        op_index: int,
        deterministic: bool,
        ctx_proto: dict[str, Any],
        allow_latency_flush: bool,
        stage_index: int,
        stage_name: str,
        collect_op_stats: bool,
    ) -> _ThreadOperatorState:
        return _ThreadOperatorState(
            node=node,
            deterministic=deterministic,
            ctx_proto=ctx_proto,
            allow_latency_flush=allow_latency_flush,
            stage_index=stage_index,
            stage_name=stage_name,
            op_index=op_index,
            collect_stats=collect_op_stats,
            queue_capacity=self._queue_capacity,
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
        state: _ThreadOperatorState,
        result: Microbatch,
        context: "ThreadStageRunner._RunContext",
        *,
        seq: Optional[int] = None,
    ) -> None:
        """Enqueue the operator output micro-batch as a single queue payload.

        ``result`` may be empty (filter) or contain more elements than the input batch
        (fan-out). Downstream queues always treat it atomically.
        """
        # In deterministic mode we MUST enqueue even empty results so the
        # reordering gate can advance emit_seq. In non-deterministic mode
        # it's fine to drop empties.
        payload: ResultItem
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
        next_queue: queue.Queue[Sequence[RunnerStreamIn] | StopToken] | None = (
            self.ops[idx + 1].input_queue if idx + 1 < len(self.ops) else None
        )

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
                    rseq, payload = cast(tuple[int, Microbatch], item)
                    state.pending_results[int(rseq)] = payload
                    # Emit in order as far as possible
                    while state.emit_seq in state.pending_results:
                        ready_elems = state.pending_results.pop(state.emit_seq)
                        state.emit_seq += 1
                        self._emit_downstream(ready_elems, next_queue, context)
                else:
                    self._emit_downstream(cast(Microbatch, item), next_queue, context)
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
        state: _ThreadOperatorState,
        next_queue: queue.Queue[Sequence[RunnerStreamIn] | StopToken] | None,
        context: "ThreadStageRunner._RunContext",
    ) -> None:
        while True:
            try:
                item = state.result_queue.get_nowait()
            except queue.Empty:
                break
            if state.deterministic:
                rseq, payload = cast(tuple[int, Microbatch], item)
                state.pending_results[int(rseq)] = payload
                while state.emit_seq in state.pending_results:
                    ready_elems = state.pending_results.pop(state.emit_seq)
                    state.emit_seq += 1
                    self._emit_downstream(ready_elems, next_queue, context)
            else:
                self._emit_downstream(cast(Microbatch, item), next_queue, context)

    def _emit_downstream(
        self,
        elements: Microbatch,
        next_queue: queue.Queue[Sequence[RunnerStreamIn] | StopToken] | None,
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

        self._emit_stage_output(elements, context)

    def _emit_stage_output(
        self,
        elements: Microbatch,
        context: "ThreadStageRunner._RunContext",
    ) -> None:
        if not elements:
            return
        if self._emit_microbatches:
            self._put_into_queue(context.stage_out_queue, elements, context)
            return

        for elem in elements:
            self._put_into_queue(context.stage_out_queue, elem, context)

    def _schedule_batch(
        self,
        context: "ThreadStageRunner._RunContext",
        state: _ThreadOperatorState,
        batch: list[RunnerStreamIn],
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
            items: list[RunnerStreamIn],
        ) -> tuple[Microbatch, int]:
            start_ns = self._node_sw.start()
            try:
                outputs: Microbatch = instance.process_many(items)
            except (NotImplementedError, AttributeError):
                out: Microbatch = []
                for element in items:
                    out.extend(instance.process_one(element))
                outputs = out
            return outputs, self._node_sw.elapsed(start_ns)

        state.inflight.increment()
        future = self._executor.submit(work, batch)

        def done_callback(
            fut: Future[tuple[Microbatch, int]],
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
        next_queue: queue.Queue[Sequence[RunnerStreamIn] | StopToken] | None,
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
        upstream: Iterable[RunnerStageIn],
        context: "ThreadStageRunner._RunContext",
    ) -> None:
        # This path (not self.ops) probably is rarely/never called. Might consider removing.
        if not self.ops:

            def passthrough() -> None:
                try:
                    for elem in upstream:
                        if context.stop_event.is_set():
                            break
                        batch: Microbatch
                        if isinstance(elem, list):
                            batch = elem
                            for item in batch:
                                if not isinstance(item, (SampleRecord, SampleBatch)):  # pyright: ignore[reportUnnecessaryIsInstance]
                                    raise TypeError(
                                        "Passthrough stage received unsupported element "
                                        + f"{type(item)!r} inside microbatch"
                                    )
                        elif isinstance(elem, (SampleRecord, SampleBatch)):
                            batch = [elem]
                        else:
                            raise TypeError(
                                "Passthrough stage received unsupported element "
                                + f"{type(elem)!r}"
                            )
                        self._emit_stage_output(batch, context)
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
                    batch: Sequence[RunnerStreamIn]
                    if isinstance(elem, list):
                        batch = elem
                    else:
                        batch = [elem]
                    self._put_into_queue(first_state.input_queue, batch, context)
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

    def run(self, upstream: Iterable[RunnerStageIn]) -> Iterator[RunnerStageOut]:
        context = self._create_context()
        with self._context_lock:
            if self._active_context is not None:
                raise RuntimeError("Stage runner already in use")
            self._active_context = context

        try:
            self._start_operator_threads(context)
            self._start_feeder(upstream, context)

            def iterator() -> Iterator[RunnerStageOut]:
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
