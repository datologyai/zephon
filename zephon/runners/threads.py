# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0
"""Thread-based stage runner for executing ops within the current process."""

import copy
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from zephon.core.constants import (
    RunnerStreamIn,
    StreamItem,
)
from zephon.core.graph import Node, Stage
from zephon.core.notify import is_sentinel
from zephon.core.op_base import Op, OpContext
from zephon.observability.config import ExecutionTrackingMode
from zephon.observability.size_estimator import estimate_bytes
from zephon.runners.concurrent import (
    ConcurrentRunContext,
    RunnerResult,
    StopToken,
    _QueueLike,
)
from zephon.runners.queue_drain import (
    QueueDrainOperatorState,
    QueueDrainStageRunner,
)


@dataclass
class _ThreadOperatorState(QueueDrainOperatorState):
    _instance_queue: queue.Queue[Op[RunnerStreamIn, StreamItem]] = field(
        init=False, repr=False
    )
    input_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] = field(init=False)
    result_queue: _QueueLike[RunnerResult] = field(init=False)

    # Thread identity of the pump thread that owns this operator state.
    # Set at the start of ``_operator_loop`` via ``_on_pump_started``.
    pump_thread_id: int = field(init=False, default=0)

    # Results produced on the pump thread that must bypass ``result_queue``.
    #
    # The pump thread is the *only* thread that drains ``result_queue``
    # (via ``_drain_results``).  If any code running on the pump thread
    # calls ``_put_result`` while the queue is full, the pump blocks on
    # its own queue and deadlocks: no other thread will ever make space.
    #
    # Two code paths hit this problem and stash results here instead:
    #
    # 1. **Sentinel batches** — ``_schedule_batch`` creates a RunnerResult
    #    for sentinel control signals directly on the pump thread.
    #    Sentinels bypass ``process_many`` entirely (they are never sent
    #    to the worker pool), so the result must be produced inline.
    #
    # 2. **Synchronous done_callbacks** — when ``future.add_done_callback``
    #    is called on an already-completed future, Python invokes the
    #    callback synchronously on the calling thread — which is the pump
    #    thread.  To break the cycle the callback detects that it is
    #    running on the pump thread (via ``pump_thread_id``) and appends
    #    the result here instead of blocking on the queue.
    #
    # ``_post_schedule_batch`` drains this list via ``_handle_result``
    # after every ``_schedule_batch`` call.  Control flows back to
    # ``_operator_loop``, and the pump handles stashed results inline —
    # draining queues, forwarding downstream, etc. — without ever
    # blocking on a queue it is responsible for draining.
    _local_results: list[RunnerResult] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._instance_queue = queue.Queue()
        for instance in self.instances:
            self._instance_queue.put(instance)

        self.input_queue = queue.Queue[Sequence[RunnerStreamIn] | StopToken](
            maxsize=self.queue_capacity
        )
        result_capacity = max(1, self.queue_capacity * self.parallelism)
        self.result_queue = queue.Queue[RunnerResult](maxsize=result_capacity)

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


class ThreadStageRunner(QueueDrainStageRunner[_ThreadOperatorState]):
    """Execute a stage locally using threads and bounded in-memory queues.

    This runner is a concrete :class:`ConcurrentStageRunner` that uses the
    standard library's :class:`concurrent.futures.ThreadPoolExecutor` and
    :class:`queue.Queue` to execute operators in the current process.  All of
    the high-level coordination – per-operator pump threads, buffering, result
    reordering, stop propagation, and the pull-driven :meth:`run` iterator –
    is implemented in :class:`ConcurrentStageRunner`.  This subclass supplies
    the thread-based worker backend and the operator instance management.

    Execution model
    ---------------

    For a given stage, :class:`ConcurrentStageRunner` creates one pump thread
    per operator.  This subclass configures the concrete state that those pump
    threads work with:

    * Each operator owns:
      - a bounded ``input_queue`` backed by :class:`queue.Queue` that carries
        micro-batches (``Sequence[RunnerStreamIn]``), and
      - a bounded ``result_queue`` backed by :class:`queue.Queue` that carries
        :class:`RunnerResult` objects produced by worker threads.

    * :class:`ConcurrentStageRunner` pump threads:
      - pull input micro-batches from ``input_queue``,
      - apply buffering rules from :class:`BaseOperatorState` to decide when
        to form a micro-batch,
      - call :meth:`_schedule_batch` on this subclass to hand the batch to the
        worker backend, and
      - drain ``result_queue`` and forward ready results downstream using the
        shared deterministic/non-deterministic logic in the base class.

    * This subclass maintains a per-operator ``_instance_queue`` with concrete
      :class:`Op` instances.  When a batch is scheduled, a worker thread:
      - acquires an instance from ``_instance_queue``,
      - runs ``process_many`` (or ``process_one``) on that instance,
      - returns the instance to the pool, and
      - enqueues a :class:`RunnerResult` into the operator's ``result_queue``.

      This allows operators to keep local state inside an instance while still
      enabling parallel execution up to the configured ``parallelism``.

    Determinism and buffering
    -------------------------

    The semantics of deterministic vs non-deterministic execution, the use of
    the ``seq`` field on :class:`RunnerResult`, and the interaction with
    operator-local buffering (including the batch-style example) are defined
    by :class:`ConcurrentStageRunner`.  See the base class docstring for a
    detailed discussion.

    Backpressure and resource limits
    --------------------------------

    The ``queue_capacity`` parameter controls how many in-flight micro-batches
    each operator can buffer between upstream and downstream.  Because the
    queues are bounded:

    * The per-operator input queue applies backpressure to upstream producers,
      preventing unbounded memory growth when operators are slow or fan out.
    * The per-operator result queue applies backpressure to worker threads:
      when it fills, completed tasks block in :meth:`_put_result` until pump
      threads make progress or the stage begins shutting down.

    The total number of concurrent worker tasks is bounded by ``max_workers``
    in the underlying :class:`ThreadPoolExecutor`.  The
    :meth:`set_parallelism` method allows adjusting the parallelism of a
    specific operator at runtime (within that per-stage bound) by cloning or
    retiring operator instances.

    Output mode and prefetching
    ---------------------------

    The ``stage_output_mode`` parameter controls whether the iterator yielded
    by :meth:`run` returns micro-batches or individual stream items.  In
    micro-batch mode, downstream consumers see the same grouping that the
    runner uses internally; in stream-item mode, elements are flattened before
    being written to the stage output queue.

    When ``prefetch_capacity`` is greater than zero, the base class wraps the
    output iterator with :func:`zephon.utils.buffered_iterable`, allowing the
    consumer to pull ahead while worker threads continue to process the next
    micro-batches in the background.

    Because all work happens in the local process and uses plain Python
    threads and queues, :class:`ThreadStageRunner` is a good default for
    single-node executions and for operators that are not CPU- or GIL-bound.
    Process-based or distributed runners can reuse the same control flow by
    subclassing :class:`ConcurrentStageRunner` with a different worker
    backend.
    """

    _OperatorState = _ThreadOperatorState

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

    def _on_pump_started(self, state: _ThreadOperatorState) -> None:
        state.pump_thread_id = threading.get_ident()

    def _post_schedule_batch(
        self,
        state: _ThreadOperatorState,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
    ) -> None:
        for result in state._local_results:
            self._handle_result(state, result, next_queue, context)
        state._local_results.clear()

    def _put_result(
        self,
        state: _ThreadOperatorState,
        result: RunnerResult,
        context: ConcurrentRunContext,
    ) -> None:
        """Enqueue a completed micro-batch into the operator's result queue.

        The thread-based runner always forwards a ``RunnerResult`` object, even when
        the payload is empty.  Deterministic stages rely on every scheduled batch
        producing a result with a monotonically increasing ``seq`` so that the
        base class can re-establish single-threaded ordering.  Non-deterministic
        stages ignore ``seq`` but still use the same result queue to propagate
        outputs and metrics.

        Backpressure is applied by blocking when the result queue is full.  Results
        are eventually drained by the operator pump thread, which invokes the
        generic reordering and downstream forwarding logic in
        :class:`ConcurrentStageRunner`.
        """
        while True:
            try:
                state.result_queue.put(result, timeout=0.1)
                return
            except queue.Full:
                if context.stop_event.is_set():
                    continue

    def _schedule_batch(
        self,
        state: _ThreadOperatorState,
        batch: list[RunnerStreamIn],
        *,
        wait_ns: int,
        context: ConcurrentRunContext,
    ) -> None:
        if not batch or context.stop_event.is_set():
            return

        # Sentinel batches (tombstones, flush signals) bypass process_many
        # entirely — they are control signals that operators should never see.
        # Create a RunnerResult inline and stash it in _local_results so the
        # pump thread never blocks on result_queue (see _local_results docstring).
        if is_sentinel(batch[0]):
            seq = state.next_seq
            state.next_seq += 1
            result = RunnerResult(
                seq=seq,
                payload=batch,
                wait_ns=0,
                consumed_elements=0,
                consumed_bytes=0,
                queue_depth_snapshot=-1,
                proc_ns=0,
                collect_metrics=False,
                from_worker=False,
            )
            state._local_results.append(result)
            return

        instance = state.acquire_instance()

        seq = state.next_seq
        state.next_seq += 1

        collect_stats = self._tracking_mode.collects_nodes
        consumed_elements = len(batch) if collect_stats else 0
        consumed_bytes = estimate_bytes(batch) if collect_stats else 0
        queue_depth_snapshot = -1
        if collect_stats:
            try:
                queue_depth_snapshot = state.input_queue.qsize()
            except NotImplementedError:
                queue_depth_snapshot = -1

        def work(
            items: list[RunnerStreamIn],
        ) -> tuple[list[StreamItem], int]:
            start_ns = self._node_sw.start()
            try:
                outputs = instance.process_many(items)
            except (NotImplementedError, AttributeError):
                outputs = None
            if outputs is None:
                out: list[StreamItem] = []
                for element in items:
                    out.extend(instance.process_one(element))
                outputs = out
            return outputs, self._node_sw.elapsed(start_ns)

        state.inflight.increment()
        future = self._executor.submit(work, batch)

        def done_callback(
            fut: Future[tuple[list[StreamItem], int]],
        ) -> None:
            try:
                result, proc_ns = fut.result()
            except Exception as exc:  # noqa: BLE001
                self._record_error(context, exc)
                result = None
                proc_ns = 0

            # Release worker capacity early to reduce idle time during backpressure.
            # Increment pending_puts BEFORE decrementing inflight to prevent race where
            # pump sees both counters at zero before put completes.
            state.release_instance(instance)
            should_put = result is not None and (result or state.deterministic)
            if should_put:
                state.pending_puts.increment()
            state.inflight.decrement()

            # Enqueue result (tracked by pending_puts for termination safety)
            if result is not None:
                collect_stats = self._tracking_mode.collects_nodes
                payload = RunnerResult(
                    seq=seq,
                    payload=result,
                    wait_ns=wait_ns if collect_stats else 0,
                    consumed_elements=consumed_elements,
                    consumed_bytes=consumed_bytes,
                    queue_depth_snapshot=queue_depth_snapshot,
                    proc_ns=proc_ns,
                    collect_metrics=collect_stats,
                )

                if should_put:
                    if threading.get_ident() == state.pump_thread_id:
                        # Synchronous callback on the pump thread — stash
                        # instead of blocking (see _local_results docstring).
                        state._local_results.append(payload)
                        state.pending_puts.decrement()
                    else:
                        try:
                            self._put_result(state, payload, context)
                        finally:
                            state.pending_puts.decrement()
                elif collect_stats:
                    # Non-deterministic + empty payload: record metrics only
                    self._record_result_metrics(state, payload)

        future.add_done_callback(done_callback)

    def set_parallelism(self, op_index: int, new_parallelism: int) -> None:
        if op_index < 0 or op_index >= len(self.ops):
            raise IndexError("op_index out of range")
        target = max(1, min(new_parallelism, self._max_workers))
        self.ops[op_index].adjust_parallelism(target)

    def close(self, *, hard: bool = False) -> None:
        with self._context_lock:
            ctx = self._active_context

        if ctx is not None:
            # tell everyone to stop
            ctx.stop_event.set()
            self._put_stage_stop(ctx)
            self._join_threads(ctx, hard=hard)

        self._executor.shutdown(wait=not hard, cancel_futures=hard)
