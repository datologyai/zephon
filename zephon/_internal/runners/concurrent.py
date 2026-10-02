# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Building blocks shared by concurrent (thread/process) stage runners."""

import queue
import threading
import time
from dataclasses import dataclass, field
from multiprocessing import queues as mp_queues
from typing import Any, Generic, Iterable, Iterator, Protocol, Sequence, TypeVar

from zephon._internal.flush_context import shutdown_flush
from zephon._internal.observability.size_estimator import estimate_bytes
from zephon._internal.runners.base import BaseOperatorState, StageRunnerBase
from zephon._internal.runners.pump_timer import PumpTimer
from zephon._internal.stream import (
    Microbatch,
    RunnerStageIn,
    RunnerStageOut,
    RunnerStreamIn,
    resolve_lazy_payloads,
)
from zephon._internal.utils import buffered_iterable
from zephon.observability.stats import BackpressureDelta, NodeMetricsDelta
from zephon.types import SampleBatch, SampleRecord

S = TypeVar("S", bound="ConcurrentOperatorState")
_T = TypeVar("_T")
_QItem = TypeVar("_QItem")

# -- Shutdown timeout constants (seconds) -----------------------------------
_GRACEFUL_THREAD_JOIN: float = 5.0
_HARD_THREAD_JOIN: float = 0.1


class _QueueLike(Protocol[_T]):
    def put(
        self, item: _T, /, block: bool = ..., timeout: float | None = ...
    ) -> None: ...
    def put_nowait(self, item: _T, /) -> None: ...
    def get(self, block: bool = ..., timeout: float | None = ...) -> _T: ...
    def get_nowait(self) -> _T: ...
    def empty(self) -> bool: ...
    def qsize(self) -> int: ...


_EMPTY_EXCEPTIONS: tuple[type[Exception], ...] = (
    queue.Empty,
    mp_queues.Empty,  # type: ignore[attr-defined]
)


class _Stop:
    """Sentinel payload used to shut down per-operator queues."""


StopToken = _Stop


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

    def peek(self) -> int:
        """Return the current count without acquiring the lock.

        Safe ONLY when the caller guarantees serial access — e.g. the Ray
        runner reads this from the pump thread, which is the sole thread
        that mutates the counter for that runner's operator state. For
        callers that share the counter across producer/consumer threads,
        use :meth:`is_zero` (or add a locked accessor if you need the
        integer value).
        """
        return self._count

    def try_decrement(self) -> bool:
        """Atomically decrement if count > 0. Returns True if decremented."""
        with self._cv:
            if self._count == 0:
                return False
            self._count -= 1
            if self._count == 0:
                self._cv.notify_all()
            return True

    def force_zero(self) -> int:
        """Atomically set count to zero. Returns previous count."""
        with self._cv:
            old = self._count
            self._count = 0
            if old > 0:
                self._cv.notify_all()
            return old


@dataclass(slots=True)
class WorkerErrorInfo:
    """Structured metadata describing a worker-side failure."""

    exc_type: str
    message: str
    formatted_traceback: str


class WorkerCrashed(RuntimeError):
    """Exception raised in the main process when a worker errors."""

    def __init__(self, info: WorkerErrorInfo):
        super().__init__(
            f"Worker raised {info.exc_type}: {info.message}\n{info.formatted_traceback}"
        )
        self.info = info


@dataclass(slots=True)
class RunnerResult:
    """Payload emitted by workers once a micro-batch completes."""

    seq: int
    payload: Microbatch
    wait_ns: int
    consumed_elements: int
    consumed_bytes: int
    queue_depth_snapshot: int
    proc_ns: int
    collect_metrics: bool = True
    ack: Any | None = None
    error: WorkerErrorInfo | None = None
    from_worker: bool = True


@dataclass
class ConcurrentOperatorState(BaseOperatorState):
    """Operator state shared by concurrent stage runners.

    Tracks the per-operator machinery common to every concurrent runner:
    the inbound queue the pump reads from, the deterministic-reordering
    buffer, the in-flight work counter, the monotonic seq counters, and
    the pump-thread timer used to attribute pump wall time to disjoint
    phase buckets.
    """

    queue_capacity: int = 1
    next_seq: int = field(init=False, default=0)
    emit_seq: int = field(init=False, default=0)
    pending_results: dict[int, RunnerResult] = field(init=False, default_factory=dict)
    inflight: _InflightCounter = field(init=False)
    input_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] = field(init=False)
    pump_timer: PumpTimer = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.inflight = _InflightCounter()
        self.pump_timer = PumpTimer(
            enabled=self.collect_stats,
            stage_index=self.stage_index,
            op_index=self.op_index,
            stage_name=self.stage_name,
            op_name=self.node.name,
        )

    @property
    def all_completions_observed(self) -> bool:
        """True iff every worker completion is visible to the pump thread.

        The pump's upstream-closed termination check uses this to decide
        whether in-flight work has fully settled. Subclasses that have an
        async worker→pump handoff gap (e.g. a bounded result_queue the
        worker pushes to) override this to close that observation window.

        The default is ``True`` because the base class makes no assumption
        about an async handoff — runners where the pump itself observes
        every completion synchronously (e.g. direct ``ray.wait()`` on the
        pump thread) have no such window.
        """
        return True


@dataclass(slots=True)
class ConcurrentRunContext:
    """Lifecycle bookkeeping for concurrent runners."""

    stop_token: StopToken
    stage_out_queue: _QueueLike[RunnerStageOut | StopToken]
    stop_event: threading.Event
    pumps: list[threading.Thread] = field(default_factory=list)
    feeder: threading.Thread | None = None
    error: BaseException | None = None
    stop_sent: bool = False


class ConcurrentStageRunner(StageRunnerBase[S], Generic[S]):
    """Skeleton for concurrent stage runners (threads, processes, ray, ...).

    This class implements the control flow shared by all concurrent runners
    without assuming how completed work travels from workers back to the pump
    thread. Subclasses implement two source-specific hooks —
    :meth:`_drain_results` (non-blocking) and :meth:`_await_one_result`
    (blocking with timeout) — and the loop drives them. The
    :class:`QueueDrainStageRunner` subclass provides bounded-queue
    implementations used by thread and process runners; a Ray-based runner
    plugs in ``ray.wait()``-based implementations.

    * The public API is a pull-driven iterator (:meth:`run`) that yields
      ``RunnerStageOut`` elements to the caller.
    * Internally, each operator in the fused stage owns a small bounded input
      queue and optional buffering rules (via :class:`BaseOperatorState`).
    * When an input micro-batch is ready, the runner schedules it on a worker
      backend (thread pool, process pool, remote worker, etc.) via
      :meth:`_schedule_batch`.
    * Completed work is observed by the pump thread via the subclass's
      :meth:`_drain_results` / :meth:`_await_one_result` hooks. Results flow
      through :meth:`_handle_result` for optional deterministic reordering
      and then downstream via :meth:`_forward_ready_result`.

    Deterministic vs non-deterministic execution
    -------------------------------------------

    The ``deterministic`` flag controls how results are ordered:

    * Non-deterministic mode (default):
      - Results are forwarded to downstream operators in *completion order*.
      - This maximizes throughput and minimizes latency at the cost of
        non-stable ordering when worker runtimes differ.
      - ``RunnerResult.seq`` is still populated but ignored for ordering.

    * Deterministic mode:
      - Each scheduled micro-batch must be assigned a monotonically increasing
        integer sequence number by :meth:`_schedule_batch`.
      - Every worker completion is reported as a :class:`RunnerResult`
        containing that ``seq``, even when the payload is empty because the
        operator buffered or filtered everything.
      - :meth:`_handle_result` stores results in ``pending_results`` and
        advances ``emit_seq`` only when the next integer is present. Out-of-
        order completions simply accumulate in the map until their predecessors
        arrive.

    Under the following constraints, the seq-based scheme reconstructs the
    exact stream a single-threaded run would produce:

    1. Operators are deterministic given a deterministic input micro-batch.
    2. Any cross-element state is confined to a single operator instance, or
       the planner constrains the operator to run with parallelism 1 when
       such state is needed.
    3. Fan-out operators derive child lineage deterministically from their
       inputs.
    4. Operators do not depend on wall-clock interleaving between invocations.

    Example (Batch operator)
    ------------------------

    Consider an operator like ``Batch`` that maintains a per-lane buffer
    inside each operator instance. By default the planner creates a
    single instance (parallelism = 1), so there is exactly one authoritative
    buffer per lane. Inputs traverse the stage in deterministic order, seq
    numbers enforce downstream ordering, and every flush produces the same
    micro-batch boundaries regardless of how worker tasks interleave.

    If for some reason we increase the operator's parallelism above 1,
    those buffers are split across instances and each instance owns an
    independent partial view of the lane. The effective flush boundaries then
    depend on which instance sees which elements first, and batching becomes
    non-deterministic. Operators with this kind of internal buffering are
    therefore currently are required to run single-threaded when deterministic behavior is required.
    Long term we might want to explore extending the buffering features offered by
    the stage runner.

    Accumulator-based buffering
    ---------------------------

    All cross-invocation state is managed by accumulators that run on the
    pump thread (serial). This ensures deterministic batch boundaries:

    * Runner buffering happens in :class:`BaseOperatorState.enqueue` before a
      batch is handed to the worker backend. The operator's accumulator
      determines batch boundaries on the pump thread, tagged with seq
      (in deterministic mode), and then scheduled.
      Time-based and count-based flushes are controlled by the accumulator.

    * When upstream is exhausted, the accumulator's ``flush()`` method is
      called to emit any remaining buffered data. Worker ``process_many``
      calls are stateless across invocations.

    Bounded input and result queues provide backpressure even when operators
    expand or filter the stream. When queues fill up, producers block until
    consumers drain enough space or until the stage is being shut down.

    Subclass responsibilities
    -------------------------

    Concrete runners (e.g. thread-based or process-based) provide the
    backend-specific wiring by implementing:

    * :meth:`_create_context` – allocate a :class:`ConcurrentRunContext`
      containing a stop token, the stage output queue, and shared stop/event
      state.
    * :meth:`_schedule_batch` – submit a micro-batch to the worker backend,
      assign a seq, measure processing time, and eventually surface a
      :class:`RunnerResult` to the pump thread.
    * :meth:`_drain_results` – non-blockingly observe any completed results
      and forward them via :meth:`_handle_result`.
    * :meth:`_await_one_result` – block up to ``timeout`` seconds for at
      least one completion and forward it.
    * :meth:`_ack_result` (optional) – perform backend-specific acknowledgements
      once a result has been forwarded downstream (e.g. freeing shared memory
      in a process-based runner).

    The rest of the coordination – feeding upstream elements into the first
    operator, driving operator pump threads, enforcing deterministic ordering,
    forwarding stop tokens, and exposing the final iterator – is shared between
    all concurrent runners implemented on top of this class.
    """

    _OperatorState: type[S]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._context_lock = threading.Lock()
        self._active_context: ConcurrentRunContext | None = None
        self._emit_backpressure_metrics: Any = self._ctx_services.get(
            "emit_backpressure_metrics"
        )
        self._emit_pump_metrics: Any = self._ctx_services.get("emit_pump_metrics")
        flush_interval_s = float(self._ctx_services.get("pump_flush_interval_s") or 5.0)
        self._pump_flush_interval_ns: int = int(flush_interval_s * 1_000_000_000)

    # -- Abstract hooks -------------------------------------------------
    def _create_context(self) -> ConcurrentRunContext:
        """Allocate the per-run context.

        Subclasses may override if they need a custom stage-output queue
        shape; the default is sufficient for thread-, process-, and
        ray-backed runners.
        """
        out_capacity = max(1, self._prefetch_capacity or self._queue_capacity)
        stage_out_queue = queue.Queue[RunnerStageOut | StopToken](maxsize=out_capacity)
        return ConcurrentRunContext(
            stop_token=StopToken(),
            stage_out_queue=stage_out_queue,
            stop_event=threading.Event(),
        )

    def _schedule_batch(
        self,
        state: S,
        batch: list[RunnerStreamIn],
        *,
        wait_ns: int,
        context: ConcurrentRunContext,
    ) -> None:
        raise NotImplementedError

    # -- Optional hooks -------------------------------------------------
    def _before_run(self, context: ConcurrentRunContext) -> None:
        """Hook invoked right after the context is created but before pumps start."""

    def _after_run(self, context: ConcurrentRunContext) -> None:
        """Hook invoked once the iterator exits and pumps are joined."""

    def _on_pump_started(self, state: S) -> None:
        """Hook invoked at the start of ``_operator_loop`` on the pump thread."""

    def _post_schedule_batch(
        self,
        state: S,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
    ) -> None:
        """Hook invoked after every ``_schedule_batch`` call.

        Subclasses can override this to handle results that were produced
        synchronously during scheduling (e.g. the thread runner's
        ``sync_result`` stash).
        """

    def _ack_result(
        self,
        state: S,
        result: RunnerResult,
        context: ConcurrentRunContext,
    ) -> None:
        """Called after a result payload has been forwarded downstream."""

    # -- Queue helpers --------------------------------------------------
    @staticmethod
    def _queue_put(q: _QueueLike[_T], item: _T, timeout: float) -> None:
        while True:
            try:
                q.put(item, timeout=timeout)
                return
            except _EMPTY_EXCEPTIONS:
                raise
            except queue.Full:
                raise

    @staticmethod
    def _queue_get_nowait(q: _QueueLike[_T]) -> _T:
        try:
            return q.get_nowait()
        except _EMPTY_EXCEPTIONS as exc:
            raise queue.Empty from exc

    @staticmethod
    def _queue_get(q: _QueueLike[_T], timeout: float) -> _T:
        try:
            return q.get(timeout=timeout)
        except _EMPTY_EXCEPTIONS as exc:
            raise queue.Empty from exc

    # -- Source-specific hooks (implemented by subclasses) --------------
    def _drain_results(
        self,
        state: S,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
    ) -> None:
        """Drain any ready results and forward each via :meth:`_handle_result`.

        Non-blocking and opportunistic; subclasses may rate-limit the sweep
        (the Ray runner does). Subclasses implement this per their result
        source — queue-pop for queue-drain runners, ``ray.wait()`` for
        ray-based runners, etc.
        """
        raise NotImplementedError

    def _await_one_result(
        self,
        state: S,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
        timeout: float,
    ) -> None:
        """Block up to ``timeout`` for at least one result and forward it.

        Called from the upstream-closed drain path when we're waiting for
        in-flight work to settle. Forwards via :meth:`_handle_result`. A
        single-attempt wait is acceptable; the outer loop retries if
        nothing arrives. Subclasses implement this per their result source.
        """
        raise NotImplementedError

    # -- Core loops -----------------------------------------------------

    def _handle_result(
        self,
        state: S,
        result: RunnerResult,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
    ) -> None:
        if result.error is not None:
            self._forward_ready_result(state, result, next_queue, context)
            return
        if state.deterministic:
            state.pending_results[result.seq] = result
            while state.emit_seq in state.pending_results:
                ready = state.pending_results.pop(state.emit_seq)
                state.emit_seq += 1
                self._forward_ready_result(state, ready, next_queue, context)
            return
        self._forward_ready_result(state, result, next_queue, context)

    def _forward_ready_result(
        self,
        state: S,
        result: RunnerResult,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
    ) -> None:
        if result.error is not None:
            exc = WorkerCrashed(result.error)
            self._record_error(context, exc)
            self._ack_result(state, result, context)
            return
        if result.payload:
            self._emit_downstream(result.payload, next_queue, context)
        self._record_result_metrics(state, result)
        self._ack_result(state, result, context)

    def _record_result_metrics(self, state: S, result: RunnerResult) -> None:
        if not result.collect_metrics or result.error is not None:
            return
        if not self._tracking_mode.collects_nodes:
            return
        metrics_meta = self._metrics_meta[state.op_index]
        stage_index, stage_name, op_index, op_name = metrics_meta
        produced_elements = len(result.payload)
        produced_bytes = estimate_bytes(result.payload)
        delta = NodeMetricsDelta(
            stage_index=stage_index,
            op_index=op_index,
            stage_name=stage_name,
            name=op_name,
            processed_ns=result.proc_ns,
            produced_elements=produced_elements,
            consumed_elements=result.consumed_elements,
            produced_bytes=produced_bytes,
            consumed_bytes=result.consumed_bytes,
            wait_ns=result.wait_ns,
            max_queue_depth=result.queue_depth_snapshot,
            min_processing_ns=result.proc_ns,
            max_processing_ns=result.proc_ns,
        )
        self._record_node_metrics(delta)

    def _emit_downstream(
        self,
        elements: Microbatch,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
    ) -> None:
        if not elements:
            return
        if next_queue is not None:
            self._put_into_queue(next_queue, elements, context)
            return
        self._emit_stage_output(elements, context)

    def _emit_stage_output(
        self,
        elements: Microbatch,
        context: ConcurrentRunContext,
    ) -> None:
        """Emit elements to the stage output queue.

        Resolves any ``LazyPayload`` instances before emission — downstream
        thread stages and the engine iterator expect materialized payloads.
        """
        if not elements:
            return
        resolve_lazy_payloads(elements)
        if self._emit_microbatches:
            self._put_into_queue(context.stage_out_queue, elements, context)
            return
        for elem in elements:
            self._put_into_queue(context.stage_out_queue, elem, context)

    def _next_queue_for(
        self, state: S
    ) -> _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None:
        """Return the input queue of the next operator, or None for the last."""
        next_index = state.op_index + 1
        if next_index < len(self.ops):
            return self.ops[next_index].input_queue
        return None

    def _operator_loop(self, idx: int, context: ConcurrentRunContext) -> None:
        state = self.ops[idx]
        next_queue = self._next_queue_for(state)

        state.reset_buffers()
        state.pump_timer.start_window()
        upstream_closed = False
        self._on_pump_started(state)

        try:
            while True:
                self._drain_results(state, next_queue, context)
                self._maybe_flush_pump_timer(state)

                if upstream_closed:
                    if (
                        state.inflight.is_zero()
                        and state.all_completions_observed
                        and (not state.pending_results or context.error is not None)
                        and not state.accumulator_impl.has_pending_data()
                    ):
                        # Final drain to catch results that arrived after the loop's drain.
                        # Closes the handoff-observation window: a worker may have completed
                        # put_result() and decremented its handoff counter between our drain
                        # and this check, in which case a fresh drain exposes the result.
                        self._drain_results(state, next_queue, context)

                        # In the error case, we don't wait for missing seq gaps.
                        if context.error is not None:
                            # Use list() to snapshot: other threads may still modify pending_results.
                            for pending in list(state.pending_results.values()):
                                self._ack_result(state, pending, context)
                            state.pending_results.clear()

                        self._signal_downstream_stop(next_queue, context)
                        return
                    self._await_one_result(state, next_queue, context, timeout=0.05)
                    continue

                if context.stop_event.is_set():
                    upstream_closed = True
                    with shutdown_flush():
                        ready = state.enqueue([], force=True)
                    for batch, wait_ns in ready:
                        self._drain_results(state, next_queue, context)
                        self._schedule_batch(
                            state,
                            batch,
                            wait_ns=wait_ns,
                            context=context,
                        )
                        self._post_schedule_batch(state, next_queue, context)
                    continue

                # Try non-blocking get first so we don't stall for 50 ms
                # when input is already available.
                try:
                    item = self._queue_get_nowait(state.input_queue)
                except queue.Empty:
                    self._drain_results(state, next_queue, context)
                    try:
                        with state.pump_timer.measure("input_wait"):
                            item = self._queue_get(state.input_queue, timeout=0.05)
                    except queue.Empty:
                        continue

                # Shutdown may have started during the queue read. Let the
                # existing stop_event branch above perform the shutdown flush.
                if context.stop_event.is_set():
                    continue

                if isinstance(item, _Stop):
                    upstream_closed = True
                    ready = state.enqueue([], force=True)
                else:
                    state.pump_timer.note("items_consumed")
                    ready = state.enqueue(item, force=False)

                # Drain results after every dispatch to keep workers unblocked.
                for batch, wait_ns in ready:
                    self._schedule_batch(
                        state,
                        batch,
                        wait_ns=wait_ns,
                        context=context,
                    )
                    self._post_schedule_batch(state, next_queue, context)
                    self._drain_results(state, next_queue, context)
        except BaseException as exc:  # noqa: BLE001
            # Any unexpected failure in the pump should propagate as a stage error
            # so the iterator can stop cleanly rather than killing the thread.
            self._record_error(context, exc)
            return
        finally:
            self._flush_pump_timer(state)

    def _maybe_flush_pump_timer(self, state: S) -> None:
        """Emit a pump-timing delta if the cadence has elapsed."""
        if self._emit_pump_metrics is None or not state.pump_timer.enabled:
            return
        now_ns = time.perf_counter_ns()
        if not state.pump_timer.should_flush(now_ns, self._pump_flush_interval_ns):
            return
        delta = state.pump_timer.flush(now_ns)
        if delta is not None:
            self._emit_pump_metrics(delta)

    def _flush_pump_timer(self, state: S) -> None:
        """Emit the final pump-timing delta on pump exit."""
        if self._emit_pump_metrics is None or not state.pump_timer.enabled:
            return
        delta = state.pump_timer.flush()
        if delta is not None and delta.has_samples():
            self._emit_pump_metrics(delta)

    def _signal_downstream_stop(
        self,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
    ) -> None:
        if next_queue is None:
            self._put_stage_stop(context)
        else:
            self._put_into_queue(next_queue, context.stop_token, context)

    def _put_into_queue(
        self,
        q: _QueueLike[_QItem],
        item: _QItem,
        context: ConcurrentRunContext,
    ) -> None:
        while True:
            try:
                q.put(item, timeout=0.1)
                return
            except queue.Full:
                # Backpressure: pump thread can't forward to downstream queue
                if self._emit_backpressure_metrics is not None:
                    self._emit_backpressure_metrics(
                        BackpressureDelta(
                            stage_index=self._stage_index,
                            put_into_queue_backpressure_events=1,
                        )
                    )
                if context.stop_event.is_set():
                    try:
                        q.put_nowait(item)
                    except queue.Full:
                        return
                continue

    def _put_stage_stop(self, context: ConcurrentRunContext) -> None:
        if context.stop_sent:
            return
        context.stop_sent = True
        self._put_into_queue(context.stage_out_queue, context.stop_token, context)

    def _record_error(
        self,
        context: ConcurrentRunContext,
        exc: BaseException,
    ) -> None:
        if context.error is None:
            context.error = exc
            context.stop_event.set()
            self._put_stage_stop(context)

    def _start_feeder(
        self,
        upstream: Iterable[RunnerStageIn],
        context: ConcurrentRunContext,
    ) -> None:
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
                    first_state.input_queue,
                    context.stop_token,
                    context,
                )

        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()
        context.feeder = feeder

    def _start_operator_threads(self, context: ConcurrentRunContext) -> None:
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

    def _join_threads(
        self, context: ConcurrentRunContext, *, hard: bool = False
    ) -> None:
        timeout = _HARD_THREAD_JOIN if hard else _GRACEFUL_THREAD_JOIN
        for thread in context.pumps:
            thread.join(timeout=timeout)
        if context.feeder is not None:
            context.feeder.join(timeout=timeout)

    # -- Public API -----------------------------------------------------
    def run(self, upstream: Iterable[RunnerStageIn]) -> Iterator[RunnerStageOut]:
        context = self._create_context()
        with self._context_lock:
            if self._active_context is not None:
                raise RuntimeError("Stage runner already in use")
            self._active_context = context

        try:
            self._before_run(context)
            self._start_operator_threads(context)
            self._start_feeder(upstream, context)

            def iterator() -> Iterator[RunnerStageOut]:
                try:
                    while True:
                        try:
                            item = self._queue_get(context.stage_out_queue, timeout=0.1)
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
            self._after_run(context)
            with self._context_lock:
                self._active_context = None
