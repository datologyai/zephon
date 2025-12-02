"""Process-based stage runner that executes ops using multiprocessing workers."""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Semaphore
from typing import Any, Callable, Iterable, Literal, Protocol, Sequence, TypeVar, cast

from zephon.core.constants import (
    Microbatch,
    RunnerStageIn,
    RunnerStageOut,
    RunnerStreamIn,
    StreamItem,
)
from zephon.core.graph import Node, Stage
from zephon.core.op_base import Op, OpContext
from zephon.observability.config import ExecutionTrackingMode
from zephon.observability.size_estimator import estimate_bytes
from zephon.runners.concurrent import (
    ConcurrentOperatorState,
    ConcurrentRunContext,
    ConcurrentStageRunner,
    RunnerResult,
    StopToken,
    WorkerCrashed,
    WorkerErrorInfo,
    _QueueLike,
)

Q = TypeVar("Q")


class _ClosableQueue(_QueueLike[Q], Protocol):
    def close(self) -> None: ...


_DEBUG = bool(os.environ.get("ZEPHON_DEBUG_PROCESS_RUNNER"))


def _debug(msg: str) -> None:  # pragma: no cover - diagnostics helper
    if _DEBUG:
        print(f"[ProcessRunner] {msg}", file=sys.stderr, flush=True)


@dataclass
class _WorkerCommand:
    kind: Literal["batch", "finalize", "stop"]
    seq: int
    batch: list[RunnerStreamIn]
    wait_ns: int
    consumed_elements: int
    consumed_bytes: int
    queue_depth_snapshot: int
    collect_metrics: bool


@dataclass
class _ServiceRequest:
    worker_id: int
    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class _RemoteServiceProxy:
    """Callable shim that forwards execution to the main process."""

    __slots__ = ("_worker_id", "_name", "_request_q", "_response_q")

    def __init__(
        self,
        worker_id: int,
        name: str,
        request_queue: _ClosableQueue[_ServiceRequest | None],
        response_queue: _ClosableQueue[tuple[bool, Any]],
    ) -> None:
        self._worker_id = worker_id
        self._name = name
        self._request_q = request_queue
        self._response_q = response_queue

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        _debug(f"proxy[{self._worker_id}:{self._name}] enqueue request")
        self._request_q.put(
            _ServiceRequest(
                worker_id=self._worker_id,
                name=self._name,
                args=tuple(args),
                kwargs=dict(kwargs),
            )
        )
        ok, payload = self._response_q.get()
        _debug(f"proxy[{self._worker_id}:{self._name}] response ok={ok}")
        if ok:
            return payload
        exc = payload if isinstance(payload, BaseException) else RuntimeError(payload)
        raise exc


@dataclass
class _ProcessWorkerConfig:
    worker_index: int
    worker_id: int
    op_proto: Op[Any, StreamItem]
    stage_index: int
    stage_name: str
    op_index: int
    collect_stats: bool
    ctx_services: dict[str, Any]
    task_queue: _QueueLike[_WorkerCommand]
    result_queue: _QueueLike[RunnerResult]
    backpressure: Semaphore


QueueFactory = Callable[..., _ClosableQueue[Any]]
SemaphoreFactory = Callable[..., Semaphore]
ProcessFactory = Callable[..., BaseProcess]


def _process_worker_main(config: _ProcessWorkerConfig) -> None:
    try:
        op_instance = copy.deepcopy(config.op_proto)
        ctx = OpContext(dict(config.ctx_services))
        op_instance.setup(
            ctx,
            config.stage_index,
            config.stage_name,
            config.op_index,
            config.collect_stats,
        )
        while True:
            command = config.task_queue.get()
            if command.kind == "stop":
                _debug(f"worker[{config.worker_index}] received stop")
                break
            if command.kind == "batch":
                _debug(
                    f"worker[{config.worker_index}] processing batch seq={command.seq}"
                )
                start_ns = time.perf_counter_ns() if command.collect_metrics else 0
                try:
                    outputs = op_instance.process_many(command.batch)
                except (NotImplementedError, AttributeError):
                    outputs = None
                if outputs is None:
                    out: list[StreamItem] = []
                    for element in command.batch:
                        out.extend(op_instance.process_one(element))
                    outputs = out
                proc_ns = (
                    time.perf_counter_ns() - start_ns if command.collect_metrics else 0
                )
            else:  # finalize
                _debug(f"worker[{config.worker_index}] finalize seq={command.seq}")
                outputs = op_instance.finalize()
                proc_ns = 0

            config.backpressure.acquire()
            result = RunnerResult(
                seq=command.seq,
                payload=outputs,
                wait_ns=command.wait_ns,
                consumed_elements=command.consumed_elements,
                consumed_bytes=command.consumed_bytes,
                queue_depth_snapshot=command.queue_depth_snapshot,
                proc_ns=proc_ns,
                collect_metrics=command.collect_metrics and command.kind == "batch",
                ack=config.worker_index,
            )
            config.result_queue.put(result)
    except BaseException as exc:  # noqa: BLE001
        try:
            config.backpressure.acquire()
        except Exception:
            pass
        info = WorkerErrorInfo(
            exc_type=type(exc).__name__,
            message=str(exc),
            formatted_traceback=traceback.format_exc(),
        )
        config.result_queue.put(
            RunnerResult(
                seq=-1,
                payload=[],
                wait_ns=0,
                consumed_elements=0,
                consumed_bytes=0,
                queue_depth_snapshot=-1,
                proc_ns=0,
                collect_metrics=False,
                ack=config.worker_index,
                error=info,
            )
        )


@dataclass
class _ProcessOperatorState(ConcurrentOperatorState):
    task_queue: _QueueLike[_WorkerCommand] | None = field(init=False, default=None)
    workers: list[BaseProcess] = field(init=False, default_factory=list)
    result_semaphores: list[Semaphore] = field(init=False, default_factory=list)
    worker_ids: list[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.input_queue = queue.Queue[Sequence[RunnerStreamIn] | StopToken](
            maxsize=self.queue_capacity
        )
        self.result_queue = queue.Queue[RunnerResult](maxsize=1)


class ProcessStageRunner(ConcurrentStageRunner[_ProcessOperatorState]):
    """Execute a stage in dedicated worker processes with IPC queues.

    This runner is a concrete :class:`ConcurrentStageRunner` that offloads
    operator execution to a pool of child processes.  It is intended for
    CPU-bound or GIL-heavy workloads where threads are not sufficient, or
    where isolation between operators is desirable.

    High-level execution model
    --------------------------

    * Each operator in the stage is represented by a :class:`_ProcessOperatorState`
      in the main process. For that operator, the runner maintains:

      - an in-process bounded ``input_queue`` for upstream micro-batches
        (``Sequence[RunnerStreamIn]`` or :class:`StopToken`), and
      - an out-of-process ``task_queue`` and ``result_queue`` backing the worker
        processes.

    * For each operator, :meth:`_launch_workers` creates a pool of worker
      processes. Each worker:

      - owns a deep-copied :class:`Op` instance,
      - receives :class:`_WorkerCommand` messages from ``task_queue``
        (kinds: ``"batch"``, ``"finalize"``, ``"stop"``),
      - runs ``process_many`` / ``process_one`` or ``finalize`` on its local
        operator instance, and
      - pushes :class:`RunnerResult` objects onto the shared ``result_queue``.

    * In the multi-operator case, the usual pump threads in
      :class:`ConcurrentStageRunner`:

      - pull micro-batches from the in-process ``input_queue``,
      - use :class:`BaseOperatorState.enqueue` to apply runner-level buffering,
      - call :meth:`_schedule_batch` to send a ``"batch"`` command to the
        worker pool, and
      - drain ``result_queue`` and forward ready results downstream using the
        generic deterministic/non-deterministic logic in the base class.

    Determinism and ordering
    ------------------------

    The deterministic vs non-deterministic semantics, the meaning of the
    ``seq`` field on :class:`RunnerResult`, and the interaction with
    operator-local buffering are defined entirely by
    :class:`ConcurrentStageRunner`.  This process-based runner simply ensures
    that every scheduled batch produces a corresponding :class:`RunnerResult`
    (or a structured :class:`WorkerErrorInfo` on failure), so the base class
    can re-establish the same logical stream a single-threaded run would
    produce when the operator determinism constraints are satisfied.

    IPC batching
    ------------

    Sending very small micro-batches across process boundaries can be
    inefficient.  To amortize IPC overhead, the runner can further partition
    or coalesce micro-batches before they are shipped to workers:

    * ``ipc_batch_size_factor`` controls how the effective IPC batch size is
      derived from the buffering hints of the first operator in the stage
      (see :meth:`_derive_ipc_batch_size`).
    * :meth:`_dispatch_ipc_batches` breaks large logical micro-batches into
      smaller chunks tuned for the worker pool and the underlying queue
      implementation, while still preserving the seq-based ordering guarantees
      enforced by :class:`ConcurrentStageRunner`.

    Context services and remote calls
    ---------------------------------

    Workers may need access to shared services (e.g. logging, metrics, model
    registries) that live in the main process.  The process runner supports
    this via a simple request/response channel:

    * Callables in ``ctx_services`` are replaced with
      :class:`_RemoteServiceProxy` instances when passed to workers.
    * When a worker calls such a proxy, it sends a :class:`_ServiceRequest`
      through a shared service queue and blocks for a reply.
    * A dedicated service thread in the main process (:meth:`_service_loop`)
      receives these requests, executes the real callable, and sends the
      result (or exception) back to the worker via a per-worker response
      queue.

    This keeps the worker processes lean while allowing them to interact with
    rich main-process services without sharing complex objects directly.

    Backpressure and resource limits
    --------------------------------

    Backpressure is applied at two levels:

    * In-process queues:
      - The per-operator ``input_queue`` and stage output queue behave like in
        the thread runner, bounding how many micro-batches can be buffered
        between operators.

    * Cross-process result flow:
      - Each worker gets a dedicated :class:`multiprocessing.Semaphore` used as
        a result-side backpressure mechanism.  Before pushing a
        :class:`RunnerResult`, the worker acquires the semaphore; the main
        process releases it from :meth:`_ack_result` after the result has been
        forwarded downstream or handled.
      - This prevents workers from out-running the main process and filling
        the shared ``result_queue`` without bound.

    Together, these mechanisms limit both in-process and cross-process memory
    usage while preserving throughput.

    Single-operator direct-IPC fast path
    ------------------------------------

    For stages that contain a single operator, the runner enables a direct IPC
    fast path:

    * Instead of spinning per-operator pump threads in the main process, the
      feeder thread (:meth:`_start_direct_ipc_feeder`) drives buffering and
      scheduling for that operator directly.
    * It calls :meth:`BaseOperatorState.enqueue` to apply buffering, dispatches
      IPC batches to workers, and drives result draining and finalization
      itself (:meth:`_drain_until_idle`, :meth:`_finalize_state`).
    * This reduces coordination overhead for the common case of a single
      heavy operator running fully in separate processes.

    Limitations and usage notes
    ---------------------------

    * :class:`ProcessStageRunner` does **not** support live scaling of operator
      parallelism; :meth:`set_parallelism` is intentionally unimplemented.
      Changing parallelism requires rebuilding the runner so that worker
      processes can be recreated with the desired configuration.

    * Because operators and inputs must be serialized and shipped across
      process boundaries, this runner is best suited for CPU-bound work or
      operators with relatively coarse-grained micro-batches.  For lightweight
      or I/O-bound operators, :class:`ThreadStageRunner` is often a better
      choice.

    * All workers are spawned using a configurable :class:`multiprocessing`
      context (``mp_context``), defaulting to the ``"spawn"`` start method to
      avoid the usual caveats of ``fork`` in multi-threaded programs.
    """

    _OperatorState = _ProcessOperatorState

    def __init__(
        self,
        stage: Stage,
        ctx_services: dict[str, Any],
        max_workers: int,
        *,
        prefetch_capacity: int = 0,
        queue_capacity: int = 4,
        ipc_batch_size_factor: int = 2,
        deterministic: bool = False,
        allow_latency_flush_in_deterministic: bool = True,
        stage_index: int = 0,
        tracking_mode: ExecutionTrackingMode = ExecutionTrackingMode.OFF,
        stage_output_mode: Literal["microbatches", "stream_items"] = "microbatches",
        mp_context: BaseContext | None = None,
    ) -> None:
        self._queue_capacity = max(1, queue_capacity)
        self._ipc_batch_size_factor = max(1, int(ipc_batch_size_factor))
        self._ipc_batch_size = self._derive_ipc_batch_size(stage)
        ctx = mp_context or mp.get_context("fork")
        queue_factory = cast(QueueFactory, getattr(ctx, "Queue"))
        semaphore_factory = cast(SemaphoreFactory, getattr(ctx, "Semaphore"))
        process_factory = cast(ProcessFactory, getattr(ctx, "Process"))
        self._queue_factory: QueueFactory = queue_factory
        self._semaphore_factory: SemaphoreFactory = semaphore_factory
        self._process_factory: ProcessFactory = process_factory
        self._service_queue = cast(
            _ClosableQueue[_ServiceRequest | None], self._queue_factory()
        )
        self._service_thread: threading.Thread | None = None
        self._service_stop = threading.Event()
        self._service_responses: dict[int, _ClosableQueue[tuple[bool, Any]]] = {}
        self._next_worker_id = 0
        self._single_op_direct_ipc = len(stage.nodes) == 1
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

    def _derive_ipc_batch_size(self, stage: Stage) -> int:
        fallback = 128
        if not stage.nodes:
            return fallback
        first = stage.nodes[0]
        buffering = None
        try:
            buffering = first.op.buffering()
        except Exception:
            buffering = None
        if buffering is None:
            return fallback
        max_batch = getattr(buffering, "max_batch", None)
        if max_batch is None:
            return fallback
        try:
            base = int(max(1, int(max_batch)))
        except Exception:
            return fallback
        factor = max(1, int(self._ipc_batch_size_factor))
        size = base * factor
        if size <= 0:
            return 1
        return size

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
    ) -> _ProcessOperatorState:
        return _ProcessOperatorState(
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

    def _create_context(self) -> ConcurrentRunContext:
        out_capacity = max(1, self._prefetch_capacity or self._queue_capacity)
        stage_out_queue = queue.Queue[RunnerStageOut | StopToken](maxsize=out_capacity)
        return ConcurrentRunContext(
            stop_token=StopToken(),
            stage_out_queue=stage_out_queue,
            stop_event=threading.Event(),
        )

    def _start_operator_threads(self, context: ConcurrentRunContext) -> None:
        if self._single_op_direct_ipc:
            return
        super()._start_operator_threads(context)

    def _before_run(self, context: ConcurrentRunContext) -> None:
        self._start_service_thread()
        for state in self.ops:
            self._launch_workers(state)

    def _after_run(self, context: ConcurrentRunContext) -> None:
        self._shutdown_workers()
        self._stop_service_thread()

    def _start_service_thread(self) -> None:
        if self._service_thread is not None:
            return
        self._service_stop.clear()
        _debug("starting service thread")
        thread = threading.Thread(target=self._service_loop, daemon=True)
        thread.start()
        self._service_thread = thread

    def _stop_service_thread(self) -> None:
        if self._service_thread is None:
            return
        self._service_stop.set()
        self._service_queue.put(None)
        self._service_thread.join(timeout=1.0)
        self._service_thread = None
        for resp in self._service_responses.values():
            try:
                resp.close()
            except Exception:
                pass
        self._service_responses.clear()

    def _start_feeder(
        self,
        upstream: Iterable[RunnerStageIn],
        context: ConcurrentRunContext,
    ) -> None:
        if self._single_op_direct_ipc:
            self._start_direct_ipc_feeder(upstream, context)
            return
        super()._start_feeder(upstream, context)

    def _start_direct_ipc_feeder(
        self,
        upstream: Iterable[RunnerStageIn],
        context: ConcurrentRunContext,
    ) -> None:
        if not self.ops:
            super()._start_feeder(upstream, context)
            return

        state = self.ops[0]

        def feed() -> None:
            try:
                state.reset_buffers()
                for elem in upstream:
                    if context.stop_event.is_set():
                        break
                    batch = self._coerce_to_batch(elem)
                    if not batch:
                        continue
                    ready = state.enqueue(batch, force=False)
                    self._dispatch_ipc_batches(state, ready, context)

                ready = state.enqueue([], force=True)
                self._dispatch_ipc_batches(state, ready, context)

                self._drain_until_idle(state, context)
                tail = self._finalize_state(state, context, None)
                if tail:
                    self._emit_downstream(tail, None, context)
            except BaseException as exc:  # noqa: BLE001
                self._record_error(context, exc)
            finally:
                self._signal_downstream_stop(None, context)

        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()
        context.feeder = feeder

    def _dispatch_ipc_batches(
        self,
        state: _ProcessOperatorState,
        ready: list[tuple[list[RunnerStreamIn], int]],
        context: ConcurrentRunContext,
    ) -> None:
        if not ready:
            return
        for batch, wait_ns in ready:
            if not batch:
                continue
            self._schedule_chunked_batch(
                state,
                batch,
                wait_ns=wait_ns,
                context=context,
                next_queue=None,
            )

    def _next_queue_for(
        self, state: _ProcessOperatorState
    ) -> _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None:
        next_index = state.op_index + 1
        if next_index < len(self.ops):
            return self.ops[next_index].input_queue
        return None

    def _schedule_chunked_batch(
        self,
        state: _ProcessOperatorState,
        batch: list[RunnerStreamIn],
        *,
        wait_ns: int,
        context: ConcurrentRunContext,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
    ) -> None:
        if not batch or context.stop_event.is_set():
            return
        size = self._ipc_batch_size
        start = 0
        while start < len(batch):
            chunk = batch[start : start + size]
            if not chunk:
                break
            self._schedule_worker_batch(
                state,
                chunk,
                wait_ns=wait_ns,
                context=context,
            )
            self._drain_results(state, next_queue, context)
            start += size

    def _drain_until_idle(
        self,
        state: _ProcessOperatorState,
        context: ConcurrentRunContext,
    ) -> None:
        while True:
            if state.inflight.is_zero() and not state.pending_results:
                break
            try:
                item = self._queue_get(state.result_queue, timeout=0.05)
            except queue.Empty:
                continue
            self._handle_result(state, item, None, context)

    def _service_loop(self) -> None:
        while not self._service_stop.is_set():
            try:
                req = self._service_queue.get(timeout=0.1)
            except Exception:
                if self._service_stop.is_set():
                    break
                continue
            if req is None:
                break
            resp = self._service_responses.get(req.worker_id)
            if resp is None:
                continue
            fn = self._ctx_services.get(req.name)
            if fn is None or not callable(fn):
                resp.put(
                    (
                        False,
                        RuntimeError(f"Unknown context service '{req.name}'"),
                    )
                )
                continue
            try:
                _debug(f"service handling worker={req.worker_id} service={req.name}")
                result = fn(*req.args, **req.kwargs)
                resp.put((True, result))
            except BaseException as exc:  # noqa: BLE001
                resp.put((False, exc))

    def _launch_workers(self, state: _ProcessOperatorState) -> None:
        state.workers = []
        state.worker_ids = []
        state.result_semaphores = []
        result_queue = cast(_ClosableQueue[RunnerResult], self._queue_factory())
        state.result_queue = result_queue
        queue_capacity = max(1, self._queue_capacity * max(1, state.parallelism))
        task_queue = cast(
            _ClosableQueue[_WorkerCommand], self._queue_factory(queue_capacity)
        )
        state.task_queue = task_queue
        for idx in range(state.parallelism):
            resp_queue = cast(_ClosableQueue[tuple[bool, Any]], self._queue_factory())
            semaphore = self._semaphore_factory(self._queue_capacity)
            worker_id = self._next_worker_id
            self._next_worker_id += 1
            ctx_payload = self._build_worker_ctx(worker_id, resp_queue)
            config = _ProcessWorkerConfig(
                worker_index=idx,
                worker_id=worker_id,
                op_proto=state.node.op,
                stage_index=state.stage_index,
                stage_name=state.stage_name,
                op_index=state.op_index,
                collect_stats=self._tracking_mode.collects_nodes,
                ctx_services=ctx_payload,
                task_queue=task_queue,
                result_queue=result_queue,
                backpressure=semaphore,
            )
            proc: BaseProcess = self._process_factory(
                target=_process_worker_main,
                args=(config,),
                daemon=True,
            )
            proc.start()
            _debug(f"started worker process idx={idx} pid={proc.pid}")
            self._service_responses[worker_id] = resp_queue
            state.worker_ids.append(worker_id)
            state.result_semaphores.append(semaphore)
            state.workers.append(proc)

    def _shutdown_workers(self) -> None:
        for state in self.ops:
            task_queue = state.task_queue
            if task_queue is None:
                continue

            # First, try to drain any pending results before sending stop commands
            # This helps avoid deadlocks where workers are blocked on semaphores
            # and can't process stop commands. We just need to acknowledge results
            # to release semaphores - we don't need to process them fully.
            _debug(f"draining results before shutdown for {len(state.workers)} workers")
            drain_start = time.time()
            drain_timeout = 5.0  # Give up after 5 seconds of draining
            while time.time() - drain_start < drain_timeout:
                drained_any = False
                try:
                    # Drain results with timeout - get() returns immediately if items available
                    while True:
                        item = self._queue_get(state.result_queue, timeout=0.1)
                        # Just acknowledge to release semaphore - don't process fully
                        self._ack_result(state, item, None)
                        state.inflight.decrement()
                        drained_any = True
                except queue.Empty:
                    pass

                if not drained_any and state.inflight.is_zero():
                    break

            # Now send stop commands to all workers
            # Note: We don't pass context here because we're in shutdown
            # and the draining above should have released semaphores already
            for _ in state.workers:
                self._send_command(
                    task_queue,
                    _WorkerCommand("stop", -1, [], 0, 0, 0, -1, False),
                    state=None,  # Don't drain during shutdown - already drained above
                    context=None,
                )

            # Wait for workers to finish, with reasonable timeout
            # We don't block indefinitely - if workers don't terminate quickly,
            # they're likely hung and we log a warning
            for proc in state.workers:
                proc.join(timeout=1.0)
                if proc.is_alive():
                    _debug(
                        f"Worker process {proc.pid} did not terminate within timeout, "
                        + "it may be hung"
                    )
            for wid in state.worker_ids:
                resp = self._service_responses.pop(wid, None)
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
            state.workers.clear()
            state.worker_ids.clear()
            state.result_semaphores.clear()
            # task_queue is guaranteed to be not None here due to check above
            close = getattr(task_queue, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            state.task_queue = None

    def _build_worker_ctx(
        self, worker_id: int, response_queue: _ClosableQueue[tuple[bool, Any]]
    ) -> dict[str, Any]:
        ctx: dict[str, Any] = {}
        for name, value in self._ctx_services.items():
            if callable(value):
                ctx[name] = _RemoteServiceProxy(
                    worker_id,
                    name,
                    self._service_queue,
                    response_queue,
                )
            else:
                ctx[name] = value
        return ctx

    def _send_command(
        self,
        queue_: _QueueLike[_WorkerCommand],
        command: _WorkerCommand,
        state: _ProcessOperatorState | None = None,
        context: ConcurrentRunContext | None = None,
    ) -> None:
        """Send command to worker queue, draining results if queue is full.

        If the queue is full, it means workers aren't consuming commands.
        This is often because workers are blocked on semaphores waiting to send results.
        By draining results, we release semaphores and allow workers to proceed.
        """
        retries = 0
        max_retries = 10  # Limit retries to avoid infinite loops
        while retries < max_retries:
            try:
                queue_.put(command, timeout=0.1)
                return
            except queue.Full:
                retries += 1
                # If queue is full, workers may be blocked on semaphores
                # Try draining results to release semaphores
                if state is not None and context is not None:
                    try:
                        # Drain a few results to release semaphores
                        for _ in range(3):  # Drain up to 3 results
                            try:
                                item = self._queue_get_nowait(state.result_queue)
                                self._handle_result(state, item, None, context)
                            except queue.Empty:
                                break
                    except Exception:
                        # If draining fails, continue retrying
                        pass
                continue
        # If we've exhausted retries, try one more time without timeout
        # This will block indefinitely, but at least we tried to drain first
        _debug(
            f"WARNING: Queue still full after {max_retries} retries with draining. "
            + "Falling back to blocking put() - this may indicate a deadlock."
        )
        queue_.put(command)

    def _schedule_batch(
        self,
        state: _ProcessOperatorState,
        batch: list[RunnerStreamIn],
        *,
        wait_ns: int,
        context: ConcurrentRunContext,
    ) -> None:
        if not batch or context.stop_event.is_set():
            return
        if not self._single_op_direct_ipc and len(self.ops) > 1 and state.op_index == 0:
            next_queue = self._next_queue_for(state)
            self._schedule_chunked_batch(
                state,
                batch,
                wait_ns=wait_ns,
                context=context,
                next_queue=next_queue,
            )
            return
        self._schedule_worker_batch(
            state,
            batch,
            wait_ns=wait_ns,
            context=context,
        )

    def _schedule_worker_batch(
        self,
        state: _ProcessOperatorState,
        batch: list[RunnerStreamIn],
        *,
        wait_ns: int,
        context: ConcurrentRunContext,
    ) -> None:
        if not batch or context.stop_event.is_set():
            return
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
        task_queue = state.task_queue
        if task_queue is None:
            raise RuntimeError("Process runner has no worker queue")
        command = _WorkerCommand(
            kind="batch",
            seq=seq,
            batch=batch,
            wait_ns=wait_ns if collect_stats else 0,
            consumed_elements=consumed_elements,
            consumed_bytes=consumed_bytes,
            queue_depth_snapshot=queue_depth_snapshot,
            collect_metrics=collect_stats,
        )
        self._send_command(task_queue, command, state=state, context=context)
        _debug(f"scheduled batch seq={seq}")
        state.inflight.increment()

    def _ack_result(
        self,
        state: _ProcessOperatorState,
        result: RunnerResult,
        context: ConcurrentRunContext | None,
    ) -> None:
        if result.ack is None:
            return
        idx = int(result.ack)
        if 0 <= idx < len(state.result_semaphores):
            state.result_semaphores[idx].release()

    def _handle_result(
        self,
        state: _ProcessOperatorState,
        result: RunnerResult,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
    ) -> None:
        _debug(
            f"handle_result seq={result.seq} error={result.error} "
            + f"payload={len(result.payload)}"
        )
        state.inflight.decrement()
        super()._handle_result(state, result, next_queue, context)

    def _finalize_state(
        self,
        state: _ProcessOperatorState,
        context: ConcurrentRunContext,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
    ) -> Microbatch:
        outputs: Microbatch = []
        task_queue = state.task_queue
        if task_queue is None:
            return outputs
        worker_count = len(state.workers)
        for _ in range(worker_count):
            seq = state.next_seq
            state.next_seq += 1
            command = _WorkerCommand(
                kind="finalize",
                seq=seq,
                batch=[],
                wait_ns=0,
                consumed_elements=0,
                consumed_bytes=0,
                queue_depth_snapshot=-1,
                collect_metrics=False,
            )
            self._send_command(task_queue, command, state=state, context=context)
            result = self._wait_for_result(state, seq, context, next_queue)
            if result.payload:
                outputs.extend(result.payload)
            self._ack_result(state, result, context)
        return outputs

    def _wait_for_result(
        self,
        state: _ProcessOperatorState,
        seq: int,
        context: ConcurrentRunContext,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
    ) -> RunnerResult:
        while True:
            item = state.result_queue.get()
            _debug(f"_wait_for_result saw seq={item.seq}")
            if item.error is not None:
                exc = WorkerCrashed(item.error)
                self._record_error(context, exc)
                self._ack_result(state, item, context)
                raise exc
            if item.seq == seq:
                return item
            # Unexpected payload; funnel through the normal path
            self._handle_result(state, item, next_queue, context)

    def set_parallelism(self, op_index: int, new_parallelism: int) -> None:
        raise NotImplementedError("ProcessStageRunner does not support live scaling")

    def close(self) -> None:
        with self._context_lock:
            ctx = self._active_context

        if ctx is not None:
            ctx.stop_event.set()
            self._put_stage_stop(ctx)
            self._join_threads(ctx)

        self._shutdown_workers()
        self._stop_service_thread()
