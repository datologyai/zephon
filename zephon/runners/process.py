"""Process-based stage runner that executes ops using multiprocessing workers.

Lambda function support
-----------------------
This runner uses cloudpickle to serialize operators before sending them to worker
processes. This enables operators to contain lambda functions, closures, and nested
functions that standard pickle cannot handle. The serialization is done surgically:
only the operator is serialized with cloudpickle, while the rest of the multiprocessing
infrastructure uses standard pickle to avoid compatibility issues.

Debugging hangs and crashes
---------------------------
Set ZEPHON_FAULTHANDLER=1 to enable faulthandler, which will dump all thread stacks
on SIGSEGV, SIGFPE, SIGABRT, SIGBUS, SIGILL crashes and on SIGUSR1 (for manual trigger).
Set ZEPHON_DEBUG_SHUTDOWN=1 to enable detailed shutdown logging.
Set ZEPHON_SHUTDOWN_WATCHDOG=<seconds> to dump thread stacks if shutdown takes too long.
"""

from __future__ import annotations

import copy
import gc
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import queues as mp_queues
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Semaphore
from typing import Any, Callable, Iterable, Literal, Protocol, Sequence, TypeVar, cast

import cloudpickle

from zephon.utils import (
    SafeSemLock,
    cleanup_semaphores,
    collect_with_finalizers,
    dump_semaphore_registry,
)
from zephon.utils.fault_handling import ShutdownWatchdog, setup_faulthandler

# Initialize faulthandler at module load time
setup_faulthandler()

from zephon.core.constants import (
    RunnerStageIn,
    RunnerStageOut,
    RunnerStreamIn,
    StreamItem,
)
from zephon.core.graph import Node, Stage
from zephon.core.op_base import OpContext
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
_SHUTDOWN_DEBUG = bool(os.environ.get("ZEPHON_DEBUG_SHUTDOWN"))
_SHUTDOWN_WATCHDOG_TIMEOUT = float(os.environ.get("ZEPHON_SHUTDOWN_WATCHDOG", "0"))


def _debug(msg: str) -> None:  # pragma: no cover - diagnostics helper
    if _DEBUG:
        print(f"[ProcessRunner] {msg}", file=sys.stderr, flush=True)


def _shutdown_debug(msg: str) -> None:  # pragma: no cover - diagnostics helper
    """Debug logging specifically for shutdown paths."""
    if _SHUTDOWN_DEBUG:
        ts = time.strftime("%H:%M:%S", time.localtime())
        ms = int((time.time() % 1) * 1000)
        print(f"[Shutdown {ts}.{ms:03d}] {msg}", file=sys.stderr, flush=True)


@dataclass(frozen=True, slots=True)
class _FeederError:
    """Sentinel enqueued when the feeder thread fails to serialize an item.

    Placed on a queue's internal ``_buffer`` when serialization fails (e.g.
    ``/dev/shm`` exhaustion with torch tensors).  Contains only strings so it
    serializes without touching ``/dev/shm``.  The consumer's ``get()`` override
    checks for it and raises :class:`QueueFeederError`.
    """

    queue_name: str
    traceback: str


class QueueFeederError(RuntimeError):
    """Raised when a queue's background feeder thread fails to serialize an item.

    CPython's ``Queue._feed`` thread silently drops items on serialization
    failure.  This exception surfaces that failure with the original traceback
    so the pipeline can fail loudly instead of silently losing batches.
    """


# Timeouts for the escalating retry in _on_queue_feeder_error.
_FEEDER_SHORT_TIMEOUT = 5.0
_FEEDER_LONG_TIMEOUT = 120.0

# Interval between gc.collect() calls in worker processes (nanoseconds).
# Time-based rather than batch-count so it adapts to both fast workers
# (many short batches) and slow workers (few long batches).
_WORKER_GC_INTERVAL_NS = 60_000_000_000  # 60 seconds


class _NamedQueue(mp_queues.Queue):
    """Queue that tags its feeder thread with a friendly name.

    Uses SafeSemLock for all internal semaphores to ensure proper cleanup in
    free-threaded Python where GC finalizers run in background threads and can
    race with explicit cleanup.

    Overrides ``_on_queue_feeder_error`` so that serialization failures in the
    background feeder thread (e.g. ``/dev/shm`` exhaustion when pickling torch
    tensors) are re-enqueued as :class:`_FeederError` sentinels through the
    queue's own pipe.  The ``get()`` override detects these and raises
    :class:`QueueFeederError` with the full original traceback.
    """

    def __init__(self, name: str, maxsize: int = 0, *, ctx: BaseContext):
        super().__init__(maxsize, ctx=ctx)
        self._ignore_epipe = True
        self._name_label = name

        # Wrap all semaphores with SafeSemLock for coordinated cleanup
        # in free-threaded Python where GC finalizers run in background threads
        self._sem = SafeSemLock.wrap(self._sem, source=f"queue:{name}:_sem")
        self._rlock = SafeSemLock.wrap(self._rlock, source=f"queue:{name}:_rlock")
        self._wlock = SafeSemLock.wrap(self._wlock, source=f"queue:{name}:_wlock")

    # -- Feeder error detection ---------------------------------------------

    def _on_queue_feeder_error(self, e: BaseException, obj: object) -> None:
        """Called by CPython's ``Queue._feed`` thread on serialization failure.

        Re-enqueues a :class:`_FeederError` sentinel through the queue's own
        internal buffer so the consumer's ``get()`` raises
        :class:`QueueFeederError` with the full original traceback.

        The ``_feed`` thread survives serialization errors (it only exits on
        EPIPE or process shutdown), so the sentinel is picked up on the next
        loop iteration and sent through the pipe like any normal item.

        Uses an escalating timeout to ensure the error reaches the consumer:

        1. Short wait (5 s) — should succeed immediately since ``_feed``
           already released a semaphore slot.
        2. Warn to stderr and block longer (120 s).
        3. Give up — log aggressively to stderr.
        """
        import traceback as tb_mod

        tb_str = tb_mod.format_exc()
        sentinel = _FeederError(self._name_label, tb_str)
        try:
            # Stage 1: short wait — _feed already released a semaphore slot.
            if self._sem.acquire(block=True, timeout=_FEEDER_SHORT_TIMEOUT):
                with self._notempty:
                    self._buffer.append(sentinel)
                    self._notempty.notify()
                return

            # Stage 2: warn and block longer.
            print(
                f"WARNING: Queue '{self._name_label}' feeder thread failed to "
                f"serialize an item and cannot re-enqueue the error sentinel "
                f"(queue full for {_FEEDER_SHORT_TIMEOUT}s). Retrying for "
                f"{_FEEDER_LONG_TIMEOUT}s before dropping.\n{tb_str}",
                file=sys.stderr,
                flush=True,
            )
            if self._sem.acquire(block=True, timeout=_FEEDER_LONG_TIMEOUT):
                with self._notempty:
                    self._buffer.append(sentinel)
                    self._notempty.notify()
                return

            # Stage 3: give up after 2+ minutes — log aggressively.
            msg = (
                f"\n{'!' * 72}\n"
                f"CRITICAL: Queue '{self._name_label}' feeder thread DROPPED "
                f"an item after failing to serialize it AND failing to enqueue "
                f"the error sentinel for "
                f"{_FEEDER_SHORT_TIMEOUT + _FEEDER_LONG_TIMEOUT}s.\n"
                f"Training results may be SILENTLY INCORRECT.\n"
                f"Original error:\n{tb_str}"
                f"\n{'!' * 72}\n"
            )
            print(msg, file=sys.stderr, flush=True)
        except Exception:
            # Last resort — something is deeply wrong.
            print(
                f"CRITICAL: Queue '{self._name_label}' feeder error AND failed "
                f"to report it:\n{tb_str}",
                file=sys.stderr,
                flush=True,
            )

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        """Retrieve an item, raising on feeder-thread errors."""
        item = super().get(block, timeout)
        if isinstance(item, _FeederError):
            raise QueueFeederError(
                f"Queue '{item.queue_name}' feeder thread failed to serialize "
                f"an item (likely /dev/shm exhaustion).  The item was silently "
                f"dropped by CPython's Queue._feed thread.\n\n"
                f"Original traceback from feeder thread:\n{item.traceback}"
            )
        return item

    # -- Thread naming & lifecycle ------------------------------------------

    def _start_thread(self) -> None:
        super()._start_thread()  # type: ignore[attr-defined]
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.name = f"QueueFeederThread[{self._name_label}]"

    def close(self) -> None:
        """Close the queue and clean up all internal semaphores."""
        try:
            super().close()
        finally:
            self._sem.cleanup()
            self._rlock.cleanup()
            self._wlock.cleanup()

    # -- Pickling (process spawning only) -----------------------------------

    def __getstate__(self) -> Any:  # noqa: D401 - custom pickle payload
        base_state = super().__getstate__()  # type: ignore[attr-defined]
        return (self._name_label, base_state)

    def __setstate__(self, state: Any) -> None:
        self._name_label, base_state = state
        super().__setstate__(base_state)  # type: ignore[attr-defined]


@dataclass
class _WorkerCommand:
    kind: Literal["batch", "stop"]
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
    op_proto_bytes: bytes  # Serialized with cloudpickle to support lambdas
    stage_index: int
    stage_name: str
    op_index: int
    collect_stats: bool
    ctx_services: dict[str, Any]
    task_queue: _QueueLike[_WorkerCommand]
    result_queue: _QueueLike[RunnerResult]
    backpressure: Semaphore | SafeSemLock


QueueFactory = Callable[..., _ClosableQueue[Any]]
SemaphoreFactory = Callable[..., Semaphore | SafeSemLock]
ProcessFactory = Callable[..., BaseProcess]


def _process_worker_main(config: _ProcessWorkerConfig) -> None:
    _debug(f"worker[{config.worker_index}] starting")
    try:
        # Deserialize operator using cloudpickle to support lambdas/closures
        op_proto = cloudpickle.loads(config.op_proto_bytes)
        op_instance = copy.deepcopy(op_proto)
        # Free the deserialized prototype immediately.  With spawn/forkserver
        # each worker gets its own cloudpickle.loads() result so the deepcopy
        # above already produced an independent instance.  Without this del,
        # op_proto (which includes everything captured in the operator closure —
        # tokenizers, transform functions, etc.) stays alive for the entire
        # worker lifetime as an unused local variable.
        del op_proto
        ctx = OpContext(dict(config.ctx_services))
        op_instance.setup(
            ctx,
            config.stage_index,
            config.stage_name,
            config.op_index,
            config.collect_stats,
        )
        _last_gc_ns = time.monotonic_ns()
        while True:
            command = config.task_queue.get()
            if command.kind == "stop":
                _debug(f"worker[{config.worker_index}] received stop, exiting cleanly")
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

            # Periodic GC: reclaim cyclic garbage (PyArrow internals, torch
            # storage objects, orphaned closures) and let the allocator
            # consolidate freed pages.  Time-based rather than batch-count
            # to adapt to both fast workers (many short batches) and slow
            # workers (few long batches).  The monotonic_ns() check is
            # essentially free (~20 ns vdso call).
            now_ns = time.monotonic_ns()
            if now_ns - _last_gc_ns >= _WORKER_GC_INTERVAL_NS:
                gc.collect()
                _last_gc_ns = now_ns
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
    result_semaphores: list[Semaphore | SafeSemLock] = field(
        init=False, default_factory=list
    )
    worker_ids: list[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.input_queue = queue.Queue[Sequence[RunnerStreamIn] | StopToken](
            maxsize=self.queue_capacity
        )
        # note that this is just a temporary q regular queue that we will replace with a proper mp.Queue
        self.result_queue = queue.Queue[RunnerResult](maxsize=1)


def _close_reader_end(q: Any) -> None:
    conn = getattr(q, "_reader", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def _close_writer_end(q: Any) -> None:
    conn = getattr(q, "_writer", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


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
        (kinds: ``"batch"``, ``"stop"``),
      - runs ``process_many`` / ``process_one`` on its local
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

    Invocation boundaries
    ---------------------

    Invocation boundaries are defined by the operator's accumulator, which
    runs on the pump thread. The runner sends each ready batch from the
    accumulator to workers without further splitting or chunking. This ensures
    that Thread and Process runners execute identical batch boundaries for
    deterministic execution.

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
      IPC batches to workers, and drives result draining itself via
      :meth:`_drain_until_idle`.
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
        deterministic: bool = False,
        allow_latency_flush_in_deterministic: bool = True,
        stage_index: int = 0,
        tracking_mode: ExecutionTrackingMode = ExecutionTrackingMode.OFF,
        stage_output_mode: Literal["microbatches", "stream_items"] = "microbatches",
        mp_context: BaseContext | None = None,
    ) -> None:
        self._queue_capacity = max(1, queue_capacity)
        ctx = mp_context or mp.get_context("spawn")
        self._mp_context: BaseContext = ctx
        # Use SafeSemLock for coordinated cleanup in free-threaded Python
        semaphore_factory: SemaphoreFactory = lambda value=1: SafeSemLock.new_semaphore(
            value, ctx=ctx
        )
        process_factory = cast(ProcessFactory, getattr(ctx, "Process"))
        self._semaphore_factory: SemaphoreFactory = semaphore_factory
        self._process_factory: ProcessFactory = process_factory
        self._service_queue = cast(
            _ClosableQueue[_ServiceRequest | None], self._make_ipc_queue("service")
        )
        self._service_thread: threading.Thread | None = None
        self._service_stop = threading.Event()
        self._service_responses: dict[int, _ClosableQueue[tuple[bool, Any]]] = {}
        self._next_worker_id = 0
        self._single_op_direct_ipc = len(stage.nodes) == 1
        self._shutdown_lock = threading.Lock()
        self._workers_shutdown = False
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
        self._log_debug_config()

    def _log_debug_config(self) -> None:
        """Log which debugging features are enabled at initialization."""
        features: list[str] = []
        if _DEBUG:
            features.append("ZEPHON_DEBUG_PROCESS_RUNNER")
        if _SHUTDOWN_DEBUG:
            features.append("ZEPHON_DEBUG_SHUTDOWN")
        if _SHUTDOWN_WATCHDOG_TIMEOUT > 0:
            features.append(f"ZEPHON_SHUTDOWN_WATCHDOG={_SHUTDOWN_WATCHDOG_TIMEOUT}s")
        if os.environ.get("ZEPHON_FAULTHANDLER"):
            features.append("ZEPHON_FAULTHANDLER")

        if features:
            print(
                f"[ProcessRunner] Debug features enabled: {', '.join(features)}",
                file=sys.stderr,
                flush=True,
            )

    def _make_ipc_queue(
        self, name: str, maxsize: int | None = None
    ) -> _ClosableQueue[Any]:
        size = 0 if maxsize is None else maxsize
        return cast(
            _ClosableQueue[Any],
            _NamedQueue(name, maxsize=size, ctx=self._mp_context),
        )

    @staticmethod
    def _close_ipc_queue(q: Any) -> None:
        if q is None:
            return

        try:
            q.close()
        except Exception:
            name = "unnamed" if not isinstance(q, _NamedQueue) else q._name_label
            print(
                f"Error closing queue {name!r}.",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)

        t = getattr(q, "_thread", None)
        if t is not None:
            try:
                t.join(timeout=5)
            except Exception:
                pass

            if t.is_alive():
                print(
                    "Queue failed to flush within 5 seconds. "
                    + "Force-closing pipe to unblock feeder thread (data loss possible).",
                    file=sys.stderr,
                )
                _close_writer_end(q)
                _close_reader_end(q)
                try:
                    q.cancel_join_thread()
                except Exception:
                    pass
                try:
                    t.join(timeout=0.5)
                except Exception:
                    pass
        else:
            try:
                q.cancel_join_thread()
            except Exception:
                pass

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
        # Reset shutdown flag for runner reuse (start -> stop -> start)
        self._workers_shutdown = False
        for state in self.ops:
            self._launch_workers(state)
        self._start_service_thread()

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
            _shutdown_debug("_stop_service_thread: no service thread")
            return
        _shutdown_debug("_stop_service_thread: setting stop flag")
        self._service_stop.set()
        try:
            self._service_queue.put(None)
        except (FileNotFoundError, EOFError, OSError, ValueError):
            pass  # i think we right now do this multiple times but anyways

        _shutdown_debug("_stop_service_thread: joining service thread")
        self._service_thread.join(timeout=10.0)
        if self._service_thread.is_alive():
            _shutdown_debug(
                "_stop_service_thread: WARNING - service thread still alive after 10s join"
            )
        self._service_thread = None
        _shutdown_debug("_stop_service_thread: closing response queues")
        for resp in self._service_responses.values():
            self._close_ipc_queue(resp)
        self._service_responses.clear()
        _shutdown_debug("_stop_service_thread: done")

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
                    # The for-loop calls next(upstream) before we can check stop_event,
                    # and next() may block indefinitely. Check here to exit early.
                    if context.stop_event.is_set():
                        break

                ready = state.enqueue([], force=True)
                self._dispatch_ipc_batches(state, ready, context)

                self._drain_until_idle(state, context)
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
        """Dispatch ready batches to workers without splitting.

        Invocation boundaries are defined by the operator's accumulator.
        Each ready batch is sent to workers as-is to ensure deterministic
        execution across Thread and Process runners.

        This method is only used for single-operator stages using the direct
        IPC path (see _spawn_ipc_feeder). For single-operator stages, there is
        no next operator, so next_queue=None is passed to _drain_results,
        which causes results to go directly to the stage output queue.
        Multi-operator stages use _operator_loop which calculates the correct
        next_queue via _next_queue_for.
        """
        if not ready:
            return
        for batch, wait_ns in ready:
            if not batch:
                continue
            if context.stop_event.is_set():
                break
            self._schedule_batch(
                state,
                batch,
                wait_ns=wait_ns,
                context=context,
            )
            # next_queue=None: results go to stage output (correct for single-op stages)
            self._drain_results(state, None, context)

    def _next_queue_for(
        self, state: _ProcessOperatorState
    ) -> _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None:
        next_index = state.op_index + 1
        if next_index < len(self.ops):
            return self.ops[next_index].input_queue
        return None

    def _drain_until_idle(
        self,
        state: _ProcessOperatorState,
        context: ConcurrentRunContext,
    ) -> None:
        while True:
            if (
                state.inflight.is_zero()
                and state.pending_puts.is_zero()
                and not state.pending_results
            ):
                break
            if context.stop_event.is_set():
                # Shutdown requested; don't wait forever for dead workers.
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

        op_label = state.node.name or f"op{state.op_index}"
        queue_label = f"{state.stage_name}:{op_label}"

        result_queue = cast(
            _ClosableQueue[RunnerResult],
            self._make_ipc_queue(f"result:{queue_label}"),
        )
        state.result_queue = result_queue
        queue_capacity = max(1, self._queue_capacity * max(1, state.parallelism))
        task_queue = cast(
            _ClosableQueue[_WorkerCommand],
            self._make_ipc_queue(f"task:{queue_label}", queue_capacity),
        )
        state.task_queue = task_queue

        # Serialize operator once using cloudpickle to support lambdas/closures
        op_proto_bytes = cloudpickle.dumps(state.node.op)

        # Phase 1: Create all worker configs and Process objects
        worker_infos: list[
            tuple[
                int,
                int,
                BaseProcess,
                Semaphore | SafeSemLock,
                _ClosableQueue[tuple[bool, Any]],
            ]
        ] = []

        for idx in range(state.parallelism):
            semaphore = self._semaphore_factory(self._queue_capacity)
            worker_id = self._next_worker_id
            resp_queue = cast(
                _ClosableQueue[tuple[bool, Any]],
                self._make_ipc_queue(
                    f"service-response:{queue_label}:{worker_id}",
                ),
            )
            self._next_worker_id += 1
            ctx_payload = self._build_worker_ctx(worker_id, resp_queue)
            config = _ProcessWorkerConfig(
                worker_index=idx,
                worker_id=worker_id,
                op_proto_bytes=op_proto_bytes,
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
            worker_infos.append((idx, worker_id, proc, semaphore, resp_queue))

        # Phase 2: Start all processes.
        # For spawn/forkserver, we start processes in parallel using threads since
        # proc.start() blocks until the process is forked. This reduces spawn time
        # from O(n * spawn_time) to O(spawn_time).
        # For fork, we start sequentially because fork() in a multithreaded program
        # is unsafe: the child inherits locks held by threads that no longer exist,
        # leading to potential deadlocks. Concurrent fork() calls exacerbate this.
        start_method = self._mp_context.get_start_method()
        if start_method in ("spawn", "forkserver"):
            spawn_threads: list[threading.Thread] = []
            for _, _, proc, _, _ in worker_infos:
                t = threading.Thread(target=proc.start, daemon=True)
                t.start()
                spawn_threads.append(t)
            for t in spawn_threads:
                t.join()
        else:
            for _, _, proc, _, _ in worker_infos:
                proc.start()

        # Phase 3: Bookkeeping after all processes have started
        for idx, worker_id, proc, semaphore, resp_queue in worker_infos:
            _debug(f"started worker process idx={idx} pid={proc.pid}")
            self._service_responses[worker_id] = resp_queue
            state.worker_ids.append(worker_id)
            state.result_semaphores.append(semaphore)
            state.workers.append(proc)
            # main -> worker (service response queue): main is producer-only
            _close_reader_end(resp_queue)

        # main -> workers (task queue): main is producer-only
        _close_reader_end(task_queue)

        # workers -> main (result queue): main is consumer-only
        _close_writer_end(result_queue)

    def _shutdown_workers(self) -> None:
        with self._shutdown_lock:
            if self._workers_shutdown:
                _shutdown_debug("_shutdown_workers: already complete, skipping")
                return
            self._workers_shutdown = True
        _shutdown_debug(f"_shutdown_workers: starting for {len(self.ops)} ops")
        for op_idx, state in enumerate(self.ops):
            task_queue = state.task_queue
            if task_queue is None:
                _shutdown_debug(
                    f"_shutdown_workers: op[{op_idx}] has no task_queue, skipping"
                )
                continue

            # First, try to drain any pending results before sending stop commands
            # This helps avoid deadlocks where workers are blocked on semaphores
            # and can't process stop commands. We just need to acknowledge results
            # to release semaphores - we don't need to process them fully.
            _shutdown_debug(
                f"_shutdown_workers: op[{op_idx}] draining results for "
                + f"{len(state.workers)} workers, inflight={state.inflight._count}"
            )
            drain_start = time.time()
            drain_timeout = 5.0  # Give up after 5 seconds of draining
            drained_total = 0
            while time.time() - drain_start < drain_timeout:
                drained_any = False
                try:
                    # Drain results with timeout - get() returns immediately if items available
                    while True:
                        try:
                            item = self._queue_get(state.result_queue, timeout=0.1)
                        except (FileNotFoundError, EOFError, OSError, ValueError):
                            # The result queue's underlying fd may vanish if workers
                            # die abruptly (torch shared memory handles). At shutdown
                            # we just stop draining and proceed with tear-down.
                            _shutdown_debug(
                                f"_shutdown_workers: op[{op_idx}] result_queue get() "
                                + "failed; worker likely exited"
                            )
                            break
                        # Just acknowledge to release semaphore - don't process fully
                        self._ack_result(state, item, None)
                        # Use atomic try_decrement to avoid TOCTOU race: a pump thread
                        # (still alive after join timeout) may have dequeued an item
                        # before shutdown and decrement between our check and decrement.
                        # Also handles error results from startup crashes (no increment).
                        state.inflight.try_decrement()
                        drained_any = True
                        drained_total += 1
                except queue.Empty:
                    pass

                if not drained_any and state.inflight.is_zero():
                    break

            _shutdown_debug(
                f"_shutdown_workers: op[{op_idx}] drain done, drained={drained_total}, "
                + f"inflight={state.inflight._count}, elapsed={time.time() - drain_start:.2f}s"
            )

            # Now send stop commands to all workers
            # Note: We don't pass context here because we're in shutdown
            # and the draining above should have released semaphores already
            _shutdown_debug(
                f"_shutdown_workers: op[{op_idx}] sending stop to {len(state.workers)} workers"
            )
            for worker_idx, _ in enumerate(state.workers):
                self._send_command(
                    task_queue,
                    _WorkerCommand("stop", -1, [], 0, 0, 0, -1, False),
                    state=None,  # Don't drain during shutdown - already drained above
                    context=None,
                    block_on_exhaustion=False,
                )

            # Wait for workers to finish, with reasonable timeout
            # We don't block indefinitely - if workers don't terminate quickly,
            # they're likely hung and we log a warning
            _shutdown_debug(f"_shutdown_workers: op[{op_idx}] joining workers")
            for worker_idx, proc in enumerate(state.workers):
                _shutdown_debug(
                    f"_shutdown_workers: op[{op_idx}] joining worker[{worker_idx}] pid={proc.pid}"
                )
                proc.join(timeout=10.0)
                if proc.is_alive():
                    _shutdown_debug(
                        f"_shutdown_workers: op[{op_idx}] worker[{worker_idx}] did NOT exit cleanly, terminating"
                    )
                    proc.terminate()
                    # Give it a moment to die gracefully, then kill
                    proc.join(timeout=5)
                    if proc.is_alive():
                        _shutdown_debug(
                            f"_shutdown_workers: op[{op_idx}] worker[{worker_idx}] still alive, killing"
                        )
                        proc.kill()
                        proc.join()

            # Atomically clear any remaining inflight count
            _shutdown_debug(
                f"_shutdown_workers: op[{op_idx}] force_zero (was {state.inflight._count})"
            )
            state.inflight.force_zero()

            # The workers are dead; they will never fill the sequence gaps.
            # Release semaphores for pending results before clearing to avoid leaks.
            # Use list() to snapshot values: pump threads may still be modifying the dict.
            _shutdown_debug(
                f"_shutdown_workers: op[{op_idx}] clearing {len(state.pending_results)} pending results"
            )
            for pending in list(state.pending_results.values()):
                self._ack_result(state, pending, None)
            state.pending_results.clear()

            # 3. Clear runner-level buffers
            # Ensure the thread doesn't think it has batched items left to schedule.
            state.reset_buffers()

            _shutdown_debug(
                f"_shutdown_workers: op[{op_idx}] closing {len(state.worker_ids)} response queues"
            )
            for wid in state.worker_ids:
                resp = self._service_responses.pop(wid, None)
                self._close_ipc_queue(resp)

            state.workers.clear()
            state.worker_ids.clear()
            # In GIL-free Python, semaphore __del__ finalizers are deferred and may
            # not run before the resource tracker checks at shutdown. Explicitly
            # clean up to prevent "leaked semaphore" warnings.
            cleanup_semaphores(state.result_semaphores)
            state.result_semaphores.clear()
            collect_with_finalizers()

            _shutdown_debug(
                f"_shutdown_workers: op[{op_idx}] closing task/result queues"
            )
            self._close_ipc_queue(state.task_queue)
            self._close_ipc_queue(state.result_queue)

            state.task_queue = None

        _shutdown_debug("_shutdown_workers: closing service queue")
        self._close_ipc_queue(self._service_queue)
        # Final cleanup: more aggressive GC for any remaining semaphores.
        collect_with_finalizers(cycles=5, yield_ms=2.0)
        # Dump semaphore debug info if ZEPHON_SEMAPHORE_DEBUG=1
        dump_semaphore_registry()
        _shutdown_debug("_shutdown_workers: done")

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
        block_on_exhaustion: bool = True,
    ) -> None:
        """Send command to worker queue, draining results if queue is full.

        If the queue is full, it means workers aren't consuming commands.
        This is often because workers are blocked on semaphores waiting to send results.
        By draining results, we release semaphores and allow workers to proceed.
        """
        retries = 0
        max_retries = 10
        warned = False
        while block_on_exhaustion or retries < max_retries:
            try:
                queue_.put(command, timeout=0.1)
                return
            except ValueError:
                # Queue was closed - only silently return if we're in shutdown
                if self._workers_shutdown:
                    return
                raise  # Re-raise if not in shutdown - this indicates a real bug
            except queue.Full:
                retries += 1
                if retries == max_retries and not warned:
                    warned = True
                    if command.kind != "batch":
                        print(
                            f"WARNING: Queue still full after {max_retries} retries "
                            + f"with draining; will continue retrying. Command: {command}",
                            file=sys.stderr,
                        )
                # If queue is full, workers may be blocked on semaphores
                # Try draining results to release semaphores
                if state is not None and context is not None:
                    try:
                        # Drain a few results to release semaphores
                        next_queue = self._next_queue_for(state)
                        for _ in range(3):
                            try:
                                item = self._queue_get_nowait(state.result_queue)
                                self._handle_result(state, item, next_queue, context)
                            except queue.Empty:
                                break
                        if context.stop_event.is_set():
                            return
                    except Exception:
                        # If draining fails, continue retrying
                        pass
                continue
        # Only reached when block_on_exhaustion=False
        print(
            f"WARNING: Queue still full after {max_retries} retries with draining. "
            + f"Skipping command {command}.",
            file=sys.stderr,
        )

    def _schedule_batch(
        self,
        state: _ProcessOperatorState,
        batch: list[RunnerStreamIn],
        *,
        wait_ns: int,
        context: ConcurrentRunContext,
    ) -> None:
        """Schedule a batch for worker execution without splitting.

        Invocation boundaries are defined by the operator's accumulator.
        Each batch is sent to workers as-is.
        """
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
        # Use try_decrement for all results: during shutdown, force_zero() may have
        # already cleared the counter while pump threads are still processing results.
        # This races when buffered_iterable's 1s join timeout expires before
        # _join_threads completes, allowing _shutdown_workers to run concurrently.
        state.inflight.try_decrement()
        super()._handle_result(state, result, next_queue, context)

    def _wait_for_result(
        self,
        state: _ProcessOperatorState,
        seq: int,
        context: ConcurrentRunContext,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
    ) -> RunnerResult:
        while True:
            try:
                item = self._queue_get(state.result_queue, timeout=0.1)
            except queue.Empty:
                if context.stop_event.is_set():
                    # Shutdown requested - return empty result to allow graceful exit.
                    # This is not an error: workers are being stopped via stop commands
                    # and the feeder should complete cleanly.
                    return RunnerResult(
                        seq=seq,
                        payload=[],
                        wait_ns=0,
                        consumed_elements=0,
                        consumed_bytes=0,
                        queue_depth_snapshot=-1,
                        proc_ns=0,
                        collect_metrics=False,
                        ack=None,
                        error=None,
                    )
                continue
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
        _shutdown_debug("close() called")
        with ShutdownWatchdog(_SHUTDOWN_WATCHDOG_TIMEOUT, "close()"):
            with self._context_lock:
                ctx = self._active_context

            if ctx is not None:
                _shutdown_debug("close(): setting stop_event")
                ctx.stop_event.set()
                self._put_stage_stop(ctx)

            _shutdown_debug("close(): calling _shutdown_workers")
            self._shutdown_workers()
            _shutdown_debug("close(): calling _stop_service_thread")
            self._stop_service_thread()

            if ctx is not None:
                _shutdown_debug("close(): calling _join_threads")
                self._join_threads(ctx)
            _shutdown_debug("close(): done")
