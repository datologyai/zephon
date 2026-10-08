"""Process-based stage runner that executes ops using multiprocessing workers.

Lambda function support
-----------------------
This runner uses cloudpickle to serialize operators before sending them to worker
processes. This enables operators to contain lambda functions, closures, and nested
functions that standard pickle cannot handle. The serialization is done surgically:
only the operator is serialized with cloudpickle, while the rest of the multiprocessing
infrastructure uses standard pickle to avoid compatibility issues.

Resilience (worker respawn + bounded retry)
-------------------------------------------
When ``max_worker_retries > 0`` (default 3 via ``RuntimeOptions``) a daemon
watchdog respawns workers killed by signals and re-dispatches their
in-flight work. Only the seq the worker was actively running at the
moment of death (the *culprit*) accrues a retry; bystanders whose
results were lost in the dead worker's outbox are re-dispatched for
free. On retry exhaustion the pipeline either fails (deterministic
mode) or drops the seq with a loud stderr warning (non-deterministic
mode). Resilient mode requires ``mp_context`` with ``spawn`` or
``forkserver`` — the constructor raises ``ValueError`` otherwise.

Debugging hangs and crashes
---------------------------
Set ZEPHON_FAULTHANDLER=1 to enable faulthandler, which will dump all thread stacks
on SIGSEGV, SIGFPE, SIGABRT, SIGBUS, SIGILL crashes and on SIGUSR1 (for manual trigger).
Set ZEPHON_DEBUG_SHUTDOWN=1 to enable detailed shutdown logging.
Set ZEPHON_DEBUG_WORKER_STARTUP=1 to log per-worker startup timing, RSS, and module inventory.
Set ZEPHON_SHUTDOWN_WATCHDOG=<seconds> to dump thread stacks if shutdown takes too long.
Set ZEPHON_WATCHDOG_POLL_S=<seconds> to override the resilient-worker watchdog poll interval (default 5.0).
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
import warnings
from dataclasses import dataclass, field
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Semaphore
from typing import Any, Callable, Iterable, Literal, Protocol, Sequence, TypeVar, cast

import cloudpickle

from zephon._internal.utils import (
    SafeSemLock,
    cleanup_semaphores,
    collect_with_finalizers,
    dump_semaphore_registry,
)
from zephon._internal.utils.fault_handling import ShutdownWatchdog, setup_faulthandler
from zephon._internal.utils.ipc import DEFAULT_IPC_BUFFER_BYTES, DEFAULT_IPC_TRANSPORT
from zephon._internal.utils.rank import rank_ctx
from zephon.options import IpcTransport

# Initialize faulthandler at module load time.  Also runs when this module
# is re-imported inside spawned workers so C-level crashes there produce
# tracebacks to stderr.  Idempotent via ``faulthandler.is_enabled()``.
setup_faulthandler()

from zephon._internal.flush_context import shutdown_flush
from zephon._internal.graph import Node, Stage
from zephon._internal.notify import is_sentinel
from zephon._internal.observability.size_estimator import estimate_bytes
from zephon._internal.runners.concurrent import (
    ConcurrentRunContext,
    RunnerResult,
    StopToken,
    WorkerCrashed,
    WorkerErrorInfo,
    _QueueLike,
)
from zephon._internal.runners.queue_drain import (
    QueueDrainOperatorState,
    QueueDrainStageRunner,
)
from zephon._internal.runners.watchdog import (
    close_abandoned_result_queues,
    recover_result_queue,
)
from zephon._internal.stream import (
    Microbatch,
    RunnerStageIn,
    RunnerStreamIn,
    resolve_lazy_payloads,
)
from zephon._internal.utils.shm_coalesce import (
    DEFAULT_SHM_MIN_SIZE,
    coalesce_microbatch,
    forward_shared_numpy,
)
from zephon.observability.config import ExecutionTrackingMode
from zephon.ops.base import OpContext, StageInfo

Q = TypeVar("Q")


class _ClosableQueue(_QueueLike[Q], Protocol):
    def close(self) -> None: ...


class _SharedFlag(Protocol):
    value: int


_DEBUG = bool(os.environ.get("ZEPHON_DEBUG_PROCESS_RUNNER"))
_SHUTDOWN_DEBUG = bool(os.environ.get("ZEPHON_DEBUG_SHUTDOWN"))
_STARTUP_DEBUG = bool(os.environ.get("ZEPHON_DEBUG_WORKER_STARTUP"))
_SHUTDOWN_WATCHDOG_TIMEOUT = float(os.environ.get("ZEPHON_SHUTDOWN_WATCHDOG", "0"))
#: Watchdog poll interval in seconds.  5s default keeps steady-state
#: overhead negligible; tests override via ``ZEPHON_WATCHDOG_POLL_S``
#: to tighten resubmit-round-trip latency.
_WATCHDOG_POLL_S = float(os.environ.get("ZEPHON_WATCHDOG_POLL_S", "5.0"))
_SERVICE_POLL_S = 0.1

# -- Shutdown timeout constants (seconds) -----------------------------------
_GRACEFUL_QUEUE_FLUSH: float = 5.0
_HARD_QUEUE_FLUSH: float = 0.5
_GRACEFUL_SERVICE_JOIN: float = 10.0
_HARD_SERVICE_JOIN: float = 1.0
_GRACEFUL_DRAIN_TIMEOUT: float = 5.0
_HARD_DRAIN_TIMEOUT: float = 0.5
_GRACEFUL_WORKER_JOIN: float = 10.0
_HARD_WORKER_JOIN: float = 0.5
_GRACEFUL_TERMINATE_JOIN: float = 5.0
_HARD_TERMINATE_JOIN: float = 0.5


def _debug(msg: str) -> None:  # pragma: no cover - diagnostics helper
    if _DEBUG:
        print(f"[ProcessRunner] {msg}", file=sys.stderr, flush=True)


def _set_worker_fds_close_on_exec() -> None:
    """Best-effort close-on-exec for open worker fds above stderr.

    Exec'd helpers (e.g. torch_shm_manager) must not retain the exit pipe and
    delay join(). Multiprocessing has no public API for its child-side fd, so this
    also covers library-owned fds. Descriptors remain usable in this worker;
    the parent's flags and plain fork inheritance are unchanged. Helpers that
    need an existing fd must receive it explicitly (e.g. via pass_fds).
    """
    try:
        fds = [int(name) for name in os.listdir("/dev/fd")]
    except OSError:
        return
    for fd in fds:
        if fd <= 2:
            continue
        try:
            os.set_inheritable(fd, False)
        except OSError:
            pass  # An fd may have closed since listdir, including its own fd.


def _shutdown_debug(msg: str) -> None:  # pragma: no cover - diagnostics helper
    """Debug logging specifically for shutdown paths."""
    if _SHUTDOWN_DEBUG:
        ts = time.strftime("%H:%M:%S", time.localtime())
        ms = int((time.time() % 1) * 1000)
        print(f"[Shutdown {ts}.{ms:03d}] {msg}", file=sys.stderr, flush=True)


from zephon._internal.runners.queue import NamedQueue


def _startup_log(msg: str) -> None:  # pragma: no cover - diagnostics helper
    """Debug logging for worker startup diagnostics."""
    if _STARTUP_DEBUG:
        print(f"[WorkerStartup] {msg}", file=sys.stderr, flush=True)


def _get_rss_mb() -> float:
    """Return current process RSS in MB.

    Uses /proc/self/statm on Linux (current RSS) and resource.getrusage
    on macOS (peak RSS — macOS doesn't expose current RSS cheaply).
    """
    try:
        # Linux: /proc/self/statm field 1 = resident pages
        with open("/proc/self/statm", "r") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (FileNotFoundError, OSError, ValueError):
        pass
    # macOS fallback: ru_maxrss is in bytes
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024  # Linux ru_maxrss is in KB


def _log_worker_startup(
    worker_index: int,
    total_ns: int,
    deser_ns: int,
    setup_ns: int,
    wall_ns: int,
    rss_entry_mb: float,
    rss_deser_mb: float,
    rss_ready_mb: float,
) -> None:
    """Log per-worker startup timing and RSS breakdown."""
    total_s = total_ns / 1e9
    deser_s = deser_ns / 1e9
    setup_s = setup_ns / 1e9
    wall_s = wall_ns / 1e9
    _startup_log(
        f"worker[{worker_index}] ready in {total_s:.2f}s "
        f"(deserialize={deser_s:.2f}s, setup={setup_s:.2f}s, "
        f"wall={wall_s:.2f}s from spawn)"
    )
    _startup_log(
        f"worker[{worker_index}] RSS: entry={rss_entry_mb:.0f} MB, "
        f"post-deserialize={rss_deser_mb:.0f} MB (+{rss_deser_mb - rss_entry_mb:.0f}), "
        f"ready={rss_ready_mb:.0f} MB (+{rss_ready_mb - rss_deser_mb:.0f})"
    )


def _log_worker_modules(worker_index: int) -> None:
    """Log module inventory, separating stdlib from external packages."""
    stdlib_names = sys.stdlib_module_names
    all_mods = sorted(sys.modules.keys())

    # Classify: skip dunder/private internal modules, split by stdlib vs external
    external = sorted(
        m
        for m in all_mods
        if not m.startswith("_") and m.split(".")[0] not in stdlib_names
    )
    stdlib = sorted(
        m for m in all_mods if not m.startswith("_") and m.split(".")[0] in stdlib_names
    )

    _startup_log(
        f"worker[{worker_index}] {len(all_mods)} modules "
        f"({len(external)} external, {len(stdlib)} stdlib)"
    )


# Interval between gc.collect() calls in worker processes (nanoseconds).
# Time-based rather than batch-count so it adapts to both fast workers
# (many short batches) and slow workers (few long batches).
_WORKER_GC_INTERVAL_NS = 60_000_000_000  # 60 seconds


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


@dataclass(frozen=True)
class _RelaxSignal:
    """Watchdog→pump message: enter relaxed-backpressure mode.

    The watchdog posts this to ``state.result_queue`` when a worker dies in
    deterministic mode with in-flight items.  The pump picks it up in its
    normal drain loop and (a) post-hoc releases the backpressure permits
    held by entries already sitting in ``pending_results`` (to unblock
    workers stuck on ``backpressure.acquire()``), and (b) switches
    ``_handle_result`` to release permits on arrival rather than on emit
    until every seq in ``recovery_seqs`` has come back.  See the
    ``Resilience and recovery`` section of :class:`ProcessStageRunner` for
    the full deadlock rationale.
    """

    recovery_seqs: frozenset[int]


@dataclass
class _ServiceRequest:
    worker_id: int
    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class _RemoteServiceProxy:
    """Callable shim that forwards execution to the main process."""

    __slots__ = (
        "_worker_id",
        "_name",
        "_request_q",
        "_response_q",
        "_cancelled",
        "_pending",
    )

    def __init__(
        self,
        worker_id: int,
        name: str,
        request_queue: _ClosableQueue[_ServiceRequest | None],
        response_queue: _ClosableQueue[tuple[bool, Any]],
        cancelled: _SharedFlag,
        pending: _SharedFlag,
    ) -> None:
        self._worker_id = worker_id
        self._name = name
        self._request_q = request_queue
        self._response_q = response_queue
        self._cancelled = cancelled
        self._pending = pending

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        _debug(f"proxy[{self._worker_id}:{self._name}] enqueue request")
        parent = mp.parent_process()
        # Queue.put() only buffers: its feeder can still be serializing/writing
        # while we wait. Keep the flag raised until the reply proves delivery.
        self._pending.value = 1
        try:
            if self._cancelled.value:
                raise RuntimeError(f"Control service {self._name!r} was cancelled")
            self._request_q.put(
                _ServiceRequest(
                    worker_id=self._worker_id,
                    name=self._name,
                    args=tuple(args),
                    kwargs=dict(kwargs),
                ),
            )
            while True:
                if self._cancelled.value:
                    raise RuntimeError(f"Control service {self._name!r} was cancelled")
                if parent is not None and not parent.is_alive():
                    self._cancelled.value = 1
                    raise RuntimeError(
                        f"Control service {self._name!r} lost its parent process"
                    )
                try:
                    ok, payload = self._response_q.get(timeout=_SERVICE_POLL_S)
                    break
                except queue.Empty:
                    continue
        finally:
            self._pending.value = 0
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
    #: Per-operator shared int64 array sized to ``parallelism``.  Each worker
    #: owns slot ``worker_index`` and writes the seq it is currently
    #: processing; ``-1`` means idle / between batches.  Single-writer /
    #: single-reader (watchdog) per slot, so no lock is needed — 64-bit
    #: stores are hardware-atomic on x86/arm64.  Allocated via
    #: ``mp_context.Array('q', [-1] * parallelism, lock=False)``.
    worker_seq_slots: Any = None
    spawn_wall_ns: int = 0  # Main process wall-clock time at spawn start
    #: When True, coalesce all tensors in the microbatch into per-dtype SHM
    #: buffers before serialization.  Reduces N POSIX SHM segments to K
    #: (K = distinct dtypes, usually 1–2).
    coalesce_tensors: bool = True
    #: Minimum payload size in bytes for SHM coalescing.  Payloads smaller
    #: than this are left inline in the pickle stream.
    shm_min_size: int = DEFAULT_SHM_MIN_SIZE


QueueFactory = Callable[..., _ClosableQueue[Any]]
SemaphoreFactory = Callable[..., Semaphore | SafeSemLock]
ProcessFactory = Callable[..., BaseProcess]


def _process_worker_main(config: _ProcessWorkerConfig) -> None:
    _set_worker_fds_close_on_exec()
    startup_t0 = time.perf_counter_ns()
    rss_entry_mb = _get_rss_mb() if _STARTUP_DEBUG else 0.0
    _debug(f"worker[{config.worker_index}] starting")
    try:
        # Deserialize operator using cloudpickle to support lambdas/closures
        op_proto = cloudpickle.loads(config.op_proto_bytes)
        op_instance = copy.deepcopy(op_proto)
        deser_ns = time.perf_counter_ns() - startup_t0
        rss_deser_mb = _get_rss_mb() if _STARTUP_DEBUG else 0.0

        # Free the deserialized prototype immediately.  With spawn/forkserver
        # each worker gets its own cloudpickle.loads() result so the deepcopy
        # above already produced an independent instance.  Without this del,
        # op_proto (which includes everything captured in the operator closure —
        # tokenizers, transform functions, etc.) stays alive for the entire
        # worker lifetime as an unused local variable.
        del op_proto
        stage_info = StageInfo(
            stage_index=config.stage_index,
            stage_name=config.stage_name,
            op_index=config.op_index,
            collect_stats=config.collect_stats,
        )
        op_instance.setup(OpContext(dict(config.ctx_services), stage_info))
        total_ns = time.perf_counter_ns() - startup_t0
        setup_ns = total_ns - deser_ns
        wall_ns = time.time_ns() - config.spawn_wall_ns if config.spawn_wall_ns else 0
        rss_ready_mb = _get_rss_mb() if _STARTUP_DEBUG else 0.0

        if _STARTUP_DEBUG:
            _log_worker_startup(
                config.worker_index,
                total_ns,
                deser_ns,
                setup_ns,
                wall_ns,
                rss_entry_mb,
                rss_deser_mb,
                rss_ready_mb,
            )
            _log_worker_modules(config.worker_index)

        _last_gc_ns = time.monotonic_ns()
        slots = config.worker_seq_slots
        while True:
            command = config.task_queue.get()
            if command.kind == "stop":
                _debug(f"worker[{config.worker_index}] received stop, exiting cleanly")
                break
            if command.kind == "batch":
                # Announce seq so the watchdog can attribute a crash to
                # this batch (see ``worker_seq_slots`` field docstring).
                if slots is not None:
                    slots[config.worker_index] = command.seq
                _debug(
                    f"worker[{config.worker_index}] processing batch seq={command.seq}"
                )
                batch = command.batch
                resolve_lazy_payloads(batch)
                start_ns = time.perf_counter_ns() if command.collect_metrics else 0
                outputs = op_instance.process_many(batch)
                proc_ns = (
                    time.perf_counter_ns() - start_ns if command.collect_metrics else 0
                )

            # Optionally coalesce tensors into per-dtype SHM buffers.
            # CoalescedMicrobatch.__reduce__ unpickles as Microbatch on the
            # consumer side, so the cast is safe.
            if config.coalesce_tensors and outputs:
                coalesced = coalesce_microbatch(outputs, config.shm_min_size)
                if coalesced is not None:
                    outputs = cast(Microbatch, coalesced)

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
            # Announce "idle" after handing off the result.  If we die
            # after this point the watchdog reads -1 and blind-resubmits
            # (false-positive rate is negligible since this window is tiny).
            if slots is not None:
                slots[config.worker_index] = -1

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
class _ProcessOperatorState(QueueDrainOperatorState):
    task_queue: _QueueLike[_WorkerCommand] | None = field(init=False, default=None)
    workers: list[BaseProcess] = field(init=False, default_factory=list)
    result_semaphores: list[Semaphore | SafeSemLock] = field(
        init=False, default_factory=list
    )
    worker_ids: list[int] = field(init=False, default_factory=list)

    # Resilient-worker bookkeeping.  ``pending_commands`` is the
    # authoritative in-flight registry (populated on dispatch, popped
    # on first-time result).  ``retry_counts`` tracks per-seq attempts
    # (incremented by watchdog on resubmit, cleared on first-time
    # success).  ``worker_seq_slots`` is the SHM crash-attribution
    # array — see ``_ProcessWorkerConfig.worker_seq_slots``.
    pending_commands: dict[int, "_WorkerCommand"] = field(
        init=False, default_factory=dict
    )
    retry_counts: dict[int, int] = field(init=False, default_factory=dict)
    worker_seq_slots: Any = field(init=False, default=None)

    # Relaxed-backpressure state.  When a deterministic-mode worker dies
    # with in-flight items, the watchdog posts a ``_RelaxSignal`` onto
    # ``result_queue``.  The pump reads it and sets ``relaxed_backpressure``
    # to True while it waits for every seq in ``relaxed_backpressure_seqs``
    # to come back.  In relaxed mode, permits are released on arrival
    # instead of on emit — this lets ``pending_results`` grow past its
    # usual cap (bounded by upstream rate × recovery time) but lets the
    # emit cascade unblock naturally once the culprit is processed.  Both
    # fields are pump-owned; the watchdog only writes them indirectly via
    # the signal.
    relaxed_backpressure: bool = field(init=False, default=False)
    relaxed_backpressure_seqs: set[int] = field(init=False, default_factory=set)

    # Results produced on the pump thread that must bypass ``result_queue``.
    #
    # The main process is the consumer side of the IPC result pipe/queue.
    # Writing a result into it from the main process would deadlock when the
    # pipe buffer is full (the reader — also the main process — is blocked
    # on writing, not reading).
    #
    # Sentinel batches create RunnerResults inline on the pump thread and
    # stash them here.  ``_post_schedule_batch`` drains this list via
    # ``_handle_result``, keeping sentinel results on the fast path without
    # touching the IPC pipe.
    _local_results: list[RunnerResult] = field(init=False, default_factory=list)

    # Old result queues abandoned by ``_rotate_result_queue`` after a
    # worker death.  We keep references so they can be closed cleanly at
    # shutdown rather than relying on GC.  (Closing them at rotation time
    # is unsafe because the pump may still be inside an old ``get()``.)
    _abandoned_result_queues: list[Any] = field(init=False, default_factory=list)

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


class ProcessStageRunner(QueueDrainStageRunner[_ProcessOperatorState]):
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

      - owns a deep-copied :class:`~zephon.ops.BaseOp` instance,
      - receives :class:`_WorkerCommand` messages from ``task_queue``
        (kinds: ``"batch"``, ``"stop"``),
      - runs ``process_many`` on its local operator instance, and
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
        coalesce_tensors: bool = True,
        shm_min_size: int = DEFAULT_SHM_MIN_SIZE,
        max_worker_retries: int = 0,
        ipc_transport: IpcTransport = DEFAULT_IPC_TRANSPORT,
        ipc_buffer_bytes: int = DEFAULT_IPC_BUFFER_BYTES,
    ) -> None:
        self._coalesce_tensors = coalesce_tensors
        self._shm_min_size = shm_min_size
        self._max_worker_retries = max(0, int(max_worker_retries))
        self._queue_capacity = max(1, queue_capacity)
        self._ipc_transport = ipc_transport
        self._ipc_buffer_bytes = ipc_buffer_bytes
        ctx = mp_context or mp.get_context("spawn")
        self._mp_context: BaseContext = ctx
        # Resilient respawn requires spawn/forkserver: fork + threads +
        # Process.start() from a non-main thread (the watchdog) is a
        # classic deadlock (child inherits locks held by threads that
        # no longer exist in the child).  If the caller's mp_context uses
        # fork (either by default on old platforms or by explicit
        # configuration), silently disable the watchdog and warn — we
        # don't want a default-on option to break existing fork-based
        # setups.
        if self._max_worker_retries > 0:
            method = ctx.get_start_method()
            if method not in ("spawn", "forkserver"):
                warnings.warn(
                    f"ProcessStageRunner: disabling worker-resilience watchdog "
                    f"(max_worker_retries={self._max_worker_retries}) because "
                    f"mp_context uses start method {method!r} — fork + threads + "
                    f"Process.start() from the watchdog deadlocks.  Pass "
                    f"mp_context=mp.get_context('spawn') to enable worker "
                    f"respawn on crash.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._max_worker_retries = 0
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
        # Single-byte, process-shared flags avoid orphaned locks if a worker is
        # killed while inside a service call. No threading.Event crosses IPC.
        self._service_cancelled = ctx.Value("b", 0, lock=False)
        self._service_pending: dict[int, _SharedFlag] = {}
        self._next_worker_id = 0
        self._single_op_direct_ipc = len(stage.nodes) == 1
        self._shutdown_lock = threading.Lock()
        self._workers_shutdown = False
        # Crash-detection watchdog.  Started in _before_run, stopped in
        # _after_run / close().  Default body is diagnostic (shout on
        # unexpected death); the resilient-worker actor overrides
        # ``_handle_dead_worker``.
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
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
        if _STARTUP_DEBUG:
            features.append("ZEPHON_DEBUG_WORKER_STARTUP")
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
            NamedQueue(
                name,
                maxsize=size,
                ctx=self._mp_context,
                transport=self._ipc_transport,
                buffer_bytes=self._ipc_buffer_bytes,
            ),
        )

    @staticmethod
    def _close_ipc_queue(q: Any, *, hard: bool = False) -> None:
        if q is None:
            return

        try:
            q.close()
        except Exception:
            name = "unnamed" if not isinstance(q, NamedQueue) else q._name_label
            print(
                f"Error closing queue {name!r}.",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)

        flush_timeout = _HARD_QUEUE_FLUSH if hard else _GRACEFUL_QUEUE_FLUSH
        t = getattr(q, "_thread", None)
        if t is not None:
            try:
                t.join(timeout=flush_timeout)
            except Exception:
                pass

            if t.is_alive():
                print(
                    f"Queue failed to flush within {flush_timeout} seconds. "
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

    def _start_operator_threads(self, context: ConcurrentRunContext) -> None:
        if self._single_op_direct_ipc:
            return
        super()._start_operator_threads(context)

    def _before_run(self, context: ConcurrentRunContext) -> None:
        # Reset shutdown flag for runner reuse (start -> stop -> start)
        self._workers_shutdown = False
        self._service_cancelled.value = 0
        for state in self.ops:
            self._launch_workers(state)
        self._start_service_thread()
        self._start_watchdog()

    def _after_run(self, context: ConcurrentRunContext) -> None:
        self._service_cancelled.value = 1
        self._stop_watchdog()
        self._shutdown_workers()
        self._stop_service_thread()

    def _start_watchdog(self) -> None:
        """Launch the crash-detection watchdog thread."""
        if self._watchdog_thread is not None:
            return
        self._watchdog_stop.clear()
        thread = threading.Thread(
            target=self._watchdog_loop,
            name="zephon-proc-runner-watchdog",
            daemon=True,
        )
        thread.start()
        self._watchdog_thread = thread

    def _stop_watchdog(self) -> None:
        thread = self._watchdog_thread
        if thread is None:
            return
        self._watchdog_stop.set()
        thread.join(timeout=1.0)
        self._watchdog_thread = None

    def _watchdog_loop(self) -> None:
        """Poll worker liveness; handle only UNEXPECTED deaths (negative exitcode).

        A worker exit is "unexpected" when ``exitcode < 0``, i.e. the
        process was terminated by a signal (SIGSEGV=-11, SIGKILL=-9,
        SIGABRT=-6, SIGBUS=-7, etc.) — these are the cases the resilient
        actor needs to recover from, because the worker had no chance to
        deliver a ``WorkerErrorInfo`` via ``result_queue``.

        Non-negative exitcodes are *clean* exits:

        - ``exitcode == 0`` via the ``if command.kind == "stop": break``
          path — worker received a stop command during shutdown and exited
          normally.
        - ``exitcode == 0`` via the ``except BaseException`` handler at
          the bottom of ``_process_worker_main`` — a Python-level
          exception was caught, ``WorkerErrorInfo`` was already pushed to
          ``result_queue``, and the function returned normally.

        In both clean cases, any required action (shutdown join, error
        propagation) is already in flight via the normal channels.
        Triggering resilient recovery on top of that caused deadlocks:
        watchdog would try to ``task_queue.put()`` resubmits while
        ``_shutdown_workers`` held no drain, blocking forever while
        holding ``_shutdown_lock``.
        """
        reported: set[int] = set()
        while not self._watchdog_stop.wait(timeout=_WATCHDOG_POLL_S):
            with self._shutdown_lock:
                if self._workers_shutdown or sys.is_finalizing():
                    return
                for state in self.ops:
                    for worker_index, proc in enumerate(list(state.workers)):
                        pid = proc.pid
                        if pid is None or pid in reported:
                            continue
                        if proc.is_alive():
                            continue
                        # Mark as seen regardless so we don't re-inspect.
                        reported.add(pid)
                        if not self._is_unexpected_death(proc):
                            if _DEBUG:
                                _debug(
                                    f"watchdog: worker_idx={worker_index} "
                                    f"pid={pid} exited cleanly "
                                    f"(exitcode={proc.exitcode}); skipping "
                                    f"resilient recovery"
                                )
                            continue
                        # Signal-terminated → real crash.
                        self._handle_dead_worker(state, worker_index, proc)

    @staticmethod
    def _is_unexpected_death(proc: BaseProcess) -> bool:
        """True iff a dead worker needs resilient-actor recovery.

        "Unexpected" = signal-terminated (SIGSEGV=-11, SIGKILL=-9,
        SIGABRT=-6, SIGBUS=-7, SIGTERM=-15, etc.), indicated by a NEGATIVE
        exitcode.  In those cases the worker had no chance to deliver a
        ``WorkerErrorInfo`` via ``result_queue`` and we need to resubmit
        + respawn.

        "Expected" = non-negative exitcode, which means one of:
        - 0 via the ``if command.kind == "stop": break`` path in
          ``_process_worker_main`` — worker received a stop command
          during shutdown and exited normally.
        - 0 via the ``except BaseException`` handler at the bottom of
          ``_process_worker_main`` — a Python exception was caught and
          ``WorkerErrorInfo`` was already pushed to ``result_queue``.
        In both cases, normal error-propagation / shutdown paths handle
        cleanup; the watchdog must stay out of the way to avoid
        deadlocking ``_shutdown_workers`` on ``_shutdown_lock``.

        ``exitcode`` can be ``None`` transiently (after ``is_alive()``
        returned False but before the exit has been fully reaped).
        Treat that as "not unexpected" — defensive.
        """
        code = proc.exitcode
        return code is not None and code < 0

    # Result-queue recovery after a worker crash lives in
    # :mod:`zephon._internal.runners.watchdog`.  The watchdog calls
    # ``recover_result_queue`` from there; OS-specific bits (POSIX
    # semaphore introspection, Linux ``/proc/PID/syscall`` parsing) are
    # kept out of this file.

    def _handle_dead_worker(
        self,
        state: "_ProcessOperatorState",
        worker_index: int,
        dead_proc: BaseProcess,
    ) -> None:
        """Log + (optionally) resubmit and respawn.

        When ``max_worker_retries == 0`` we only log the death: the pipeline
        will hang/fail like it did before resilience existed.  When
        ``max_worker_retries > 0`` we resubmit in-flight work for this
        worker (targeted via the SHM slot when possible; blind fallback
        otherwise), bump per-seq retry counts, escalate / drop on
        exhaustion, then spawn a replacement into the same slot.
        """
        pid = dead_proc.pid
        exitcode = dead_proc.exitcode
        # Determine which seq the worker was ACTIVELY processing at the
        # moment of death (the likely culprit), versus innocent bystanders.
        #
        # A worker's ``multiprocessing.Queue.put()`` buffers to an
        # in-worker feeder thread; when the worker dies, any already-
        # completed results sitting in that outbox die with it.  So on
        # crash we MUST resubmit every seq we haven't received a result
        # for — not just the SHM-slot value.  The slot still tells us
        # which seq was actively in ``process_many`` (the crash
        # candidate), so only THAT seq has its retry count bumped; the
        # outboxed bystanders are resubmitted with retry_count untouched.
        culprit_seq = -1
        if state.worker_seq_slots is not None:
            try:
                culprit_seq = int(state.worker_seq_slots[worker_index])
            except Exception:
                culprit_seq = -1

        to_consider: list[int] = []
        if self._max_worker_retries > 0 and state.pending_commands:
            to_consider = list(state.pending_commands.keys())

        self._log_worker_death(
            state, worker_index, pid, exitcode, to_consider, culprit_seq
        )

        worker_id = state.worker_ids[worker_index]
        pending = self._service_pending.get(worker_id)
        if pending is not None and pending.value:
            # Result-queue recovery cannot repair a control queue whose writer
            # died mid-message. Fail this execution instead of reusing it.
            self._service_cancelled.value = 1
            context = self._active_context
            if context is not None:
                self._record_error(
                    context,
                    RuntimeError(
                        f"Worker {worker_id} died during a control service call; "
                        + "the control transport cannot be safely reused"
                    ),
                )
            return

        if self._max_worker_retries <= 0:
            # Diagnostic-only mode: leave pending_commands / workers alone.
            # Pipeline will surface an error or hang per pre-resilience
            # behavior.  Still report the death loudly above.
            return

        # Recover from a possibly-wedged ``result_queue._wlock``: the
        # mp.Queue ``_feed`` thread holds it across ``send_bytes`` and
        # a worker that dies inside that critical section leaves the
        # POSIX semaphore permanently acquired.  Strategy chain (queue
        # rotation for parallelism=1, then timed-acquire +
        # ``/proc/PID/syscall`` for parallelism>1) lives in
        # :mod:`zephon._internal.runners.watchdog`.
        recover_result_queue(
            state=state,
            dead_proc=dead_proc,
            worker_index=worker_index,
            make_ipc_queue=self._make_ipc_queue,
        )

        # Drop the dead worker's response queue so it doesn't leak.
        old_wid = state.worker_ids[worker_index]
        old_resp = self._service_responses.pop(old_wid, None)
        self._service_pending.pop(old_wid, None)
        if old_resp is not None:
            self._close_ipc_queue(old_resp)

        # Resubmit needs a live consumer: blind re-dispatch can push
        # ``task_queue`` past capacity, and a blocked ``put()`` here
        # would deadlock with ``_shutdown_lock`` held.  Spawn first so
        # the replacement drains as we re-dispatch.
        try:
            op_proto_bytes = cloudpickle.dumps(state.node.op)
            proc, sem, wid, resp_queue = self._build_worker(
                state, worker_index, op_proto_bytes, time.time_ns()
            )
            if state.worker_seq_slots is not None:
                try:
                    state.worker_seq_slots[worker_index] = -1
                except Exception:
                    pass
            self._start_worker(proc, wid)
            self._install_worker(state, worker_index, proc, sem, wid, resp_queue)
            print(
                f"[zephon] Respawned worker: stage={state.stage_name!r} "
                f"op={state.node.name!r} worker_idx={worker_index} "
                f"old_pid={pid} new_pid={proc.pid} ({rank_ctx()})",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort respawn
            traceback.print_exc(file=sys.stderr)
            print(
                f"[zephon] FAILED to respawn worker {worker_index} for "
                f"stage={state.stage_name!r} op={state.node.name!r}: {exc!r} "
                f"({rank_ctx()})",
                file=sys.stderr,
                flush=True,
            )
            # Respawn failure is fatal to resilient mode: the dead worker
            # is gone and no replacement is coming.  Fail every remaining
            # in-flight seq so the pipeline error-propagates cleanly
            # instead of hanging forever on the retry path.  In non-det
            # mode we still fail loudly — respawn failure is an env
            # problem (fork/spawn/fd limits), not a per-sample issue.
            for seq in list(state.pending_commands.keys()):
                info = WorkerErrorInfo(
                    exc_type="WorkerRespawnFailed",
                    message=(
                        f"Could not respawn worker {worker_index} after "
                        f"pid={pid} exit={exitcode}: {exc!r}"
                    ),
                    formatted_traceback=traceback.format_exc(),
                )
                err_result = RunnerResult(
                    seq=seq,
                    payload=[],
                    wait_ns=0,
                    consumed_elements=0,
                    consumed_bytes=0,
                    queue_depth_snapshot=-1,
                    proc_ns=0,
                    collect_metrics=False,
                    ack=None,
                    error=info,
                )
                try:
                    state.result_queue.put(err_result)
                except Exception:  # noqa: BLE001 - queue dead, last-ditch
                    break
                state.pending_commands.pop(seq, None)
                state.retry_counts.pop(seq, None)
            return

        # Split ``to_consider`` into "actually resubmit" vs "exhaust".  Only
        # the culprit's retry count is incremented; bystanders re-dispatch
        # for free (their results just happened to be in the dying worker's
        # outbox and got lost — not their fault).
        to_resubmit: list[int] = []
        for seq in to_consider:
            cmd = state.pending_commands.get(seq)
            if cmd is None:
                # Raced with the main pump's result handling — nothing to do.
                continue
            if seq == culprit_seq:
                attempts = state.retry_counts.get(seq, 0) + 1
                if attempts > self._max_worker_retries:
                    self._exhaust_seq(state, seq, pid, exitcode)
                    continue
                state.retry_counts[seq] = attempts
            to_resubmit.append(seq)

        # Deterministic mode only: post a ``_RelaxSignal`` so the pump
        # unblocks workers stuck on ``backpressure.acquire()``.
        #
        # Why this is necessary: in det mode, permits held by buffered
        # entries in ``pending_results`` are released on emit — but emit
        # is stuck on the dead worker's culprit seq.  Workers B/C/D
        # exhaust their permits, block at ``acquire()`` with unfinished
        # results in hand, and never reach ``task_queue.get()``.  The
        # pump's ``_send_command`` drain-retry loop observes an empty
        # ``result_queue`` (workers can't put), so its flag-check never
        # fires.  The signal breaks this by post-hoc releasing permits
        # from the pump itself, then switching to arrival-release until
        # the resubmits have all come back.  See the
        # ``Resilience and recovery`` section below for the full trace.
        #
        # Non-det mode doesn't deadlock this way — permits are released
        # on arrival already via ``_forward_ready_result``, so workers
        # never back up against a stuck emit.  Skip the signal.
        if state.deterministic and to_resubmit:
            try:
                state.result_queue.put(_RelaxSignal(frozenset(to_resubmit)))
            except ValueError:
                # Queue closed mid-shutdown.  Pipeline is tearing down;
                # skipping the signal is the right call.
                return

        # Resubmit.  The fresh worker is already draining ``task_queue`` so
        # these puts can't deadlock.
        task_queue = state.task_queue
        if task_queue is not None:
            for seq in to_resubmit:
                cmd = state.pending_commands.get(seq)
                if cmd is None:
                    continue
                try:
                    task_queue.put(cmd)
                except ValueError:
                    # Queue closed mid-shutdown.
                    return

    def _log_worker_death(
        self,
        state: "_ProcessOperatorState",
        worker_index: int,
        pid: int | None,
        exitcode: int | None,
        to_resubmit: Sequence[int],
        culprit_seq: int,
    ) -> None:
        """Loud, multi-line stderr banner announcing a worker death.

        Distinguishes the *culprit* seq (what the worker was actively
        processing when it died — most likely cause of the crash) from
        *bystanders* (completed seqs that were still in the worker's
        in-process ``Queue`` outbox and got lost when the worker died).
        Only the culprit accrues a retry count; bystanders re-dispatch
        for free.
        """
        if not to_resubmit:
            resubmit_desc = "none"
        else:
            preview = to_resubmit[:16]
            overflow = "..." if len(to_resubmit) > len(preview) else ""
            if culprit_seq >= 0 and culprit_seq in to_resubmit:
                bystanders = [s for s in preview if s != culprit_seq]
                resubmit_desc = (
                    f"{len(to_resubmit)} items — culprit seq={culprit_seq}, "
                    f"bystanders={bystanders}{overflow} "
                    f"(completed results lost in worker outbox on crash)"
                )
            else:
                # No SHM slot attribution — every in-flight seq is suspect.
                resubmit_desc = (
                    f"{len(to_resubmit)} items [no culprit attribution] "
                    f"seqs={list(preview)}{overflow}"
                )
        print(
            "\n"
            + "!" * 78
            + "\n"
            + f"[zephon] WORKER DIED: stage={state.stage_name!r} "
            + f"op={state.node.name!r} worker_idx={worker_index} "
            + f"worker_pid={pid} exit={exitcode}. ({rank_ctx()})\n"
            + f"         Resubmitting: {resubmit_desc}\n"
            + "         The culprit may have triggered the crash; the "
            + "next worker will retry it.\n"
            + "         Common causes: OOM (exit=-9/-6), segfault "
            + "(exit=-11), unhandled C-level crash.\n"
            + "!" * 78,
            file=sys.stderr,
            flush=True,
        )

    def _exhaust_seq(
        self,
        state: "_ProcessOperatorState",
        seq: int,
        dead_pid: int | None,
        exitcode: int | None,
    ) -> None:
        """Handle retry-budget exhaustion for a single seq.

        Deterministic mode: inject a synthetic error result so the pump's
        existing ``_forward_ready_result`` → ``_record_error`` path fails
        the pipeline with a ``WorkerCrashed(MaxWorkerRetriesExceeded)``.

        Non-deterministic mode: drop the sample silently (log loudly).
        Downstream ordering is not a contract, so consumers see a
        pipeline that simply skipped a poisoned sample.
        """
        state.pending_commands.pop(seq, None)
        state.retry_counts.pop(seq, None)
        if state.deterministic:
            info = WorkerErrorInfo(
                exc_type="MaxWorkerRetriesExceeded",
                message=(
                    f"seq={seq} failed {self._max_worker_retries} retries "
                    f"(last crash: pid={dead_pid} exit={exitcode})"
                ),
                formatted_traceback="",
            )
            err_result = RunnerResult(
                seq=seq,
                payload=[],
                wait_ns=0,
                consumed_elements=0,
                consumed_bytes=0,
                queue_depth_snapshot=-1,
                proc_ns=0,
                collect_metrics=False,
                ack=None,
                error=info,
            )
            try:
                state.result_queue.put(err_result)
            except Exception:  # noqa: BLE001 - queue closed during shutdown
                pass
            print(
                f"[zephon] EXHAUSTED seq={seq} after "
                f"{self._max_worker_retries} retries — escalating to "
                f"WorkerCrashed (deterministic). "
                f"stage={state.stage_name!r} op={state.node.name!r} "
                f"({rank_ctx()})",
                file=sys.stderr,
                flush=True,
            )
        else:
            # Drop: decrement inflight so shutdown can eventually reach
            # idle; emit nothing downstream.  We do not rely on the pump
            # to observe this — we adjust the counter directly.
            state.inflight.try_decrement()
            print(
                f"[zephon] DROPPED seq={seq} after "
                f"{self._max_worker_retries} retries (non-deterministic mode). "
                f"Sample lost; pipeline continues. "
                f"stage={state.stage_name!r} op={state.node.name!r} "
                f"({rank_ctx()})",
                file=sys.stderr,
                flush=True,
            )

    def _start_service_thread(self) -> None:
        if self._service_thread is not None:
            return
        self._service_stop.clear()
        _debug("starting service thread")
        thread = threading.Thread(target=self._service_loop, daemon=True)
        thread.start()
        self._service_thread = thread

    def _stop_service_thread(self, *, hard: bool = False) -> None:
        self._service_cancelled.value = 1
        # Grab a local ref to avoid races with concurrent callers
        # (_after_run and close() can both reach here).
        thread = self._service_thread
        if thread is None:
            _shutdown_debug("_stop_service_thread: no service thread")
            return
        _shutdown_debug("_stop_service_thread: setting stop flag")
        self._service_stop.set()
        try:
            self._service_queue.put(None)
        except (FileNotFoundError, EOFError, OSError, ValueError):
            pass  # i think we right now do this multiple times but anyways

        join_timeout = _HARD_SERVICE_JOIN if hard else _GRACEFUL_SERVICE_JOIN
        _shutdown_debug("_stop_service_thread: joining service thread")
        thread.join(timeout=join_timeout)
        if thread.is_alive():
            _shutdown_debug(
                f"_stop_service_thread: WARNING - service thread still alive after {join_timeout}s join"
            )
        self._service_thread = None
        _shutdown_debug("_stop_service_thread: closing response queues")
        for resp in self._service_responses.values():
            self._close_ipc_queue(resp, hard=hard)
        self._service_responses.clear()
        self._service_pending.clear()
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

                with shutdown_flush(enabled=context.stop_event.is_set()):
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
            # Drain locally stashed sentinel results before draining the IPC queue.
            self._post_schedule_batch(state, None, context)
            # next_queue=None: results go to stage output (correct for single-op stages)
            self._drain_results(state, None, context)

    def _drain_until_idle(
        self,
        state: _ProcessOperatorState,
        context: ConcurrentRunContext,
    ) -> None:
        last_log_ns = 0
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
                if _DEBUG:
                    now = time.monotonic_ns()
                    if now - last_log_ns > 1_000_000_000:  # 1s
                        last_log_ns = now
                        _debug(
                            f"drain_until_idle waiting: inflight={state.inflight._count} "
                            f"pending_cmds={len(state.pending_commands)} "
                            f"pending_puts={state.pending_puts._count} "
                            f"pending_results={len(state.pending_results)}"
                        )
                continue
            self._handle_result(state, item, None, context)

    def _service_loop(self) -> None:
        try:
            self._serve_requests()
        except BaseException as exc:  # noqa: BLE001 - report transport failures
            if not self._service_stop.is_set():
                self._service_cancelled.value = 1
                context = self._active_context
                if context is not None:
                    self._record_error(context, exc)

    def _serve_requests(self) -> None:
        while not self._service_stop.is_set():
            try:
                req = self._service_queue.get(timeout=0.1)
            except queue.Empty:
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
            except BaseException as exc:  # noqa: BLE001
                resp.put((False, exc))
            else:
                resp.put((True, result))

    def _build_worker(
        self,
        state: _ProcessOperatorState,
        worker_index: int,
        op_proto_bytes: bytes,
        spawn_wall_ns: int,
    ) -> tuple[
        BaseProcess,
        Semaphore | SafeSemLock,
        int,
        _ClosableQueue[tuple[bool, Any]],
    ]:
        """Allocate per-worker resources and create (but don't start) the Process.

        Shared by :meth:`_launch_workers` (initial launch) and
        :meth:`_replace_worker` (crash-driven respawn).  Returns the new
        ``Process`` plus its fresh backpressure semaphore, monotonically-
        allocated ``worker_id``, and service-response queue.  Caller is
        responsible for ``proc.start()`` and ``_install_worker``.
        """
        op_label = state.node.name or f"op{state.op_index}"
        queue_label = f"{state.stage_name}:{op_label}"
        worker_id = self._next_worker_id
        self._next_worker_id += 1
        semaphore = self._semaphore_factory(self._queue_capacity)
        resp_queue = cast(
            _ClosableQueue[tuple[bool, Any]],
            self._make_ipc_queue(
                f"service-response:{queue_label}:{worker_id}",
            ),
        )
        # Install the service endpoint before start: a replacement can consume
        # queued work immediately while the service thread is already running.
        # Keep both pipe ends open until the child has inherited/serialized them.
        self._service_responses[worker_id] = resp_queue
        self._service_pending[worker_id] = self._mp_context.Value("b", 0, lock=False)
        ctx_payload = self._build_worker_ctx(worker_id, resp_queue)
        config = _ProcessWorkerConfig(
            worker_index=worker_index,
            worker_id=worker_id,
            op_proto_bytes=op_proto_bytes,
            stage_index=state.stage_index,
            stage_name=state.stage_name,
            op_index=state.op_index,
            collect_stats=self._tracking_mode.collects_nodes,
            ctx_services=ctx_payload,
            task_queue=cast(_QueueLike[_WorkerCommand], state.task_queue),
            result_queue=state.result_queue,
            backpressure=semaphore,
            worker_seq_slots=state.worker_seq_slots,
            spawn_wall_ns=spawn_wall_ns,
            coalesce_tensors=self._coalesce_tensors,
            shm_min_size=self._shm_min_size,
        )
        proc: BaseProcess = self._process_factory(
            target=_process_worker_main,
            args=(config,),
            daemon=True,
        )
        return proc, semaphore, worker_id, resp_queue

    def _install_worker(
        self,
        state: _ProcessOperatorState,
        worker_index: int,
        proc: BaseProcess,
        semaphore: Semaphore | SafeSemLock,
        worker_id: int,
        resp_queue: _ClosableQueue[tuple[bool, Any]],
    ) -> None:
        """Register a started worker into state and its response queue.

        Appends when ``worker_index == len(state.workers)`` (initial
        launch), otherwise overwrites the existing slot (replacement).
        """
        _debug(f"started worker process idx={worker_index} pid={proc.pid}")
        # main -> worker (service response queue): main is producer-only
        _close_reader_end(resp_queue)
        if worker_index < len(state.workers):
            state.workers[worker_index] = proc
            state.result_semaphores[worker_index] = semaphore
            state.worker_ids[worker_index] = worker_id
        else:
            state.worker_ids.append(worker_id)
            state.result_semaphores.append(semaphore)
            state.workers.append(proc)

    def _start_worker(self, proc: BaseProcess, worker_id: int) -> None:
        """Start with the reply endpoint installed, cleaning it on failure."""
        try:
            proc.start()
        except BaseException:
            response = self._service_responses.pop(worker_id, None)
            self._service_pending.pop(worker_id, None)
            self._close_ipc_queue(response, hard=True)
            raise

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

        # Serialize operator once using cloudpickle to support lambdas/closures.
        # Local-only: we re-pickle on respawn rather than retaining this in
        # main-process memory for the lifetime of the runner.
        op_proto_bytes = cloudpickle.dumps(state.node.op)

        # See ``_ProcessWorkerConfig.worker_seq_slots`` for the slot contract.
        state.worker_seq_slots = self._mp_context.Array(
            "q", [-1] * state.parallelism, lock=False
        )

        # Reset resilient-worker bookkeeping.
        state.pending_commands.clear()
        state.retry_counts.clear()
        state.relaxed_backpressure = False
        state.relaxed_backpressure_seqs.clear()

        # Record wall-clock time so workers can compute cross-process startup duration
        spawn_wall_ns = time.time_ns()

        # Phase 1: Build all worker Process objects via the shared helper.
        built: list[
            tuple[
                int,
                BaseProcess,
                Semaphore | SafeSemLock,
                int,
                _ClosableQueue[tuple[bool, Any]],
            ]
        ] = []
        for idx in range(state.parallelism):
            proc, sem, wid, resp_queue = self._build_worker(
                state, idx, op_proto_bytes, spawn_wall_ns
            )
            built.append((idx, proc, sem, wid, resp_queue))

        # Phase 2: Start all processes.
        # For spawn/forkserver, we start processes in parallel using threads since
        # proc.start() blocks until the process is forked. This reduces spawn time
        # from O(n * spawn_time) to O(spawn_time).
        # For fork, we start sequentially because fork() in a multithreaded program
        # is unsafe: the child inherits locks held by threads that no longer exist,
        # leading to potential deadlocks. Concurrent fork() calls exacerbate this.
        spawn_t0 = time.perf_counter_ns()
        start_method = self._mp_context.get_start_method()
        start_errors: list[BaseException] = []
        errors_lock = threading.Lock()

        def start(proc: BaseProcess, worker_id: int) -> None:
            try:
                self._start_worker(proc, worker_id)
            except BaseException as exc:
                with errors_lock:
                    start_errors.append(exc)

        if start_method in ("spawn", "forkserver"):
            spawn_threads: list[threading.Thread] = []
            for _, proc, _, wid, _ in built:
                t = threading.Thread(target=start, args=(proc, wid), daemon=True)
                t.start()
                spawn_threads.append(t)
            for t in spawn_threads:
                t.join()
        else:
            for _, proc, _, wid, _ in built:
                start(proc, wid)
        spawn_s = (time.perf_counter_ns() - spawn_t0) / 1e9
        _startup_log(
            f"spawned {state.parallelism} workers for {queue_label} "
            f"in {spawn_s:.2f}s ({start_method})"
        )

        if start_errors:
            # Do not install a partial pool: worker indices also index reply
            # permits, so missing slots cannot be compacted safely.
            for _, proc, sem, wid, resp_queue in built:
                if proc.pid is not None:
                    proc.terminate()
                    proc.join(timeout=_GRACEFUL_TERMINATE_JOIN)
                    if proc.is_alive():
                        proc.kill()
                        proc.join()
                response = self._service_responses.pop(wid, None)
                self._service_pending.pop(wid, None)
                if response is not None:
                    self._close_ipc_queue(response, hard=True)
                cleanup_semaphores([sem])
            raise start_errors[0]

        # Phase 3: Install started processes into state.
        for idx, proc, sem, wid, resp_queue in built:
            self._install_worker(state, idx, proc, sem, wid, resp_queue)

        # fd hygiene in non-resilient mode only.
        #
        # Normally we close main's unused ends of the two IPC queues:
        #   - reader end of task_queue (main is producer-only),
        #   - writer end of result_queue (main is consumer-only).
        # This lets the kernel signal EOF on worker exit and reduces fd count.
        #
        # In resilient mode we keep BOTH ends open for two reasons:
        # 1. ``proc.start()`` for a replacement pickles the queues, which
        #    requires both ``_reader`` and ``_writer`` to be live fds in the
        #    parent.  Closing the unused end breaks respawn with
        #    ``OSError('handle is closed')``.
        # 2. On the single-worker result_queue, closing main's writer-end
        #    would let the reader see EOF during the respawn window (zero
        #    writers momentarily) and the replacement's new writes can't
        #    un-EOF an already-EOF'd Connection.
        # The watchdog is the primary crash-detection signal in resilient
        # mode, so EOF on the queue isn't load-bearing here.
        if self._max_worker_retries <= 0:
            _close_reader_end(task_queue)
            _close_writer_end(result_queue)

    def _shutdown_workers(self, *, hard: bool = False) -> None:
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
            drain_timeout = _HARD_DRAIN_TIMEOUT if hard else _GRACEFUL_DRAIN_TIMEOUT
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
                        # Skip in-flight watchdog signals — they aren't results,
                        # and we're tearing everything down anyway so there's
                        # nothing to relax.
                        if isinstance(item, _RelaxSignal):
                            drained_any = True
                            continue
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
            join_timeout = _HARD_WORKER_JOIN if hard else _GRACEFUL_WORKER_JOIN
            terminate_timeout = (
                _HARD_TERMINATE_JOIN if hard else _GRACEFUL_TERMINATE_JOIN
            )
            _shutdown_debug(f"_shutdown_workers: op[{op_idx}] joining workers")
            for worker_idx, proc in enumerate(state.workers):
                _shutdown_debug(
                    f"_shutdown_workers: op[{op_idx}] joining worker[{worker_idx}] pid={proc.pid}"
                )
                proc.join(timeout=join_timeout)
                if proc.is_alive():
                    _shutdown_debug(
                        f"_shutdown_workers: op[{op_idx}] worker[{worker_idx}] did NOT exit cleanly, terminating"
                    )
                    proc.terminate()
                    # Give it a moment to die gracefully, then kill
                    proc.join(timeout=terminate_timeout)
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
                self._service_pending.pop(wid, None)
                self._close_ipc_queue(resp, hard=hard)

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
            self._close_ipc_queue(state.task_queue, hard=hard)
            self._close_ipc_queue(state.result_queue, hard=hard)
            close_abandoned_result_queues(
                state, lambda q: self._close_ipc_queue(q, hard=hard)
            )

            state.task_queue = None

        _shutdown_debug("_shutdown_workers: closing service queue")
        self._close_ipc_queue(self._service_queue, hard=hard)
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
                    self._service_cancelled,
                    self._service_pending[worker_id],
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
                                state.pump_timer.note("batches_completed")
                                with state.pump_timer.measure("result_handle"):
                                    self._handle_result(
                                        state, item, next_queue, context
                                    )
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

        Sentinel batches (tombstones, flush signals) bypass worker execution
        entirely — they are control signals that operators should never see.
        A RunnerResult is created inline and stashed in ``_local_results``
        so the pump thread never writes to the IPC result pipe.
        """
        if not batch or context.stop_event.is_set():
            return

        # Sentinel batches bypass workers — create result inline.
        # _post_schedule_batch drains _local_results via _handle_result,
        # which respects deterministic sequence ordering — sentinels cannot
        # overtake prior worker results.
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
        transport_batch = batch
        if self._coalesce_tensors:
            forwarded = forward_shared_numpy(batch)
            if forwarded is not None:
                # The envelope's reducer reconstructs a list in the worker.
                transport_batch = cast(list[RunnerStreamIn], forwarded)
        command = _WorkerCommand(
            kind="batch",
            seq=seq,
            batch=transport_batch,
            wait_ns=wait_ns if collect_stats else 0,
            consumed_elements=consumed_elements,
            consumed_bytes=consumed_bytes,
            queue_depth_snapshot=queue_depth_snapshot,
            collect_metrics=collect_stats,
        )
        # Record for the watchdog BEFORE putting on the task queue so a
        # worker that picks up, crashes, and gets observed by the watchdog
        # can never find pending_commands missing an in-flight seq.
        state.pending_commands[seq] = command
        # Exclude result_handle: a full task queue makes _send_command drain
        # results, which would otherwise double-count into dispatch_active.
        with state.pump_timer.measure_excluding("dispatch_active", "result_handle"):
            self._send_command(task_queue, command, state=state, context=context)
        state.pump_timer.note("batches_submitted")
        _debug(f"scheduled batch seq={seq}")
        state.inflight.increment()

    def _post_schedule_batch(
        self,
        state: _ProcessOperatorState,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
    ) -> None:
        for result in state._local_results:
            state.pump_timer.note("batches_completed")
            with state.pump_timer.measure("result_handle"):
                self._handle_result(state, result, next_queue, context)
        state._local_results.clear()

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

    def _handle_relax_signal(
        self,
        state: _ProcessOperatorState,
        signal: _RelaxSignal,
        context: ConcurrentRunContext,
    ) -> None:
        """Enter relaxed-backpressure mode on watchdog notification.

        The watchdog posts ``_RelaxSignal`` to ``result_queue`` when a
        worker dies with in-flight items.  This handler runs on the pump
        thread.  Two actions:

        1. Post-hoc release the permits held by entries already in
           ``pending_results``.  Those entries are waiting to emit but
           their permits are held by workers that may be blocked on
           ``backpressure.acquire()`` with an unfinished result in hand.
           Releasing the permits unblocks the workers so they can put
           their results and loop back to ``task_queue.get()`` (which in
           turn frees a slot for the pump's blocked ``task_queue.put()``
           retry loop).  We null out ``entry.ack`` to prevent
           double-release when the entry later emits.

        2. Turn on the relaxed flag and record the seqs the watchdog
           resubmitted.  Until every one of those seqs arrives, the main
           ``_handle_result`` path releases permits on arrival instead
           of on emit — otherwise each fresh result would accumulate a
           held permit again and we'd deadlock exactly the same way.
        """
        for entry in state.pending_results.values():
            if entry.ack is not None:
                self._ack_result(state, entry, context)
                entry.ack = None
        state.relaxed_backpressure_seqs.update(signal.recovery_seqs)
        if state.relaxed_backpressure_seqs:
            state.relaxed_backpressure = True

    def _handle_result(
        self,
        state: _ProcessOperatorState,
        result: RunnerResult,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
    ) -> None:
        # Watchdog → pump signal: not a RunnerResult, handled inline and returned.
        if isinstance(result, _RelaxSignal):
            self._handle_relax_signal(state, result, context)
            return
        _debug(
            f"handle_result seq={result.seq} error={result.error} "
            + f"payload={len(result.payload)}"
        )
        # Identity-aware inflight tracking.  ``pending_commands`` is the
        # authoritative map of "what is in flight"; the legacy
        # ``state.inflight`` counter is kept in sync alongside it so the
        # base class's ``is_zero()`` idle check keeps working.
        #
        # - First-time result for a seq: pop the command + reset retry count.
        # - Duplicate result (resubmit race): no entry in pending_commands;
        #   we still ack the producer's backpressure semaphore so the live
        #   worker that sent it doesn't leak a slot, then drop the payload.
        # - Sentinel / startup-error results (from_worker=False, seq=-1):
        #   skipped — they were never recorded in pending_commands.
        if result.from_worker and result.seq >= 0:
            cmd = state.pending_commands.pop(result.seq, None)
            state.retry_counts.pop(result.seq, None)
            if cmd is None and result.error is None:
                # Duplicate from a resubmit race.  Release the producer's
                # semaphore (see _ack_result) and drop the payload; do not
                # touch inflight (it was already decremented on first arrival).
                # The dup path must NOT discard from relaxed_backpressure_seqs —
                # the original (non-dup) arrival already did that.
                self._ack_result(state, result, context)
                return
            if cmd is not None:
                # Use try_decrement: during shutdown, force_zero() may have
                # already cleared the counter while pump threads are still
                # processing results (buffered_iterable's 1s join timeout
                # expires before _join_threads completes, allowing
                # _shutdown_workers to run concurrently).
                state.inflight.try_decrement()
                # Relaxed-backpressure arrival release: during the recovery
                # window, release permits on arrival instead of on emit so
                # ``pending_results`` entries don't re-accumulate held permits.
                # Null ack to prevent double-release when the entry emits.
                if (
                    state.relaxed_backpressure
                    and result.error is None
                    and result.ack is not None
                ):
                    self._ack_result(state, result, context)
                    result.ack = None
                # Flag-down transition: once every resubmitted seq has
                # arrived (first-time only; dup path doesn't touch this set),
                # drop back to normal on-emit permit release.
                if state.relaxed_backpressure_seqs:
                    state.relaxed_backpressure_seqs.discard(result.seq)
                    if not state.relaxed_backpressure_seqs:
                        state.relaxed_backpressure = False
        elif result.from_worker:
            # Non-seq worker result (e.g. startup-error with seq=-1).  Keep
            # the legacy counter in sync and fall through.
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
            if isinstance(item, _RelaxSignal):
                # Single-op sync path: route relaxed-mode setup through
                # the pump's handler so pending_results permits get
                # released here too, then keep waiting.
                self._handle_result(state, item, next_queue, context)
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

    def close(self, *, hard: bool = False) -> None:
        self._service_cancelled.value = 1
        _shutdown_debug("close() called")
        with ShutdownWatchdog(_SHUTDOWN_WATCHDOG_TIMEOUT, "close()"):
            with self._context_lock:
                ctx = self._active_context

            if ctx is not None:
                _shutdown_debug("close(): setting stop_event")
                ctx.stop_event.set()
                self._put_stage_stop(ctx)

            _shutdown_debug("close(): stopping watchdog")
            self._stop_watchdog()
            _shutdown_debug("close(): calling _shutdown_workers")
            self._shutdown_workers(hard=hard)
            _shutdown_debug("close(): calling _stop_service_thread")
            self._stop_service_thread(hard=hard)

            if ctx is not None:
                _shutdown_debug("close(): calling _join_threads")
                self._join_threads(ctx, hard=hard)
            _shutdown_debug("close(): done")
