# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""RemoteStageRunner: executes pipeline stages on Ray actor pools.

Integrates with ConcurrentStageRunner to reuse the shared pump-thread,
feeder, deterministic reordering, and shutdown coordination logic.

Backpressure is achieved via per-actor idle tracking: the submit thread
blocks until an actor is free, and actors are only recycled after the
pump has forwarded their result downstream (in ``_ack_result``).
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import ray

logger = logging.getLogger(__name__)

from zephon.core.constants import RunnerStageOut, RunnerStreamIn
from zephon.core.graph import Node, Stage
from zephon.core.notify import is_sentinel
from zephon.observability.config import ExecutionTrackingMode
from zephon.runners.concurrent import (
    ConcurrentRunContext,
    RunnerResult,
    StopToken,
    WorkerErrorInfo,
    _QueueLike,
)
from zephon.runners.queue_drain import (
    QueueDrainOperatorState,
    QueueDrainStageRunner,
)
from zephon.runners.ray.service import (
    _RayActorGroup,
)
from zephon.utils.thread_utils import THREAD_SUPPRESSION_ENV_VARS

# Sentinel pushed into the submit queue to tell the submit thread to exit.
_STOP_SUBMIT = object()


def _noop(*_args: Any, **_kwargs: Any) -> None:
    """No-op placeholder for non-serializable callables sent to Ray actors."""


# ---------------------------------------------------------------------------
# Submit and collector thread loops
# ---------------------------------------------------------------------------


def _submit_loop(
    submit_queue: queue.Queue[Any],
    actors: list[ray.actor.ActorHandle],
    idle_queue: queue.Queue[int],
    refs_queue: queue.Queue[tuple[ray.ObjectRef, int]],
    stop_event: threading.Event,
) -> None:
    """Bridge: pull batches from submit_queue, dispatch to idle actors."""
    while True:
        # Block until a batch arrives (true sleep — no spin).
        try:
            item = submit_queue.get(timeout=0.5)
        except queue.Empty:
            if stop_event.is_set():
                return
            continue

        if item is _STOP_SUBMIT:
            return

        batch, seq = item

        # Block until an actor is free (true sleep — no spin).
        while not stop_event.is_set():
            try:
                actor_idx = idle_queue.get(timeout=0.5)
                break
            except queue.Empty:
                continue
        else:
            return  # stop_event was set

        ref = actors[actor_idx].process.remote(batch, seq)
        refs_queue.put((ref, actor_idx))


def _collector_loop(
    refs_queue: queue.Queue[tuple[ray.ObjectRef, int]],
    result_queue: queue.Queue[RunnerResult],
    stop_event: threading.Event,
    num_actors: int,
    op_name: str,
) -> None:
    """Bridge: wait for actor results, push into result_queue."""
    pending: dict[ray.ObjectRef, int] = {}
    dead_actors: set[int] = set()

    while True:
        # Drain newly submitted refs from the submit thread.
        while True:
            try:
                ref, actor_idx = refs_queue.get_nowait()
                pending[ref] = actor_idx
            except queue.Empty:
                break

        if not pending:
            # No in-flight work. Block on refs_queue for new refs.
            try:
                ref, actor_idx = refs_queue.get(timeout=0.5)
                pending[ref] = actor_idx
            except queue.Empty:
                if stop_event.is_set():
                    return  # shutdown + nothing pending + no new refs coming
                continue

        # Fast sweep: grab everything that's already ready (no blocking).
        # list() required: ray.wait needs list[ObjectRef], not dict_keys.
        refs = list(pending)
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
        if not ready:
            # Nothing ready — block until at least one completes (true sleep).
            ready, _ = ray.wait(refs, num_returns=1, timeout=1.0)

        for ref in ready:
            actor_idx = pending.pop(ref)
            try:
                # TODO(perf): ray.get() deserializes the full payload
                # (including large tensors) on the driver. For reordering
                # we only need metadata (seq number). Splitting each actor
                # return into a metadata ref + payload ref would let us
                # resolve only metadata here and pass the payload ref
                # through to the next stage without deserialization.
                result: RunnerResult = ray.get(ref)
            except Exception as exc:
                # Actor died — synthesize error result, do NOT recycle actor.
                dead_actors.add(actor_idx)
                alive = num_actors - len(dead_actors)
                logger.warning(
                    "Actor %d for op '%s' died: %s (%d/%d actors remaining)",
                    actor_idx,
                    op_name,
                    exc,
                    alive,
                    num_actors,
                )
                result = RunnerResult(
                    seq=-1,
                    payload=[],
                    wait_ns=0,
                    consumed_elements=0,
                    consumed_bytes=0,
                    queue_depth_snapshot=0,
                    proc_ns=0,
                    collect_metrics=False,
                    error=WorkerErrorInfo(
                        exc_type=type(exc).__name__,
                        message=str(exc),
                        formatted_traceback=str(exc),
                    ),
                    from_worker=True,
                    ack=None,  # dead actor won't be returned to idle_queue
                )
                if alive == 0:
                    logger.error(
                        "All actors for op '%s' are dead, stopping stage",
                        op_name,
                    )
                    stop_event.set()
            else:
                result.ack = actor_idx

            # Block until result_queue has space (output backpressure).
            while not stop_event.is_set():
                try:
                    result_queue.put(result, timeout=0.5)
                    break
                except queue.Full:
                    continue


# ---------------------------------------------------------------------------
# Operator state
# ---------------------------------------------------------------------------


@dataclass
class _RayOperatorState(QueueDrainOperatorState):
    """Operator state for the Ray runner.

    Extends ConcurrentOperatorState with an actor group and the bridge
    queues that connect the pump thread to the submit/collector threads.
    """

    actor_group: _RayActorGroup | None = field(default=None, init=False)
    input_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] = field(init=False)
    result_queue: _QueueLike[RunnerResult] = field(init=False)

    # Bridge queues for submit/collector threads.
    submit_queue: queue.Queue[Any] = field(init=False)
    refs_queue: queue.Queue[tuple[ray.ObjectRef, int]] = field(init=False)

    # Thread handles.
    submit_thread: threading.Thread | None = field(default=None, init=False)
    collector_thread: threading.Thread | None = field(default=None, init=False)

    # Sentinel results created inline on the pump thread (same pattern
    # as ThreadStageRunner._local_results).
    _local_results: list[RunnerResult] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Throwaway queues: replaced with real queues in _before_run().
        self.input_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.submit_queue = queue.Queue(maxsize=1)
        self.refs_queue = queue.Queue()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class RemoteStageRunner(QueueDrainStageRunner["_RayOperatorState"]):
    """Stage runner that executes operators on per-operator Ray actor groups.

    Each operator in the stage gets its own group of Ray actors. The
    inherited ConcurrentStageRunner machinery handles the pump threads,
    feeder, deterministic reordering, stop propagation, and the
    pull-driven run() iterator. This subclass provides the Ray-specific
    wiring: actor group creation, submit/collector bridge threads, batch
    submission via queues, and backpressure via per-actor idle tracking.
    """

    _OperatorState = _RayOperatorState

    def __init__(
        self,
        stage: Stage,
        ctx_services: dict[str, Any],
        max_workers: int,
        *,
        num_cpus_per_actor: float = 0,
        prefetch_capacity: int = 0,
        queue_capacity: int = 4,
        deterministic: bool = True,
        allow_latency_flush_in_deterministic: bool = True,
        stage_index: int = 0,
        tracking_mode: ExecutionTrackingMode = ExecutionTrackingMode.OFF,
        stage_output_mode: Literal["microbatches", "stream_items"] = "microbatches",
        runtime_env: dict[str, Any] | None = None,
        actor_env_vars: dict[str, str] | None = None,
    ) -> None:
        self._num_cpus_per_actor = num_cpus_per_actor
        self._queue_capacity = max(1, queue_capacity)

        if runtime_env is None:
            # UV vars ensure the actor process doesn't try to sync a
            # virtual environment on the remote node.
            # Thread-suppression vars from THREAD_SUPPRESSION_ENV_VARS
            # are set here so they're present *before* import time;
            # suppress_library_threads() called inside actor __init__
            # additionally handles runtime calls (e.g. torch.set_num_threads).
            default_env_vars: dict[str, str] = {
                "VIRTUAL_ENV": "",
                "UV_PROJECT_ENVIRONMENT": "",
                "UV_NO_SYNC": "1",
                **THREAD_SUPPRESSION_ENV_VARS,
            }
            if actor_env_vars:
                default_env_vars.update(actor_env_vars)
            self._runtime_env: dict[str, Any] = {"env_vars": default_env_vars}
        else:
            self._runtime_env = runtime_env

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

    def _build_actor_ctx(self) -> dict[str, Any]:
        """Build context services for actors.

        Callables (e.g. metrics callbacks) cannot be serialized across Ray
        actor boundaries via cloudpickle, so we replace them with no-ops.
        This means actors currently run without observability callbacks —
        metrics are only recorded when results arrive back on the driver.

        TODO: implement Ray-native observability (e.g. via Ray's built-in
        metrics, a dedicated reporting actor, or a similar mechanism to the
        process runner's task_queue approach).
        """
        ctx: dict[str, Any] = {}
        for key, val in self._ctx_services.items():
            if callable(val):
                ctx[key] = _noop
            else:
                ctx[key] = val
        return ctx

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
    ) -> _RayOperatorState:
        return _RayOperatorState(
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

    def _before_run(self, context: ConcurrentRunContext) -> None:
        """Initialize actor groups, bridge queues, and submit/collector threads."""
        actor_ctx = self._build_actor_ctx()

        for state in self.ops:
            traits = state.node.op.traits()
            num_actors = max(1, state.node.parallelism or traits.parallelism)

            group = _RayActorGroup(
                node=state.node,
                op_index=state.op_index,
                num_actors=num_actors,
                tokens_per_actor=self._queue_capacity,
                stage_index=self._stage_index,
                stage_name=self._stage_name,
                collect_stats=self._tracking_mode.collects_nodes,
                ctx_services=actor_ctx,
            )
            group.init(self._num_cpus_per_actor, self._runtime_env)
            state.actor_group = group

            state.input_queue = queue.Queue(maxsize=self._queue_capacity)
            state.submit_queue = queue.Queue(maxsize=self._queue_capacity)
            state.result_queue = queue.Queue(maxsize=self._queue_capacity)
            # Bounded by total tokens: submit thread consumes one token
            # from idle_queue before each put, so refs_queue can never
            # exceed num_actors * queue_capacity entries.
            state.refs_queue = queue.Queue(maxsize=num_actors * self._queue_capacity)

            state.submit_thread = threading.Thread(
                target=_submit_loop,
                args=(
                    state.submit_queue,
                    group.actors,
                    group.idle_queue,
                    state.refs_queue,
                    context.stop_event,
                ),
                daemon=True,
                name=f"ray-submit-{state.node.name}",
            )
            state.collector_thread = threading.Thread(
                target=_collector_loop,
                args=(
                    state.refs_queue,
                    state.result_queue,
                    context.stop_event,
                    num_actors,
                    state.node.name,
                ),
                daemon=True,
                name=f"ray-collector-{state.node.name}",
            )
            state.submit_thread.start()
            state.collector_thread.start()

    def _after_run(self, context: ConcurrentRunContext) -> None:
        """Shutdown submit/collector threads and actor groups."""
        for state in self.ops:
            # Signal submit thread to exit.
            if state.submit_thread is not None:
                state.submit_queue.put(_STOP_SUBMIT)

        # Join submit threads first — guarantees refs_queue is sealed.
        for state in self.ops:
            if state.submit_thread is not None:
                state.submit_thread.join(timeout=10.0)
                state.submit_thread = None

        # Now set stop_event so collector threads know no more refs will arrive.
        context.stop_event.set()

        for state in self.ops:
            if state.collector_thread is not None:
                state.collector_thread.join(timeout=10.0)
                state.collector_thread = None

        for state in self.ops:
            if state.actor_group is not None:
                state.actor_group.shutdown()
                state.actor_group = None

    def _schedule_batch(
        self,
        state: _RayOperatorState,
        batch: list[RunnerStreamIn],
        *,
        wait_ns: int,
        context: ConcurrentRunContext,
    ) -> None:
        if not batch or context.stop_event.is_set():
            return

        # Sentinel batches bypass actors — handle inline like ThreadStageRunner.
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

        assert state.actor_group is not None

        seq = state.next_seq
        state.next_seq += 1

        # Put into submit_queue with drain-on-full to avoid deadlock.
        # Increment inflight *after* the put succeeds so an early return
        # cannot leave inflight permanently inflated (which would hang shutdown).
        while True:
            try:
                state.submit_queue.put((batch, seq), timeout=0.1)
                state.inflight.increment()
                break
            except queue.Full:
                # Submit queue is full — drain results to release actors
                # back to idle_queue via _ack_result. The submit thread
                # must then pop from submit_queue before a slot opens, so
                # the next put may not succeed immediately (same pattern
                # as ProcessStageRunner._send_command). The 0.1s timeout
                # above naturally yields to the submit thread.
                self._drain_results(state, self._next_queue_for(state), context)
                if context.stop_event.is_set():
                    return

    def _post_schedule_batch(
        self,
        state: _RayOperatorState,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
    ) -> None:
        """Drain locally-stashed results (sentinels + drain_callback results)."""
        for result in state._local_results:
            self._handle_result(state, result, next_queue, context)
        state._local_results.clear()

    def _ack_result(
        self,
        state: _RayOperatorState,
        result: RunnerResult,
        context: ConcurrentRunContext,
    ) -> None:
        """Acknowledge a result: decrement inflight, recycle actor if alive."""
        if result.from_worker:
            state.inflight.decrement()
            # ack is None when the actor died (see _collector_loop error path).
            # group may be None during shutdown (_after_run / close set it to
            # None after killing actors). Local-bind to avoid TOCTOU race.
            group = state.actor_group
            if result.ack is not None and group is not None:
                group.release(result.ack)

    def close(self, *, hard: bool = False) -> None:
        """Tear down bridge threads, actor groups, and join pump threads."""
        with self._context_lock:
            ctx = self._active_context

        if ctx is not None:
            ctx.stop_event.set()
            self._put_stage_stop(ctx)
            self._join_threads(ctx, hard=hard)

        for state in self.ops:
            # Ensure bridge threads are stopped.
            if state.submit_thread is not None:
                try:
                    state.submit_queue.put_nowait(_STOP_SUBMIT)
                except queue.Full:
                    pass
                state.submit_thread.join(timeout=5.0)
                state.submit_thread = None
            if state.collector_thread is not None:
                state.collector_thread.join(timeout=5.0)
                state.collector_thread = None
            if state.actor_group is not None:
                state.actor_group.shutdown()
                state.actor_group = None
