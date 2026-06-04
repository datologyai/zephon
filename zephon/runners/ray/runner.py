# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Ray-backed stage runner on top of ConcurrentStageRunner."""

from __future__ import annotations

import logging
import os
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import ray

from zephon.core.constants import RunnerStreamIn
from zephon.core.graph import Node, Stage
from zephon.core.notify import is_sentinel
from zephon.observability.config import ExecutionTrackingMode
from zephon.runners.concurrent import (
    ConcurrentOperatorState,
    ConcurrentRunContext,
    ConcurrentStageRunner,
    RunnerResult,
    StopToken,
    WorkerErrorInfo,
    _QueueLike,
)
from zephon.runners.ray.service import _RayActorGroup
from zephon.utils.thread_utils import THREAD_SUPPRESSION_ENV_VARS

logger = logging.getLogger(__name__)

# Minimum interval between opportunistic result sweeps; see _drain_results.
_DRAIN_INTERVAL_NS: int = 2_000_000


def _noop(*_args: Any, **_kwargs: Any) -> None:
    """No-op placeholder for non-serializable callables sent to Ray actors."""


def _available_ray_node_ids() -> list[str]:
    """Return alive Ray node ids with CPU resources in stable order."""
    nodes = [
        node
        for node in ray.nodes()
        if bool(node.get("Alive"))
        and node.get("NodeID") is not None
        and float(node.get("Resources", {}).get("CPU", 0)) > 0
    ]
    nodes.sort(
        key=lambda node: (
            str(node.get("NodeManagerAddress", "")),
            str(node.get("NodeID", "")),
        )
    )
    return [str(node["NodeID"]) for node in nodes]


def _plan_actor_node_ids(num_actors: int, node_ids: list[str]) -> list[str]:
    """Assign actor placements round-robin across Ray node ids."""
    if num_actors <= 0 or not node_ids:
        return []
    return [node_ids[idx % len(node_ids)] for idx in range(num_actors)]


# ---------------------------------------------------------------------------
# Pending-ref bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _PendingRefs:
    """In-flight ObjectRef tracking with O(1) per-actor load queries.

    Replaces the prior ``dict[ObjectRef, int]`` + per-query O(N) scan
    pattern. Only accessed from the pump thread, so no locking.
    """

    refs: dict[ray.ObjectRef, int] = field(default_factory=dict)
    _counts: list[int] = field(default_factory=list)

    def init(self, num_actors: int) -> None:
        self.refs.clear()
        self._counts = [0] * num_actors

    def add(self, ref: ray.ObjectRef, actor_idx: int) -> None:
        self.refs[ref] = actor_idx
        self._counts[actor_idx] += 1

    def pop(self, ref: ray.ObjectRef) -> int:
        actor_idx = self.refs.pop(ref)
        self._counts[actor_idx] -= 1
        return actor_idx

    def count_for(self, actor_idx: int) -> int:
        return self._counts[actor_idx]

    def total(self) -> int:
        return len(self.refs)

    def is_empty(self) -> bool:
        return not self.refs

    def active(self) -> list[ray.ObjectRef]:
        """Snapshot of pending refs; ``ray.wait`` needs a list, not a view."""
        return list(self.refs)

    def least_loaded(self, dead: set[int]) -> int | None:
        """Return the least-loaded live actor index, or None if all are dead."""
        best: int | None = None
        best_count = 0
        for idx, count in enumerate(self._counts):
            if idx in dead:
                continue
            if best is None or count < best_count:
                best = idx
                best_count = count
        return best


# ---------------------------------------------------------------------------
# Operator state
# ---------------------------------------------------------------------------


@dataclass
class _RayOperatorState(ConcurrentOperatorState):
    """Operator state for RemoteStageRunner."""

    actor_group: _RayActorGroup | None = field(default=None, init=False)
    pending_refs: _PendingRefs = field(init=False, default_factory=_PendingRefs)
    dead_actors: set[int] = field(init=False, default_factory=set)

    # Last opportunistic sweep; rate-limits ray.wait polls (pump-thread only).
    last_drain_ns: int = field(default=0, init=False)

    # Sentinel results created inline on the pump thread (same pattern as
    # ThreadStageRunner._local_results). Sentinel batches bypass actors and
    # are processed immediately; stashing here avoids the pump-thread put
    # deadlock the base class documents for _local_results.
    _local_results: list[RunnerResult] = field(init=False, default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class RemoteStageRunner(ConcurrentStageRunner["_RayOperatorState"]):
    """Stage runner that executes operators on per-operator Ray actor groups.

    Each operator in the stage gets its own group of Ray actors. The
    inherited :class:`ConcurrentStageRunner` machinery handles the pump
    threads, feeder, deterministic reordering, stop propagation, and the
    pull-driven ``run()`` iterator. This subclass provides the Ray-specific
    wiring: actor group creation, direct ``ray.wait()`` result collection,
    and combined per-actor + global backpressure.
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
        ray_hard_node_affinity: bool | None = None,
    ) -> None:
        self._num_cpus_per_actor = num_cpus_per_actor
        self._queue_capacity = max(1, queue_capacity)
        if ray_hard_node_affinity is None:
            ray_hard_node_affinity = bool(
                int(os.environ.get("ZEPHON_RAY_HARD_NODE_AFFINITY", "0"))
            )
        self._hard_node_affinity = ray_hard_node_affinity

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
            # Pass through the zero-copy tensor flag — actors don't inherit the
            # driver's os.environ, and it must be set on both ends to take effect.
            zero_copy = os.environ.get("RAY_ENABLE_ZERO_COPY_TORCH_TENSORS")
            if zero_copy is not None:
                default_env_vars["RAY_ENABLE_ZERO_COPY_TORCH_TENSORS"] = zero_copy
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

        Callables cannot be serialized across Ray actor boundaries via
        cloudpickle, so we replace them with no-ops before shipping the
        context to actors.

        TODO: implement Ray-native observability (e.g. via Ray's built-in
        metrics, a dedicated reporting actor, or a similar mechanism to the
        process runner's task_queue approach). Pump-level metrics now flow
        back to the driver, but node-level metrics inside actors are still
        no-ops.
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

    # -- Ray-specific result collection -------------------------------------

    def _sweep_ready_refs(
        self,
        state: _RayOperatorState,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
        *,
        block: bool = False,
        timeout: float = 0.5,
    ) -> int:
        """Resolve completed ObjectRefs via ``ray.wait`` and forward results.

        Args:
            block: If True and refs are pending but none ready, block up
                to ``timeout`` seconds for at least one to complete.
            timeout: Blocking wait timeout in seconds. Ignored if
                ``block`` is False.

        Returns:
            Number of results resolved this call.
        """
        if state.pending_refs.is_empty():
            return 0

        refs = state.pending_refs.active()
        state.pump_timer.note("sweeps")
        state.pump_timer.note("sweep_refs", len(refs))

        # Non-blocking sweep first.
        with state.pump_timer.measure("result_wait"):
            ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
        if not ready and block:
            with state.pump_timer.measure("result_wait"):
                ready, _ = ray.wait(refs, num_returns=1, timeout=timeout)

        resolved = 0
        for ref in ready:
            actor_idx = state.pending_refs.pop(ref)
            try:
                with state.pump_timer.measure("result_collect"):
                    result: RunnerResult = ray.get(ref)
            except Exception as exc:
                # Actor died — synthesize error result.
                state.dead_actors.add(actor_idx)
                num_actors = state.actor_group.num_actors if state.actor_group else 0
                alive = num_actors - len(state.dead_actors)
                # Demote to DEBUG when stop_event is set — actor kills during
                # intentional teardown are expected and shouldn't spam logs.
                log = logger.debug if context.stop_event.is_set() else logger.warning
                log(
                    "Actor %d for op '%s' died: %s (%d/%d actors remaining)",
                    actor_idx,
                    state.node.name,
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
                )
                if alive == 0:
                    logger.error(
                        "All actors for op '%s' are dead, stopping stage",
                        state.node.name,
                    )
                    context.stop_event.set()

            state.pump_timer.note("batches_completed")
            with state.pump_timer.measure("result_handle"):
                self._handle_result(state, result, next_queue, context)
            resolved += 1

        return resolved

    # -- ConcurrentStageRunner hooks ---------------------------------------

    def _drain_results(
        self,
        state: _RayOperatorState,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
    ) -> None:
        """Non-blocking sweep of ready ObjectRefs, rate-limited to a cadence.

        Each sweep is a ``ray.wait`` RPC (~100-200us), so the pump loop's
        per-item call frequency would otherwise dominate pump time on
        fine-grained input. Blocking paths (:meth:`_await_one_result`,
        capacity stalls) bypass this and are never skipped.
        """
        now_ns = time.perf_counter_ns()
        if now_ns - state.last_drain_ns < _DRAIN_INTERVAL_NS:
            return
        state.last_drain_ns = now_ns
        with state.pump_timer.measure_excluding(
            "idle_drain", "result_wait", "result_collect", "result_handle"
        ):
            self._sweep_ready_refs(state, next_queue, context, block=False)

    def _await_one_result(
        self,
        state: _RayOperatorState,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
        timeout: float,
    ) -> None:
        """Block up to ``timeout`` for at least one pending ref to complete."""
        self._sweep_ready_refs(state, next_queue, context, block=True, timeout=timeout)

    def _before_run(self, context: ConcurrentRunContext) -> None:
        """Initialize actor groups and per-operator input queues."""
        actor_ctx = self._build_actor_ctx()
        cluster_node_ids = _available_ray_node_ids()

        for state in self.ops:
            traits = state.node.op.traits()
            num_actors = max(1, state.node.parallelism or traits.parallelism)
            preferred_node_ids = _plan_actor_node_ids(num_actors, cluster_node_ids)

            group = _RayActorGroup(
                node=state.node,
                op_index=state.op_index,
                num_actors=num_actors,
                stage_index=self._stage_index,
                stage_name=self._stage_name,
                collect_stats=self._tracking_mode.collects_nodes,
                ctx_services=actor_ctx,
            )
            group.init(
                self._num_cpus_per_actor,
                self._runtime_env,
                preferred_node_ids=preferred_node_ids or None,
                hard_node_affinity=self._hard_node_affinity,
            )
            state.actor_group = group
            state.pending_refs.init(num_actors)
            state.dead_actors.clear()
            state._local_results.clear()

            state.input_queue = queue.Queue(maxsize=self._queue_capacity)

    def _after_run(self, context: ConcurrentRunContext) -> None:
        """Shutdown actor groups."""
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

        # Sentinel batches bypass actors — handle inline on the pump thread.
        # Stashed in _local_results; _post_schedule_batch drains them.
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

        max_per_actor = self._queue_capacity
        next_queue = self._next_queue_for(state)

        # Two-layer backpressure:
        #   * per-actor cap — fairness across actors
        #   * global in-flight cap — memory bound that also covers results
        #     parked in `pending_results` waiting to close a seq gap
        #     (inflight is decremented in _ack_result AFTER emit downstream,
        #     so it naturally counts submitted-but-not-yet-emitted work).
        while True:
            if context.stop_event.is_set():
                return

            actor_idx = state.pending_refs.least_loaded(state.dead_actors)
            if actor_idx is None:
                context.stop_event.set()
                return

            n_live = state.actor_group.num_actors - len(state.dead_actors)
            max_total = max_per_actor * n_live

            at_actor_cap = state.pending_refs.count_for(actor_idx) >= max_per_actor
            at_total_cap = state.inflight.peek() >= max_total
            if not at_actor_cap and not at_total_cap:
                break

            state.pump_timer.note("capacity_stalls")
            with state.pump_timer.measure_excluding(
                "dispatch_wait", "result_wait", "result_collect", "result_handle"
            ):
                self._sweep_ready_refs(
                    state, next_queue, context, block=True, timeout=0.5
                )

        with state.pump_timer.measure("dispatch_active"):
            seq = state.next_seq
            state.next_seq += 1

            ref = state.actor_group.actors[actor_idx].process.remote(batch, seq)
            state.pending_refs.add(ref, actor_idx)
            state.inflight.increment()
        state.pump_timer.note("batches_submitted")

    def _post_schedule_batch(
        self,
        state: _RayOperatorState,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
    ) -> None:
        """Drain locally-stashed sentinel results."""
        for result in state._local_results:
            state.pump_timer.note("batches_completed")
            with state.pump_timer.measure("result_handle"):
                self._handle_result(state, result, next_queue, context)
        state._local_results.clear()

    def _ack_result(
        self,
        state: _RayOperatorState,
        result: RunnerResult,
        context: ConcurrentRunContext,
    ) -> None:
        """Decrement inflight once the result has been forwarded downstream.

        Called by :meth:`_forward_ready_result` on the base class, *after*
        :meth:`_emit_downstream` has placed the payload into the next
        operator's input queue (or stage output). That ordering makes
        ``inflight`` a precise count of submitted-but-not-yet-emitted work,
        which is the quantity the global backpressure cap bounds.
        """
        if result.from_worker:
            state.inflight.decrement()

    def close(self, *, hard: bool = False) -> None:
        """Tear down actor groups and join pump threads."""
        with self._context_lock:
            ctx = self._active_context

        if ctx is not None:
            ctx.stop_event.set()
            self._put_stage_stop(ctx)
            self._join_threads(ctx, hard=hard)

        for state in self.ops:
            if state.actor_group is not None:
                state.actor_group.shutdown()
                state.actor_group = None
