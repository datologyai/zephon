# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Ray actor and operator pool primitives used by the remote runner."""

from __future__ import annotations

import logging
import os
import queue
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import cloudpickle
import ray
from ray.util.actor_pool import ActorPool

from zephon.core.graph import Node
from zephon.core.op_base import Op, OpContext
from zephon.observability.size_estimator import estimate_bytes
from zephon.runners.concurrent import RunnerResult, WorkerErrorInfo
from zephon.utils.thread_utils import suppress_library_threads

logger = logging.getLogger(__name__)

_SHUTDOWN_DEBUG = bool(os.environ.get("ZEPHON_DEBUG_SHUTDOWN"))


# Uses print(stderr) instead of logger so it works on remote Ray nodes
# without having to configure logging across the cluster.
# TODO: replace with proper structured logging once we have a Ray log aggregation story.
def _shutdown_debug(msg: str) -> None:
    """Print timestamped shutdown debug message when ZEPHON_DEBUG_SHUTDOWN=1."""
    if _SHUTDOWN_DEBUG:
        ts = time.strftime("%H:%M:%S", time.localtime())
        ms = int((time.time() % 1) * 1000)
        print(f"[Shutdown {ts}.{ms:03d}] {msg}", file=sys.stderr, flush=True)


@ray.remote
class RaySingleOpActor:
    """Actor that owns and runs a single operator."""

    def __init__(
        self,
        op_bytes: bytes,
        ctx_services: dict[str, Any],
        stage_index: int,
        stage_name: str,
        op_index: int,
        op_name: str,
        collect_stats: bool,
    ) -> None:
        """Initialize the actor with a single operator instance."""
        suppress_library_threads()

        self._op: Op[Any, Any] = cloudpickle.loads(op_bytes)
        self._op_index = op_index
        self._op_name = op_name
        self._collect_stats = collect_stats

        ctx = OpContext(dict(ctx_services))
        self._op.setup(
            ctx,
            stage_index,
            stage_name,
            op_index,
            collect_stats,
        )

    def process(self, batch: list[Any], seq: int) -> RunnerResult:
        """Process one input batch and return a RunnerResult."""
        try:
            consumed_elements = len(batch)
            consumed_bytes = estimate_bytes(batch) if self._collect_stats else 0

            start_ns = time.perf_counter_ns() if self._collect_stats else 0

            result = self._op.process_many(batch)

            proc_ns = time.perf_counter_ns() - start_ns if self._collect_stats else 0

            return RunnerResult(
                seq=seq,
                payload=result,
                wait_ns=0,
                consumed_elements=consumed_elements,
                consumed_bytes=consumed_bytes,
                queue_depth_snapshot=0,
                proc_ns=proc_ns,
                collect_metrics=self._collect_stats,
                from_worker=True,
            )
        except Exception as exc:
            tb = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            logger.error("Actor %s error: %s", self._op_name, tb)
            return RunnerResult(
                seq=seq,
                payload=[],
                wait_ns=0,
                consumed_elements=0,
                consumed_bytes=0,
                queue_depth_snapshot=0,
                proc_ns=0,
                collect_metrics=False,
                error=WorkerErrorInfo(
                    exc_type=type(exc).__name__,
                    message=tb,
                    formatted_traceback=tb,
                ),
                from_worker=True,
            )


# ---------------------------------------------------------------------------
# _RayOperatorPool
# ---------------------------------------------------------------------------


@dataclass
class _RayOperatorPool:
    """Actor pool for a single operator.

    Spawns Ray actors and manages them through a ``ray.util.ActorPool``.
    """

    node: Node
    op_index: int
    num_actors: int
    stage_index: int
    stage_name: str
    collect_stats: bool
    ctx_services: dict[str, Any]

    actors: list[ray.actor.ActorHandle] = field(default_factory=list)
    pool: ActorPool | None = field(default=None, init=False)

    def init(
        self,
        num_cpus_per_actor: float,
        runtime_env: dict[str, Any] | None = None,
    ) -> None:
        """Create actors and build the ActorPool.

        Args:
            num_cpus_per_actor: CPU fraction each actor reserves.
            runtime_env: Optional Ray runtime environment dict.  Ray actors
                on the *same* node inherit the driver's OS-level env vars,
                but actors scheduled on *other* nodes do not — they start
                from that node's default environment.  Use this to forward
                env vars (e.g. ``{"env_vars": {"UV_INDEX": "..."}}``) or
                specify pip dependencies / working-dir overrides so that
                remote actors see the same environment as the driver.
        """
        op_bytes = cloudpickle.dumps(self.node.op)

        options: dict[str, Any] = {
            "num_cpus": num_cpus_per_actor,
            # Allow unlimited queued calls per actor so that submit() never
            # raises.  Back-pressure is handled at the runner level (the
            # ConcurrentStageRunner controls how many batches are in-flight)
            # rather than inside Ray's per-actor call queue.
            "max_pending_calls": -1,
        }
        if runtime_env is not None:
            options["runtime_env"] = runtime_env

        for _ in range(self.num_actors):
            actor = RaySingleOpActor.options(**options).remote(
                op_bytes,
                self.ctx_services,
                self.stage_index,
                self.stage_name,
                self.op_index,
                self.node.name,
                self.collect_stats,
            )
            self.actors.append(actor)

        self.pool = ActorPool(self.actors)

    def submit(self, batch: list[Any], seq: int) -> None:
        """Submit a batch to the pool. Blocks if all actors are busy."""
        assert self.pool is not None
        self.pool.submit(lambda a, v: a.process.remote(v, seq), batch)

    def has_next(self) -> bool:
        """Return True if there are pending results."""
        assert self.pool is not None
        return self.pool.has_next()

    def get_next_unordered(self, timeout: float | None = None) -> RunnerResult:
        """Return the next completed result (completion order)."""
        assert self.pool is not None
        return self.pool.get_next_unordered(timeout=timeout)

    def shutdown(self) -> None:
        """Kill all actors and clear the pool."""
        _shutdown_debug(f"_RayOperatorPool.shutdown: killing {len(self.actors)} actors")

        # Drain any pending results to avoid warnings
        if self.pool is not None:
            while self.pool.has_next():
                try:
                    self.pool.get_next_unordered(timeout=0)
                except Exception:
                    break

        for actor in self.actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                logger.debug("Failed to kill actor during shutdown", exc_info=True)

        self.pool = None
        self.actors = []

        _shutdown_debug("_RayOperatorPool.shutdown: done")


# ---------------------------------------------------------------------------
# _RayResultQueueAdapter — adapter for _QueueLike protocol
# ---------------------------------------------------------------------------


class _RayResultQueueAdapter:
    """Adapts _RayOperatorPool to the _QueueLike protocol.

    This lets ConcurrentStageRunner._drain_results() pull results from a
    Ray actor pool using the same get/get_nowait/empty interface it uses
    for thread and process result queues.

    Write methods (put/put_nowait) raise ``NotImplementedError`` because
    Ray actors push results internally — this adapter is read-only.
    """

    def __init__(self, pool: _RayOperatorPool) -> None:
        self._pool = pool

    def put(
        self, item: RunnerResult, /, block: bool = True, timeout: float | None = None
    ) -> None:
        """Not supported — Ray actors push results internally."""
        raise NotImplementedError("Ray result queue adapter is read-only")

    def put_nowait(self, item: RunnerResult, /) -> None:
        """Not supported — Ray actors push results internally."""
        raise NotImplementedError("Ray result queue adapter is read-only")

    def get_nowait(self) -> RunnerResult:
        """Non-blocking drain — raises queue.Empty if nothing ready."""
        try:
            return self._pool.get_next_unordered(timeout=0)
        except (StopIteration, TimeoutError):
            raise queue.Empty

    def get(self, block: bool = True, timeout: float | None = None) -> RunnerResult:
        """Blocking drain with optional timeout."""
        if not block:
            return self.get_nowait()
        try:
            return self._pool.get_next_unordered(timeout=timeout)
        except (StopIteration, TimeoutError):
            raise queue.Empty

    def empty(self) -> bool:
        """Return True when no results are pending."""
        return not self._pool.has_next()

    def qsize(self) -> int:
        """Approximate pending count — always 0 or 1.

        ActorPool doesn't expose a real pending count, so this only
        distinguishes "empty" from "has at least one result".
        """
        return 0 if self.empty() else 1
