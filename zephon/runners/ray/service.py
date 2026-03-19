# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Ray actor and actor group primitives used by the remote runner."""

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
# _RayActorGroup
# ---------------------------------------------------------------------------


@dataclass
class _RayActorGroup:
    """Manages actor lifecycle and idle-actor tracking.

    Direct actor management gives us per-actor backpressure via an
    ``idle_queue``.  Each actor is seeded with ``tokens_per_actor``
    tokens so that multiple batches can be in-flight to the same actor
    simultaneously (double-buffering).  The submit thread consumes one
    token per dispatch; ``_ack_result`` returns one token per completed
    result, maintaining the invariant that at most ``tokens_per_actor``
    batches are queued/processing on any single actor.
    """

    node: Node
    op_index: int
    num_actors: int
    tokens_per_actor: int
    stage_index: int
    stage_name: str
    collect_stats: bool
    ctx_services: dict[str, Any]

    actors: list[ray.actor.ActorHandle] = field(default_factory=list)
    idle_queue: queue.Queue[int] = field(init=False)

    def init(
        self,
        num_cpus_per_actor: float,
        runtime_env: dict[str, Any] | None = None,
    ) -> None:
        """Create actors and populate the idle queue.

        Args:
            num_cpus_per_actor: CPU fraction each actor reserves.
            runtime_env: Optional Ray runtime environment dict.
        """
        self.idle_queue = queue.Queue(maxsize=self.num_actors * self.tokens_per_actor)

        op_bytes = cloudpickle.dumps(self.node.op)

        options: dict[str, Any] = {
            "num_cpus": num_cpus_per_actor,
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

        # Seed idle queue: each actor gets tokens_per_actor tokens so
        # that the next batch is already in the actor's mailbox when it
        # finishes the current one (double-buffering).
        for i in range(self.num_actors):
            for _ in range(self.tokens_per_actor):
                self.idle_queue.put(i)

    def release(self, actor_idx: int) -> None:
        """Return an actor to the idle queue (called by _ack_result)."""
        self.idle_queue.put(actor_idx)

    def shutdown(self) -> None:
        """Kill all actors."""
        _shutdown_debug(f"_RayActorGroup.shutdown: killing {len(self.actors)} actors")

        for actor in self.actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                logger.debug("Failed to kill actor during shutdown", exc_info=True)

        self.actors = []

        _shutdown_debug("_RayActorGroup.shutdown: done")
