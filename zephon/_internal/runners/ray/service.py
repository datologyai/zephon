# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Ray actor and actor group primitives used by the remote runner."""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, ClassVar

import cloudpickle
import ray
from ray.exceptions import RayError
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from zephon._internal.graph import Node
from zephon._internal.observability.size_estimator import estimate_bytes
from zephon._internal.op_base import Op
from zephon._internal.runners.concurrent import RunnerResult, WorkerErrorInfo
from zephon._internal.utils.thread_utils import suppress_library_threads
from zephon.ops.base import OpContext

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

    def get_node_id(self) -> str:
        """Return the Ray node id this actor is running on.

        Used by ``_RayActorGroup.init`` as a placement-readiness probe under
        hard node affinity: an actor that responds confirms it has been
        scheduled.
        """
        return ray.get_runtime_context().get_node_id()

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
    """Manages actor lifecycle for a single operator.

    Holds the list of Ray actor handles and owns their init/shutdown.
    Backpressure has moved into the runner (combined per-actor +
    global-in-flight cap on the pump thread), so this class no longer
    tracks idle actors or tokens.
    """

    _PLACEMENT_TIMEOUT_S: ClassVar[float] = 60.0

    node: Node
    op_index: int
    num_actors: int
    stage_index: int
    stage_name: str
    collect_stats: bool
    ctx_services: dict[str, Any]

    actors: list[ray.actor.ActorHandle] = field(default_factory=list)

    def init(
        self,
        num_cpus_per_actor: float,
        runtime_env: dict[str, Any] | None = None,
        preferred_node_ids: list[str] | None = None,
        hard_node_affinity: bool = False,
    ) -> None:
        """Create actors.

        Args:
            num_cpus_per_actor: CPU fraction each actor reserves.
            runtime_env: Optional Ray runtime environment dict.
            preferred_node_ids: Optional per-actor Ray node id plan; each
                entry pins the matching actor via NodeAffinity. If provided,
                the list length must equal ``num_actors``.
            hard_node_affinity: If True the plan is strict; otherwise a soft
                hint Ray may override.

        Raises:
            RuntimeError: When ``hard_node_affinity`` is True and the
                resulting actors fail to land on their requested nodes
                (either timing out in PENDING past ``_PLACEMENT_TIMEOUT_S``
                or being rejected synchronously by Ray). Half-spawned
                actors are shut down before re-raising.
        """
        if (
            preferred_node_ids is not None
            and len(preferred_node_ids) != self.num_actors
        ):
            raise ValueError(
                f"preferred_node_ids has {len(preferred_node_ids)} entries but "
                f"num_actors is {self.num_actors}; caller must supply a full plan or None."
            )

        op_bytes = cloudpickle.dumps(self.node.op)

        options: dict[str, Any] = {
            "num_cpus": num_cpus_per_actor,
        }
        if runtime_env is not None:
            options["runtime_env"] = runtime_env

        for idx in range(self.num_actors):
            actor_options = dict(options)
            if preferred_node_ids is not None:
                actor_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                    node_id=preferred_node_ids[idx],
                    soft=not hard_node_affinity,
                )
            actor = RaySingleOpActor.options(**actor_options).remote(
                op_bytes,
                self.ctx_services,
                self.stage_index,
                self.stage_name,
                self.op_index,
                self.node.name,
                self.collect_stats,
            )
            self.actors.append(actor)

        if hard_node_affinity and preferred_node_ids is not None and self.actors:
            self._probe_hard_placement(preferred_node_ids)

    def _probe_hard_placement(self, planned: list[str]) -> None:
        """Verify hard-pinned actors have actually been scheduled.

        Under ``soft=False`` Ray will leave an actor PENDING indefinitely if
        the requested node is unavailable, and the pump thread later blocks
        on ``ray.wait`` for results that never come. Two failure modes are
        converted to a descriptive RuntimeError here:

        * **Pending past the deadline** — the requested node existed at
          snapshot time but was unavailable when Ray tried to schedule.
        * **Immediate Ray rejection** — Ray detected the affinity target
          is unknown / infeasible and surfaced a :class:`RayError` on
          ``ray.get`` (e.g. ``ActorUnschedulableError``).

        Half-spawned actors are shut down before raising so the group
        doesn't leak handles.
        """
        refs = [actor.get_node_id.remote() for actor in self.actors]
        _, pending = ray.wait(
            refs, num_returns=len(refs), timeout=self._PLACEMENT_TIMEOUT_S
        )
        if pending:
            self.shutdown()
            raise RuntimeError(
                f"Actor placement timed out for op '{self.node.name}': "
                f"{len(pending)}/{len(refs)} actors still pending after "
                f"{self._PLACEMENT_TIMEOUT_S}s. Requested node ids: {planned}."
            )
        try:
            ray.get(refs)
        except RayError as exc:
            self.shutdown()
            raise RuntimeError(
                f"Actor placement failed for op '{self.node.name}': Ray rejected "
                f"the requested placement: {exc}. Requested node ids: {planned}."
            ) from exc

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
