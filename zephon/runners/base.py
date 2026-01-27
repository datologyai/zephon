# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for local stage runners."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Sequence, TypeVar

from zephon.core.accumulators import Accumulator
from zephon.core.constants import (
    Microbatch,
    RunnerStageIn,
    RunnerStreamIn,
    SampleBatch,
    SampleRecord,
    StreamItem,
)
from zephon.core.graph import Node, Stage
from zephon.core.op_base import Op, OpContext
from zephon.observability.config import ExecutionTrackingMode
from zephon.observability.stats import NodeMetricsDelta
from zephon.observability.stopwatch import Stopwatch


@dataclass
class BaseOperatorState:
    """Common operator wiring shared across stage runners.

    Accumulators and Buffering
    --------------------------
    Each operator provides an accumulator via the ``accumulator()`` method.
    The accumulator runs on the pump thread and defines invocation batch
    boundaries for parallel workers. This ensures deterministic execution
    regardless of parallelism level.

    The ``enqueue()`` method delegates to the accumulator's ``push_many()``
    and ``flush()`` methods to determine when batches are ready.
    """

    node: Node
    deterministic: bool
    ctx_proto: dict[str, Any]
    allow_latency_flush: bool = False
    stage_index: int = 0
    stage_name: str = ""
    op_index: int = 0
    collect_stats: bool = False
    instances: list[Op[RunnerStreamIn, StreamItem]] = field(
        init=False, default_factory=list
    )
    parallelism: int = field(init=False)
    accumulator_impl: Accumulator[RunnerStreamIn] = field(init=False)

    def __post_init__(self) -> None:
        traits = self.node.op.traits()
        self.parallelism = max(1, self.node.parallelism or traits.parallelism or 1)
        # Enforce parallelism=1 for operators that require serial state in deterministic mode
        if (
            self.deterministic
            and traits.requires_serial_state
            and self.parallelism != 1
        ):
            op_name = type(self.node.op).__name__
            raise RuntimeError(
                f"{op_name} operator must run with parallelism=1 in deterministic mode "
                + "because it requires serial state (requires_serial_state=True)"
            )

        base_ctx = dict(self.ctx_proto)
        base_ctx.setdefault("deterministic", self.deterministic)

        for _ in range(self.parallelism):
            instance = copy.deepcopy(self.node.op)
            ctx = OpContext(dict(base_ctx))
            instance.setup(
                ctx,
                self.stage_index,
                self.stage_name,
                self.op_index,
                self.collect_stats,
            )
            self.instances.append(instance)

        # Get accumulator from operator
        self.accumulator_impl = self.node.op.accumulator(
            deterministic=self.deterministic,
            ctx=base_ctx,
        )

    def reset_buffers(self) -> None:
        """Reset the accumulator by recreating it.

        This is called during shutdown cleanup to ensure no buffered state
        persists across runs when a runner is reused.
        """
        self.accumulator_impl = self.node.op.accumulator(
            deterministic=self.deterministic,
            ctx=self.ctx_proto,
        )

    def enqueue(
        self, elems: Sequence[RunnerStreamIn], *, force: bool = False
    ) -> list[tuple[list[RunnerStreamIn], int]]:
        """Accumulate elements and return ready batches.

        Uses the operator's accumulator to determine batch boundaries.
        The accumulator runs on the pump thread (serial) and maintains
        any cross-invocation state needed for deterministic batching.

        Args:
            elems: Input elements to accumulate.
            force: If True, flush all remaining buffered elements.

        Returns:
            List of (batch, wait_ns) tuples ready for worker dispatch.
        """
        ready: list[tuple[list[RunnerStreamIn], int]] = []

        if not elems and not force:
            return ready

        # Push elements through accumulator
        if elems:
            ready.extend(self.accumulator_impl.push_many(elems))

        # Flush remaining on force
        if force:
            ready.extend(self.accumulator_impl.flush())

        return ready


StateT = TypeVar("StateT", bound=BaseOperatorState)


class StageRunnerBase(Generic[StateT], ABC):
    """Skeleton for concrete stage runners."""

    def __init__(
        self,
        stage: Stage,
        ctx_services: dict[str, Any],
        max_workers: int,
        *,
        prefetch_capacity: int = 0,
        deterministic: bool = False,
        allow_latency_flush_in_deterministic: bool = True,
        stage_index: int = 0,
        tracking_mode: ExecutionTrackingMode = ExecutionTrackingMode.OFF,
        stage_output_mode: str = "microbatches",
    ) -> None:
        if stage_output_mode not in {"microbatches", "stream_items"}:
            raise ValueError(f"Unsupported stage_output_mode '{stage_output_mode}'.")
        self._stage = stage
        self._ctx_services = dict(ctx_services)
        self._max_workers = max(1, max_workers)
        self._prefetch_capacity = max(0, prefetch_capacity)
        self._stage_index = stage_index
        self._tracking_mode = tracking_mode
        self._emit_microbatches = stage_output_mode == "microbatches"
        self._record_node_metrics: Callable[[NodeMetricsDelta], None] = (
            self._ctx_services["record_node_metrics"]
        )
        self._metrics_meta: list[tuple[int, str, int, str]] = []
        self._node_sw = Stopwatch(self._tracking_mode.collects_nodes)
        self._stage_name = stage.name or f"stage_{stage_index}"

        self.ops: list[StateT] = []
        collect_op_stats = self._tracking_mode.collects_nodes
        for op_index, node in enumerate(stage.nodes):
            state = self._make_operator_state(
                node=node,
                op_index=op_index,
                deterministic=deterministic,
                ctx_proto=self._ctx_services,
                allow_latency_flush=allow_latency_flush_in_deterministic,
                stage_index=stage_index,
                stage_name=self._stage_name,
                collect_op_stats=collect_op_stats,
            )
            self.ops.append(state)
            self._metrics_meta.append(
                (self._stage_index, self._stage_name, op_index, node.name)
            )

    @abstractmethod
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
    ) -> StateT:
        """Instantiate a concrete operator state for this runner."""

    @staticmethod
    def _coerce_to_batch(elem: RunnerStageIn) -> Sequence[RunnerStreamIn]:
        if isinstance(elem, list):
            return elem
        return [elem]

    @staticmethod
    def _normalize_passthrough_batch(elem: RunnerStageIn) -> Microbatch:
        if isinstance(elem, list):
            batch = elem
            for item in batch:
                if not isinstance(item, (SampleRecord, SampleBatch)):  # pyright: ignore[reportUnnecessaryIsInstance]
                    raise TypeError(
                        "Passthrough stage received unsupported element "
                        + f"{type(item)!r} inside microbatch"
                    )
            return batch
        if isinstance(elem, (SampleRecord, SampleBatch)):
            return [elem]
        raise TypeError(
            "Passthrough stage received unsupported element " + f"{type(elem)!r}"
        )

    def run_one(self, elem: RunnerStreamIn) -> StreamItem:
        if not self.ops:
            if not isinstance(elem, (SampleRecord, SampleBatch)):
                raise TypeError(
                    f"Passthrough stage received unsupported element {type(elem)!r}"
                )
            return elem
        value: Any = elem
        for state in self.ops:
            instance = state.instances[0]
            outputs = instance.process_one(value)
            if len(outputs) != 1:
                raise RuntimeError("Indexable path requires 1->1 ops through the stage")
            value = outputs[0]
        if not isinstance(value, (SampleRecord, SampleBatch)):
            raise TypeError(
                f"Stage produced unsupported element {type(value)!r} in run_one()"
            )
        return value
