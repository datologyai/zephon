# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for local stage runners."""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar

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
from zephon.core.traits import Buffering
from zephon.observability.config import ExecutionTrackingMode
from zephon.observability.stats import NodeMetricsDelta
from zephon.observability.stopwatch import Stopwatch
from zephon.ops.batch import Batch


@dataclass
class BaseOperatorState:
    """Common operator wiring shared across stage runners."""

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
    buffer_cfg: Buffering | None = field(init=False)
    buffer: list[RunnerStreamIn] = field(init=False, default_factory=list)
    buffer_ts_ns: list[int] = field(
        init=False, default_factory=list
    )  # arrival timestamp per buffered element
    first_ts_ns: Optional[int] = field(init=False, default=None)

    def __post_init__(self) -> None:
        traits = self.node.op.traits()
        self.parallelism = max(1, self.node.parallelism or traits.parallelism or 1)
        if (
            self.deterministic
            and isinstance(self.node.op, Batch)
            and self.parallelism != 1
        ):
            raise RuntimeError(
                "Batch operator must run with parallelism=1 in deterministic mode"
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

        cfg = self.node.op.buffering()
        if cfg is None:
            self.buffer_cfg = None
        else:
            self.buffer_cfg = Buffering(
                max_batch=cfg.max_batch,
                max_latency_ms=(
                    cfg.max_latency_ms
                    if (not self.deterministic or self.allow_latency_flush)
                    else None
                ),
            )

    def reset_buffers(self) -> None:
        self.buffer = []
        self.buffer_ts_ns = []
        self.first_ts_ns = None

    def enqueue(
        self, elems: Sequence[RunnerStreamIn], *, force: bool = False
    ) -> list[tuple[list[RunnerStreamIn], int]]:
        ready: list[tuple[list[RunnerStreamIn], int]] = []
        if not elems and not force:
            return ready
        if self.buffer_cfg is None:
            # Preserve the upstream microbatch as-as (no internal batching),
            # so process_many can vectorize when available.
            if elems:
                batch = elems if isinstance(elems, list) else list(elems)
                ready.append((batch, 0))
            # there is no internal buffer in the 'None' path, so nothing to flush on 'force'
            return ready

        now_ns = time.perf_counter_ns()
        for elem in elems:
            arrival_ns = time.perf_counter_ns()
            if not self.buffer:
                self.first_ts_ns = arrival_ns
            self.buffer.append(elem)
            self.buffer_ts_ns.append(arrival_ns)
            if (
                self.buffer_cfg.max_batch
                and len(self.buffer) >= self.buffer_cfg.max_batch
            ):
                ready.append(self._pop_batch(self.buffer_cfg.max_batch, now_ns))
            elif (
                self.buffer_cfg.max_latency_ms is not None
                and self.first_ts_ns is not None
                and (now_ns - self.first_ts_ns) / 1_000_000
                >= self.buffer_cfg.max_latency_ms
            ):
                ready.append(self._drain_buffer(now_ns=now_ns))
            now_ns = time.perf_counter_ns()
        if force and self.buffer:
            ready.append(self._drain_buffer())
        return ready

    def _pop_batch(self, size: int, now_ns: int) -> tuple[list[RunnerStreamIn], int]:
        current_ns = now_ns
        chunk = self.buffer[:size]
        timestamps = self.buffer_ts_ns[:size]
        self.buffer = self.buffer[size:]
        self.buffer_ts_ns = self.buffer_ts_ns[size:]
        wait_ns = 0
        if timestamps:
            wait_ns = max(0, current_ns - timestamps[0])
        if not self.buffer:
            self.first_ts_ns = None
        else:
            self.first_ts_ns = self.buffer_ts_ns[0]
        return list(chunk), wait_ns

    def _drain_buffer(
        self, *, now_ns: Optional[int] = None
    ) -> tuple[list[RunnerStreamIn], int]:
        current_ns = now_ns if now_ns is not None else time.perf_counter_ns()
        return self._pop_batch(len(self.buffer), current_ns)

    def finalize(self) -> Microbatch:
        tail: Microbatch = []
        for instance in self.instances:
            tail.extend(instance.finalize())
        return tail


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
        for state in self.ops:
            tail = state.instances[0].finalize()
            if tail:
                raise RuntimeError("Finalize emitted data in run_one path")
        if not isinstance(value, (SampleRecord, SampleBatch)):
            raise TypeError(
                f"Stage produced unsupported element {type(value)!r} in run_one()"
            )
        return value
