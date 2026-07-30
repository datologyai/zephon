# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for local stage runners."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Sequence, TypeVar

from zephon._internal.graph import Node, Stage
from zephon._internal.notify import is_sentinel
from zephon._internal.observability.stopwatch import Stopwatch
from zephon._internal.ops.batch import Batch
from zephon._internal.stream import (
    Microbatch,
    RunnerStageIn,
    RunnerStreamIn,
    resolve_lazy_payloads,
)
from zephon.observability.config import ExecutionTrackingMode
from zephon.observability.stats import NodeMetricsDelta
from zephon.ops.accumulators import Accumulator
from zephon.ops.base import BaseOp, OpContext, StageInfo
from zephon.types import SampleBatch, SampleRecord, StreamItem


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
    instances: list[BaseOp] = field(init=False, default_factory=list)
    parallelism: int = field(init=False)
    accumulator_impl: Accumulator[RunnerStreamIn] = field(init=False)
    _preserves_cursor_order: bool = field(init=False)
    _stall_on_epoch_boundary: bool = field(init=False)
    _epoch_floor: int | None = field(init=False, default=None)
    _stalled_sentinels: list[tuple[SampleRecord, int]] = field(
        init=False, default_factory=list
    )

    def __post_init__(self) -> None:
        traits = self.node.op.traits()
        self._preserves_cursor_order = bool(traits.preserves_cursor_order)
        # Batch(drop_last=True) may carry a partial batch across an epoch boundary:
        # checkpoints are cut only after complete batches, and ReplayFilter removes
        # records already delivered before replay reaches Batch. Other operators would
        # need extra replay state; if we add that, make _stall_on_epoch_boundary an
        # explicit operator property again.
        op = self.node.op
        self._stall_on_epoch_boundary = isinstance(op, Batch) and op.drop_last
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
        stage_info = StageInfo(
            stage_index=self.stage_index,
            stage_name=self.stage_name,
            op_index=self.op_index,
            collect_stats=self.collect_stats,
        )

        for _ in range(self.parallelism):
            instance = copy.deepcopy(self.node.op)
            instance.setup(OpContext(dict(base_ctx), stage_info))
            self.instances.append(instance)

        # Get accumulator from operator
        self.accumulator_impl = self.node.op.accumulator(
            deterministic=self.deterministic,
            ctx=base_ctx,
        )

    def _update_epoch_floor(self, elems: Sequence[RunnerStreamIn]) -> None:
        """Track the minimum chunk_id that entered this operator.

        Only tracked for ``preserves_cursor_order=False`` ops — order-preserving
        ops (Batch, passthrough) don't have the cross-chunk state problem.
        """
        if self._preserves_cursor_order:
            return
        for e in elems:
            cid: int | None = None
            if isinstance(e, SampleRecord):
                cid = e.meta.chunk_id
            elif isinstance(e, SampleBatch):
                for r in e.records:
                    c = r.meta.chunk_id
                    if self._epoch_floor is None or c < self._epoch_floor:
                        self._epoch_floor = c
                continue
            if cid is not None and (
                self._epoch_floor is None or cid < self._epoch_floor
            ):
                self._epoch_floor = cid

    def reset_epoch_floor(self) -> None:
        """Reset epoch floor after sentinel flush (new epoch starts)."""
        self._epoch_floor = None

    def _try_release_stalled_sentinels(
        self, ready: list[tuple[list[RunnerStreamIn], int]]
    ) -> None:
        """Release stalled sentinels whose pre-boundary records have drained.

        Sentinels are per-lane, so each is checked against its own lane via
        ``accumulator.try_epoch_reset(boundary_cid, lane_id)``.  Per-lane FIFO
        order is preserved (a lane's later sentinel waits for its earlier one),
        but lanes are independent: a blocked lane does not hold back another
        lane's ready sentinel.
        """
        if not self._stalled_sentinels:
            return  # common case: nothing stalled — skip the rebuild allocation

        blocked_lanes: set[int] = set()
        remaining: list[tuple[SampleRecord, int]] = []
        for sentinel, boundary_cid in self._stalled_sentinels:
            lane = sentinel.meta.lane_id
            if lane in blocked_lanes or not self.accumulator_impl.try_epoch_reset(
                boundary_cid, lane
            ):
                blocked_lanes.add(lane)
                remaining.append((sentinel, boundary_cid))
                continue
            ready.append(([sentinel], 0))
            # Advance epoch floor now that the reset is committed.
            if not self._preserves_cursor_order:
                self._epoch_floor = boundary_cid
        self._stalled_sentinels = remaining

    def _force_release_all_stalled_sentinels(
        self, ready: list[tuple[list[RunnerStreamIn], int]]
    ) -> None:
        """Unconditionally release all stalled sentinels (end-of-stream)."""
        for sentinel, _boundary_cid in self._stalled_sentinels:
            ready.append(([sentinel], 0))
        self._stalled_sentinels.clear()

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

        Sentinel records (tombstones, flush signals, etc.) bypass the
        accumulator entirely and are emitted as individual ready batches.
        This ensures operators never see sentinels unless they create them.

        Flush sentinels trigger ``accumulator.flush(reset=True, lane_id=lane)``
        to drain that lane's buffered data before the sentinel passes
        downstream.
        This is critical: if the sentinel overtakes buffered data,
        downstream stateful operators see the epoch boundary before the
        data, breaking deterministic replay.

        ``Batch(drop_last=True)`` is the exception: if it has pending data at an
        epoch boundary, it keeps the buffer and stalls the sentinel instead of flushing.

        Every other operator must fully flush at epoch boundaries.
        If ``flush(reset=True)`` returns while ``has_pending_data()``
        is still True, the flush contract has been violated and the runner
        raises. This keeps unsupported delayed-reset behavior from slipping
        past the checkpoint/replay contract.

        Stalling semantics ("delayed but guaranteed reset"):

        - The sentinel is held behind buffered data.
        - After each ``push_many()`` call, ``try_epoch_reset()`` checks
          whether all pre-boundary records have left the buffer.
        - Once they have, the accumulator resets its epoch-dependent
          ordering state and the sentinel is released downstream.

        Stalling is currently only supported for ``Batch(drop_last=True)``.
        Its replay story relies on Batch-specific properties: the consumer
        cursor is batch-aligned, the live Batch buffer is empty at every
        checkpoint cut, and the planner inserts ``ReplayFilter`` immediately
        before Batch.  Other stalled operators would need a replay capsule to
        reconstruct cross-epoch consumption state after eviction.

        For ``preserves_cursor_order=False`` ops, flush sentinels also
        advance the epoch floor to ``_boundary_cid``.

        Elements are processed in order — regular elements before a flush
        sentinel enter ``push_many()`` before the flush fires.  This
        preserves the invariant that pre-sentinel data is in the old epoch
        and post-sentinel data starts a new epoch.

        Args:
            elems: Input elements to accumulate.
            force: If True, flush all remaining buffered elements.

        Returns:
            List of (batch, wait_ns) tuples ready for worker dispatch.
        """
        ready: list[tuple[list[RunnerStreamIn], int]] = []

        if not elems and not force:
            return ready

        # Collect non-sentinel elements between sentinel boundaries.
        # When a sentinel is encountered, the preceding regular elements
        # are flushed through push_many() first so they enter the
        # accumulator before the sentinel triggers its flush/stall.
        pre_sentinel: list[RunnerStreamIn] = []

        for e in elems:
            if is_sentinel(e):
                # Flush preceding regular elements into the accumulator.
                if pre_sentinel:
                    if self.accumulator_impl.reads_payload:
                        resolve_lazy_payloads(pre_sentinel)
                    self._update_epoch_floor(pre_sentinel)
                    batches = self.accumulator_impl.push_many(pre_sentinel)
                    ready.extend(batches)
                    self._try_release_stalled_sentinels(ready)
                    pre_sentinel = []

                # Tombstones contribute to epoch floor tracking — their chunk_id
                # must constrain eviction even though they bypass the accumulator.
                # Flush sentinels have dummy chunk_id=0 and must NOT affect the floor.
                if not e.meta.is_flush_sentinel:
                    self._update_epoch_floor([e])

                if e.meta.is_flush_sentinel:
                    boundary_cid_tag = e.meta.tags.get("_boundary_cid")
                    bcid = int(boundary_cid_tag) if boundary_cid_tag is not None else 0
                    lane = e.meta.lane_id
                    stalled = False

                    if (
                        self._stall_on_epoch_boundary
                        and self.accumulator_impl.has_pending_data(lane)
                    ):
                        # Explicit stall: skip flush, preserve buffer across
                        # epoch boundary.  The sentinel is held behind the
                        # buffered data and released via try_epoch_reset()
                        # once all pre-boundary records have been emitted.
                        self._stalled_sentinels.append((e, bcid))
                        stalled = True
                    else:
                        # Flush accumulator to drain buffered data before the
                        # sentinel passes downstream.
                        flushed = self.accumulator_impl.flush(reset=True, lane_id=lane)
                        ready.extend(flushed)

                        if self.accumulator_impl.has_pending_data(lane):
                            op_name = type(self.node.op).__name__
                            raise RuntimeError(
                                f"{op_name}: flush(reset=True) left "
                                "pending data at an epoch boundary. "
                                "Intentional stalling is only supported for "
                                "Batch(drop_last=True); other operators must "
                                "fully flush and reset at the sentinel."
                            )
                        else:
                            ready.append(([e], 0))

                    # Epoch floor advance (preserves_cursor_order=False only).
                    # Only advance when the sentinel actually passes through.
                    # When stalled, the floor stays at the old value so that
                    # accum_floor blocks eviction until the delayed reset.
                    if not stalled and not self._preserves_cursor_order:
                        if boundary_cid_tag is not None:
                            self._epoch_floor = bcid
                        else:
                            self.reset_epoch_floor()
                else:
                    # Non-flush sentinels (tombstones) pass through immediately.
                    ready.append(([e], 0))
            else:
                pre_sentinel.append(e)

        # Process remaining regular elements after the last sentinel.
        if pre_sentinel:
            if self.accumulator_impl.reads_payload:
                resolve_lazy_payloads(pre_sentinel)
            self._update_epoch_floor(pre_sentinel)
            batches = self.accumulator_impl.push_many(pre_sentinel)
            ready.extend(batches)
            self._try_release_stalled_sentinels(ready)

        # Force-flush on upstream close.
        if force:
            ready.extend(self.accumulator_impl.flush())
            self._force_release_all_stalled_sentinels(ready)

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

    def epoch_floor(self) -> int | None:
        """Return the minimum epoch floor across all operator states.

        The epoch floor is the lowest chunk_id that influenced any
        ``preserves_cursor_order=False`` accumulator since the last sentinel
        flush.  Chunks below this floor are safe to consider for eviction
        (subject to the ``all_done`` bitmap check).
        """
        floor: int | None = None
        for op_state in self.ops:
            wm = op_state._epoch_floor
            if wm is not None:
                floor = min(floor, wm) if floor is not None else wm
        return floor

    @abstractmethod
    def close(self, *, hard: bool = False) -> None:
        """Tear down all runner resources.

        Args:
            hard: When True, use minimal join timeouts for fast exit.
        """

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
