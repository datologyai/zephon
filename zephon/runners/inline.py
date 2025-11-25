# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Inline stage runner that executes operators synchronously."""

import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Literal, Sequence, cast

from zephon.core.constants import (
    Microbatch,
    RunnerStageIn,
    RunnerStageOut,
    RunnerStreamIn,
)
from zephon.core.graph import Node, Stage
from zephon.core.op_base import Op
from zephon.observability.config import ExecutionTrackingMode
from zephon.observability.size_estimator import estimate_bytes
from zephon.observability.stats import NodeMetricsDelta
from zephon.runners.base import BaseOperatorState, StageRunnerBase
from zephon.utils import buffered_iterable


@dataclass
class _InlineOperatorState(BaseOperatorState):
    rr_cursor: int = field(init=False, default=0)

    def acquire_instance(self) -> Op[RunnerStreamIn, Any]:
        instance = self.instances[self.rr_cursor]
        self.rr_cursor = (self.rr_cursor + 1) % len(self.instances)
        return instance


class InlineStageRunner(StageRunnerBase[_InlineOperatorState]):
    """Run a stage synchronously without per-operator worker threads."""

    _OperatorState = _InlineOperatorState

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
        stage_output_mode: Literal["microbatches", "stream_items"] = "microbatches",
    ) -> None:
        self._context_lock = threading.Lock()
        self._active = False
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
    ) -> _InlineOperatorState:
        return _InlineOperatorState(
            node=node,
            deterministic=deterministic,
            ctx_proto=ctx_proto,
            allow_latency_flush=allow_latency_flush,
            stage_index=stage_index,
            stage_name=stage_name,
            op_index=op_index,
            collect_stats=collect_op_stats,
        )

    def _emit_stage_output(self, elements: Microbatch) -> Iterator[RunnerStageOut]:
        if not elements:
            return iter(())
        if self._emit_microbatches:
            return iter((elements,))
        return iter(elements)

    def _run_passthrough(
        self, upstream: Iterable[RunnerStageIn], stop_event: threading.Event
    ) -> Iterator[RunnerStageOut]:
        for elem in upstream:
            if stop_event.is_set():
                break
            batch = self._normalize_passthrough_batch(elem)
            yield from self._emit_stage_output(batch)

    def _apply_operator_state(
        self,
        state: _InlineOperatorState,
        inputs: Sequence[RunnerStreamIn],
        *,
        force: bool,
    ) -> Microbatch:
        ready = state.enqueue(inputs, force=force)
        outputs: Microbatch = []
        for batch, wait_ns in ready:
            outputs.extend(self._process_batch(state, batch, wait_ns))
        return outputs

    def _process_batch(
        self,
        state: _InlineOperatorState,
        batch: list[RunnerStreamIn],
        wait_ns: int,
    ) -> Microbatch:
        if not batch:
            return []
        instance = state.acquire_instance()

        collect_stats = self._tracking_mode.collects_nodes
        consumed_elements = len(batch) if collect_stats else 0
        consumed_bytes = estimate_bytes(batch) if collect_stats else 0
        metrics_meta = self._metrics_meta[state.op_index] if collect_stats else None

        start_ns = self._node_sw.start()
        try:
            try:
                outputs: Microbatch = instance.process_many(batch)
            except (NotImplementedError, AttributeError):
                out: Microbatch = []
                for element in batch:
                    out.extend(instance.process_one(element))
                outputs = out
        finally:
            proc_ns = self._node_sw.elapsed(start_ns)

        if collect_stats and metrics_meta is not None:
            stage_index, stage_name, op_index, op_name = metrics_meta
            produced_elements = len(outputs)
            produced_bytes = estimate_bytes(outputs)
            delta = NodeMetricsDelta(
                stage_index=stage_index,
                op_index=op_index,
                stage_name=stage_name,
                name=op_name,
                processed_ns=proc_ns,
                produced_elements=produced_elements,
                consumed_elements=consumed_elements,
                produced_bytes=produced_bytes,
                consumed_bytes=consumed_bytes,
                wait_ns=wait_ns,
                max_queue_depth=-1,
                min_processing_ns=proc_ns,
                max_processing_ns=proc_ns,
            )
            self._record_node_metrics(delta)

        return outputs

    def _process_pipeline(
        self,
        payloads: Sequence[RunnerStreamIn],
        *,
        force: bool,
    ) -> Microbatch:
        current: Sequence[RunnerStreamIn] = list(payloads)
        for state in self.ops:
            current = self._apply_operator_state(state, current, force=force)
        return cast(Microbatch, current)

    def _finalize_pipeline(self) -> Microbatch:
        if not self.ops:
            return []
        pending: Sequence[RunnerStreamIn] = ()
        for state in self.ops:
            outputs = self._apply_operator_state(state, pending, force=True)
            tail = state.finalize()
            if tail:
                outputs.extend(tail)
            pending = outputs
        return cast(Microbatch, pending)

    def _cancel_upstream(self, upstream_iter: Iterator[RunnerStageIn] | None) -> None:
        if upstream_iter is None:
            return
        close = getattr(upstream_iter, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def run(self, upstream: Iterable[RunnerStageIn]) -> Iterator[RunnerStageOut]:
        stop_event = threading.Event()
        upstream_holder: dict[str, Iterator[RunnerStageIn] | None] = {"iter": None}

        with self._context_lock:
            if self._active:
                raise RuntimeError("Stage runner already in use")
            self._active = True

        def iterator() -> Iterator[RunnerStageOut]:
            upstream_iter = iter(upstream)
            upstream_holder["iter"] = upstream_iter
            try:
                if not self.ops:
                    yield from self._run_passthrough(upstream_iter, stop_event)
                    return

                for state in self.ops:
                    state.reset_buffers()

                for elem in upstream_iter:
                    if stop_event.is_set():
                        break
                    batch = self._coerce_to_batch(elem)
                    if not batch:
                        continue
                    outputs = self._process_pipeline(batch, force=False)
                    if outputs:
                        yield from self._emit_stage_output(outputs)

                tail = self._finalize_pipeline()
                if tail:
                    yield from self._emit_stage_output(tail)
            finally:
                stop_event.set()
                self._cancel_upstream(upstream_holder.get("iter"))
                upstream_holder["iter"] = None
                with self._context_lock:
                    self._active = False

        stream = iterator()
        if self._prefetch_capacity > 0:

            def _on_stop() -> None:
                stop_event.set()
                self._cancel_upstream(upstream_holder.get("iter"))

            stream = buffered_iterable(
                stream,
                self._prefetch_capacity,
                on_stop=_on_stop,
            )
        return stream

    def close(self) -> None:
        with self._context_lock:
            self._active = False
