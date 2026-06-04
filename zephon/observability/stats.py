"""Aggregated observability data structures shared between workers."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from typing import Iterator, Mapping, MutableMapping

from .config import ExecutionTrackingMode


@dataclass(slots=True)
class NodeMetricsDelta:
    """Incremental metrics produced by a runner for a single operator."""

    stage_index: int
    op_index: int
    stage_name: str
    name: str
    processed_ns: int = 0
    produced_elements: int = 0
    consumed_elements: int = 0
    produced_bytes: int = 0
    consumed_bytes: int = 0
    wait_ns: int = 0
    max_queue_depth: int | None = None
    min_processing_ns: int | None = None
    max_processing_ns: int | None = None

    @classmethod
    def empty(
        cls, stage_index: int, op_index: int, stage_name: str, name: str
    ) -> "NodeMetricsDelta":
        return cls(
            stage_index=stage_index,
            op_index=op_index,
            stage_name=stage_name,
            name=name,
        )


@dataclass(slots=True)
class NodeSummary:
    """Aggregated metrics for an operator."""

    stage_index: int
    op_index: int
    name: str
    processed_ns: int = 0
    produced_elements: int = 0
    consumed_elements: int = 0
    produced_bytes: int = 0
    consumed_bytes: int = 0
    wait_ns: int = 0
    min_processing_ns: int = field(default_factory=lambda: sys.maxsize)
    max_processing_ns: int = 0
    max_queue_depth: int = 0
    wait_ratio: float = 0.0

    # Derived metrics.
    def apply(self, delta: NodeMetricsDelta) -> None:
        self.processed_ns += delta.processed_ns
        self.produced_elements += delta.produced_elements
        self.consumed_elements += delta.consumed_elements
        self.produced_bytes += delta.produced_bytes
        self.consumed_bytes += delta.consumed_bytes
        self.wait_ns += delta.wait_ns
        if delta.max_queue_depth is not None:
            self.max_queue_depth = max(self.max_queue_depth, delta.max_queue_depth)

        if delta.min_processing_ns is not None:
            self.min_processing_ns = min(
                self.min_processing_ns, delta.min_processing_ns
            )
        if delta.max_processing_ns is not None:
            self.max_processing_ns = max(
                self.max_processing_ns, delta.max_processing_ns
            )

    @property
    def has_samples(self) -> bool:
        return self.produced_elements > 0 or self.processed_ns > 0

    @property
    def avg_processing_ns(self) -> float:
        if self.produced_elements == 0:
            return 0.0
        return self.processed_ns / max(self.produced_elements, 1)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage_index,
            "op": self.op_index,
            "name": self.name,
            "processed_ns": self.processed_ns,
            "produced_elements": self.produced_elements,
            "consumed_elements": self.consumed_elements,
            "produced_bytes": self.produced_bytes,
            "consumed_bytes": self.consumed_bytes,
            "wait_ns": self.wait_ns,
            "min_processing_ns": None
            if self.min_processing_ns == sys.maxsize
            else self.min_processing_ns,
            "max_processing_ns": self.max_processing_ns,
            "avg_processing_ns": self.avg_processing_ns,
            "max_queue_depth": self.max_queue_depth,
            "wait_ratio": self.wait_ratio,
        }


@dataclass(slots=True)
class StageSummary:
    """Aggregated metrics for a pipeline stage."""

    index: int
    name: str
    nodes: MutableMapping[int, NodeSummary] = field(default_factory=dict)

    def apply(self, delta: NodeMetricsDelta) -> None:
        if delta.stage_name and not self.name:
            self.name = delta.stage_name
        node = self.nodes.get(delta.op_index)
        if node is None:
            node = NodeSummary(
                stage_index=delta.stage_index,
                op_index=delta.op_index,
                name=delta.name,
            )
            self.nodes[delta.op_index] = node
        node.apply(delta)

    def merge(self, other: "StageSummary") -> None:
        for op_index, node in other.nodes.items():
            self.apply(
                NodeMetricsDelta(
                    stage_index=self.index,
                    op_index=op_index,
                    stage_name=self.name,
                    name=node.name,
                    processed_ns=node.processed_ns,
                    produced_elements=node.produced_elements,
                    consumed_elements=node.consumed_elements,
                    produced_bytes=node.produced_bytes,
                    consumed_bytes=node.consumed_bytes,
                    wait_ns=node.wait_ns,
                    max_queue_depth=node.max_queue_depth,
                    min_processing_ns=(
                        None
                        if node.min_processing_ns == sys.maxsize
                        else node.min_processing_ns
                    ),
                    max_processing_ns=node.max_processing_ns,
                )
            )

    def iter_nodes(self) -> Iterator[NodeSummary]:
        for _, node in sorted(self.nodes.items()):
            yield node

    def copy(self) -> "StageSummary":
        clone = StageSummary(index=self.index, name=self.name)
        for node in self.iter_nodes():
            clone.apply(
                NodeMetricsDelta(
                    stage_index=self.index,
                    op_index=node.op_index,
                    stage_name=self.name,
                    name=node.name,
                    processed_ns=node.processed_ns,
                    produced_elements=node.produced_elements,
                    consumed_elements=node.consumed_elements,
                    produced_bytes=node.produced_bytes,
                    consumed_bytes=node.consumed_bytes,
                    wait_ns=node.wait_ns,
                    max_queue_depth=node.max_queue_depth,
                    min_processing_ns=(
                        None
                        if node.min_processing_ns == sys.maxsize
                        else node.min_processing_ns
                    ),
                    max_processing_ns=node.max_processing_ns or None,
                )
            )
        return clone

    @property
    def total_processing_ns(self) -> int:
        return sum(node.processed_ns for node in self.nodes.values())

    @property
    def total_wait_ns(self) -> int:
        return sum(node.wait_ns for node in self.nodes.values())

    @property
    def produced_elements(self) -> int:
        return sum(node.produced_elements for node in self.nodes.values())


@dataclass(slots=True)
class FetchTimingDelta:
    """Incremental fetch metrics emitted per shard group."""

    stage_index: int
    shard_id: int
    samples: int
    group_ns: int
    resolve_ns: int
    open_ns: int
    read_ns: int
    close_ns: int
    retries: int
    cache_hits: int
    cache_misses: int
    shard_reopens: int = 0


@dataclass(slots=True)
class FetchTimingTotals:
    """Aggregated timing totals accumulated across deltas."""

    samples: int = 0
    groups: int = 0
    group_ns: int = 0
    resolve_ns: int = 0
    open_ns: int = 0
    read_ns: int = 0
    close_ns: int = 0
    retries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    shard_reopens: int = 0

    def apply(self, delta: FetchTimingDelta) -> None:
        self.samples += max(0, delta.samples)
        self.groups += 1
        self.group_ns += max(0, delta.group_ns)
        self.resolve_ns += max(0, delta.resolve_ns)
        self.open_ns += max(0, delta.open_ns)
        self.read_ns += max(0, delta.read_ns)
        self.close_ns += max(0, delta.close_ns)
        self.retries += max(0, delta.retries)
        self.cache_hits += max(0, delta.cache_hits)
        self.cache_misses += max(0, delta.cache_misses)
        self.shard_reopens += max(0, delta.shard_reopens)

    def merge(self, other: "FetchTimingTotals") -> None:
        self.samples += other.samples
        self.groups += other.groups
        self.group_ns += other.group_ns
        self.resolve_ns += other.resolve_ns
        self.open_ns += other.open_ns
        self.read_ns += other.read_ns
        self.close_ns += other.close_ns
        self.retries += other.retries
        self.cache_hits += other.cache_hits
        self.cache_misses += other.cache_misses
        self.shard_reopens += other.shard_reopens

    def copy(self) -> "FetchTimingTotals":
        clone = FetchTimingTotals()
        clone.samples = self.samples
        clone.groups = self.groups
        clone.group_ns = self.group_ns
        clone.resolve_ns = self.resolve_ns
        clone.open_ns = self.open_ns
        clone.read_ns = self.read_ns
        clone.close_ns = self.close_ns
        clone.retries = self.retries
        clone.cache_hits = self.cache_hits
        clone.cache_misses = self.cache_misses
        clone.shard_reopens = self.shard_reopens
        return clone

    @property
    def avg_group_ns(self) -> float:
        if self.groups <= 0:
            return 0.0
        return self.group_ns / max(self.groups, 1)

    @property
    def avg_resolve_ns(self) -> float:
        if self.samples <= 0:
            return 0.0
        return self.resolve_ns / max(self.samples, 1)

    @property
    def avg_open_ns(self) -> float:
        if self.samples <= 0:
            return 0.0
        return self.open_ns / max(self.samples, 1)

    @property
    def avg_read_ns(self) -> float:
        if self.samples <= 0:
            return 0.0
        return self.read_ns / max(self.samples, 1)

    @property
    def avg_close_ns(self) -> float:
        if self.samples <= 0:
            return 0.0
        return self.close_ns / max(self.samples, 1)

    @property
    def cache_hit_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total <= 0:
            return 0.0
        return self.cache_hits / total


@dataclass(slots=True)
class FetchStageSummary:
    """Aggregated fetch timings for a single pipeline stage."""

    index: int
    totals: FetchTimingTotals = field(default_factory=FetchTimingTotals)
    shard_totals: MutableMapping[int, FetchTimingTotals] = field(default_factory=dict)

    def apply(self, delta: FetchTimingDelta) -> None:
        self.totals.apply(delta)
        shard = self.shard_totals.get(delta.shard_id)
        if shard is None:
            shard = FetchTimingTotals()
            self.shard_totals[delta.shard_id] = shard
        shard.apply(delta)

    def merge(self, other: "FetchStageSummary") -> None:
        self.totals.merge(other.totals)
        for shard_id, totals in other.shard_totals.items():
            existing = self.shard_totals.get(shard_id)
            if existing is None:
                self.shard_totals[shard_id] = totals.copy()
            else:
                existing.merge(totals)

    def copy(self) -> "FetchStageSummary":
        clone = FetchStageSummary(index=self.index)
        clone.totals = self.totals.copy()
        for shard_id, totals in self.shard_totals.items():
            clone.shard_totals[shard_id] = totals.copy()
        return clone

    def _build_record(
        self,
        *,
        plan_id: str | None,
        stage_name: str | None,
        tracking_mode: ExecutionTrackingMode,
        shard_id: int | None,
        totals: FetchTimingTotals,
    ) -> dict:
        return {
            "plan_id": plan_id,
            "tracking_mode": tracking_mode.value,
            "stage": self.index,
            "stage_name": stage_name,
            "shard_id": shard_id,
            "samples": totals.samples,
            "groups": totals.groups,
            "group_ns": totals.group_ns,
            "resolve_ns": totals.resolve_ns,
            "open_ns": totals.open_ns,
            "read_ns": totals.read_ns,
            "close_ns": totals.close_ns,
            "avg_group_ns": totals.avg_group_ns,
            "avg_resolve_ns": totals.avg_resolve_ns,
            "avg_open_ns": totals.avg_open_ns,
            "avg_read_ns": totals.avg_read_ns,
            "avg_close_ns": totals.avg_close_ns,
            "retries": totals.retries,
            "cache_hits": totals.cache_hits,
            "cache_misses": totals.cache_misses,
            "cache_hit_ratio": totals.cache_hit_ratio,
            "shard_reopens": totals.shard_reopens,
        }

    def to_records(
        self,
        *,
        plan_id: str | None,
        stage_name: str | None,
        tracking_mode: ExecutionTrackingMode,
        include_shards: bool = True,
    ) -> list[dict]:
        records: list[dict] = [
            self._build_record(
                plan_id=plan_id,
                stage_name=stage_name,
                tracking_mode=tracking_mode,
                shard_id=None,
                totals=self.totals,
            )
        ]
        if include_shards:
            for shard_id, totals in sorted(self.shard_totals.items()):
                records.append(
                    self._build_record(
                        plan_id=plan_id,
                        stage_name=stage_name,
                        tracking_mode=tracking_mode,
                        shard_id=shard_id,
                        totals=totals,
                    )
                )
        return records


@dataclass
class FetchTimingSummary:
    """Aggregated fetch metrics for an entire pipeline."""

    plan_id: str | None = None
    reporting_interval_s: float | None = None
    tracking_mode: ExecutionTrackingMode = ExecutionTrackingMode.OFF
    stages: MutableMapping[int, FetchStageSummary] = field(default_factory=dict)

    def apply(self, delta: FetchTimingDelta) -> None:
        stage = self.stages.get(delta.stage_index)
        if stage is None:
            stage = FetchStageSummary(index=delta.stage_index)
            self.stages[delta.stage_index] = stage
        stage.apply(delta)

    def merge(self, other: "FetchTimingSummary") -> None:
        if self.plan_id is None:
            self.plan_id = other.plan_id
        if self.reporting_interval_s is None:
            self.reporting_interval_s = other.reporting_interval_s
        for stage_index, stage in other.stages.items():
            existing = self.stages.get(stage_index)
            if existing is None:
                self.stages[stage_index] = stage.copy()
            else:
                existing.merge(stage)

    def iter_stages(self) -> Iterator[FetchStageSummary]:
        for _, stage in sorted(self.stages.items()):
            yield stage

    def to_records(
        self,
        stage_names: Mapping[int, str] | None = None,
        *,
        include_shards: bool = True,
    ) -> list[dict]:
        records: list[dict] = []
        for stage in self.iter_stages():
            name = stage_names.get(stage.index) if stage_names else None
            records.extend(
                stage.to_records(
                    plan_id=self.plan_id,
                    stage_name=name,
                    tracking_mode=self.tracking_mode,
                    include_shards=include_shards,
                )
            )
        return records

    def clone(self) -> "FetchTimingSummary":
        clone = FetchTimingSummary(
            plan_id=self.plan_id,
            reporting_interval_s=self.reporting_interval_s,
            tracking_mode=self.tracking_mode,
        )
        for stage in self.iter_stages():
            clone.stages[stage.index] = stage.copy()
        return clone

    def has_samples(self) -> bool:
        return any(stage.totals.samples > 0 for stage in self.stages.values())


@dataclass(slots=True)
class PrefetchTimingDelta:
    """Incremental prefetch metrics emitted per batch."""

    stage_index: int
    batch_size: int
    prefetch_requests: int
    prefetch_succeeded: int
    prefetch_failed: int


@dataclass(slots=True)
class PrefetchTimingTotals:
    """Aggregated prefetch totals accumulated across deltas."""

    batches: int = 0
    samples: int = 0
    prefetch_requests: int = 0
    prefetch_succeeded: int = 0
    prefetch_failed: int = 0

    def apply(self, delta: PrefetchTimingDelta) -> None:
        self.batches += 1
        self.samples += max(0, delta.batch_size)
        self.prefetch_requests += max(0, delta.prefetch_requests)
        self.prefetch_succeeded += max(0, delta.prefetch_succeeded)
        self.prefetch_failed += max(0, delta.prefetch_failed)

    def merge(self, other: "PrefetchTimingTotals") -> None:
        self.batches += other.batches
        self.samples += other.samples
        self.prefetch_requests += other.prefetch_requests
        self.prefetch_succeeded += other.prefetch_succeeded
        self.prefetch_failed += other.prefetch_failed

    def copy(self) -> "PrefetchTimingTotals":
        clone = PrefetchTimingTotals()
        clone.batches = self.batches
        clone.samples = self.samples
        clone.prefetch_requests = self.prefetch_requests
        clone.prefetch_succeeded = self.prefetch_succeeded
        clone.prefetch_failed = self.prefetch_failed
        return clone


@dataclass
class PrefetchTimingSummary:
    """Aggregated prefetch metrics for an entire pipeline."""

    plan_id: str | None = None
    reporting_interval_s: float | None = None
    tracking_mode: ExecutionTrackingMode = ExecutionTrackingMode.OFF
    stages: MutableMapping[int, PrefetchTimingTotals] = field(default_factory=dict)

    def apply(self, delta: PrefetchTimingDelta) -> None:
        stage = self.stages.get(delta.stage_index)
        if stage is None:
            stage = PrefetchTimingTotals()
            self.stages[delta.stage_index] = stage
        stage.apply(delta)

    def merge(self, other: "PrefetchTimingSummary") -> None:
        if self.plan_id is None:
            self.plan_id = other.plan_id
        if self.reporting_interval_s is None:
            self.reporting_interval_s = other.reporting_interval_s
        for stage_index, totals in other.stages.items():
            existing = self.stages.get(stage_index)
            if existing is None:
                self.stages[stage_index] = totals.copy()
            else:
                existing.merge(totals)

    def clone(self) -> "PrefetchTimingSummary":
        clone = PrefetchTimingSummary(
            plan_id=self.plan_id,
            reporting_interval_s=self.reporting_interval_s,
            tracking_mode=self.tracking_mode,
        )
        for stage_index, totals in self.stages.items():
            clone.stages[stage_index] = totals.copy()
        return clone

    def has_samples(self) -> bool:
        return any(totals.samples > 0 for totals in self.stages.values())

    @property
    def success_rate(self) -> float:
        total = self.prefetch_requests
        if total <= 0:
            return 0.0
        return self.prefetch_succeeded / total


@dataclass(slots=True)
class BackpressureDelta:
    """Incremental backpressure metrics emitted by ConcurrentStageRunner.

    Tracks queue.Full events in _put_into_queue when stages cannot forward
    results downstream due to backpressure.
    """

    stage_index: int
    put_into_queue_backpressure_events: int = 0


@dataclass(slots=True)
class BackpressureTotals:
    """Aggregated backpressure event counts."""

    put_into_queue_backpressure_events: int = 0

    def apply(self, delta: BackpressureDelta) -> None:
        self.put_into_queue_backpressure_events += (
            delta.put_into_queue_backpressure_events
        )

    def merge(self, other: "BackpressureTotals") -> None:
        self.put_into_queue_backpressure_events += (
            other.put_into_queue_backpressure_events
        )

    def copy(self) -> "BackpressureTotals":
        return BackpressureTotals(
            put_into_queue_backpressure_events=self.put_into_queue_backpressure_events,
        )


@dataclass
class BackpressureSummary:
    """Aggregated backpressure events for an entire pipeline."""

    plan_id: str | None = None
    reporting_interval_s: float | None = None
    tracking_mode: ExecutionTrackingMode = ExecutionTrackingMode.OFF
    stages: MutableMapping[int, BackpressureTotals] = field(default_factory=dict)

    def apply(self, delta: BackpressureDelta) -> None:
        stage = self.stages.get(delta.stage_index)
        if stage is None:
            stage = BackpressureTotals()
            self.stages[delta.stage_index] = stage
        stage.apply(delta)

    def merge(self, other: "BackpressureSummary") -> None:
        if self.plan_id is None:
            self.plan_id = other.plan_id
        if self.reporting_interval_s is None:
            self.reporting_interval_s = other.reporting_interval_s
        for stage_index, totals in other.stages.items():
            existing = self.stages.get(stage_index)
            if existing is None:
                self.stages[stage_index] = totals.copy()
            else:
                existing.merge(totals)

    def clone(self) -> "BackpressureSummary":
        clone = BackpressureSummary(
            plan_id=self.plan_id,
            reporting_interval_s=self.reporting_interval_s,
            tracking_mode=self.tracking_mode,
        )
        for stage_index, totals in self.stages.items():
            clone.stages[stage_index] = totals.copy()
        return clone

    def has_events(self) -> bool:
        return any(
            totals.put_into_queue_backpressure_events > 0
            for totals in self.stages.values()
        )

    @property
    def total_events(self) -> int:
        """Total backpressure events across all stages."""
        return sum(
            totals.put_into_queue_backpressure_events for totals in self.stages.values()
        )


@dataclass(slots=True)
class PumpCounts:
    """Shape of a pump-thread measurement record.

    The seven ``*_ns`` buckets are disjoint slices of pump wall time; their
    sum approximates the elapsed wall clock modulo small unaccounted Python
    overhead. The counter fields track discrete events. Used as the shared
    base for ``PumpTimingDelta`` (per-flush emit), ``PumpTimingNodeTotals``
    (aggregated in the collector), and ``PumpTimer`` (the live accumulator
    on the operator state). Field names live here once; everything else
    derives from :func:`dataclasses.fields`.
    """

    input_wait_ns: int = 0
    dispatch_wait_ns: int = 0
    dispatch_active_ns: int = 0
    result_wait_ns: int = 0
    result_collect_ns: int = 0
    result_handle_ns: int = 0
    idle_drain_ns: int = 0
    batches_submitted: int = 0
    batches_completed: int = 0
    capacity_stalls: int = 0
    items_consumed: int = 0
    sweeps: int = 0
    sweep_refs: int = 0

    def add_from(self, other: "PumpCounts") -> None:
        for name in _PUMP_ALL_FIELDS:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def copy_to(self, other: "PumpCounts") -> None:
        for name in _PUMP_ALL_FIELDS:
            setattr(other, name, getattr(self, name))

    def reset_counts(self) -> None:
        for name in _PUMP_ALL_FIELDS:
            setattr(self, name, 0)

    @property
    def total_ns(self) -> int:
        return sum(getattr(self, name) for name in _PUMP_BUCKET_FIELDS)

    def has_samples(self) -> bool:
        """True if any bucket accrued time or any counter fired."""
        return any(getattr(self, name) for name in _PUMP_ALL_FIELDS)


# Derived from PumpCounts so the field list is named exactly once.
_PUMP_BUCKET_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(PumpCounts) if f.name.endswith("_ns")
)
_PUMP_COUNTER_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(PumpCounts) if not f.name.endswith("_ns")
)
_PUMP_ALL_FIELDS: tuple[str, ...] = _PUMP_BUCKET_FIELDS + _PUMP_COUNTER_FIELDS


@dataclass(slots=True)
class PumpTimingDelta(PumpCounts):
    """Incremental pump-thread timing emitted by a concurrent stage runner."""

    stage_index: int = 0
    op_index: int = 0
    stage_name: str = ""
    name: str = ""


@dataclass(slots=True)
class PumpTimingNodeTotals(PumpCounts):
    """Aggregated pump timing for one operator."""

    stage_index: int = 0
    op_index: int = 0
    name: str = ""

    def apply(self, delta: PumpTimingDelta) -> None:
        self.add_from(delta)

    def to_dict(self) -> dict:
        record: dict = {
            "stage": self.stage_index,
            "op": self.op_index,
            "name": self.name,
        }
        for name in _PUMP_ALL_FIELDS:
            record[name] = getattr(self, name)
        record["total_ns"] = self.total_ns
        return record


@dataclass(slots=True)
class PumpTimingStageSummary:
    """Aggregated pump timing for one stage."""

    index: int
    name: str
    nodes: MutableMapping[int, PumpTimingNodeTotals] = field(default_factory=dict)

    def apply(self, delta: PumpTimingDelta) -> None:
        if delta.stage_name and not self.name:
            self.name = delta.stage_name
        node = self.nodes.get(delta.op_index)
        if node is None:
            node = PumpTimingNodeTotals(
                stage_index=delta.stage_index,
                op_index=delta.op_index,
                name=delta.name,
            )
            self.nodes[delta.op_index] = node
        node.apply(delta)

    def _delta_from_node(
        self, node: PumpTimingNodeTotals, stage_name: str
    ) -> PumpTimingDelta:
        delta = PumpTimingDelta(
            stage_index=self.index,
            op_index=node.op_index,
            stage_name=stage_name,
            name=node.name,
        )
        node.copy_to(delta)
        return delta

    def merge(self, other: "PumpTimingStageSummary") -> None:
        for node in other.nodes.values():
            self.apply(self._delta_from_node(node, self.name))

    def iter_nodes(self) -> Iterator[PumpTimingNodeTotals]:
        for _, node in sorted(self.nodes.items()):
            yield node

    def copy(self) -> "PumpTimingStageSummary":
        clone = PumpTimingStageSummary(index=self.index, name=self.name)
        for node in self.iter_nodes():
            clone.apply(self._delta_from_node(node, self.name))
        return clone


@dataclass
class PumpTimingSummary:
    """Aggregated pump-thread timing for an entire pipeline."""

    plan_id: str | None = None
    reporting_interval_s: float | None = None
    tracking_mode: ExecutionTrackingMode = ExecutionTrackingMode.OFF
    stages: MutableMapping[int, PumpTimingStageSummary] = field(default_factory=dict)

    def apply(self, delta: PumpTimingDelta) -> None:
        stage = self.stages.get(delta.stage_index)
        if stage is None:
            stage = PumpTimingStageSummary(
                index=delta.stage_index, name=delta.stage_name
            )
            self.stages[delta.stage_index] = stage
        stage.apply(delta)

    def merge(self, other: "PumpTimingSummary") -> None:
        if self.plan_id is None:
            self.plan_id = other.plan_id
        if self.reporting_interval_s is None:
            self.reporting_interval_s = other.reporting_interval_s
        for stage_index, stage in other.stages.items():
            existing = self.stages.get(stage_index)
            if existing is None:
                new_stage = PumpTimingStageSummary(index=stage.index, name=stage.name)
                new_stage.merge(stage)
                self.stages[stage_index] = new_stage
            else:
                existing.merge(stage)

    def iter_stages(self) -> Iterator[PumpTimingStageSummary]:
        for _, stage in sorted(self.stages.items()):
            yield stage

    def clone(self) -> "PumpTimingSummary":
        clone = PumpTimingSummary(
            plan_id=self.plan_id,
            reporting_interval_s=self.reporting_interval_s,
            tracking_mode=self.tracking_mode,
        )
        for stage in self.iter_stages():
            clone.stages[stage.index] = stage.copy()
        return clone

    def to_records(self) -> list[dict]:
        records: list[dict] = []
        for stage in self.iter_stages():
            for node in stage.iter_nodes():
                record = node.to_dict()
                record["stage_name"] = stage.name
                record["plan_id"] = self.plan_id
                record["tracking_mode"] = self.tracking_mode.value
                records.append(record)
        return records

    def has_samples(self) -> bool:
        return any(stage.nodes for stage in self.stages.values())


@dataclass
class PipelineSummary:
    """Aggregated metrics for an entire pipeline invocation."""

    plan_id: str | None = None
    reporting_interval_s: float | None = None
    tracking_mode: ExecutionTrackingMode = ExecutionTrackingMode.OFF
    stages: MutableMapping[int, StageSummary] = field(default_factory=dict)
    total_wait_ratio: float | None = None

    def apply(self, delta: NodeMetricsDelta) -> None:
        stage = self.stages.get(delta.stage_index)
        if stage is None:
            stage = StageSummary(index=delta.stage_index, name=delta.stage_name)
            self.stages[delta.stage_index] = stage
        stage.apply(delta)

    def merge(self, other: "PipelineSummary") -> None:
        if self.plan_id is None:
            self.plan_id = other.plan_id
        if self.reporting_interval_s is None:
            self.reporting_interval_s = other.reporting_interval_s
        for stage_index, stage in other.stages.items():
            existing = self.stages.get(stage_index)
            if existing is None:
                new_stage = StageSummary(index=stage.index, name=stage.name)
                new_stage.merge(stage)
                self.stages[stage_index] = new_stage
            else:
                existing.merge(stage)

    def iter_stages(self) -> Iterator[StageSummary]:
        for _, stage in sorted(self.stages.items()):
            yield stage

    def to_records(self) -> list[dict]:
        records: list[dict] = []
        for stage in self.iter_stages():
            for node in stage.iter_nodes():
                record = node.to_dict()
                record["stage_name"] = stage.name
                record["plan_id"] = self.plan_id
                record["tracking_mode"] = self.tracking_mode.value
                records.append(record)
        return records

    def clone(self) -> "PipelineSummary":
        clone = PipelineSummary(
            plan_id=self.plan_id,
            reporting_interval_s=self.reporting_interval_s,
            tracking_mode=self.tracking_mode,
        )
        for stage in self.iter_stages():
            clone.stages[stage.index] = stage.copy()
        clone.total_wait_ratio = self.total_wait_ratio
        return clone

    def compute_wait_ratios(self) -> None:
        """Derive wait ratios for the pipeline based on processing and wait times."""
        total_wait_ns = sum(stage.total_wait_ns for stage in self.stages.values())
        if total_wait_ns <= 0:
            self.total_wait_ratio = 0.0
            return
        for stage in self.stages.values():
            for node in stage.nodes.values():
                node.wait_ratio = (
                    (node.wait_ns / total_wait_ns) if total_wait_ns else 0.0
                )
        self.total_wait_ratio = 1.0


def pretty_format_ns(value: int) -> str:
    """Return a human-readable nanosecond duration string."""
    if value < 1_000:
        return f"{value}ns"
    if value < 1_000_000:
        return f"{value / 1_000:.2f}µs"
    if value < 1_000_000_000:
        return f"{value / 1_000_000:.2f}ms"
    return f"{value / 1_000_000_000:.2f}s"


def pretty_format_bytes(value: int) -> str:
    """Return a human-readable byte count string."""
    if value < 1024:
        return f"{value}B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    scaled = float(value)
    for unit in units:
        scaled /= 1024.0
        if scaled < 1024.0:
            return f"{scaled:.2f}{unit}"
    return f"{scaled:.2f}PiB"


def pretty_format_ratio(value: float) -> str:
    """Format a ratio (0-1) as a percentage string."""
    return f"{value * 100:.2f}%"
