"""Utilities to collect and aggregate observability metrics."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable

from .config import ExecutionTrackingMode, MetricsSinkConfig
from .stats import (
    BackpressureDelta,
    BackpressureSummary,
    FetchTimingDelta,
    FetchTimingSummary,
    NodeMetricsDelta,
    PipelineSummary,
    PrefetchTimingDelta,
    PrefetchTimingSummary,
    PumpTimingDelta,
    PumpTimingSummary,
)


@dataclass
class CollectorConfig:
    """Runtime configuration for a collector."""

    tracking_mode: ExecutionTrackingMode = ExecutionTrackingMode.OFF
    sink: MetricsSinkConfig | None = None
    report_interval_s: float = 5.0
    plan_id: str | None = None


class PipelineCollector:
    """Thread-safe accumulator for per-node execution metrics."""

    def __init__(self, config: CollectorConfig):
        self._config = config
        self._lock = threading.RLock()
        self._summary = PipelineSummary(
            plan_id=config.plan_id,
            reporting_interval_s=config.report_interval_s,
            tracking_mode=config.tracking_mode,
        )
        self._fetch_summary = FetchTimingSummary(
            plan_id=config.plan_id,
            reporting_interval_s=config.report_interval_s,
            tracking_mode=config.tracking_mode,
        )
        self._prefetch_summary = PrefetchTimingSummary(
            plan_id=config.plan_id,
            reporting_interval_s=config.report_interval_s,
            tracking_mode=config.tracking_mode,
        )
        self._backpressure_summary = BackpressureSummary(
            plan_id=config.plan_id,
            reporting_interval_s=config.report_interval_s,
            tracking_mode=config.tracking_mode,
        )
        self._pump_summary = PumpTimingSummary(
            plan_id=config.plan_id,
            reporting_interval_s=config.report_interval_s,
            tracking_mode=config.tracking_mode,
        )

    @property
    def tracking_mode(self) -> ExecutionTrackingMode:
        return self._config.tracking_mode

    def record(self, delta: NodeMetricsDelta) -> None:
        if self._config.tracking_mode is ExecutionTrackingMode.OFF:
            return
        with self._lock:
            self._summary.apply(delta)

    def extend(self, deltas: Iterable[NodeMetricsDelta]) -> None:
        for delta in deltas:
            self.record(delta)

    def record_fetch(self, delta: FetchTimingDelta) -> None:
        if not self._config.tracking_mode.collects_nodes:
            return
        with self._lock:
            self._fetch_summary.apply(delta)

    def record_prefetch(self, delta: PrefetchTimingDelta) -> None:
        if not self._config.tracking_mode.collects_nodes:
            return
        with self._lock:
            self._prefetch_summary.apply(delta)

    def record_backpressure(self, delta: BackpressureDelta) -> None:
        if not self._config.tracking_mode.collects_nodes:
            return
        with self._lock:
            self._backpressure_summary.apply(delta)

    def record_pump_timing(self, delta: PumpTimingDelta) -> None:
        if not self._config.tracking_mode.collects_nodes:
            return
        with self._lock:
            self._pump_summary.apply(delta)

    def snapshot(self) -> PipelineSummary:
        with self._lock:
            return self._summary.clone()

    def snapshot_fetch(self) -> FetchTimingSummary:
        with self._lock:
            return self._fetch_summary.clone()

    def snapshot_prefetch(self) -> PrefetchTimingSummary:
        with self._lock:
            return self._prefetch_summary.clone()

    def snapshot_backpressure(self) -> BackpressureSummary:
        with self._lock:
            return self._backpressure_summary.clone()

    def snapshot_pump_timing(self) -> PumpTimingSummary:
        with self._lock:
            return self._pump_summary.clone()

    def reset(self) -> None:
        with self._lock:
            self._summary = PipelineSummary(
                plan_id=self._summary.plan_id,
                reporting_interval_s=self._summary.reporting_interval_s,
                tracking_mode=self._summary.tracking_mode,
            )
            self._fetch_summary = FetchTimingSummary(
                plan_id=self._fetch_summary.plan_id,
                reporting_interval_s=self._fetch_summary.reporting_interval_s,
                tracking_mode=self._fetch_summary.tracking_mode,
            )
            self._prefetch_summary = PrefetchTimingSummary(
                plan_id=self._prefetch_summary.plan_id,
                reporting_interval_s=self._prefetch_summary.reporting_interval_s,
                tracking_mode=self._prefetch_summary.tracking_mode,
            )
            self._backpressure_summary = BackpressureSummary(
                plan_id=self._backpressure_summary.plan_id,
                reporting_interval_s=self._backpressure_summary.reporting_interval_s,
                tracking_mode=self._backpressure_summary.tracking_mode,
            )
            self._pump_summary = PumpTimingSummary(
                plan_id=self._pump_summary.plan_id,
                reporting_interval_s=self._pump_summary.reporting_interval_s,
                tracking_mode=self._pump_summary.tracking_mode,
            )

    def merge_summary(
        self,
        other: PipelineSummary,
        fetch: FetchTimingSummary | None = None,
        prefetch: PrefetchTimingSummary | None = None,
        backpressure: BackpressureSummary | None = None,
        pump: PumpTimingSummary | None = None,
    ) -> None:
        if self._config.tracking_mode is ExecutionTrackingMode.OFF:
            return
        with self._lock:
            self._summary.merge(other)
            if fetch is not None:
                self._fetch_summary.merge(fetch)
            if prefetch is not None:
                self._prefetch_summary.merge(prefetch)
            if backpressure is not None:
                self._backpressure_summary.merge(backpressure)
            if pump is not None:
                self._pump_summary.merge(pump)

    def plan_id(self) -> str | None:
        return self._summary.plan_id

    def set_plan_id(self, value: str) -> None:
        with self._lock:
            self._summary.plan_id = value
            self._fetch_summary.plan_id = value
            self._prefetch_summary.plan_id = value
            self._backpressure_summary.plan_id = value
            self._pump_summary.plan_id = value
