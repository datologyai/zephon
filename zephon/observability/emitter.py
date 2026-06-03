"""Background reporting utilities for Zephon observability."""

from __future__ import annotations

import json
import logging
import socket
import threading
from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableMapping

from .collector import PipelineCollector
from .config import ExecutionTrackingMode, MetricsSinkConfig
from .stats import (
    FetchTimingSummary,
    PipelineSummary,
    PumpTimingSummary,
    pretty_format_bytes,
    pretty_format_ns,
    pretty_format_ratio,
)

logger = logging.getLogger(__name__)


def _sanitize_component(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_").replace("-", "_").replace(":", "_")


@dataclass
class StatsdEmitter:
    """Minimal StatsD client for counters, timers, and gauges."""

    config: MetricsSinkConfig
    _sock: socket.socket = field(init=False, repr=False)
    _addr: tuple[str, int] = field(init=False, repr=False)
    _lock: threading.Lock = field(
        init=False, repr=False, default_factory=threading.Lock
    )
    _buffer: list[str] = field(init=False, repr=False, default_factory=list)
    _errored: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (self.config.statsd_host, int(self.config.statsd_port))

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._sock.close()

    def flush(self) -> None:
        with self._lock:
            while self._buffer:
                payload = "\n".join(self._buffer).encode("utf-8")
                self._buffer.clear()
                try:
                    self._sock.sendto(payload, self._addr)
                except OSError:
                    if not self._errored:
                        logger.warning(
                            "StatsD emission failed; disabling further sending."
                        )
                        self._errored = True
                    break

    def _format_tags(self, tags: Mapping[str, str] | None) -> str:
        all_tags = dict(self.config.default_tags)
        if tags:
            all_tags.update(tags)
        if not all_tags:
            return ""
        pairs = [
            f"{_sanitize_component(k)}:{_sanitize_component(v)}"
            for k, v in sorted(all_tags.items())
        ]
        return "|#" + ",".join(pairs)

    def _metric_name(self, metric: str) -> str:
        namespace = self.config.namespace.rstrip(".")
        return f"{namespace}.{metric}"

    def _queue_message(self, message: str) -> None:
        if self._errored:
            return
        with self._lock:
            self._buffer.append(message)
            if len(self._buffer) >= self.config.max_batch_size:
                self.flush()

    def gauge(
        self, metric: str, value: float, tags: Mapping[str, str] | None = None
    ) -> None:
        message = f"{self._metric_name(metric)}:{value}|g{self._format_tags(tags)}"
        self._queue_message(message)

    def counter(
        self, metric: str, value: float, tags: Mapping[str, str] | None = None
    ) -> None:
        message = f"{self._metric_name(metric)}:{value}|c{self._format_tags(tags)}"
        self._queue_message(message)

    def timer(
        self, metric: str, value_ms: float, tags: Mapping[str, str] | None = None
    ) -> None:
        message = f"{self._metric_name(metric)}:{value_ms}|ms{self._format_tags(tags)}"
        self._queue_message(message)


class MetricsReporter:
    """Periodically snapshot pipeline metrics and emit to configured sinks."""

    def __init__(
        self,
        collector: PipelineCollector,
        sink_config: MetricsSinkConfig | None,
        *,
        rank_id: int = 0,
        worker_id: int = 0,
    ) -> None:
        self._collector = collector
        self._sink_config = sink_config
        self._rank_id = rank_id
        self._worker_id = worker_id
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._statsd: StatsdEmitter | None = None
        if sink_config and sink_config.should_emit_statsd():
            self._statsd = StatsdEmitter(sink_config)
        self._prev_totals: MutableMapping[tuple[int, int], dict[str, float]] = {}
        self._prev_fetch_totals: MutableMapping[int, dict[str, float]] = {}
        self._prev_pump_totals: MutableMapping[tuple[int, int], dict[str, int]] = {}

    def start(self) -> None:
        if self._collector.tracking_mode is ExecutionTrackingMode.OFF:
            return
        if self._thread is not None:
            return
        interval = self._sink_config.flush_interval_s if self._sink_config else 5.0
        self._thread = threading.Thread(
            target=self._loop,
            args=(interval,),
            daemon=True,
            name="ZephonMetricsReporter",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._statsd is not None:
            self._statsd.close()

    def _loop(self, interval: float) -> None:
        while not self._stop_event.wait(interval):
            try:
                self.publish_snapshot()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Metrics reporter failed: %s", exc)

    def publish_snapshot(self) -> None:
        summary = self._collector.snapshot()
        fetch_summary = self._collector.snapshot_fetch()
        pump_summary = self._collector.snapshot_pump_timing()
        if summary.tracking_mode is ExecutionTrackingMode.OFF:
            return
        summary.compute_wait_ratios()
        records = summary.to_records()
        if not records:
            return

        stage_names = {stage.index: stage.name for stage in summary.iter_stages()}
        fetch_records_logs: list[dict] = []
        fetch_records_statsd: list[dict] = []
        if (
            fetch_summary.tracking_mode is not ExecutionTrackingMode.OFF
            and fetch_summary.has_samples()
        ):
            fetch_records_logs = fetch_summary.to_records(
                stage_names, include_shards=True
            )
            fetch_records_statsd = fetch_summary.to_records(
                stage_names, include_shards=False
            )

        pump_records: list[dict] = []
        if pump_summary.has_samples():
            pump_records = pump_summary.to_records()

        if self._sink_config and self._sink_config.should_log():
            self._emit_logs(summary, records)
            if fetch_records_logs:
                interval = summary.reporting_interval_s or (
                    self._sink_config.flush_interval_s if self._sink_config else 5.0
                )
                self._emit_fetch_logs(
                    fetch_summary,
                    fetch_records_logs,
                    interval=interval,
                )
            if pump_records:
                self._emit_pump_logs(pump_summary, pump_records)
        if self._statsd is not None:
            self._emit_statsd(records, summary.plan_id)
            if fetch_records_statsd:
                self._emit_fetch_statsd(fetch_records_statsd, summary.plan_id)
            if pump_records:
                self._emit_pump_statsd(pump_records, summary.plan_id)
        if self._statsd is not None:
            self._statsd.flush()

    def _emit_logs(self, summary: PipelineSummary, records: Iterable[dict]) -> None:
        plan_id = summary.plan_id or "unknown"
        interval = summary.reporting_interval_s or (
            self._sink_config.flush_interval_s if self._sink_config else 5.0
        )
        for record in records:
            log_payload = {
                "event": "zephon.metrics",
                "plan_id": plan_id,
                "stage": record["stage"],
                "stage_name": record["stage_name"],
                "op_index": record["op"],
                "op_name": record["name"],
                "processed_ns": record["processed_ns"],
                "produced_elements": record["produced_elements"],
                "consumed_elements": record["consumed_elements"],
                "produced_bytes": record["produced_bytes"],
                "consumed_bytes": record["consumed_bytes"],
                "avg_processing_ns": record["avg_processing_ns"],
                "min_processing_ns": record["min_processing_ns"],
                "max_processing_ns": record["max_processing_ns"],
                "wait_ns": record["wait_ns"],
                "wait_ratio": record.get("wait_ratio", 0.0),
                "interval_s": interval,
                "rank": self._rank_id,
                "worker": self._worker_id,
            }
            if self._sink_config and self._sink_config.json_logs:
                logger.info(json.dumps(log_payload))
            else:
                logger.info(
                    "[zephon][metrics] plan=%s stage=%s/%s op=%s/%s avg=%s total=%s produced=%d bytes=%s wait=%s ratio=%s",
                    plan_id,
                    record["stage"],
                    record["stage_name"],
                    record["op"],
                    record["name"],
                    pretty_format_ns(int(record["avg_processing_ns"])),
                    pretty_format_ns(int(record["processed_ns"])),
                    int(record["produced_elements"]),
                    pretty_format_bytes(int(record["produced_bytes"])),
                    pretty_format_ns(int(record["wait_ns"])),
                    pretty_format_ratio(float(record.get("wait_ratio", 0.0))),
                )

    def _emit_statsd(self, records: Iterable[dict], plan_id: str | None) -> None:
        assert self._statsd is not None
        base_tags = {
            "plan": plan_id or "unknown",
            "rank": str(self._rank_id),
            "worker": str(self._worker_id),
        }
        for record in records:
            key = (record["stage"], record["op"])
            prev = self._prev_totals.get(key, {})
            delta_processed = record["processed_ns"] - prev.get("processed_ns", 0)
            delta_produced = record["produced_elements"] - prev.get(
                "produced_elements", 0
            )
            delta_consumed = record["consumed_elements"] - prev.get(
                "consumed_elements", 0
            )
            delta_bytes_out = record["produced_bytes"] - prev.get("produced_bytes", 0)
            metric_base = f"{_sanitize_component(record['stage_name'])}.{_sanitize_component(record['name'])}"
            tags = dict(base_tags)
            tags.update(
                {
                    "stage": str(record["stage"]),
                    "op": str(record["op"]),
                }
            )
            if delta_processed > 0:
                self._statsd.timer(
                    f"{metric_base}.processing",
                    delta_processed / 1_000_000,
                    tags,
                )
            if delta_produced > 0:
                self._statsd.counter(f"{metric_base}.produced", delta_produced, tags)
            if delta_consumed > 0:
                self._statsd.counter(f"{metric_base}.consumed", delta_consumed, tags)
            if delta_bytes_out > 0:
                self._statsd.counter(
                    f"{metric_base}.bytes_emitted", delta_bytes_out, tags
                )
            self._statsd.gauge(
                f"{metric_base}.wait_ratio",
                float(record.get("wait_ratio", 0.0)) * 100.0,
                tags,
            )
            self._statsd.gauge(
                f"{metric_base}.queue_max",
                float(record.get("max_queue_depth", 0)),
                tags,
            )
            self._prev_totals[key] = {
                "processed_ns": record["processed_ns"],
                "produced_elements": record["produced_elements"],
                "consumed_elements": record["consumed_elements"],
                "produced_bytes": record["produced_bytes"],
            }

    def _emit_fetch_logs(
        self,
        summary: FetchTimingSummary,
        records: Iterable[dict],
        *,
        interval: float,
    ) -> None:
        plan_id = summary.plan_id or "unknown"
        for record in records:
            shard_id = record["shard_id"]
            payload = {
                "event": "zephon.fetch_metrics",
                "plan_id": plan_id,
                "stage": record["stage"],
                "stage_name": record.get("stage_name"),
                "shard_id": shard_id,
                "samples": record["samples"],
                "groups": record["groups"],
                "group_ns": record["group_ns"],
                "resolve_ns": record["resolve_ns"],
                "open_ns": record["open_ns"],
                "read_ns": record["read_ns"],
                "close_ns": record["close_ns"],
                "avg_group_ns": record["avg_group_ns"],
                "avg_resolve_ns": record["avg_resolve_ns"],
                "avg_open_ns": record["avg_open_ns"],
                "avg_read_ns": record["avg_read_ns"],
                "avg_close_ns": record["avg_close_ns"],
                "retries": record["retries"],
                "cache_hits": record["cache_hits"],
                "cache_misses": record["cache_misses"],
                "cache_hit_ratio": record["cache_hit_ratio"],
                "shard_reopens": record["shard_reopens"],
                "interval_s": interval,
                "rank": self._rank_id,
                "worker": self._worker_id,
            }
            if self._sink_config and self._sink_config.json_logs:
                logger.info(json.dumps(payload))
            else:
                stage_name = payload["stage_name"] or "unknown"
                shard_label = (
                    f"shard={int(shard_id)}" if shard_id is not None else "stage"
                )
                logger.info(
                    "[zephon][fetch] plan=%s %s=%s/%s samples=%d avg_resolve=%s avg_read=%s avg_close=%s retries=%d cache_hit_ratio=%s reopens=%d",
                    plan_id,
                    shard_label,
                    record["stage"],
                    stage_name,
                    record["samples"],
                    pretty_format_ns(int(payload["avg_resolve_ns"])),
                    pretty_format_ns(int(payload["avg_read_ns"])),
                    pretty_format_ns(int(payload["avg_close_ns"])),
                    int(payload["retries"]),
                    pretty_format_ratio(float(payload["cache_hit_ratio"])),
                    int(payload["shard_reopens"]),
                )

    def _emit_fetch_statsd(self, records: Iterable[dict], plan_id: str | None) -> None:
        assert self._statsd is not None
        base_tags = {
            "plan": plan_id or "unknown",
            "rank": str(self._rank_id),
            "worker": str(self._worker_id),
        }
        for record in records:
            stage_index = record["stage"]
            key = stage_index
            prev = self._prev_fetch_totals.get(key, {})
            delta_samples = record["samples"] - prev.get("samples", 0)
            delta_resolve_ns = record["resolve_ns"] - prev.get("resolve_ns", 0)
            delta_open_ns = record["open_ns"] - prev.get("open_ns", 0)
            delta_read_ns = record["read_ns"] - prev.get("read_ns", 0)
            delta_close_ns = record["close_ns"] - prev.get("close_ns", 0)
            delta_retries = record["retries"] - prev.get("retries", 0)
            delta_hits = record["cache_hits"] - prev.get("cache_hits", 0)
            delta_misses = record["cache_misses"] - prev.get("cache_misses", 0)
            delta_reopens = record["shard_reopens"] - prev.get("shard_reopens", 0)
            samples = max(delta_samples, 0)
            tags = dict(base_tags)
            tags.update(
                {
                    "stage": str(stage_index),
                    "stage_name": record.get("stage_name") or "unknown",
                }
            )
            denom = max(samples, 1)
            resolve_ms = (delta_resolve_ns / denom) / 1_000_000
            open_ms = (delta_open_ns / denom) / 1_000_000
            read_ms = (delta_read_ns / denom) / 1_000_000
            close_ms = (delta_close_ns / denom) / 1_000_000
            if samples > 0:
                self._statsd.gauge("fetch.resolve_ms", resolve_ms, tags)
                self._statsd.gauge("fetch.open_ms", open_ms, tags)
                self._statsd.gauge("fetch.read_ms", read_ms, tags)
                self._statsd.gauge("fetch.close_ms", close_ms, tags)
            if delta_retries > 0:
                self._statsd.counter("fetch.retries", delta_retries, tags)
            if delta_reopens > 0:
                self._statsd.counter("fetch.shard_reopens", delta_reopens, tags)
            total_accesses = max(delta_hits + delta_misses, 0)
            if total_accesses > 0:
                hit_ratio = delta_hits / total_accesses
                self._statsd.gauge("fetch.cache_hit_ratio", hit_ratio * 100.0, tags)
            self._prev_fetch_totals[key] = {
                "samples": record["samples"],
                "resolve_ns": record["resolve_ns"],
                "open_ns": record["open_ns"],
                "read_ns": record["read_ns"],
                "close_ns": record["close_ns"],
                "retries": record["retries"],
                "cache_hits": record["cache_hits"],
                "cache_misses": record["cache_misses"],
                "shard_reopens": record["shard_reopens"],
            }

    def _emit_pump_logs(
        self, summary: PumpTimingSummary, records: Iterable[dict]
    ) -> None:
        plan_id = summary.plan_id or "unknown"
        for record in records:
            payload = {
                "event": "zephon.pump_metrics",
                "plan_id": plan_id,
                "stage": record["stage"],
                "stage_name": record["stage_name"],
                "op_index": record["op"],
                "op_name": record["name"],
                "input_wait_ns": record["input_wait_ns"],
                "dispatch_wait_ns": record["dispatch_wait_ns"],
                "dispatch_active_ns": record["dispatch_active_ns"],
                "result_wait_ns": record["result_wait_ns"],
                "result_collect_ns": record["result_collect_ns"],
                "result_handle_ns": record["result_handle_ns"],
                "idle_drain_ns": record["idle_drain_ns"],
                "total_ns": record["total_ns"],
                "batches_submitted": record["batches_submitted"],
                "batches_completed": record["batches_completed"],
                "capacity_stalls": record["capacity_stalls"],
                "rank": self._rank_id,
                "worker": self._worker_id,
            }
            if self._sink_config and self._sink_config.json_logs:
                logger.info(json.dumps(payload))
            else:
                total_ns = record["total_ns"] or 1
                logger.info(
                    "[zephon][pump] plan=%s stage=%s/%s op=%s/%s "
                    "submitted=%d completed=%d stalls=%d | "
                    "input_wait=%s dispatch_wait=%s dispatch=%s "
                    "result_wait=%s result_collect=%s result_handle=%s idle=%s",
                    plan_id,
                    record["stage"],
                    record["stage_name"],
                    record["op"],
                    record["name"],
                    record["batches_submitted"],
                    record["batches_completed"],
                    record["capacity_stalls"],
                    pretty_format_ratio(record["input_wait_ns"] / total_ns),
                    pretty_format_ratio(record["dispatch_wait_ns"] / total_ns),
                    pretty_format_ratio(record["dispatch_active_ns"] / total_ns),
                    pretty_format_ratio(record["result_wait_ns"] / total_ns),
                    pretty_format_ratio(record["result_collect_ns"] / total_ns),
                    pretty_format_ratio(record["result_handle_ns"] / total_ns),
                    pretty_format_ratio(record["idle_drain_ns"] / total_ns),
                )

    def _emit_pump_statsd(self, records: Iterable[dict], plan_id: str | None) -> None:
        assert self._statsd is not None
        base_tags = {
            "plan": plan_id or "unknown",
            "rank": str(self._rank_id),
            "worker": str(self._worker_id),
        }
        for record in records:
            key = (record["stage"], record["op"])
            prev = self._prev_pump_totals.get(key, {})
            tags = dict(base_tags)
            tags.update(
                {
                    "stage": str(record["stage"]),
                    "stage_name": record.get("stage_name") or "unknown",
                    "op": str(record["op"]),
                    "op_name": record.get("name") or "unknown",
                }
            )
            metric_base = f"{_sanitize_component(record['stage_name'])}.{_sanitize_component(record['name'])}.pump"
            for bucket in (
                "input_wait_ns",
                "dispatch_wait_ns",
                "dispatch_active_ns",
                "result_wait_ns",
                "result_collect_ns",
                "result_handle_ns",
                "idle_drain_ns",
            ):
                delta_ns = record[bucket] - prev.get(bucket, 0)
                if delta_ns > 0:
                    self._statsd.timer(
                        f"{metric_base}.{bucket[:-3]}",
                        delta_ns / 1_000_000,
                        tags,
                    )
            for counter in (
                "batches_submitted",
                "batches_completed",
                "capacity_stalls",
            ):
                delta = record[counter] - prev.get(counter, 0)
                if delta > 0:
                    self._statsd.counter(f"{metric_base}.{counter}", delta, tags)
            self._prev_pump_totals[key] = {
                field: record[field]
                for field in (
                    "input_wait_ns",
                    "dispatch_wait_ns",
                    "dispatch_active_ns",
                    "result_wait_ns",
                    "result_collect_ns",
                    "result_handle_ns",
                    "idle_drain_ns",
                    "batches_submitted",
                    "batches_completed",
                    "capacity_stalls",
                )
            }
