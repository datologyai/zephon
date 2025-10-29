"""Configuration primitives for Zephon observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, MutableMapping


class ExecutionTrackingMode(str, Enum):
    """Controls how much execution detail is collected."""

    OFF = "off"
    STAGES = "stages"
    NODES = "nodes"

    @classmethod
    def from_value(
        cls, value: "ExecutionTrackingMode | str | None"
    ) -> "ExecutionTrackingMode":
        if value is None:
            return cls.OFF
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported execution tracking mode: {value!r}") from exc

    @property
    def collects_nodes(self) -> bool:
        return self is ExecutionTrackingMode.NODES

    @property
    def collects_stages(self) -> bool:
        return self in (ExecutionTrackingMode.STAGES, ExecutionTrackingMode.NODES)


class MetricsSinkMode(str, Enum):
    """Where metrics should be emitted."""

    LOG = "log"
    STATSD = "statsd"
    BOTH = "both"

    @classmethod
    def from_value(cls, value: "MetricsSinkMode | str | None") -> "MetricsSinkMode":
        if value is None:
            return cls.LOG
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported metrics sink mode: {value!r}") from exc

    def includes_logs(self) -> bool:
        return self in (MetricsSinkMode.LOG, MetricsSinkMode.BOTH)

    def includes_statsd(self) -> bool:
        return self in (MetricsSinkMode.STATSD, MetricsSinkMode.BOTH)


@dataclass(frozen=True)
class MetricsSinkConfig:
    """Describe how observability data is exported."""

    mode: MetricsSinkMode = MetricsSinkMode.LOG
    namespace: str = "zephon.pipeline"
    statsd_host: str = "127.0.0.1"
    statsd_port: int = 8125
    default_tags: Mapping[str, str] = field(default_factory=dict)
    json_logs: bool = True
    max_batch_size: int = 32
    flush_interval_s: float = 5.0

    def should_log(self) -> bool:
        return self.mode.includes_logs()

    def should_emit_statsd(self) -> bool:
        return self.mode.includes_statsd()

    def with_additional_tags(self, extra: Mapping[str, str]) -> "MetricsSinkConfig":
        if not extra:
            return self
        merged: MutableMapping[str, str] = dict(self.default_tags)
        merged.update(extra)
        return MetricsSinkConfig(
            mode=self.mode,
            namespace=self.namespace,
            statsd_host=self.statsd_host,
            statsd_port=self.statsd_port,
            default_tags=merged,
            json_logs=self.json_logs,
            max_batch_size=self.max_batch_size,
            flush_interval_s=self.flush_interval_s,
        )
