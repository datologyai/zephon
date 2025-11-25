# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures/helpers for runner tests."""

from __future__ import annotations

from typing import Iterable

from zephon.core.constants import SampleMeta, SampleRecord
from zephon.observability.stats import NodeMetricsDelta


def _noop_metrics(_: NodeMetricsDelta) -> None:  # pragma: no cover - trivial helper
    return None


def _ctx_services(extra: dict[str, object] | None = None) -> dict[str, object]:
    services: dict[str, object] = {"record_node_metrics": _noop_metrics}
    if extra:
        services.update(extra)
    return services


def _mk_record(value: int) -> SampleRecord:
    meta = SampleMeta(
        sample_id=(0, 0, value),
        lane_id=0,
        chunk_id=0,
        chunk_offset=value,
    )
    return SampleRecord(meta=meta, payload={"value": value})


def _mk_records(values: Iterable[int]) -> list[SampleRecord]:
    return [_mk_record(int(v)) for v in values]


def _extract_values(records: Iterable[SampleRecord]) -> list[int]:
    return [int(rec.payload["value"]) for rec in records]
