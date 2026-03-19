# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures/helpers for runner tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from zephon.core.constants import SampleMeta, SampleRecord

if TYPE_CHECKING:
    from zephon.core.graph import Node, Stage
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


def _collect(runner: object, data: Iterable[int]) -> list[int]:
    """Run data through a runner and extract output values."""
    records = _mk_records(data)
    out = list(runner.run(iter(records)))  # type: ignore[union-attr]
    # Flatten microbatches if needed
    flat: list[SampleRecord] = []
    for item in out:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return _extract_values(flat)


def _make_stage(
    *,
    max_delay_ms: float = 1.0,
    parallelism: int = 2,
    placement: str = "auto",
    num_ops: int = 1,
    nodes: "list[Node] | None" = None,
) -> "Stage":
    """Create a Stage for testing with configurable options."""
    from zephon.core.graph import Node, Stage
    from zephon.ops.delay import DelayById

    if nodes is not None:
        return Stage(
            name="test_stage",
            nodes=nodes,
            placement=placement,
            break_reason="test",
        )
    built_nodes = [
        Node(
            name=f"delay{i}",
            op=DelayById(max_delay_ms=max_delay_ms),
            parallelism=parallelism,
        )
        for i in range(num_ops)
    ]
    return Stage(
        name="test_stage",
        nodes=built_nodes,
        placement=placement,
        break_reason="test",
    )
