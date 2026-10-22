# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures/helpers for runner tests."""

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from zephon._internal.graph import Node, Stage
from zephon.ops.base import BaseOp, OpContext
from zephon.ops.traits import OpTraits
from zephon.types import SampleMeta, SampleRecord

if TYPE_CHECKING:
    from zephon._internal.runners.base import StageRunnerBase
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
    from zephon._internal.graph import Node, Stage
    from zephon._internal.ops.delay import DelayById

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


class _StageInfoProbe(BaseOp):
    def __init__(self, key: str = "stage_info") -> None:
        super().__init__()
        self._key = key

    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True)

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        info = self.stage_info
        stamp = (info.stage_index, info.stage_name, info.op_index, info.collect_stats)
        return [
            SampleRecord(meta=e.meta, payload={**e.payload, self._key: stamp})
            for e in elems
        ]


def _probe_stage(*keys: str) -> "Stage":
    from zephon._internal.graph import Node, Stage

    nodes = [Node(name=key, op=_StageInfoProbe(key=key)) for key in keys]
    return Stage(name="probe_stage", nodes=nodes, placement="auto", break_reason="test")


class _ResourceOp(BaseOp):
    def __init__(self, root: Path, *, fail_setup: bool = False) -> None:
        super().__init__()
        self.root = root
        self.fail_setup = fail_setup
        self.resource: Path | None = None

    def setup(self, ctx: OpContext) -> None:
        super().setup(ctx)
        self.resource = self.root / f"{os.getpid()}-{uuid.uuid4()}.live"
        self.resource.touch()
        if self.fail_setup:
            raise ValueError("setup failed after acquiring a resource")

    def teardown(self) -> None:
        if self.resource is not None:
            self.resource.rename(self.resource.with_suffix(".closed"))
            self.resource = None

    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True)

    def process_many(self, elems: list[Any]) -> list[Any]:
        assert self.resource is not None and self.resource.exists()
        return elems


def _resource_stage(*ops: BaseOp) -> Stage:
    return Stage(
        name="resources",
        nodes=[Node(name=f"op{i}", op=op) for i, op in enumerate(ops)],
        placement="auto",
        break_reason="test",
    )


class _BlockingOp(BaseOp):
    def setup(self, ctx: OpContext) -> None:
        super().setup(ctx)
        self.started = ctx.get("started")
        self.proceed = ctx.get("proceed")
        self.closed = ctx.get("closed")

    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True)

    def process_many(self, elems: list[Any]) -> list[Any]:
        self.started.set()
        assert self.proceed.wait(5)
        assert not self.closed.is_set(), "teardown overlapped operator execution"
        return elems

    def teardown(self) -> None:
        self.closed.set()


def _assert_close_waits_for_operator_execution(
    runner_type: type[StageRunnerBase[Any]],
) -> None:
    started, proceed, closed, finished = (threading.Event() for _ in range(4))
    runner = runner_type(
        _resource_stage(_BlockingOp()),
        _ctx_services({"started": started, "proceed": proceed, "closed": closed}),
        max_workers=1,
        prefetch_capacity=1,
    )
    errors: list[BaseException] = []

    def consume() -> None:
        try:
            list(runner.run(_mk_records([1])))
        except BaseException as exc:
            errors.append(exc)

    def close() -> None:
        try:
            runner.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    consumer = threading.Thread(target=consume, daemon=True)
    closer = threading.Thread(target=close, daemon=True)
    consumer.start()
    try:
        assert started.wait(5)
        closer.start()
        assert not finished.wait(0.1)
        assert not closed.is_set()
    finally:
        proceed.set()
        consumer.join(5)
        if closer.ident is not None:
            closer.join(5)
    assert not consumer.is_alive() and not closer.is_alive()
    assert finished.is_set() and closed.is_set()
    assert not errors
