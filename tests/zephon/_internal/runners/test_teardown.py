"""Operator resources are released by runner shutdown, independently of GC."""

import os
import threading
import uuid
from pathlib import Path
from typing import Any

import pytest

from tests.zephon._internal.runners._helpers import _ctx_services, _mk_records
from zephon._internal.graph import Node, Stage
from zephon._internal.runners.inline import InlineStageRunner
from zephon._internal.runners.process import ProcessStageRunner
from zephon._internal.runners.threads import ThreadStageRunner
from zephon.ops.base import BaseOp, OpContext
from zephon.ops.traits import OpTraits

_RunnerType = (
    type[InlineStageRunner] | type[ThreadStageRunner] | type[ProcessStageRunner]
)


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


def _stage(*ops: BaseOp) -> Stage:
    return Stage(
        name="resources",
        nodes=[Node(name=f"op{i}", op=op) for i, op in enumerate(ops)],
        placement="auto",
        break_reason="test",
    )


@pytest.mark.parametrize(
    "runner_type", [InlineStageRunner, ThreadStageRunner, ProcessStageRunner]
)
def test_runner_closes_local_and_worker_resources(
    tmp_path: Path, runner_type: _RunnerType
) -> None:
    runner = runner_type(
        _stage(_ResourceOp(tmp_path)),
        _ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )
    instances = list(runner.ops[0].instances)
    try:
        assert list(runner.run(_mk_records([1, 2]))) == _mk_records([1, 2])
    finally:
        runner.close()
    runner.close()

    assert instances and all(op.resource is None for op in instances)
    assert not list(tmp_path.glob("*.live"))
    expected = 2 if runner_type is ProcessStageRunner else 1
    assert len(list(tmp_path.glob("*.closed"))) == expected


@pytest.mark.parametrize("runner_type", [InlineStageRunner, ThreadStageRunner])
def test_setup_failure_releases_partial_and_previous_operators(
    tmp_path: Path, runner_type: _RunnerType
) -> None:
    with pytest.raises(ValueError, match="setup failed"):
        runner_type(
            _stage(_ResourceOp(tmp_path), _ResourceOp(tmp_path, fail_setup=True)),
            _ctx_services(),
            max_workers=1,
        )
    assert not list(tmp_path.glob("*.live"))
    assert len(list(tmp_path.glob("*.closed"))) == 2


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


@pytest.mark.parametrize("runner_type", [InlineStageRunner, ThreadStageRunner])
def test_close_waits_for_operator_execution(runner_type: _RunnerType) -> None:
    started, proceed, closed, finished = (threading.Event() for _ in range(4))
    runner = runner_type(
        _stage(_BlockingOp()),
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
