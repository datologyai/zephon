# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Process control services across startup, transport, crashes, and shutdown."""

import multiprocessing as mp
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.zephon._internal.runners._helpers import _ctx_services, _mk_records
from zephon._internal.graph import Node, Stage
from zephon._internal.ops.delay import DelayById
from zephon._internal.runners.process import ProcessStageRunner
from zephon.ops.base import BaseOp, OpContext
from zephon.ops.traits import OpTraits

pytestmark = pytest.mark.integration


class _DieDuringSerialization:
    def __reduce__(self) -> Any:
        os.kill(os.getpid(), signal.SIGKILL)


class _PublishingOp(BaseOp):
    def __init__(self, mode: str, marker: str = "") -> None:
        super().__init__()
        self.mode = mode
        self.marker = marker

    def setup(self, ctx: OpContext) -> None:
        super().setup(ctx)
        self._hook = ctx.get("custom_service")
        if (
            self.mode == "before"
            and Path(self.marker).exists()
            and mp.parent_process() is not None
        ):
            self._hook("replacement")

    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True)

    def process_many(self, elems: list[Any]) -> list[Any]:
        if self.mode == "before":
            marker = Path(self.marker)
            if not marker.exists():
                marker.touch()
                os.kill(os.getpid(), signal.SIGKILL)
        elif self.mode == "during":
            self._hook(_DieDuringSerialization())
        elif self.mode == "accepted":
            self._hook(os.getpid())
        else:
            self._hook("publish")
        return elems


def _runner(
    hook: Any,
    *,
    transport: str = "pipe",
    multi: bool = False,
    op: Any = None,
    start: str = "spawn",
) -> ProcessStageRunner:
    nodes = [Node(name="publish", op=op or _PublishingOp("normal"), parallelism=1)]
    if multi:
        nodes.append(Node(name="delay", op=DelayById(max_delay_ms=0), parallelism=1))
    return ProcessStageRunner(
        Stage(name="control", nodes=nodes, placement="auto", break_reason="test"),
        ctx_services=_ctx_services({"custom_service": hook}),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
        ipc_transport=transport,
        mp_context=mp.get_context(start),
        max_worker_retries=0 if start == "fork" else 1,
    )


def _collect(runner: ProcessStageRunner) -> list[Any]:
    return list(runner.run(_mk_records([0])))


# Cover startup methods on one transport; pair transports with execution/failure
# modes below rather than repeating the full Cartesian product.
@pytest.mark.parametrize("start", mp.get_all_start_methods())
def test_initial_worker_can_publish_during_setup(start: str, tmp_path: Path) -> None:
    marker = tmp_path / "publish_immediately"
    marker.touch()
    accepted = []
    runner = _runner(
        accepted.append,
        start=start,
        op=_PublishingOp("before", str(marker)),
    )
    assert len(_collect(runner)) == 1
    assert accepted == ["replacement"]


@pytest.mark.parametrize(
    ("transport", "multi"), [("pipe", False), ("socketpair", True)]
)
def test_no_output_before_control_acknowledgement(transport: str, multi: bool) -> None:
    entered, release = threading.Event(), threading.Event()

    def register(_: Any) -> None:
        entered.set()
        assert release.wait(15), "test did not release acknowledgement"

    runner = _runner(register, transport=transport, multi=multi)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(_collect, runner)
        try:
            assert entered.wait(15)
            assert not result.done(), "data escaped before acknowledgement"
        finally:
            release.set()
        assert len(result.result(timeout=15)) == 1


def test_handler_failure_reaches_consumer() -> None:
    def reject(_: Any) -> None:
        raise ValueError("conflicting metadata")

    with pytest.raises(Exception, match="conflicting metadata"):
        _collect(_runner(reject))


def test_close_interrupts_acknowledgement_wait() -> None:
    entered, release = threading.Event(), threading.Event()

    def register(_: Any) -> None:
        entered.set()
        release.wait(15)

    runner = _runner(register)
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = pool.submit(_collect, runner)
        try:
            assert entered.wait(15)
            closed = pool.submit(runner.close, hard=True)
            closed.result(timeout=10)
            assert runner._service_cancelled.value
        finally:
            release.set()
        try:
            assert result.result(timeout=10) == []
        except RuntimeError as exc:
            assert "cancel" in str(exc).lower()


def test_replacement_can_publish_before_worker_installation(
    tmp_path: Path,
) -> None:
    registered = threading.Event()

    def register(_: Any) -> None:
        registered.set()

    runner = _runner(
        register,
        op=_PublishingOp("before", str(tmp_path / "crashed")),
        start="forkserver",
    )
    install = runner._install_worker

    def install_after_report(
        state: Any, index: int, proc: Any, sem: Any, worker_id: int, response: Any
    ) -> None:
        assert worker_id in runner._service_responses
        if worker_id > 0:
            assert registered.wait(15), "replacement's early service request was lost"
        install(state, index, proc, sem, worker_id, response)

    with patch.object(runner, "_install_worker", install_after_report):
        assert len(_collect(runner)) == 1
    assert registered.is_set()


@pytest.mark.parametrize(
    ("transport", "mode"), [("pipe", "during"), ("socketpair", "accepted")]
)
def test_death_during_control_call_fails_without_reusing_transport(
    transport: str, mode: str
) -> None:
    accepted = []

    def register(pid: int) -> None:
        accepted.append(pid)
        os.kill(pid, signal.SIGKILL)

    runner = _runner(
        register, transport=transport, op=_PublishingOp(mode), start="forkserver"
    )
    with pytest.raises(RuntimeError, match="died during a control service call"):
        _collect(runner)
    assert bool(accepted) == (mode == "accepted")
    assert runner._next_worker_id == 1  # Fail; result-queue recovery is insufficient.


def test_partial_start_failure_cleans_every_endpoint_and_started_process() -> None:
    runner = ProcessStageRunner(
        Stage(
            "start",
            [Node("publish", _PublishingOp("normal"), parallelism=2)],
            "auto",
            "test",
        ),
        ctx_services=_ctx_services({"custom_service": lambda _: None}),
        max_workers=2,
        deterministic=True,
        mp_context=mp.get_context("spawn"),
    )
    start = runner._start_worker
    attempted = []

    def fail_second(proc: Any, worker_id: int) -> None:
        attempted.append(proc)
        if worker_id == 1:
            raise OSError("second worker cannot start")
        start(proc, worker_id)

    with patch.object(runner, "_start_worker", fail_second):
        with pytest.raises(OSError, match="second worker cannot start"):
            _collect(runner)
    assert not runner._service_responses and not runner._service_pending
    assert all(proc.pid is None or not proc.is_alive() for proc in attempted)
