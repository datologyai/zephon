# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Process workers must not outlive an owner killed without cleanup."""

import multiprocessing as mp
import os
import signal
import sys
import time
from typing import Any

import pytest

from tests.zephon._internal.runners._helpers import _ctx_services, _mk_records
from zephon._internal.graph import Node, Stage
from zephon._internal.ops.delay import DelayById
from zephon._internal.runners.process import ProcessStageRunner

pytestmark = pytest.mark.integration


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        # Orphans may stay zombies where PID 1 does not reap them.
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def _own_workers_until_killed(conn: Any) -> None:
    """Start a runner mid-stream, report its worker pids, then wait."""
    node = Node(name="delay", op=DelayById(max_delay_ms=0.0))
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")
    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
    )
    stream = runner.run(iter(_mk_records(range(64))))
    next(stream)
    conn.send([proc.pid for state in runner.ops for proc in state.workers])
    time.sleep(300)


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX signals")
@pytest.mark.timeout(120)
def test_workers_exit_when_owner_is_killed() -> None:
    ctx = mp.get_context("spawn")
    reader, writer = ctx.Pipe(duplex=False)
    owner = ctx.Process(target=_own_workers_until_killed, args=(writer,))
    owner.start()
    writer.close()
    pids: list[int] = []
    try:
        assert reader.poll(60), "owner never started its workers"
        pids = reader.recv()
        assert pids and all(_pid_running(pid) for pid in pids)
        os.kill(owner.pid, signal.SIGKILL)
        owner.join(timeout=10)
        # Workers poll for parent death every second.
        deadline = time.monotonic() + 10
        while any(_pid_running(pid) for pid in pids):
            assert time.monotonic() < deadline, "worker outlived its killed owner"
            time.sleep(0.05)
    finally:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if owner.is_alive():
            owner.kill()
        owner.join(timeout=5)
        owner.close()
