# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""A spawn bootstrap failure reaches the consumer without waiting for output."""

import multiprocessing as mp
from typing import Any

import pytest

from tests.zephon._internal.runners._helpers import _ctx_services, _mk_records
from zephon._internal.graph import Node, Stage
from zephon._internal.ops.delay import DelayById
from zephon._internal.runners import process as process_mod
from zephon._internal.runners.process import ProcessStageRunner

pytestmark = pytest.mark.integration


def _fail_unpickling() -> None:
    raise ImportError("intentional worker bootstrap failure")


class _BootstrapFailure:
    def __reduce__(self) -> Any:
        return _fail_unpickling, ()


@pytest.mark.timeout(15)
def test_spawn_bootstrap_failure_reaches_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(process_mod, "_WATCHDOG_POLL_S", 0.05)
    runner = ProcessStageRunner(
        Stage(
            name="startup",
            nodes=[Node(name="delay", op=DelayById(max_delay_ms=0), parallelism=1)],
            placement="auto",
            break_reason="test",
        ),
        ctx_services=_ctx_services({"bootstrap": _BootstrapFailure()}),
        max_workers=1,
        # Retry-capable pools retain queue endpoints for replacement workers;
        # a bootstrap failure must not wait for a result or an EOF indefinitely.
        max_worker_retries=1,
        mp_context=mp.get_context("spawn"),
        stage_output_mode="stream_items",
    )
    try:
        with pytest.raises(RuntimeError, match="exited with code 1.*unpickle"):
            list(runner.run(_mk_records([1])))
    finally:
        runner.close(hard=True)
