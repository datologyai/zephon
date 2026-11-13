"""Tests for idle_drain accounting in the queue-drain runners' result sweep."""

from __future__ import annotations

import queue
import time
from types import SimpleNamespace
from typing import Any

import pytest

from zephon._internal.runners.process import ProcessStageRunner
from zephon._internal.runners.pump_timer import PumpTimer
from zephon._internal.runners.queue_drain import QueueDrainStageRunner
from zephon._internal.runners.threads import ThreadStageRunner

_READ_S = 0.01
_READ_NS = int(_READ_S * 1e9)


class _SlowQueue:
    """Result queue whose reads, empty or not, sleep: the first ``first_s``."""

    def __init__(self, items: list[Any], first_s: float = _READ_S) -> None:
        self._items = list(items)
        self._next_s = first_s

    def get_nowait(self) -> Any:
        time.sleep(self._next_s)
        self._next_s = _READ_S
        if not self._items:
            raise queue.Empty
        return self._items.pop(0)


def _drain(
    runner_cls: type[QueueDrainStageRunner[Any]],
    items: list[Any],
    first_s: float = _READ_S,
) -> tuple[PumpTimer, list[Any]]:
    # _drain_results only touches the queue, the timer, and these two hooks,
    # so skip building a real stage.
    runner = object.__new__(runner_cls)
    handled: list[Any] = []
    runner._handle_result = lambda state, item, next_queue, ctx: handled.append(item)  # type: ignore[method-assign]
    runner._record_error = lambda ctx, exc: pytest.fail(f"unexpected error {exc!r}")  # type: ignore[method-assign]
    state = SimpleNamespace(
        result_queue=_SlowQueue(items, first_s), pump_timer=PumpTimer(enabled=True)
    )
    runner._drain_results(state, None, SimpleNamespace())  # type: ignore[arg-type]
    return state.pump_timer, handled


def test_process_drain_times_every_read() -> None:
    timer, handled = _drain(ProcessStageRunner, ["a", "b"])
    assert handled == ["a", "b"]
    assert timer.batches_completed == 2
    # Both successful reads (receive + unpickle on a real IPC queue) and the
    # final empty poll stay in idle_drain.
    assert timer.idle_drain_ns >= 3 * _READ_NS


def test_process_drain_times_empty_poll() -> None:
    timer, handled = _drain(ProcessStageRunner, [])
    assert handled == []
    assert timer.idle_drain_ns >= _READ_NS


def test_thread_drain_skips_timing_empty_poll() -> None:
    timer, handled = _drain(ThreadStageRunner, [])
    assert handled == []
    assert timer.idle_drain_ns == 0


def test_thread_drain_times_reads_after_the_first() -> None:
    first_s = 0.5
    timer, handled = _drain(ThreadStageRunner, ["a", "b"], first_s=first_s)
    assert handled == ["a", "b"]
    assert timer.batches_completed == 2
    # The (slow) first read is untimed; the second read and the final empty
    # poll are inside idle_drain.
    assert 2 * _READ_NS <= timer.idle_drain_ns < int(first_s * 1e9)


@pytest.mark.parametrize("runner_cls", [ThreadStageRunner, ProcessStageRunner])
@pytest.mark.parametrize("wait", [False, True])
def test_result_reads_use_queue_replacement_hook(
    runner_cls: type[QueueDrainStageRunner[Any]], wait: bool
) -> None:
    items: list[Any] = ["replacement result"]
    handled: list[Any] = []
    if not wait and runner_cls._UNTIMED_FIRST_POLL:
        items.insert(0, None)

    class ReplacedQueueRunner(runner_cls):
        def _get_result(self, state: Any, timeout: float | None = None) -> Any:
            if not items:
                raise queue.Empty
            return items.pop(0)

    class RetiredQueue:
        def get_nowait(self) -> Any:
            raise AssertionError("result hook must select the replacement queue")

    runner = object.__new__(ReplacedQueueRunner)
    runner._handle_result = lambda state, item, next_queue, ctx: handled.append(item)
    state = SimpleNamespace(
        result_queue=RetiredQueue(), pump_timer=PumpTimer(enabled=True)
    )
    if wait:
        runner._await_one_result(state, None, SimpleNamespace(), timeout=0.1)
    else:
        runner._drain_results(state, None, SimpleNamespace())
    assert handled == ["replacement result"]
    assert state.pump_timer.batches_completed == 1
