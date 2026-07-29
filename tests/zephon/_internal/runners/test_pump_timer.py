"""Tests for the pump-thread timer used by concurrent stage runners."""

from __future__ import annotations

import time

from zephon._internal.runners.pump_timer import PumpTimer


def test_disabled_timer_is_noop() -> None:
    timer = PumpTimer(enabled=False)
    with timer.measure("input_wait"):
        time.sleep(0.001)
    timer.note("batches_submitted")
    assert timer.input_wait_ns == 0
    assert timer.batches_submitted == 0
    assert timer.flush() is None


def test_measure_accumulates_into_bucket() -> None:
    timer = PumpTimer(enabled=True)
    with timer.measure("input_wait"):
        time.sleep(0.005)
    with timer.measure("input_wait"):
        time.sleep(0.005)
    assert timer.input_wait_ns > 0
    # Both measurements add to the same bucket.
    snapshot = timer.input_wait_ns
    with timer.measure("dispatch_active"):
        time.sleep(0.005)
    assert timer.dispatch_active_ns > 0
    assert timer.input_wait_ns == snapshot  # untouched by the second measure


def test_measure_excluding_subtracts_inner_growth() -> None:
    timer = PumpTimer(enabled=True)
    with timer.measure_excluding("idle_drain", "result_wait"):
        with timer.measure("result_wait"):
            time.sleep(0.01)
        # A bit of unaccounted residual remains in the outer bucket
        # from loop overhead and the inner measure() bookkeeping.
    # result_wait carries the inner duration; idle_drain has only the residual.
    assert timer.result_wait_ns > 0
    assert timer.idle_drain_ns >= 0
    assert timer.idle_drain_ns < timer.result_wait_ns


def test_note_increments_counter() -> None:
    timer = PumpTimer(enabled=True)
    timer.note("batches_submitted")
    timer.note("batches_submitted", count=3)
    timer.note("capacity_stalls")
    assert timer.batches_submitted == 4
    assert timer.capacity_stalls == 1


def test_should_flush_respects_interval() -> None:
    timer = PumpTimer(enabled=True)
    timer.start_window(now_ns=0)
    assert not timer.should_flush(now_ns=999, interval_ns=1_000)
    assert timer.should_flush(now_ns=1_000, interval_ns=1_000)


def test_flush_resets_window_and_returns_delta() -> None:
    timer = PumpTimer(
        enabled=True,
        stage_index=2,
        op_index=1,
        stage_name="stageX",
        op_name="opY",
    )
    timer.start_window(now_ns=0)
    with timer.measure("input_wait"):
        time.sleep(0.002)
    timer.note("batches_submitted", count=5)

    delta = timer.flush()
    assert delta is not None
    assert delta.stage_index == 2
    assert delta.op_index == 1
    assert delta.stage_name == "stageX"
    assert delta.name == "opY"
    assert delta.input_wait_ns > 0
    assert delta.batches_submitted == 5

    # After flush, all accumulators reset to zero.
    assert timer.input_wait_ns == 0
    assert timer.batches_submitted == 0
