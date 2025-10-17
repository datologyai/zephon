from typing import Any

from zephon.core.graph import Node, Stage
from zephon.ops.delay import DelayById
from zephon.runners.threads import ThreadStageRunner


def _collect(runner: ThreadStageRunner, data: list[Any]) -> list[Any]:
    return list(runner.run(iter(data)))


def test_runner_emits_in_input_order_when_deterministic() -> None:
    # Build a single-stage plan with a delay op that will reorder completions
    op = DelayById(max_delay_ms=2.0)
    node = Node(name="delay", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")

    data = list(range(100))

    # Deterministic: outputs must match the input order exactly
    det_runner = ThreadStageRunner(
        stage, ctx_services={}, max_workers=8, deterministic=True
    )
    out_det = _collect(det_runner, data)
    assert out_det == data

    # Non-deterministic: should still be a permutation; may equal by chance
    nondet_runner = ThreadStageRunner(
        stage, ctx_services={}, max_workers=8, deterministic=False
    )
    out_nondet = _collect(nondet_runner, data)
    assert sorted(out_nondet) == sorted(data)


def test_run_one_returns_through_single_op_stage() -> None:
    # Single-op stage: DelayById is identity on payloads; run_one should pass through
    op = DelayById(max_delay_ms=0.0)
    node = Node(name="delay", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")

    runner = ThreadStageRunner(
        stage, ctx_services={}, max_workers=2, deterministic=True
    )
    out = runner.run_one(7)
    assert out == 7


def test_set_parallelism_errors_and_adjustments() -> None:
    op = DelayById(max_delay_ms=0.0)
    node = Node(name="delay", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")
    runner = ThreadStageRunner(
        stage, ctx_services={}, max_workers=4, deterministic=True
    )

    # Invalid op index raises
    try:
        runner.set_parallelism(-1, 2)
        assert False, "expected IndexError"
    except IndexError:
        pass
    try:
        runner.set_parallelism(99, 2)
        assert False, "expected IndexError"
    except IndexError:
        pass

    # Grow then shrink while idle should succeed
    runner.set_parallelism(0, 3)
    runner.set_parallelism(0, 1)


def test_prefetching_stage_iterator_close_is_clean() -> None:
    # With prefetch_capacity > 0 we wrap with buffered_iterable; closing early must be clean
    op = DelayById(max_delay_ms=0.0)
    node = Node(name="delay", op=op)
    stage = Stage(name="s", nodes=[node], placement="auto", break_reason="test")
    runner = ThreadStageRunner(
        stage, ctx_services={}, max_workers=2, deterministic=True, prefetch_capacity=4
    )

    it = runner.run(iter(range(100)))
    # pull a few then close early
    got = []
    for _ in range(5):
        got.append(next(it))
    assert got == list(range(5))
    # Explicitly close iterator; should not raise or hang
    if hasattr(it, "close"):
        it.close()  # type: ignore[call-arg]


def test_passthrough_stage_forwards_stream() -> None:
    # Empty stage (no ops) must pass through StreamOut elements
    stage = Stage(name="empty", nodes=[], placement="auto", break_reason="test")
    runner = ThreadStageRunner(
        stage, ctx_services={}, max_workers=2, deterministic=True
    )
    data = list(range(10))
    out = list(runner.run(iter(data)))
    assert out == data
