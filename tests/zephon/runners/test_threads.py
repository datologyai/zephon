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
