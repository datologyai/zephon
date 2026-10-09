# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for RuntimeSpec dataclasses and resolve_runtime_spec."""

import pytest

from zephon._internal.graph import Graph
from zephon._internal.ops.delay import DelayById
from zephon._internal.planner import Planner
from zephon._internal.runtime_spec import (
    StageRuntimeSpec,
    apportion,
    resolve_runtime_spec,
    stage_parallelism,
)
from zephon.options import RuntimeOptions


def _mk_plan(pars=(2, 5, 1)):
    """Build a multi-stage plan by using placement changes to force stage breaks."""
    g = Graph()
    g.add("op0", DelayById(max_delay_ms=0.0), parallelism=pars[0])
    g.add(
        "op1",
        DelayById(max_delay_ms=0.0),
        *g.nodes,
        placement="local",
        parallelism=pars[1],
    )
    g.add(
        "op2",
        DelayById(max_delay_ms=0.0),
        *g.nodes,
        placement="local2",
        parallelism=pars[2],
    )
    return Planner().make_plan(g)


def _mk_simple_plan():
    """Single-stage plan with one op."""
    g = Graph()
    g.add("delay", DelayById(max_delay_ms=0.0), parallelism=3)
    return Planner().make_plan(g)


# ---------------------------------------------------------------------------
# RuntimeSpec dataclass tests
# ---------------------------------------------------------------------------


def test_runtime_spec_is_frozen():
    spec = resolve_runtime_spec(_mk_simple_plan(), RuntimeOptions())
    with pytest.raises(AttributeError):
        spec.deterministic = True  # type: ignore[misc]


def test_stage_runtime_spec_is_frozen():
    spec = resolve_runtime_spec(_mk_simple_plan(), RuntimeOptions())
    with pytest.raises(AttributeError):
        spec.stages[0].worker_cap = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# resolve_runtime_spec — allocation modes
# ---------------------------------------------------------------------------


def test_fit_to_ops_allocation():
    plan = _mk_plan((2, 5, 1))
    spec = resolve_runtime_spec(plan, RuntimeOptions())
    assert spec.worker_allocation == "fit_to_ops"
    caps = [s.worker_cap for s in spec.stages]
    assert caps == [2, 5, 1]


def test_per_stage_fixed_allocation():
    plan = _mk_plan((2, 5, 1))
    opts = RuntimeOptions(worker_allocation="per_stage_fixed", max_workers=7)
    spec = resolve_runtime_spec(plan, opts)
    assert spec.worker_allocation == "per_stage_fixed"
    caps = [s.worker_cap for s in spec.stages]
    assert caps == [7, 7, 7]


def test_global_allocation():
    plan = _mk_plan((2, 5, 1))
    opts = RuntimeOptions(worker_allocation="global", max_workers=16)
    spec = resolve_runtime_spec(plan, opts)
    caps = [s.worker_cap for s in spec.stages]
    assert caps == [4, 10, 2]


def test_global_bumps_when_less_than_stages():
    plan = _mk_plan((2, 5, 1))
    with pytest.warns(RuntimeWarning):
        spec = resolve_runtime_spec(
            plan,
            RuntimeOptions(worker_allocation="global", max_workers=2),
        )
    caps = [s.worker_cap for s in spec.stages]
    assert caps == [1, 1, 1]


def test_autotune_raises():
    plan = _mk_simple_plan()
    with pytest.raises(NotImplementedError):
        resolve_runtime_spec(
            plan, RuntimeOptions(worker_allocation="autotune", max_workers=8)
        )


# ---------------------------------------------------------------------------
# Runner type selection
# ---------------------------------------------------------------------------


def test_default_runner_is_threads():
    plan = _mk_simple_plan()
    spec = resolve_runtime_spec(plan, RuntimeOptions())
    assert spec.stages[0].runner_type == "threads"


def test_runner_override():
    plan = _mk_simple_plan()
    spec = resolve_runtime_spec(plan, RuntimeOptions(runner="process"))
    assert spec.stages[0].runner_type == "process"


def test_inside_worker_demotes_process_to_threads():
    plan = _mk_simple_plan()
    spec = resolve_runtime_spec(
        plan, RuntimeOptions(runner="process"), inside_worker=True
    )
    assert spec.stages[0].runner_type == "threads"


# ---------------------------------------------------------------------------
# Output mode
# ---------------------------------------------------------------------------


def test_last_stage_is_stream_items():
    plan = _mk_plan((2, 5, 1))
    spec = resolve_runtime_spec(plan, RuntimeOptions())
    assert spec.stages[-1].output_mode == "stream_items"
    # Non-last stages are microbatches
    for s in spec.stages[:-1]:
        assert s.output_mode == "microbatches"


# ---------------------------------------------------------------------------
# explain() format
# ---------------------------------------------------------------------------


def test_explain_contains_allocation_header():
    plan = _mk_simple_plan()
    spec = resolve_runtime_spec(plan, RuntimeOptions())
    text = spec.explain(plan)
    assert "Allocation=fit_to_ops" in text


def test_explain_contains_stage_info():
    plan = _mk_plan((2, 5, 1))
    spec = resolve_runtime_spec(plan, RuntimeOptions())
    text = spec.explain(plan)
    assert "Stage[0]" in text
    assert "Stage[1]" in text
    assert "Stage[2]" in text
    assert "pipeline_end" in text


def test_explain_contains_bookkeeping():
    plan = _mk_simple_plan()
    spec = resolve_runtime_spec(plan, RuntimeOptions())
    text = spec.explain(plan)
    assert "Bookkeeping=" in text


# ---------------------------------------------------------------------------
# Pipeline-level flags
# ---------------------------------------------------------------------------


def test_deterministic_flag():
    plan = _mk_simple_plan()
    spec = resolve_runtime_spec(plan, RuntimeOptions(deterministic=True))
    assert spec.deterministic is True

    spec2 = resolve_runtime_spec(plan, RuntimeOptions(deterministic=False))
    assert spec2.deterministic is False


def test_coalesce_tensors_defaults_on() -> None:
    plan = _mk_simple_plan()
    spec = resolve_runtime_spec(plan, RuntimeOptions())
    assert spec.stages[0].coalesce_tensors is True


def test_final_prefetch():
    plan = _mk_simple_plan()
    spec = resolve_runtime_spec(plan, RuntimeOptions(prefetch_batches=8))
    assert spec.final_prefetch == 8


# ---------------------------------------------------------------------------
# apportion edge cases
# ---------------------------------------------------------------------------


def test_apportion_zero_total():
    assert apportion(0, [1, 2, 3]) == [0, 0, 0]


def test_apportion_empty_weights():
    assert apportion(5, []) == []


def test_apportion_all_zero_weights():
    result = apportion(5, [0, 0, 0, 0])
    assert sum(result) == 5
    assert all(r >= 1 for r in result)


def test_apportion_negative_weights():
    # Non-positive weights → equal split
    result = apportion(5, [0, 0, -1, 0])
    assert sum(result) == 5
    assert all(r >= 1 for r in result)


def test_apportion_proportional():
    assert apportion(16, [2, 5, 1]) == [4, 10, 2]


# ---------------------------------------------------------------------------
# stage_parallelism
# ---------------------------------------------------------------------------


def test_stage_parallelism_sums_node_parallelism():
    g = Graph()
    g.add("a", DelayById(max_delay_ms=0.0), parallelism=3)
    g.add("b", DelayById(max_delay_ms=0.0), *g.nodes, parallelism=5)
    plan = Planner().make_plan(g)
    # Single stage with both ops
    assert len(plan.stages) == 1
    par = stage_parallelism(plan.stages[0])
    assert par == 8  # 3 + 5


# ---------------------------------------------------------------------------
# max_worker_retries plumbing (RuntimeOptions -> StageRuntimeSpec)
# ---------------------------------------------------------------------------


def test_max_worker_retries_default_and_override() -> None:
    """``RuntimeOptions.max_worker_retries`` defaults to 3 and accepts overrides."""
    assert RuntimeOptions().max_worker_retries == 3

    opts = RuntimeOptions()
    opts.max_worker_retries = 11
    assert opts.max_worker_retries == 11


def test_max_worker_retries_flows_through_runtime_spec() -> None:
    """``.options(max_worker_retries=N)`` lands in each ``StageRuntimeSpec``."""
    plan = _mk_plan((2, 5, 1))
    spec = resolve_runtime_spec(plan, RuntimeOptions(max_worker_retries=11))
    for stage_spec in spec.stages:
        assert stage_spec.max_worker_retries == 11


def test_stage_runtime_spec_default_max_worker_retries_is_zero() -> None:
    """Direct ``StageRuntimeSpec`` construction (no ``RuntimeOptions``) defaults to 0.

    This default keeps backwards compatibility for callers that build specs
    by hand — they opt into resilience by passing the field explicitly.
    """
    stage_spec = StageRuntimeSpec(
        stage_index=0,
        runner_type="threads",
        worker_cap=1,
        queue_capacity=1,
        prefetch_capacity=0,
        output_mode="microbatches",
        allow_latency_flush=True,
        coalesce_tensors=False,
        shm_min_size=1,
    )
    assert stage_spec.max_worker_retries == 0


def test_payload_memory_policy_options_reach_process_stages() -> None:
    opts = RuntimeOptions(
        shm_min_size=128,
        shm_min_buffer_size=2048,
        shm_min_reuse_size=1024,
        coalesce_tensors=False,
        shm_max_retained_ratio=4,
        shm_min_reclaim_bytes=1024,
        shm_coalesce_max_size=8192,
    )
    spec = resolve_runtime_spec(_mk_plan((2, 1, 1)), opts)
    for stage in spec.stages:
        assert stage.shm_min_size == 128
        assert stage.shm_min_buffer_size == 2048
        assert stage.shm_min_reuse_size == 1024
        assert not stage.coalesce_tensors
        assert stage.shm_max_retained_ratio == 4
        assert stage.shm_min_reclaim_bytes == 1024
        assert stage.shm_coalesce_max_size == 8192
    with pytest.raises(ValueError, match="shm_max_retained_ratio"):
        resolve_runtime_spec(
            _mk_plan((1, 1, 1)), RuntimeOptions(shm_max_retained_ratio=0.5)
        )
