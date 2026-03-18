# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for ray service: RaySingleOpActor and _RayResultQueueAdapter."""

from __future__ import annotations

import os
import queue

import pytest

from tests.zephon.runners._helpers import (
    _extract_values,
    _mk_records,
)

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    """Initialize Ray for testing."""
    session_scope = os.environ.get("ZEPHON_RAY_SESSION_SCOPE") == "session"
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_cpus=4,
            runtime_env={"working_dir": None},
        )
    yield
    if not session_scope and ray.is_initialized():
        ray.shutdown()


def _make_pool(*, num_actors: int = 2, max_delay_ms: float = 0.0):
    """Create a pool with DelayById actors for testing."""
    from zephon.core.graph import Node
    from zephon.ops.delay import DelayById
    from zephon.runners.ray.service import _RayOperatorPool

    op = DelayById(max_delay_ms=max_delay_ms)
    node = Node(name="test_op", op=op, parallelism=num_actors)
    pool = _RayOperatorPool(
        node=node,
        op_index=0,
        num_actors=num_actors,
        stage_index=0,
        stage_name="test",
        collect_stats=False,
        ctx_services={},
    )
    pool.init(num_cpus_per_actor=0.1)
    return pool


@pytest.fixture
def pool():
    """Create a pool and guarantee shutdown even if the test fails."""
    p = _make_pool(num_actors=2)
    yield p
    p.shutdown()


@pytest.fixture
def single_actor_pool():
    """Single-actor pool for tests that need exactly one worker."""
    p = _make_pool(num_actors=1)
    yield p
    p.shutdown()


# ===================================================================
# RaySingleOpActor
# ===================================================================


def _spawn_actor(op, *, op_name: str = "test_op", collect_stats: bool = False):
    """Spawn a single RaySingleOpActor. Caller must kill it when done."""
    import cloudpickle

    from zephon.runners.ray.service import RaySingleOpActor

    op_bytes = cloudpickle.dumps(op)
    return RaySingleOpActor.remote(
        op_bytes,
        ctx_services={},
        stage_index=0,
        stage_name="test_stage",
        op_index=0,
        op_name=op_name,
        collect_stats=collect_stats,
    )


@pytest.fixture
def delay_actor():
    """Spawn a DelayById actor and kill it after the test."""
    from zephon.ops.delay import DelayById

    actor = _spawn_actor(DelayById(max_delay_ms=0.1), collect_stats=True)
    yield actor
    ray.kill(actor)


class TestRaySingleOpActor:
    """Tests for RaySingleOpActor per-operator execution."""

    def test_process_returns_runner_result_with_stats(self, delay_actor) -> None:
        """Actor returns RunnerResult with correct payload and stats."""
        from zephon.runners.concurrent import RunnerResult

        records = _mk_records([1, 2, 3])
        result = ray.get(delay_actor.process.remote(records, 0))

        assert isinstance(result, RunnerResult)
        assert result.error is None
        assert _extract_values(result.payload) == [1, 2, 3]
        assert result.consumed_elements == 3
        assert result.proc_ns > 0

    def test_error_returns_runner_result_with_error(self) -> None:
        """Errors are wrapped in RunnerResult.error (not raised)."""
        from zephon.core.op_base import DefaultSetup
        from zephon.core.traits import OpTraits
        from zephon.runners.concurrent import RunnerResult

        class FailOp(DefaultSetup):
            def traits(self) -> OpTraits:
                return OpTraits(indexable=True, preserves_cursor_order=True)

            def process_one(self, elem):
                raise ValueError("intentional failure")

            def process_many(self, elems):
                raise ValueError("intentional failure")

        actor = _spawn_actor(FailOp(), op_name="fail_op")
        try:
            records = _mk_records([1])
            result = ray.get(actor.process.remote(records, 0))

            assert isinstance(result, RunnerResult)
            assert result.error is not None
            assert "intentional failure" in result.error.message
        finally:
            ray.kill(actor)

    def test_attribute_error_in_process_many_not_masked(self) -> None:
        """AttributeError raised inside process_many is surfaced as an error."""
        from zephon.core.op_base import DefaultSetup
        from zephon.core.traits import OpTraits
        from zephon.runners.concurrent import RunnerResult

        class BrokenProcessManyOp(DefaultSetup):
            def traits(self) -> OpTraits:
                return OpTraits(indexable=True, preserves_cursor_order=True)

            def process_many(self, elems):
                return [self.missing_attr for _ in elems]

            def process_one(self, elem):
                return [elem]

        actor = _spawn_actor(BrokenProcessManyOp(), op_name="broken_op")
        try:
            records = _mk_records([1])
            result = ray.get(actor.process.remote(records, 0))

            assert isinstance(result, RunnerResult)
            assert result.error is not None
            assert "missing_attr" in result.error.message
        finally:
            ray.kill(actor)


# ===================================================================
# _RayResultQueueAdapter
# ===================================================================


class TestRayResultQueueAdapter:
    """Tests for the _RayResultQueueAdapter."""

    def test_get_returns_runner_result(self, pool) -> None:
        """get() returns RunnerResult directly from the pool."""
        from zephon.runners.concurrent import RunnerResult
        from zephon.runners.ray.service import _RayResultQueueAdapter

        records = _mk_records([10, 20, 30])
        pool.submit(records, seq=0)

        rq = _RayResultQueueAdapter(pool)
        result = rq.get(timeout=5.0)

        assert isinstance(result, RunnerResult)
        assert _extract_values(result.payload) == [10, 20, 30]
        assert result.seq == 0
        assert result.error is None

    def test_get_nowait_raises_empty_when_nothing_ready(
        self, single_actor_pool
    ) -> None:
        """get_nowait() raises queue.Empty when no results are available."""
        from zephon.runners.ray.service import _RayResultQueueAdapter

        rq = _RayResultQueueAdapter(single_actor_pool)

        with pytest.raises(queue.Empty):
            rq.get_nowait()

    def test_empty_reflects_pool_state(self, single_actor_pool) -> None:
        """empty() returns True when no work is pending."""
        from zephon.runners.ray.service import _RayResultQueueAdapter

        rq = _RayResultQueueAdapter(single_actor_pool)

        assert rq.empty()

        single_actor_pool.submit([1], seq=0)
        assert not rq.empty()

        rq.get(timeout=5.0)
        assert rq.empty()

    def test_error_result_has_worker_error_info(self) -> None:
        """Errors in actor come back as WorkerErrorInfo in RunnerResult."""
        from zephon.core.graph import Node
        from zephon.core.op_base import DefaultSetup
        from zephon.core.traits import OpTraits
        from zephon.runners.ray.service import _RayOperatorPool, _RayResultQueueAdapter

        class FailOp(DefaultSetup):
            def traits(self) -> OpTraits:
                return OpTraits(indexable=True, preserves_cursor_order=True)

            def process_one(self, elem):
                raise ValueError("test error")

            def process_many(self, elems):
                raise ValueError("test error")

        node = Node(name="fail", op=FailOp(), parallelism=1)
        pool = _RayOperatorPool(
            node=node,
            op_index=0,
            num_actors=1,
            stage_index=0,
            stage_name="test",
            collect_stats=False,
            ctx_services={},
        )
        pool.init(num_cpus_per_actor=0.1)

        try:
            pool.submit(_mk_records([1]), seq=0)

            rq = _RayResultQueueAdapter(pool)
            result = rq.get(timeout=5.0)

            assert result.error is not None
            assert "test error" in result.error.message
        finally:
            pool.shutdown()
