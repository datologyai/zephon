# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for ray service: RaySingleOpActor and _RayActorGroup."""

from __future__ import annotations

import os

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


def _make_group(
    *, num_actors: int = 2, tokens_per_actor: int = 1, max_delay_ms: float = 0.0
):
    """Create an actor group with DelayById actors for testing."""
    from zephon.core.graph import Node
    from zephon.ops.delay import DelayById
    from zephon.runners.ray.service import _RayActorGroup

    op = DelayById(max_delay_ms=max_delay_ms)
    node = Node(name="test_op", op=op, parallelism=num_actors)
    group = _RayActorGroup(
        node=node,
        op_index=0,
        num_actors=num_actors,
        tokens_per_actor=tokens_per_actor,
        stage_index=0,
        stage_name="test",
        collect_stats=False,
        ctx_services={},
    )
    group.init(num_cpus_per_actor=0.1)
    return group


@pytest.fixture
def group():
    """Create a group and guarantee shutdown even if the test fails."""
    g = _make_group(num_actors=2)
    yield g
    g.shutdown()


@pytest.fixture
def single_actor_group():
    """Single-actor group for tests that need exactly one worker."""
    g = _make_group(num_actors=1)
    yield g
    g.shutdown()


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
# _RayActorGroup
# ===================================================================


class TestRayActorGroup:
    """Tests for the _RayActorGroup."""

    def test_init_populates_idle_queue(self, group) -> None:
        """After init, all actor indices are in the idle queue."""
        # 2 actors → idle_queue should have 2 entries.
        assert group.idle_queue.qsize() == 2

    def test_release_returns_actor_to_idle_queue(self, single_actor_group) -> None:
        """release() puts the actor index back into the idle queue."""
        # Drain the idle queue (simulating the submit thread taking the actor).
        idx = single_actor_group.idle_queue.get_nowait()
        assert single_actor_group.idle_queue.empty()

        single_actor_group.release(idx)
        assert single_actor_group.idle_queue.qsize() == 1

    def test_direct_actor_dispatch_and_get(self, group) -> None:
        """Dispatch directly to an actor and get the result."""
        from zephon.runners.concurrent import RunnerResult

        records = _mk_records([10, 20, 30])
        actor_idx = group.idle_queue.get_nowait()
        ref = group.actors[actor_idx].process.remote(records, 0)
        result = ray.get(ref)

        assert isinstance(result, RunnerResult)
        assert _extract_values(result.payload) == [10, 20, 30]
        assert result.seq == 0
        assert result.error is None

        group.release(actor_idx)

    def test_multi_token_idle_queue_seeding(self) -> None:
        """Multiple tokens per actor are seeded correctly."""
        g = _make_group(num_actors=2, tokens_per_actor=3)
        try:
            assert g.idle_queue.qsize() == 6

            # Drain all tokens and verify each actor appears exactly 3 times.
            tokens: list[int] = []
            while not g.idle_queue.empty():
                tokens.append(g.idle_queue.get_nowait())
            assert tokens.count(0) == 3
            assert tokens.count(1) == 3
        finally:
            g.shutdown()

    def test_shutdown_kills_actors(self) -> None:
        """shutdown() kills all actors and clears the list."""
        g = _make_group(num_actors=1)
        assert len(g.actors) == 1
        g.shutdown()
        assert len(g.actors) == 0
