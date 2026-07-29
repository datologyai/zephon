# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for ray service: RaySingleOpActor and _RayActorGroup."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.requires_ray, pytest.mark.usefixtures("ray_init")]

from tests.zephon._internal.runners._helpers import (
    _extract_values,
    _mk_records,
)

ray = pytest.importorskip("ray")

from zephon._internal.graph import Node
from zephon._internal.ops.delay import DelayById
from zephon._internal.runners.ray.service import _RayActorGroup


def _make_group(*, num_actors: int = 2, max_delay_ms: float = 0.0):
    """Create an actor group with DelayById actors for testing."""
    op = DelayById(max_delay_ms=max_delay_ms)
    node = Node(name="test_op", op=op, parallelism=num_actors)
    group = _RayActorGroup(
        node=node,
        op_index=0,
        num_actors=num_actors,
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

    from zephon._internal.runners.ray.service import RaySingleOpActor

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
    from zephon._internal.ops.delay import DelayById

    actor = _spawn_actor(DelayById(max_delay_ms=0.1), collect_stats=True)
    yield actor
    ray.kill(actor)


class TestRaySingleOpActor:
    """Tests for RaySingleOpActor per-operator execution."""

    def test_process_returns_runner_result_with_stats(self, delay_actor) -> None:
        """Actor returns RunnerResult with correct payload and stats."""
        from zephon._internal.runners.concurrent import RunnerResult

        records = _mk_records([1, 2, 3])
        result = ray.get(delay_actor.process.remote(records, 0))

        assert isinstance(result, RunnerResult)
        assert result.error is None
        assert _extract_values(result.payload) == [1, 2, 3]
        assert result.consumed_elements == 3
        assert result.proc_ns > 0

    def test_actor_setup_provides_stage_info(self) -> None:
        from zephon._internal.runners.concurrent import RunnerResult
        from zephon.ops.base import BaseOp
        from zephon.ops.traits import OpTraits
        from zephon.types import SampleRecord

        class StageInfoProbe(BaseOp):
            def traits(self) -> OpTraits:
                return OpTraits(preserves_cursor_order=True)

            def process_many(self, elems):
                info = self.stage_info
                stamp = (
                    info.stage_index,
                    info.stage_name,
                    info.op_index,
                    info.collect_stats,
                )
                return [
                    SampleRecord(
                        meta=e.meta, payload={**e.payload, "stage_info": stamp}
                    )
                    for e in elems
                ]

        actor = _spawn_actor(StageInfoProbe(), op_name="probe_op")
        try:
            result = ray.get(actor.process.remote(_mk_records([1]), 0))
            assert isinstance(result, RunnerResult)
            assert result.error is None
            (rec,) = result.payload
            assert rec.payload["stage_info"] == (0, "test_stage", 0, False)
        finally:
            ray.kill(actor)

    def test_error_returns_runner_result_with_error(self) -> None:
        """Errors are wrapped in RunnerResult.error (not raised)."""
        from zephon._internal.runners.concurrent import RunnerResult
        from zephon.ops.base import BaseOp
        from zephon.ops.traits import OpTraits

        class FailOp(BaseOp):
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
        from zephon._internal.runners.concurrent import RunnerResult
        from zephon.ops.base import BaseOp
        from zephon.ops.traits import OpTraits

        class BrokenProcessManyOp(BaseOp):
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

    def test_init_creates_actors(self, group) -> None:
        """After init, the requested number of actor handles exist."""
        assert len(group.actors) == 2

    def test_direct_actor_dispatch_and_get(self, group) -> None:
        """Dispatch directly to an actor and get the result."""
        from zephon._internal.runners.concurrent import RunnerResult

        records = _mk_records([10, 20, 30])
        ref = group.actors[0].process.remote(records, 0)
        result = ray.get(ref)

        assert isinstance(result, RunnerResult)
        assert _extract_values(result.payload) == [10, 20, 30]
        assert result.seq == 0
        assert result.error is None

    def test_shutdown_kills_actors(self) -> None:
        """shutdown() kills all actors and clears the list."""
        g = _make_group(num_actors=1)
        assert len(g.actors) == 1
        g.shutdown()
        assert len(g.actors) == 0

    def test_init_rejects_mismatched_planned_length(self) -> None:
        """preferred_node_ids must have length num_actors when provided."""
        op = DelayById(max_delay_ms=0.0)
        node = Node(name="mismatch", op=op, parallelism=2)
        group = _RayActorGroup(
            node=node,
            op_index=0,
            num_actors=2,
            stage_index=0,
            stage_name="test",
            collect_stats=False,
            ctx_services={},
        )
        with pytest.raises(ValueError, match=r"preferred_node_ids has 1 entries"):
            group.init(num_cpus_per_actor=0.1, preferred_node_ids=["only-one"])
        # No actors were spawned, so nothing to shut down.
        assert group.actors == []

    def test_hard_affinity_bogus_node_id_raises(self, monkeypatch) -> None:
        """Hard affinity with an unschedulable node id fails fast, not silently hang.

        Ray surfaces this two ways depending on timing — either a
        synchronous ActorUnschedulableError (caught via ``ray.get``) or
        indefinite PENDING (caught via the ``ray.wait`` deadline). We
        accept either error message since both indicate the probe did
        its job.
        """
        # Shrink the timeout so the test fails fast if Ray takes the pending path.
        monkeypatch.setattr(_RayActorGroup, "_PLACEMENT_TIMEOUT_S", 1.0)

        op = DelayById(max_delay_ms=0.0)
        node = Node(name="bogus_pin", op=op, parallelism=1)
        group = _RayActorGroup(
            node=node,
            op_index=0,
            num_actors=1,
            stage_index=0,
            stage_name="test",
            collect_stats=False,
            ctx_services={},
        )
        try:
            # Ray node IDs are 56-char hex strings — use one that's the right
            # shape but doesn't correspond to any node in the cluster.
            bogus_node_id = "0" * 56
            with pytest.raises(
                RuntimeError, match=r"Actor placement (timed out|failed)"
            ):
                group.init(
                    num_cpus_per_actor=0.1,
                    preferred_node_ids=[bogus_node_id],
                    hard_node_affinity=True,
                )
            assert group.actors == []
        finally:
            group.shutdown()
