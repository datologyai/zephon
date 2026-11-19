"""Unit tests for Pipeline plan cache invalidation on graph mutation (PLAT-1443)."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon import Pipeline as PublicPipeline
from zephon.types import SampleBatch, SampleRecord


def test_add_node_mutation_invalidates_plan() -> None:
    """Path A: batch() after explain() yields SampleBatch (not SampleRecord).

    Reproduces PLAT-1443. Without invalidation, cached plan has no batch node.
    """
    ds = make_inmem_dataset("tiny", [{"text": f"x{i}"} for i in range(8)])
    ws = FakeIndexableWorkSource(ds, chunk_size=4)

    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .options(
            deterministic=True,
            prefetch_batches=0,
            default_stage_prefetch=0,
            max_workers=1,
        )
    )
    pipe.explain()  # Trigger plan cache
    pipe.batch(microbatch_size=4, drop_last=False)  # Mutate graph after cache

    items = list(pipe)
    assert len(items) >= 1
    assert all(isinstance(item, SampleBatch) for item in items)
    assert not any(isinstance(item, SampleRecord) for item in items)


def test_prefetch_mutation_invalidates_plan() -> None:
    """Path B: prefetch() after explain() rebuilds plan with prefetch node."""
    ds = make_inmem_dataset("tiny", [{"text": "a"}, {"text": "b"}])
    ws = FakeIndexableWorkSource(ds, chunk_size=2)

    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .options(
            deterministic=True,
            prefetch_batches=0,
            default_stage_prefetch=0,
            max_workers=1,
        )
    )
    pipe.explain()  # Trigger plan cache
    pipe.prefetch(buffer_size=64, parallelism=1)  # Mutate graph after cache

    exp = pipe.explain()
    assert "prefetch" in exp


def test_invalidate_plan_closes_engine() -> None:
    """Plan invalidation must call Engine.close() before clearing _engine to avoid leaking threads/resources."""
    ds = make_inmem_dataset("tiny", [{"text": "a"}, {"text": "b"}])
    ws = FakeIndexableWorkSource(ds, chunk_size=2)

    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .options(
            deterministic=True,
            prefetch_batches=0,
            default_stage_prefetch=0,
            max_workers=1,
        )
    )
    mock_engine = MagicMock()
    pipe._engine = mock_engine

    pipe.batch(microbatch_size=2, drop_last=False)  # Mutate graph -> _invalidate_plan()

    mock_engine.close.assert_called_once()
    assert pipe._engine is None


def test_options_does_not_invalidate_plan() -> None:
    """Non-mutation: options() must not clear plan cache (graph unchanged)."""
    ds = make_inmem_dataset("tiny", [{"text": "a"}, {"text": "b"}])
    ws = FakeIndexableWorkSource(ds, chunk_size=2)

    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .batch(microbatch_size=2, drop_last=False)
        .options(
            deterministic=True,
            prefetch_batches=0,
            default_stage_prefetch=0,
            max_workers=1,
        )
    )
    pipe.explain()  # Trigger plan cache
    pipe.options(deterministic=True)  # Non-mutation: options only

    assert pipe._plan is not None


def test_all_graph_mutating_methods_invalidate_plan() -> None:
    """Structural guard: every graph-mutating Pipeline public method must invalidate the plan.

    Accepted patterns (either is sufficient):
    1. The method carries a ``_mutates_graph`` sentinel attribute set by the
       ``@_mutates_graph`` decorator.
    2. The method's source contains the string ``_invalidate_plan`` (explicit call).

    Add to ``SAFE_METHODS`` only when a method provably does NOT mutate the graph or
    plan cache (e.g. read-only queries, pure configuration, or transparent delegation
    to another method that already invalidates).
    """
    import inspect

    # Methods that intentionally do NOT need to invalidate the plan themselves.
    SAFE_METHODS: frozenset[str] = frozenset(
        {
            # Pure reads / queries
            "explain",
            "compile",
            "checkpoint",
            "restore",
            "metrics_snapshot",
            "fetch_timing_snapshot",
            "prefetch_timing_snapshot",
            "validate",
            "preflight_tokenizers",
            "inflight_summary",
            "mtp_queue_stats",
            # Runtime cleanup leaves the graph and compiled plan intact.
            "close",
            # Adapters — return a wrapper object, do not mutate the graph
            "to_torch_dataset",
            "to_indexable_torch_dataset",
            # Configuration-only: mutates _options/_runtime_spec, not the graph
            "options",
            "enable_observability",
            # Transparent delegation: calls fetch_parallelism() which invalidates
            "fetch",
        }
    )

    violations: list[str] = []
    for name, method in inspect.getmembers(
        PublicPipeline, predicate=inspect.isfunction
    ):
        if name.startswith("_"):
            continue
        if name in SAFE_METHODS:
            continue
        has_sentinel = bool(getattr(method, "_mutates_graph", False))
        has_explicit_call = "_invalidate_plan" in inspect.getsource(method)
        if not has_sentinel and not has_explicit_call:
            violations.append(name)

    assert not violations, (
        "The following Pipeline public methods are missing plan cache invalidation:\n"
        + "\n".join(f"  - {name}()" for name in sorted(violations))
        + "\n\nDecorate with @_mutates_graph or call self._invalidate_plan() at the top."
    )


_SERIAL_OPTS: dict[str, Any] = {
    "deterministic": True,
    "prefetch_batches": 0,
    "default_stage_prefetch": 0,
    "max_workers": 1,
}


def test_planning_leaves_graph_unchanged() -> None:
    """Replay filters exist only in the plan, never in the user's graph.

    The plan cache is invalidated on every graph mutation, but edits the
    planner wrote into the graph would survive that and pile up.
    """
    ds = make_inmem_dataset("tiny", [{"text": "a"}, {"text": "b"}])
    ws = FakeIndexableWorkSource(ds, chunk_size=2)

    pipe = PublicPipeline(ws).decode_text().options(**_SERIAL_OPTS)
    pipe.explain()  # Plans with a tail replay filter
    assert [n.name for n in pipe._graph.nodes] == ["fetch", "decode_text"]

    pipe.batch(microbatch_size=2, drop_last=False)
    batch_node = pipe._graph.nodes[-1]
    pipe.explain()  # Plans with a replay filter before batch

    assert [n.name for n in pipe._graph.nodes] == ["fetch", "decode_text", "batch"]
    assert [n.name for n in batch_node.inputs] == ["decode_text"]
    assert pipe._plan is not None
    plan_names = [n.name for stage in pipe._plan.stages for n in stage.nodes]
    assert [n for n in plan_names if n.endswith("_replay_filter")] == [
        "batch_replay_filter"
    ]


_RESTORE_ROWS = [{"text": f"s{i}"} for i in range(8)]


def _build_for_restore(*, last_op: str, plan_before_last_op: bool) -> PublicPipeline:
    ws = FakeIndexableWorkSource(
        make_inmem_dataset("rows", _RESTORE_ROWS), chunk_size=3
    )
    pipe = PublicPipeline(ws).decode_text().options(**_SERIAL_OPTS)
    if plan_before_last_op:
        pipe.explain()
    if last_op == "batch":
        return pipe.batch(microbatch_size=2, drop_last=False)
    return pipe.map_transform(lambda payload: payload)


def _texts(items: Iterable[Any]) -> list[str]:
    texts: list[str] = []
    for item in items:
        records = item.records if isinstance(item, SampleBatch) else [item]
        texts.extend(record.payload["text"] for record in records)
    return texts


@pytest.mark.parametrize("last_op", ["map_transform", "batch"])
@pytest.mark.parametrize("plan_before_last_op", [False, True])
def test_restore_after_planning_mid_build_resumes_remaining_rows(
    last_op: str, plan_before_last_op: bool
) -> None:
    """Planning before the last op is added must not change what a restore yields.

    Regression: the planner used to write its tail replay filter into the
    graph, so the next op landed after it and a second filter was planned.
    On restore the first filter turned the cursor record into a tombstone,
    the second never saw it, and every remaining record was dropped.
    """
    first = _build_for_restore(last_op=last_op, plan_before_last_op=plan_before_last_op)
    it = iter(first)
    head = _texts(itertools.islice(it, 3))
    ckpt = first.checkpoint()
    it.close()

    resumed = _build_for_restore(
        last_op=last_op, plan_before_last_op=plan_before_last_op
    )
    resumed.restore(ckpt)
    rest = _texts(resumed)

    assert head
    assert head + rest == [row["text"] for row in _RESTORE_ROWS]
