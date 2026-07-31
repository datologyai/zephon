"""Unit tests for Pipeline plan cache invalidation on graph mutation (PLAT-1443)."""

from __future__ import annotations

from unittest.mock import MagicMock

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
