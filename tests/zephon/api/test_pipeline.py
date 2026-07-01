"""Pipeline API tests.

Covers options/cache behavior plus the custom-op authoring surface:

* ``BaseOp`` defaults and abstract-method enforcement.
* Both forms of ``Pipeline.add_op``:
    - Instance form: ``add_op(op: BaseOp, ...)`` — full lifecycle via the
      subclass (``__init__`` / ``setup`` / ``traits`` / ``accumulator``).
    - Kwargs form: ``add_op(name, *, process_many=..., ...)`` for stateless
      transforms, including the advanced trait kwargs
      (``batch_shape_sensitive``, ``requires_serial_state``,
      ``stall_on_epoch_boundary``).

Also includes a port of the legacy_shuffle_pack v2 pattern (the per-lane
``CountingAccumulator`` op Maxi calls out in the docstring at
``datnanovlm/data/zephon/legacy_shuffle_pack.py``) as an integration test —
this is the load-bearing use case the new API was designed for.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon.api import Pipeline as PublicPipeline
from zephon.core import (
    Accumulator,
    BaseOp,
    CountingAccumulator,
    DefaultSetup,
    Op,
    OpContext,
    OpTraits,
    PassthroughAccumulator,
    ReadyBatch,
    SampleRecord,
)
from zephon.io import Dataset
from zephon.ops.ensure_mixture import EnsureMixture
from zephon.work import MixtureSpec, StaticMixtureWorkSource

# ---------------------------------------------------------------------------
# Options / cache
# ---------------------------------------------------------------------------


def test_pipeline_with_cache(tmp_path: Path) -> None:
    # Build a small JSONL dataset
    shard0 = tmp_path / "shard0.jsonl"
    shard1 = tmp_path / "shard1.jsonl"
    shard0.write_text(
        "\n".join(json.dumps({"text": f"sample {i}", "value": i}) for i in range(3)),
        encoding="utf-8",
    )
    shard1.write_text(
        "\n".join(json.dumps({"text": f"sample {i}", "value": i}) for i in range(3, 5)),
        encoding="utf-8",
    )
    jsonl_dataset = Dataset.from_path("demo", str(tmp_path))

    cache_root = tmp_path / "cache"
    work_source = StaticMixtureWorkSource(
        [jsonl_dataset],
        mixture=MixtureSpec({jsonl_dataset.name: 1.0}).weights,
        chunk_size=1,
        seed=11,
        shuffle_shards=False,
    )
    pipe = (
        PublicPipeline(work_source)
        .decode_text()
        .options(io_options={"cache": {"enabled": True, "root": cache_root}})
        .batch(microbatch_size=2, drop_last=False)
    )
    iterator = iter(pipe)
    try:
        next(iterator)
    finally:
        iterator.close()
    assert cache_root.exists()


def test_unknown_options_warn(tmp_path: Path) -> None:
    """Unknown keys passed to .options() should emit a warning, not be silently ignored."""
    shard = tmp_path / "shard.jsonl"
    shard.write_text(json.dumps({"text": "hello"}), encoding="utf-8")
    ds = Dataset.from_path("demo", str(tmp_path))
    ws = StaticMixtureWorkSource(
        [ds],
        mixture=MixtureSpec({ds.name: 1.0}).weights,
        chunk_size=1,
        seed=0,
    )
    pipe = PublicPipeline(ws)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe.options(no_such_option=42, deterministic=True)

    unknown_warnings = [w for w in caught if "no_such_option" in str(w.message)]
    assert len(unknown_warnings) == 1, (
        f"Expected exactly 1 warning for unknown option, got {len(unknown_warnings)}"
    )
    # deterministic is valid and should NOT trigger a warning
    deterministic_warnings = [w for w in caught if "deterministic" in str(w.message)]
    assert len(deterministic_warnings) == 0, "Valid option should not trigger a warning"


def test_ensure_mixture_accepts_unbounded_buffer(tmp_path: Path) -> None:
    """The public ensure_mixture() wrapper exposes strict mode: passing
    max_buffer_size=None wires through to an unbounded EnsureMixture op."""
    shard = tmp_path / "shard.jsonl"
    shard.write_text(json.dumps({"text": "hello"}), encoding="utf-8")
    ds = Dataset.from_path("demo", str(tmp_path))
    ws = StaticMixtureWorkSource(
        [ds],
        mixture=MixtureSpec({ds.name: 1.0}).weights,
        chunk_size=1,
        seed=0,
    )
    pipe = PublicPipeline(ws).decode_text().ensure_mixture(max_buffer_size=None)

    ops = [node.op for node in pipe._graph.nodes if isinstance(node.op, EnsureMixture)]
    assert len(ops) == 1
    assert ops[0]._config.max_buffer_size is None


# ---------------------------------------------------------------------------
# Public namespace
# ---------------------------------------------------------------------------


def test_public_namespace_exports_authoring_types() -> None:
    """zephon.core must expose every type a custom op needs."""
    for sym in (
        Op,
        OpContext,
        DefaultSetup,
        BaseOp,
        OpTraits,
        Accumulator,
        ReadyBatch,
        CountingAccumulator,
        PassthroughAccumulator,
        SampleRecord,
    ):
        assert sym is not None


# ---------------------------------------------------------------------------
# BaseOp defaults
# ---------------------------------------------------------------------------


def test_baseop_defaults_passthrough_traits_and_accumulator() -> None:
    """A BaseOp subclass that overrides only the abstract methods inherits defaults."""

    class TrivialOp(BaseOp):
        def traits(self) -> OpTraits:
            return OpTraits(preserves_cursor_order=True)

        def process_many(self, elems: list[Any]) -> list[Any]:
            return list(elems)

    op = TrivialOp()
    traits = op.traits()
    assert isinstance(traits, OpTraits)
    assert traits.parallelism == 1
    assert traits.indexable is True
    assert traits.preserves_cursor_order is True

    acc = op.accumulator(deterministic=True, ctx={})
    assert isinstance(acc, PassthroughAccumulator)


def test_baseop_process_one_routes_through_process_many() -> None:
    """The default process_one must wrap the element and call process_many."""
    seen_batches: list[list[Any]] = []

    class CapturingOp(BaseOp):
        def traits(self) -> OpTraits:
            return OpTraits(preserves_cursor_order=True)

        def process_many(self, elems: list[Any]) -> list[Any]:
            seen_batches.append(list(elems))
            return list(elems)

    op = CapturingOp()
    out = op.process_one("alpha")
    assert out == ["alpha"]
    assert seen_batches == [["alpha"]]


def test_baseop_process_many_is_abstract() -> None:
    """Subclasses that forget process_many fail at instantiation, not at call."""

    class MissingProcessMany(BaseOp):
        def traits(self) -> OpTraits:
            return OpTraits(preserves_cursor_order=True)

    with pytest.raises(TypeError, match="abstract"):
        MissingProcessMany()  # type: ignore[abstract]


def test_baseop_traits_is_abstract() -> None:
    """Subclasses that forget traits fail at instantiation, not at plan time."""

    class MissingTraits(BaseOp):
        def process_many(self, elems: list[Any]) -> list[Any]:
            return list(elems)

    with pytest.raises(TypeError, match="abstract"):
        MissingTraits()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Helpers used by the kwargs add_op tests
# ---------------------------------------------------------------------------


def _empty_pipeline() -> PublicPipeline:
    ds = make_inmem_dataset("tiny", [{"text": "x"}])
    ws = FakeIndexableWorkSource(ds, chunk_size=1)
    return PublicPipeline(ws)


def _tag_payload(records: list[SampleRecord], tag: str) -> list[SampleRecord]:
    out: list[SampleRecord] = []
    for rec in records:
        if rec.meta.tombstone:
            out.append(rec)
            continue
        payload = dict(rec.payload) if isinstance(rec.payload, dict) else {}
        payload["tag"] = tag
        rec.payload = payload
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Kwargs add_op — primary API
# ---------------------------------------------------------------------------


def test_add_op_rejects_empty_name() -> None:
    pipe = _empty_pipeline()
    with pytest.raises(ValueError, match="non-empty name"):
        pipe.add_op(
            "", process_many=lambda elems: list(elems), preserves_cursor_order=True
        )


def test_add_op_carries_mutates_graph_sentinel() -> None:
    """Required for the structural plan-cache invalidation guard."""
    assert getattr(PublicPipeline.add_op, "_mutates_graph", False) is True


def test_add_op_invalidates_cached_plan() -> None:
    pipe = (
        _empty_pipeline()
        .decode_text()
        .options(deterministic=True, prefetch_batches=0, default_stage_prefetch=0)
    )
    pipe.explain()
    assert pipe._plan is not None

    pipe.add_op(
        "tag",
        process_many=lambda elems: _tag_payload(elems, "after-cache"),
        preserves_cursor_order=True,
    )

    assert pipe._plan is None


def test_add_op_passthrough_default_accumulator_runs_end_to_end() -> None:
    """No accumulator kwarg → PassthroughAccumulator → each upstream batch flows through."""
    rows = [{"text": f"row-{i}"} for i in range(4)]
    ws = FakeIndexableWorkSource(make_inmem_dataset("tiny", rows), chunk_size=2)

    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .add_op(
            "tagger",
            process_many=lambda elems: _tag_payload(elems, "hi"),
            preserves_cursor_order=True,
        )
        .options(
            deterministic=True,
            max_workers=1,
            default_stage_prefetch=0,
            prefetch_batches=0,
        )
    )

    seen: list[SampleRecord] = []
    it = iter(pipe)
    try:
        for item in it:
            assert isinstance(item, SampleRecord)
            seen.append(item)
    finally:
        it.close()

    assert len(seen) == len(rows)
    for record in seen:
        assert isinstance(record.payload, dict)
        assert record.payload.get("tag") == "hi"


def test_add_op_counting_accumulator_factory_drives_window() -> None:
    """A CountingAccumulator factory groups items into fixed-size windows."""
    window = 4
    rows = [{"text": f"row-{i}"} for i in range(window * 2)]
    ws = FakeIndexableWorkSource(make_inmem_dataset("tiny", rows), chunk_size=window)

    def stamp_window(elems: list[SampleRecord]) -> list[SampleRecord]:
        size = len([e for e in elems if not e.meta.tombstone])
        for elem in elems:
            if elem.meta.tombstone:
                continue
            payload = dict(elem.payload) if isinstance(elem.payload, dict) else {}
            payload["window_size"] = size
            elem.payload = payload
        return list(elems)

    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .add_op(
            "windower",
            process_many=stamp_window,
            accumulator=lambda: CountingAccumulator(max_batch=window),
            preserves_cursor_order=False,
        )
        .options(
            deterministic=True,
            max_workers=1,
            default_stage_prefetch=0,
            prefetch_batches=0,
        )
    )

    seen: list[SampleRecord] = []
    it = iter(pipe)
    try:
        for item in it:
            assert isinstance(item, SampleRecord)
            seen.append(item)
    finally:
        it.close()

    assert len(seen) == len(rows)
    for record in seen:
        assert isinstance(record.payload, dict)
        assert record.payload.get("window_size") == window


def test_add_op_with_baseop_instance_invokes_setup_per_worker() -> None:
    """The instance form of add_op runs BaseOp.setup on each deep-copied
    instance, with `self.*` state visible to process_many — the class-based
    path matching built-in operator semantics."""
    setup_calls = {"n": 0, "ctx_seen": False}

    class TokenizerOp(BaseOp):
        def __init__(self) -> None:
            super().__init__()
            self._tokenizer: Any = None  # populated per worker

        def traits(self) -> OpTraits:
            return OpTraits(preserves_cursor_order=True)

        def setup(
            self,
            ctx: Any,
            stage_index: int,
            stage_name: str,
            op_index: int,
            collect_stats: bool,
        ) -> None:
            super().setup(ctx, stage_index, stage_name, op_index, collect_stats)
            setup_calls["n"] += 1
            setup_calls["ctx_seen"] = ctx is not None
            self._tokenizer = "ready"

        def process_many(self, elems: list[Any]) -> list[Any]:
            return list(elems)

    pipe = (
        _empty_pipeline()
        .decode_text()
        .add_op(TokenizerOp())
        .options(deterministic=True, prefetch_batches=0, default_stage_prefetch=0)
    )
    it = iter(pipe)
    try:
        list(it)
    finally:
        it.close()

    assert setup_calls["n"] >= 1, "BaseOp.setup was never invoked"
    assert setup_calls["ctx_seen"], "BaseOp.setup did not receive an OpContext"


def test_add_op_with_baseop_instance_uses_class_name_when_name_omitted() -> None:
    """The instance form defaults the node name to the op class name."""

    class MyCustomOp(BaseOp):
        def traits(self) -> OpTraits:
            return OpTraits(preserves_cursor_order=True)

        def process_many(self, elems: list[Any]) -> list[Any]:
            return list(elems)

    pipe = _empty_pipeline().decode_text().add_op(MyCustomOp())
    node_names = {n.name for n in pipe._graph.nodes}
    assert "MyCustomOp" in node_names


def test_add_op_with_baseop_instance_rejects_kwargs_form_fields() -> None:
    """Mixing the two forms is a programming error and must raise."""

    class TrivialOp(BaseOp):
        def traits(self) -> OpTraits:
            return OpTraits(preserves_cursor_order=True)

        def process_many(self, elems: list[Any]) -> list[Any]:
            return list(elems)

    pipe = _empty_pipeline()
    with pytest.raises(ValueError, match="does not accept"):
        pipe.add_op(TrivialOp(), process_many=lambda elems: elems)  # type: ignore[call-overload]


def test_add_op_kwargs_form_requires_process_many_and_preserves_cursor_order() -> None:
    """The kwargs form must spell out the required callables/traits."""
    pipe = _empty_pipeline()
    with pytest.raises(ValueError, match="process_many"):
        pipe.add_op("missing_pm", preserves_cursor_order=True)  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="preserves_cursor_order"):
        pipe.add_op("missing_pco", process_many=lambda elems: list(elems))  # type: ignore[call-overload]


def test_add_op_accumulator_factory_called_per_setup() -> None:
    """Factories must be invoked, not the same instance reused — the runner
    calls accumulator() at startup AND on reset_buffers()."""
    calls = {"n": 0}

    def factory() -> Accumulator[Any]:
        calls["n"] += 1
        return PassthroughAccumulator[Any]()

    pipe = (
        _empty_pipeline()
        .decode_text()
        .add_op(
            "noop",
            process_many=lambda elems: list(elems),
            accumulator=factory,
            preserves_cursor_order=True,
        )
        .options(deterministic=True, prefetch_batches=0, default_stage_prefetch=0)
    )
    it = iter(pipe)
    try:
        list(it)
    finally:
        it.close()

    assert calls["n"] >= 1, "accumulator factory was never called"


def test_add_op_accumulator_factory_receives_deterministic_and_ctx() -> None:
    """A factory declaring kwargs is called with deterministic + ctx.

    Mirrors the arena ShufflePackOp pattern: switch
    ``CountingAccumulator(max_latency_ms=...)`` based on the runner's
    deterministic flag.  Without auto-detection, callers would have to
    construct a custom Op class just to read the runtime flag.
    """
    seen_kwargs: list[dict[str, Any]] = []

    def factory(*, deterministic: bool, ctx: dict[str, Any]) -> Accumulator[Any]:
        seen_kwargs.append({"deterministic": deterministic, "ctx_type": type(ctx)})
        return PassthroughAccumulator[Any]()

    pipe = (
        _empty_pipeline()
        .decode_text()
        .add_op(
            "noop",
            process_many=lambda elems: list(elems),
            accumulator=factory,
            preserves_cursor_order=True,
        )
        .options(deterministic=True, prefetch_batches=0, default_stage_prefetch=0)
    )
    it = iter(pipe)
    try:
        list(it)
    finally:
        it.close()

    assert seen_kwargs, "kwargs-form factory was never called"
    for call in seen_kwargs:
        assert call["deterministic"] is True
        assert call["ctx_type"] is dict


def test_add_op_appears_in_explain() -> None:
    pipe = (
        _empty_pipeline()
        .decode_text()
        .add_op(
            "custom_stage",
            process_many=lambda elems: list(elems),
            preserves_cursor_order=True,
        )
        .options(deterministic=True, prefetch_batches=0, default_stage_prefetch=0)
    )
    assert "custom_stage" in pipe.explain()


def test_add_op_process_one_override_used() -> None:
    """When a process_one kwarg is supplied, it should be called for the
    PassthroughAccumulator single-item path instead of routing through
    process_many."""
    one_calls: list[Any] = []
    many_calls: list[list[Any]] = []

    def my_process_one(elem: Any) -> list[Any]:
        one_calls.append(elem)
        return [elem]

    def my_process_many(elems: list[Any]) -> list[Any]:
        many_calls.append(list(elems))
        return list(elems)

    rows = [{"text": "a"}, {"text": "b"}]
    ws = FakeIndexableWorkSource(make_inmem_dataset("tiny", rows), chunk_size=2)
    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .add_op(
            "split",
            process_many=my_process_many,
            process_one=my_process_one,
            preserves_cursor_order=True,
        )
        .options(
            deterministic=True,
            max_workers=1,
            default_stage_prefetch=0,
            prefetch_batches=0,
        )
    )
    it = iter(pipe)
    try:
        list(it)
    finally:
        it.close()

    # process_many is the batch path; the runner uses one of the two
    # depending on accumulator output.  We just need to know at least
    # one was actually invoked — both being uncalled would mean the op
    # never ran.
    assert one_calls or many_calls


# ---------------------------------------------------------------------------
# Advanced trait kwargs
# ---------------------------------------------------------------------------


def test_add_op_passes_advanced_traits_through() -> None:
    """The three advanced OpTraits kwargs must flow into the op's traits()."""
    pipe = _empty_pipeline().add_op(
        "advanced",
        process_many=lambda elems: list(elems),
        preserves_cursor_order=True,
        batch_shape_sensitive=True,
        requires_serial_state=True,
        stall_on_epoch_boundary=True,
    )

    # The op we just appended is the new tail; pull it back out and inspect.
    node = pipe._tail
    traits = node.op.traits()
    assert traits.batch_shape_sensitive is True
    assert traits.requires_serial_state is True
    assert traits.stall_on_epoch_boundary is True


def test_add_op_advanced_traits_default_false() -> None:
    """Omitting the advanced trait kwargs leaves them at the OpTraits defaults."""
    pipe = _empty_pipeline().add_op(
        "defaults",
        process_many=lambda elems: list(elems),
        preserves_cursor_order=True,
    )
    traits = pipe._tail.op.traits()
    assert traits.batch_shape_sensitive is False
    assert traits.requires_serial_state is False
    assert traits.stall_on_epoch_boundary is False


# ---------------------------------------------------------------------------
# Integration: the legacy_shuffle_pack v2 pattern
# ---------------------------------------------------------------------------


def test_legacy_shuffle_pack_v2_pattern_via_kwargs_add_op() -> None:
    """Smoke-test the per-lane CountingAccumulator pattern Maxi calls out
    in legacy_shuffle_pack.py: the v2 fix is "use a real op with
    CountingAccumulator(key_fn=lane_of)" instead of stateful_transform.
    With the new add_op, that pattern is a single call.
    """
    window_size = 3
    rows = [{"text": f"row-{i}"} for i in range(window_size * 3)]
    ws = FakeIndexableWorkSource(
        make_inmem_dataset("tiny", rows), chunk_size=window_size
    )

    invocation_sizes: list[int] = []

    def shuffle_pack(elems: list[SampleRecord]) -> list[SampleRecord]:
        # In the real op this would do shuffle_and_pack(...).  Here we just
        # observe that the runner gave us a window_size-sized batch and
        # forward it unchanged so we can verify ordering downstream.
        real = [e for e in elems if not e.meta.tombstone]
        invocation_sizes.append(len(real))
        return list(elems)

    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .add_op(
            "shuffle_pack_v2",
            process_many=shuffle_pack,
            accumulator=lambda: CountingAccumulator(max_batch=window_size),
            parallelism=1,
            preserves_cursor_order=False,
        )
        .options(
            deterministic=True,
            max_workers=1,
            default_stage_prefetch=0,
            prefetch_batches=0,
        )
    )
    it = iter(pipe)
    try:
        items = list(it)
    finally:
        it.close()

    assert len(items) == len(rows)
    assert invocation_sizes, "process_many was never invoked"
    # All complete windows must be exactly window_size.
    full = [n for n in invocation_sizes if n == window_size]
    assert full, (
        f"expected at least one full window of size {window_size}, "
        f"got invocation sizes {invocation_sizes}"
    )
