# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Builder and planner behavior of ``Pipeline.branch()``.

No runner executes branches yet, so these tests stop at the graph and plan.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon import Pipeline
from zephon.ops import CountingAccumulator
from zephon.pipeline import Branch


def _route(payload: Any) -> str:
    return "chat" if "messages" in payload else "text"


def _identity(payload: Any) -> Any:
    return payload


def _pipeline() -> Pipeline:
    ds = make_inmem_dataset("tiny", [{"text": "a"}, {"messages": []}])
    return Pipeline(FakeIndexableWorkSource(ds, chunk_size=2)).options(
        deterministic=True, prefetch_batches=0, default_stage_prefetch=0
    )


def _names(vertices: list[Any]) -> list[str]:
    return [vertex.name for vertex in vertices]


def test_branch_adds_split_branch_ops_and_merge() -> None:
    pipe = (
        _pipeline()
        .map_transform(_identity)
        .branch(
            _route,
            {
                "text": lambda b: b.decode_text(),
                "chat": lambda b: b.expand_chat(),
            },
        )
        .materialize()
    )

    fetch, upstream, text, chat, after = pipe._graph.nodes
    assert _names([fetch, upstream, text, chat, after]) == [
        "fetch",
        "map_transform",
        "decode_text",
        "expand_chat",
        "materialize",
    ]
    (split,) = pipe._graph.splits
    (merge,) = pipe._graph.merges
    assert split.name == "map_transform_split"
    assert split.input is upstream
    assert split.router.branches == ("text", "chat")
    assert split.router.default is None
    assert text.inputs[0] is split and chat.inputs[0] is split
    assert merge.name == "map_transform_merge"
    assert merge.split is split
    assert list(merge.branches) == ["text", "chat"]
    assert merge.branches["text"] is text and merge.branches["chat"] is chat
    assert len(after.inputs) == 1 and after.inputs[0] is merge
    # decode_text keeps its forced placement inside a branch.
    assert text.placement == "local"


def test_passthrough_branch_is_an_identity_branch_op() -> None:
    pipe = _pipeline().branch(
        _route,
        {"text": lambda b: b, "chat": lambda b: b.expand_chat()},
        default="text",
    )

    _, text, _ = pipe._graph.nodes
    (split,) = pipe._graph.splits
    (merge,) = pipe._graph.merges
    assert split.router.default == "text"
    assert text.name == "passthrough" and text.inputs[0] is split
    assert text.op.process_many(["a", "b"]) == ["a", "b"]
    assert merge.branches["text"] is text


def test_every_branch_may_be_a_passthrough() -> None:
    pipe = _pipeline().branch(_route, {"text": lambda b: b, "chat": lambda b: b})

    _, text, chat = pipe._graph.nodes
    assert _names([text, chat]) == ["passthrough", "passthrough"]
    assert "'[chat] passthrough@p1'" in pipe.explain()


def test_branch_keeps_op_arguments() -> None:
    pipe = _pipeline().branch(
        _route,
        {
            "text": lambda b: b.map_transform(_identity, parallelism=3),
            "chat": lambda b: b.add_op(
                "custom",
                process_many=lambda items: items,
                preserves_cursor_order=True,
                parallelism=2,
            ),
        },
    )

    _, text, chat = pipe._graph.nodes
    assert (text.name, text.parallelism) == ("map_transform", 3)
    assert (chat.name, chat.parallelism) == ("custom", 2)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"by": _route, "branches": {}}, ValueError, "at least one branch"),
        (
            {"by": _route, "branches": {"": lambda b: b.expand_chat()}},
            ValueError,
            "non-empty strings",
        ),
        (
            {
                "by": _route,
                "branches": {"chat": lambda b: b.expand_chat()},
                "default": "text",
            },
            ValueError,
            "default='text' names no branch",
        ),
        (
            {"by": "chat", "branches": {"chat": lambda b: b.expand_chat()}},
            TypeError,
            "callable `by`",
        ),
        (
            {"by": _route, "branches": {"chat": lambda b: None}},
            TypeError,
            "must return the Branch it received",
        ),
        (
            {
                "by": _route,
                "branches": {
                    "chat": lambda b: b.expand_chat().map_transform(_identity)
                },
            },
            ValueError,
            "at most one op",
        ),
    ],
)
def test_branch_rejects_invalid_arguments(
    kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    pipe = _pipeline().map_transform(_identity)
    nodes_before = list(pipe._graph.nodes)
    tail_before = pipe._tail

    with pytest.raises(error, match=match):
        pipe.branch(**kwargs)

    # A rejected call leaves the graph as it was.
    assert pipe._graph.nodes == nodes_before
    assert pipe._graph.splits == [] and pipe._graph.merges == []
    assert pipe._tail is tail_before


def test_branch_cannot_directly_follow_another_branch() -> None:
    pipe = _pipeline().branch(
        _route, {"text": lambda b: b.decode_text(), "chat": lambda b: b.expand_chat()}
    )

    with pytest.raises(ValueError, match="cannot follow another branch"):
        pipe.branch(_route, {"text": lambda b: b.decode_text()})

    # An op in between reads the first merge, so a second split works.
    pipe.map_transform(_identity).branch(_route, {"text": lambda b: b.decode_text()})
    assert len(pipe._graph.splits) == len(pipe._graph.merges) == 2
    pipe.explain()


def test_branch_builder_is_public_and_offers_only_record_level_methods() -> None:
    seen: dict[str, Any] = {}

    def capture(builder: Branch) -> Branch:
        seen["builder"] = builder
        return builder.decode_text()

    _pipeline().branch(_route, {"text": capture})

    builder = seen["builder"]
    assert isinstance(builder, Branch)
    assert type(builder).__module__ == "zephon.pipeline"
    for name in ("map_transform", "add_op", "decode_text", "tokenize", "tokenize_chat"):
        assert callable(getattr(builder, name))
    for name in ("branch", "batch", "shuffle", "pack_flat", "ensure_mixture"):
        assert not hasattr(builder, name)


def test_explain_shows_split_branch_ops_and_merge_in_order() -> None:
    pipe = _pipeline().branch(
        _route, {"text": lambda b: b.decode_text(), "chat": lambda b: b.expand_chat()}
    )

    plan = pipe.explain().splitlines()[0]
    marks = [
        "'fetch@p4'",
        "'fetch_split(text, chat)'",
        "'[text] decode_text@p",
        "'[chat] expand_chat@p",
        "'fetch_merge'",
    ]
    positions = [plan.index(mark) for mark in marks]
    assert positions == sorted(positions)


def test_split_and_merge_are_placed_with_their_branch_ops() -> None:
    pipe = _pipeline().branch(
        _route,
        {
            "text": lambda b: b.map_transform(_identity, placement="remote"),
            "chat": lambda b: b.map_transform(_identity, placement="remote"),
        },
    )
    pipe.explain()

    assert pipe._plan is not None
    upstream_stage, branch_stage = pipe._plan.stages
    assert _names(upstream_stage.nodes) == ["fetch"]
    assert upstream_stage.splits == [] and upstream_stage.merges == []
    assert branch_stage.splits == pipe._graph.splits
    assert branch_stage.merges == pipe._graph.merges


def test_tail_replay_filter_reads_the_merge() -> None:
    pipe = _pipeline().branch(
        _route, {"text": lambda b: b, "chat": lambda b: b.expand_chat()}
    )
    pipe.explain()

    assert pipe._plan is not None
    replay_filter = pipe._plan.stages[-1].nodes[-1]
    assert replay_filter.name == "fetch_merge_replay_filter"
    assert replay_filter.inputs == pipe._graph.merges


def test_replay_filter_before_batch_reads_the_merge() -> None:
    pipe = _pipeline().branch(
        _route, {"text": lambda b: b.decode_text(), "chat": lambda b: b.expand_chat()}
    )
    pipe.batch(2)
    pipe.explain()

    assert pipe._plan is not None
    batch_stage = pipe._plan.stages[-1]
    assert _names(batch_stage.nodes) == ["batch_replay_filter", "batch"]
    assert batch_stage.nodes[0].inputs == pipe._graph.merges


def test_ops_after_batch_read_the_plans_batch() -> None:
    pipe = _pipeline().branch(
        _route, {"text": lambda b: b.decode_text(), "chat": lambda b: b.expand_chat()}
    )
    pipe.batch(2).map_batch(_identity)
    pipe.explain()

    assert pipe._plan is not None
    plan_nodes = [node for stage in pipe._plan.stages for node in stage.nodes]
    batch, map_batch = plan_nodes[-2:]
    assert _names([batch, map_batch]) == ["batch", "map_batch"]
    # The planner copies Batch to put its replay filter in front, and the op
    # after it reads that copy, so every edge in the plan stays in the plan.
    assert map_batch.inputs[0] is batch
    # The graph keeps its own edges.
    graph_batch, graph_map_batch = pipe._graph.nodes[-2:]
    assert graph_map_batch.inputs[0] is graph_batch
    assert graph_batch.inputs == pipe._graph.merges


@pytest.mark.parametrize(
    "before",
    [
        pytest.param(lambda pipe: pipe.batch(2), id="right-after-batch"),
        pytest.param(
            lambda pipe: pipe.batch(2).map_batch(_identity), id="later-after-batch"
        ),
    ],
)
def test_planner_rejects_branch_after_batch(before: Callable[[Any], Any]) -> None:
    pipe = before(_pipeline()).branch(
        _route, {"text": lambda b: b, "chat": lambda b: b.decode_text()}
    )

    with pytest.raises(ValueError, match=r"branch\(\) cannot come after batch\(\)"):
        pipe.explain()


class _HoldingAccumulator(CountingAccumulator[Any]):
    """A subclass may hold records across batches, so branches reject it."""


@pytest.mark.parametrize(
    ("op_kwargs", "match"),
    [
        ({"preserves_cursor_order": False}, "does not preserve cursor order"),
        (
            {"preserves_cursor_order": True, "batch_shape_sensitive": True},
            "depends on batch shape",
        ),
        (
            {"preserves_cursor_order": True, "requires_serial_state": True},
            "keeps serial state",
        ),
        (
            {
                "preserves_cursor_order": True,
                "accumulator": lambda: _HoldingAccumulator(max_batch=4),
            },
            "_HoldingAccumulator may hold records",
        ),
    ],
)
def test_planner_rejects_branch_ops_that_are_not_stateless(
    op_kwargs: dict[str, Any], match: str
) -> None:
    add: Callable[[Any], Any] = lambda b: b.add_op(  # noqa: E731
        "custom", process_many=lambda items: items, **op_kwargs
    )
    pipe = _pipeline().branch(_route, {"text": add, "chat": lambda b: b})

    with pytest.raises(ValueError, match=rf"op 'custom' in branch 'text'.*{match}"):
        pipe.explain()


@pytest.mark.parametrize("padding", [True, "longest"])
def test_planner_rejects_tokenize_that_pads_to_the_batch(padding: bool | str) -> None:
    pipe = _pipeline().branch(
        _route,
        {
            "text": lambda b: b.tokenize(
                tokenizer_id="__fallback__", field="text", padding=padding
            ),
            "chat": lambda b: b,
        },
    )

    with pytest.raises(ValueError, match="op 'tokenize'.*depends on batch shape"):
        pipe.explain()


def test_planner_accepts_tokenize_that_pads_to_a_fixed_length() -> None:
    pipe = _pipeline().branch(
        _route,
        {
            "text": lambda b: b.tokenize(
                tokenizer_id="__fallback__",
                field="text",
                padding="max_length",
                max_length=8,
            ),
            "chat": lambda b: b,
        },
    )

    assert "[text] tokenize@p" in pipe.explain()


def test_planner_accepts_counting_and_passthrough_accumulators() -> None:
    pipe = _pipeline().branch(
        _route,
        {
            "text": lambda b: b.add_op(
                "counting",
                process_many=lambda items: items,
                preserves_cursor_order=True,
                accumulator=lambda: CountingAccumulator(max_batch=4),
            ),
            "chat": lambda b: b.add_op(
                "passthrough",
                process_many=lambda items: items,
                preserves_cursor_order=True,
            ),
        },
    )

    assert "[text] counting" in pipe.explain()


def test_planner_requires_branch_ops_to_share_a_stage() -> None:
    pipe = _pipeline().branch(
        _route,
        {
            "text": lambda b: b.map_transform(_identity, placement="local"),
            "chat": lambda b: b.map_transform(_identity, placement="remote"),
        },
    )

    with pytest.raises(ValueError, match="landed in different stages"):
        pipe.explain()


def test_branched_plan_id_reflects_edges() -> None:
    chain = _pipeline().map_transform(_identity).map_transform(_identity)
    branched = _pipeline().branch(_route, {"a": lambda b: b.map_transform(_identity)})
    branched.map_transform(_identity)
    chain.explain()
    branched.explain()

    assert chain._plan is not None and branched._plan is not None
    # Same ops in the same order; only the split, merge and edges differ.
    assert [_names(stage.nodes) for stage in chain._plan.stages] == [
        _names(stage.nodes) for stage in branched._plan.stages
    ]
    assert chain._plan.plan_id != branched._plan.plan_id


def test_iterating_a_branched_pipeline_fails_until_runners_support_it() -> None:
    pipe = _pipeline().branch(
        _route, {"text": lambda b: b.decode_text(), "chat": lambda b: b.expand_chat()}
    )

    with pytest.raises(NotImplementedError, match="cannot run yet"):
        next(iter(pipe))
