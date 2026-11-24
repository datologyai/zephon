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


def _route(payload: Any) -> str:
    return "chat" if "messages" in payload else "text"


def _identity(payload: Any) -> Any:
    return payload


def _pipeline() -> Pipeline:
    ds = make_inmem_dataset("tiny", [{"text": "a"}, {"messages": []}])
    return Pipeline(FakeIndexableWorkSource(ds, chunk_size=2)).options(
        deterministic=True, prefetch_batches=0, default_stage_prefetch=0
    )


def _names(nodes: list[Any]) -> list[str]:
    return [node.name for node in nodes]


def test_branch_wires_one_op_per_branch_after_upstream() -> None:
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

    fetch, upstream, text, chat, merge = pipe._graph.nodes
    assert _names([fetch, upstream, text, chat, merge]) == [
        "fetch",
        "map_transform",
        "decode_text",
        "expand_chat",
        "materialize",
    ]
    assert upstream.router is not None
    assert upstream.router.branches == ("text", "chat")
    assert upstream.router.default is None
    assert (text.branch, text.inputs) == ("text", [upstream])
    assert (chat.branch, chat.inputs) == ("chat", [upstream])
    assert merge.branch is None
    assert merge.inputs == [text, chat]
    # decode_text keeps its forced placement inside a branch.
    assert text.placement == "local"


def test_passthrough_branch_is_an_identity_branch_op() -> None:
    pipe = (
        _pipeline()
        .map_transform(_identity)
        .branch(
            _route,
            {"text": lambda b: b, "chat": lambda b: b.expand_chat()},
            default="text",
        )
        .materialize()
    )

    _, upstream, text, chat, merge = pipe._graph.nodes
    assert upstream.router.default == "text"
    assert (text.name, text.branch, text.inputs) == ("passthrough", "text", [upstream])
    assert text.op.process_many(["a", "b"]) == ["a", "b"]
    # The merge reads exactly the branch op nodes, never U.
    assert merge.inputs == [text, chat]


def test_every_branch_may_be_a_passthrough() -> None:
    pipe = _pipeline().branch(_route, {"text": lambda b: b, "chat": lambda b: b})

    _, text, chat = pipe._graph.nodes
    assert _names([text, chat]) == ["passthrough", "passthrough"]
    assert (text.branch, chat.branch) == ("text", "chat")
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

    with pytest.raises(error, match=match):
        pipe.branch(**kwargs)

    # A rejected call leaves the graph as it was.
    assert pipe._graph.nodes == nodes_before
    assert all(node.router is None for node in pipe._graph.nodes)


def test_branch_cannot_directly_follow_another_branch() -> None:
    pipe = _pipeline().branch(
        _route, {"text": lambda b: b.decode_text(), "chat": lambda b: b.expand_chat()}
    )

    with pytest.raises(ValueError, match="cannot follow another branch"):
        pipe.branch(_route, {"text": lambda b: b.decode_text()})

    # An op in between merges the first branches, so a second split works.
    pipe.map_transform(_identity).branch(_route, {"text": lambda b: b.decode_text()})
    assert sum(node.router is not None for node in pipe._graph.nodes) == 2


def test_branch_builder_offers_only_record_level_methods() -> None:
    seen: dict[str, Any] = {}

    def capture(builder: Any) -> Any:
        seen["builder"] = builder
        return builder.decode_text()

    _pipeline().branch(_route, {"text": capture})

    builder = seen["builder"]
    for name in ("map_transform", "add_op", "decode_text", "tokenize", "tokenize_chat"):
        assert callable(getattr(builder, name))
    for name in ("branch", "batch", "shuffle", "pack_flat", "ensure_mixture"):
        assert not hasattr(builder, name)


def test_explain_marks_router_and_branch_ops() -> None:
    pipe = _pipeline().branch(
        _route, {"text": lambda b: b.decode_text(), "chat": lambda b: b.expand_chat()}
    )

    plan = pipe.explain().splitlines()[0]
    assert "'fetch@p4 -> branch(text, chat)'" in plan
    assert "'[text] decode_text@p" in plan
    assert "'[chat] expand_chat@p" in plan


def test_tail_replay_filter_reads_every_branch_output() -> None:
    pipe = _pipeline().branch(
        _route, {"text": lambda b: b, "chat": lambda b: b.expand_chat()}
    )
    pipe.explain()

    assert pipe._plan is not None
    replay_filter = pipe._plan.stages[-1].nodes[-1]
    assert replay_filter.name == "expand_chat_replay_filter"
    assert _names(replay_filter.inputs) == ["passthrough", "expand_chat"]


def test_replay_filter_before_batch_reads_every_branch_output() -> None:
    pipe = _pipeline().branch(
        _route, {"text": lambda b: b.decode_text(), "chat": lambda b: b.expand_chat()}
    )
    pipe.batch(2)
    pipe.explain()

    assert pipe._plan is not None
    batch_stage = pipe._plan.stages[-1]
    assert _names(batch_stage.nodes) == ["batch_replay_filter", "batch"]
    assert _names(batch_stage.nodes[0].inputs) == ["decode_text", "expand_chat"]


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
    # Same node names and ops in the same order; only the edges differ.
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
