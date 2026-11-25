# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Router, split and merge checks, stateless rule and runner check for branches."""

from __future__ import annotations

from typing import Any, Callable

import pytest

from zephon._internal import branching
from zephon._internal.branching import (
    Router,
    require_runner_support,
    stateless_violation,
)
from zephon._internal.graph import Graph, Merge, Node, Plan, Split, Stage
from zephon._internal.ops.delay import DelayById
from zephon._internal.ops.identity import Identity
from zephon._internal.ops.map_transform import MapTransform
from zephon._internal.ops.shuffle_buffer import ShuffleBuffer
from zephon._internal.ops.tokenize_text import TokenizeText
from zephon._internal.planner import Planner


def _by_field(payload: Any) -> str:
    return payload["kind"]


def test_router_routes_by_payload_and_falls_back_to_default() -> None:
    router = Router(by=_by_field, branches=("text", "chat"), default="text")

    assert router.route({"kind": "chat"}) == "chat"
    assert router.route({"kind": "text"}) == "text"
    assert router.route({"kind": "image"}) == "text"


def test_router_without_default_rejects_unknown_keys() -> None:
    router = Router(by=_by_field, branches=("text", "chat"))

    with pytest.raises(ValueError, match="returned 'image', which names no branch"):
        router.route({"kind": "image"})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"branches": ()}, "at least one branch"),
        ({"branches": ("a", "a")}, "must be unique"),
        ({"branches": ("a",), "default": "b"}, "names no branch"),
    ],
)
def test_router_validates_its_branches(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Router(by=_by_field, **kwargs)


def test_router_identity_is_stable_and_hashable() -> None:
    first = Router(by=_by_field, branches=("a", "b"), default="b")
    second = Router(by=_by_field, branches=("a", "b"), default="b")
    reordered = Router(by=_by_field, branches=("b", "a"), default="b")

    assert first.identity() == second.identity()
    assert hash(first.identity()) == hash(second.identity())
    assert first.identity() != reordered.identity()
    assert first.identity()[:2] == (__name__, "_by_field")


def test_stateless_rule_accepts_record_local_ops_and_rejects_stateful_ones() -> None:
    graph = Graph()
    mapped = graph.add("map", MapTransform(lambda p: p))
    passthrough = graph.add("passthrough", Identity())
    shuffled = graph.add("shuffle", ShuffleBuffer(buffer_size=4))

    assert stateless_violation(mapped) is None
    assert stateless_violation(passthrough) is None
    assert stateless_violation(shuffled) is not None


@pytest.mark.parametrize(
    ("kwargs", "sensitive"),
    [
        ({}, False),
        ({"padding": True}, True),
        ({"padding": "longest"}, True),
        ({"padding": "max_length", "max_length": 8}, False),
        ({"padding": True, "split_long_samples": True, "max_length": 8}, False),
    ],
)
def test_tokenize_reports_batch_padding_as_batch_shape_sensitive(
    kwargs: dict[str, Any], sensitive: bool
) -> None:
    op = TokenizeText(tokenizer_id="__fallback__", field="text", **kwargs)

    traits = op.traits()
    assert traits.batch_shape_sensitive is sensitive
    assert traits.calibration_replayable is not sensitive
    node = Graph().add("tokenize", op)
    assert (stateless_violation(node) is not None) is sensitive


def _branched(graph: Graph) -> tuple[Node[Any], Split, Node[Any], Merge]:
    upstream = graph.add("u", DelayById())
    split = graph.add_split("u_split", upstream, Router(by=_by_field, branches=("a",)))
    op = graph.add("a", DelayById(), split)
    merge = graph.add_merge("u_merge", split, {"a": op})
    return upstream, split, op, merge


def _branched_stages() -> tuple[Stage, Stage]:
    upstream, split, op, merge = _branched(Graph())
    upstream_only = Stage(
        name="u", nodes=[upstream], placement="auto", break_reason="start"
    )
    with_branch = Stage(
        name="a",
        nodes=[op],
        placement="auto",
        break_reason="end",
        splits=[split],
        merges=[merge],
    )
    return upstream_only, with_branch


def test_runner_check_rejects_only_stages_holding_a_split() -> None:
    upstream_only, with_branch = _branched_stages()

    # A stage ending at U hands its output on; the stage with the split routes it.
    require_runner_support(upstream_only, "threads")
    with pytest.raises(NotImplementedError, match="'threads' runner cannot run yet"):
        require_runner_support(with_branch, "threads")


def test_runner_check_passes_supported_runners(monkeypatch: pytest.MonkeyPatch) -> None:
    _, with_branch = _branched_stages()
    monkeypatch.setattr(branching, "SUPPORTED_RUNNERS", frozenset({"inline"}))

    require_runner_support(with_branch, "inline")
    with pytest.raises(NotImplementedError, match="supported runners: inline"):
        require_runner_support(with_branch, "threads")


def _split_without_merge(graph: Graph) -> None:
    upstream = graph.add("u", DelayById())
    split = graph.add_split("u_split", upstream, Router(by=_by_field, branches=("a",)))
    graph.add("a", DelayById(), split)


def _merge_with_other_branches(graph: Graph) -> None:
    upstream = graph.add("u", DelayById())
    split = graph.add_split(
        "u_split", upstream, Router(by=_by_field, branches=("a", "b"))
    )
    graph.add_merge("u_merge", split, {"a": graph.add("a", DelayById(), split)})


def _stray_split_reader(graph: Graph) -> None:
    _, split, _, _ = _branched(graph)
    graph.add("stray", DelayById(), split)


def _branch_op_read_past_the_merge(graph: Graph) -> None:
    _, _, op, merge = _branched(graph)
    graph.add("j", DelayById(), merge)
    graph.add("stray", DelayById(), op)


def _op_reading_merge_and_more(graph: Graph) -> None:
    upstream, _, _, merge = _branched(graph)
    graph.add("j", DelayById(), merge, upstream)


def _branch_not_from_its_split(graph: Graph) -> None:
    upstream = graph.add("u", DelayById())
    split = graph.add_split("u_split", upstream, Router(by=_by_field, branches=("a",)))
    graph.add("a", DelayById(), split)
    graph.add_merge("u_merge", split, {"a": upstream})


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (_split_without_merge, "has 0 merges"),
        (_merge_with_other_branches, r"rejoins branches \['a'\]"),
        (_stray_split_reader, "must feed the first op of each of its branches"),
        (_branch_op_read_past_the_merge, "reads branch op 'a' directly"),
        (_op_reading_merge_and_more, "reads 'u_merge' and other inputs"),
        (_branch_not_from_its_split, "does not lead back to split 'u_split'"),
    ],
)
def test_planner_checks_split_and_merge_pairing(
    build: Callable[[Graph], None], match: str
) -> None:
    graph = Graph()
    build(graph)

    with pytest.raises(ValueError, match=match):
        Planner().make_plan(graph)


def _plan(branched: bool) -> Plan:
    graph = Graph()
    if branched:
        _, _, _, merge = _branched(graph)
        graph.add("j", DelayById(), merge)
    else:
        upstream = graph.add("u", DelayById())
        graph.add("j", DelayById(), graph.add("a", DelayById(), upstream))
    return Planner().make_plan(graph)


def test_chain_signature_keeps_its_shape() -> None:
    stages = _plan(branched=False)._signature()[0]

    # (name, placement, break reason, runner hint, nodes) per stage and
    # (name, placement, parallelism, op identity) per node, as before branches.
    assert {len(stage) for stage in stages} == {5}
    assert {len(node) for stage in stages for node in stage[4]} == {4}


def test_branched_signature_adds_edges_splits_and_merges() -> None:
    plan = _plan(branched=True)
    (stage,) = plan._signature()[0]

    nodes = {node[0]: node for node in stage[4]}
    names = [node[0] for node in stage[4]]
    assert nodes["u"][4] == ()
    assert nodes["a"][4] == (("split", 0),)
    assert nodes["j"][4] == (("merge", 0),)
    router = Router(by=_by_field, branches=("a",))
    assert stage[5] == (("u_split", ("node", names.index("u")), router.identity()),)
    assert stage[6] == (
        ("u_merge", ("split", 0), (("a", ("node", names.index("a"))),)),
    )
    assert plan.plan_id != _plan(branched=False).plan_id


def test_branched_signature_rejects_edges_that_leave_the_plan() -> None:
    upstream, split, op, merge = _branched(Graph())
    outside = Node(name="outside", op=DelayById())
    op.inputs = [outside]
    plan = Plan(
        stages=[
            Stage(
                name="s",
                nodes=[upstream, op],
                placement="auto",
                break_reason="start",
                splits=[split],
                merges=[merge],
            )
        ],
        explain="",
        indexable=True,
        preserves_cursor_order=True,
    )

    with pytest.raises(RuntimeError, match="'outside' is read by the plan"):
        plan.plan_id
