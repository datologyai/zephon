# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Router, stateless rule and runner check for ``Pipeline.branch()``."""

from __future__ import annotations

from typing import Any

import pytest

from zephon._internal import branching
from zephon._internal.branching import (
    Router,
    is_branched,
    require_runner_support,
    stateless_violation,
)
from zephon._internal.graph import Graph, Plan, Stage
from zephon._internal.ops.delay import DelayById
from zephon._internal.ops.identity import Identity
from zephon._internal.ops.map_transform import MapTransform
from zephon._internal.ops.shuffle_buffer import ShuffleBuffer
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


def test_is_branched_sees_routers_and_branch_ops() -> None:
    graph = Graph()
    upstream = graph.add("u", DelayById())
    assert not is_branched(graph.nodes)

    upstream.router = Router(by=_by_field, branches=("a",))
    assert is_branched(graph.nodes)


def _branched_stage() -> tuple[Stage, Stage]:
    graph = Graph()
    upstream = graph.add("u", DelayById())
    upstream.router = Router(by=_by_field, branches=("a",))
    op = graph.add("a", DelayById(), upstream, branch="a")
    upstream_only = Stage(
        name="u", nodes=[upstream], placement="auto", break_reason="start"
    )
    with_branch = Stage(name="a", nodes=[op], placement="auto", break_reason="end")
    return upstream_only, with_branch


def test_runner_check_rejects_only_stages_holding_branch_ops() -> None:
    upstream_only, with_branch = _branched_stage()

    # A stage ending at U hands its output on; the next stage routes it.
    require_runner_support(upstream_only, "threads")
    with pytest.raises(NotImplementedError, match="'threads' runner cannot run yet"):
        require_runner_support(with_branch, "threads")


def test_runner_check_passes_supported_runners(monkeypatch: pytest.MonkeyPatch) -> None:
    _, with_branch = _branched_stage()
    monkeypatch.setattr(branching, "SUPPORTED_RUNNERS", frozenset({"inline"}))

    require_runner_support(with_branch, "inline")
    with pytest.raises(NotImplementedError, match="supported runners: inline"):
        require_runner_support(with_branch, "threads")


def _plan(branched: bool) -> Plan:
    graph = Graph()
    upstream = graph.add("u", DelayById())
    op = graph.add("op", DelayById(), upstream, branch="a" if branched else None)
    if branched:
        upstream.router = Router(by=_by_field, branches=("a",))
    graph.add("merge", DelayById(), op)
    return Planner().make_plan(graph)


def test_chain_signature_keeps_its_shape() -> None:
    signature = _plan(branched=False)._signature()
    stages = signature[0]

    # (name, placement, parallelism, op identity) per node, as before branches.
    assert {len(node) for stage in stages for node in stage[4]} == {4}


def test_branched_signature_adds_inputs_branch_and_router() -> None:
    plan = _plan(branched=True)
    nodes = [node for stage in plan._signature()[0] for node in stage[4]]

    names = [node[0] for node in nodes]
    upstream, op, merge = (nodes[names.index(name)] for name in ("u", "op", "merge"))
    assert upstream[4:] == ((), None, Router(by=_by_field, branches=("a",)).identity())
    assert op[4:] == ((names.index("u"),), "a", None)
    assert merge[4:] == ((names.index("op"),), None, None)
    assert plan.plan_id != _plan(branched=False).plan_id
