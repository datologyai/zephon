# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""In-memory representation of a Zephon pipeline graph and execution plan."""

from dataclasses import dataclass, field
from typing import Optional

from zephon.core.op_base import Op


@dataclass
class Node:
    """A vertex in the logical pipeline graph bound to an operator instance."""

    name: str
    op: Op
    inputs: list["Node"] = field(default_factory=list)
    placement: str = "auto"
    parallelism: Optional[int] = None


class Graph:
    """Mutable DAG used by the pipeline builder to describe operator topology."""

    def __init__(self) -> None:
        """Initialize an empty graph in insertion order."""
        self.nodes: list[Node] = []

    def add(
        self,
        name: str,
        op: Op,
        *inputs: Node,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> Node:
        """Create a node, infer defaults, and append it to the graph order."""
        if parallelism is None:
            parallelism = max(1, op.traits().parallelism)
        node = Node(
            name=name,
            op=op,
            inputs=list(inputs),
            placement=placement,
            parallelism=parallelism,
        )
        self.nodes.append(node)
        return node


@dataclass
class Stage:
    """A contiguous run of graph nodes that execute under the same runner."""

    name: str
    nodes: list[Node]
    placement: str
    break_reason: str


@dataclass
class Plan:
    """Plannable pipeline describing stage layout and derived metadata."""

    stages: list[Stage]
    explain: str
    indexable: bool
