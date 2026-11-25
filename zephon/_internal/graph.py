# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""In-memory representation of a Zephon pipeline graph and execution plan."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar, Union

from zephon._internal.branching import Router
from zephon.ops.base import BaseOp

# TypeVar for operator type, enabling typed Node[OpT] where OpT is the specific op
OpT = TypeVar("OpT", bound=BaseOp)


@dataclass
class Node(Generic[OpT]):
    """A vertex in the logical pipeline graph bound to an operator instance."""

    name: str
    op: OpT
    inputs: list["Vertex"] = field(default_factory=list)
    placement: str = "auto"
    parallelism: Optional[int] = None


@dataclass(eq=False)
class Split:
    """Sends each record from ``input`` to exactly one branch.

    A split is a graph vertex, not an op. The first op of every branch takes
    the split as its only input, and the split's `Merge` rejoins the
    branches.
    """

    name: str
    input: Node[Any]
    router: Router


@dataclass(eq=False)
class Merge:
    """Rejoins the branches of ``split`` in input order.

    A merge is a graph vertex, not an op. ``branches`` maps each branch name,
    in the router's order, to the last op of that branch. The op after the
    branches takes the merge as its only input.
    """

    name: str
    split: Split
    branches: dict[str, Node[Any]]


# Anything an op can take as input.
Vertex = Union[Node[Any], Split, Merge]


class Graph:
    """Mutable DAG used by the pipeline builder to describe operator topology.

    Internal DAG representation. Do not add operators here — use
    ``Pipeline.add_op()`` or ``Pipeline.map_transform()``.
    """

    def __init__(self) -> None:
        """Initialize an empty graph in insertion order."""
        self.nodes: list[Node[Any]] = []
        self.splits: list[Split] = []
        self.merges: list[Merge] = []

    def add(
        self,
        name: str,
        op: OpT,
        *inputs: Vertex,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> Node[OpT]:
        """Create a node, infer defaults, and append it to the graph order."""
        if parallelism is None:
            parallelism = max(1, op.traits().parallelism)
        node: Node[OpT] = Node(
            name=name,
            op=op,
            inputs=list(inputs),
            placement=placement,
            parallelism=parallelism,
        )
        self.nodes.append(node)
        return node

    def add_split(self, name: str, input: Node[Any], router: Router) -> Split:
        """Add a split that routes ``input``'s records with ``router``."""
        split = Split(name=name, input=input, router=router)
        self.splits.append(split)
        return split

    def add_merge(
        self, name: str, split: Split, branches: Mapping[str, Node[Any]]
    ) -> Merge:
        """Add the merge that rejoins ``split``'s branches.

        Args:
            name: Vertex name shown by ``explain()``.
            split: The split whose branches this merge rejoins.
            branches: Branch name to the last op of that branch.
        """
        merge = Merge(name=name, split=split, branches=dict(branches))
        self.merges.append(merge)
        return merge


@dataclass
class Stage:
    """A contiguous run of graph nodes that execute under the same runner.

    ``splits`` and ``merges`` are the branch points whose branch ops run in
    this stage; the runner routes and rejoins them here.
    """

    name: str
    nodes: list[Node[Any]]
    placement: str
    break_reason: str
    runner_hint: str | None = None
    splits: list[Split] = field(default_factory=list)
    merges: list[Merge] = field(default_factory=list)


@dataclass
class Plan:
    """Plannable pipeline describing stage layout and derived metadata."""

    stages: list[Stage]
    explain: str
    indexable: bool
    preserves_cursor_order: bool
    batch_size_hint: int | None = None

    # --- Deterministic identity helpers ---
    def _freeze(self, value: Any) -> Any:
        """Convert common Python containers into hashable, ordered tuples."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple, set)):
            return tuple(self._freeze(v) for v in value)
        if isinstance(value, dict):
            return tuple(
                (str(k), self._freeze(v))
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            )
        raise TypeError(f"non-freezable value for identity: {type(value)!r}")

    def _op_identity(self, op: BaseOp) -> tuple[Any, ...]:
        """Return an operator signature capturing class, config, and traits."""
        config_items: list[tuple[str, Any]] = []
        for key, val in sorted(getattr(op, "__dict__", {}).items()):
            if key.startswith("_"):
                continue
            try:
                config_items.append((key, self._freeze(val)))
            except TypeError:
                continue

        traits_payload: tuple[Any, ...] = ()
        try:
            traits = op.traits()
            traits_payload = tuple(
                (name, self._freeze(getattr(traits, name)))
                for name in sorted(vars(traits))
            )
        except Exception:
            traits_payload = ()

        return (
            op.plan_identity(),
            tuple(config_items),
            traits_payload,
        )

    def _signature(self) -> tuple[Any, ...]:
        """Compact tuple describing the plan for hashing purposes.

        A chain's order already defines its edges. A branched plan also lists
        each node's inputs and each stage's splits and merges, so two graphs
        with the same ops but different edges differ. Chains keep their
        previous signature, and so their plan ids.
        """
        branched = any(st.splits or st.merges for st in self.stages)
        refs = self._vertex_refs() if branched else {}

        def ref(vertex: Vertex) -> tuple[str, int]:
            try:
                return refs[id(vertex)]
            except KeyError:
                raise RuntimeError(
                    f"plan vertex {vertex.name!r} is read by the plan but is not "
                    "part of it"
                ) from None

        stages_sig: list[Any] = []
        for st in self.stages:
            nodes_sig: list[Any] = []
            for nd in st.nodes:
                node_sig: tuple[Any, ...] = (
                    nd.name,
                    nd.placement,
                    int(nd.parallelism) if nd.parallelism else 1,
                    self._op_identity(nd.op),
                )
                if branched:
                    node_sig += (tuple(ref(inp) for inp in nd.inputs),)
                nodes_sig.append(node_sig)
            stage_sig: tuple[Any, ...] = (
                st.name,
                st.placement,
                st.break_reason,
                st.runner_hint,
                tuple(nodes_sig),
            )
            if branched:
                stage_sig += (
                    tuple(
                        (sp.name, ref(sp.input), sp.router.identity())
                        for sp in st.splits
                    ),
                    tuple(
                        (
                            mg.name,
                            ref(mg.split),
                            tuple(
                                (name, ref(tail)) for name, tail in mg.branches.items()
                            ),
                        )
                        for mg in st.merges
                    ),
                )
            stages_sig.append(stage_sig)

        return (
            tuple(stages_sig),
            bool(self.indexable),
            bool(self.preserves_cursor_order),
            self.batch_size_hint,
        )

    def _vertex_refs(self) -> dict[int, tuple[str, int]]:
        """Number every node, split and merge of the plan by kind, in plan order."""
        refs: dict[int, tuple[str, int]] = {}
        for kind, vertices in (
            ("node", [nd for st in self.stages for nd in st.nodes]),
            ("split", [sp for st in self.stages for sp in st.splits]),
            ("merge", [mg for st in self.stages for mg in st.merges]),
        ):
            for index, vertex in enumerate(vertices):
                refs[id(vertex)] = (kind, index)
        return refs

    def fingerprint(self) -> str:
        """Return a stable SHA-256 hex digest for this plan."""
        blob = repr(self._signature()).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    @property
    def plan_id(self) -> str:
        """Short identifier derived from the plan fingerprint.

        First 16 hex characters of the SHA-256 digest; suitable for logs/paths.
        """
        return self.fingerprint()[:16]
