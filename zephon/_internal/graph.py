# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""In-memory representation of a Zephon pipeline graph and execution plan."""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

from zephon._internal.op_base import Op

# TypeVar for operator type, enabling typed Node[OpT] where OpT is the specific op
OpT = TypeVar("OpT", bound=Op[Any, Any])


@dataclass
class Node(Generic[OpT]):
    """A vertex in the logical pipeline graph bound to an operator instance."""

    name: str
    op: OpT
    inputs: list["Node[Any]"] = field(default_factory=list)
    placement: str = "auto"
    parallelism: Optional[int] = None


class Graph:
    """Mutable DAG used by the pipeline builder to describe operator topology.

    Internal DAG representation. Do not add operators here — use
    ``Pipeline.add_op()`` or ``Pipeline.map_transform()``.
    """

    def __init__(self) -> None:
        """Initialize an empty graph in insertion order."""
        self.nodes: list[Node[Any]] = []

    def add(
        self,
        name: str,
        op: OpT,
        *inputs: Node[Any],
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


@dataclass
class Stage:
    """A contiguous run of graph nodes that execute under the same runner."""

    name: str
    nodes: list[Node[Any]]
    placement: str
    break_reason: str
    runner_hint: str | None = None


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

    def _op_identity(self, op: Any) -> tuple[Any, ...]:
        """Return an operator signature capturing class, config, and traits."""
        cls = op.__class__
        fqcn = f"{cls.__module__}.{cls.__qualname__}"

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
            fqcn,
            tuple(config_items),
            traits_payload,
        )

    def _signature(self) -> tuple[Any, ...]:
        """Compact tuple describing the plan for hashing purposes."""
        stages_sig: list[Any] = []
        for st in self.stages:
            nodes_sig: list[Any] = []
            for nd in st.nodes:
                nodes_sig.append(
                    (
                        nd.name,
                        nd.placement,
                        int(nd.parallelism) if nd.parallelism else 1,
                        self._op_identity(nd.op),
                    )
                )
            stages_sig.append(
                (
                    st.name,
                    st.placement,
                    st.break_reason,
                    st.runner_hint,
                    tuple(nodes_sig),
                )
            )

        return (
            tuple(stages_sig),
            bool(self.indexable),
            bool(self.preserves_cursor_order),
            self.batch_size_hint,
        )

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
