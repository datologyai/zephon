"""Shared metadata, output types, and reader interface for LitData."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import optree


@dataclass(slots=True)
class FlatPyTree:
    """Lightweight wrapper around flattened pytree leaves with lazy reconstruction."""

    leaves: list[Any]
    spec: optree.PyTreeSpec

    def materialize(self) -> Any:
        """Reconstruct the original pytree structure."""
        return optree.tree_unflatten(self.spec, self.leaves)

    # Alias for ergonomics
    to_tree = materialize


class Interval(NamedTuple):
    """Represents a half-open interval [chunk_start, chunk_end) for a chunk."""

    chunk_start: int
    roi_start_idx: int
    roi_end_idx: int
    chunk_end: int


def row_intervals(
    chunks: Sequence[Mapping[str, Any]],
    region_of_interest: list[tuple[int, int]] | None = None,
) -> list[Interval]:
    """Compute row intervals shared by serialized-pytree and Arrow chunks."""
    intervals: list[Interval] = []
    begin = 0
    end = 0
    for idx, chunk in enumerate(chunks):
        chunk_size = int(chunk["chunk_size"])
        end += chunk_size
        start_idx = begin
        end_idx = end
        if region_of_interest is not None:
            roi = region_of_interest[idx]
            start_idx = begin + roi[0]
            end_idx = begin + roi[1]
        intervals.append(Interval(begin, start_idx, end_idx, end))
        begin += chunk_size
    return intervals


class BaseItemLoader(ABC):
    """Shared metadata and read interface for LitData layout readers."""

    def setup(
        self,
        config: Mapping[str, Any],
        chunks: list[Mapping[str, Any]],
        serializers: Mapping[str, Any] | None,
        region_of_interest: list[tuple[int, int]] | None = None,
    ) -> None:
        # The serializer argument is consumed by the binary layout readers.
        self._config = dict(config)
        self._chunks = [dict(chunk) for chunk in chunks]
        self.region_of_interest = region_of_interest
        flag = config.get("return_flat_leaves")
        self._return_flat_leaves = (
            flag
            if isinstance(flag, bool)
            else getattr(self, "_return_flat_leaves", False)
        )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def generate_intervals(self) -> list[Interval]:
        return row_intervals(self._chunks, self.region_of_interest)

    @abstractmethod
    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
    ) -> Any: ...

    def load_items_from_chunk(
        self,
        indices: list[int],
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
    ) -> list[Any]:
        """Load multiple items from a chunk efficiently.

        Default implementation falls back to calling load_item_from_chunk
        for each index. Subclasses should override for better performance.
        """
        return [
            self.load_item_from_chunk(
                index, chunk_index, chunk_filepath, begin, filesize_bytes
            )
            for index in indices
        ]

    def load_item_from_bytes(
        self, raw_bytes: bytes, chunk_index: int
    ) -> Any:  # pragma: no cover - rarely used
        raise NotImplementedError

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        if os.path.exists(chunk_filepath):
            os.remove(chunk_filepath)


def treespec_loads(serialized: str) -> optree.PyTreeSpec:
    """Deserialize a PyTreeSpec from the legacy LitData JSON representation."""
    _protocol, json_schema = json.loads(serialized)
    return _convert_legacy_treespec(json_schema)


def _convert_legacy_treespec(schema: Mapping[str, Any]) -> optree.PyTreeSpec:
    # Use the public constructors available at our OpTree 0.12 floor. The
    # optree.treespec convenience namespace was only added in 0.14.1.
    if (
        schema.get("type") is None
        and schema.get("context") is None
        and len(schema.get("children_spec", [])) == 0
    ):
        return optree.treespec_leaf()

    children = [
        _convert_legacy_treespec(child) for child in schema.get("children_spec", [])
    ]

    type_name = schema.get("type")
    context = json.loads(schema["context"]) if schema.get("context") else None

    if type_name == "builtins.dict":
        keys = context if isinstance(context, list) else []
        pairs = [(key, child) for key, child in zip(keys, children)]
        return optree.treespec_ordereddict(pairs)
    if type_name == "builtins.list":
        return optree.treespec_list(children)
    if type_name == "builtins.tuple":
        return optree.treespec_tuple(children)
    if type_name == "collections.OrderedDict":
        keys = context if isinstance(context, list) else []
        pairs = [(key, child) for key, child in zip(keys, children)]
        return optree.treespec_ordereddict(pairs)
    return optree.treespec_list(children)


def treespec_dumps(spec: optree.PyTreeSpec) -> str:
    """Serialize an optree PyTreeSpec into the legacy LitData JSON representation."""

    def _encode(node: optree.PyTreeSpec) -> dict[str, Any]:
        if node.is_leaf():
            return {"type": None, "context": None, "children_spec": []}

        children = [_encode(child) for child in node.children()]
        kind: optree.PyTreeKind = node.kind  # type: ignore[assignment]
        context: Any = None

        if kind == optree.PyTreeKind.TUPLE:
            type_name = "builtins.tuple"
        elif kind == optree.PyTreeKind.LIST:
            type_name = "builtins.list"
        elif kind == optree.PyTreeKind.DICT:
            type_name = "builtins.dict"
            context = list(node.entries())
        elif kind == optree.PyTreeKind.ORDEREDDICT:
            type_name = "collections.OrderedDict"
            context = list(node.entries())
        else:
            type_name = "builtins.list"

        return {
            "type": type_name,
            "context": json.dumps(context) if context is not None else None,
            "children_spec": children,
        }

    schema = _encode(spec)
    return json.dumps([0, schema])


__all__ = [
    "BaseItemLoader",
    "FlatPyTree",
    "Interval",
    "row_intervals",
    "treespec_dumps",
    "treespec_loads",
]
