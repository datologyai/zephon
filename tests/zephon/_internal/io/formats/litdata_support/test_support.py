"""Unit tests for shared LitData metadata and output types."""

import json
from collections import OrderedDict

import pytest

optree = pytest.importorskip("optree")

from zephon._internal.io.formats.litdata_support.support import (
    FlatPyTree,
    Interval,
    row_intervals,
    treespec_dumps,
    treespec_loads,
)


@pytest.mark.parametrize(
    "structure",
    [
        "leaf",
        [],
        (),
        {},
        ("a", 1),
        {"x": [1, 2], "y": {"z": 3}},
        [{"alpha": 1}, {"beta": (2, 3)}],
        OrderedDict([("z", [1, ()]), ("a", {"nested": [2, 3]})]),
    ],
)
def test_treespec_roundtrip(structure):
    spec = optree.tree_structure(structure)
    serialized = treespec_dumps(spec)
    restored_spec = treespec_loads(serialized)
    leaves = optree.tree_leaves(structure)
    reconstructed = optree.tree_unflatten(restored_spec, leaves)
    assert reconstructed == optree.tree_unflatten(spec, leaves)


def test_litdata_dict_spec_preserves_serialized_key_order() -> None:
    # LitData's dict leaves follow the serialized key order; OpTree's ordinary
    # dict constructor sorts keys and would silently associate the wrong values.
    leaf = {"type": None, "context": None, "children_spec": []}
    serialized = json.dumps(
        [
            0,
            {
                "type": "builtins.dict",
                "context": json.dumps(["z", "a"]),
                "children_spec": [leaf, leaf],
            },
        ]
    )
    reconstructed = optree.tree_unflatten(treespec_loads(serialized), [23, 1])
    assert list(reconstructed.items()) == [("z", 23), ("a", 1)]


@pytest.mark.parametrize(
    ("region", "expected"),
    [
        (None, [Interval(0, 0, 2, 2), Interval(2, 2, 5, 5)]),
        ([(0, 1), (1, 3)], [Interval(0, 0, 1, 2), Interval(2, 3, 5, 5)]),
    ],
)
def test_row_intervals(
    region: list[tuple[int, int]] | None, expected: list[Interval]
) -> None:
    assert row_intervals([{"chunk_size": 2}, {"chunk_size": 3}], region) == expected


def test_flat_pytree_materializes_structure() -> None:
    expected = {"text": ("payload", 7), "values": [1, 2]}
    leaves, spec = optree.tree_flatten(expected)
    flat = FlatPyTree(leaves, spec)
    assert flat.leaves == leaves
    assert flat.materialize() == expected
    assert flat.to_tree() == expected
