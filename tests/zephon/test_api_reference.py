# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Check the API reference's ``:type:`` lines against the aliases they describe.

``docs/source/api/*.rst`` spells out each public type alias by hand, because a
bare ``py:data`` directive renders without one. Nothing ties those strings to
the definitions, so an alias can gain a union arm or a ``Literal`` member while
the docs keep showing the old shape. Both sides are read with ``ast``: no Sphinx
build, and ``zephon`` is never imported.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
API_DOCS = ROOT / "docs" / "source" / "api"
PACKAGE = ROOT / "zephon"

# ``.. py:data:: Name``, then its ``:type:``, tolerating other options between.
DIRECTIVE = re.compile(
    r"^\.\. py:data:: (\w+)\n(?:[ \t]+:[\w-]+:.*\n)*?[ \t]+:type: *(.+)$",
    re.MULTILINE,
)


class _Unquote(ast.NodeTransformer):
    """Resolve forward references, so ``list["X"]`` reads as ``list[X]``."""

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # Literal members are values, not forward references; unquoting them
        # would let a member renamed to an identifier compare equal to a type.
        if isinstance(node.value, ast.Name) and node.value.id == "Literal":
            return node
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str) and node.value.isidentifier():
            return ast.Name(id=node.value, ctx=ast.Load())
        return node


def _normalize(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    return ast.unparse(ast.fix_missing_locations(_Unquote().visit(tree)).body)


def _source_aliases() -> dict[str, set[str]]:
    """Module-level assignments in the public package, keyed by name."""
    aliases: dict[str, set[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE)
        # Private directories are out entirely; ``__init__.py`` is the one
        # underscore-prefixed file that is still public API.
        if any(part.startswith("_") for part in relative.parts[:-1]):
            continue
        if relative.name.startswith("_") and relative.name != "__init__.py":
            continue
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.AnnAssign):
                target, value = node.target, node.value
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            else:
                continue
            if isinstance(target, ast.Name) and value is not None:
                aliases.setdefault(target.id, set()).add(ast.unparse(value))
    return aliases


RST_FILES = sorted(API_DOCS.rglob("*.rst"))
DOCUMENTED = [
    (rst.relative_to(ROOT).as_posix(), name, declared)
    for rst in RST_FILES
    for name, declared in DIRECTIVE.findall(rst.read_text())
]
DECLARED_COUNT = sum(
    len(re.findall(r"^\.\.\s+py:data::", rst.read_text(), re.MULTILINE))
    for rst in RST_FILES
)
SOURCE = _source_aliases()


def test_every_py_data_directive_declares_a_type():
    # The check below is parametrized, so a directive this file fails to pair
    # with its ``:type:`` is simply never collected and nothing fails.
    assert DOCUMENTED and len(DOCUMENTED) == DECLARED_COUNT, (
        f"{DECLARED_COUNT} `.. py:data::` directives under {API_DOCS}, but "
        f"{len(DOCUMENTED)} carry a `:type:` this test could read. Give the alias "
        "a `:type:`, or the reference documents a shape nothing verifies."
    )


@pytest.mark.parametrize(
    ("rst", "name", "declared"), DOCUMENTED, ids=[n for _, n, _ in DOCUMENTED]
)
def test_documented_type_matches_source(rst: str, name: str, declared: str):
    defined = SOURCE.get(name)
    assert defined, f"{rst} documents `{name}`, which no public module defines"
    found = sorted(_normalize(value) for value in defined)
    assert _normalize(declared) in found, (
        f"`{name}` has drifted: {rst} says {_normalize(declared)}, source says {found}"
    )
