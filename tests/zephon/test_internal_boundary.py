# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""External surfaces must import only supported paths.

Examples, README, and docs must never use ``zephon._internal`` or the non-public
``zephon.{api,core,runners,utils}`` names. Zephon's own tests may use internals.
"""

import importlib
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_NAMES = "_internal|api|core|runners|utils"
# A dotted reference (``zephon._internal.x``), bounded by a trailing `.` or
# word-end so doc-page names like ``zephon_api`` (underscore) don't match; or a
# ``from zephon import _internal`` style import of one of the names.
_FORBIDDEN = re.compile(
    rf"\bzephon\.(?:{_NAMES})(?=\.|\b)"
    rf"|\bfrom\s+zephon\s+import\b[^\n]*\b(?:{_NAMES})\b"
)


def _consumer_files() -> list[Path]:
    files: list[Path] = []
    files.extend((_REPO / "examples").rglob("*.py"))
    files.extend((_REPO / "docs").rglob("*.md"))
    files.extend((_REPO / "docs").rglob("*.rst"))
    readme = _REPO / "README.md"
    if readme.exists():
        files.append(readme)
    return files


def test_consumer_surfaces_use_only_supported_paths():
    offenders: list[str] = []
    for f in _consumer_files():
        for i, line in enumerate(f.read_text().splitlines(), 1):
            m = _FORBIDDEN.search(line)
            if m:
                offenders.append(f"{f.relative_to(_REPO)}:{i}: {line.strip()}")
    assert not offenders, (
        "external surfaces must use only supported import paths:\n"
        + "\n".join(offenders)
    )


def test_internal_package_ships_but_is_not_exported():
    import zephon

    importlib.import_module("zephon._internal")  # must ship / be importable
    assert "_internal" not in zephon.__all__
