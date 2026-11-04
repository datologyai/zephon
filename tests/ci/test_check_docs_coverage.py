# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / ".github" / "scripts" / "check_docs_coverage.py"
_spec = importlib.util.spec_from_file_location("check_docs_coverage", _SCRIPT)
assert _spec and _spec.loader
cdc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cdc)


class _Shared:
    pass


class _RootOnly:
    pass


@pytest.fixture
def fake_package(monkeypatch):
    # A root that re-exports one submodule name and adds one of its own.
    root = types.ModuleType("fakepkg")
    root.__all__ = ["Shared", "RootOnly", "__version__"]
    root.Shared, root.RootOnly, root.__version__ = _Shared, _RootOnly, "0"
    sub = types.ModuleType("fakepkg.sub")
    sub.__all__ = ["Shared"]
    sub.Shared = _Shared
    monkeypatch.setitem(sys.modules, "fakepkg", root)
    monkeypatch.setitem(sys.modules, "fakepkg.sub", sub)


def test_public_modules_include_root():
    assert "zephon" in cdc.public_modules()


@pytest.mark.usefixtures("fake_package")
def test_root_only_export_is_reported():
    missing = cdc.find_undocumented({"fakepkg.sub.Shared"}, ["fakepkg", "fakepkg.sub"])
    assert missing == ["fakepkg.RootOnly"]


@pytest.mark.usefixtures("fake_package")
def test_any_documented_path_covers_a_shared_object():
    documented = {"fakepkg.sub.Shared", "fakepkg.RootOnly"}
    assert cdc.find_undocumented(documented, ["fakepkg", "fakepkg.sub"]) == []
