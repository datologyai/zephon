# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

import zephon.build_index as bi


def test_build_index_delegates(monkeypatch):
    # A known format imports its (builder-registering) module, then delegates to
    # the internal create_index. Stub both so the test needs no format deps.
    imported: list[str] = []
    monkeypatch.setattr(bi.importlib, "import_module", lambda m: imported.append(m))

    seen: dict[str, object] = {}

    def fake_create_index(fmt, dataset_dir, *, output_path=None, progress=True):
        seen.update(fmt=fmt, dir=dataset_dir, out=output_path, progress=progress)
        return Path("/idx/index.json")

    monkeypatch.setattr(bi, "_create_index", fake_create_index)

    result = bi.build_index("parquet", "/data")

    assert imported == ["zephon._internal.io.index.parquet_index"]
    assert seen == {"fmt": "parquet", "dir": "/data", "out": None, "progress": True}
    assert str(result) == "/idx/index.json"


def test_build_index_imports_jsonl_builder(monkeypatch):
    """JSONL is registered as an indexable CLI format."""
    imported: list[str] = []
    monkeypatch.setattr(bi.importlib, "import_module", lambda m: imported.append(m))
    monkeypatch.setattr(
        bi,
        "_create_index",
        lambda *_args, **_kwargs: Path("/idx/index.json"),
    )

    bi.build_index("jsonl", "/data")

    assert imported == ["zephon._internal.io.index.jsonl_index"]


def test_build_index_unknown_format_raises():
    with pytest.raises(ValueError):
        bi.build_index("no_such_format", "/data")


def test_cli_prints_public_usage_and_exits(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["build_index"])  # too few args
    with pytest.raises(SystemExit) as exc:
        bi._cli()
    assert exc.value.code == 1
    assert "python -m zephon.build_index" in capsys.readouterr().out
