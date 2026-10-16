# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
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

    def fake_create_index(
        fmt, dataset_dir, *, output_path=None, progress=True, max_workers=8
    ):
        seen.update(
            fmt=fmt,
            dir=dataset_dir,
            out=output_path,
            progress=progress,
            max_workers=max_workers,
        )
        return "/idx/index.json"

    monkeypatch.setattr(bi, "_create_index", fake_create_index)

    result = bi.build_index("parquet", "/data", max_workers=2)

    assert imported == ["zephon._internal.io.index.parquet_index"]
    assert seen == {
        "fmt": "parquet",
        "dir": "/data",
        "out": None,
        "progress": True,
        "max_workers": 2,
    }
    assert result == "/idx/index.json"


def test_build_index_imports_jsonl_builder(monkeypatch):
    """JSONL is registered as an indexable CLI format."""
    imported: list[str] = []
    monkeypatch.setattr(bi.importlib, "import_module", lambda m: imported.append(m))
    monkeypatch.setattr(
        bi,
        "_create_index",
        lambda *_args, **_kwargs: "/idx/index.json",
    )

    bi.build_index("jsonl", "/data")

    assert imported == ["zephon._internal.io.index.jsonl_index"]


def test_build_index_unknown_format_raises():
    with pytest.raises(ValueError):
        bi.build_index("no_such_format", "/data")


@pytest.mark.parametrize("path_input", [False, True])
@pytest.mark.parametrize("output", ["default", "path", "str"])
def test_build_index_local_output_is_string(
    tmp_path: Path, path_input: bool, output: str
) -> None:
    (tmp_path / "data.jsonl").write_text('{"i": 0}\n', encoding="utf-8")
    target = tmp_path / ("index.json" if output == "default" else "custom.json")
    output_path: str | Path | None = None
    if output == "path":
        output_path = target
    elif output == "str":
        output_path = str(target)

    result = bi.build_index(
        "jsonl",
        tmp_path if path_input else str(tmp_path),
        output_path=output_path,
        progress=False,
    )

    assert result == str(target)
    assert json.loads(target.read_text(encoding="utf-8"))["shards"][0]["num_rows"] == 1


def test_cli_prints_public_usage_and_exits(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["build_index"])  # too few args
    with pytest.raises(SystemExit) as exc:
        bi._cli()
    assert exc.value.code == 1
    assert "python -m zephon.build_index" in capsys.readouterr().out


def test_cli_builds_mixed_jsonl_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.jsonl").write_text('{"i":0}\n\nnull', encoding="utf-8")
    with gzip.open(tmp_path / "b.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write('{"i":1}\n')
    monkeypatch.setattr(sys, "argv", ["build_index", "jsonl", str(tmp_path)])

    bi._cli()

    data = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert [shard["basename"] for shard in data["shards"]] == ["a.jsonl", "b.jsonl.gz"]
    assert [shard["num_rows"] for shard in data["shards"]] == [2, 1]
