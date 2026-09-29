# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the catalog handle's build-timing logs.

A first-contact ``finalize()`` runs full discovery (which can take minutes on a
remote root), so it announces the build up front and then reports how long it
took.
"""

import json
import logging
import re
from pathlib import Path

import pytest

import zephon._internal.io.catalog.handle as handle_mod
from zephon._internal.io.catalog import (
    DatasetHeader,
    ShardCatalogHandle,
    clear_registry,
    finalize,
    set_catalog_dir,
)
from zephon._internal.io.catalog.builder import BuiltCatalog, build_catalog
from zephon.io.options import StoreOptions


@pytest.fixture
def catalog_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ZEPHON_CATALOG_DIR", str(tmp_path))
    set_catalog_dir(StoreOptions())
    clear_registry()
    yield tmp_path
    clear_registry()


def _make_jsonl(root: Path, shards: dict[str, int]) -> None:
    for name, count in shards.items():
        path = root / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps({"i": i}) + "\n" for i in range(count)),
            encoding="utf-8",
        )


def _header(root: Path, name: str = "ds") -> DatasetHeader:
    return DatasetHeader(name=name, root=str(root), format="jsonl", path=str(root))


def test_finalize_brackets_build_with_timing_logs(
    catalog_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A first-contact build emits the start line, then a completion line with a
    real elapsed time."""
    root = tmp_path / "ds"
    root.mkdir()
    _make_jsonl(root, {"a": 3, "b": 5})

    with caplog.at_level(logging.INFO, logger="zephon._internal.io.catalog.handle"):
        finalize(ShardCatalogHandle(dataset=_header(root)))

    timing = [
        r.getMessage()
        for r in caplog.records
        if "shard catalog for dataset 'ds'" in r.getMessage()
    ]
    assert len(timing) == 2
    assert timing[0].startswith("Building shard catalog for dataset 'ds'")
    assert str(root) in timing[0]
    assert re.fullmatch(r"Built shard catalog for dataset 'ds' in \d+\.\d+s", timing[1])


def test_only_discovery_inputs_key_a_scan_catalog(
    catalog_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decoded copies written beside the shards must not force a rebuild."""
    builds: list[BuiltCatalog] = []

    def _counting_build(header: DatasetHeader) -> BuiltCatalog:
        builds.append(build_catalog(header))
        return builds[-1]

    monkeypatch.setattr(handle_mod, "build_catalog", _counting_build)
    root = tmp_path / "ds"
    root.mkdir()
    _make_jsonl(root, {"a": 3})
    finalize(ShardCatalogHandle(dataset=_header(root)))
    assert len(builds) == 1

    (root / "b.jsonl.gz.raw").write_text("decoded copy\n", encoding="utf-8")
    finalize(ShardCatalogHandle(dataset=_header(root)))
    assert len(builds) == 1

    _make_jsonl(root, {"c": 2})  # a new shard does change the dataset
    finalize(ShardCatalogHandle(dataset=_header(root)))
    assert len(builds) == 2
