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

from zephon._internal.io.catalog import (
    DatasetHeader,
    ShardCatalogHandle,
    clear_registry,
    finalize,
    set_catalog_dir,
)
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
