# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Public Vortex discovery and index roundtrip."""

import json
from pathlib import Path

import pytest

pytest.importorskip("vortex.io", reason="vortex-data not installed")
import vortex

from zephon._internal.io.formats import vortex as vortex_format
from zephon.build_index import build_index
from zephon.io import Dataset

pytestmark = pytest.mark.integration


def test_vortex_discovery_index_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index preserves discovered counts and removes the need to read footers."""
    for name, count in (("b.vortex", 3), ("a.vortex", 2)):
        vortex.io.write(
            vortex.array([{"value": i} for i in range(count)]), str(tmp_path / name)
        )

    discovered = Dataset.from_path("discovered", str(tmp_path))
    assert discovered.backend["kind"] == "vortex"
    assert discovered.shard_count() == 2
    assert discovered.total() == 5
    assert len(discovered) == 5
    target = tmp_path / "custom-index.json"
    assert build_index("vortex", tmp_path, output_path=target, progress=False) == target
    index = json.loads(target.read_text())
    assert [shard["basename"] for shard in index["shards"]] == ["a.vortex", "b.vortex"]
    assert [shard["extra"] for shard in index["shards"]] == [
        {"length": 2},
        {"length": 3},
    ]
    target.rename(tmp_path / "index.json")

    monkeypatch.setattr(vortex_format, "_vortex", None)
    indexed = Dataset.from_path("indexed", str(tmp_path), fmt="vortex")
    assert indexed.shard_count() == discovered.shard_count()
    assert indexed.total() == discovered.total()
