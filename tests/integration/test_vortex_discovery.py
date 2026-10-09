# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Public dataset discovery and index-building integration for Vortex."""

from pathlib import Path

import pytest

vortex = pytest.importorskip("vortex", reason="vortex-data not installed")

from zephon._internal.io.formats import vortex as vortex_format
from zephon._internal.io.formats import vortex_metadata
from zephon.build_index import build_index
from zephon.io.dataset import Dataset

pytestmark = pytest.mark.integration


def test_vortex_discovery_index_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index preserves discovered counts and removes the need to read footers."""
    for name, count in (("b.vortex", 3), ("a.vortex", 2)):
        vortex.io.write(
            vortex.array([{"value": i} for i in range(count)]), str(tmp_path / name)
        )

    discovered = Dataset.from_path("discovered", str(tmp_path), fmt="vortex")
    assert discovered.shard_count() == 2
    assert discovered.total() == 5
    assert build_index("vortex", tmp_path, progress=False) == tmp_path / "index.json"

    monkeypatch.setattr(vortex_format, "_vortex", None)
    monkeypatch.setattr(vortex_metadata, "_vortex", None)
    indexed = Dataset.from_path("indexed", str(tmp_path), fmt="vortex")
    assert indexed.shard_count() == discovered.shard_count()
    assert indexed.total() == discovered.total()
