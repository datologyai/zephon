# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Index-path tests for ``VortexFormat.discover_counts``.

Separate from ``test_vortex.py``, which skips entirely without the
``vortex-data`` package: the index fast path never opens a ``.vortex`` file,
so count-only discovery must work — and stay in agreement with ``discover`` —
without the package installed.
"""

import json
from pathlib import Path

import pytest

from zephon.io.formats.vortex import VortexFormat
from zephon.io.storage.local import LocalFSBackend


def _write_index(tmp_path: Path, shards: list[dict[str, object]]) -> None:
    (tmp_path / "index.json").write_text(
        json.dumps({"format_version": 1, "shards": shards}), encoding="utf-8"
    )


def test_discover_counts_matches_discover_on_index_path(tmp_path: Path) -> None:
    """Counts agree with ``discover`` on ``(shard_id, num_rows)``."""
    _write_index(
        tmp_path,
        [
            {"basename": "a.vortex", "bytes": 10, "num_rows": 3},
            {"basename": "b.vortex", "bytes": 20, "num_rows": 5},
        ],
    )
    handler = VortexFormat()
    storage = LocalFSBackend(tmp_path)

    shard_index, _ = handler.discover(str(tmp_path), storage)
    ids, counts = handler.discover_counts(str(tmp_path), storage)

    assert ids.tolist() == sorted(shard_index)
    assert counts.tolist() == [shard_index[sid] for sid in sorted(shard_index)]


def test_discover_counts_agrees_with_discover_on_empty_index(tmp_path: Path) -> None:
    """An index with no shards fails count-only discovery just like ``discover``."""
    _write_index(tmp_path, [])
    handler = VortexFormat()
    storage = LocalFSBackend(tmp_path)

    with pytest.raises(ValueError, match="contains no shards"):
        handler.discover(str(tmp_path), storage)
    with pytest.raises(ValueError, match="contains no shards"):
        handler.discover_counts(str(tmp_path), storage)
