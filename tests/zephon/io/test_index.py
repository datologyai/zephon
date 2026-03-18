"""Tests for the index utility module."""

import json
from pathlib import Path

import pytest

from zephon.io.index import (
    find_and_load_index,
    is_litdata_index,
    is_mds_index,
    is_shard_index,
)
from zephon.io.storage import RouterStorageBackend


def test_find_and_load_index_returns_data_when_exists(tmp_path: Path) -> None:
    """find_and_load_index returns parsed data when index.json exists (ShardIndex)."""
    index_data = {
        "format_version": 1,
        "shards": [
            {
                "basename": "part-00000.parquet",
                "bytes": 1024,
                "num_rows": 100,
                "hashes": {},
                "extra": {},
            }
        ],
    }
    (tmp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")
    storage = RouterStorageBackend()

    result = find_and_load_index(str(tmp_path), storage)

    assert result is not None
    assert is_shard_index(result)
    assert result == index_data


def test_find_and_load_index_returns_none_when_absent(tmp_path: Path) -> None:
    """find_and_load_index returns None when no candidate file exists."""
    storage = RouterStorageBackend()

    result = find_and_load_index(str(tmp_path), storage)

    assert result is None


def test_find_and_load_index_uses_index_json_when_both_exist(tmp_path: Path) -> None:
    """find_and_load_index prefers index.json over _index.json when both exist."""
    (tmp_path / "index.json").write_text(
        '{"shards": [{"samples": 1}]}', encoding="utf-8"
    )
    (tmp_path / "_index.json").write_text(
        '{"shards": [{"samples": 2}]}', encoding="utf-8"
    )
    storage = RouterStorageBackend()

    result = find_and_load_index(str(tmp_path), storage)

    assert result is not None
    assert result["shards"][0]["samples"] == 1


def test_find_and_load_index_uses_underscore_index_when_only_it_exists(
    tmp_path: Path,
) -> None:
    """find_and_load_index loads _index.json when index.json is absent."""
    index_data = {"shards": [{"samples": 3}]}
    (tmp_path / "_index.json").write_text(
        '{"shards": [{"samples": 3}]}', encoding="utf-8"
    )
    storage = RouterStorageBackend()

    result = find_and_load_index(str(tmp_path), storage)

    assert result is not None
    assert result == index_data


def test_find_and_load_index_mds_index(tmp_path: Path) -> None:
    """find_and_load_index returns parsed data for MdsIndex format."""
    index_data = {"shards": [{"raw_data": "foo.bin", "samples": 42}]}
    (tmp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")
    storage = RouterStorageBackend()

    result = find_and_load_index(str(tmp_path), storage)

    assert result is not None
    assert is_mds_index(result)
    assert result == index_data


def test_find_and_load_index_litdata_index(tmp_path: Path) -> None:
    """find_and_load_index returns parsed data for LitDataIndex format."""
    index_data = {"config": {"version": 1}, "chunks": [{"filename": "chunk_0.bin"}]}
    (tmp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")
    storage = RouterStorageBackend()

    result = find_and_load_index(str(tmp_path), storage)

    assert result is not None
    assert is_litdata_index(result)
    assert result == index_data


def test_find_and_load_index_raises_on_invalid_format(tmp_path: Path) -> None:
    """find_and_load_index raises ValueError when index does not match any known schema."""
    (tmp_path / "index.json").write_text(
        '{"unknown": "structure", "foo": 123}', encoding="utf-8"
    )
    storage = RouterStorageBackend()

    with pytest.raises(ValueError, match="Invalid index format"):
        find_and_load_index(str(tmp_path), storage)
