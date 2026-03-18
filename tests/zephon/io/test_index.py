"""Tests for the index utility module."""

from pathlib import Path

from zephon.io.index import find_and_load_index
from zephon.io.storage import RouterStorageBackend


def test_find_and_load_index_returns_data_when_exists(tmp_path: Path) -> None:
    """find_and_load_index returns parsed data when index.json exists."""
    index_data = {"shards": [{"samples": 5}]}
    (tmp_path / "index.json").write_text(
        '{"shards": [{"samples": 5}]}', encoding="utf-8"
    )
    storage = RouterStorageBackend()

    result = find_and_load_index(str(tmp_path), storage)

    assert result is not None
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
