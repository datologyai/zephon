"""Tests for ObstoreBackend's shared behavior against a real obstore store."""

from typing import Any

import obstore as obs
import pytest
from obstore.store import MemoryStore

from zephon._internal.io.storage.obstore import ObstoreBackend


class _MemoryBackend(ObstoreBackend):
    valid_schemes = frozenset({"mem"})

    def __init__(self) -> None:
        self.store = MemoryStore()

    def _get_store(self, bucket: str) -> Any:
        return self.store


def _keys(store: MemoryStore) -> list[str]:
    return sorted(obj["path"] for chunk in obs.list(store) for obj in chunk)


def test_mkdir_writes_no_marker_objects() -> None:
    backend = _MemoryBackend()

    backend.mkdir("mem://b/checkpoints/run", parents=True, exist_ok=True)
    backend.put("mem://b/checkpoints/run/state.json", b"{}")

    # A marker for ``checkpoints/`` would be stored as a file named
    # ``checkpoints``, since obstore strips the trailing slash.
    assert _keys(backend.store) == ["checkpoints/run/state.json"]


def test_mkdir_exist_ok_false_raises_only_for_non_empty_prefix() -> None:
    backend = _MemoryBackend()
    backend.mkdir("mem://b/fresh")
    backend.put("mem://b/used/a.json", b"{}")

    with pytest.raises(FileExistsError):
        backend.mkdir("mem://b/used")
    backend.mkdir("mem://b/used", exist_ok=True)
    # A sibling sharing the name as a prefix is not the directory.
    backend.mkdir("mem://b/use")


def test_mkdir_rejects_foreign_scheme() -> None:
    with pytest.raises(ValueError, match="Invalid path"):
        _MemoryBackend().mkdir("s3://b/dir")
