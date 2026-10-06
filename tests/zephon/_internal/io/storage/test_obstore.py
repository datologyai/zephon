"""Tests for ObstoreBackend's shared behavior against a real obstore store."""

import importlib.util
import sys
from typing import Any

import obstore as obs
import pytest
from obstore.store import MemoryStore

from zephon._internal.io.storage.azure import AzureBackend
from zephon._internal.io.storage.gcs import GCSBackend
from zephon._internal.io.storage.obstore import ObstoreBackend
from zephon._internal.io.storage.s3 import S3Backend
from zephon.io import Dataset


class _MemoryBackend(ObstoreBackend):
    valid_schemes = frozenset({"mem"})

    def __init__(self) -> None:
        super().__init__()
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


@pytest.fixture
def obstore_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
        return None if name == "obstore" else real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.delitem(sys.modules, "obstore", raising=False)


@pytest.mark.parametrize(
    ("backend_cls", "schemes"),
    [
        (S3Backend, "s3://"),
        (GCSBackend, "gcs://, gs://"),
        (AzureBackend, "abfs://, abfss://, az://, azure://"),
    ],
)
@pytest.mark.usefixtures("obstore_missing")
def test_missing_obstore_names_the_cloud_extra(backend_cls: type, schemes: str) -> None:
    with pytest.raises(
        ImportError, match=rf"Accessing {schemes} paths .*zephon\[cloud\]"
    ):
        backend_cls()


@pytest.mark.usefixtures("obstore_missing")
def test_missing_obstore_fails_dataset_discovery_with_install_hint() -> None:
    with pytest.raises(ImportError, match=r"s3://.*zephon\[cloud\]"):
        Dataset.from_path("web", "s3://example-bucket/web/")
