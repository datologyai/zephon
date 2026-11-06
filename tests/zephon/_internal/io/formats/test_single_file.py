# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shared single-file discovery and locator validation, without native readers."""

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from zephon._internal.io.formats.single_file import SingleFileFormat, shard_length
from zephon._internal.io.protocols import RandomAccessShard
from zephon._internal.io.storage import StorageBackend
from zephon._internal.io.storage.local import LocalFSBackend
from zephon._internal.io.types import LocalShardFile, LocalShardRef, ShardLocator
from zephon.io.dataset import Dataset


class StubFormat(SingleFileFormat):
    kind = "stub"
    label = "Stub"
    suffixes = (".stub",)

    def _read_row_count(self, path: str, storage: StorageBackend, size: int) -> int:
        raise AssertionError("unexpected shard read")

    def open_shard(
        self, locator: ShardLocator, local_ref: LocalShardRef
    ) -> RandomAccessShard:
        raise AssertionError("unexpected shard open")


@pytest.mark.parametrize(
    "update",
    [
        {"num_rows": None},
        {"num_rows": -1},
        {"num_rows": True},
        {"num_rows": 1 << 63},
        {"num_rows": "3"},
        {"bytes": -1},
        {"basename": "wrong.jsonl"},
        {"hashes": []},
        {"extra": None},
        {"extra": {"length": 2}},
    ],
)
def test_shared_index_validation(tmp_path: Path, update: dict[str, Any]) -> None:
    handler = StubFormat()
    shard = {
        "basename": "s" + handler.suffixes[0],
        "bytes": 100,
        "num_rows": 3,
        **update,
    }
    if update.get("num_rows", 0) is None:
        del shard["num_rows"]
    (tmp_path / "index.json").write_text(
        json.dumps({"format_version": 1, "shards": [shard]})
    )
    for method in (handler.discover, handler.discover_counts):
        with pytest.raises(ValueError, match="index"):
            method(str(tmp_path), LocalFSBackend(tmp_path))


def test_index_counts_and_metadata_agree_without_reading_shards(tmp_path: Path) -> None:
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "shards": [
                    {
                        "basename": "a.stub",
                        "bytes": 10,
                        "num_rows": 3,
                        "extra": {"source": "test"},
                    },
                    {"basename": "b.stub", "bytes": 20, "num_rows": 0},
                ],
            }
        )
    )
    handler = StubFormat()
    storage = LocalFSBackend(tmp_path)
    counts, metadata = handler.discover(".", storage)
    ids, values = handler.discover_counts(".", storage)
    assert counts == {0: 3, 1: 0}
    assert ids.tolist() == [0, 1]
    assert values.tolist() == [3, 0]
    assert metadata[0]["extra"] == {"source": "test", "length": 3}
    assert metadata[1]["extra"] == {"length": 0}


@pytest.mark.parametrize(
    "index", [{"format_version": 2, "shards": []}, {"format_version": 1, "shards": []}]
)
def test_rejects_invalid_index_header(tmp_path: Path, index: dict[str, Any]) -> None:
    (tmp_path / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="index"):
        StubFormat().discover(".", LocalFSBackend(tmp_path))


def test_scan_sorts_shards_and_passes_backend_paths_and_sizes(tmp_path: Path) -> None:
    (tmp_path / "b.stub").write_bytes(b"bb")
    (tmp_path / "a.stub").write_bytes(b"a")
    (tmp_path / "notes.txt").write_text("ignored")
    handler = StubFormat()
    storage = LocalFSBackend(tmp_path)
    with mock.patch.object(
        handler, "_read_row_count", side_effect=lambda path, backend, size: size + 2
    ) as read:
        counts, metadata = handler.discover(".", storage)
    assert counts == {0: 3, 1: 4}
    assert metadata[0]["raw"]["basename"] == "a.stub"
    assert metadata[0]["extra"] == {"length": 3}
    assert sorted(call.args[0] for call in read.call_args_list) == [
        "./a.stub",
        "./b.stub",
    ]
    assert all(call.args[1] is storage for call in read.call_args_list)
    assert sorted(call.args[2] for call in read.call_args_list) == [1, 2]


def test_scan_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No .stub shards"):
        StubFormat().discover(".", LocalFSBackend(tmp_path))


def test_scan_preserves_metadata_error_context(tmp_path: Path) -> None:
    (tmp_path / "a.stub").touch()
    handler = StubFormat()
    with mock.patch.object(handler, "_read_row_count", side_effect=OSError("missing")):
        with pytest.raises(ValueError, match="a.stub: missing") as error:
            handler.discover(".", LocalFSBackend(tmp_path))
    assert isinstance(error.value.__cause__, OSError)


@pytest.mark.parametrize(
    "bad_shards",
    [
        {0: {}},
        {0: {"raw": {"bytes": 1}}},
        {0: {"raw": {"basename": "s.stub"}}},
        {0: {"raw": {"basename": "s.stub", "bytes": object()}}},
    ],
)
def test_build_locators_rejects_bad_metadata(tmp_path: Path, bad_shards: Any) -> None:
    ds = Dataset(
        name="bad",
        backend={"kind": "stub", "path": str(tmp_path), "shards": bad_shards},
        path=str(tmp_path),
    )
    with pytest.raises(ValueError):
        StubFormat().build_locators(ds)


def test_build_locators_preserves_shard_metadata(tmp_path: Path) -> None:
    dataset = Dataset(
        name="ds",
        path=str(tmp_path),
        backend={
            "kind": "stub",
            "path": str(tmp_path),
            "shards": {
                4: {
                    "raw": {
                        "basename": "part.stub",
                        "bytes": 100,
                        "hashes": {"sha256": "abc"},
                    },
                    "extra": {"length": 3},
                }
            },
        },
    )
    locator = StubFormat().build_locators(dataset)[4]
    assert locator.dataset == "ds"
    assert locator.shard_id == 4
    assert locator.raw.basename == "part.stub"
    assert locator.raw.bytes == 100
    assert locator.raw.hashes == {"sha256": "abc"}
    assert locator.extra == {"length": 3}


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("4", 4), (4, 4), ("bad", None), (float("inf"), None)],
)
def test_optional_length_hint(tmp_path: Path, value: Any, expected: int | None) -> None:
    ref = LocalShardRef(
        raw=LocalShardFile(path=tmp_path / "s.stub", bytes=1), extra={"length": value}
    )
    assert shard_length(ref) == expected
