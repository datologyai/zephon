"""Unit tests for LitData Arrow footer detection and decoding."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("optree")

from tests.helpers.litdata_chunks import write_litdata_fixture
from zephon._internal.io.formats.litdata_support import arrow
from zephon._internal.io.formats.litdata_support.arrow import (
    ArrowLoader,
    arrow_footer_span,
)
from zephon._internal.io.formats.litdata_support.support import FlatPyTree


def _open_loader(root: Path) -> tuple[ArrowLoader, Path]:
    index = json.loads((root / "index.json").read_text())
    loader = ArrowLoader()
    loader.setup(index["config"], index["chunks"], None)
    return loader, root / index["chunks"][0]["filename"]


def _load_rows(loader: ArrowLoader, path: Path, indices: list[int]) -> list[Any]:
    return loader.load_items_from_chunk(indices, 0, str(path), 0, path.stat().st_size)


@pytest.mark.parametrize("ipc_compression", [None, "zstd"])
@pytest.mark.parametrize("flat", [False, True])
def test_arrow_nested_rows_are_independent(
    tmp_path: Path, ipc_compression: str | None, flat: bool
) -> None:
    pytest.importorskip("pyarrow")
    rows = [
        {
            "text": {"content": "zero"},
            "images": b"\x00\xff",
            "values": [1, 2],
            "optional": None,
        },
        {
            "text": {"content": "one"},
            "images": b"\x01",
            "values": [],
            "optional": "present",
        },
        {
            "text": {"content": "two"},
            "images": b"\x02",
            "values": [3],
            "optional": None,
        },
    ]
    write_litdata_fixture(
        tmp_path,
        rows,
        layout="file",
        ipc_compression=ipc_compression,
    )
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text())
    index["config"]["return_flat_leaves"] = flat
    index_path.write_text(json.dumps(index))
    loader, path = _open_loader(tmp_path)
    result = _load_rows(loader, path, [2, 0, 2])
    if flat:
        assert all(isinstance(row, FlatPyTree) for row in result)
        result = [row.materialize() for row in result]
    assert result == [rows[i] for i in [2, 0, 2]]
    result[0]["values"].append(99)
    result[0]["text"]["content"] = "changed"
    assert result[2] == rows[2]
    again = loader.load_item_from_chunk(2, 0, str(path), 0, path.stat().st_size)
    assert (again.materialize() if flat else again) == rows[2]
    loader.close(0)
    loader.close(0)
    assert result[1] == rows[0]


def test_arrow_missing_pyarrow_has_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pyarrow")
    write_litdata_fixture(tmp_path, [{"id": 0, "text": "row"}], layout="file")

    def missing() -> None:
        raise ImportError("No module named pyarrow")

    monkeypatch.setattr(arrow, "require_pyarrow", missing)
    loader, path = _open_loader(tmp_path)
    try:
        with pytest.raises(
            ImportError, match="Arrow-backed LitData chunks require pyarrow"
        ):
            _load_rows(loader, path, [0])
    finally:
        loader.close(0)


@pytest.mark.parametrize("size", [0, 9999])
def test_invalid_arrow_footer_length(tmp_path: Path, size: int) -> None:
    path = tmp_path / "chunk.bin"
    path.write_bytes(b"payload" + struct.pack("<I", size) + b"LDARW01\0")
    with pytest.raises(ValueError, match="Invalid LitData Arrow footer length"):
        arrow_footer_span(path)


@pytest.mark.parametrize("layout", ["file", "stream"])
@pytest.mark.parametrize("declared_rows", [1, 3])
def test_arrow_row_count_must_match_index(
    tmp_path: Path, layout: str, declared_rows: int
) -> None:
    pytest.importorskip("pyarrow")
    rows = [{"id": i, "text": "row"} for i in range(2)]
    write_litdata_fixture(tmp_path, rows, layout=layout)
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text())
    index["chunks"][0]["chunk_size"] = declared_rows
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="row count.*chunk_size"):
        loader, path = _open_loader(tmp_path)
        try:
            _load_rows(loader, path, [0])
        finally:
            loader.close(0)


def test_arrow_loader_reuses_files_with_global_intervals(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    paths = []
    chunks = []
    expected = []
    for group in range(2):
        rows = [{"id": group * 10 + i, "text": f"row-{group}-{i}"} for i in range(3)]
        path = write_litdata_fixture(tmp_path / str(group), rows, layout="file")
        paths.append(path)
        chunks.append({"chunk_size": len(rows), "chunk_bytes": path.stat().st_size})
        expected.append(rows)
    loader = ArrowLoader()
    loader.setup({}, chunks, None)
    try:
        for group in [0, 1, 0]:
            begin = 3 * group
            rows = loader.load_items_from_chunk(
                [begin + i for i in [2, 0, 2]],
                group,
                str(paths[group]),
                begin,
                chunks[group]["chunk_bytes"],
            )
            assert rows == [expected[group][i] for i in [2, 0, 2]]
    finally:
        loader.close(0)


@pytest.mark.parametrize("layout", ["file", "stream", "hybrid"])
def test_arrow_loader_reads_layout(tmp_path: Path, layout: str) -> None:
    pytest.importorskip("pyarrow")
    rows = [{"id": i, "text": f"row-{i}"} for i in range(600)]
    write_litdata_fixture(
        tmp_path,
        rows,
        layout=layout,
        batch_sizes=(2, 0, 597, 1) if layout == "stream" else None,
    )
    loader, path = _open_loader(tmp_path)
    try:
        indices = [599, 0, 255, 256, 511, 512, 599, 1]
        assert _load_rows(loader, path, indices) == [rows[i] for i in indices]
        assert _load_rows(loader, path, []) == []
        assert (
            loader.load_item_from_chunk(3, 0, str(path), 0, path.stat().st_size)
            == rows[3]
        )
    finally:
        loader.close(0)
    loader.close(0)


@pytest.mark.parametrize("ipc_compression", [None, "zstd"])
def test_arrow_reopen_decodes_only_requested_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ipc_compression: str | None
) -> None:
    pa = pytest.importorskip("pyarrow")
    rows = [{"id": i, "text": f"row-{i}"} for i in range(2000)]
    write_litdata_fixture(
        tmp_path, rows, layout="file", ipc_compression=ipc_compression
    )
    decoded = []
    open_file = pa.ipc.open_file

    class TrackedReader:
        def __init__(self, reader: Any) -> None:
            self.reader = reader
            self.num_record_batches = reader.num_record_batches

        def get_batch(self, index: int) -> Any:
            decoded.append(index)
            return self.reader.get_batch(index)

    def tracked_open_file(source: Any) -> TrackedReader:
        return TrackedReader(open_file(source))

    monkeypatch.setattr(pa.ipc, "open_file", tracked_open_file)
    indices = [1792, 1793, 256, 511, 1792, 1999, 512]
    # The store constructs a fresh shard for each read, so caches cannot hide
    # prefix decoding or double decoding across these requests.
    for _ in range(2):
        decoded.clear()
        loader, path = _open_loader(tmp_path)
        try:
            assert _load_rows(loader, path, indices) == [rows[i] for i in indices]
            assert decoded == [7, 1, 2]
        finally:
            loader.close(0)


def test_arrow_rejects_unexpected_batch_count(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    rows = [{"id": i, "text": f"row-{i}"} for i in range(6)]
    write_litdata_fixture(tmp_path, rows, layout="file", batch_sizes=(2, 4))
    loader, path = _open_loader(tmp_path)
    try:
        with pytest.raises(ValueError, match="batch count 2.*expected 1"):
            _load_rows(loader, path, [0])
    finally:
        loader.close(0)


@pytest.mark.parametrize("index, expected_rows", [(0, 256), (256, 256), (512, 88)])
def test_arrow_validates_each_decoded_batch(
    tmp_path: Path, index: int, expected_rows: int
) -> None:
    pytest.importorskip("pyarrow")
    rows = [{"id": i, "text": f"row-{i}"} for i in range(600)]
    write_litdata_fixture(tmp_path, rows, layout="file", batch_sizes=(255, 258, 87))
    loader, path = _open_loader(tmp_path)
    try:
        with pytest.raises(ValueError, match=f"expected {expected_rows} for 256-row"):
            _load_rows(loader, path, [index])
    finally:
        loader.close(0)


def test_arrow_footer_rejects_incomplete_file_before_sniffing(tmp_path: Path) -> None:
    path = tmp_path / "chunk.bin"
    path.write_bytes(b"truncated")
    with pytest.raises(FileNotFoundError, match="not found or incomplete"):
        arrow_footer_span(path, filesize_bytes=100)
