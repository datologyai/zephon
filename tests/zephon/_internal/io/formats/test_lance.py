# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Lance discovery, version pinning, refusals and projected shard reads."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from zephon._internal.io.formats.lance import LanceFormat, LanceShardOpener
from zephon._internal.io.formats.lance_manifest import (
    dataset_root_of,
    latest_manifest_path,
)
from zephon._internal.io.storage.local import LocalFSBackend
from zephon._internal.io.types import LocalShardFile, LocalShardRef
from zephon.io import StoreOptions
from zephon.io.dataset import Dataset

pytest.importorskip("lance", reason="pylance not installed")
import lance  # noqa: E402


@pytest.fixture
def storage() -> LocalFSBackend:
    return LocalFSBackend(root=Path("/"))


@pytest.fixture
def pa() -> Any:
    return pytest.importorskip("pyarrow")


def _table(pa: Any, count: int, start: int = 0) -> Any:
    return pa.table(
        {
            "id": pa.array(range(start, start + count), pa.int64()),
            "text": pa.array([f"row-{i}" for i in range(start, start + count)]),
            "vec": pa.array(
                [[float(i), float(i + 1)] for i in range(start, start + count)],
                pa.list_(pa.float32(), 2),
            ),
        }
    )


@pytest.fixture
def dataset_path(tmp_path: Path, pa: Any) -> Path:
    path = tmp_path / "ds.lance"
    lance.write_dataset(_table(pa, 10), str(path), max_rows_per_file=4)
    return path


def _rows(path: Path, shard_ids: list[int] | None = None) -> list[dict]:
    """Read every shard of ``path`` through the format handler, in shard order."""
    handler = LanceFormat()
    storage = LocalFSBackend(root=Path("/"))
    manifest = handler.resolve_root(str(path), storage)
    counts, meta = handler.discover(manifest, storage)
    data_root = Path(dataset_root_of(manifest)) / "data"
    dataset = Dataset(
        name="ds", backend={"kind": "lance", "path": manifest, "shards": meta}
    )
    locators = handler.build_locators(dataset)
    opener = LanceShardOpener(StoreOptions())
    out: list[dict] = []
    try:
        for shard_id in shard_ids if shard_ids is not None else sorted(counts):
            locator = locators[shard_id]
            raw = data_root / locator.raw.basename
            shard = opener.open_shard(
                locator,
                LocalShardRef(
                    raw=LocalShardFile(path=raw, bytes=raw.stat().st_size),
                    extra=locator.extra,
                ),
            )
            assert len(shard) == counts[shard_id]
            out.extend(shard.getsamples(list(range(len(shard)))))
            shard.close()
    finally:
        opener.close()
    return out


def _data_file_path(data_file: Any) -> str:
    # pylance turned DataFile.path from a method into an attribute.
    path = data_file.path
    return path() if callable(path) else path


def _as_plain(row: dict) -> dict:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in row.items()
    }


def _assert_matches_pylance(path: Path, rows: list[dict]) -> None:
    expected = lance.dataset(str(path)).to_table().to_pylist()
    assert [_as_plain(row) for row in rows] == expected


def test_discovery_matches_pylance_fragments(
    dataset_path: Path, storage: LocalFSBackend
) -> None:
    counts, meta = LanceFormat().discover(str(dataset_path), storage)
    fragments = lance.dataset(str(dataset_path)).get_fragments()

    assert list(counts.values()) == [f.count_rows() for f in fragments]
    assert [meta[i]["raw"]["basename"] for i in sorted(meta)] == [
        _data_file_path(f.data_files()[0]) for f in fragments
    ]


def test_discovery_needs_no_pylance(
    dataset_path: Path, storage: LocalFSBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest is the index, so discovery reads it without the decoder."""
    monkeypatch.setattr("zephon._internal.io.formats.lance._lance_file", None)

    counts, _meta = LanceFormat().discover(str(dataset_path), storage)

    assert sum(counts.values()) == 10


def test_shard_bytes_come_from_the_manifest(
    dataset_path: Path, storage: LocalFSBackend
) -> None:
    _counts, meta = LanceFormat().discover(str(dataset_path), storage)

    for shard in meta.values():
        actual = (dataset_path / "data" / shard["raw"]["basename"]).stat().st_size
        assert shard["raw"]["bytes"] == actual


def test_reads_match_pylance(dataset_path: Path) -> None:
    _assert_matches_pylance(dataset_path, _rows(dataset_path))


def test_auto_detection_and_version_pinning(dataset_path: Path, pa: Any) -> None:
    first = Dataset.from_path("ds", str(dataset_path))
    assert first.backend["kind"] == "lance"
    assert first.path is not None and first.path.endswith(".manifest")
    assert first.total() == 10

    lance.write_dataset(
        _table(pa, 6, start=100), str(dataset_path), mode="append", max_rows_per_file=4
    )

    # The pinned manifest keeps naming the version it was resolved at.
    assert Dataset.from_path("ds", first.path).total() == 10
    assert Dataset.from_path("ds", str(dataset_path)).total() == 16


def test_catalog_carries_the_pinned_manifest_to_every_shard(dataset_path: Path) -> None:
    """Shards resolve under ``data/`` even though the catalog is keyed by version."""
    dataset = Dataset.from_path("ds", str(dataset_path))
    assert dataset._catalog_handle is not None
    catalog = dataset._catalog_handle.ensure_attached()

    assert catalog.root == str(dataset_path / "data")
    assert catalog.num_rows().tolist() == dataset.counts().tolist()
    # The opener rebuilds its read plan from this path in every worker.
    for slot in range(catalog.shard_count):
        extra = catalog.locator_at(slot, dataset_name="ds").extra
        assert extra is not None and extra["manifest"] == dataset.path


def test_row_order_and_duplicates_are_preserved(dataset_path: Path) -> None:
    rows = _rows(dataset_path, shard_ids=[0])
    handler = LanceFormat()
    storage = LocalFSBackend(root=Path("/"))
    manifest = handler.resolve_root(str(dataset_path), storage)
    _counts, meta = handler.discover(manifest, storage)
    dataset = Dataset(
        name="ds", backend={"kind": "lance", "path": manifest, "shards": meta}
    )
    locator = handler.build_locators(dataset)[0]
    raw = dataset_path / "data" / locator.raw.basename
    opener = LanceShardOpener(StoreOptions())
    try:
        shard = opener.open_shard(
            locator,
            LocalShardRef(
                raw=LocalShardFile(path=raw, bytes=raw.stat().st_size),
                extra=locator.extra,
            ),
        )
        requested = [3, 0, 3, 1]

        got = shard.getsamples(requested)

        assert [_as_plain(row) for row in got] == [
            _as_plain(rows[i]) for i in requested
        ]
        # Duplicate rows must not share a mutable container.
        got[0]["text"] = "changed"
        assert got[2]["text"] != "changed"
    finally:
        opener.close()


def test_one_shard_reads_from_several_threads(tmp_path: Path, pa: Any) -> None:
    """A retained reader is shared across fetch threads; pylance rejects reentry."""
    import random
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "shared.lance"
    lance.write_dataset(_table(pa, 200), str(path), max_rows_per_file=200)
    handler = LanceFormat()
    storage = LocalFSBackend(root=Path("/"))
    manifest = handler.resolve_root(str(path), storage)
    _counts, meta = handler.discover(manifest, storage)
    dataset = Dataset(
        name="ds", backend={"kind": "lance", "path": manifest, "shards": meta}
    )
    locator = handler.build_locators(dataset)[0]
    raw = path / "data" / locator.raw.basename
    local_ref = LocalShardRef(
        raw=LocalShardFile(path=raw, bytes=raw.stat().st_size), extra=locator.extra
    )
    opener = LanceShardOpener(StoreOptions())

    def read(seed: int) -> bool:
        generator = random.Random(seed)
        for _ in range(20):
            wanted = generator.sample(range(200), 6)
            shard = opener.open_shard(locator, local_ref)
            rows = shard.getsamples(wanted)
            shard.close()
            if [int(row["id"]) for row in rows] != wanted:
                return False
        return True

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            assert all(pool.map(read, range(16)))
    finally:
        opener.close()


def test_out_of_range_index_raises(dataset_path: Path) -> None:
    handler = LanceFormat()
    storage = LocalFSBackend(root=Path("/"))
    manifest = handler.resolve_root(str(dataset_path), storage)
    _counts, meta = handler.discover(manifest, storage)
    dataset = Dataset(
        name="ds", backend={"kind": "lance", "path": manifest, "shards": meta}
    )
    locator = handler.build_locators(dataset)[0]
    raw = dataset_path / "data" / locator.raw.basename
    opener = LanceShardOpener(StoreOptions())
    try:
        shard = opener.open_shard(
            locator,
            LocalShardRef(
                raw=LocalShardFile(path=raw, bytes=raw.stat().st_size),
                extra=locator.extra,
            ),
        )
        with pytest.raises(IndexError):
            shard.getsamples([len(shard)])
    finally:
        opener.close()


def test_deleted_rows_are_skipped(tmp_path: Path, pa: Any) -> None:
    path = tmp_path / "deleted.lance"
    lance.write_dataset(_table(pa, 10), str(path), max_rows_per_file=4)
    lance.dataset(str(path)).delete("id in (0, 2, 5, 9)")

    rows = _rows(path)

    assert [row["id"] for row in rows] == [1, 3, 4, 6, 7, 8]
    _assert_matches_pylance(path, rows)


def test_emptied_fragments_leave_the_shard_set(tmp_path: Path, pa: Any) -> None:
    """Lance drops a fragment whose rows are all deleted, so no shard is empty."""
    path = tmp_path / "gap.lance"
    lance.write_dataset(_table(pa, 8), str(path), max_rows_per_file=4)
    lance.dataset(str(path)).delete("id < 4")

    counts, _meta = LanceFormat().discover(str(path), LocalFSBackend(root=Path("/")))

    assert list(counts.values()) == [4]
    _assert_matches_pylance(path, _rows(path))


def test_dense_deletions_use_a_bitmap_tombstone(tmp_path: Path, pa: Any) -> None:
    """Lance switches tombstone encodings by density; both must read the same."""
    path = tmp_path / "dense.lance"
    rows = 40000
    lance.write_dataset(
        pa.table({"id": pa.array(range(rows), pa.int64())}),
        str(path),
        max_rows_per_file=rows,
    )
    lance.dataset(str(path)).delete("id % 2 = 0")
    fragment = lance.dataset(str(path)).get_fragments()[0]
    assert fragment.metadata.deletion_file.file_type == "bitmap"

    read = _rows(path)

    assert [row["id"] for row in read] == list(range(1, rows, 2))


def test_dropped_columns_are_not_read(tmp_path: Path, pa: Any) -> None:
    path = tmp_path / "dropped.lance"
    lance.write_dataset(_table(pa, 6), str(path), max_rows_per_file=6)
    lance.dataset(str(path)).drop_columns(["text"])

    rows = _rows(path)

    assert set(rows[0]) == {"id", "vec"}
    _assert_matches_pylance(path, rows)


def test_nested_columns_map_to_the_right_file_columns(tmp_path: Path, pa: Any) -> None:
    """A list or struct spans several schema fields but one column in the file."""
    path = tmp_path / "nested.lance"
    lance.write_dataset(
        pa.table(
            {
                "tags": pa.array([["a", "b"], ["c"]], pa.list_(pa.string())),
                "pair": pa.array([{"x": 1, "y": "l"}, {"x": 2, "y": "r"}]),
                "id": pa.array([0, 1], pa.int64()),
            }
        ),
        str(path),
    )

    rows = _rows(path)

    assert [row["tags"] for row in rows] == [["a", "b"], ["c"]]
    assert [row["id"] for row in rows] == [0, 1]
    _assert_matches_pylance(path, rows)


def test_a_dropped_nested_column_does_not_shift_the_rest(
    tmp_path: Path, pa: Any
) -> None:
    """The file keeps the dropped column's columns, so positions must skip them."""
    path = tmp_path / "nested_drop.lance"
    lance.write_dataset(
        pa.table(
            {
                "tags": pa.array([["a", "b"], ["c"]], pa.list_(pa.string())),
                "pair": pa.array([{"x": 1, "y": "l"}, {"x": 2, "y": "r"}]),
                "id": pa.array([0, 1], pa.int64()),
            }
        ),
        str(path),
    )
    lance.dataset(str(path)).drop_columns(["pair"])

    rows = _rows(path)

    assert set(rows[0]) == {"tags", "id"}
    _assert_matches_pylance(path, rows)


def test_renamed_columns_use_the_manifest_name(tmp_path: Path, pa: Any) -> None:
    """A rename updates the manifest only; the data file keeps the old name."""
    path = tmp_path / "renamed.lance"
    lance.write_dataset(_table(pa, 6), str(path), max_rows_per_file=6)
    lance.dataset(str(path)).alter_columns({"path": "text", "name": "body"})

    rows = _rows(path)

    assert set(rows[0]) == {"id", "body", "vec"}
    _assert_matches_pylance(path, rows)


def test_multi_file_fragments_are_refused(tmp_path: Path, pa: Any) -> None:
    path = tmp_path / "added.lance"
    lance.write_dataset(_table(pa, 6), str(path), max_rows_per_file=3)
    lance.dataset(str(path)).add_columns({"doubled": "id * 2"})

    with pytest.raises(ValueError, match="spans 2 data files"):
        LanceFormat().discover(str(path), LocalFSBackend(root=Path("/")))


def test_compacting_makes_a_multi_file_dataset_readable(
    tmp_path: Path, pa: Any
) -> None:
    path = tmp_path / "added.lance"
    lance.write_dataset(_table(pa, 6), str(path), max_rows_per_file=3)
    lance.dataset(str(path)).add_columns({"doubled": "id * 2"})
    lance.dataset(str(path)).optimize.compact_files()

    rows = _rows(path)

    assert [row["doubled"] for row in rows] == [0, 2, 4, 6, 8, 10]
    _assert_matches_pylance(path, rows)


def test_blob_columns_are_refused(tmp_path: Path, pa: Any) -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field(
                "payload",
                pa.large_binary(),
                metadata={"lance-encoding:blob": "true"},
            ),
        ]
    )
    path = tmp_path / "blob.lance"
    lance.write_dataset(
        pa.table(
            {
                "id": pa.array(range(4), pa.int64()),
                "payload": pa.array([b"x" * 8] * 4, pa.large_binary()),
            },
            schema=schema,
        ),
        str(path),
    )

    with pytest.raises(ValueError, match="blob column"):
        LanceFormat().discover(str(path), LocalFSBackend(root=Path("/")))


def test_unknown_reader_feature_flags_are_refused(
    dataset_path: Path, storage: LocalFSBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zephon._internal.io.formats import lance as lance_format

    manifest_path = latest_manifest_path(storage, str(dataset_path))
    original = lance_format.read_manifest

    def with_future_flag(backend: Any, path: str) -> Any:
        import dataclasses

        return dataclasses.replace(
            original(backend, path), reader_feature_flags=1 << 40
        )

    monkeypatch.setattr(lance_format, "read_manifest", with_future_flag)

    with pytest.raises(ValueError, match="reader feature flags"):
        LanceFormat().discover(manifest_path, storage)


def test_known_but_irrelevant_flags_are_accepted(
    dataset_path: Path, storage: LocalFSBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stable row ids and table config do not change a positional read."""
    from zephon._internal.io.formats import lance as lance_format

    manifest_path = latest_manifest_path(storage, str(dataset_path))
    original = lance_format.read_manifest

    def with_known_flags(backend: Any, path: str) -> Any:
        import dataclasses

        return dataclasses.replace(original(backend, path), reader_feature_flags=2 | 8)

    monkeypatch.setattr(lance_format, "read_manifest", with_known_flags)

    counts, _meta = LanceFormat().discover(manifest_path, storage)

    assert sum(counts.values()) == 10
