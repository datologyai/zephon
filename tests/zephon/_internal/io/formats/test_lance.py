# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Lance discovery, version pinning, refusals and projected shard reads."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pytest

from tests.helpers.storage import _install_obstore_stubs
from zephon._internal.io.formats.lance import LanceFormat, LanceShardOpener
from zephon._internal.io.formats.lance_support.manifest import (
    dataset_root_of,
    latest_manifest_path,
    manifest_names,
    manifest_version,
    read_manifest,
)
from zephon._internal.io.protocols import RandomAccessShard
from zephon._internal.io.resolvers.base import ShardResolver
from zephon._internal.io.storage import RouterStorageBackend
from zephon._internal.io.storage.local import LocalFSBackend
from zephon._internal.io.stores.multi import build_multi_dataset_store
from zephon._internal.io.stores.resilient import ResilientShard
from zephon._internal.io.types import LocalShardFile, LocalShardRef, ShardLocator
from zephon.io import CacheOptions, StoreOptions
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


Write = Callable[..., None]


# 2.0 lists a struct's own field id among the file's columns; 2.1+ only its leaves.
@pytest.fixture(params=["2.0", "2.1", None], ids=["v2.0", "v2.1", "default"])
def write(request: pytest.FixtureRequest) -> Write:
    def write(data: Any, path: Path, **kwargs: Any) -> None:
        if request.param is not None:
            kwargs.setdefault("data_storage_version", request.param)
        lance.write_dataset(data, str(path), **kwargs)

    return write


@pytest.fixture
def dataset_path(tmp_path: Path, pa: Any, write: Write) -> Path:
    path = tmp_path / "ds.lance"
    write(_table(pa, 10), path, max_rows_per_file=4)
    return path


class _Store:
    """Discovery plus a store-scoped opener, wired as the shard store wires them."""

    def __init__(self, path: Path, opener: LanceShardOpener | None = None) -> None:
        handler = LanceFormat()
        storage = LocalFSBackend(root=Path("/"))
        self.manifest = handler.resolve_root(str(path), storage)
        self.counts, meta = handler.discover(self.manifest, storage)
        self.locators = handler.build_locators(
            Dataset(
                name="ds",
                backend={"kind": "lance", "path": self.manifest, "shards": meta},
            )
        )
        self._data_root = Path(dataset_root_of(self.manifest)) / "data"
        # A real store shares one opener across every Lance dataset it serves.
        self._opener = opener or LanceShardOpener(
            StoreOptions(), RouterStorageBackend()
        )

    def open(self, shard_id: int) -> RandomAccessShard:
        """Open one shard from its cached data file, as the resolver would hand it over."""
        locator = self.locators[shard_id]
        raw = self._data_root / locator.raw.basename
        return self._opener.open_shard(
            locator,
            LocalShardRef(
                raw=LocalShardFile(path=raw, bytes=locator.raw.bytes),
                extra=locator.extra,
            ),
        )

    def rows(self, shard_ids: list[int] | None = None) -> list[dict]:
        """Read whole shards in shard order."""
        out: list[dict] = []
        for shard_id in self.counts if shard_ids is None else shard_ids:
            shard = self.open(shard_id)
            assert len(shard) == self.counts[shard_id]
            out.extend(shard.getsamples(list(range(len(shard)))))
            shard.close()
        return out


@contextmanager
def _store(path: Path) -> Iterator[_Store]:
    store = _Store(path)
    try:
        yield store
    finally:
        store._opener.close()


def _rows(path: Path, shard_ids: list[int] | None = None) -> list[dict]:
    with _store(path) as store:
        return store.rows(shard_ids)


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
    requested = [3, 0, 3, 1]

    with _store(dataset_path) as store:
        got = store.open(0).getsamples(requested)

    assert [_as_plain(row) for row in got] == [_as_plain(rows[i]) for i in requested]
    # Duplicate rows must not share a mutable container.
    got[0]["text"] = "changed"
    assert got[2]["text"] != "changed"


def test_one_shard_reads_from_several_threads(
    tmp_path: Path, pa: Any, write: Write
) -> None:
    import random
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "shared.lance"
    write(_table(pa, 200), path, max_rows_per_file=200)

    with _store(path) as store:

        def read(seed: int) -> bool:
            generator = random.Random(seed)
            for _ in range(20):
                wanted = generator.sample(range(200), 6)
                shard = store.open(0)
                rows = shard.getsamples(wanted)
                shard.close()
                if [int(row["id"]) for row in rows] != wanted:
                    return False
            return True

        with ThreadPoolExecutor(max_workers=8) as pool:
            assert all(pool.map(read, range(16)))


def test_an_evicted_data_file_is_not_served_from_a_retained_reader(
    dataset_path: Path,
) -> None:
    """An open reader keeps an unlinked file readable, hiding the eviction."""
    with _store(dataset_path) as store:
        shard = store.open(0)
        shard.getsamples([0])
        shard.close()
        (dataset_path / "data" / store.locators[0].raw.basename).unlink()

        with pytest.raises(FileNotFoundError):
            store.open(0)


def test_a_resilient_shard_retries_a_file_evicted_before_open(
    dataset_path: Path,
) -> None:
    """The eviction retry only catches OSError, which pylance does not raise."""
    with _store(dataset_path) as store:
        locator = store.locators[0]
        data_file = dataset_path / "data" / locator.raw.basename
        evicted = data_file.with_suffix(".evicted")

        class Resolver(ShardResolver):
            def __init__(self) -> None:
                self.calls = 0

            def resolve(
                self, locator: ShardLocator, *, blocking: bool = True
            ) -> LocalShardRef:
                self.calls += 1
                # The first resolution hands out a path the cache then evicts.
                if self.calls == 1:
                    data_file.rename(evicted)
                else:
                    evicted.rename(data_file)
                return LocalShardRef(
                    raw=LocalShardFile(path=data_file, bytes=locator.raw.bytes),
                    extra=locator.extra,
                )

            def touch(self, locator: ShardLocator) -> None:
                pass

        resolver = Resolver()
        shard = ResilientShard(
            locator=locator,
            resolver=resolver,
            opener=store._opener,
            length=store.counts[0],
            retry_attempts=2,
            retry_initial_backoff=0,
            retry_max_backoff=0,
        )

        rows, stats = shard.getsamples([1, 0])

        assert [row["id"] for row in rows] == [1, 0]
        assert all(item.retries == 1 for item in stats)
        assert resolver.calls == 2


def test_out_of_range_index_raises(dataset_path: Path) -> None:
    with _store(dataset_path) as store:
        shard = store.open(0)

        with pytest.raises(IndexError):
            shard.getsamples([len(shard)])


def test_deleted_rows_are_skipped(tmp_path: Path, pa: Any, write: Write) -> None:
    path = tmp_path / "deleted.lance"
    write(_table(pa, 10), path, max_rows_per_file=4)
    lance.dataset(str(path)).delete("id in (0, 2, 5, 9)")

    rows = _rows(path)

    assert [row["id"] for row in rows] == [1, 3, 4, 6, 7, 8]
    _assert_matches_pylance(path, rows)


class _RecordingBackend(LocalFSBackend):
    """Local storage that remembers which files were read through it."""

    def __init__(self) -> None:
        super().__init__(root=Path("/"))
        self.read: list[str] = []

    def open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        self.read.append(path)
        return super().open(path, mode, **kwargs)

    def read_range(self, path: str, start: int, **kwargs: Any) -> Any:
        self.read.append(path)
        return super().read_range(path, start, **kwargs)


@pytest.mark.parametrize("cache_enabled", [False, True], ids=["direct", "cached"])
def test_store_reads_manifest_and_tombstones_through_its_storage(
    tmp_path: Path, pa: Any, write: Write, cache_enabled: bool
) -> None:
    path = tmp_path / "deleted.lance"
    write(_table(pa, 10), path, max_rows_per_file=4)
    lance.dataset(str(path)).delete("id in (0, 2, 5, 9)")
    dataset = Dataset.from_path("ds", str(path))
    storage = _RecordingBackend()

    store = build_multi_dataset_store(
        {0: dataset},
        storage=storage,
        options=StoreOptions(
            cache=CacheOptions(enabled=cache_enabled, root=tmp_path / "cache")
        ),
    )
    try:
        shard, _ = store.for_dataset(0).open(0)
        rows, _stats = shard.getsamples([0, 1])
    finally:
        store.close()

    assert [row["id"] for row in rows] == [1, 3]
    assert dataset.path in storage.read
    assert any(f"{os.sep}_deletions{os.sep}" in read for read in storage.read)


def test_emptied_fragments_leave_the_shard_set(
    tmp_path: Path, pa: Any, write: Write, storage: LocalFSBackend
) -> None:
    """Lance drops a fragment whose rows are all deleted, so no shard is empty."""
    path = tmp_path / "gap.lance"
    write(_table(pa, 8), path, max_rows_per_file=4)
    lance.dataset(str(path)).delete("id < 4")

    counts, _meta = LanceFormat().discover(str(path), storage)

    assert list(counts.values()) == [4]
    _assert_matches_pylance(path, _rows(path))


def test_dense_deletions_use_a_bitmap_tombstone(
    tmp_path: Path, pa: Any, write: Write
) -> None:
    """Lance switches tombstone encodings by density; both must read the same."""
    path = tmp_path / "dense.lance"
    rows = 40000
    write(
        pa.table({"id": pa.array(range(rows), pa.int64())}),
        path,
        max_rows_per_file=rows,
    )
    lance.dataset(str(path)).delete("id % 2 = 0")
    fragment = lance.dataset(str(path)).get_fragments()[0]
    assert fragment.metadata.deletion_file.file_type == "bitmap"

    read = _rows(path)

    assert [row["id"] for row in read] == list(range(1, rows, 2))


def test_nested_columns_map_to_the_right_file_columns(
    tmp_path: Path, pa: Any, write: Write
) -> None:
    """A list or struct spans several schema fields but one column in the file."""
    path = tmp_path / "nested.lance"
    write(
        pa.table(
            {
                "tags": pa.array([["a", "b"], ["c"]], pa.list_(pa.string())),
                "pair": pa.array([{"x": 1, "y": "l"}, {"x": 2, "y": "r"}]),
                "id": pa.array([0, 1], pa.int64()),
            }
        ),
        path,
    )

    rows = _rows(path)

    assert [row["tags"] for row in rows] == [["a", "b"], ["c"]]
    assert [row["id"] for row in rows] == [0, 1]
    _assert_matches_pylance(path, rows)


def test_dropped_columns_are_refused(
    tmp_path: Path, pa: Any, write: Write, storage: LocalFSBackend
) -> None:
    path = tmp_path / "dropped.lance"
    write(_table(pa, 6), path, max_rows_per_file=6)
    lance.dataset(str(path)).drop_columns(["text"])

    with pytest.raises(ValueError, match="still stores a dropped column"):
        LanceFormat().discover(str(path), storage)


def test_compacting_a_lone_fragment_keeps_a_dropped_column(
    tmp_path: Path, pa: Any, write: Write, storage: LocalFSBackend
) -> None:
    """Compaction leaves a lone fragment's file as it is."""
    path = tmp_path / "dropped.lance"
    write(_table(pa, 6), path, max_rows_per_file=6)
    lance.dataset(str(path)).drop_columns(["text"])
    lance.dataset(str(path)).optimize.compact_files()

    with pytest.raises(ValueError, match="rewrite the dataset"):
        LanceFormat().discover(str(path), storage)


def test_rewriting_makes_a_dropped_column_dataset_readable(
    tmp_path: Path, pa: Any, write: Write
) -> None:
    path = tmp_path / "dropped.lance"
    write(_table(pa, 6), path, max_rows_per_file=6)
    lance.dataset(str(path)).drop_columns(["text"])
    rewritten = tmp_path / "rewritten.lance"
    write(lance.dataset(str(path)), rewritten)

    rows = _rows(rewritten)

    assert set(rows[0]) == {"id", "vec"}
    _assert_matches_pylance(rewritten, rows)


def test_renamed_columns_use_the_manifest_name(
    tmp_path: Path, pa: Any, write: Write
) -> None:
    """A rename updates the manifest only; the data file keeps the old name."""
    path = tmp_path / "renamed.lance"
    write(_table(pa, 6), path, max_rows_per_file=6)
    lance.dataset(str(path)).alter_columns({"path": "text", "name": "body"})

    rows = _rows(path)

    assert set(rows[0]) == {"id", "body", "vec"}
    _assert_matches_pylance(path, rows)


def test_two_pinned_versions_do_not_share_an_opened_fragment(
    tmp_path: Path, pa: Any, write: Write
) -> None:
    """Versions share unchanged data files, and without the cache, local paths."""
    path = tmp_path / "versions.lance"
    write(_table(pa, 4), path, max_rows_per_file=4)
    with _store(path) as old:
        assert set(old.rows()[0]) == {"id", "text", "vec"}
        lance.dataset(str(path)).alter_columns({"path": "text", "name": "body"})
        # A second dataset pinned to the newer version, served by the same
        # opener, reading the same unchanged file from the same path.
        new = _Store(path, opener=old._opener)
        assert new.manifest != old.manifest

        assert set(new.rows()[0]) == {"id", "body", "vec"}


def test_two_pinned_versions_do_not_share_a_deletion_vector(
    tmp_path: Path, pa: Any, write: Write
) -> None:
    """A delete leaves the data file alone, so both versions read one path."""
    path = tmp_path / "versions.lance"
    write(_table(pa, 4), path, max_rows_per_file=4)
    with _store(path) as old:
        assert [row["id"] for row in old.rows()] == [0, 1, 2, 3]
        lance.dataset(str(path)).delete("id = 1")

        new = _Store(path, opener=old._opener)

        assert [row["id"] for row in new.rows()] == [0, 2, 3]


def test_multi_file_fragments_are_refused(
    tmp_path: Path, pa: Any, write: Write
) -> None:
    path = tmp_path / "added.lance"
    write(_table(pa, 6), path, max_rows_per_file=3)
    lance.dataset(str(path)).add_columns({"doubled": "id * 2"})

    with pytest.raises(ValueError, match="spans 2 data files"):
        LanceFormat().discover(str(path), LocalFSBackend(root=Path("/")))


def test_compacting_makes_a_multi_file_dataset_readable(
    tmp_path: Path, pa: Any, write: Write
) -> None:
    path = tmp_path / "added.lance"
    write(_table(pa, 6), path, max_rows_per_file=3)
    lance.dataset(str(path)).add_columns({"doubled": "id * 2"})
    lance.dataset(str(path)).optimize.compact_files()

    rows = _rows(path)

    assert [row["doubled"] for row in rows] == [0, 2, 4, 6, 8, 10]
    _assert_matches_pylance(path, rows)


def test_blob_columns_are_refused(tmp_path: Path, pa: Any, write: Write) -> None:
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
    write(
        pa.table(
            {
                "id": pa.array(range(4), pa.int64()),
                "payload": pa.array([b"x" * 8] * 4, pa.large_binary()),
            },
            schema=schema,
        ),
        path,
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


def test_stable_row_ids_stay_out_of_the_data_columns(
    tmp_path: Path, pa: Any, write: Write, storage: LocalFSBackend
) -> None:
    """Lineage columns among the data columns would shift every position."""
    path = tmp_path / "stable.lance"
    write(_table(pa, 10), path, max_rows_per_file=4, enable_stable_row_ids=True)
    lance.dataset(str(path)).delete("id in (1, 6)")
    # Compaction moves rows, so Lance has to record their lineage.
    lance.dataset(str(path)).optimize.compact_files()
    manifest = read_manifest(storage, latest_manifest_path(storage, str(path)))
    assert manifest.reader_feature_flags & 2  # stable row ids
    assert all(fragment.deletion is None for fragment in manifest.fragments)

    rows = _rows(path)

    assert set(rows[0]) == {"id", "text", "vec"}
    _assert_matches_pylance(path, rows)


def test_discovery_through_object_storage_walks_a_stale_hint(
    tmp_path: Path, pa: Any, write: Write, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote discovery never downloads a data file."""
    import json
    import sys

    from zephon._internal.io.storage.s3 import S3Backend

    state = _install_obstore_stubs(monkeypatch)
    path = tmp_path / "remote.lance"
    write(_table(pa, 10), path, max_rows_per_file=4)
    lance.write_dataset(
        _table(pa, 6, start=100), str(path), mode="append", max_rows_per_file=4
    )
    # As left by a writer that crashed before updating the hint.
    (path / "_versions" / "latest_version_hint.json").write_text(
        json.dumps({"version": 1})
    )
    for file in path.rglob("*"):
        if file.is_file():
            key = f"ds.lance/{file.relative_to(path).as_posix()}"
            state["objects"][("bucket", key)] = file.read_bytes()

    probed: list[str] = []
    obstore = sys.modules["obstore"]
    head = obstore.head

    def recording_head(store: Any, key: str) -> Any:
        probed.append(key)
        return head(store, key)

    monkeypatch.setattr(obstore, "head", recording_head)

    backend = S3Backend()
    download = backend.download

    def small_downloads_only(src: str, dst: str, timeout: float | None = None) -> None:
        if not src.endswith(".json"):
            pytest.fail(f"discovery downloaded {src}")
        download(src, dst, timeout)

    monkeypatch.setattr(backend, "download", small_downloads_only)

    handler = LanceFormat()
    manifest = handler.resolve_root("s3://bucket/ds.lance", backend)
    counts, meta = handler.discover(manifest, backend)

    # Whichever naming scheme this pylance writes.
    v2 = next(
        name for name in os.listdir(path / "_versions") if manifest_version(name) == 2
    )
    v3 = manifest_names(3)
    assert manifest == f"s3://bucket/ds.lance/_versions/{v2}"
    assert sum(counts.values()) == 16
    # Walked past the hint, stopping at the first missing version.
    assert f"ds.lance/_versions/{v2}" in probed
    assert all(f"ds.lance/_versions/{name}" in probed for name in v3)
    ranged = {call["key"] for call in state["range_calls"]}
    assert ranged == {f"ds.lance/_versions/{v2}"}
    assert {shard["raw"]["basename"] for shard in meta.values()} <= {
        key.rsplit("/", 1)[1]
        for _bucket, key in state["objects"]
        if key.startswith("ds.lance/data/")
    }
