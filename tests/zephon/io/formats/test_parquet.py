# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for Parquet format support."""

import json
from io import BytesIO
from pathlib import Path

import pytest

# Check if pyarrow is available
pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq

from tests._helpers import catalog_locators
from tests.helpers.storage import _install_obstore_stubs
from zephon.io.catalog import extra_codec
from zephon.io.dataset import Dataset
from zephon.io.formats.parquet import (
    ParquetFormat,
    ParquetShard,
    _arrow_table_to_numpy,
    _extract_row,
    _ParquetExtraCodec,
    _RowGroupCache,
)
from zephon.io.index.parquet_index import ParquetIndexBuilder
from zephon.io.storage import LocalFSBackend
from zephon.io.storage.router import RouterStorageBackend
from zephon.io.types import LocalShardFile, LocalShardRef, ShardFile, ShardLocator


def create_test_parquet_file(path: Path, num_rows: int, row_group_size: int) -> None:
    """Create a test Parquet file with multiple row groups.

    Args:
        path: Output file path
        num_rows: Total number of rows
        row_group_size: Rows per row group
    """
    table = pa.table(
        {
            "id": pa.array(range(num_rows)),
            "text": pa.array([f"item_{i}" for i in range(num_rows)]),
            "value": pa.array([i * 2 for i in range(num_rows)]),
        }
    )
    pq.write_table(table, str(path), row_group_size=row_group_size)


@pytest.fixture
def parquet_dataset_dir(tmp_path):
    """Create a test Parquet dataset with multiple shards."""
    dataset_dir = tmp_path / "parquet_dataset"
    dataset_dir.mkdir()

    # Create 3 parquet files with different sizes
    create_test_parquet_file(
        dataset_dir / "data_000.parquet", num_rows=10000, row_group_size=2000
    )
    create_test_parquet_file(
        dataset_dir / "data_001.parquet", num_rows=8000, row_group_size=2000
    )
    create_test_parquet_file(
        dataset_dir / "data_002.parquet", num_rows=12000, row_group_size=3000
    )

    return dataset_dir


@pytest.fixture
def parquet_dataset_with_index(parquet_dataset_dir):
    """Create a Parquet dataset with index.json."""
    builder = ParquetIndexBuilder()
    builder.create_index(parquet_dataset_dir, progress=False)
    return parquet_dataset_dir


class TestParquetIndexCreation:
    """Tests for the preprocessing tool."""

    def test_create_index(self, parquet_dataset_dir):
        """Test index.json creation."""
        builder = ParquetIndexBuilder()
        builder.create_index(parquet_dataset_dir, progress=False)

        index_path = parquet_dataset_dir / "index.json"
        assert index_path.exists()

        with open(index_path) as f:
            data = json.load(f)

        assert data["format_version"] == 1
        assert len(data["shards"]) == 3

        # Check first shard (using IndexBuilder format)
        shard0 = data["shards"][0]
        assert shard0["basename"] == "data_000.parquet"
        assert shard0["num_rows"] == 10000
        assert shard0["extra"]["num_row_groups"] == 5  # 10000 / 2000
        assert len(shard0["extra"]["row_groups"]) == 5
        assert all("num_rows" in rg for rg in shard0["extra"]["row_groups"])

    def test_create_index_no_files(self, tmp_path):
        """Test error when no Parquet files found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        builder = ParquetIndexBuilder()
        with pytest.raises(ValueError, match="No \\*.parquet files found"):
            builder.create_index(empty_dir, progress=False)


class TestParquetFormatDiscovery:
    """Tests for Parquet format discovery."""

    def test_discover_from_index(self, parquet_dataset_with_index):
        """Test fast discovery from index.json."""
        format_handler = ParquetFormat()
        storage = LocalFSBackend(root=Path("/"))

        shard_index, shard_meta = format_handler.discover(
            str(parquet_dataset_with_index), storage
        )

        assert len(shard_index) == 3
        assert shard_index[0] == 10000
        assert shard_index[1] == 8000
        assert shard_index[2] == 12000

        # Check metadata
        assert 0 in shard_meta
        meta0 = shard_meta[0]
        assert meta0["raw"]["basename"] == "data_000.parquet"
        assert meta0["extra"]["num_rows"] == 10000
        assert meta0["extra"]["num_row_groups"] == 5
        assert len(meta0["extra"]["row_groups"]) == 5

    def test_discover_from_files_fallback(self, parquet_dataset_dir):
        """Test fallback discovery by reading Parquet metadata directly."""
        format_handler = ParquetFormat()
        storage = LocalFSBackend(root=Path("/"))

        # No index.json, should fall back to direct reads
        shard_index, shard_meta = format_handler.discover(
            str(parquet_dataset_dir), storage
        )

        assert len(shard_index) == 3
        assert shard_index[0] == 10000
        assert shard_index[1] == 8000
        assert shard_index[2] == 12000

        # Metadata should be the same as index-based discovery
        meta0 = shard_meta[0]
        assert "row_groups" in meta0["extra"]

    def test_discover_counts_matches_discover_with_index(
        self, parquet_dataset_with_index
    ):
        """Index path: counts agree with ``discover`` on ``(shard_id, num_rows)``."""
        format_handler = ParquetFormat()
        storage = LocalFSBackend(root=Path("/"))

        shard_index, _ = format_handler.discover(
            str(parquet_dataset_with_index), storage
        )
        ids, counts = format_handler.discover_counts(
            str(parquet_dataset_with_index), storage
        )

        assert ids.tolist() == sorted(shard_index)
        assert counts.tolist() == [shard_index[sid] for sid in sorted(shard_index)]

    def test_discover_counts_matches_discover_without_index(self, parquet_dataset_dir):
        """No index: the footer-scan fallback still agrees with ``discover``."""
        format_handler = ParquetFormat()
        storage = LocalFSBackend(root=Path("/"))

        shard_index, _ = format_handler.discover(str(parquet_dataset_dir), storage)
        ids, counts = format_handler.discover_counts(str(parquet_dataset_dir), storage)

        assert ids.tolist() == sorted(shard_index)
        assert counts.tolist() == [10000, 8000, 12000]

    def test_discover_from_files_s3_does_not_download_full_object(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test cloud discovery does not rely on full-object downloads."""
        state = _install_obstore_stubs(monkeypatch)

        dataset_dir = tmp_path / "parquet_dataset"
        dataset_dir.mkdir()
        shard_path = dataset_dir / "data_000.parquet"
        create_test_parquet_file(shard_path, num_rows=1000, row_group_size=250)
        state["objects"][("bucket", "dataset/data_000.parquet")] = (
            shard_path.read_bytes()
        )

        from zephon.io.storage.s3 import S3Backend

        backend = S3Backend()

        def fail_download(src: str, dst: str, timeout: float | None = None) -> None:
            del src, dst, timeout
            pytest.fail("discover() performed a full-object download")

        backend.download = fail_download  # type: ignore[assignment]

        format_handler = ParquetFormat()
        shard_index, shard_meta = format_handler.discover(
            "s3://bucket/dataset", backend
        )

        assert shard_index == {0: 1000}
        assert shard_meta[0]["raw"]["basename"] == "data_000.parquet"
        assert shard_meta[0]["extra"]["num_row_groups"] == 4

    def test_discover_from_files_router_falls_back_without_read_range(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        class _S3NoRangeBackend:
            def __init__(self, object_data: bytes) -> None:
                self._object_data = object_data

            def exists(self, path: str) -> bool:
                del path
                return False

            def listdir(self, path: str) -> list[str]:
                assert path == "s3://bucket/dataset"
                return ["data_000.parquet"]

            def stat(self, path: str) -> dict[str, int]:
                assert path == "s3://bucket/dataset/data_000.parquet"
                return {"size": len(self._object_data)}

            def open(self, path: str, mode: str = "rb", **kwargs):
                del kwargs
                assert path == "s3://bucket/dataset/data_000.parquet"
                assert mode == "rb"
                return BytesIO(self._object_data)

        dataset_dir = tmp_path / "parquet_dataset"
        dataset_dir.mkdir()
        shard_path = dataset_dir / "data_000.parquet"
        create_test_parquet_file(shard_path, num_rows=1000, row_group_size=250)

        backend = _S3NoRangeBackend(shard_path.read_bytes())
        monkeypatch.setattr(
            "zephon.io.storage.router._make_s3_backend", lambda: backend
        )

        router = RouterStorageBackend()
        format_handler = ParquetFormat()
        shard_index, shard_meta = format_handler.discover("s3://bucket/dataset", router)

        assert shard_index == {0: 1000}
        assert shard_meta[0]["raw"]["basename"] == "data_000.parquet"
        assert shard_meta[0]["extra"]["num_row_groups"] == 4

    def test_discover_no_files(self, tmp_path):
        """Test error when no Parquet files found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        format_handler = ParquetFormat()
        storage = LocalFSBackend(root=Path("/"))

        with pytest.raises(ValueError, match="No .parquet shards found"):
            format_handler.discover(str(empty_dir), storage)


class TestParquetShard:
    """Tests for ParquetShard random access and bulk reads."""

    @pytest.fixture
    def shard_file(self, tmp_path):
        """Create a single Parquet file for testing."""
        shard_path = tmp_path / "test_shard.parquet"
        create_test_parquet_file(shard_path, num_rows=10000, row_group_size=2000)
        return shard_path

    @pytest.fixture
    def shard_metadata(self, shard_file):
        """Get metadata for the test shard."""
        metadata = pq.read_metadata(str(shard_file))
        row_groups = [
            {
                "num_rows": metadata.row_group(i).num_rows,
                "total_byte_size": metadata.row_group(i).total_byte_size,
            }
            for i in range(metadata.num_row_groups)
        ]
        return row_groups

    def test_shard_random_access(self, shard_file, shard_metadata):
        """Test single-record random access."""
        row_groups = shard_metadata
        shard = ParquetShard(shard_file, row_groups, rg_cache=_RowGroupCache())

        assert len(shard) == 10000

        # Test accessing first row
        row0 = shard[0]
        assert row0["id"] == 0
        assert row0["text"] == "item_0"
        assert row0["value"] == 0

        # Test accessing middle row
        row5000 = shard[5000]
        assert row5000["id"] == 5000
        assert row5000["text"] == "item_5000"
        assert row5000["value"] == 10000

        # Test accessing last row
        row9999 = shard[9999]
        assert row9999["id"] == 9999
        assert row9999["text"] == "item_9999"
        assert row9999["value"] == 19998

    def test_shard_row_group_boundaries(self, shard_file, shard_metadata):
        """Test accessing rows at row group boundaries."""
        row_groups = shard_metadata
        shard = ParquetShard(shard_file, row_groups, rg_cache=_RowGroupCache())

        # First row of each row group (row_group_size = 2000)
        for rg_idx in range(5):
            first_idx = rg_idx * 2000
            row = shard[first_idx]
            assert row["id"] == first_idx

        # Last row of each row group
        for rg_idx in range(5):
            last_idx = (rg_idx + 1) * 2000 - 1
            row = shard[last_idx]
            assert row["id"] == last_idx

    def test_shard_out_of_bounds(self, shard_file, shard_metadata):
        """Test accessing out-of-bounds indices."""
        row_groups = shard_metadata
        shard = ParquetShard(shard_file, row_groups, rg_cache=_RowGroupCache())

        with pytest.raises(IndexError):
            _ = shard[-1]

        with pytest.raises(IndexError):
            _ = shard[10000]

        with pytest.raises(IndexError):
            _ = shard[10001]

    def test_shard_getsamples_bulk(self, shard_file, shard_metadata):
        """Test bulk read with getsamples()."""
        row_groups = shard_metadata
        shard = ParquetShard(shard_file, row_groups, rg_cache=_RowGroupCache())

        # Bulk read spanning multiple row groups
        indices = [0, 100, 2000, 2001, 5000, 7500, 9999]
        rows = shard.getsamples(indices)

        assert len(rows) == len(indices)
        for i, idx in enumerate(indices):
            assert rows[i]["id"] == idx
            assert rows[i]["text"] == f"item_{idx}"
            assert rows[i]["value"] == idx * 2

    def test_shard_getsamples_order_preservation(self, shard_file, shard_metadata):
        """Test that getsamples() preserves input order."""
        row_groups = shard_metadata
        shard = ParquetShard(shard_file, row_groups, rg_cache=_RowGroupCache())

        # Random order
        indices = [9999, 0, 5000, 100, 7500]
        rows = shard.getsamples(indices)

        assert len(rows) == len(indices)
        for i, idx in enumerate(indices):
            assert rows[i]["id"] == idx

    def test_shard_getsamples_duplicates(self, shard_file, shard_metadata):
        """Test that getsamples() handles duplicate indices."""
        row_groups = shard_metadata
        shard = ParquetShard(shard_file, row_groups, rg_cache=_RowGroupCache())

        # Include duplicates
        indices = [100, 100, 200, 100, 300]
        rows = shard.getsamples(indices)

        assert len(rows) == len(indices)
        # Each duplicate should return the same row
        assert rows[0]["id"] == 100
        assert rows[1]["id"] == 100
        assert rows[2]["id"] == 200
        assert rows[3]["id"] == 100
        assert rows[4]["id"] == 300

    def test_shard_getsamples_empty(self, shard_file, shard_metadata):
        """Test getsamples() with empty list."""
        row_groups = shard_metadata
        shard = ParquetShard(shard_file, row_groups, rg_cache=_RowGroupCache())

        rows = shard.getsamples([])
        assert rows == []

    def test_shard_getsamples_out_of_bounds(self, shard_file, shard_metadata):
        """Test getsamples() with out-of-bounds indices."""
        row_groups = shard_metadata
        shard = ParquetShard(shard_file, row_groups, rg_cache=_RowGroupCache())

        with pytest.raises(IndexError):
            shard.getsamples([0, 100, 10000])

        with pytest.raises(IndexError):
            shard.getsamples([0, -1, 100])


class TestParquetMetadataCaching:
    """Tests for cached metadata reuse during shard opens."""

    def test_open_shard_reuses_cached_metadata(self, tmp_path: Path) -> None:
        shard_path = tmp_path / "cached.parquet"
        create_test_parquet_file(shard_path, num_rows=200, row_group_size=20)

        metadata = pq.read_metadata(str(shard_path))
        row_groups = [
            {
                "num_rows": metadata.row_group(i).num_rows,
                "total_byte_size": metadata.row_group(i).total_byte_size,
            }
            for i in range(metadata.num_row_groups)
        ]
        format_handler = ParquetFormat()
        local_ref = LocalShardRef(
            raw=LocalShardFile(path=shard_path, bytes=shard_path.stat().st_size),
            extra={"row_groups": row_groups},
        )
        locator = ShardLocator(
            dataset="test",
            shard_id=0,
            format="parquet",
            root=str(tmp_path),
            raw=ShardFile(
                basename=shard_path.name,
                bytes=shard_path.stat().st_size,
                hashes={},
            ),
            extra={"row_groups": row_groups},
        )

        calls = 0
        real_read_metadata = pq.read_metadata

        def counting_read_metadata(path_arg):
            nonlocal calls
            calls += 1
            return real_read_metadata(path_arg)

        original = pq.read_metadata
        pq.read_metadata = counting_read_metadata
        try:
            shard1 = format_handler.open_shard(locator, local_ref)
            shard1.close()
            shard2 = format_handler.open_shard(locator, local_ref)
            shard2.close()
        finally:
            pq.read_metadata = original

        assert calls == 1


class TestArrowToNumpy:
    """Tests for _arrow_table_to_numpy conversion with various Arrow types."""

    def test_scalar_columns(self):
        """Scalar int/float columns produce 1-D numpy arrays."""
        table = pa.table({"x": pa.array([1, 2, 3]), "y": pa.array([1.5, 2.5, 3.5])})
        result, decoded_bytes = _arrow_table_to_numpy(table)

        assert set(result.keys()) == {"x", "y"}
        import numpy as np

        np.testing.assert_array_equal(result["x"], [1, 2, 3])
        np.testing.assert_array_equal(result["y"], [1.5, 2.5, 3.5])
        assert result["x"].shape == (3,)
        assert decoded_bytes == int(table.nbytes)

    def test_fixed_size_list_column(self):
        """fixed_size_list<uint32>[N] is reshaped to (n_rows, N)."""
        import numpy as np

        inner = pa.array([10, 20, 30, 40, 50, 60], type=pa.uint32())
        fsl = pa.FixedSizeListArray.from_arrays(inner, list_size=3)
        table = pa.table({"tokens": fsl})
        result, _ = _arrow_table_to_numpy(table)

        assert result["tokens"].shape == (2, 3)
        np.testing.assert_array_equal(result["tokens"][0], [10, 20, 30])
        np.testing.assert_array_equal(result["tokens"][1], [40, 50, 60])

    def test_variable_length_list_column(self):
        """Variable-length list columns produce an object array of numpy arrays."""
        import numpy as np

        list_arr = pa.array([[1, 2], [3, 4, 5], [6]], type=pa.list_(pa.int64()))
        table = pa.table({"ragged": list_arr})
        result, _ = _arrow_table_to_numpy(table)

        assert result["ragged"].dtype == object
        assert len(result["ragged"]) == 3
        np.testing.assert_array_equal(result["ragged"][0], [1, 2])
        np.testing.assert_array_equal(result["ragged"][1], [3, 4, 5])
        np.testing.assert_array_equal(result["ragged"][2], [6])

    def test_string_column_fallback(self):
        """String columns fall back to object dtype numpy array."""

        table = pa.table({"s": pa.array(["hello", "world"])})
        result, _ = _arrow_table_to_numpy(table)

        assert len(result["s"]) == 2
        assert result["s"][0] == "hello"

    def test_extract_row_from_mixed_table(self):
        """_extract_row returns a dict with correct per-column values."""
        import numpy as np

        inner = pa.array(list(range(12)), type=pa.uint32())
        fsl = pa.FixedSizeListArray.from_arrays(inner, list_size=4)
        table = pa.table({"id": pa.array([10, 20, 30]), "tokens": fsl})
        columns, _ = _arrow_table_to_numpy(table)

        row = _extract_row(columns, 1)
        assert row["id"] == 20
        np.testing.assert_array_equal(row["tokens"], [4, 5, 6, 7])

    def test_roundtrip_through_parquet_with_fixed_size_list(self, tmp_path):
        """End-to-end: write parquet with fixed_size_list, read via ParquetShard."""
        import numpy as np

        n_rows = 100
        list_size = 8
        flat = list(range(n_rows * list_size))
        inner = pa.array(flat, type=pa.uint32())
        fsl = pa.FixedSizeListArray.from_arrays(inner, list_size=list_size)
        table = pa.table({"id": pa.array(range(n_rows)), "tokens": fsl})

        path = tmp_path / "fsl.parquet"
        pq.write_table(table, str(path), row_group_size=50)

        metadata = pq.read_metadata(str(path))
        row_groups = [
            {
                "num_rows": metadata.row_group(i).num_rows,
                "total_byte_size": metadata.row_group(i).total_byte_size,
            }
            for i in range(metadata.num_row_groups)
        ]

        shard = ParquetShard(path, row_groups, rg_cache=_RowGroupCache())

        # Single access
        row = shard[0]
        assert row["id"] == 0
        np.testing.assert_array_equal(row["tokens"], list(range(list_size)))

        # Bulk access across row groups
        rows = shard.getsamples([0, 49, 50, 99])
        assert rows[0]["id"] == 0
        assert rows[1]["id"] == 49
        assert rows[2]["id"] == 50
        assert rows[3]["id"] == 99
        np.testing.assert_array_equal(
            rows[2]["tokens"], list(range(50 * list_size, 51 * list_size))
        )


class TestRowGroupCache:
    """Tests for row group caching and file descriptor management."""

    @pytest.fixture
    def shard_file(self, tmp_path):
        """Create a single Parquet file with 5 row groups."""
        shard_path = tmp_path / "test_cache.parquet"
        create_test_parquet_file(shard_path, num_rows=10000, row_group_size=2000)
        return shard_path

    @pytest.fixture
    def shard_metadata(self, shard_file):
        metadata = pq.read_metadata(str(shard_file))
        return [
            {
                "num_rows": metadata.row_group(i).num_rows,
                "total_byte_size": metadata.row_group(i).total_byte_size,
            }
            for i in range(metadata.num_row_groups)
        ]

    def test_cache_hit_avoids_file_open(self, shard_file, shard_metadata):
        """Reading the same row group twice should only open the file once."""
        cache = _RowGroupCache()
        shard = ParquetShard(shard_file, shard_metadata, rg_cache=cache)

        open_calls = 0
        real_parquet_file = pq.ParquetFile

        def counting_open(*args, **kwargs):
            nonlocal open_calls
            open_calls += 1
            return real_parquet_file(*args, **kwargs)

        original = pq.ParquetFile
        pq.ParquetFile = counting_open
        try:
            # First read: cache miss, opens file
            _ = shard[0]
            assert open_calls == 1

            # Second read from same row group: cache hit, no file open
            _ = shard[1]
            assert open_calls == 1

            # Read from different row group: cache miss, opens file again
            _ = shard[2000]
            assert open_calls == 2

            # Re-read from first row group: still cached
            _ = shard[100]
            assert open_calls == 2
        finally:
            pq.ParquetFile = original

    def test_lru_eviction(self, shard_file, shard_metadata):
        """Cache should evict LRU entries when RAM cap is exceeded."""
        # Read one row group to measure its size, then set cap to fit exactly one.
        probe = _RowGroupCache()
        probe_shard = ParquetShard(shard_file, shard_metadata, rg_cache=probe)
        _ = probe_shard[0]
        one_rg_bytes = probe.used_bytes
        assert one_rg_bytes > 0

        # Cap = just over one row group so a second evicts the first.
        cache = _RowGroupCache(max_bytes=one_rg_bytes + 1)
        shard = ParquetShard(shard_file, shard_metadata, rg_cache=cache)

        _ = shard[0]  # rg 0 — fits
        assert len(cache) == 1

        _ = shard[2000]  # rg 1 — exceeds cap, rg 0 evicted
        assert len(cache) == 1
        assert cache.get(str(shard_file), 0) is None
        assert cache.get(str(shard_file), 1) is not None

    def test_shared_cache_across_shard_instances(self, shard_file, shard_metadata):
        """Cache should persist data across ParquetShard open/close cycles."""
        cache = _RowGroupCache()

        open_calls = 0
        real_parquet_file = pq.ParquetFile

        def counting_open(*args, **kwargs):
            nonlocal open_calls
            open_calls += 1
            return real_parquet_file(*args, **kwargs)

        original = pq.ParquetFile
        pq.ParquetFile = counting_open
        try:
            # First shard instance reads row group 0
            shard1 = ParquetShard(shard_file, shard_metadata, rg_cache=cache)
            _ = shard1[0]
            shard1.close()
            assert open_calls == 1

            # Second shard instance reads same row group: cache hit
            shard2 = ParquetShard(shard_file, shard_metadata, rg_cache=cache)
            _ = shard2[0]
            shard2.close()
            assert open_calls == 1  # no additional file open
        finally:
            pq.ParquetFile = original

    def test_no_persistent_file_descriptor(self, shard_file, shard_metadata):
        """ParquetShard should not hold any persistent file handle."""
        cache = _RowGroupCache()
        shard = ParquetShard(shard_file, shard_metadata, rg_cache=cache)

        # No _pq_file attribute should exist
        assert not hasattr(shard, "_pq_file")

        # Read some data
        _ = shard[0]
        _ = shard.getsamples([0, 2000, 4000])

        # Still no persistent file handle
        assert not hasattr(shard, "_pq_file")

    def test_getsamples_uses_cache(self, shard_file, shard_metadata):
        """getsamples should use the cache for repeated row group access."""
        cache = _RowGroupCache()

        open_calls = 0
        real_parquet_file = pq.ParquetFile

        def counting_open(*args, **kwargs):
            nonlocal open_calls
            open_calls += 1
            return real_parquet_file(*args, **kwargs)

        original = pq.ParquetFile
        pq.ParquetFile = counting_open
        try:
            shard = ParquetShard(shard_file, shard_metadata, rg_cache=cache)

            # First getsamples: opens file for each row group accessed
            _ = shard.getsamples([0, 2000, 4000])
            first_calls = open_calls

            # Second getsamples with same row groups: all cache hits
            _ = shard.getsamples([1, 2001, 4001])
            assert open_calls == first_calls  # no additional opens
        finally:
            pq.ParquetFile = original

    def test_format_handler_injects_shared_cache(self, tmp_path):
        """ParquetFormat.open_shard should inject its shared cache."""
        shard_path = tmp_path / "shared.parquet"
        create_test_parquet_file(shard_path, num_rows=200, row_group_size=20)

        metadata = pq.read_metadata(str(shard_path))
        row_groups = [
            {
                "num_rows": metadata.row_group(i).num_rows,
                "total_byte_size": metadata.row_group(i).total_byte_size,
            }
            for i in range(metadata.num_row_groups)
        ]

        format_handler = ParquetFormat()
        local_ref = LocalShardRef(
            raw=LocalShardFile(path=shard_path, bytes=shard_path.stat().st_size),
            extra={"row_groups": row_groups},
        )
        locator = ShardLocator(
            dataset="test",
            shard_id=0,
            format="parquet",
            root=str(tmp_path),
            raw=ShardFile(
                basename=shard_path.name,
                bytes=shard_path.stat().st_size,
                hashes={},
            ),
            extra={"row_groups": row_groups},
        )

        shard1 = format_handler.open_shard(locator, local_ref)
        shard2 = format_handler.open_shard(locator, local_ref)

        # Both shards should share the same cache instance
        assert shard1._rg_cache is shard2._rg_cache
        assert shard1._rg_cache is format_handler._rg_cache

    def test_low_memory_mode(self, shard_file, shard_metadata):
        """max_bytes=0 disables caching: every read opens the file."""
        cache = _RowGroupCache(max_bytes=0)
        shard = ParquetShard(shard_file, shard_metadata, rg_cache=cache)

        open_calls = 0
        real_parquet_file = pq.ParquetFile

        def counting_open(*args, **kwargs):
            nonlocal open_calls
            open_calls += 1
            return real_parquet_file(*args, **kwargs)

        original = pq.ParquetFile
        pq.ParquetFile = counting_open
        try:
            _ = shard[0]
            _ = shard[1]  # same row group, but not cached
            assert open_calls == 2
            assert len(cache) == 0
        finally:
            pq.ParquetFile = original

    def test_ram_cap_eviction(self, shard_file, shard_metadata):
        """used_bytes tracks actual numpy buffer sizes."""
        cache = _RowGroupCache()
        shard = ParquetShard(shard_file, shard_metadata, rg_cache=cache)

        _ = shard[0]
        assert cache.used_bytes > 0

        before = cache.used_bytes
        _ = shard[2000]  # second row group
        assert cache.used_bytes > before


def _create_struct_parquet_file(
    path: Path, num_rows: int, row_group_size: int, content_bytes: int
) -> int:
    """Create a Parquet file whose ``text`` column is a struct with a heavy
    ``content`` field, mimicking the IngestedDocument shape that triggered
    the byte-accounting bug. Returns the table's Arrow ``nbytes`` so callers
    can size cache caps relative to the actual decoded payload.
    """
    payload = "x" * content_bytes
    table = pa.table(
        {
            "id": pa.array(range(num_rows), type=pa.int64()),
            "text": pa.array(
                [{"content": payload, "hash": f"h{i}"} for i in range(num_rows)],
                type=pa.struct([("content", pa.string()), ("hash", pa.string())]),
            ),
        }
    )
    pq.write_table(table, str(path), row_group_size=row_group_size)
    return int(table.nbytes)


class TestRowGroupCacheAccounting:
    """Regression tests for parquet row-group LRU cache fixes."""

    def _struct_shard(self, tmp_path, *, num_rows, row_group_size, content_bytes):
        path = tmp_path / "struct.parquet"
        decoded = _create_struct_parquet_file(
            path,
            num_rows=num_rows,
            row_group_size=row_group_size,
            content_bytes=content_bytes,
        )
        metadata = pq.read_metadata(str(path))
        shard_meta = [
            {
                "num_rows": metadata.row_group(i).num_rows,
                "total_byte_size": metadata.row_group(i).total_byte_size,
            }
            for i in range(metadata.num_row_groups)
        ]
        return path, shard_meta, decoded

    def test_used_bytes_counts_struct_payload(self, tmp_path):
        """Cache ``used_bytes`` must reflect struct-column payload
        bytes, not just the numpy object-pointer table.

        Before the fix, an object-dtype numpy array reported
        ``nbytes == len(arr) * 8``, so a 100-row row group with 100 KiB of
        text per row (~10 MiB decoded) was booked as ~800 bytes. After the
        fix we use Arrow ``table.nbytes`` which counts the underlying
        buffer bytes.
        """
        path, shard_meta, decoded_bytes = self._struct_shard(
            tmp_path, num_rows=100, row_group_size=100, content_bytes=100 * 1024
        )
        # Sanity check: the single row group really is ~10 MiB on the wire.
        assert decoded_bytes >= 9_500_000, decoded_bytes

        cache = _RowGroupCache()
        shard = ParquetShard(path, shard_meta, rg_cache=cache)
        _ = shard[0]

        # Pre-fix this would be ~1 KiB. Post-fix it must be within an order
        # of magnitude of the Arrow decoded size.
        assert cache.used_bytes >= 5_000_000, (
            f"used_bytes={cache.used_bytes} too small for a "
            f"~10 MiB struct row group; accounting still broken"
        )

    def test_lru_eviction_with_struct_columns(self, tmp_path):
        """eviction cap actually fires for struct-dtype
        row groups. Pre-fix the cap was meaningless for these workloads
        because ``used_bytes`` under-counted by ~1000x.
        """
        # 5 row groups of ~5 MiB each; cap ≈ 7 MiB so a second insert
        # always evicts the previous one.
        path, shard_meta, _ = self._struct_shard(
            tmp_path, num_rows=250, row_group_size=50, content_bytes=100 * 1024
        )
        cache = _RowGroupCache(max_bytes=7 * 1024 * 1024)
        shard = ParquetShard(path, shard_meta, rg_cache=cache)

        _ = shard[0]
        assert len(cache) == 1
        first_used = cache.used_bytes

        _ = shard[50]  # next row group
        _ = shard[100]
        _ = shard[150]
        # Cap is below 2x one row group, so eviction must keep us at 1 entry.
        assert len(cache) == 1, (
            f"expected cap to evict down to 1 entry, got len={len(cache)}, "
            f"used_bytes={cache.used_bytes}, first_used={first_used}"
        )
        # And the most recently inserted row group is still there.
        assert cache.get(str(path), 3) is not None

    def test_getsamples_without_row_group_cache(self, tmp_path):
        """When max_bytes=0, put is a no-op. getsamples must still work without it."""
        path, shard_meta, _ = self._struct_shard(
            tmp_path, num_rows=200, row_group_size=50, content_bytes=4 * 1024
        )
        cache = _RowGroupCache(max_bytes=0)
        shard = ParquetShard(path, shard_meta, rg_cache=cache)

        # Indices spanning multiple row groups, in non-monotonic order.
        results = shard.getsamples([0, 51, 199, 100, 1, 150])
        assert len(results) == 6
        assert results[0]["id"] == 0
        assert results[1]["id"] == 51
        assert results[2]["id"] == 199
        assert results[3]["id"] == 100
        assert results[4]["id"] == 1
        assert results[5]["id"] == 150
        # Cache must still be empty (max_bytes=0 ⇒ put is a no-op).
        assert len(cache) == 0
        assert cache.used_bytes == 0

    def test_put_same_key_does_not_double_count(self, tmp_path):
        """Re-``put``ing the same ``(path, rg_id)`` must not double-count
        ``used_bytes``. Guards the dedup branch in ``_RowGroupCache.put``.
        """
        path, shard_meta, _ = self._struct_shard(
            tmp_path, num_rows=50, row_group_size=50, content_bytes=1024
        )
        cache = _RowGroupCache()
        shard = ParquetShard(path, shard_meta, rg_cache=cache)

        _ = shard[0]
        used_after_first = cache.used_bytes
        len_after_first = len(cache)
        assert used_after_first > 0

        # Re-insert the same row group directly. Without the dedup branch
        # this would double the booked usage and add a second entry.
        path_key = str(path)
        cached = cache.get(path_key, 0)
        assert cached is not None
        cache.put(path_key, 0, cached, used_after_first)

        assert cache.used_bytes == used_after_first
        assert len(cache) == len_after_first

    def test_extract_row_returns_independent_dict(self, tmp_path):
        """Mutating a returned struct value must not corrupt the
        cached row group. Pre-fix the returned dict was the same Python
        object that lived in the cache's numpy object array, so any
        downstream mutation (or downstream-held reference outliving cache
        eviction) created a "shadow set" of pinned dicts.
        """
        path, shard_meta, _ = self._struct_shard(
            tmp_path, num_rows=50, row_group_size=50, content_bytes=512
        )
        cache = _RowGroupCache()
        shard = ParquetShard(path, shard_meta, rg_cache=cache)

        first = shard[0]
        original_content = first["text"]["content"]

        # Mutating the returned dict's nested struct field must NOT affect
        # the cache's view of the row.
        first["text"]["content"] = "MUTATED"
        first["text"]["hash"] = "MUTATED"

        second = shard[0]  # served from cache
        assert second["text"]["content"] == original_content
        assert second["text"]["hash"] != "MUTATED"
        # And the two reads are not the same Python object identity.
        assert first["text"] is not second["text"]


class TestParquetAutoDetection:
    """Tests for auto-detection of Parquet datasets."""

    def test_auto_detect_with_index(self, parquet_dataset_with_index):
        """Test auto-detection with index.json."""
        dataset = Dataset.from_path(
            name="test_dataset", path=str(parquet_dataset_with_index)
        )

        assert dataset.backend["kind"] == "parquet"
        assert dataset.shard_count() == 3
        assert dataset.total() == 30000  # 10000 + 8000 + 12000

    def test_auto_detect_without_index(self, parquet_dataset_dir):
        """Test auto-detection without index.json (fallback)."""
        dataset = Dataset.from_path(name="test_dataset", path=str(parquet_dataset_dir))

        assert dataset.backend["kind"] == "parquet"
        assert dataset.shard_count() == 3
        assert dataset.total() == 30000

    def test_explicit_format_specification(self, parquet_dataset_dir):
        """Test explicitly specifying format='parquet'."""
        dataset = Dataset.from_path(
            name="test_dataset", path=str(parquet_dataset_dir), fmt="parquet"
        )

        assert dataset.backend["kind"] == "parquet"


class TestParquetIntegration:
    """Integration tests with full pipeline."""

    def test_end_to_end_reading(self, parquet_dataset_with_index):
        """Test complete workflow from Dataset creation to reading."""
        dataset = Dataset.from_path(
            name="test_dataset", path=str(parquet_dataset_with_index)
        )

        # Verify dataset structure
        assert dataset.shard_count() == 3
        total_samples = dataset.total()
        assert total_samples == 30000

        # Note: Full integration with FetchOp/Pipeline would require
        # more complex setup with engine and workers. This test verifies
        # the dataset structure is correct for such integration.


class TestParquetExtraCodec:
    """Catalog round-trip of the parquet row-group metadata (``extra``)."""

    def _expected_extras(self, dataset_dir: Path) -> dict[int, dict[str, object]]:
        """Ground truth straight from the parquet footers, keyed by shard id.

        Shard ids follow sorted-basename order (the discovery contract), so this
        is independent of the discovery/locator code under test.
        """
        out: dict[int, dict[str, object]] = {}
        names = sorted(p.name for p in dataset_dir.glob("*.parquet"))
        for shard_id, name in enumerate(names):
            md = pq.ParquetFile(str(dataset_dir / name)).metadata
            out[shard_id] = {
                "num_rows": md.num_rows,
                "num_row_groups": md.num_row_groups,
                "row_groups": [
                    {
                        "num_rows": md.row_group(i).num_rows,
                        "total_byte_size": md.row_group(i).total_byte_size,
                    }
                    for i in range(md.num_row_groups)
                ],
            }
        return out

    def _synthesized_extras(
        self, dataset_dir: Path, name: str
    ) -> dict[int, dict[str, object]]:
        dataset = Dataset.from_path(name=name, path=str(dataset_dir))
        _, locators = catalog_locators(dataset)
        return {
            sid: dict(loc.extra) if loc.extra else {} for sid, loc in locators.items()
        }

    def test_catalog_extra_matches_parquet_footers(self, parquet_dataset_with_index):
        """Codec decode reconstructs num_rows/num_row_groups/row_groups exactly."""
        assert self._synthesized_extras(
            parquet_dataset_with_index, "pq_codec"
        ) == self._expected_extras(parquet_dataset_with_index)

    def test_extra_roundtrips_through_default_codec(
        self, parquet_dataset_with_index, monkeypatch
    ):
        """The generic rest path alone preserves ``extra`` field-for-field.

        With no parquet codec registered, the catch-all per-shard blob must
        reconstruct ``extra`` exactly.
        """
        monkeypatch.delitem(extra_codec._CODECS, "parquet", raising=False)
        assert self._synthesized_extras(
            parquet_dataset_with_index, "pq_default"
        ) == self._expected_extras(parquet_dataset_with_index)

    def test_catalog_locator_serves_reader(self, parquet_dataset_with_index):
        """A catalog-synthesized locator's decoded ``extra`` feeds ``open_shard``.

        The row-group offsets come back from the ragged int64 columns; a wrong
        slice would break row-group pruning at read time.
        """
        dataset = Dataset.from_path(
            name="pq_read", path=str(parquet_dataset_with_index)
        )
        _, locators = catalog_locators(dataset)

        # data_000.parquet: 10000 rows in row groups of 2000.
        locator = locators[0]
        local_ref = LocalShardRef(
            raw=LocalShardFile(
                path=parquet_dataset_with_index / locator.raw.basename,
                bytes=locator.raw.bytes,
            ),
            extra=locator.extra,
        )
        shard = ParquetFormat().open_shard(locator, local_ref)
        assert len(shard) == 10000
        # Rows on both sides of a row-group boundary exercise the rg offsets.
        assert shard[0] == {"id": 0, "text": "item_0", "value": 0}
        assert shard[1999] == {"id": 1999, "text": "item_1999", "value": 3998}
        assert shard[2000] == {"id": 2000, "text": "item_2000", "value": 4000}
        assert shard[9999] == {"id": 9999, "text": "item_9999", "value": 19998}

    def test_codec_raises_on_num_rows_contradiction(self):
        """``encode`` refuses an extra ``num_rows`` that disagrees with the counts.

        Decode substitutes the catalog's ``num_rows`` column, so a contradictory
        extra cannot round-trip; it must fail at build time rather than be
        silently rewritten.
        """
        import numpy as np

        codec = _ParquetExtraCodec()
        metas = [
            {
                "num_rows": 5,
                "num_row_groups": 1,
                "row_groups": [{"num_rows": 5, "total_byte_size": 10}],
            }
        ]
        with pytest.raises(ValueError, match="self-contradictory"):
            codec.encode(metas, np.array([7], dtype=np.int64))

    def test_codec_raises_on_row_group_count_contradiction(self):
        """``encode`` refuses ``num_row_groups`` != ``len(row_groups)``."""
        import numpy as np

        codec = _ParquetExtraCodec()
        metas = [
            {
                "num_rows": 5,
                "num_row_groups": 2,
                "row_groups": [{"num_rows": 5, "total_byte_size": 10}],
            }
        ]
        with pytest.raises(ValueError, match="num_row_groups"):
            codec.encode(metas, np.array([5], dtype=np.int64))

    def test_codec_owns_nothing_for_foreign_extras(self, tmp_path):
        """Non-discovery-shaped extras ride the generic rest path untouched.

        Round-trips through a real packed catalog: decode must step aside (no
        ``rg_off`` columns) so the rest path alone reconstructs the extra.
        """
        import numpy as np

        from tests._helpers import catalog_set_from_locators

        codec = _ParquetExtraCodec()
        extra = {"custom": 1}
        encoded = codec.encode([dict(extra)], np.array([5], dtype=np.int64))
        assert encoded.owned_keys == frozenset()
        assert encoded.int_columns == {}

        locator = ShardLocator(
            dataset="pq_foreign",
            shard_id=0,
            format="parquet",
            root=str(tmp_path),
            raw=ShardFile(basename="s0.parquet", bytes=10, hashes={}),
            extra=dict(extra),
        )
        catalog_set = catalog_set_from_locators([locator], tmp_path / "_catalogs")
        assert dict(catalog_set.locator_at(0).extra) == extra


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
