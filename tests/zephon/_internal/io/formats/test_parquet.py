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

from tests._catalog_helpers import catalog_locators
from tests.helpers.storage import _install_obstore_stubs
from zephon._internal.io.catalog import extra_codec
from zephon._internal.io.formats.parquet import (
    ParquetFormat,
    ParquetShard,
    ParquetShardOpener,
    _ParquetExtraCodec,
)
from zephon._internal.io.index.parquet_index import ParquetIndexBuilder
from zephon._internal.io.storage import LocalFSBackend
from zephon._internal.io.storage.router import RouterStorageBackend
from zephon._internal.io.types import (
    LocalShardFile,
    LocalShardRef,
    ShardFile,
    ShardLocator,
)
from zephon.io.dataset import Dataset


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

        from zephon._internal.io.storage.s3 import S3Backend

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
            "zephon._internal.io.storage.router._make_s3_backend", lambda: backend
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
        shard = ParquetShard(shard_file, row_groups)

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
        shard = ParquetShard(shard_file, row_groups)

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
        shard = ParquetShard(shard_file, row_groups)

        with pytest.raises(IndexError):
            _ = shard[-1]

        with pytest.raises(IndexError):
            _ = shard[10000]

        with pytest.raises(IndexError):
            _ = shard[10001]

    def test_shard_getsamples_bulk(self, shard_file, shard_metadata):
        """Test bulk read with getsamples()."""
        row_groups = shard_metadata
        shard = ParquetShard(shard_file, row_groups)

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
        shard = ParquetShard(shard_file, row_groups)

        # Random order
        indices = [9999, 0, 5000, 100, 7500]
        rows = shard.getsamples(indices)

        assert len(rows) == len(indices)
        for i, idx in enumerate(indices):
            assert rows[i]["id"] == idx

    def test_shard_getsamples_duplicates(self, shard_file, shard_metadata):
        """Test that getsamples() handles duplicate indices."""
        row_groups = shard_metadata
        shard = ParquetShard(shard_file, row_groups)

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
        shard = ParquetShard(shard_file, row_groups)

        rows = shard.getsamples([])
        assert rows == []

    def test_shard_getsamples_out_of_bounds(self, shard_file, shard_metadata):
        """Test getsamples() with out-of-bounds indices."""
        row_groups = shard_metadata
        shard = ParquetShard(shard_file, row_groups)

        with pytest.raises(IndexError):
            shard.getsamples([0, 100, 10000])

        with pytest.raises(IndexError):
            shard.getsamples([0, -1, 100])


def _metadata_lifecycle_refs(
    tmp_path: Path,
) -> tuple[LocalShardRef, ShardLocator]:
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
    return local_ref, locator


class TestParquetMetadataLifecycle:
    """Tests for lazy metadata access during shard opens."""

    def test_open_shard_does_not_read_parquet_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        local_ref, locator = _metadata_lifecycle_refs(tmp_path)
        format_handler = ParquetFormat()
        calls = 0
        read_metadata = pq.read_metadata

        def counting_read_metadata(path_arg):
            nonlocal calls
            calls += 1
            return read_metadata(path_arg)

        monkeypatch.setattr(pq, "read_metadata", counting_read_metadata)
        shard1 = format_handler.open_shard(locator, local_ref)
        shard1.close()
        shard2 = format_handler.open_shard(locator, local_ref)
        shard2.close()

        assert calls == 0

    def test_parquet_misses_reuse_lazily_loaded_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        local_ref, locator = _metadata_lifecycle_refs(tmp_path)
        opener = ParquetShardOpener(
            decoded_cache=None,
            index=None,
        )
        calls = 0
        read_metadata = pq.read_metadata

        def counting_read_metadata(path_arg):
            nonlocal calls
            calls += 1
            return read_metadata(path_arg)

        monkeypatch.setattr(pq, "read_metadata", counting_read_metadata)
        first = opener.open_shard(locator, local_ref)
        first.getsamples([0])
        second = opener.open_shard(locator, local_ref)
        second.getsamples([1])

        assert calls == 1


def test_roundtrip_through_parquet_with_fixed_size_list(tmp_path: Path) -> None:
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

    shard = ParquetShard(path, row_groups)

    row = shard[0]
    assert row["id"] == 0
    np.testing.assert_array_equal(row["tokens"], list(range(list_size)))

    rows = shard.getsamples([0, 49, 50, 99])
    assert rows[0]["id"] == 0
    assert rows[1]["id"] == 49
    assert rows[2]["id"] == 50
    assert rows[3]["id"] == 99
    np.testing.assert_array_equal(
        rows[2]["tokens"], list(range(50 * list_size, 51 * list_size))
    )


def test_shard_variable_list_preserves_numpy_via_take(tmp_path):
    """Take-based decode keeps variable-length list cells as writable numpy
    arrays (not Python lists), including across a row-group boundary."""
    import numpy as np

    list_arr = pa.array(
        [[1, 2], [3, 4, 5], [6], [7, 8, 9, 10]], type=pa.list_(pa.int64())
    )
    table = pa.table({"id": pa.array(range(4)), "tok": list_arr})
    path = tmp_path / "vl.parquet"
    pq.write_table(table, str(path), row_group_size=2)

    metadata = pq.read_metadata(str(path))
    row_groups = [
        {
            "num_rows": metadata.row_group(i).num_rows,
            "total_byte_size": metadata.row_group(i).total_byte_size,
        }
        for i in range(metadata.num_row_groups)
    ]
    shard = ParquetShard(path, row_groups)

    row = shard[1]
    assert isinstance(row["tok"], np.ndarray)
    assert row["tok"].dtype == np.int64
    assert row["tok"].flags.writeable  # Materialization copies list cells
    np.testing.assert_array_equal(row["tok"], [3, 4, 5])

    # getsamples spanning both row groups, arbitrary order.
    rows = shard.getsamples([3, 0, 1])
    np.testing.assert_array_equal(rows[0]["tok"], [7, 8, 9, 10])
    np.testing.assert_array_equal(rows[1]["tok"], [1, 2])
    np.testing.assert_array_equal(rows[2]["tok"], [3, 4, 5])


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

        from tests._catalog_helpers import catalog_set_from_locators

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
