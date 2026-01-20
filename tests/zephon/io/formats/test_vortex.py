import sys
from pathlib import Path
from unittest import mock

import pytest

from zephon.io.dataset import Dataset
from zephon.io.formats.vortex import VortexFormat, VortexShard
from zephon.io.storage.local import LocalFSBackend
from zephon.io.types import LocalShardFile, LocalShardRef

# Skip vortex tests if vortex-data is not available (requires Python 3.11+)
# Note: there's a different "vortex" package on PyPI, so we check for vortex.io
vortex = pytest.importorskip("vortex.io", reason="vortex-data not installed")
import vortex


def _create_vortex_file(path: Path, rows: list[dict[str, object]]) -> None:
    """Create a Vortex file from a list of row dictionaries."""
    # Create vortex array directly from list of dicts and write to file
    vortex_array = vortex.array(rows)
    vortex.io.write(vortex_array, str(path))


def test_vortex_discover_local(tmp_path: Path) -> None:
    """Test discovering Vortex shards in a directory."""
    shard0 = tmp_path / "a.vortex"
    shard1 = tmp_path / "b.vortex"
    _create_vortex_file(shard0, [{"i": i} for i in range(2)])
    _create_vortex_file(shard1, [{"i": i} for i in range(3)])

    handler = VortexFormat()
    shard_index, shard_meta = handler.discover(str(tmp_path), LocalFSBackend(tmp_path))

    # Check total row count
    assert sum(int(v) for v in shard_index.values()) == 5
    assert len(shard_meta) == len(shard_index)
    # Ensure basic metadata captured
    for meta in shard_meta.values():
        assert isinstance(meta.get("raw"), dict)
        assert isinstance(meta.get("extra"), dict)


def test_vortex_build_locators_and_open(tmp_path: Path) -> None:
    """Test building locators and opening a Vortex shard."""
    shard0 = tmp_path / "shard0.vortex"
    _create_vortex_file(shard0, [{"x": 1}, {"x": 2}])

    handler = VortexFormat()
    shard_index, shard_meta = handler.discover(str(tmp_path), LocalFSBackend(tmp_path))

    ds = Dataset(
        name="demo",
        shard_index=shard_index,
        backend={"kind": "vortex", "path": str(tmp_path), "shards": shard_meta},
        path=str(tmp_path),
    )

    locators = handler.build_locators(ds)
    assert set(locators.keys()) == set(shard_index.keys())
    loc = next(iter(locators.values()))

    # Open and verify shard contents
    ref = LocalShardRef(raw=LocalShardFile(path=shard0, bytes=shard0.stat().st_size))
    shard = handler.open_shard(loc, ref)
    assert len(shard) == 2
    shard.close()

    # Open with extra length hint
    ref2 = LocalShardRef(
        raw=LocalShardFile(path=shard0, bytes=shard0.stat().st_size),
        extra={"length": 2},
    )
    shard2 = handler.open_shard(loc, ref2)
    assert len(shard2) == 2
    shard2.close()


@pytest.mark.parametrize(
    "bad_shards",
    [
        # Missing raw
        {0: {}},
        # Missing basename
        {0: {"raw": {"bytes": 1}}},
        # Missing bytes
        {0: {"raw": {"basename": "shard0.vortex"}}},
        # Invalid bytes type
        {0: {"raw": {"basename": "shard0.vortex", "bytes": object()}}},
    ],
)
def test_vortex_build_locators_rejects_bad_metadata(tmp_path: Path, bad_shards) -> None:
    """Test that build_locators rejects invalid metadata."""
    ds = Dataset(
        name="bad",
        shard_index={0: 1},
        backend={"kind": "vortex", "path": str(tmp_path), "shards": bad_shards},
        path=str(tmp_path),
    )
    handler = VortexFormat()
    with pytest.raises(ValueError):
        _ = handler.build_locators(ds)


def test_vortex_shard_getsamples(tmp_path: Path) -> None:
    """Test VortexShard.getsamples with sorted indices."""
    p = tmp_path / "shard.vortex"
    rows = [{"v": i} for i in range(6)]
    _create_vortex_file(p, rows)

    shard = VortexShard(p)

    # Test sorted indices with consecutive duplicates (as fetch.py may provide)
    out = shard.getsamples([0, 1, 1, 4, 4])
    assert [r["v"] for r in out] == [0, 1, 1, 4, 4]

    # Test empty indices
    assert shard.getsamples([]) == []

    # Test out of range
    with pytest.raises(IndexError):
        _ = shard.getsamples([10])

    shard.close()


def test_vortex_shard_indexing(tmp_path: Path) -> None:
    """Test VortexShard single item indexing."""
    p = tmp_path / "shard.vortex"
    rows = [{"a": i, "b": f"val_{i}"} for i in range(3)]
    _create_vortex_file(p, rows)

    shard = VortexShard(p)
    assert len(shard) == 3

    # Test negative index
    with pytest.raises(IndexError):
        _ = shard[-1]

    # Test out of range
    with pytest.raises(IndexError):
        _ = shard[3]

    shard.close()


def test_vortex_import_error_handling() -> None:
    """Test that VortexShard raises helpful error when vortex not installed."""
    # Temporarily hide vortex module to test import error path
    with mock.patch.dict(sys.modules, {"vortex": None}):
        # Need to reload the module to trigger the import check

        from zephon.io.formats import vortex as vortex_module

        # Save original value
        original_vortex = vortex_module._vortex
        vortex_module._vortex = None

        try:
            with pytest.raises(RuntimeError, match="vortex-data"):
                VortexShard(Path("/fake/path.vortex"))
        finally:
            # Restore original value
            vortex_module._vortex = original_vortex


def test_vortex_discover_no_shards(tmp_path: Path) -> None:
    """Test that discover raises ValueError when no .vortex files found."""
    # Create a directory with no .vortex files
    (tmp_path / "other.txt").write_text("hello")

    handler = VortexFormat()

    # If vortex is installed, should raise ValueError for no shards
    with pytest.raises(ValueError, match="No .vortex shards found"):
        handler.discover(str(tmp_path), LocalFSBackend(tmp_path))


def test_vortex_multiple_shards_data_integrity(tmp_path: Path) -> None:
    """Test reading data from multiple shards preserves values correctly."""
    # Create 4 shards with distinct data ranges
    for i in range(4):
        shard_path = tmp_path / f"shard_{i:02d}.vortex"
        start = i * 10
        rows = [{"idx": start + j, "shard": i} for j in range(10)]
        _create_vortex_file(shard_path, rows)

    handler = VortexFormat()
    shard_index, shard_meta = handler.discover(str(tmp_path), LocalFSBackend(tmp_path))

    # Should have 4 shards with 10 rows each
    assert len(shard_index) == 4
    assert sum(shard_index.values()) == 40

    # Verify each shard's data
    ds = Dataset(
        name="multi",
        shard_index=shard_index,
        backend={"kind": "vortex", "path": str(tmp_path), "shards": shard_meta},
        path=str(tmp_path),
    )
    locators = handler.build_locators(ds)

    for shard_id, loc in locators.items():
        shard_path = tmp_path / loc.raw.basename
        ref = LocalShardRef(
            raw=LocalShardFile(path=shard_path, bytes=shard_path.stat().st_size),
            extra=shard_meta[shard_id].get("extra"),
        )
        shard = handler.open_shard(loc, ref)
        # Each shard should have 10 rows
        assert len(shard) == 10
        # Verify first and last row
        first = shard[0]
        last = shard[9]
        assert first["shard"] == last["shard"]  # Same shard marker
        assert last["idx"] - first["idx"] == 9  # Sequential indices
        shard.close()


def test_vortex_mixed_column_types(tmp_path: Path) -> None:
    """Test handling of various column types."""
    rows = [
        {
            "int_col": 42,
            "float_col": 3.14159,
            "str_col": "hello world",
            "bool_col": True,
            "none_col": None,
        },
        {
            "int_col": -100,
            "float_col": 2.71828,
            "str_col": "goodbye",
            "bool_col": False,
            "none_col": None,
        },
    ]
    p = tmp_path / "mixed.vortex"
    _create_vortex_file(p, rows)

    shard = VortexShard(p)
    assert len(shard) == 2

    row0 = shard[0]
    assert row0["int_col"] == 42
    assert abs(row0["float_col"] - 3.14159) < 0.0001
    assert row0["str_col"] == "hello world"
    assert row0["bool_col"] is True

    row1 = shard[1]
    assert row1["int_col"] == -100
    assert row1["bool_col"] is False

    shard.close()


def test_vortex_wide_table(tmp_path: Path) -> None:
    """Test handling of tables with many columns."""
    num_cols = 100
    num_rows = 5

    rows = []
    for i in range(num_rows):
        row = {f"col_{j:03d}": i * num_cols + j for j in range(num_cols)}
        rows.append(row)

    p = tmp_path / "wide.vortex"
    _create_vortex_file(p, rows)

    shard = VortexShard(p)
    assert len(shard) == num_rows

    # Verify column count and values
    row = shard[2]
    assert len(row) == num_cols
    assert row["col_000"] == 2 * num_cols
    assert row["col_099"] == 2 * num_cols + 99

    shard.close()


def test_vortex_single_row_shard(tmp_path: Path) -> None:
    """Test handling of shards with a single row."""
    p = tmp_path / "single.vortex"
    _create_vortex_file(p, [{"only": "row"}])

    shard = VortexShard(p)
    assert len(shard) == 1
    assert shard[0]["only"] == "row"

    # getsamples with single item
    result = shard.getsamples([0])
    assert len(result) == 1
    assert result[0]["only"] == "row"

    # Out of bounds
    with pytest.raises(IndexError):
        _ = shard[1]

    shard.close()


def test_vortex_shard_ordering_deterministic(tmp_path: Path) -> None:
    """Test that shard discovery ordering is deterministic."""
    # Create shards with names that might sort differently
    names = ["z_last.vortex", "a_first.vortex", "m_middle.vortex"]
    for name in names:
        _create_vortex_file(tmp_path / name, [{"name": name}])

    handler = VortexFormat()

    # Discover multiple times and verify consistent ordering
    results = []
    for _ in range(3):
        shard_index, shard_meta = handler.discover(
            str(tmp_path), LocalFSBackend(tmp_path)
        )
        basenames = [
            shard_meta[i]["raw"]["basename"] for i in sorted(shard_meta.keys())
        ]
        results.append(basenames)

    # All discoveries should produce the same order
    assert results[0] == results[1] == results[2]
    # Should be alphabetically sorted
    assert results[0] == sorted(names)


def test_vortex_unicode_strings(tmp_path: Path) -> None:
    """Test handling of unicode content."""
    rows = [
        {"text": "Hello, 世界!"},
        {"text": "Привет мир"},
        {"text": "🎉🚀🔥"},
        {"text": "café résumé naïve"},
    ]
    p = tmp_path / "unicode.vortex"
    _create_vortex_file(p, rows)

    shard = VortexShard(p)
    assert len(shard) == 4
    assert shard[0]["text"] == "Hello, 世界!"
    assert shard[2]["text"] == "🎉🚀🔥"

    samples = shard.getsamples([1, 3])
    assert samples[0]["text"] == "Привет мир"
    assert samples[1]["text"] == "café résumé naïve"

    shard.close()


def test_vortex_large_values(tmp_path: Path) -> None:
    """Test handling of large string values."""
    large_string = "x" * 100_000  # 100KB string
    rows = [{"data": large_string, "idx": i} for i in range(3)]

    p = tmp_path / "large.vortex"
    _create_vortex_file(p, rows)

    shard = VortexShard(p)
    assert len(shard) == 3

    row = shard[1]
    assert len(row["data"]) == 100_000
    assert row["data"] == large_string

    shard.close()


def test_vortex_getsamples_all_rows(tmp_path: Path) -> None:
    """Test getsamples retrieving rows with sorted indices."""
    rows = [{"v": i} for i in range(100)]
    p = tmp_path / "hundred.vortex"
    _create_vortex_file(p, rows)

    shard = VortexShard(p)

    # Forward order
    forward = shard.getsamples(list(range(100)))
    assert [r["v"] for r in forward] == list(range(100))

    # Every other row
    evens = shard.getsamples(list(range(0, 100, 2)))
    assert [r["v"] for r in evens] == list(range(0, 100, 2))

    shard.close()


def test_vortex_getsamples_contiguous_optimization(tmp_path: Path) -> None:
    """Test that getsamples handles contiguous indices correctly.

    This exercises the optimized code path that uses a single slice()
    for contiguous ranges instead of individual scalar_at() calls.
    """
    rows = [{"id": i, "text": f"row_{i}"} for i in range(50)]
    p = tmp_path / "contiguous.vortex"
    _create_vortex_file(p, rows)

    shard = VortexShard(p)

    # Contiguous range at start (uses slice optimization)
    result = shard.getsamples(list(range(10)))
    assert [r["id"] for r in result] == list(range(10))
    assert [r["text"] for r in result] == [f"row_{i}" for i in range(10)]

    # Contiguous range in middle (uses slice optimization)
    result = shard.getsamples(list(range(20, 35)))
    assert [r["id"] for r in result] == list(range(20, 35))

    # Contiguous range at end (uses slice optimization)
    result = shard.getsamples(list(range(40, 50)))
    assert [r["id"] for r in result] == list(range(40, 50))

    # Single element (no contiguity check needed)
    result = shard.getsamples([25])
    assert result[0]["id"] == 25

    shard.close()


def test_vortex_getsamples_non_contiguous(tmp_path: Path) -> None:
    """Test that getsamples handles non-contiguous sorted indices.

    This exercises the scan API with gaps in the index sequence.
    """
    rows = [{"id": i, "text": f"row_{i}"} for i in range(100)]
    p = tmp_path / "scattered.vortex"
    _create_vortex_file(p, rows)

    shard = VortexShard(p)

    # Sorted scattered indices
    scattered = [5, 10, 25, 50, 75]
    result = shard.getsamples(scattered)
    assert [r["id"] for r in result] == scattered

    # Every 10th row
    every_tenth = list(range(0, 100, 10))
    result = shard.getsamples(every_tenth)
    assert [r["id"] for r in result] == every_tenth

    # Sorted with consecutive duplicates
    with_dups = [5, 5, 10, 10, 10]
    result = shard.getsamples(with_dups)
    assert [r["id"] for r in result] == with_dups

    # Non-contiguous with gap at 3
    with_gap = [0, 1, 2, 4, 5, 6]
    result = shard.getsamples(with_gap)
    assert [r["id"] for r in result] == with_gap

    shard.close()


def test_vortex_getitem_scalar_at(tmp_path: Path) -> None:
    """Test that __getitem__ works correctly with scalar_at optimization."""
    rows = [{"id": i, "value": i * 1.5, "name": f"item_{i}"} for i in range(20)]
    p = tmp_path / "scalar.vortex"
    _create_vortex_file(p, rows)

    shard = VortexShard(p)

    # Test various indices
    assert shard[0] == {"id": 0, "value": 0.0, "name": "item_0"}
    assert shard[10] == {"id": 10, "value": 15.0, "name": "item_10"}
    assert shard[19] == {"id": 19, "value": 28.5, "name": "item_19"}

    # Verify all rows accessible
    for i in range(20):
        row = shard[i]
        assert row["id"] == i
        assert row["value"] == i * 1.5
        assert row["name"] == f"item_{i}"

    shard.close()


# --- Auto-detection and integration tests ---


def test_vortex_auto_detect_format(tmp_path: Path) -> None:
    """Test auto-detection of Vortex format from directory contents."""
    # Create vortex files
    _create_vortex_file(tmp_path / "shard_0.vortex", [{"x": i} for i in range(10)])
    _create_vortex_file(tmp_path / "shard_1.vortex", [{"x": i} for i in range(10, 20)])

    dataset = Dataset.from_path(name="test_vortex", path=str(tmp_path))

    assert dataset.backend["kind"] == "vortex"
    assert len(dataset.shard_index) == 2
    assert sum(dataset.shard_index.values()) == 20


def test_vortex_explicit_format_specification(tmp_path: Path) -> None:
    """Test explicitly specifying format='vortex'."""
    _create_vortex_file(tmp_path / "data.vortex", [{"val": i} for i in range(5)])

    dataset = Dataset.from_path(name="explicit", path=str(tmp_path), fmt="vortex")

    assert dataset.backend["kind"] == "vortex"
    assert len(dataset.shard_index) == 1
    assert dataset.shard_index[0] == 5


def test_vortex_end_to_end_reading(tmp_path: Path) -> None:
    """Test complete workflow from Dataset creation to shard reading."""
    # Create multi-shard dataset
    for i in range(3):
        rows = [{"shard_id": i, "row_id": j, "value": i * 100 + j} for j in range(50)]
        _create_vortex_file(tmp_path / f"part_{i:02d}.vortex", rows)

    dataset = Dataset.from_path(name="e2e_test", path=str(tmp_path))

    # Verify dataset structure
    assert len(dataset.shard_index) == 3
    total_samples = sum(dataset.shard_index.values())
    assert total_samples == 150  # 3 shards * 50 rows

    # Verify we can build locators and open shards
    handler = VortexFormat()
    locators = handler.build_locators(dataset)
    assert len(locators) == 3

    # Open each shard and verify data
    shard_meta = dataset.backend["shards"]
    for shard_id, loc in locators.items():
        shard_path = tmp_path / loc.raw.basename
        ref = LocalShardRef(
            raw=LocalShardFile(path=shard_path, bytes=shard_path.stat().st_size),
            extra=shard_meta[shard_id].get("extra"),
        )
        shard = handler.open_shard(loc, ref)
        assert len(shard) == 50

        # Verify first row of each shard
        first_row = shard[0]
        assert first_row["row_id"] == 0
        assert first_row["value"] == first_row["shard_id"] * 100

        shard.close()


def test_vortex_dataset_len(tmp_path: Path) -> None:
    """Test Dataset.__len__ returns total sample count."""
    _create_vortex_file(tmp_path / "a.vortex", [{"x": i} for i in range(100)])
    _create_vortex_file(tmp_path / "b.vortex", [{"x": i} for i in range(200)])
    _create_vortex_file(tmp_path / "c.vortex", [{"x": i} for i in range(50)])

    dataset = Dataset.from_path(name="len_test", path=str(tmp_path))

    assert len(dataset) == 350  # 100 + 200 + 50
