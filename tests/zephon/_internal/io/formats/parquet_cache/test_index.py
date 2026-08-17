# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from tests._catalog_helpers import catalog_set_from_locators
from zephon._internal.io.formats import ensure_builtin_formats
from zephon._internal.io.formats.parquet_cache.index import ParquetRGIndex
from zephon._internal.io.types import ShardFile, ShardLocator


@pytest.fixture(autouse=True)
def _register_builtin_formats() -> None:
    ensure_builtin_formats(required={"jsonl", "parquet"})


def _parquet_locator(
    dataset: str,
    shard_id: int,
    row_group_rows: list[int],
) -> ShardLocator:
    return ShardLocator(
        dataset=dataset,
        shard_id=shard_id,
        format="parquet",
        root=f"/data/{dataset}",
        raw=ShardFile(
            basename=f"{shard_id:05d}.parquet",
            bytes=sum(row_group_rows) * 10,
            hashes={},
        ),
        zip=None,
        compression=None,
        extra={
            "num_rows": sum(row_group_rows),
            "num_row_groups": len(row_group_rows),
            "row_groups": [
                {"num_rows": rows, "total_byte_size": rows * 10}
                for rows in row_group_rows
            ],
        },
    )


def _jsonl_locator(dataset: str, shard_id: int) -> ShardLocator:
    return ShardLocator(
        dataset=dataset,
        shard_id=shard_id,
        format="jsonl",
        root=f"/data/{dataset}",
        raw=ShardFile(basename=f"{shard_id}.jsonl", bytes=10, hashes={}),
        zip=None,
        compression=None,
        extra={"length": 1},
    )


def test_dense_slots_include_only_parquet_and_follow_name_shard_rg_order(
    tmp_path: Path,
) -> None:
    locators = [
        _parquet_locator("zeta", 7, [1]),
        _jsonl_locator("middle", 0),
        _parquet_locator("alpha", 9, [2]),
        _parquet_locator("alpha", 3, [3, 4]),
    ]
    catalog_set = catalog_set_from_locators(
        locators,
        tmp_path,
        counts={7: 1, 0: 1, 9: 2, 3: 7},
    )
    index = ParquetRGIndex(catalog_set)

    assert index.num_row_groups == 4
    assert index.slot_of("alpha", 3, 0) == 0
    assert index.slot_of("alpha", 3, 1) == 1
    assert index.slot_of("alpha", 9, 0) == 2
    assert index.slot_of("zeta", 7, 0) == 3
    assert index.slot_of("middle", 0, 0) is None
    assert index.slot_of("alpha", 3, 2) is None

    locations = [index.locate(slot) for slot in range(index.num_row_groups)]
    assert [
        (location.dataset_name, location.shard_id, location.rg_id)
        for location in locations
    ] == [
        ("alpha", 3, 0),
        ("alpha", 3, 1),
        ("alpha", 9, 0),
        ("zeta", 7, 0),
    ]
    assert locations[0].locator.raw.basename == "00003.parquet"


def test_rg_offsets_remain_zero_copy_catalog_views(tmp_path: Path) -> None:
    catalog_set = catalog_set_from_locators(
        [_parquet_locator("ds", 0, [1, 2])],
        tmp_path,
        counts={0: 3},
    )
    catalog = catalog_set.catalog_for("ds")
    offsets = catalog.extra_int_column("rg_off")

    assert offsets is not None
    assert not offsets.flags.writeable
    assert np.shares_memory(offsets, catalog._loaded._mmap)


def test_fingerprint_ignores_non_parquet_catalogs(tmp_path: Path) -> None:
    parquet = _parquet_locator("parquet", 0, [1])
    first = catalog_set_from_locators(
        [parquet, _jsonl_locator("other", 0)],
        tmp_path / "first",
        counts={0: 1},
    )
    second = catalog_set_from_locators(
        [parquet, _jsonl_locator("renamed", 0)],
        tmp_path / "second",
        counts={0: 1},
    )

    assert ParquetRGIndex(first).fingerprint == ParquetRGIndex(second).fingerprint


def test_index_bounds_and_pickle_guard(tmp_path: Path) -> None:
    catalog_set = catalog_set_from_locators(
        [_parquet_locator("ds", 0, [1])],
        tmp_path,
        counts={0: 1},
    )
    index = ParquetRGIndex(catalog_set)

    with pytest.raises(IndexError):
        index.locate(-1)
    with pytest.raises(IndexError):
        index.locate(1)
    with pytest.raises(TypeError, match="must not be pickled"):
        pickle.dumps(index)


def test_identity_is_cached_per_row_group(tmp_path: Path) -> None:
    catalog_set = catalog_set_from_locators(
        [_parquet_locator("ds", 0, [1])],
        tmp_path,
        counts={0: 1},
    )
    index = ParquetRGIndex(catalog_set)

    assert index.identity_for(0) is index.identity_for(0)
