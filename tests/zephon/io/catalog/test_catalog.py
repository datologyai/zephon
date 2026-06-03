# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the zero-copy catalog views (``ShardCatalog`` / ``CatalogSet``).

Catalogs are built here from synthetic locators via
:func:`tests._helpers.catalog_set_from_locators` (``pack_locators`` -> on-disk
artifact -> ``mmap``), which exercises the same columnarize/synthesize path the
production node-local build uses.
"""

import mmap as _mmap
import pickle
from pathlib import Path

import numpy as np
import pytest

from tests._helpers import catalog_set_from_locators
from zephon.io.catalog import CatalogSet
from zephon.io.types import ShardFile, ShardLocator


def _loc(
    dataset: str,
    shard_id: int,
    basename: str,
    *,
    root: str | None = None,
    zip_name: str | None = None,
    compression: str | None = None,
    extra: dict | None = None,
) -> ShardLocator:
    return ShardLocator(
        dataset=dataset,
        shard_id=shard_id,
        format="jsonl",
        root=root or f"/data/{dataset}",
        raw=ShardFile(basename=basename, bytes=len(basename), hashes={}),
        zip=ShardFile(basename=zip_name, bytes=1, hashes={}) if zip_name else None,
        compression=compression,
        extra=extra or {},
    )


# --------------------------------------------------------------------------
# ShardCatalog: slot mapping, mmap backing, reverse lookup
# --------------------------------------------------------------------------


def test_shardcatalog_sparse_ids_and_counts(tmp_path: Path) -> None:
    cset = catalog_set_from_locators(
        [_loc("ds", 0, "s0.jsonl"), _loc("ds", 5, "s5.jsonl")],
        tmp_path,
        counts={0: 3, 5: 9},
    )
    cat = cset.catalog_for("ds")
    np.testing.assert_array_equal(cat.ids(), np.array([0, 5]))
    np.testing.assert_array_equal(cat.num_rows(), np.array([3, 9]))
    assert cat.total() == 12
    assert cat.max_count() == 9
    assert cat.slot_of(0) == 0
    assert cat.slot_of(5) == 1
    with pytest.raises(KeyError):
        cat.slot_of(1)  # a gap in the sparse id space is not addressable


def test_shardcatalog_reverse_lookup_raw_and_zip(tmp_path: Path) -> None:
    cset = catalog_set_from_locators(
        [
            _loc("ds", 0, "s0.jsonl", zip_name="s0.jsonl.zip"),
            _loc("ds", 5, "s5.jsonl"),
        ],
        tmp_path,
    )
    cat = cset.catalog_for("ds")
    assert cat.reverse_lookup("s0.jsonl") == (0, "raw")
    assert cat.reverse_lookup("s5.jsonl") == (1, "raw")
    assert cat.reverse_lookup("s0.jsonl.zip") == (0, "zip")
    assert cat.reverse_lookup("not-a-shard") is None


def test_shardcatalog_columns_are_mmap_views(tmp_path: Path) -> None:
    # The view's columns must be zero-copy, read-only windows into the file's
    # mmap (not private anonymous copies) — that is what makes the catalog
    # page-shared across ranks/workers on a node.
    cset = catalog_set_from_locators(
        [_loc("ds", 0, "s0.jsonl"), _loc("ds", 1, "s1.jsonl")], tmp_path
    )
    cat = cset.catalog_for("ds")
    num_rows = cat.num_rows()
    assert isinstance(cat._loaded._mmap, _mmap.mmap)
    assert not num_rows.flags.writeable
    assert np.shares_memory(num_rows, np.frombuffer(cat._loaded._mmap, dtype=np.uint8))


# --------------------------------------------------------------------------
# CatalogSet: name-sorted dense slot space, fingerprint, pickle guard
# --------------------------------------------------------------------------


def test_catalog_set_composition(tmp_path: Path) -> None:
    cset = catalog_set_from_locators(
        [
            _loc("zeta", 0, "z0.jsonl"),
            _loc("alpha", 0, "a0.jsonl"),
            _loc("alpha", 1, "a1.jsonl"),
        ],
        tmp_path,
    )
    # Slots are name-sorted then shard-id-sorted: alpha(0,1) then zeta(2).
    assert cset.cacheable_names == ("alpha", "zeta")
    assert cset.num_shards == 3
    assert cset.slot_of("alpha", 0) == 0
    assert cset.slot_of("alpha", 1) == 1
    assert cset.slot_of("zeta", 0) == 2
    assert cset.slot_of("missing", 0) is None
    assert cset.locator_at(2).dataset == "zeta"

    loc = cset.locator_at(0)
    assert loc.dataset == "alpha"
    assert cset.reverse_lookup("alpha", loc.raw.basename) == (0, "raw")
    assert cset.reverse_lookup("zeta", "z0.jsonl") == (2, "raw")
    assert cset.global_fingerprint.startswith("sha256:")


def test_catalog_set_summary(tmp_path: Path) -> None:
    cset = catalog_set_from_locators(
        [_loc("alpha", 0, "a0.jsonl"), _loc("zeta", 0, "z0.jsonl")], tmp_path
    )
    summary = cset.summary()
    assert [s["name"] for s in summary] == ["alpha", "zeta"]
    assert all(s["shard_count"] == 1 for s in summary)


def test_catalog_set_rejects_duplicate_names(tmp_path: Path) -> None:
    cset = catalog_set_from_locators([_loc("dup", 0, "d0.jsonl")], tmp_path)
    cat = cset.catalog_for("dup")
    with pytest.raises(ValueError, match="Duplicate dataset name"):
        CatalogSet({0: ("dup", cat), 1: ("dup", cat)})


def test_catalog_set_not_picklable(tmp_path: Path) -> None:
    cset = catalog_set_from_locators([_loc("x", 0, "x0.jsonl")], tmp_path)
    with pytest.raises(TypeError, match="must not be pickled"):
        pickle.dumps(cset)
