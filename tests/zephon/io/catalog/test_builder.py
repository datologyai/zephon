# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the catalog builder (``build_catalog`` / ``pack_locators``).

The central guarantee is *fidelity*: a ``ShardLocator`` synthesized from the
columnar artifact must reproduce, field for field, what the format handler's
``build_locators`` produces — otherwise the catalog would silently hand workers
a different ``shard_id -> file`` mapping than discovery did.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from zephon.io.catalog import build_catalog
from zephon.io.catalog import io as catalog_io
from zephon.io.catalog.builder import DatasetHeader, pack_locators
from zephon.io.catalog.catalog import ShardCatalog
from zephon.io.types import ShardFile, ShardLocator


def _make_jsonl(root: Path, shards: dict[str, int]) -> None:
    for name, count in shards.items():
        path = root / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps({"i": i}) + "\n" for i in range(count)),
            encoding="utf-8",
        )


def _header(root: Path, name: str = "ds") -> DatasetHeader:
    return DatasetHeader(name=name, root=str(root), format="jsonl", path=str(root))


def _load(built, tmp_path: Path) -> ShardCatalog:
    path = tmp_path / "art"
    catalog_io.write_atomic(path, built.file_bytes)
    return ShardCatalog(catalog_io.load_mmap(path))


# --------------------------------------------------------------------------
# build_catalog: full discovery -> columnarize -> synthesize
# --------------------------------------------------------------------------


def test_build_catalog_jsonl_known_values(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir()
    _make_jsonl(root, {"a": 3, "b": 5})
    catalog = _load(build_catalog(_header(root)), tmp_path)

    assert catalog.shard_count == 2
    assert catalog.total() == 8
    assert catalog.max_count() == 5
    np.testing.assert_array_equal(catalog.ids(), np.array([0, 1]))
    np.testing.assert_array_equal(catalog.num_rows(), np.array([3, 5]))

    loc = catalog.locator_at(catalog.slot_of(1), dataset_name="ds")
    assert loc.shard_id == 1
    assert loc.format == "jsonl"
    assert loc.root == str(root)
    assert dict(loc.extra) == {"length": 5}


def test_build_catalog_locators_match_build_locators(tmp_path: Path) -> None:
    """``locator_at`` reproduces ``handler.build_locators`` field for field."""
    from zephon.io.dataset import Dataset
    from zephon.io.formats import ensure_builtin_formats
    from zephon.io.formats.base import get_format
    from zephon.io.storage import RouterStorageBackend

    root = tmp_path / "ds"
    root.mkdir()
    _make_jsonl(root, {"a": 3, "b": 5, "c": 7})
    header = _header(root)

    # Canonical locators straight from the handler: exactly what build_catalog
    # columnarizes, and what the catalog must synthesize identically.
    ensure_builtin_formats(required={"jsonl"})
    handler = get_format("jsonl")
    shard_index, shard_meta = handler.discover(str(root), RouterStorageBackend())
    tmp_ds = Dataset(
        name="ds",
        backend={"kind": "jsonl", "path": str(root), "shards": shard_meta},
        path=str(root),
    )
    canonical = dict(handler.build_locators(tmp_ds))

    catalog = _load(build_catalog(header), tmp_path)
    assert set(catalog.ids().tolist()) == set(canonical)
    for sid, want in canonical.items():
        got = catalog.locator_at(catalog.slot_of(int(sid)), dataset_name="ds")
        assert got.shard_id == want.shard_id
        assert got.format == want.format
        assert got.root == want.root
        assert got.compression == want.compression
        assert got.raw.basename == want.raw.basename
        assert got.raw.bytes == want.raw.bytes
        assert dict(got.raw.hashes) == dict(want.raw.hashes or {})
        if want.zip is None:
            assert got.zip is None
        else:
            assert got.zip is not None
            assert got.zip.basename == want.zip.basename
            assert got.zip.bytes == want.zip.bytes
            assert dict(got.zip.hashes) == dict(want.zip.hashes or {})
        assert dict(got.extra or {}) == dict(want.extra or {})


def test_fingerprint_stable_and_name_independent(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir()
    _make_jsonl(root, {"a": 3, "b": 5})
    fp1 = build_catalog(_header(root, name="alpha")).fingerprint
    fp2 = build_catalog(_header(root, name="alpha")).fingerprint
    fp_alias = build_catalog(_header(root, name="BETA")).fingerprint
    assert fp1 == fp2  # deterministic
    assert fp1 == fp_alias  # name excluded from fingerprint


def test_fingerprint_changes_with_root(tmp_path: Path) -> None:
    root1 = tmp_path / "ds1"
    root2 = tmp_path / "ds2"
    for r in (root1, root2):
        r.mkdir()
        _make_jsonl(r, {"a": 3, "b": 5})
    fp1 = build_catalog(_header(root1)).fingerprint
    fp2 = build_catalog(_header(root2)).fingerprint
    assert fp1 != fp2  # physical identity (root) is part of the hash


# --------------------------------------------------------------------------
# pack_locators: columnarize synthetic locators (hashes / zip / compression)
# --------------------------------------------------------------------------


def test_pack_locators_roundtrips_all_locator_fields(tmp_path: Path) -> None:
    """The optional columns (hashes, zip, compression, extra) survive a round-trip."""
    locators = {
        0: ShardLocator(
            dataset="d",
            shard_id=0,
            format="jsonl",
            root="/data/d",
            raw=ShardFile(basename="s0.jsonl", bytes=10, hashes={"sha256": "aa"}),
            zip=ShardFile(basename="s0.jsonl.zip", bytes=4, hashes={"crc32": "bb"}),
            compression="zstd",
            extra={"length": 3, "custom": "x"},
        ),
        2: ShardLocator(
            dataset="d",
            shard_id=2,
            format="jsonl",
            root="/data/d",
            raw=ShardFile(basename="s2.jsonl", bytes=20, hashes={}),
            zip=None,
            compression=None,
            extra={"length": 7, "custom": "y"},
        ),
    }
    header = DatasetHeader(name="d", root="/data/d", format="jsonl", path="/data/d")
    catalog = _load(pack_locators(header, locators, {0: 3, 2: 7}), tmp_path)

    # Sparse ids 0 and 2 -> slots 0 and 1; the gap (1) is not addressable.
    np.testing.assert_array_equal(catalog.ids(), np.array([0, 2]))
    assert catalog.slot_of(0) == 0
    assert catalog.slot_of(2) == 1
    with pytest.raises(KeyError):
        catalog.slot_of(1)

    l0 = catalog.locator_at(catalog.slot_of(0), dataset_name="d")
    assert l0.raw.basename == "s0.jsonl"
    assert l0.raw.bytes == 10
    assert dict(l0.raw.hashes) == {"sha256": "aa"}
    assert l0.zip is not None
    assert l0.zip.basename == "s0.jsonl.zip"
    assert l0.zip.bytes == 4
    assert dict(l0.zip.hashes) == {"crc32": "bb"}
    assert l0.compression == "zstd"
    assert dict(l0.extra) == {"length": 3, "custom": "x"}

    l2 = catalog.locator_at(catalog.slot_of(2), dataset_name="d")
    assert l2.raw.basename == "s2.jsonl"
    assert dict(l2.raw.hashes) == {}
    assert l2.zip is None
    assert l2.compression is None
    assert dict(l2.extra) == {"length": 7, "custom": "y"}


def test_pack_locators_dense_ids_use_identity_slots(tmp_path: Path) -> None:
    locators = {
        i: ShardLocator(
            dataset="d",
            shard_id=i,
            format="jsonl",
            root="/data/d",
            raw=ShardFile(basename=f"s{i}.jsonl", bytes=i + 1, hashes={}),
            extra={"length": i + 1},
        )
        for i in range(3)
    }
    catalog = _load(
        pack_locators(
            DatasetHeader(name="d", root="/data/d", format="jsonl", path="/data/d"),
            locators,
            {0: 1, 1: 2, 2: 3},
        ),
        tmp_path,
    )
    assert catalog.shard_count == 3
    assert [catalog.slot_of(i) for i in range(3)] == [0, 1, 2]
    assert catalog.total() == 6


def test_pack_locators_hoists_constant_extra_to_header(tmp_path: Path) -> None:
    """Extra identical across shards collapses to one header blob, not a column.

    The generic "rest" path splits per-dataset from per-shard metadata purely by
    value-constancy: when every shard's leftover ``extra`` is byte-identical it
    must hoist to a single header blob (the memory win), while each shard still
    reconstructs the full mapping. No other test exercises this branch.
    """
    shared = {"schema": "v2", "codec": "raw"}
    locators = {
        i: ShardLocator(
            dataset="d",
            shard_id=i,
            format="jsonl",
            root="/data/d",
            raw=ShardFile(basename=f"s{i}.jsonl", bytes=i + 1, hashes={}),
            extra=dict(shared),
        )
        for i in range(3)
    }
    catalog = _load(
        pack_locators(
            DatasetHeader(name="d", root="/data/d", format="jsonl", path="/data/d"),
            locators,
            {0: 1, 1: 2, 2: 3},
        ),
        tmp_path,
    )

    # The hoist actually fired: one shared blob, no per-shard "rest" column.
    assert catalog._loaded.header["extra_flags"]["rest_constant"] is True
    assert "extra_rest_header_blob" in catalog._loaded.columns
    assert "extra_rest_off" not in catalog._loaded.columns

    # ...and every shard still reconstructs the full extra.
    for i in range(3):
        loc = catalog.locator_at(catalog.slot_of(i), dataset_name="d")
        assert dict(loc.extra) == shared


def test_pack_locators_empty_catalog(tmp_path: Path) -> None:
    """A zero-shard catalog is well-defined: empty ids, zero totals, no slots."""
    catalog = _load(
        pack_locators(
            DatasetHeader(name="d", root="/data/d", format="jsonl", path="/data/d"),
            {},
            {},
        ),
        tmp_path,
    )
    assert catalog.shard_count == 0
    assert catalog.total() == 0
    # max_count() keeps its empty-guard: np.max() over an empty array would raise.
    assert catalog.max_count() == 0
    assert catalog.ids().tolist() == []
    with pytest.raises(KeyError):
        catalog.slot_of(0)
    assert catalog.reverse_lookup("anything") is None
