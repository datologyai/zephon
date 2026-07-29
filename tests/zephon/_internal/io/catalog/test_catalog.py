# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the node-local shard catalog: zero-copy views and lifecycle.

The view tests (``ShardCatalog`` / ``CatalogSet``) build catalogs from synthetic
locators via :func:`tests._helpers.catalog_set_from_locators` (``pack_locators``
-> on-disk artifact -> ``mmap``), exercising the same columnarize/synthesize path
the production node-local build uses. The build/finalize/attach tests drive the
full lifecycle (``build_catalog`` -> source-key-locked ``finalize`` ->
registry/file/rebuild ``attach``) over real on-disk datasets.
"""

import json
import mmap as _mmap
import os
import pickle
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from tests._helpers import catalog_set_from_locators
from zephon._internal.io import formats as formats_mod
from zephon._internal.io.catalog import (
    CATALOG_CACHE_SUBDIR,
    SCHEMA_VERSION,
    CatalogSet,
    DatasetHeader,
    ShardCatalog,
    ShardCatalogHandle,
    attach,
    build_catalog,
    clear_registry,
    finalize,
    resolve_catalog_dir,
    set_catalog_dir,
)
from zephon._internal.io.catalog import extra_codec as extra_codec_mod
from zephon._internal.io.catalog import handle as handle_mod
from zephon._internal.io.types import ShardFile, ShardLocator
from zephon.io.options import CacheOptions, StoreOptions


def _loc(
    dataset: str,
    shard_id: int,
    basename: str,
    *,
    root: str | None = None,
    zip_name: str | None = None,
    compression: str | None = None,
    extra: dict | None = None,
    raw_bytes: int | None = None,
) -> ShardLocator:
    return ShardLocator(
        dataset=dataset,
        shard_id=shard_id,
        format="jsonl",
        root=root or f"/data/{dataset}",
        raw=ShardFile(
            basename=basename,
            bytes=len(basename) if raw_bytes is None else raw_bytes,
            hashes={},
        ),
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


def test_shardcatalog_raw_bytes_in_slot_order(tmp_path: Path) -> None:
    # raw_bytes() returns the per-shard raw file sizes aligned with ids(), i.e.
    # in slot (sorted shard_id) order, not locator-argument order.
    cset = catalog_set_from_locators(
        [
            _loc("ds", 5, "s5.jsonl", raw_bytes=90),
            _loc("ds", 0, "s0.jsonl", raw_bytes=10),
        ],
        tmp_path,
    )
    cat = cset.catalog_for("ds")
    np.testing.assert_array_equal(cat.ids(), np.array([0, 5]))
    np.testing.assert_array_equal(cat.raw_bytes(), np.array([10, 90]))


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


# --------------------------------------------------------------------------
# build / finalize / attach lifecycle (handle-backed, real on-disk datasets)
# --------------------------------------------------------------------------


@pytest.fixture
def catalog_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ZEPHON_CATALOG_DIR", str(tmp_path))
    set_catalog_dir(StoreOptions())
    clear_registry()
    yield tmp_path
    clear_registry()


def _make_jsonl(root: Path, shards: dict[str, int]) -> None:
    for name, count in shards.items():
        path = root / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps({"i": i}) + "\n" for i in range(count)),
            encoding="utf-8",
        )


def _header(root: Path, name: str = "ds") -> DatasetHeader:
    return DatasetHeader(name=name, root=str(root), format="jsonl", path=str(root))


def test_finalize_then_attach_jsonl(catalog_dir: Path, tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir()
    _make_jsonl(root, {"a": 3, "b": 5})
    handle = ShardCatalogHandle(dataset=_header(root))

    fp = finalize(handle)
    assert fp.startswith("sha256:")
    assert handle.fingerprint == fp

    catalog = attach(handle)
    assert isinstance(catalog, ShardCatalog)
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


def test_attach_requires_finalized_handle(catalog_dir: Path, tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir()
    _make_jsonl(root, {"a": 1})
    handle = ShardCatalogHandle(dataset=_header(root))
    # The message must explain the usual cause: pickled before finalize().
    with pytest.raises(RuntimeError, match="pickled before finalize"):
        attach(handle)


def test_lazy_extra_is_mapping(catalog_dir: Path, tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir()
    _make_jsonl(root, {"a": 2})
    handle = ShardCatalogHandle(dataset=_header(root))
    finalize(handle)
    loc = attach(handle).locator_at(0, dataset_name="ds")
    assert isinstance(loc.extra, Mapping)
    assert loc.extra["length"] == 2
    assert list(loc.extra.keys()) == ["length"]
    assert dict(loc.extra) == {"length": 2}


# --------------------------------------------------------------------------
# fingerprint boundary
# --------------------------------------------------------------------------


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
# concurrency: exactly one builder, the rest hit the pointer
# --------------------------------------------------------------------------


def test_concurrent_finalize_builds_once(
    catalog_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ds"
    root.mkdir()
    _make_jsonl(root, {"a": 3, "b": 5, "c": 7})

    builds = {"n": 0}
    lock = threading.Lock()
    real_build = build_catalog

    def counting_build(header):
        with lock:
            builds["n"] += 1
        return real_build(header)

    monkeypatch.setattr(handle_mod, "build_catalog", counting_build)

    handles = [ShardCatalogHandle(dataset=_header(root)) for _ in range(8)]
    results: list[str] = []
    rlock = threading.Lock()

    def worker(h):
        fp = finalize(h)
        with rlock:
            results.append(fp)

    threads = [threading.Thread(target=worker, args=(h,)) for h in handles]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(results)) == 1  # all converge on one fingerprint
    assert builds["n"] == 1  # source-key lock + pointer => exactly one build


def test_attach_rebuild_matches_fingerprint(catalog_dir: Path, tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir()
    _make_jsonl(root, {"a": 3, "b": 5})
    handle = ShardCatalogHandle(dataset=_header(root))
    fp = finalize(handle)

    # Simulate a fresh machine: clear the registry and delete the file, keep the
    # baked fingerprint. attach() must rebuild and reproduce the same fp.
    clear_registry()
    (catalog_dir / f"v{SCHEMA_VERSION}" / fp).unlink()
    catalog = attach(handle)
    assert catalog.fingerprint == fp
    assert catalog.total() == 8


def test_attach_rebuild_mismatch_raises(
    catalog_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ds"
    root.mkdir()
    _make_jsonl(root, {"a": 3})
    handle = ShardCatalogHandle(dataset=_header(root))
    finalize(handle)
    handle.fingerprint = "sha256:deadbeef"  # pretend the driver baked a different fp
    clear_registry()
    with pytest.raises(handle_mod.CatalogFingerprintMismatch):
        attach(handle)


def test_attach_rebuild_concurrent_with_finalize_builds_once(
    catalog_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # attach()'s cross-node rebuild and finalize() share the per-source lock:
    # overlapping calls for one source must produce exactly one build.
    root = tmp_path / "ds"
    root.mkdir()
    _make_jsonl(root, {"a": 3, "b": 5})
    baked = ShardCatalogHandle(dataset=_header(root))
    fp = finalize(baked)
    clear_registry()
    (catalog_dir / f"v{SCHEMA_VERSION}" / fp).unlink()  # simulate a fresh node

    builds = {"n": 0}
    lock = threading.Lock()
    entered = threading.Event()
    release = threading.Event()
    real_build = build_catalog

    def gated_build(header):
        with lock:
            builds["n"] += 1
        entered.set()
        assert release.wait(timeout=30)
        return real_build(header)

    monkeypatch.setattr(handle_mod, "build_catalog", gated_build)

    attached: list[ShardCatalog] = []
    finalized: list[str] = []
    t_attach = threading.Thread(target=lambda: attached.append(attach(baked)))
    t_attach.start()
    assert entered.wait(timeout=30)  # the rebuild holds the source-key lock
    fresh = ShardCatalogHandle(dataset=_header(root))
    t_finalize = threading.Thread(target=lambda: finalized.append(finalize(fresh)))
    t_finalize.start()
    release.set()
    t_attach.join(timeout=30)
    t_finalize.join(timeout=30)

    assert builds["n"] == 1  # finalize waited, then loaded the rebuilt file
    assert finalized == [fp]
    assert attached[0].fingerprint == fp


def test_attach_cold_process_resolves_codec(
    catalog_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """attach() must import the owning format; the default codec would drop keys."""
    root = tmp_path / "ds"
    root.mkdir()
    index = {
        "shards": [
            {
                "samples": 3,
                "raw": {"basename": "s0.mds", "bytes": 1},
                "column_encodings": ["str"],
                "column_names": ["text"],
            }
        ]
    }
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    handle = ShardCatalogHandle(
        dataset=DatasetHeader(name="ds", root=str(root), format="mds", path=str(root))
    )
    finalize(handle)
    warm = dict(attach(handle).locator_at(0, dataset_name="ds").extra)
    assert "_streaming_template" in warm  # the key the mds codec owns

    # Simulate a fresh worker: no mmapped catalogs, empty codec registry, format
    # module not imported.
    clear_registry()
    monkeypatch.setattr(extra_codec_mod, "_CODECS", {})
    monkeypatch.setattr(formats_mod, "_INITIALIZED_FORMATS", set())
    monkeypatch.delitem(sys.modules, "zephon._internal.io.formats.mds")
    cold = dict(attach(handle).locator_at(0, dataset_name="ds").extra)
    assert cold == warm


def test_unwritable_catalog_dir_error_names_override(
    catalog_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only catalog dir fails loud, naming $ZEPHON_CATALOG_DIR."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores file modes")
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    monkeypatch.setenv("ZEPHON_CATALOG_DIR", str(ro / "catalog"))
    set_catalog_dir(StoreOptions())
    root = tmp_path / "ds"
    root.mkdir()
    _make_jsonl(root, {"a": 1})
    try:
        with pytest.raises(OSError, match="ZEPHON_CATALOG_DIR"):
            finalize(ShardCatalogHandle(dataset=_header(root)))
    finally:
        ro.chmod(0o755)
        monkeypatch.setenv("ZEPHON_CATALOG_DIR", str(catalog_dir))
        set_catalog_dir(StoreOptions())


def test_stale_source_sig_forces_rebuild(catalog_dir: Path, tmp_path: Path) -> None:
    # An index-bearing dataset whose index changes in place under the same root
    # gets a new source key -> pointer miss -> rebuild (no stale catalog served).
    root = tmp_path / "ds"
    root.mkdir()
    index = {"shards": [{"samples": 3, "raw": {"basename": "s0.mds", "bytes": 1}}]}
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    h1 = ShardCatalogHandle(
        dataset=DatasetHeader(name="ds", root=str(root), format="mds", path=str(root))
    )
    fp1 = finalize(h1)

    # Mutate the index in place (different samples) and bump mtime.
    index2 = {
        "shards": [
            {"samples": 4, "raw": {"basename": "s0.mds", "bytes": 1}},
            {"samples": 9, "raw": {"basename": "s1.mds", "bytes": 1}},
        ]
    }
    (root / "index.json").write_text(json.dumps(index2), encoding="utf-8")
    import os
    import time

    future = time.time() + 10
    os.utime(root / "index.json", (future, future))

    h2 = ShardCatalogHandle(
        dataset=DatasetHeader(name="ds", root=str(root), format="mds", path=str(root))
    )
    fp2 = finalize(h2)
    assert fp2 != fp1
    assert attach(h2).total() == 13


def test_source_sig_recognizes_underscore_index(tmp_path: Path) -> None:
    # _index.json must short-circuit the O(N) scan, like discovery. The branches
    # differ on an unrelated sibling: the index branch folds only the index file
    # (sig stable), the scan branch folds every file (sig moves).
    idx_root = tmp_path / "with_index"
    idx_root.mkdir()
    index = {"shards": [{"samples": 3, "raw": {"basename": "s0.mds", "bytes": 1}}]}
    (idx_root / "_index.json").write_text(json.dumps(index), encoding="utf-8")
    idx_header = DatasetHeader(
        name="ds", root=str(idx_root), format="mds", path=str(idx_root)
    )
    idx_sig = handle_mod._source_sig(idx_header)
    (idx_root / "s0.mds").write_bytes(b"x")
    assert handle_mod._source_sig(idx_header) == idx_sig

    scan_root = tmp_path / "no_index"
    scan_root.mkdir()
    (scan_root / "s0.mds").write_bytes(b"x")
    scan_header = DatasetHeader(
        name="ds", root=str(scan_root), format="mds", path=str(scan_root)
    )
    scan_sig = handle_mod._source_sig(scan_header)
    (scan_root / "s1.mds").write_bytes(b"x")
    assert handle_mod._source_sig(scan_header) != scan_sig


def test_catalog_is_mmap_backed_not_anonymous(
    catalog_dir: Path, tmp_path: Path
) -> None:
    # finalize/attach must register the mmap of the file on disk, not the
    # private in-memory build buffers: columns stay zero-copy, read-only views
    # into the file mapping, so all ranks/workers on a node share its pages.
    root = tmp_path / "ds"
    root.mkdir()
    _make_jsonl(root, {"a": 3, "b": 5})
    handle = ShardCatalogHandle(dataset=_header(root))
    finalize(handle)
    catalog = attach(handle)
    num_rows = catalog.num_rows()
    assert isinstance(catalog._loaded._mmap, _mmap.mmap)
    assert not num_rows.flags.writeable
    assert np.shares_memory(
        num_rows, np.frombuffer(catalog._loaded._mmap, dtype=np.uint8)
    )


# --------------------------------------------------------------------------
# catalog directory: filesystem classification + resolution precedence
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fstype", "expected"),
    [
        (None, False),  # no procfs (e.g. macOS): can't detect -> assume node-local
        ("", True),  # procfs present but mount undetermined -> fail safe
        ("ext4", False),
        ("xfs", False),
        ("nfs4", True),
        ("lustre", True),
        ("tmpfs", True),
        ("fuse.sshfs", True),  # any fuse mount
    ],
)
def test_is_network_or_ram_classifies_fstype(
    fstype: str | None, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(handle_mod, "_fs_type", lambda _p: fstype)
    assert handle_mod._is_network_or_ram(Path("/whatever")) is expected


def test_resolve_catalog_dir_prefers_local_cache_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(handle_mod, "_is_network_or_ram", lambda _p: False)
    opts = StoreOptions(cache=CacheOptions(enabled=True, root=tmp_path))
    assert resolve_catalog_dir(opts) == tmp_path / CATALOG_CACHE_SUBDIR


def test_resolve_catalog_dir_skips_network_cache_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A network/RAM-backed cache root is skipped (mmap page-sharing degrades
    # there); resolution falls through to the env override.
    monkeypatch.setattr(handle_mod, "_is_network_or_ram", lambda _p: True)
    monkeypatch.setenv("ZEPHON_CATALOG_DIR", str(tmp_path / "env"))
    opts = StoreOptions(cache=CacheOptions(enabled=True, root=tmp_path))
    assert resolve_catalog_dir(opts) == tmp_path / "env"


def test_resolve_catalog_dir_env_over_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Cache off -> skip the cache branch; the env override beats the tmp default.
    monkeypatch.setenv("ZEPHON_CATALOG_DIR", str(tmp_path / "env"))
    assert resolve_catalog_dir(StoreOptions()) == tmp_path / "env"


def test_resolve_catalog_dir_falls_back_to_tmp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No cache, no env override (``None`` mirrors the no-Engine direct/test path).
    monkeypatch.delenv("ZEPHON_CATALOG_DIR", raising=False)
    assert resolve_catalog_dir(None) == handle_mod._default_tmp_dir()
