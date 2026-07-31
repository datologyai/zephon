import gc
import json
import os
import threading
import time
import weakref
from multiprocessing import Process, Queue
from pathlib import Path
from typing import IO, Any, Mapping

import pytest

import zephon._internal.io.resolvers.cache.manager as manager_mod
from tests._catalog_helpers import catalog_set_from_locators
from zephon._internal.io.resolvers.cache import (
    CacheInUseError,
    CacheManager,
    PermanentSourceMissing,
    ShardNotReady,
)
from zephon._internal.io.resolvers.cache.shared_state import _ShardState
from zephon._internal.io.storage.local import LocalFSBackend
from zephon._internal.io.types import LocalShardRef, ShardFile, ShardLocator
from zephon.io.dataset import Dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_file(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _locator(
    dataset: str,
    shard_id: int,
    root: str,
    *,
    raw_name: str,
    raw_bytes: int,
    zip_name: str | None = None,
    zip_bytes: int = 0,
    compression: str | None = None,
    hashes: Mapping[str, str] | None = None,
) -> ShardLocator:
    raw = ShardFile(basename=raw_name, bytes=int(raw_bytes), hashes=hashes or {})
    zip_file = (
        ShardFile(basename=zip_name, bytes=int(zip_bytes), hashes={})
        if zip_name
        else None
    )
    return ShardLocator(
        dataset=dataset,
        shard_id=int(shard_id),
        format="dummy",
        root=root,
        raw=raw,
        zip=zip_file,
        compression=compression,
        extra=None,
    )


def _make_manager(
    cache_root: Path,
    storage,
    locators: list[ShardLocator],
    *,
    cls: type[CacheManager] = CacheManager,
    **kwargs: Any,
) -> CacheManager:
    """Construct a CacheManager over a CatalogSet built from synthetic locators."""
    catalog_set = catalog_set_from_locators(locators, cache_root.parent / "_catalogs")
    return cls(cache_root, storage, catalog_set=catalog_set, **kwargs)


# ---------------------------------------------------------------------------
# Single-process functional tests
# ---------------------------------------------------------------------------


def test_resolve_raw_only_downloads_and_updates_stats(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data = b"hello world" * 123
    raw_name = "shard0.bin"
    _make_file(remote / raw_name, data)

    storage = LocalFSBackend(root=remote)
    loc = _locator(
        dataset="demo",
        shard_id=0,
        root=str(remote),
        raw_name=raw_name,
        raw_bytes=len(data),
    )
    mgr = _make_manager(cache_root, storage, [loc])

    try:
        ref = mgr.resolve(loc)
        assert isinstance(ref, LocalShardRef)
        expected_local = cache_root / "demo" / raw_name
        assert ref.raw.path == expected_local
        assert ref.raw.path.is_file()
        assert ref.raw.bytes == len(data)
        assert ref.zip is None

        stats = mgr.stats()
        assert stats.shards == 1
        assert stats.bytes_used == len(data)

        # Second resolve is a cache hit and should not change stats.
        ref2 = mgr.resolve(loc)
        assert ref2.raw.path == expected_local
        assert mgr.stats() == stats
    finally:
        mgr.close()


def test_resolve_with_gzip_decompression_and_keep_zip(tmp_path: Path) -> None:
    import gzip

    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    raw_name = "raw1.txt"
    zip_name = raw_name + ".gz"
    payload = b"sample payload for gzip\n" * 10
    _make_file(remote / zip_name, gzip.compress(payload))

    storage = LocalFSBackend(root=remote)
    loc = _locator(
        dataset="ds",
        shard_id=1,
        root=str(remote),
        raw_name=raw_name,
        raw_bytes=len(payload),
        zip_name=zip_name,
        zip_bytes=(remote / zip_name).stat().st_size,
        compression="gzip",
    )

    mgr1 = _make_manager(cache_root, storage, [loc], keep_zip=False)
    try:
        ref1 = mgr1.resolve(loc)
        assert ref1.raw.path.is_file()
        assert ref1.raw.bytes == len(payload)
        assert ref1.zip is None
        assert not (cache_root / "ds" / zip_name).exists()
        stats1 = mgr1.stats()
        assert stats1.bytes_used == len(payload)
        assert stats1.shards == 1
    finally:
        mgr1.close()

    # Fresh session with keep_zip=True: wipe behavior expected because no
    # prior session record exists (FIRST_INIT always wipes).
    mgr2 = _make_manager(cache_root, storage, [loc], keep_zip=True, persist_state=False)
    try:
        ref2 = mgr2.resolve(loc)
        assert ref2.raw.path.is_file()
        assert ref2.zip is not None
        assert ref2.zip.path.exists()
        stats2 = mgr2.stats()
        assert stats2.shards == 1
        assert stats2.bytes_used == len(payload) + (remote / zip_name).stat().st_size
    finally:
        mgr2.close()


def test_validate_hash_mismatch_raises_and_removes_file(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    raw_name = "shard2.bin"
    data = b"abc" * 333
    _make_file(remote / raw_name, data)

    storage = LocalFSBackend(root=remote)
    loc = _locator(
        dataset="demo",
        shard_id=2,
        root=str(remote),
        raw_name=raw_name,
        raw_bytes=len(data),
        hashes={"md5": "0" * 32},
    )
    mgr = _make_manager(cache_root, storage, [loc], validate_hash="md5")

    try:
        with pytest.raises(ValueError):
            mgr.resolve(loc)

        expected_local = cache_root / "demo" / raw_name
        assert not expected_local.exists()
    finally:
        mgr.close()


def test_cache_manager_cleanup_runs_on_gc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    remote = tmp_path / "remote"
    remote.mkdir(parents=True, exist_ok=True)
    cache_root = tmp_path / "cache"
    called: list[bool] = []
    orig = manager_mod._close_cache_manager_resources

    def wrapped(*args, **kwargs) -> None:
        called.append(True)
        orig(*args, **kwargs)

    monkeypatch.setattr(manager_mod, "_close_cache_manager_resources", wrapped)

    loc = _locator(
        dataset="demo",
        shard_id=0,
        root=str(remote),
        raw_name="dummy.bin",
        raw_bytes=1,
    )
    mgr = _make_manager(cache_root, LocalFSBackend(root=remote), [loc])

    fin = mgr._close_finalizer
    mgr_ref = weakref.ref(mgr)
    mgr = None  # drop strong ref

    for _ in range(200):
        if not fin.alive:
            break
        gc.collect()
        time.sleep(0.01)

    assert len(called) == 1
    assert fin.alive is False
    assert mgr_ref() is None


def test_blocking_false_raises_not_ready_when_preparing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    raw_name = "shard3.bin"
    data = b"x" * 1024
    _make_file(remote / raw_name, data)

    storage = LocalFSBackend(root=remote)
    loc = _locator(
        dataset="demo",
        shard_id=3,
        root=str(remote),
        raw_name=raw_name,
        raw_bytes=len(data),
    )
    mgr = _make_manager(cache_root, storage, [loc])

    started = threading.Event()
    release = threading.Event()

    real_prepare = mgr._prepare

    def slow_prepare(*args: Any, **kwargs: Any) -> None:
        started.set()
        release.wait(timeout=5)
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(mgr, "_prepare", slow_prepare)

    result: dict[str, Any] = {}

    def run_resolve() -> None:
        try:
            result["ref"] = mgr.resolve(loc)
        except Exception as exc:  # pragma: no cover
            result["err"] = exc

    t = threading.Thread(target=run_resolve)
    t.start()
    assert started.wait(timeout=2)

    with pytest.raises(ShardNotReady):
        mgr.resolve(loc, blocking=False)

    release.set()
    t.join(timeout=5)
    assert "ref" in result
    assert isinstance(result["ref"], LocalShardRef)
    mgr.close()


def test_missing_source_raises_permanent(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    raw_name = "missing.bin"

    storage = LocalFSBackend(root=remote)
    loc = _locator(
        dataset="demo",
        shard_id=4,
        root=str(remote),
        raw_name=raw_name,
        raw_bytes=123,
    )
    mgr = _make_manager(cache_root, storage, [loc])

    try:
        with pytest.raises(PermanentSourceMissing):
            mgr.resolve(loc)
    finally:
        mgr.close()


def test_re_resolve_recovers_if_local_file_missing(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    raw_name = "shard4.bin"
    data = b"z" * 8192
    _make_file(remote / raw_name, data)

    storage = LocalFSBackend(root=remote)
    loc = _locator("demo", 5, str(remote), raw_name=raw_name, raw_bytes=len(data))
    mgr = _make_manager(cache_root, storage, [loc])
    try:
        ref = mgr.resolve(loc)
        assert ref.raw.path.exists()
        stats1 = mgr.stats()

        ref.raw.path.unlink()

        ref2 = mgr.resolve(loc)
        assert ref2.raw.path.exists()
        stats2 = mgr.stats()
        assert stats2.shards == 1
        assert stats2.bytes_used == stats1.bytes_used
    finally:
        mgr.close()


def test_limit_too_small_raises(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    raw_name = "big.bin"
    data = b"A" * 4096
    _make_file(remote / raw_name, data)

    storage = LocalFSBackend(root=remote)
    slack = max(512 * 1024, int(len(data) * 0.005))
    slack = min(64 * 1024 * 1024, slack)
    limit_bytes = len(data) + slack - 1
    loc = _locator("demo", 6, str(remote), raw_name=raw_name, raw_bytes=len(data))
    mgr = _make_manager(cache_root, storage, [loc], limit_bytes=limit_bytes)
    try:
        with pytest.raises(ValueError):
            mgr.resolve(loc)
    finally:
        mgr.close()


def test_eviction_chooses_coldest_when_capacity_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data_a = b"A" * 1024
    data_b = b"B" * 1024
    data_c = b"C" * 5120
    _make_file(remote / "a.bin", data_a)
    _make_file(remote / "b.bin", data_b)
    _make_file(remote / "c.bin", data_c)

    storage = LocalFSBackend(root=remote)
    loc_a = _locator("demo", 10, str(remote), raw_name="a.bin", raw_bytes=len(data_a))
    loc_b = _locator("demo", 11, str(remote), raw_name="b.bin", raw_bytes=len(data_b))
    loc_c = _locator("demo", 12, str(remote), raw_name="c.bin", raw_bytes=len(data_c))

    mgr = _make_manager(
        cache_root, storage, [loc_a, loc_b, loc_c], limit_bytes=6 * 1024
    )
    try:
        monkeypatch.setattr(mgr, "_required_bytes", lambda loc: int(loc.raw.bytes))

        ref_a = mgr.resolve(loc_a)
        assert ref_a.raw.path.exists()
        mgr.touch(loc_a)
        time.sleep(0.01)

        ref_b = mgr.resolve(loc_b)
        assert ref_b.raw.path.exists()

        ref_c = mgr.resolve(loc_c)
        assert ref_c.raw.path.exists()

        path_a = cache_root / "demo" / "a.bin"
        path_b = cache_root / "demo" / "b.bin"
        path_c = cache_root / "demo" / "c.bin"
        assert not path_a.exists()
        assert path_b.exists()
        assert path_c.exists()
    finally:
        mgr.close()


def test_unsupported_compression_raises_and_cleans_part_files(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    raw_name = "raw.txt"
    zip_name = "raw.txt.bad"
    _make_file(remote / zip_name, b"not actually compressed")

    storage = LocalFSBackend(root=remote)
    loc = _locator(
        dataset="demo",
        shard_id=20,
        root=str(remote),
        raw_name=raw_name,
        raw_bytes=1,
        zip_name=zip_name,
        zip_bytes=(remote / zip_name).stat().st_size,
        compression="unknown",
    )
    mgr = _make_manager(cache_root, storage, [loc])
    try:
        with pytest.raises(ValueError):
            mgr.resolve(loc)

        ds_root = cache_root / "demo"
        for p in ds_root.rglob("*.part"):
            pytest.fail(f"Unexpected leftover part file: {p}")
        for p in ds_root.rglob("*.tmp"):
            pytest.fail(f"Unexpected leftover tmp file: {p}")
    finally:
        mgr.close()


def test_download_retry_succeeds_after_transient_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    raw_name = "flaky.bin"
    data = b"retry me" * 100
    _make_file(remote / raw_name, data)

    class FlakyStorage:
        def __init__(self, root: Path) -> None:
            self.root = root
            self.calls: dict[str, int] = {}

        def open(
            self, path: str, mode: str = "rb", **kwargs: Any
        ) -> IO[bytes] | IO[str]:
            return open(path, mode, **kwargs)

        def exists(self, path: str) -> bool:
            return Path(path).exists()

        def download(self, src: str, dst: str, timeout: float | None = None) -> None:
            self.calls[src] = self.calls.get(src, 0) + 1
            if self.calls[src] == 1:
                raise IOError("transient failure")
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            contents = Path(src).read_bytes()
            Path(dst).write_bytes(contents)

        def listdir(self, path: str) -> list[str]:  # pragma: no cover
            return []

        def stat(self, path: str) -> Mapping[str, int]:  # pragma: no cover
            return {"size": Path(path).stat().st_size}

    storage = FlakyStorage(remote)
    loc = _locator("demo", 30, str(remote), raw_name=raw_name, raw_bytes=len(data))
    mgr = _make_manager(cache_root, storage, [loc], download_retry=2)

    try:
        monkeypatch.setattr(time, "sleep", lambda _: None)
        ref = mgr.resolve(loc)
        assert ref.raw.path.read_bytes() == data
        src_full = os.path.join(str(remote), raw_name)
        assert storage.calls.get(src_full) == 2
    finally:
        mgr.close()


# ---------------------------------------------------------------------------
# Session state machine: six branches from plan §5
# ---------------------------------------------------------------------------


def test_first_init_wipes_preexisting_files(tmp_path: Path) -> None:
    """No session.json: first owner wipes regardless of persist_state."""
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    raw_name = "persist.bin"
    data = b"persist test"
    _make_file(remote / raw_name, data)
    # Pre-existing file in the cache root that isn't from a tracked session.
    orphan_dir = cache_root / "demo"
    _make_file(orphan_dir / "leftover.bin", b"old run")

    storage = LocalFSBackend(root=remote)
    loc = _locator("demo", 40, str(remote), raw_name=raw_name, raw_bytes=len(data))
    mgr = _make_manager(cache_root, storage, [loc], persist_state=True)
    try:
        assert not (orphan_dir / "leftover.bin").exists()
    finally:
        mgr.close()


def test_first_init_wipe_preserves_catalog_subdir(tmp_path: Path) -> None:
    """The node-local shard catalog under ``.catalog`` survives a cache wipe."""
    from zephon._internal.io.catalog import CATALOG_CACHE_SUBDIR

    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    raw_name = "persist.bin"
    data = b"persist test"
    _make_file(remote / raw_name, data)
    catalog_file = cache_root / CATALOG_CACHE_SUBDIR / "v1" / "sha256:feedface"
    _make_file(catalog_file, b"catalog-bytes")
    _make_file(cache_root / "demo" / "leftover.bin", b"old run")

    storage = LocalFSBackend(root=remote)
    loc = _locator("demo", 40, str(remote), raw_name=raw_name, raw_bytes=len(data))
    mgr = _make_manager(cache_root, storage, [loc])
    try:
        # FIRST_INIT wiped the stale shard but not the co-located catalog.
        assert not (cache_root / "demo" / "leftover.bin").exists()
        assert catalog_file.read_bytes() == b"catalog-bytes"
    finally:
        mgr.close()


def test_fresh_reset_wipes_on_new_run_persist_state_false(tmp_path: Path) -> None:
    """persist_state=False + no live owners: wipe between runs."""
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data = b"persist test"
    raw_name = "persist.bin"
    _make_file(remote / raw_name, data)

    storage = LocalFSBackend(root=remote)
    loc = _locator("demo", 40, str(remote), raw_name=raw_name, raw_bytes=len(data))

    mgr1 = _make_manager(cache_root, storage, [loc], persist_state=False)
    try:
        ref = mgr1.resolve(loc)
        assert ref.raw.path.exists()

        # New manager while first is still alive: JOIN, no wipe.
        mgr_same = _make_manager(cache_root, storage, [loc], persist_state=False)
        try:
            assert (cache_root / "demo" / raw_name).exists()
        finally:
            mgr_same.close()
    finally:
        mgr1.close()

    # All owners gone: next init wipes (persist_state=False).
    mgr2 = _make_manager(cache_root, storage, [loc], persist_state=False)
    try:
        assert not (cache_root / "demo" / raw_name).exists()
    finally:
        mgr2.close()


def test_resume_preserves_files_with_matching_fingerprint(tmp_path: Path) -> None:
    """persist_state=True + matching fingerprint + no live owners: RESUME.

    Files on disk survive and are reconciled to LOCAL.
    """
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data = b"resume payload" * 8
    raw_name = "resume.bin"
    _make_file(remote / raw_name, data)

    storage = LocalFSBackend(root=remote)
    loc = _locator("demo", 50, str(remote), raw_name=raw_name, raw_bytes=len(data))

    mgr1 = _make_manager(cache_root, storage, [loc], persist_state=True)
    try:
        ref = mgr1.resolve(loc)
        assert ref.raw.path.exists()
    finally:
        mgr1.close()

    file_path = cache_root / "demo" / raw_name
    assert file_path.exists()

    # Second run, same locator set -> fingerprint matches -> RESUME.
    mgr2 = _make_manager(cache_root, storage, [loc], persist_state=True)
    try:
        assert file_path.exists(), "file should survive RESUME"
        stats = mgr2.stats()
        assert stats.shards == 1
        assert stats.bytes_used == len(data)

        # Subsequent resolve is a cache hit — no re-download.
        ref2 = mgr2.resolve(loc)
        assert ref2.cache_hit is True
    finally:
        mgr2.close()


def test_cold_reset_on_fingerprint_mismatch_persist_state_true(tmp_path: Path) -> None:
    """persist_state=True + fingerprint mismatch + no live owners: COLD_RESET."""
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data_v1 = b"v1 content"
    data_v2 = b"v2 content that differs"
    raw_name = "shifting.bin"
    _make_file(remote / raw_name, data_v1)

    storage = LocalFSBackend(root=remote)
    loc_v1 = _locator(
        "demo", 60, str(remote), raw_name=raw_name, raw_bytes=len(data_v1)
    )

    mgr1 = _make_manager(cache_root, storage, [loc_v1], persist_state=True)
    try:
        mgr1.resolve(loc_v1)
        assert (cache_root / "demo" / raw_name).exists()
    finally:
        mgr1.close()

    # Rewrite the source with different bytes; locator declares the new size
    # so the fingerprint changes.
    _make_file(remote / raw_name, data_v2)
    loc_v2 = _locator(
        "demo", 60, str(remote), raw_name=raw_name, raw_bytes=len(data_v2)
    )

    mgr2 = _make_manager(cache_root, storage, [loc_v2], persist_state=True)
    try:
        # Cache root was wiped on mismatch; subsequent resolve re-downloads
        # the new bytes.
        ref = mgr2.resolve(loc_v2)
        assert ref.raw.path.read_bytes() == data_v2
    finally:
        mgr2.close()


def test_hard_error_on_fingerprint_mismatch_with_live_owners(tmp_path: Path) -> None:
    """Live owners + fingerprint mismatch: refuse to reset, raise CacheInUseError.

    Also asserts the per-dataset diff is included in the message: the
    second manager registers an extra shard for the same dataset, so the
    operator should see ``shard_count`` and ``total_raw_bytes`` deltas
    plus both fingerprints, not just an opaque hex digest.
    """
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data = b"content"
    _make_file(remote / "a.bin", data)
    _make_file(remote / "b.bin", data)

    storage = LocalFSBackend(root=remote)
    loc_a = _locator("demo", 70, str(remote), raw_name="a.bin", raw_bytes=len(data))
    loc_b = _locator("demo", 71, str(remote), raw_name="b.bin", raw_bytes=len(data))

    # First manager registers one shard in the session; we hold it open while
    # a second manager with a different locator set tries to join.
    mgr1 = _make_manager(cache_root, storage, [loc_a], persist_state=True)
    try:
        with pytest.raises(CacheInUseError) as excinfo:
            _make_manager(cache_root, storage, [loc_a, loc_b], persist_state=True)
    finally:
        mgr1.close()

    err = excinfo.value
    assert err.current_fingerprint is not None
    assert err.existing_fingerprint != err.current_fingerprint
    msg = str(err)
    # Both fingerprints visible.
    assert err.existing_fingerprint in msg
    assert err.current_fingerprint in msg
    # Structured per-dataset deltas, not just a hash.
    assert "Differences (per-dataset summary):" in msg
    diff_text = "\n".join(err.diff_lines)
    assert "shard_count" in diff_text
    assert "1" in diff_text and "2" in diff_text  # 1 -> 2 shards
    assert "total_raw_bytes" in diff_text
    assert "summary missing" not in msg


def test_hard_error_diff_falls_back_when_legacy_session_lacks_summary(
    tmp_path: Path,
) -> None:
    """Sessions written before the summary schema landed downgrade gracefully.

    The first manager writes session.json, then we strip the ``summary``
    key by hand to simulate an upgrade-in-place. The second manager hits
    HARD_ERROR and the message reports "summary missing" rather than
    pretending there's structured data to diff.
    """
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data = b"legacy"
    _make_file(remote / "a.bin", data)
    _make_file(remote / "b.bin", data)

    storage = LocalFSBackend(root=remote)
    loc_a = _locator("demo", 72, str(remote), raw_name="a.bin", raw_bytes=len(data))
    loc_b = _locator("demo", 73, str(remote), raw_name="b.bin", raw_bytes=len(data))

    mgr1 = _make_manager(cache_root, storage, [loc_a], persist_state=True)
    try:
        # Strip the summary key in place to simulate a pre-schema session.json.
        session_path = cache_root / ".zephon_cache_state" / "session.json"
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session.pop("summary", None)
        session_path.write_text(json.dumps(session, sort_keys=True), encoding="utf-8")

        with pytest.raises(CacheInUseError) as excinfo:
            _make_manager(cache_root, storage, [loc_a, loc_b], persist_state=True)
    finally:
        mgr1.close()

    err = excinfo.value
    msg = str(err)
    assert "summary missing" in msg
    # Both fingerprints still visible even on the fallback path.
    assert err.existing_fingerprint in msg
    assert err.current_fingerprint is not None
    assert err.current_fingerprint in msg


def test_join_succeeds_with_matching_fingerprint_and_live_owners(
    tmp_path: Path,
) -> None:
    """JOIN branch: matching fingerprint + live owner increments instances."""
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data = b"joinable"
    _make_file(remote / "j.bin", data)
    storage = LocalFSBackend(root=remote)
    loc = _locator("demo", 80, str(remote), raw_name="j.bin", raw_bytes=len(data))

    mgr1 = _make_manager(cache_root, storage, [loc], persist_state=True)
    try:
        mgr1.resolve(loc)
        mgr2 = _make_manager(cache_root, storage, [loc], persist_state=True)
        try:
            # Same session — the file is still present and a resolve is a hit.
            ref = mgr2.resolve(loc)
            assert ref.cache_hit is True
        finally:
            mgr2.close()
    finally:
        mgr1.close()


def test_reconciliation_ignores_tmp_part_and_lock_files(tmp_path: Path) -> None:
    """RESUME scan skips .tmp / .part / .locks entries."""
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data = b"reconcile me"
    raw_name = "rec.bin"
    _make_file(remote / raw_name, data)

    storage = LocalFSBackend(root=remote)
    loc = _locator("demo", 90, str(remote), raw_name=raw_name, raw_bytes=len(data))

    mgr1 = _make_manager(cache_root, storage, [loc], persist_state=True)
    try:
        mgr1.resolve(loc)
    finally:
        mgr1.close()

    # Plant noise into the dataset dir that reconciliation must ignore.
    ds_dir = cache_root / "demo"
    (ds_dir / "rec.bin.tmp").write_bytes(b"tmp")
    (ds_dir / "rec.bin.part").write_bytes(b"part")
    (ds_dir / ".locks").mkdir(exist_ok=True)
    (ds_dir / ".locks" / "90.lock").write_bytes(b"")
    (ds_dir / "orphan.bin").write_bytes(b"not a known shard")

    mgr2 = _make_manager(cache_root, storage, [loc], persist_state=True)
    try:
        stats = mgr2.stats()
        # Only the raw shard should count.
        assert stats.shards == 1
        assert stats.bytes_used == len(data)
    finally:
        mgr2.close()


def test_reconciliation_zip_only_is_not_local(tmp_path: Path) -> None:
    """A dataset dir containing only the zip file for a shard stays REMOTE."""
    import gzip

    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    raw_name = "zonly.txt"
    zip_name = raw_name + ".gz"
    payload = b"zip only body"
    _make_file(remote / zip_name, gzip.compress(payload))

    storage = LocalFSBackend(root=remote)
    loc = _locator(
        dataset="demo",
        shard_id=95,
        root=str(remote),
        raw_name=raw_name,
        raw_bytes=len(payload),
        zip_name=zip_name,
        zip_bytes=(remote / zip_name).stat().st_size,
        compression="gzip",
    )

    mgr1 = _make_manager(cache_root, storage, [loc], persist_state=True, keep_zip=True)
    try:
        ref = mgr1.resolve(loc)
        assert ref.zip is not None and ref.zip.path.exists()
    finally:
        mgr1.close()

    # Delete the raw but keep the zip: RESUME should leave shard REMOTE.
    (cache_root / "demo" / raw_name).unlink(missing_ok=True)

    mgr2 = _make_manager(cache_root, storage, [loc], persist_state=True, keep_zip=True)
    try:
        stats = mgr2.stats()
        assert stats.shards == 0
        assert stats.bytes_used == 0
    finally:
        mgr2.close()


def test_reconciliation_includes_zip_bytes_when_keep_zip_true(tmp_path: Path) -> None:
    import gzip

    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    raw_name = "rzk.txt"
    zip_name = raw_name + ".gz"
    payload = b"payload for zip-kept reconcile" * 4
    _make_file(remote / zip_name, gzip.compress(payload))

    storage = LocalFSBackend(root=remote)
    zip_size = (remote / zip_name).stat().st_size
    loc = _locator(
        dataset="demo",
        shard_id=96,
        root=str(remote),
        raw_name=raw_name,
        raw_bytes=len(payload),
        zip_name=zip_name,
        zip_bytes=zip_size,
        compression="gzip",
    )

    mgr1 = _make_manager(cache_root, storage, [loc], persist_state=True, keep_zip=True)
    try:
        mgr1.resolve(loc)
    finally:
        mgr1.close()

    mgr2 = _make_manager(cache_root, storage, [loc], persist_state=True, keep_zip=True)
    try:
        stats = mgr2.stats()
        assert stats.shards == 1
        assert stats.bytes_used == len(payload) + zip_size
    finally:
        mgr2.close()


# ---------------------------------------------------------------------------
# Multiprocess coordination
# ---------------------------------------------------------------------------


class SlowLocalFSBackend(LocalFSBackend):
    """Local backend that sleeps in download to simulate long PREPARING."""

    def download(self, src: str, dst: str, timeout: float | None = None) -> None:
        Path(self.root / "download_started").write_text("start", encoding="utf-8")
        deadline = time.time() + 5.0
        while time.time() < deadline and not (self.root / "allow_finish").exists():
            time.sleep(0.05)
        super().download(src, dst, timeout)


def _proc_resolve_slow(
    cache_root: str, remote: str, loc: ShardLocator, q: Queue
) -> None:  # type: ignore[no-redef]
    mgr = _make_manager(
        Path(cache_root),
        SlowLocalFSBackend(root=Path(remote)),
        [loc],
        persist_state=True,
    )
    try:
        ref = mgr.resolve(loc)
        q.put({"ok": True, "path": str(ref.raw.path)})
    except Exception as exc:  # pragma: no cover
        q.put({"ok": False, "err": repr(exc)})
    finally:
        mgr.close()


def test_multiprocess_preparing_blocks_and_unblocks(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data = b"M" * 4096
    raw_name = "mp.bin"
    _make_file(remote / raw_name, data)

    loc = _locator("mp", 1, str(remote), raw_name=raw_name, raw_bytes=len(data))

    q: Queue = Queue()
    p = Process(target=_proc_resolve_slow, args=(str(cache_root), str(remote), loc, q))
    p.start()

    started_flag = remote / "download_started"
    for _ in range(50):
        if started_flag.exists():
            break
        time.sleep(0.05)
    assert started_flag.exists()

    mgr2 = _make_manager(
        cache_root, LocalFSBackend(root=remote), [loc], persist_state=True
    )
    with pytest.raises(ShardNotReady):
        mgr2.resolve(loc, blocking=False)

    (remote / "allow_finish").write_text("ok", encoding="utf-8")

    ref2 = mgr2.resolve(loc, blocking=True)
    assert ref2.raw.path.exists()

    p.join(timeout=10)
    assert p.exitcode == 0
    res = q.get_nowait()
    assert res["ok"] is True
    mgr2.close()
    (remote / "download_started").unlink(missing_ok=True)
    (remote / "allow_finish").unlink(missing_ok=True)


class TinySlackCacheManager(CacheManager):
    def _required_bytes(self, locator: ShardLocator) -> int:  # type: ignore[override]
        return int(locator.raw.bytes)


def _proc_resolve_with_tiny_slack(
    cache_root: str, remote: str, loc: ShardLocator, q: Queue
) -> None:  # type: ignore[no-redef]
    mgr = _make_manager(
        Path(cache_root),
        LocalFSBackend(root=Path(remote)),
        [loc],
        cls=TinySlackCacheManager,
        limit_bytes=5 * 1024,
        persist_state=True,
    )
    try:
        ref = mgr.resolve(loc)
        q.put({"ok": True, "path": str(ref.raw.path)})
    except Exception as exc:  # pragma: no cover
        q.put({"ok": False, "err": repr(exc)})
    finally:
        mgr.close()


def _proc_resolve_with_full_set(
    cache_root: str,
    remote: str,
    locators: list[ShardLocator],
    target: ShardLocator,
    q: Queue,
) -> None:  # type: ignore[no-redef]
    mgr = _make_manager(
        Path(cache_root),
        LocalFSBackend(root=Path(remote)),
        locators,
        cls=TinySlackCacheManager,
        limit_bytes=5 * 1024,
        persist_state=True,
    )
    try:
        ref = mgr.resolve(target)
        q.put({"ok": True, "path": str(ref.raw.path)})
    except Exception as exc:  # pragma: no cover
        q.put({"ok": False, "err": repr(exc)})
    finally:
        mgr.close()


def test_multiprocess_eviction_coldest(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data_a = b"A" * 2048
    data_b = b"B" * 2048
    data_c = b"C" * 3072
    _make_file(remote / "a.bin", data_a)
    _make_file(remote / "b.bin", data_b)
    _make_file(remote / "c.bin", data_c)

    loc_a = _locator("demo", 100, str(remote), raw_name="a.bin", raw_bytes=len(data_a))
    loc_b = _locator("demo", 101, str(remote), raw_name="b.bin", raw_bytes=len(data_b))
    loc_c = _locator("demo", 102, str(remote), raw_name="c.bin", raw_bytes=len(data_c))

    # All three locators must be registered up-front so every process shares
    # the same dense index and fingerprint.
    mgr_parent = _make_manager(
        cache_root,
        LocalFSBackend(root=remote),
        [loc_a, loc_b, loc_c],
        cls=TinySlackCacheManager,
        limit_bytes=5 * 1024,
        persist_state=True,
    )
    _ = mgr_parent.resolve(loc_a)
    time.sleep(0.02)

    q: Queue = Queue()
    p = Process(
        target=_proc_resolve_with_full_set,
        args=(str(cache_root), str(remote), [loc_a, loc_b, loc_c], loc_b, q),
    )
    p.start()
    p.join(timeout=10)
    assert p.exitcode == 0
    child_res = q.get_nowait()
    assert child_res["ok"] is True

    _ = mgr_parent.resolve(loc_c)

    path_a = cache_root / "demo" / "a.bin"
    path_b = cache_root / "demo" / "b.bin"
    path_c = cache_root / "demo" / "c.bin"
    idx_a = mgr_parent._catalog_set.slot_of("demo", 100)
    idx_b = mgr_parent._catalog_set.slot_of("demo", 101)
    idx_c = mgr_parent._catalog_set.slot_of("demo", 102)
    shared = mgr_parent._shared
    assert shared is not None
    state_a = _ShardState(shared.shard_states[idx_a])
    state_b = _ShardState(shared.shard_states[idx_b])
    state_c = _ShardState(shared.shard_states[idx_c])
    assert state_a != _ShardState.LOCAL
    assert state_b == _ShardState.LOCAL
    assert state_c == _ShardState.LOCAL
    remaining_files = [p for p in (path_a, path_b, path_c) if p.exists()]
    assert len(remaining_files) <= 2
    mgr_parent.close()


# ---------------------------------------------------------------------------
# Regression: SHM survives creator exit while joiners are live
# ---------------------------------------------------------------------------


def test_creator_exit_leaves_shm_attachable_for_future_joiners(tmp_path: Path) -> None:
    """A creates SHM, B joins, A exits, C must still be able to join.

    Regression: earlier SHM lifetime was tied to the creator manager.
    When A exited first, its finalizer unlinked names that B still held
    and C would fail to attach. The current model only unlinks when the
    last session owner departs.
    """
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data = b"shm-lifetime" * 8
    _make_file(remote / "shlife.bin", data)
    storage = LocalFSBackend(root=remote)
    loc = _locator("demo", 200, str(remote), raw_name="shlife.bin", raw_bytes=len(data))

    mgr_a = _make_manager(cache_root, storage, [loc], persist_state=True)
    mgr_b = _make_manager(cache_root, storage, [loc], persist_state=True)
    mgr_a.resolve(loc)
    # A (the creator) exits first while B is still attached.
    mgr_a.close()

    # C joins now; SHM names must still resolve because B holds the session.
    mgr_c = _make_manager(cache_root, storage, [loc], persist_state=True)
    try:
        ref = mgr_c.resolve(loc)
        assert ref.cache_hit is True
    finally:
        mgr_c.close()
        mgr_b.close()


# ---------------------------------------------------------------------------
# Guards: all-inmem and duplicate-name datasets
# ---------------------------------------------------------------------------


def test_build_resolver_with_all_inmem_datasets_does_not_crash(tmp_path: Path) -> None:
    """Cache enabled + no file-backed shards: builder must not blow up.

    Before the fix, CacheManager was instantiated with num_shards=0 and
    CacheSharedState rejected capacity <= 0.
    """
    from zephon._internal.io.protocols import RandomAccessShard
    from zephon._internal.io.resolvers import DirectResolver
    from zephon._internal.io.stores.multi import build_catalog_set, build_resolver
    from zephon.io.options import StoreOptions

    # An all-inmem datasets map yields no catalog set.
    inmem_shards: Mapping[int, RandomAccessShard] = {}
    datasets = {
        0: Dataset(
            name="only-inmem",
            backend={"kind": "inmem", "shards": inmem_shards},
            path=None,
        )
    }
    cache_root = tmp_path / "cache"
    store_opts = StoreOptions.from_any(
        {"cache": {"enabled": True, "root": str(cache_root)}}
    )
    catalog_set = build_catalog_set(datasets)
    assert catalog_set is None
    resolver = build_resolver(catalog_set, options=store_opts)
    # No cacheable shards => fall back to DirectResolver rather than
    # building a zero-capacity CacheManager.
    assert isinstance(resolver, DirectResolver)


def test_duplicate_cacheable_dataset_names_raise(tmp_path: Path) -> None:
    """Two file-backed datasets with the same .name must not silently collide."""
    from zephon._internal.io.stores.multi import build_catalog_set

    remote = tmp_path / "remote"
    _make_file(remote / "x.bin", b"x")
    datasets = {
        0: Dataset(
            name="same",
            backend={"kind": "dummy", "path": str(remote)},
            path=str(remote),
        ),
        1: Dataset(
            name="same",
            backend={"kind": "dummy", "path": str(remote)},
            path=str(remote),
        ),
    }
    with pytest.raises(ValueError, match="Duplicate dataset name"):
        build_catalog_set(datasets)


def test_duplicate_name_rejected_across_inmem_and_file_backed(
    tmp_path: Path,
) -> None:
    """Inmem + file-backed sharing a .name is also ambiguous and must raise.

    The inmem dataset never touches the cache, but its .name would still
    collide semantically with the file-backed one in any by-name lookup.
    """
    from zephon._internal.io.protocols import RandomAccessShard
    from zephon._internal.io.stores.multi import build_catalog_set

    remote = tmp_path / "remote"
    _make_file(remote / "x.bin", b"x")
    inmem_shards: Mapping[int, RandomAccessShard] = {}
    datasets = {
        0: Dataset(
            name="same",
            backend={"kind": "dummy", "path": str(remote)},
            path=str(remote),
        ),
        1: Dataset(
            name="same",
            backend={"kind": "inmem", "shards": inmem_shards},
            path=None,
        ),
    }
    with pytest.raises(ValueError, match="Duplicate dataset name"):
        build_catalog_set(datasets)


def test_join_recovers_when_shm_missing_behind_live_owners(tmp_path: Path) -> None:
    """session.json with live owners + missing SHM must fall back to stale branch.

    Reproduces the CI failure where the creator had already unlinked SHM
    while joiners were still registered as live owners of the session.
    The join attempt must not raise FileNotFoundError — it must
    gracefully downgrade into the no-live-owners branch.

    Exercising the *live_owners* path (rather than the empty-owners
    stale-session path) requires mgr1 to stay alive when mgr2 is
    constructed, so session.json still names an active owner. Closing
    mgr1 first would zero the owner count and route through the
    stale-session branch instead, bypassing the recovery logic.
    """
    from zephon._internal.io.resolvers.cache.shared_state import CacheSharedState

    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data = b"recovery" * 4
    _make_file(remote / "r.bin", data)
    storage = LocalFSBackend(root=remote)
    loc = _locator("demo", 300, str(remote), raw_name="r.bin", raw_bytes=len(data))

    mgr1 = _make_manager(cache_root, storage, [loc], persist_state=True)
    try:
        mgr1.resolve(loc)

        # mgr1 is still alive — session.json records us as a live owner.
        # Unlink SHM behind mgr1's back to simulate /dev/shm cleanup or a
        # creator-exits-first race on an older manager build.
        session = json.loads(mgr1._session_path.read_text(encoding="utf-8"))
        shm_names = session["shm_names"]
        assert session["owners"], "sanity: mgr1 should be registered as owner"
        CacheSharedState.unlink_by_names(shm_names)

        # Construct mgr2 while mgr1 is still alive. mgr2 must hit the
        # live_owners + FileNotFoundError branch specifically (not the
        # empty-owners stale-session branch) and recover.
        mgr2 = _make_manager(cache_root, storage, [loc], persist_state=True)
        try:
            ref = mgr2.resolve(loc)
            assert ref.raw.path.exists()
            # After recovery, session.json is rewritten with fresh SHM
            # names. Verify the new SHM is actually attached by both
            # managers independently from mgr2's perspective.
            new_session = json.loads(mgr2._session_path.read_text(encoding="utf-8"))
            assert new_session["shm_names"] != shm_names
        finally:
            mgr2.close()
    finally:
        mgr1.close()


# ---------------------------------------------------------------------------
# pid-recycle detection (procfs start-time)
# ---------------------------------------------------------------------------


def test_parse_proc_stat_starttime_extracts_field_22() -> None:
    """Field 22 of /proc/<pid>/stat is starttime; parser must handle the
    awkward ``comm`` field (spaces, parens) by anchoring to the rightmost
    ``)`` rather than tokenising the whole line.

    Factored out as a standalone helper specifically so this parsing logic
    is unit-testable on platforms without ``/proc`` (Mac CI, etc.).
    """
    from zephon._internal.io.resolvers.cache.manager import _parse_proc_stat_starttime

    # Build a tail of exactly 20 fields (state through starttime). Index
    # 19 in this tail is starttime — set it to 4242 with a sentinel; the
    # other slots are filler. Trailing fields (vsize, rss, ...) are
    # appended to mirror real procfs output and prove the parser doesn't
    # care about them.
    after_comm = ["S"] + ["0"] * 18 + ["4242"] + ["111", "222", "333"]
    tail = (" ".join(after_comm)).encode()

    # Plain comm with no special chars.
    stat = b"123 (python) " + tail
    assert _parse_proc_stat_starttime(stat) == 4242

    # Comm with spaces and parens — common for renamed processes — must
    # not confuse the parser since we anchor on the rightmost ``)``.
    stat_tricky = b"123 (weird (proc) name) " + tail
    assert _parse_proc_stat_starttime(stat_tricky) == 4242

    # Malformed inputs return None rather than raising.
    assert _parse_proc_stat_starttime(b"") is None
    assert _parse_proc_stat_starttime(b"123 nocommparen") is None
    assert _parse_proc_stat_starttime(b"123 (foo) S 1 2 3") is None


def test_owner_entry_records_compound_key_and_process_start_ns(
    tmp_path: Path,
) -> None:
    """FIRST_INIT writes a compound owner key and the process start-time.

    Belt-and-suspenders identity: the key itself encodes
    ``f"{pid}-{start_time_ns}"`` and the entry value carries
    ``process_start_ns``, so either signal alone is sufficient to detect
    a recycled pid on the next prune.
    """
    from zephon._internal.io.resolvers.cache.manager import _process_start_time_ns

    if _process_start_time_ns(os.getpid()) is None:
        pytest.skip("procfs/psutil unavailable on this platform")

    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data = b"x"
    _make_file(remote / "a.bin", data)
    storage = LocalFSBackend(root=remote)
    loc = _locator("demo", 100, str(remote), raw_name="a.bin", raw_bytes=len(data))

    mgr = _make_manager(cache_root, storage, [loc], persist_state=True)
    try:
        session = json.loads(
            (cache_root / ".zephon_cache_state" / "session.json").read_text(
                encoding="utf-8"
            )
        )
        start_ns = _process_start_time_ns(os.getpid())
        expected_key = f"{os.getpid()}-{start_ns}"
        assert expected_key in session["owners"]
        owner = session["owners"][expected_key]
        assert owner.get("process_start_ns") == start_ns
    finally:
        mgr.close()


def test_recycled_pid_does_not_block_init_when_start_time_mismatches(
    tmp_path: Path,
) -> None:
    """A ghost owner (live pid + stale start-time) is pruned, no HARD_ERROR.

    Simulates the scenario where a prior run crashed without cleanup and
    its owner pid was later reused by an unrelated process. Without the
    start-time check, the new manager would see a "live" owner with a
    different fingerprint and refuse to reset. With it, the bogus
    start-time exposes the ghost via *both* signals (the compound key's
    suffix and the entry's ``process_start_ns``) and we COLD_RESET.
    """
    from zephon._internal.io.resolvers.cache.manager import _process_start_time_ns

    if _process_start_time_ns(os.getpid()) is None:
        pytest.skip("procfs/psutil unavailable on this platform")

    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data = b"recycle"
    _make_file(remote / "a.bin", data)
    _make_file(remote / "b.bin", data)
    storage = LocalFSBackend(root=remote)
    loc_a = _locator("demo", 110, str(remote), raw_name="a.bin", raw_bytes=len(data))
    loc_b = _locator("demo", 111, str(remote), raw_name="b.bin", raw_bytes=len(data))

    # Seed a session with the v1 fingerprint, then close cleanly so the
    # owner is dropped — leaves session.json with empty owners and stale
    # SHM names that the next init will unlink.
    mgr1 = _make_manager(cache_root, storage, [loc_a], persist_state=True)
    mgr1.close()

    # Hand-craft a ghost owner: our current pid (so os.kill says "alive")
    # but with a bogus start-time in BOTH the compound key and the entry
    # value. Either alone would trip the recycle check; together they
    # form belt-and-suspenders identity.
    session_path = cache_root / ".zephon_cache_state" / "session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    bogus_key = f"{os.getpid()}-1"  # start-time "1ns" — far from current
    session["owners"] = {
        bogus_key: {
            "instances": 1,
            "started_ns": int(time.time_ns()),
            "process_start_ns": 1,
        }
    }
    session_path.write_text(json.dumps(session, sort_keys=True), encoding="utf-8")

    # Different locator set ⇒ fingerprint mismatch. With pid-recycle
    # detection, the ghost is pruned and we COLD_RESET cleanly.
    mgr2 = _make_manager(cache_root, storage, [loc_a, loc_b], persist_state=True)
    try:
        ref = mgr2.resolve(loc_a)
        assert ref.raw.path.exists()
    finally:
        mgr2.close()


def test_legacy_owner_entry_without_start_time_is_treated_as_alive(
    tmp_path: Path,
) -> None:
    """Sessions written before this schema landed still work.

    The owner key is a bare ``str(pid)`` and the entry has no
    ``process_start_ns``; ``_parse_owner_key`` returns ``(pid, None)``,
    prune falls back to the bare kill check, sees the pid is alive, and
    JOIN proceeds normally.
    """
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data = b"legacy"
    _make_file(remote / "a.bin", data)
    storage = LocalFSBackend(root=remote)
    loc = _locator("demo", 120, str(remote), raw_name="a.bin", raw_bytes=len(data))

    mgr1 = _make_manager(cache_root, storage, [loc], persist_state=True)
    try:
        mgr1.resolve(loc)  # populate so a JOIN would observe a cache hit

        # Rewrite session.json into the legacy shape: pid-only key, no
        # process_start_ns field. Mirrors a session.json written by a
        # pre-schema build of the manager.
        session_path = cache_root / ".zephon_cache_state" / "session.json"
        session = json.loads(session_path.read_text(encoding="utf-8"))
        legacy_owners: dict = {}
        for entry in session["owners"].values():
            entry.pop("process_start_ns", None)
            legacy_owners[str(os.getpid())] = entry
        session["owners"] = legacy_owners
        session_path.write_text(json.dumps(session, sort_keys=True), encoding="utf-8")

        # Same fingerprint + live owner ⇒ JOIN. No HARD_ERROR, no reset.
        mgr2 = _make_manager(cache_root, storage, [loc], persist_state=True)
        try:
            ref = mgr2.resolve(loc)
            assert ref.cache_hit is True
        finally:
            mgr2.close()
    finally:
        mgr1.close()


def test_resolve_takes_over_preparing_from_dead_peer(tmp_path: Path) -> None:
    """Orphaned PREPARING state is taken over via shard_lock flock.

    Regression test for the resilient-workers crash-recovery path: a
    worker that dies mid-download leaves the shard's state as PREPARING
    in shared memory. Without the flock-based recovery, peers would
    poll forever waiting for a peer that will never finish. We simulate
    a dead peer by setting PREPARING in shared state without holding
    the per-shard flock; the next ``resolve()`` must take over and
    complete instead of polling forever.
    """
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data = b"orphan recovery"
    raw_name = "orphan.bin"
    _make_file(remote / raw_name, data)
    storage = LocalFSBackend(root=remote)
    loc = _locator("demo", 200, str(remote), raw_name=raw_name, raw_bytes=len(data))

    mgr = _make_manager(cache_root, storage, [loc])
    try:
        # Force the shard into PREPARING without holding shard_lock —
        # mimics a worker that started a download then died mid-flight
        # (kernel released its shard_lock automatically on death).
        index = mgr._index_for(loc)
        assert index is not None
        with mgr._cache_lock:
            mgr._shared.shard_states[index] = _ShardState.PREPARING

        # resolve() must complete (not loop). Bound the call with a
        # generous deadline — anything more than a couple of seconds
        # would indicate the recovery path failed.
        start = time.time()
        ref = mgr.resolve(loc)
        elapsed = time.time() - start
        assert elapsed < 10.0, f"resolve took {elapsed:.2f}s — recovery failed"
        assert ref.raw.path.exists()
        assert ref.raw.path.read_bytes() == data
    finally:
        mgr.close()
