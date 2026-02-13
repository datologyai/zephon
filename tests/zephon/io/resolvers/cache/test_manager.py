import gc
import os
import threading
import time
import weakref
from multiprocessing import Process, Queue
from pathlib import Path
from typing import IO, Any, Mapping

import pytest

import zephon.io.resolvers.cache.manager as manager_mod
from zephon.io.resolvers.cache import (
    CacheManager,
    PermanentSourceMissing,
    ShardNotReady,
)
from zephon.io.resolvers.cache.shared_state import _ShardState
from zephon.io.storage.local import LocalFSBackend
from zephon.io.types import LocalShardRef, ShardFile, ShardLocator


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


def test_resolve_raw_only_downloads_and_updates_stats(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data = b"hello world" * 123
    raw_name = "shard0.bin"
    _make_file(remote / raw_name, data)

    storage = LocalFSBackend(root=remote)
    mgr = CacheManager(cache_root, storage, num_shards=128)

    loc = _locator(
        dataset="demo",
        shard_id=0,
        root=str(remote),
        raw_name=raw_name,
        raw_bytes=len(data),
    )

    ref = mgr.resolve(loc)
    assert isinstance(ref, LocalShardRef)
    # On-disk location under dataset namespace
    expected_local = cache_root / "demo" / raw_name
    assert ref.raw.path == expected_local
    assert ref.raw.path.is_file()
    assert ref.raw.bytes == len(data)
    assert ref.zip is None

    stats = mgr.stats()
    assert stats.shards == 1
    assert stats.bytes_used == len(data)

    # Second resolve is a cache hit and should not change stats
    ref2 = mgr.resolve(loc)
    assert ref2.raw.path == expected_local
    assert mgr.stats() == stats
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

    # keep_zip = False (default): zip should not be retained
    mgr1 = CacheManager(cache_root, storage, num_shards=128, keep_zip=False)
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
    ref1 = mgr1.resolve(loc)
    assert ref1.raw.path.is_file()
    assert ref1.raw.bytes == len(payload)
    assert ref1.zip is None
    assert not (cache_root / "ds" / zip_name).exists()
    stats1 = mgr1.stats()
    assert stats1.bytes_used == len(payload)
    assert stats1.shards == 1

    mgr1.close()

    # keep_zip = True with a fresh session: zip is kept and accounted in stats
    mgr2 = CacheManager(
        cache_root, storage, num_shards=128, keep_zip=True, persist_state=False
    )
    ref2 = mgr2.resolve(loc)
    assert ref2.raw.path.is_file()
    assert ref2.zip is not None
    assert ref2.zip.path.exists()
    stats2 = mgr2.stats()
    assert stats2.shards == 1
    assert stats2.bytes_used == len(payload) + (remote / zip_name).stat().st_size
    mgr2.close()


def test_validate_hash_mismatch_raises_and_removes_file(tmp_path: Path) -> None:
    # Prepare a raw file and an incorrect declared hash
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    raw_name = "shard2.bin"
    data = b"abc" * 333
    _make_file(remote / raw_name, data)

    storage = LocalFSBackend(root=remote)
    mgr = CacheManager(cache_root, storage, num_shards=128, validate_hash="md5")

    try:
        # Declare wrong hash (all zeros) to force mismatch
        loc = _locator(
            dataset="demo",
            shard_id=2,
            root=str(remote),
            raw_name=raw_name,
            raw_bytes=len(data),
            hashes={"md5": "0" * 32},
        )

        with pytest.raises(ValueError):
            mgr.resolve(loc)

        # File should not be present after failure
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
    called = {"flag": False}

    orig = manager_mod._close_cache_manager

    def wrapped(ref) -> None:
        called["flag"] = True
        orig(ref)

    monkeypatch.setattr(manager_mod, "_close_cache_manager", wrapped)
    mgr = CacheManager(cache_root, LocalFSBackend(root=remote), num_shards=128)

    fin = mgr._close_finalizer
    mgr_ref = weakref.ref(mgr)
    mgr = None  # drop strong ref

    for _ in range(200):
        if not fin.alive:
            break
        gc.collect()
        time.sleep(0.01)

    assert called["flag"] is True
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
    mgr = CacheManager(cache_root, storage, num_shards=128)

    loc = _locator(
        dataset="demo",
        shard_id=3,
        root=str(remote),
        raw_name=raw_name,
        raw_bytes=len(data),
    )

    # Gate the internal _prepare so that the state remains PREPARING for a bit
    started = threading.Event()
    release = threading.Event()

    real_prepare = mgr._prepare

    def slow_prepare(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-untyped-def]
        started.set()
        # wait to simulate long preparation
        release.wait(timeout=5)
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(mgr, "_prepare", slow_prepare)

    result: dict[str, Any] = {}

    def run_resolve() -> None:
        try:
            result["ref"] = mgr.resolve(loc)
        except Exception as exc:  # pragma: no cover - should not happen
            result["err"] = exc

    t = threading.Thread(target=run_resolve)
    t.start()
    # Wait until we are inside _prepare
    assert started.wait(timeout=2)

    with pytest.raises(ShardNotReady):
        mgr.resolve(loc, blocking=False)

    # Allow prepare to complete and join
    release.set()
    t.join(timeout=5)
    assert "ref" in result
    assert isinstance(result["ref"], LocalShardRef)
    mgr.close()


def test_missing_source_raises_permanent(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    raw_name = "missing.bin"
    # Deliberately do not create the file

    storage = LocalFSBackend(root=remote)
    mgr = CacheManager(cache_root, storage, num_shards=128)

    try:
        loc = _locator(
            dataset="demo",
            shard_id=4,
            root=str(remote),
            raw_name=raw_name,
            raw_bytes=123,
        )

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
    mgr = CacheManager(cache_root, storage, num_shards=128)
    try:
        loc = _locator("demo", 5, str(remote), raw_name=raw_name, raw_bytes=len(data))

        ref = mgr.resolve(loc)
        assert ref.raw.path.exists()
        stats1 = mgr.stats()

        # Delete local file behind cache manager's back
        ref.raw.path.unlink()

        # Next resolve should re-prepare the shard and restore size accounting
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

    # With real required_bytes (includes default slack), any limit < additional should raise
    storage = LocalFSBackend(root=remote)
    slack = max(512 * 1024, int(len(data) * 0.005))
    slack = min(64 * 1024 * 1024, slack)
    limit_bytes = len(data) + slack - 1
    mgr = CacheManager(cache_root, storage, num_shards=128, limit_bytes=limit_bytes)
    try:
        loc = _locator("demo", 6, str(remote), raw_name=raw_name, raw_bytes=len(data))
        with pytest.raises(ValueError):
            mgr.resolve(loc)
    finally:
        mgr.close()


def test_eviction_chooses_coldest_when_capacity_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # To keep this test tractable (avoid default slack), monkeypatch required_bytes
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data_a = b"A" * 1024
    data_b = b"B" * 1024
    _make_file(remote / "a.bin", data_a)
    _make_file(remote / "b.bin", data_b)

    storage = LocalFSBackend(root=remote)
    mgr = CacheManager(cache_root, storage, num_shards=128, limit_bytes=6 * 1024)
    try:
        # Only count the declared raw size (no slack) for testing eviction
        monkeypatch.setattr(mgr, "_required_bytes", lambda loc: int(loc.raw.bytes))

        loc_a = _locator(
            "demo", 10, str(remote), raw_name="a.bin", raw_bytes=len(data_a)
        )
        loc_b = _locator(
            "demo", 11, str(remote), raw_name="b.bin", raw_bytes=len(data_b)
        )

        ref_a = mgr.resolve(loc_a)
        assert ref_a.raw.path.exists()
        # Touch A so it appears more recent, then sleep to ensure time moves
        mgr.touch(loc_a)
        time.sleep(0.01)

        # Now resolve B; since limit is 6KB and each shard ~1KB, it should evict none
        # when preparing B (usage 1KB + additional 1KB <= limit). After B local, usage ~2KB.
        # Next, force another resolve of B to trigger capacity check with skip_index for B.
        ref_b = mgr.resolve(loc_b)
        assert ref_b.raw.path.exists()

        # Now prepare a third shard C to trigger eviction of the coldest (which is A if older)
        data_c = b"C" * 5120  # 5KB
        _make_file(remote / "c.bin", data_c)
        loc_c = _locator(
            "demo", 12, str(remote), raw_name="c.bin", raw_bytes=len(data_c)
        )
        # Reduce limit so that usage (2KB) + additional (4KB) > limit (6KB) => evict one
        # (mgr already has limit 6KB). Ensure A is the colder one by not touching it again.
        ref_c = mgr.resolve(loc_c)
        assert ref_c.raw.path.exists()

        # A should be evicted (file removed), B and C should exist
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
    mgr = CacheManager(cache_root, storage, num_shards=128)
    try:
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

        with pytest.raises(ValueError):
            mgr.resolve(loc)

        ds_root = cache_root / "demo"
        # Ensure no stray temp/part files remain
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
    src_path = _make_file(remote / raw_name, data)

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
            # Fail the first time for this src, then succeed
            self.calls[src] = self.calls.get(src, 0) + 1
            if self.calls[src] == 1:
                raise IOError("transient failure")
            # Copy on second (or later) attempt
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            data = Path(src).read_bytes()
            Path(dst).write_bytes(data)

        def listdir(self, path: str) -> list[str]:  # pragma: no cover - unused
            return []

        def stat(self, path: str) -> Mapping[str, int]:  # pragma: no cover - unused
            return {"size": Path(path).stat().st_size}

    storage = FlakyStorage(remote)
    mgr = CacheManager(cache_root, storage, num_shards=128, download_retry=2)

    try:
        # Speed up retry backoff
        monkeypatch.setattr(time, "sleep", lambda _: None)

        loc = _locator("demo", 30, str(remote), raw_name=raw_name, raw_bytes=len(data))
        ref = mgr.resolve(loc)
        assert ref.raw.path.read_bytes() == data
        # Ensure we hit the retry path exactly once before success
        src_full = os.path.join(str(remote), raw_name)
        assert storage.calls.get(src_full) == 2
    finally:
        mgr.close()


def test_cache_root_reset_when_persist_state_false(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    cache_root = tmp_path / "cache"
    data = b"persist test"
    raw_name = "persist.bin"
    _make_file(remote / raw_name, data)

    storage = LocalFSBackend(root=remote)
    loc = _locator("demo", 40, str(remote), raw_name=raw_name, raw_bytes=len(data))

    mgr1 = CacheManager(cache_root, storage, num_shards=128, persist_state=False)
    ref = mgr1.resolve(loc)
    assert ref.raw.path.exists()

    # New manager while first still alive should not reset
    mgr_same = CacheManager(cache_root, storage, num_shards=128, persist_state=False)
    assert (cache_root / "demo" / raw_name).exists()

    # After existing managers close, the next initializer should reset
    mgr1.close()
    mgr_same.close()

    mgr2 = CacheManager(cache_root, storage, num_shards=128, persist_state=False)
    assert not (cache_root / "demo" / raw_name).exists()
    mgr2.close()


# ---------------------------
# Multiprocess coordination
# ---------------------------


class SlowLocalFSBackend(LocalFSBackend):
    """Local backend that sleeps in download to simulate long PREPARING."""

    def download(self, src: str, dst: str, timeout: float | None = None) -> None:
        # Signal start, wait for external allow signal, then perform real copy
        Path(self.root / "download_started").write_text("start", encoding="utf-8")
        deadline = time.time() + 5.0
        while time.time() < deadline and not (self.root / "allow_finish").exists():
            time.sleep(0.05)
        super().download(src, dst, timeout)


def _proc_resolve_slow(
    cache_root: str, remote: str, loc: ShardLocator, q: Queue
) -> None:  # type: ignore[no-redef]
    mgr = CacheManager(
        Path(cache_root),
        SlowLocalFSBackend(root=Path(remote)),
        num_shards=128,
        persist_state=True,
    )
    try:
        ref = mgr.resolve(loc)
        q.put({"ok": True, "path": str(ref.raw.path)})
    except Exception as exc:  # pragma: no cover - fail path
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

    # Wait for download to start
    started_flag = remote / "download_started"
    for _ in range(50):
        if started_flag.exists():
            break
        time.sleep(0.05)
    assert started_flag.exists()

    # Second process attempts non-blocking resolve and gets ShardNotReady
    mgr2 = CacheManager(
        cache_root, LocalFSBackend(root=remote), num_shards=128, persist_state=True
    )
    with pytest.raises(ShardNotReady):
        mgr2.resolve(loc, blocking=False)

    # Allow the first process to finish the download
    (remote / "allow_finish").write_text("ok", encoding="utf-8")

    # Blocking resolve succeeds after the first finishes
    ref2 = mgr2.resolve(loc, blocking=True)
    assert ref2.raw.path.exists()

    p.join(timeout=10)
    assert p.exitcode == 0
    res = q.get_nowait()
    assert res["ok"] is True
    mgr2.close()
    # Ensure child manager releases ownership before the next test runs
    (remote / "download_started").unlink(missing_ok=True)
    (remote / "allow_finish").unlink(missing_ok=True)


# ---------------------------
# Multiprocess eviction with tiny slack
# ---------------------------


class TinySlackCacheManager(CacheManager):
    def _required_bytes(self, locator: ShardLocator) -> int:  # type: ignore[override]
        return int(locator.raw.bytes)


def _proc_resolve_with_tiny_slack(
    cache_root: str, remote: str, loc: ShardLocator, q: Queue
) -> None:  # type: ignore[no-redef]
    mgr = TinySlackCacheManager(
        Path(cache_root),
        LocalFSBackend(root=Path(remote)),
        num_shards=128,
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

    # Parent loads A first (older), then a short sleep to ensure access time order
    mgr_parent = TinySlackCacheManager(
        cache_root, LocalFSBackend(root=remote), num_shards=128, limit_bytes=5 * 1024
    )
    _ = mgr_parent.resolve(loc_a)
    time.sleep(0.02)

    # Child resolves B concurrently
    q: Queue = Queue()
    p = Process(
        target=_proc_resolve_with_tiny_slack,
        args=(str(cache_root), str(remote), loc_b, q),
    )
    p.start()
    p.join(timeout=10)
    assert p.exitcode == 0
    child_res = q.get_nowait()
    assert child_res["ok"] is True

    # Now resolve C in parent, which should evict the coldest (A)
    _ = mgr_parent.resolve(loc_c)

    path_a = cache_root / "demo" / "a.bin"
    path_b = cache_root / "demo" / "b.bin"
    path_c = cache_root / "demo" / "c.bin"
    shared = mgr_parent._shared
    entry_a = shared.lookup("demo", 100)
    entry_b = shared.lookup("demo", 101)
    entry_c = shared.lookup("demo", 102)
    assert entry_a is not None
    assert entry_b is not None
    assert entry_c is not None
    state_a = _ShardState(shared.shard_states[entry_a.index])
    state_b = _ShardState(shared.shard_states[entry_b.index])
    state_c = _ShardState(shared.shard_states[entry_c.index])
    assert state_a != _ShardState.LOCAL
    assert state_b == _ShardState.LOCAL
    assert state_c == _ShardState.LOCAL
    # Files are best-effort removal; ensure at most two local shard files remain.
    remaining_files = [p for p in (path_a, path_b, path_c) if p.exists()]
    assert len(remaining_files) <= 2
    mgr_parent.close()
