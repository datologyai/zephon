import gc
import json
import time
from multiprocessing import Process, Queue
from pathlib import Path

import pytest

import zephon.io.resolvers.cache.shared_state as shared_state_mod
from zephon.io.resolvers.cache.shared_state import (
    CacheEntry,
    CacheSharedState,
    _ShardState,
)
from zephon.io.types import ShardFile, ShardLocator


def _locator(
    dataset: str, shard_id: int, raw_name: str, zip_name: str | None = None
) -> ShardLocator:
    raw = ShardFile(basename=raw_name, bytes=0, hashes={})
    zip_file = ShardFile(basename=zip_name, bytes=0, hashes={}) if zip_name else None
    return ShardLocator(
        dataset=dataset,
        shard_id=shard_id,
        format="dummy",
        root="/unused",
        raw=raw,
        zip=zip_file,
        compression="gzip" if zip_name else None,
        extra=None,
    )


def test_shared_state_initializes_and_persists_meta(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    ss1 = CacheSharedState(root, capacity=64)
    try:
        # Meta directory/files created
        state_dir = root / ".zephon_cache_state"
        meta_path = state_dir / "meta.json"
        assert state_dir.is_dir()
        assert meta_path.is_file()

        # Vectors have expected shapes
        assert ss1.capacity == 64
        assert ss1.shard_states.shape == (ss1.capacity,)
        assert ss1.shard_access_ns.shape == (ss1.capacity,)
        assert ss1.shard_sizes.shape == (ss1.capacity,)
        assert ss1.get_cache_usage() == 0

        # Register one entry
        loc = _locator("ds", 0, "raw0.bin")
        e1 = ss1.ensure_entry(loc)
        assert isinstance(e1, CacheEntry)
        assert e1.dataset == "ds"
        assert e1.shard_id == 0
        assert e1.raw == "raw0.bin"

        # New instance should reload meta mapping
        ss2 = CacheSharedState(root, capacity=64)
        try:
            e2 = ss2.lookup("ds", 0)
            assert e2 is not None
            assert e2.index == e1.index
            assert ss2.entry_by_index(e1.index) is not None
        finally:
            ss2.close()
    finally:
        ss1.close()


def test_shared_state_updates_entry_metadata(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    ss = CacheSharedState(root, capacity=64)
    try:
        loc1 = _locator("demo", 7, "a.bin", None)
        e = ss.ensure_entry(loc1)
        # Update both raw and zip names via ensure_entry
        loc2 = _locator("demo", 7, "b.bin", "b.bin.gz")
        _ = ss.ensure_entry(loc2)

        # The in-memory object returned earlier may not reflect updated names;
        # retrieve a fresh view via lookup and verify changes.
        e2 = ss.lookup("demo", 7)
        assert e2 is not None
        assert e2.index == e.index
        assert e2.raw == "b.bin"
        assert e2.zip == "b.bin.gz"

        # Meta file reflects update
        meta_path = root / ".zephon_cache_state" / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mapping = meta["mapping"]["demo"][str(7)]
        assert mapping["raw"] == "b.bin"
        assert mapping["zip"] == "b.bin.gz"
    finally:
        ss.close()


def test_shared_state_usage_and_count_local(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    ss = CacheSharedState(root, capacity=64)
    try:
        loc = _locator("foo", 1, "x.bin")
        entry = ss.ensure_entry(loc)
        # Initially nothing local
        assert ss.count_local() == 0

        # Mark LOCAL and set size
        ss.shard_states[entry.index] = _ShardState.LOCAL
        ss.shard_sizes[entry.index] = 1234
        ss.add_cache_usage(1234)
        ts = time.time_ns()
        ss.set_access_time(entry.index, ts)
        assert ss.count_local() == 1
        assert int(ss.shard_access_ns[entry.index]) == ts
        assert ss.get_cache_usage() == 1234

        # Reset usage explicitly
        ss.set_cache_usage(42)
        assert ss.get_cache_usage() == 42
    finally:
        # idempotent close
        ss.close()
        ss.close()


# ---------------------------
# Multiprocess helpers/tests
# ---------------------------


def _child_set_local_and_usage(
    root: str, dataset: str, shard_id: int, size: int, q: Queue
) -> None:  # type: ignore[no-redef]
    ss = CacheSharedState(Path(root), capacity=64)
    try:
        entry = ss.lookup(dataset, shard_id)
        if entry is None:
            q.put({"ok": False, "err": "entry-missing"})
            return
        ss.shard_states[entry.index] = _ShardState.LOCAL
        ss.shard_sizes[entry.index] = size
        ss.set_cache_usage(size)
        ts = 123456789
        ss.set_access_time(entry.index, ts)
        q.put({"ok": True, "index": entry.index, "ts": ts})
    finally:
        ss.close()


def _child_ensure_entry(root: str, loc: ShardLocator, q: Queue) -> None:  # type: ignore[no-redef]
    ss = CacheSharedState(Path(root), capacity=64)
    try:
        entry = ss.ensure_entry(loc)
        q.put({"ok": True, "index": entry.index})
    finally:
        ss.close()


def test_shared_state_is_visible_across_processes(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    ss = CacheSharedState(root, capacity=64)
    try:
        # Create entry in parent
        loc = _locator("multi", 1, "raw.bin")
        e = ss.ensure_entry(loc)

        # Child marks it LOCAL and sets usage/ts
        q: Queue = Queue()
        p = Process(
            target=_child_set_local_and_usage, args=(str(root), "multi", 1, 2048, q)
        )
        p.start()
        p.join(timeout=10)
        assert p.exitcode == 0
        msg = q.get_nowait()
        assert msg["ok"] is True
        assert msg["index"] == e.index

        # Parent observes changes through shared memory
        assert ss.count_local() == 1
        assert ss.get_cache_usage() == 2048
        assert int(ss.shard_sizes[e.index]) == 2048

        # Child can also create new entries and parent sees them
        q2: Queue = Queue()
        loc2 = _locator("multi", 2, "raw2.bin")
        p2 = Process(target=_child_ensure_entry, args=(str(root), loc2, q2))
        p2.start()
        p2.join(timeout=10)
        assert p2.exitcode == 0
        _ = q2.get_nowait()
        # Lookup after child update
        e2 = ss.lookup("multi", 2)
        assert e2 is not None
        assert e2.raw == "raw2.bin"
    finally:
        ss.close()


def test_cache_shared_state_cleanup_runs_on_gc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "cache"
    called = {"flag": False}
    orig = shared_state_mod._close_cache_shared_state

    def wrapped(ref) -> None:
        called["flag"] = True
        orig(ref)

    monkeypatch.setattr(shared_state_mod, "_close_cache_shared_state", wrapped)
    ss = CacheSharedState(root, capacity=64)
    fin = ss._close_finalizer
    ss = None

    for _ in range(200):
        if not fin.alive:
            break
        gc.collect()
        time.sleep(0.01)

    assert called["flag"] is True
    assert fin.alive is False


def test_ensure_entry_raises_when_capacity_exceeded(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    ss = CacheSharedState(root, capacity=2)
    try:
        loc0 = _locator("ds", 0, "a.bin")
        loc1 = _locator("ds", 1, "b.bin")
        loc2 = _locator("ds", 2, "c.bin")

        ss.ensure_entry(loc0)
        ss.ensure_entry(loc1)

        with pytest.raises(IndexError, match="exceeds capacity"):
            ss.ensure_entry(loc2)
    finally:
        ss.close()


def test_capacity_resize_recreates_shared_memory(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    ss1 = CacheSharedState(root, capacity=4)
    try:
        loc = _locator("ds", 0, "raw.bin")
        ss1.ensure_entry(loc)
    finally:
        ss1.close()

    # Re-open with larger capacity — should resize
    ss2 = CacheSharedState(root, capacity=16)
    try:
        assert ss2.capacity == 16
        assert ss2.shard_states.shape == (16,)
        # Old mapping was cleared during resize
        assert ss2.lookup("ds", 0) is None
        # Can register shards up to new capacity
        for i in range(16):
            ss2.ensure_entry(_locator("ds", i, f"s{i}.bin"))
    finally:
        ss2.close()


def test_capacity_smaller_or_equal_adopts_stored(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    ss1 = CacheSharedState(root, capacity=32)
    try:
        loc = _locator("ds", 0, "raw.bin")
        ss1.ensure_entry(loc)
    finally:
        ss1.close()

    # Re-open with smaller capacity — should keep stored (32)
    ss2 = CacheSharedState(root, capacity=16)
    try:
        assert ss2.capacity == 32
        e = ss2.lookup("ds", 0)
        assert e is not None
        assert e.raw == "raw.bin"
    finally:
        ss2.close()
