import gc
import time
import weakref
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import pytest

from zephon.io.resolvers.cache.shared_state import (
    CacheSharedState,
    _ShardState,
)


def test_capacity_validation(tmp_path: Path) -> None:
    del tmp_path
    with pytest.raises(ValueError):
        CacheSharedState(capacity=0)
    with pytest.raises(ValueError):
        CacheSharedState(capacity=-1)


def test_create_allocates_arrays_and_returns_names() -> None:
    ss = CacheSharedState(capacity=8)
    try:
        assert ss.capacity == 8
        assert ss.created_regions
        names = ss.shm_names
        assert set(names.keys()) == {"states", "access", "sizes", "usage"}
        # All four names must be distinct
        assert len(set(names.values())) == 4

        assert ss.shard_states.shape == (8,)
        assert ss.shard_access_ns.shape == (8,)
        assert ss.shard_sizes.shape == (8,)
        assert ss.get_cache_usage() == 0
        # Freshly created arrays are zeroed / INVALID
        assert int(ss.shard_states[0]) == int(_ShardState.INVALID)
    finally:
        ss.close()


def test_attach_second_process_in_same_process_sees_writes() -> None:
    """Attaching twice in the same process reflects writes through both views."""
    creator = CacheSharedState(capacity=4)
    attacher = None
    try:
        names = creator.shm_names
        attacher = CacheSharedState(capacity=4, shm_names=names)
        assert not attacher.created_regions
        assert attacher.capacity == 4

        # Write through one view, read through the other.
        creator.shard_states[2] = _ShardState.LOCAL
        creator.shard_sizes[2] = 123
        creator.set_access_time(2, 42)
        creator.add_cache_usage(123)

        assert int(attacher.shard_states[2]) == int(_ShardState.LOCAL)
        assert int(attacher.shard_sizes[2]) == 123
        assert int(attacher.shard_access_ns[2]) == 42
        assert attacher.get_cache_usage() == 123
    finally:
        if attacher is not None:
            attacher.close()
        creator.close()


def test_attach_missing_raises() -> None:
    """Attaching with names that were never created is an error."""
    fake_names = {
        "states": "nonexistent_states_xxx",
        "access": "nonexistent_access_xxx",
        "sizes": "nonexistent_sizes_xxx",
        "usage": "nonexistent_usage_xxx",
    }
    with pytest.raises(FileNotFoundError):
        CacheSharedState(capacity=4, shm_names=fake_names)


def test_unlink_by_names_is_idempotent_on_missing() -> None:
    """Calling unlink_by_names on absent segments is safe."""
    # Fully missing — should not raise.
    CacheSharedState.unlink_by_names(
        {
            "states": "definitely_missing_states_zzz",
            "access": "definitely_missing_access_zzz",
            "sizes": "definitely_missing_sizes_zzz",
            "usage": "definitely_missing_usage_zzz",
        }
    )
    # Empty/None inputs should not raise.
    CacheSharedState.unlink_by_names({})
    CacheSharedState.unlink_by_names(
        {"states": None, "access": "", "sizes": 123, "usage": "x"}
    )  # type: ignore[dict-item]


def test_close_does_not_unlink_segments() -> None:
    """close() releases local handles only; names stay alive.

    This is load-bearing for the session model: if a creator manager
    exits while joiners are still attached, their SHM attachment must
    keep working, AND later managers must still be able to attach via
    the names recorded in ``session.json``.
    """
    ss1 = CacheSharedState(capacity=4)
    names = ss1.shm_names
    ss1.close()
    # Names must still be attachable after close — proves close does not unlink.
    ss2 = CacheSharedState(capacity=4, shm_names=names)
    try:
        assert ss2.capacity == 4
        assert not ss2.created_regions
    finally:
        ss2.close()
    # Explicit unlink releases the OS-level name.
    CacheSharedState.unlink_by_names(names)
    with pytest.raises(FileNotFoundError):
        CacheSharedState(capacity=4, shm_names=names)


def test_numeric_accessors_roundtrip() -> None:
    ss = CacheSharedState(capacity=3)
    try:
        ss.set_cache_usage(1000)
        assert ss.get_cache_usage() == 1000
        ss.add_cache_usage(-250)
        assert ss.get_cache_usage() == 750

        ss.set_access_time(0, 111)
        ss.set_access_time(1, 222)
        ss.set_access_time(2, 333)
        assert int(ss.shard_access_ns[0]) == 111
        assert int(ss.shard_access_ns[2]) == 333

        ss.shard_states[0] = _ShardState.LOCAL
        ss.shard_states[1] = _ShardState.REMOTE
        ss.shard_states[2] = _ShardState.LOCAL
        assert ss.count_local() == 2
    finally:
        ss.close()


def _child_attach_and_write(names: dict, q: Queue) -> None:  # type: ignore[no-redef]
    try:
        ss = CacheSharedState(capacity=4, shm_names=names)
        ss.shard_states[0] = _ShardState.LOCAL
        ss.set_access_time(0, 99)
        ss.close()
        q.put({"ok": True})
    except Exception as exc:  # pragma: no cover
        q.put({"ok": False, "err": repr(exc)})


def test_attach_across_processes() -> None:
    """A child process can attach to segments created by the parent."""
    ss = CacheSharedState(capacity=4)
    try:
        q: Queue = Queue()
        p = Process(target=_child_attach_and_write, args=(ss.shm_names, q))
        p.start()
        p.join(timeout=5)
        assert p.exitcode == 0
        result = q.get_nowait()
        assert result["ok"] is True

        assert int(ss.shard_states[0]) == int(_ShardState.LOCAL)
        assert int(ss.shard_access_ns[0]) == 99
    finally:
        ss.close()


def test_finalizer_releases_segments_on_gc() -> None:
    ss = CacheSharedState(capacity=2)
    fin = ss._close_finalizer
    ss_ref = weakref.ref(ss)
    ss = None  # drop strong ref

    for _ in range(200):
        if fin is None or not fin.alive:
            break
        gc.collect()
        time.sleep(0.01)

    assert ss_ref() is None


def test_accessor_raises_after_close() -> None:
    ss = CacheSharedState(capacity=2)
    ss.close()
    with pytest.raises(RuntimeError):
        _ = ss.shard_states
    with pytest.raises(RuntimeError):
        _ = ss.get_cache_usage()


def test_views_reflect_numpy_semantics() -> None:
    """Views are numpy arrays backed by shared memory — basic sanity."""
    ss = CacheSharedState(capacity=5)
    try:
        assert ss.shard_states.dtype == np.uint8
        assert ss.shard_access_ns.dtype == np.uint64
        assert ss.shard_sizes.dtype == np.int64
        ss.shard_sizes[:] = np.arange(5, dtype=np.int64)
        assert list(ss.shard_sizes.tolist()) == [0, 1, 2, 3, 4]
    finally:
        ss.close()
