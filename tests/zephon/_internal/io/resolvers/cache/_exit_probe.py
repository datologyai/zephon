"""Run cache-lifetime scenarios in an interpreter that exits normally."""

import gc
import json
import os
import signal
import sys
import time
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zephon._internal.io.resolvers.cache import CacheManager


class _Anchor:
    pass


_retained: list[Any] = []


def shm_exists(name: str) -> bool:
    """Inspect a POSIX SHM name without registering it with a resource tracker."""
    import _posixshmem

    try:
        # SharedMemory prepends a slash to the stored name, including when
        # CacheSharedState's generated name already starts with one.
        fd = _posixshmem.shm_open("/" + name, os.O_RDONLY, 0o600)
    except FileNotFoundError:
        return False
    os.close(fd)
    return True


def _install_observer(root: Path) -> tuple[list[dict[str, Any]], list[float]]:
    records: list[dict[str, Any]] = []
    exit_started: list[float] = []
    pid = os.getpid()

    def observe(anchor: _Anchor) -> None:
        if os.getpid() != pid or not exit_started:
            return
        observations = []
        for record in records:
            session = json.loads(Path(record["session_path"]).read_text())
            observations.append(
                {
                    "owners": session["owners"],
                    "session_id": session["session_id"],
                    "shm_exists": {
                        name: shm_exists(name) for name in record["shm_names"]
                    },
                }
            )
        (root / "exit_observation.json").write_text(
            json.dumps(
                {
                    "sessions": observations,
                    "elapsed": time.monotonic() - exit_started[0],
                }
            )
        )

    # This observer is created before cache finalizers, so it observes their
    # completed exit work before Python's separate resource tracker can unlink.
    anchor = _Anchor()
    weakref.finalize(anchor, observe, anchor)
    return records, exit_started


def _make_manager(root: Path) -> "CacheManager":
    from tests._catalog_helpers import catalog_set_from_locators
    from zephon._internal.io.resolvers.cache import CacheManager
    from zephon._internal.io.storage.local import LocalFSBackend
    from zephon._internal.io.types import ShardFile, ShardLocator

    remote = root / "remote"
    remote.mkdir(parents=True, exist_ok=True)
    payload = b"cache exit probe"
    (remote / "shard.bin").write_bytes(payload)
    loc = ShardLocator(
        dataset="demo",
        shard_id=0,
        format="dummy",
        root=str(remote),
        raw=ShardFile(basename="shard.bin", bytes=len(payload), hashes={}),
        zip=None,
        compression=None,
        extra=None,
    )
    catalogs = catalog_set_from_locators([loc], root / "catalogs")
    return CacheManager(
        root / "cache",
        LocalFSBackend(root=remote),
        catalog_set=catalogs,
        persist_state=True,
    )


def _record(manager: "CacheManager", records: list[dict[str, Any]]) -> None:
    session = json.loads(manager._session_path.read_text())
    assert all(shm_exists(name) for name in session["shm_names"].values())
    records.append(
        {
            "session_path": str(manager._session_path),
            "session_id": session["session_id"],
            "shm_names": list(session["shm_names"].values()),
            "owner_key": manager._owner_key,
        }
    )


def _wait_child(pid: int) -> None:
    deadline = time.monotonic() + 5.0
    while True:
        waited, status = os.waitpid(pid, os.WNOHANG)
        if waited:
            assert os.waitstatus_to_exitcode(status) == 0
            return
        if time.monotonic() >= deadline:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            raise AssertionError("forked exit probe did not finish")
        time.sleep(0.01)


def main(case: str, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    records, exit_started = _install_observer(root)

    # A different library may have installed weakref's atexit handler before
    # the cache module starts its worker or registers its own exit finalizer.
    dependency = _Anchor()
    weakref.finalize(dependency, lambda: None)
    _retained.append(dependency)

    import zephon._internal.io.resolvers.cache.manager as manager_mod

    manager_mod._DEFERRED_RELEASE_EXIT_TIMEOUT = 0.2 if case == "contended" else 1.0
    manager = _make_manager(root)
    _retained.append(manager)
    _record(manager, records)

    if case == "retained":
        _retained.append(_make_manager(root))
    elif case == "burst":
        for _ in range(7):
            _retained.append(_make_manager(root))
        for index in range(1, 8):
            cache_root = root / f"group{index}"
            group = [_make_manager(cache_root) for _ in range(8)]
            _retained.extend(group)
            _record(group[0], records)
    elif case == "explicit":
        manager.close()
        assert json.loads(manager._session_path.read_text())["owners"] == {}
    elif case == "explicit_timeout":
        from filelock import FileLock

        lock = FileLock(manager._reset_lock.lock_file)
        lock.acquire()
        manager_mod.remaining_cleanup_timeout = lambda: 0.02
        manager.close()
        assert not manager._close_finalizer.alive
        assert (
            manager._owner_key
            in json.loads(manager._session_path.read_text())["owners"]
        )
        lock.release()
    elif case == "delayed":
        original_release = manager_mod._release_session_owner

        def release_after_delay(*args: Any, **kwargs: Any) -> bool:
            time.sleep(0.08)
            return original_release(*args, **kwargs)

        manager_mod._release_session_owner = release_after_delay
    elif case == "contended":
        from filelock import FileLock

        lock = FileLock(manager._reset_lock.lock_file)
        lock.acquire()
        _retained.append(lock)
    elif case == "worker_failure":
        attempts: list[None] = []

        def fail_release(*args: Any, **kwargs: Any) -> bool:
            attempts.append(None)
            raise OSError("injected deferred release failure")

        manager_mod._release_session_owner = fail_release
        manager._close_finalizer()
        drains = [manager_mod._drain_deferred_releases(1.0) for _ in range(2)]
        assert drains == [False, False]
        assert len(attempts) == 1
        assert not manager._close_finalizer.alive
        (root / "failure_observation.json").write_text(
            json.dumps({"attempts": len(attempts), "drains": drains})
        )
    elif case == "sentinel_gc":
        sentinel = manager_mod._release_exit_finalizer
        assert sentinel is not None
        info = sentinel.peek()
        assert info is not None
        anchor_ref = weakref.ref(info[0])
        del info
        manager_mod._release_exit_finalizer = None
        for _ in range(3):
            gc.collect()
        assert sentinel.alive
        assert anchor_ref() is not None
        manager_mod._release_exit_finalizer = sentinel
    elif case in {"fork_finalizer", "fork_close"}:
        child_root = root / "child"
        child_root.mkdir()
        pid = os.fork()
        if pid == 0:
            child_records, child_exit_started = _install_observer(child_root)
            if case == "fork_close":
                manager.close()
            child_manager = _make_manager(root)
            _retained.append(child_manager)
            _record(child_manager, child_records)
            child_exit_started.append(time.monotonic())
            sys.exit(0)
        _wait_child(pid)
        session = json.loads(manager._session_path.read_text())
        assert set(session["owners"]) == {manager._owner_key}
        assert session["owners"][manager._owner_key]["instances"] == 1
        assert all(shm_exists(name) for name in session["shm_names"].values())
        (root / "fork_observation.json").write_text(json.dumps(session))
    else:
        raise ValueError(case)

    (root / "probe_info.json").write_text(json.dumps(records))
    # Without a wait, an exit-time enqueue need not yield the GIL to its worker.
    # A long switch interval makes this scheduling hazard deterministic here.
    sys.setswitchinterval(10.0)
    exit_started.append(time.monotonic())


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
