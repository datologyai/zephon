# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Capability-gated open-file-description byte-range locks.

Python exposes OFD command constants but does not provide a high-level helper
that accepts a byte range. This module contains the small native-ABI boundary
needed to use ``fcntl(F_OFD_SETLK)`` safely on supported 64-bit Linux and
macOS hosts.
"""

from __future__ import annotations

import errno
import os
import platform
import stat
import struct
import sys
import tempfile
import threading
import time
import weakref
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import ModuleType

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-Unix hosts
    fcntl = None  # type: ignore[assignment]


_DARWIN_FLOCK = struct.Struct("@qqihh")
_LINUX_64_FLOCK = struct.Struct("@hh4xqqi4x")
# Linux UAPI include/uapi/asm-generic/fcntl.h defines F_OFD_SETLK as 37.
_LINUX_F_OFD_SETLK = 37
_CONFLICT_ERRNOS = {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}
_FORK_GUARD = threading.RLock()
_LIVE_LEASES: weakref.WeakSet[OFDLease] = weakref.WeakSet()
_LIVE_FILES: weakref.WeakSet[OFDLockFile] = weakref.WeakSet()
_PENDING_ACQUISITION_FDS: set[int] = set()


class OFDLockUnavailable(RuntimeError):
    """Raised when this system cannot safely provide the requested lock.

    Callers can catch this error to disable an optimization or use another way
    to prevent concurrent access.
    """


class OFDLockMode(Enum):
    """Control whether other workers may use the same resource concurrently.

    A shared lock may coexist with other shared locks. An exclusive lock keeps
    both shared and exclusive callers out until it is released.
    """

    SHARED = "shared"
    EXCLUSIVE = "exclusive"


@dataclass(frozen=True)
class OFDBackendInfo:
    """Diagnostic details about the operating-system locking implementation.

    Most callers do not need this information. It is useful in logs and support
    reports when lock availability differs between machines.
    """

    abi_id: str
    architecture: str
    struct_size: int
    setlk_command: int


@dataclass(frozen=True)
class OFDProbeResult:
    """Report whether this locking mechanism works in a particular directory.

    Filesystems can differ even on the same machine. When ``supported`` is
    false, ``reason`` explains why and callers should use a fallback.
    """

    supported: bool
    backend: OFDBackendInfo | None
    reason: str | None = None


class _NativeOFDBackend:
    def __init__(
        self,
        *,
        info: OFDBackendInfo,
        flock_struct: struct.Struct,
        darwin_field_order: bool,
    ) -> None:
        self.info = info
        self._flock_struct = flock_struct
        self._darwin_field_order = darwin_field_order

    def _pack_flock(self, lock_type: int, start: int, length: int) -> bytes:
        if start < 0:
            raise ValueError(f"OFD lock start must be non-negative, got {start}")
        if length <= 0:
            raise ValueError(f"OFD lock length must be positive, got {length}")

        if self._darwin_field_order:
            return self._flock_struct.pack(
                start,
                length,
                0,
                lock_type,
                os.SEEK_SET,
            )
        return self._flock_struct.pack(
            lock_type,
            os.SEEK_SET,
            start,
            length,
            0,
        )

    def try_lock(
        self,
        fd: int,
        mode: OFDLockMode,
        start: int,
        length: int,
    ) -> bool:
        fcntl_module = _require_fcntl()
        if mode is OFDLockMode.SHARED:
            lock_type = fcntl_module.F_RDLCK
        elif mode is OFDLockMode.EXCLUSIVE:
            lock_type = fcntl_module.F_WRLCK
        else:  # pragma: no cover - Enum makes this unreachable for typed callers
            raise TypeError(f"Unknown OFD lock mode: {mode!r}")

        request = self._pack_flock(lock_type, start, length)
        try:
            fcntl_module.fcntl(fd, self.info.setlk_command, request)
        except OSError as exc:
            if exc.errno in _CONFLICT_ERRNOS:
                return False
            raise
        return True

    def unlock(self, fd: int, start: int, length: int) -> None:
        fcntl_module = _require_fcntl()
        request = self._pack_flock(fcntl_module.F_UNLCK, start, length)
        fcntl_module.fcntl(fd, self.info.setlk_command, request)


def _require_fcntl() -> ModuleType:
    if fcntl is None:
        raise OFDLockUnavailable("Python has no fcntl module")
    return fcntl


@lru_cache(maxsize=None)
def _select_backend(
    *,
    sys_platform: str | None = None,
    architecture: str | None = None,
    pointer_size: int | None = None,
) -> _NativeOFDBackend:
    fcntl_module = _require_fcntl()
    selected_platform = sys.platform if sys_platform is None else sys_platform
    selected_arch = (
        platform.machine() if architecture is None else architecture
    ).lower()
    selected_pointer_size = (
        struct.calcsize("P") if pointer_size is None else pointer_size
    )
    if selected_pointer_size != 8:
        raise OFDLockUnavailable(f"Unsupported {selected_pointer_size * 8}-bit OFD ABI")

    if selected_platform == "darwin" and selected_arch in {"arm64", "x86_64"}:
        if not hasattr(fcntl_module, "F_OFD_SETLK"):
            raise OFDLockUnavailable(
                "Python's fcntl module does not expose F_OFD_SETLK, and no safe "
                "numeric fallback is available for Darwin"
            )
        if _DARWIN_FLOCK.size != 24:
            raise OFDLockUnavailable(
                f"Unexpected Darwin struct flock size: {_DARWIN_FLOCK.size}"
            )
        return _NativeOFDBackend(
            info=OFDBackendInfo(
                abi_id="darwin-64",
                architecture=selected_arch,
                struct_size=_DARWIN_FLOCK.size,
                setlk_command=fcntl_module.F_OFD_SETLK,
            ),
            flock_struct=_DARWIN_FLOCK,
            darwin_field_order=True,
        )

    if selected_platform.startswith("linux") and selected_arch in {
        "aarch64",
        "x86_64",
    }:
        setlk_command = getattr(
            fcntl_module,
            "F_OFD_SETLK",
            _LINUX_F_OFD_SETLK,
        )
        if _LINUX_64_FLOCK.size != 32:
            raise OFDLockUnavailable(
                f"Unexpected Linux struct flock size: {_LINUX_64_FLOCK.size}"
            )
        return _NativeOFDBackend(
            info=OFDBackendInfo(
                abi_id="linux-64",
                architecture=selected_arch,
                struct_size=_LINUX_64_FLOCK.size,
                setlk_command=setlk_command,
            ),
            flock_struct=_LINUX_64_FLOCK,
            darwin_field_order=False,
        )

    raise OFDLockUnavailable(
        f"Unsupported OFD ABI: platform={selected_platform!r}, "
        + f"architecture={selected_arch!r}"
    )


def ofd_backend_info() -> OFDBackendInfo:
    """Return native OFD backend information or raise if unsupported."""
    return _select_backend().info


def probe_ofd_support(directory: str | os.PathLike[str]) -> OFDProbeResult:
    """Probe OFD range semantics on ``directory`` without retaining a file."""
    try:
        backend = _select_backend()
    except OFDLockUnavailable as exc:
        return OFDProbeResult(supported=False, backend=None, reason=str(exc))

    # The probe owns transient descriptors and leases that are not registered in
    # the live-object sets.  Keep fork outside their entire lifetime so a child
    # cannot inherit an untracked probe descriptor.
    with _FORK_GUARD:
        return _probe_ofd_support_guarded(directory, backend)


def _probe_ofd_support_guarded(
    directory: str | os.PathLike[str],
    backend: _NativeOFDBackend,
) -> OFDProbeResult:
    first_fd = -1
    second_fd = -1
    probe_path: str | None = None
    try:
        first_fd, probe_path = tempfile.mkstemp(
            prefix=".zephon-ofd-probe-", dir=directory
        )
        second_fd, _ = _open_regular_file(probe_path)
        os.unlink(probe_path)
        probe_path = None

        if not backend.try_lock(first_fd, OFDLockMode.EXCLUSIVE, 0, 1):
            raise OFDLockUnavailable("initial exclusive OFD probe lock conflicted")
        if backend.try_lock(second_fd, OFDLockMode.SHARED, 0, 1):
            raise OFDLockUnavailable(
                "independent open file descriptions did not conflict"
            )
        if not backend.try_lock(second_fd, OFDLockMode.EXCLUSIVE, 1, 1):
            raise OFDLockUnavailable("disjoint OFD byte ranges conflicted")
        backend.unlock(second_fd, 1, 1)
        backend.unlock(first_fd, 0, 1)

        if not backend.try_lock(first_fd, OFDLockMode.SHARED, 0, 1):
            raise OFDLockUnavailable("first shared OFD probe lock conflicted")
        if not backend.try_lock(second_fd, OFDLockMode.SHARED, 0, 1):
            raise OFDLockUnavailable("shared OFD probe locks did not coexist")
        backend.unlock(second_fd, 0, 1)
        backend.unlock(first_fd, 0, 1)

        if not backend.try_lock(first_fd, OFDLockMode.EXCLUSIVE, 2, 1):
            raise OFDLockUnavailable("last-close OFD probe lock conflicted")
        os.close(first_fd)
        first_fd = -1
        if not backend.try_lock(second_fd, OFDLockMode.EXCLUSIVE, 2, 1):
            raise OFDLockUnavailable("last close did not release OFD lease")
        backend.unlock(second_fd, 2, 1)
    except (OSError, OFDLockUnavailable, ValueError) as exc:
        return OFDProbeResult(
            supported=False,
            backend=backend.info,
            reason=f"{type(exc).__name__}: {exc}",
        )
    finally:
        for fd in (second_fd, first_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if probe_path is not None:
            try:
                os.unlink(probe_path)
            except OSError:
                pass

    return OFDProbeResult(supported=True, backend=backend.info)


def _open_regular_file(
    path: str | os.PathLike[str],
) -> tuple[int, os.stat_result]:
    flags = os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise OFDLockUnavailable(f"OFD lock path is not a regular file: {path}")
    except BaseException:
        os.close(fd)
        raise
    return fd, file_stat


def _create_regular_file(
    path: str | os.PathLike[str],
    *,
    identity: bytes,
) -> tuple[int, os.stat_result]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    file_stat: os.stat_result | None = None
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise OFDLockUnavailable(f"OFD lock path is not a regular file: {path}")
        os.fchmod(fd, 0o600)
        remaining = memoryview(identity)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("Unable to write OFD lock identity")
            remaining = remaining[written:]
        os.fsync(fd)
        return fd, file_stat
    except BaseException:
        os.close(fd)
        if file_stat is not None:
            try:
                current = os.stat(path, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == (
                    file_stat.st_dev,
                    file_stat.st_ino,
                ):
                    os.unlink(path)
            except OSError:
                pass
        raise


def _validate_file_identity(
    fd: int,
    path: str | os.PathLike[str],
    expected: bytes,
) -> None:
    observed = os.pread(fd, len(expected) + 1, 0)
    if observed != expected:
        raise OFDLockUnavailable(f"OFD lock identity changed at {path}")


class OFDLease:
    """Represent a lock that has already been acquired.

    Instances are returned by :class:`OFDLockFile`; callers should not create
    them directly. Receiving a lease means the lock is active. ``with lease``
    does not acquire it again; it only guarantees release when the block ends,
    including when the protected work raises an exception.

    Keep the lease alive while using the protected resource. A lease may also
    be changed between shared and exclusive mode. Use :meth:`try_convert` when
    an upgrade can encounter another shared holder.
    """

    def __init__(
        self,
        *,
        fd: int,
        backend: _NativeOFDBackend,
        mode: OFDLockMode,
        start: int,
        length: int,
    ) -> None:
        self._fd = fd
        self._backend = backend
        self.mode = mode
        self.start = start
        self.length = length
        self._creator_pid = os.getpid()
        with _FORK_GUARD:
            _LIVE_LEASES.add(self)

    @property
    def closed(self) -> bool:
        """Whether the lock has been released."""
        return self._fd < 0

    def try_convert(self, mode: OFDLockMode) -> bool:
        """Try to change this active lock's mode without waiting.

        ``False`` means another lock prevents an upgrade to exclusive mode.
        The lease remains active in its previous mode in that case.
        """
        with _FORK_GUARD:
            if self._fd < 0:
                raise OFDLockUnavailable("Cannot convert a closed OFD lease")
            if os.getpid() != self._creator_pid:
                raise OFDLockUnavailable("Cannot convert an inherited OFD lease")
            if not self._backend.try_lock(self._fd, mode, self.start, self.length):
                return False
            self.mode = mode
            return True

    def convert(self, mode: OFDLockMode) -> None:
        """Change this active lock's mode without waiting.

        Use :meth:`try_convert` when upgrading a shared lease: another shared
        holder can make that routine operation fail. This method raises
        :class:`BlockingIOError` for such contention.
        """
        if not self.try_convert(mode):
            raise BlockingIOError(errno.EAGAIN, "OFD lease conversion conflicted")

    def close(self) -> None:
        """Release the lock; calling this more than once has no effect."""
        with _FORK_GUARD:
            fd = self._fd
            if fd < 0:
                return
            self._fd = -1
            _LIVE_LEASES.discard(self)

            unlock_error: BaseException | None = None
            if os.getpid() == self._creator_pid:
                try:
                    # Do not rely on close: an untracked duplicate of this open
                    # file description could otherwise keep the lock alive.
                    self._backend.unlock(fd, self.start, self.length)
                except BaseException as exc:  # close must still run
                    unlock_error = exc
            try:
                os.close(fd)
            except OSError:
                if unlock_error is None:
                    raise
            if unlock_error is not None:
                raise unlock_error

    def _close_after_fork(self) -> None:
        fd = self._fd
        self._fd = -1
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> OFDLease:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


class OFDLockFile:
    """Keep multiple workers from changing the same shared resource at once.

    This is useful when workers might update the same cache entry, download the
    same object, or create the same output. Give each resource an integer using
    ``start``. Workers using different integers proceed independently. Workers
    using the same integer may share access or require exclusive access,
    according to ``mode``.

    Unlike ``fcntl.flock``, this does not lock the whole file and unnecessarily
    block unrelated resource numbers. Unlike ``multiprocessing.Lock``, it does
    not require processes to share the same Python lock object; independently
    started processes can coordinate by agreeing on the lock-file path.

    OFD stands for "open file description." The important property is that each
    successful acquisition is independent: releasing one lock does not release
    another lock held by the same process.

    The lock file serves only as the shared identifier for the locks and may
    remain empty; it is not the data being protected. ``start`` selects a byte
    position only as a numeric key, and all workers must agree on which number
    represents each resource. The operating system tracks active locks outside
    the file's contents.

    All cooperating processes must open the same stable lock file. Use
    :meth:`create` when the caller has established that it is safe to create a
    new anchor; the normal constructor only opens an existing one. Entering the
    ``OFDLockFile`` context only arranges to close that file afterward; it does
    not lock a resource. :meth:`acquire` performs the actual lock operation and
    returns an already-active :class:`OFDLease`, or ``None`` on timeout. Entering
    the lease context does not acquire it again; it guarantees release.

    A creator may store an opaque identity in the anchor and require that value
    when reopening it. The identity does not participate in locking; it lets an
    owner detect path replacement across separate process lifetimes.

    Example:
        Suppose two counters live in a database or data file. A separate, empty
        ``counters.lock`` file can use ``start=0`` for the first counter and
        ``start=1`` for the second. This updates the first counter::

            with OFDLockFile("counters.lock") as locks:  # no resource lock yet
                lease = locks.acquire(
                    start=0,
                    mode=OFDLockMode.EXCLUSIVE,
                    timeout=1.0,
                )  # the lock is active before acquire() returns
                if lease is None:
                    raise TimeoutError("the first counter is busy")
                with lease:  # already locked; release when the block ends
                    increment_first_counter()
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        expected_identity: bytes | None = None,
    ) -> None:
        lock_path = Path(path)
        self.path = lock_path
        self._anchor_fd = -1
        backend = _select_backend()
        self._backend = backend
        self._creator_pid = os.getpid()
        self._identity = (-1, -1)
        with _FORK_GUARD:
            fd, anchor_stat = _open_regular_file(lock_path)
            try:
                if expected_identity is not None:
                    _validate_file_identity(fd, lock_path, expected_identity)
                self._adopt_anchor(lock_path, fd, anchor_stat, backend)
            except BaseException:
                os.close(fd)
                raise

    @classmethod
    def create(
        cls,
        path: str | os.PathLike[str],
        *,
        identity: bytes = b"",
    ) -> OFDLockFile:
        """Exclusively create, initialize, and adopt a private lock anchor.

        The caller decides whether creating this path is safe. This method never
        opens an existing path and returns with the exact newly-created inode as
        the live anchor, avoiding a create-close-reopen identity gap.
        """
        lock_path = Path(path)
        backend = _select_backend()
        with _FORK_GUARD:
            fd, anchor_stat = _create_regular_file(lock_path, identity=identity)
            self = cls.__new__(cls)
            try:
                self._adopt_anchor(lock_path, fd, anchor_stat, backend)
            except BaseException:
                os.close(fd)
                raise
            return self

    def _adopt_anchor(
        self,
        path: Path,
        fd: int,
        anchor_stat: os.stat_result,
        backend: _NativeOFDBackend,
    ) -> None:
        self.path = path
        self._anchor_fd = -1
        self._backend = backend
        self._creator_pid = os.getpid()
        self._identity = (anchor_stat.st_dev, anchor_stat.st_ino)
        try:
            _LIVE_FILES.add(self)
        except BaseException:
            self._anchor_fd = -1
            raise
        self._anchor_fd = fd

    @property
    def backend_info(self) -> OFDBackendInfo:
        """Return the native backend used by this lock file."""
        return self._backend.info

    @property
    def closed(self) -> bool:
        """Whether this object can no longer acquire locks."""
        return self._anchor_fd < 0

    def try_acquire(
        self,
        *,
        start: int,
        length: int = 1,
        mode: OFDLockMode,
    ) -> OFDLease | None:
        """Try to lock the resource now.

        A returned lease is already active. ``None`` means another caller
        currently holds a conflicting lock.
        """
        fd = self._open_acquisition_fd()
        try:
            lease = self._try_acquire_fd(
                fd,
                start=start,
                length=length,
                mode=mode,
                recheck_path=False,
            )
        except BaseException:
            _close_pending_acquisition_fd(fd, suppress_errors=True)
            raise
        if lease is None:
            _close_pending_acquisition_fd(fd, suppress_errors=False)
        return lease

    def acquire(
        self,
        *,
        start: int,
        length: int = 1,
        mode: OFDLockMode,
        timeout: float,
        initial_backoff: float = 0.0005,
        max_backoff: float = 0.01,
    ) -> OFDLease | None:
        """Wait up to ``timeout`` to lock the requested resource.

        A returned lease is already active. ``None`` means the timeout expired
        while another caller held a conflicting lock.
        """
        if timeout < 0:
            raise ValueError(f"timeout must be non-negative, got {timeout}")
        if initial_backoff <= 0 or max_backoff <= 0:
            raise ValueError("OFD retry backoffs must be positive")
        if initial_backoff > max_backoff:
            raise ValueError("initial_backoff cannot exceed max_backoff")

        fd = self._open_acquisition_fd()
        deadline = time.monotonic() + timeout
        backoff = initial_backoff
        first_attempt = True
        try:
            while True:
                lease = self._try_acquire_fd(
                    fd,
                    start=start,
                    length=length,
                    mode=mode,
                    recheck_path=not first_attempt,
                )
                if lease is not None:
                    return lease
                first_attempt = False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(backoff, remaining))
                backoff = min(backoff * 2, max_backoff)
        except BaseException:
            _close_pending_acquisition_fd(fd, suppress_errors=True)
            raise

        _close_pending_acquisition_fd(fd, suppress_errors=False)
        return None

    def _open_acquisition_fd(self) -> int:
        with _FORK_GUARD:
            self._check_usable()
            fd, opened_stat = _open_regular_file(self.path)
            try:
                if (opened_stat.st_dev, opened_stat.st_ino) != self._identity:
                    raise OFDLockUnavailable(
                        f"OFD lock inode changed while opening {self.path}"
                    )
                _PENDING_ACQUISITION_FDS.add(fd)
                return fd
            except BaseException:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise

    def _try_acquire_fd(
        self,
        fd: int,
        *,
        start: int,
        length: int,
        mode: OFDLockMode,
        recheck_path: bool,
    ) -> OFDLease | None:
        with _FORK_GUARD:
            self._check_usable()
            if recheck_path:
                self._check_path_identity()
            if not self._backend.try_lock(fd, mode, start, length):
                return None
            lease = OFDLease(
                fd=fd,
                backend=self._backend,
                mode=mode,
                start=start,
                length=length,
            )
            _PENDING_ACQUISITION_FDS.discard(fd)
            return lease

    def _check_path_identity(self) -> None:
        path_stat = os.stat(self.path, follow_symlinks=False)
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or (path_stat.st_dev, path_stat.st_ino) != self._identity
        ):
            raise OFDLockUnavailable(
                f"OFD lock inode changed while opening {self.path}"
            )

    def close(self) -> None:
        """Close the anchor descriptor without affecting independent leases."""
        with _FORK_GUARD:
            fd = self._anchor_fd
            if fd < 0:
                return
            self._anchor_fd = -1
            _LIVE_FILES.discard(self)
            os.close(fd)

    def _check_usable(self) -> None:
        if self._anchor_fd < 0:
            raise OFDLockUnavailable(f"OFD lock file is closed: {self.path}")
        if os.getpid() != self._creator_pid:
            raise OFDLockUnavailable(
                f"OFD lock file was inherited across fork: {self.path}"
            )

    def _close_after_fork(self) -> None:
        fd = self._anchor_fd
        self._anchor_fd = -1
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> OFDLockFile:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


def _before_fork() -> None:
    _FORK_GUARD.acquire()


def _after_fork_parent() -> None:
    _FORK_GUARD.release()


def _after_fork_child() -> None:
    try:
        for fd in tuple(_PENDING_ACQUISITION_FDS):
            try:
                os.close(fd)
            except OSError:
                pass
        _PENDING_ACQUISITION_FDS.clear()
        for lease in tuple(_LIVE_LEASES):
            lease._close_after_fork()
        for lock_file in tuple(_LIVE_FILES):
            lock_file._close_after_fork()
        _LIVE_LEASES.clear()
        _LIVE_FILES.clear()
    finally:
        _FORK_GUARD.release()


def _close_pending_acquisition_fd(fd: int, *, suppress_errors: bool) -> None:
    with _FORK_GUARD:
        _PENDING_ACQUISITION_FDS.discard(fd)
        try:
            os.close(fd)
        except OSError:
            if not suppress_errors:
                raise


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_before_fork,
        after_in_parent=_after_fork_parent,
        after_in_child=_after_fork_child,
    )


__all__ = [
    "OFDBackendInfo",
    "OFDLease",
    "OFDLockFile",
    "OFDLockMode",
    "OFDLockUnavailable",
    "OFDProbeResult",
    "ofd_backend_info",
    "probe_ofd_support",
]
