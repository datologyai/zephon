# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Lifetime-lease session for the node-shared decoded Parquet RG cache."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import stat
import uuid
import weakref
from pathlib import Path
from typing import Any

from filelock import FileLock

from zephon._internal.io.formats.parquet_cache.control import (
    CONTROL_FORMAT_VERSION,
    PAYLOAD_FORMAT_VERSION,
    ParquetRGControl,
)
from zephon._internal.io.ofd_lock import (
    OFDLease,
    OFDLockFile,
    OFDLockMode,
    OFDLockUnavailable,
    probe_ofd_support,
)
from zephon._internal.utils.atomic import atomic_write_bytes

_MARKER_FILENAME = ".zephon-parquet-rg-cache-owner"
_RESET_LOCK_FILENAME = ".reset.lock"
_LOCK_FILENAME = "locks.ofd"
_GENERATION_DIRNAME = "v1"
_CONTROL_FILENAME = "control.bin"
_ENTRIES_DIRNAME = "entries"
_MARKER_FORMAT_VERSION = 2
_SESSION_RANGE = 2


class ParquetRGSessionError(RuntimeError):
    """Raised when a decoded-cache namespace cannot be safely joined/reset."""


class ParquetRGSessionInUseError(ParquetRGSessionError):
    """Raised when live peers use incompatible decoded-cache configuration."""


def _load_marker(path: Path) -> dict[str, Any] | None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        with os.fdopen(os.open(path, flags), "r", encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _close_session_resources(
    *,
    creator_pid: int,
    control: ParquetRGControl,
    lifetime_lease: OFDLease,
    lock_file: OFDLockFile,
) -> None:
    if os.getpid() != creator_pid:
        return
    for resource in (control, lifetime_lease, lock_file):
        with contextlib.suppress(Exception):
            resource.close()


class ParquetRGSession:
    """Own one process's attachment to a decoded-cache generation.

    It validates the cache root, creates or joins a compatible generation, and
    keeps its control mapping and OFD lifetime lease alive for ``ParquetRGCache``.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        slot_count: int,
        limit_bytes: int,
        catalog_fingerprint: str,
        configuration_fingerprint: str,
    ) -> None:
        if slot_count <= 0:
            raise ValueError("Decoded RG session requires at least one row group")
        if limit_bytes <= 0:
            raise ValueError("Decoded RG cache limit must be positive")
        if not configuration_fingerprint:
            raise ValueError("Decoded RG configuration fingerprint must not be empty")

        requested_root = Path(root).expanduser()
        requested_root.mkdir(parents=True, exist_ok=True)
        self.root = requested_root.resolve()
        self._marker_path = self.root / _MARKER_FILENAME
        self._reset_lock = FileLock(str(self.root / _RESET_LOCK_FILENAME))
        self._locks_path = self.root / _LOCK_FILENAME
        self._generation_dir = self.root / _GENERATION_DIRNAME
        self.entries_dir = self._generation_dir / _ENTRIES_DIRNAME
        self._control_path = self._generation_dir / _CONTROL_FILENAME
        self._creator_pid = os.getpid()
        self._namespace_id = ""
        self._lock_id = ""
        self.session_id = ""
        self._control: ParquetRGControl | None = None
        self._lock_file: OFDLockFile | None = None
        self._lifetime_lease: OFDLease | None = None
        self._closed = False

        try:
            with self._reset_lock:
                self._namespace_id, self._lock_id = self._ensure_owned_root_locked()
                effective_configuration = self._effective_configuration_fingerprint(
                    configuration_fingerprint
                )
                self._initialize_locked(
                    slot_count=slot_count,
                    limit_bytes=limit_bytes,
                    catalog_fingerprint=catalog_fingerprint,
                    configuration_fingerprint=effective_configuration,
                )
        except BaseException:
            self._close_partial()
            raise

        assert self._control is not None
        assert self._lock_file is not None
        assert self._lifetime_lease is not None
        self._close_finalizer = weakref.finalize(
            self,
            _close_session_resources,
            creator_pid=self._creator_pid,
            control=self._control,
            lifetime_lease=self._lifetime_lease,
            lock_file=self._lock_file,
        )

    @property
    def control(self) -> ParquetRGControl:
        self._check_usable()
        assert self._control is not None
        return self._control

    @property
    def lock_file(self) -> OFDLockFile:
        self._check_usable()
        assert self._lock_file is not None
        return self._lock_file

    def close(self) -> None:
        """Release this attachment; correctness does not depend on this call."""
        if self._closed:
            return
        self._closed = True
        if os.getpid() == self._creator_pid:
            self._close_finalizer()
        self._control = None
        self._lifetime_lease = None
        self._lock_file = None

    def _check_usable(self) -> None:
        if os.getpid() != self._creator_pid:
            raise RuntimeError("Decoded RG session was inherited across fork")
        if self._closed:
            raise RuntimeError("Decoded RG session is closed")

    def _ensure_owned_root_locked(self) -> tuple[str, str]:
        marker = _load_marker(self._marker_path)
        if marker is None:
            allowed = {_RESET_LOCK_FILENAME}
            if any(child.name not in allowed for child in self.root.iterdir()):
                raise ParquetRGSessionError(
                    f"Refusing unowned non-empty decoded RG cache root {self.root}"
                )
            namespace_id = str(uuid.uuid4())
            lock_id = str(uuid.uuid4())
            atomic_write_bytes(
                self._marker_path,
                json.dumps(
                    {
                        "format_version": _MARKER_FORMAT_VERSION,
                        "namespace_id": namespace_id,
                        "lock_id": lock_id,
                    },
                    sort_keys=True,
                ).encode("utf-8"),
                fsync=True,
            )
            return namespace_id, lock_id
        if marker.get("format_version") != _MARKER_FORMAT_VERSION:
            raise ParquetRGSessionError(
                f"Decoded RG cache marker version mismatch at {self._marker_path}"
            )
        try:
            return (
                str(uuid.UUID(str(marker["namespace_id"]))),
                str(uuid.UUID(str(marker["lock_id"]))),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ParquetRGSessionError(
                f"Invalid decoded RG ownership marker at {self._marker_path}"
            ) from exc

    def _initialize_locked(
        self,
        *,
        slot_count: int,
        limit_bytes: int,
        catalog_fingerprint: str,
        configuration_fingerprint: str,
    ) -> None:
        probe = probe_ofd_support(self.root)
        if not probe.supported:
            raise OFDLockUnavailable(
                f"Decoded RG cache filesystem lacks OFD range locks: {probe.reason}"
            )
        lock_identity = self._lock_id.encode("ascii")
        try:
            self._lock_file = OFDLockFile(
                self._locks_path,
                expected_identity=lock_identity,
            )
        except FileNotFoundError:
            if self._control_path.exists():
                raise ParquetRGSessionError(
                    "Decoded RG control exists but its stable lock file is missing"
                )
            self._lock_file = OFDLockFile.create(
                self._locks_path,
                identity=lock_identity,
            )
        except (OSError, OFDLockUnavailable) as exc:
            # The ownership marker is published before the stable lock during
            # first bootstrap. If that bootstrap dies between those writes,
            # there cannot be a published generation or a live session lease.
            # Rebuild only that provably incomplete case; once control.bin is
            # published, a changed lock pathname is never safe to repair.
            if self._control_path.exists():
                raise ParquetRGSessionError(
                    f"Cannot validate decoded RG stable lock {self._locks_path}: {exc}"
                ) from exc
            self._locks_path.unlink()
            self._lock_file = OFDLockFile.create(
                self._locks_path,
                identity=lock_identity,
            )

        exclusive = self._lock_file.try_acquire(
            start=_SESSION_RANGE,
            mode=OFDLockMode.EXCLUSIVE,
        )
        if exclusive is not None:
            self._reset_and_create_locked(
                exclusive,
                slot_count=slot_count,
                limit_bytes=limit_bytes,
                catalog_fingerprint=catalog_fingerprint,
                configuration_fingerprint=configuration_fingerprint,
            )
            return

        shared = self._lock_file.try_acquire(
            start=_SESSION_RANGE,
            mode=OFDLockMode.SHARED,
        )
        if shared is None:
            raise ParquetRGSessionError("Unable to join live decoded RG session")
        try:
            control = ParquetRGControl.attach(
                self._control_path,
                slot_count=slot_count,
                limit_bytes=limit_bytes,
                catalog_fingerprint=catalog_fingerprint,
                configuration_fingerprint=configuration_fingerprint,
                namespace_uuid=uuid.UUID(self._namespace_id).bytes,
            )
        except Exception as exc:
            shared.close()
            exclusive = self._lock_file.try_acquire(
                start=_SESSION_RANGE,
                mode=OFDLockMode.EXCLUSIVE,
            )
            if exclusive is not None:
                self._reset_and_create_locked(
                    exclusive,
                    slot_count=slot_count,
                    limit_bytes=limit_bytes,
                    catalog_fingerprint=catalog_fingerprint,
                    configuration_fingerprint=configuration_fingerprint,
                )
                return
            raise ParquetRGSessionInUseError(
                "Decoded RG cache has a live incompatible or invalid generation "
                + f"at {self.root}"
            ) from exc
        self._control = control
        self._lifetime_lease = shared
        self.session_id = str(uuid.UUID(bytes=control.session_uuid))

    def _reset_and_create_locked(
        self,
        exclusive: OFDLease,
        *,
        slot_count: int,
        limit_bytes: int,
        catalog_fingerprint: str,
        configuration_fingerprint: str,
    ) -> None:
        try:
            self._reset_generation_locked()
            self.session_id = str(uuid.uuid4())
            temporary = (
                self._generation_dir / f".{_CONTROL_FILENAME}.{uuid.uuid4().hex}.tmp"
            )
            control = ParquetRGControl.create(
                temporary,
                slot_count=slot_count,
                limit_bytes=limit_bytes,
                catalog_fingerprint=catalog_fingerprint,
                configuration_fingerprint=configuration_fingerprint,
                session_uuid=uuid.UUID(self.session_id).bytes,
                namespace_uuid=uuid.UUID(self._namespace_id).bytes,
            )
            try:
                control.publish_as(self._control_path)
            except BaseException:
                control.close()
                raise
            exclusive.convert(OFDLockMode.SHARED)
            self._control = control
            self._lifetime_lease = exclusive
        except BaseException:
            exclusive.close()
            raise

    def _reset_generation_locked(self) -> None:
        try:
            info = self._generation_dir.lstat()
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(info.st_mode):
                raise ParquetRGSessionError(
                    f"Refusing symlinked decoded RG generation {self._generation_dir}"
                )
            if not stat.S_ISDIR(info.st_mode):
                raise ParquetRGSessionError(
                    f"Decoded RG generation is not a directory: {self._generation_dir}"
                )
            shutil.rmtree(self._generation_dir)
        self.entries_dir.mkdir(parents=True)

    def _effective_configuration_fingerprint(self, requested: str) -> str:
        root_stat = self.root.stat()
        payload = {
            "requested": requested,
            "namespace_id": self._namespace_id,
            "canonical_root": os.fspath(self.root),
            "device": root_stat.st_dev,
            "control_format_version": CONTROL_FORMAT_VERSION,
            "payload_format_version": PAYLOAD_FORMAT_VERSION,
            "lock_format_version": 1,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def _close_partial(self) -> None:
        for resource in (self._control, self._lifetime_lease, self._lock_file):
            if resource is None:
                continue
            with contextlib.suppress(Exception):
                resource.close()

    def __enter__(self) -> "ParquetRGSession":
        self._check_usable()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


__all__ = [
    "ParquetRGSession",
    "ParquetRGSessionError",
    "ParquetRGSessionInUseError",
]
