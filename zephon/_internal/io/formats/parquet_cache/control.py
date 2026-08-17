# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Compact file-backed state for the node-shared decoded-RG cache.

The file is mapped ``MAP_SHARED`` and indexed by a dense row-group slot. Locking
is deliberately external: exact-slot and capacity OFD leases define when these
accessors may mutate shared state.
"""

from __future__ import annotations

import contextlib
import mmap
import os
import stat
import struct
import threading
import weakref
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import TracebackType

CONTROL_FORMAT_VERSION = 4
PAYLOAD_FORMAT_VERSION = 1

_MAGIC = b"ZEPHRGC4"
_HEADER_BYTES = 4096
_PAGE_BYTES = 4096
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_I64 = struct.Struct("<q")

_OFF_MAGIC = 0
_OFF_CONTROL_VERSION = 8
_OFF_PAYLOAD_VERSION = 12
_OFF_SLOT_COUNT = 16
_OFF_FILE_BYTES = 24
_OFF_LIMIT_BYTES = 32
_OFF_ACCOUNTED_BYTES = 40
_OFF_INFLIGHT_BYTES = 48
_OFF_CAPACITY_EPOCH = 56
_OFF_CLOCK_HAND = 64
_OFF_EVICTION_PENDING = 72
_OFF_DIRTY = 73
_OFF_DISABLED = 74
_OFF_DIRTY_SLOT = 80
_OFF_CATALOG_DIGEST = 96
_OFF_CONFIGURATION_DIGEST = 128
_OFF_SESSION_UUID = 160
_OFF_NAMESPACE_UUID = 176
_OFF_THRASH_ARMED = 192
_OFF_POST_FILL_PUBLICATIONS = 200
_OFF_POST_FILL_RELOADS = 208
_MAX_U64 = (1 << 64) - 1

_FORK_GUARD = threading.RLock()
_LIVE_CONTROLS: weakref.WeakSet[ParquetRGControl] = weakref.WeakSet()


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class RGControlLayout:
    """Byte extents of one decoded-RG control generation."""

    slot_count: int
    states_offset: int
    refbits_offset: int
    evicted_offset: int
    sizes_offset: int
    file_bytes: int

    @classmethod
    def for_slots(cls, slot_count: int) -> "RGControlLayout":
        if slot_count <= 0:
            raise ValueError("Decoded RG control slot count must be positive")
        states_offset = _HEADER_BYTES
        refbits_offset = states_offset + slot_count
        evicted_offset = refbits_offset + slot_count
        sizes_offset = _align_up(evicted_offset + slot_count, _U64.size)
        file_bytes = _align_up(sizes_offset + slot_count * _U64.size, _PAGE_BYTES)
        return cls(
            slot_count=slot_count,
            states_offset=states_offset,
            refbits_offset=refbits_offset,
            evicted_offset=evicted_offset,
            sizes_offset=sizes_offset,
            file_bytes=file_bytes,
        )


class DecodedRGState(IntEnum):
    """Shared lifecycle state for one decoded row-group payload."""

    EMPTY = 0
    BUILDING = 1
    READY = 2
    EVICTING = 3


@dataclass(frozen=True)
class RGControlHeader:
    """Snapshot of the small shared header."""

    slot_count: int
    limit_bytes: int
    accounted_bytes: int
    inflight_bytes: int
    capacity_epoch: int
    clock_hand: int
    eviction_pending: bool
    dirty: bool
    dirty_slot: int | None
    disabled: bool
    thrash_armed: bool
    post_fill_publications: int
    post_fill_reloads: int


@dataclass(frozen=True)
class RGSlotSnapshot:
    """State of one dense RG slot."""

    state: DecodedRGState
    refbit: bool
    bytes: int


class RGControlMutation:
    """One slot mutation protected by a persistent dirty-slot marker.

    The old values live only in this process and are used to check the normal
    path. If the process dies, the shared marker remains set and recovery
    discards this slot and recomputes both byte counters from the dense arrays.
    """

    def __init__(self, control: "ParquetRGControl", slot: int) -> None:
        self._control = control
        self._slot = slot
        self._old_slot: RGSlotSnapshot | None = None
        self._old_accounted = 0
        self._old_inflight = 0
        self._entered = False

    def __enter__(self) -> "RGControlMutation":
        old_slot, accounted, inflight = self._control._begin_mutation(self._slot)
        self._old_slot = old_slot
        self._old_accounted = accounted
        self._old_inflight = inflight
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        try:
            if exc_type is None:
                assert self._old_slot is not None
                self._control._finish_mutation(
                    self._slot,
                    old_slot=self._old_slot,
                    old_accounted=self._old_accounted,
                    old_inflight=self._old_inflight,
                )
        finally:
            self._entered = False
        return False

    def set_slot(
        self,
        *,
        state: DecodedRGState,
        refbit: bool,
        bytes: int,
    ) -> None:
        """Replace the slot and derive global byte deltas automatically."""
        if not self._entered:
            raise RuntimeError("Decoded RG control mutation is not active")
        self._control._validate_slot_fields(state, bytes)
        mapping = self._control._check_slot(self._slot)
        self._control._set_slot_unchecked(
            mapping,
            self._slot,
            state=state,
            refbit=refbit,
            bytes=bytes,
        )


class ParquetRGControl:
    """Shared metadata for one decoded-cache generation, backed by ``control.bin``.

    ``ParquetRGSession`` creates or attaches it, while ``ParquetRGAdmission``
    updates slot states and byte counters under locks. It never stores row data.
    """

    def __init__(
        self,
        *,
        path: Path,
        fd: int,
        mapping: mmap.mmap,
        layout: RGControlLayout,
        limit_bytes: int,
        session_uuid: bytes,
    ) -> None:
        self._path = path
        self._fd = fd
        self._mapping: mmap.mmap | None = mapping
        self._layout = layout
        self.slot_count = layout.slot_count
        self.limit_bytes = limit_bytes
        self.session_uuid = session_uuid
        self._creator_pid = os.getpid()
        with _FORK_GUARD:
            _LIVE_CONTROLS.add(self)

    @classmethod
    def create(
        cls,
        path: str | os.PathLike[str],
        *,
        slot_count: int,
        limit_bytes: int,
        catalog_fingerprint: str,
        configuration_fingerprint: str,
        session_uuid: bytes,
        namespace_uuid: bytes,
    ) -> "ParquetRGControl":
        """Create, fully allocate, initialize, and map an unpublished file."""
        layout = RGControlLayout.for_slots(slot_count)
        if limit_bytes <= 0:
            raise ValueError("Decoded RG cache limit must be positive")
        catalog_digest = _parse_fingerprint(catalog_fingerprint)
        configuration_digest = _parse_fingerprint(configuration_fingerprint)
        _validate_uuid_bytes("session_uuid", session_uuid)
        _validate_uuid_bytes("namespace_uuid", namespace_uuid)

        control_path = Path(path)
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(control_path, flags, 0o600)
        mapping: mmap.mmap | None = None
        try:
            os.fchmod(fd, 0o600)
            _allocate_file(fd, layout.file_bytes)
            mapping = mmap.mmap(fd, layout.file_bytes, access=mmap.ACCESS_WRITE)
            mapping[_OFF_MAGIC : _OFF_MAGIC + len(_MAGIC)] = _MAGIC
            _U32.pack_into(mapping, _OFF_CONTROL_VERSION, CONTROL_FORMAT_VERSION)
            _U32.pack_into(mapping, _OFF_PAYLOAD_VERSION, PAYLOAD_FORMAT_VERSION)
            _U64.pack_into(mapping, _OFF_SLOT_COUNT, slot_count)
            _U64.pack_into(mapping, _OFF_FILE_BYTES, layout.file_bytes)
            _U64.pack_into(mapping, _OFF_LIMIT_BYTES, limit_bytes)
            _I64.pack_into(mapping, _OFF_DIRTY_SLOT, -1)
            mapping[_OFF_CATALOG_DIGEST : _OFF_CATALOG_DIGEST + 32] = catalog_digest
            mapping[_OFF_CONFIGURATION_DIGEST : _OFF_CONFIGURATION_DIGEST + 32] = (
                configuration_digest
            )
            mapping[_OFF_SESSION_UUID : _OFF_SESSION_UUID + 16] = session_uuid
            mapping[_OFF_NAMESPACE_UUID : _OFF_NAMESPACE_UUID + 16] = namespace_uuid
            return cls(
                path=control_path,
                fd=fd,
                mapping=mapping,
                layout=layout,
                limit_bytes=limit_bytes,
                session_uuid=session_uuid,
            )
        except BaseException:
            if mapping is not None:
                with contextlib.suppress(Exception):
                    mapping.close()
            with contextlib.suppress(OSError):
                os.close(fd)
            with contextlib.suppress(OSError):
                control_path.unlink()
            raise

    @classmethod
    def attach(
        cls,
        path: str | os.PathLike[str],
        *,
        slot_count: int,
        limit_bytes: int,
        catalog_fingerprint: str,
        configuration_fingerprint: str,
        namespace_uuid: bytes,
    ) -> "ParquetRGControl":
        """Validate and map the fixed published control file."""
        layout = RGControlLayout.for_slots(slot_count)
        control_path = Path(path)
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(control_path, flags)
        mapping: mmap.mmap | None = None
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(f"Decoded RG control is not regular: {control_path}")
            if info.st_size != layout.file_bytes:
                raise RuntimeError(
                    f"Decoded RG control length mismatch at {control_path}: "
                    + f"expected {layout.file_bytes}, got {info.st_size}"
                )
            mapping = mmap.mmap(fd, layout.file_bytes, access=mmap.ACCESS_WRITE)
            _validate_mapping(
                mapping,
                layout=layout,
                limit_bytes=limit_bytes,
                catalog_digest=_parse_fingerprint(catalog_fingerprint),
                configuration_digest=_parse_fingerprint(configuration_fingerprint),
                namespace_uuid=namespace_uuid,
            )
            return cls(
                path=control_path,
                fd=fd,
                mapping=mapping,
                layout=layout,
                limit_bytes=limit_bytes,
                session_uuid=bytes(mapping[_OFF_SESSION_UUID : _OFF_SESSION_UUID + 16]),
            )
        except BaseException:
            if mapping is not None:
                with contextlib.suppress(Exception):
                    mapping.close()
            with contextlib.suppress(OSError):
                os.close(fd)
            raise

    def publish_as(self, path: str | os.PathLike[str]) -> None:
        """Atomically publish a fully initialized temporary control file."""
        self._check_live()
        destination = Path(path)
        if destination.parent != self._path.parent:
            raise ValueError(
                "Decoded RG control publication must stay in one directory"
            )
        os.replace(self._path, destination)
        self._path = destination

    def header(self) -> RGControlHeader:
        mapping = self._check_live()
        return self._header_unchecked(mapping)

    def _header_unchecked(self, mapping: mmap.mmap) -> RGControlHeader:
        dirty_slot = _I64.unpack_from(mapping, _OFF_DIRTY_SLOT)[0]
        return RGControlHeader(
            slot_count=self.slot_count,
            limit_bytes=self.limit_bytes,
            accounted_bytes=_U64.unpack_from(mapping, _OFF_ACCOUNTED_BYTES)[0],
            inflight_bytes=_U64.unpack_from(mapping, _OFF_INFLIGHT_BYTES)[0],
            capacity_epoch=_U64.unpack_from(mapping, _OFF_CAPACITY_EPOCH)[0],
            clock_hand=_U64.unpack_from(mapping, _OFF_CLOCK_HAND)[0],
            eviction_pending=bool(mapping[_OFF_EVICTION_PENDING]),
            dirty=bool(mapping[_OFF_DIRTY]),
            dirty_slot=dirty_slot if dirty_slot >= 0 else None,
            disabled=bool(mapping[_OFF_DISABLED]),
            thrash_armed=bool(mapping[_OFF_THRASH_ARMED]),
            post_fill_publications=_U64.unpack_from(
                mapping, _OFF_POST_FILL_PUBLICATIONS
            )[0],
            post_fill_reloads=_U64.unpack_from(mapping, _OFF_POST_FILL_RELOADS)[0],
        )

    def slot(self, slot: int) -> RGSlotSnapshot:
        mapping = self._check_slot(slot)
        return self._slot_unchecked(mapping, slot)

    def _slot_unchecked(self, mapping: mmap.mmap, slot: int) -> RGSlotSnapshot:
        return RGSlotSnapshot(
            state=self._slot_state(mapping, slot),
            refbit=bool(mapping[self._layout.refbits_offset + slot]),
            bytes=_U64.unpack_from(
                mapping, self._layout.sizes_offset + slot * _U64.size
            )[0],
        )

    def count_slots(self, state: DecodedRGState) -> int:
        """Count slots in ``state`` after one mapping-liveness check."""
        mapping = self._check_live()
        start = self._layout.states_offset
        return mapping[start : start + self.slot_count].count(int(state))

    def mutation(self, slot: int) -> RGControlMutation:
        """Create one dirty-marked exact-slot update."""
        self._check_slot(slot)
        return RGControlMutation(self, slot)

    def set_replacement_state(
        self,
        *,
        clock_hand: int | None = None,
        eviction_pending: bool | None = None,
    ) -> None:
        """Update non-accounting CLOCK fields while capacity EX is held."""
        if clock_hand is None and eviction_pending is None:
            raise ValueError("Replacement-state update must change at least one field")
        self._check_live()
        if clock_hand is not None:
            if clock_hand < 0 or clock_hand >= self.slot_count:
                raise ValueError(f"CLOCK hand out of range: {clock_hand}")
            self._set_u64(_OFF_CLOCK_HAND, clock_hand)
        if eviction_pending is not None:
            self._set_byte(_OFF_EVICTION_PENDING, int(eviction_pending))
        self._set_u64(_OFF_CAPACITY_EPOCH, self._get_u64(_OFF_CAPACITY_EPOCH) + 1)

    def touch_ready(self, slot: int) -> bool:
        """Set a READY entry's CLOCK refbit under exact SH or EX."""
        mapping = self._check_slot(slot)
        if self._slot_state(mapping, slot) is not DecodedRGState.READY:
            return False
        refbit_offset = self._layout.refbits_offset + slot
        if not mapping[refbit_offset]:
            mapping[refbit_offset] = 1
        return True

    def clear_ready_refbit(self, slot: int) -> bool:
        """Give a READY entry its CLOCK second chance under exact EX."""
        mapping = self._check_slot(slot)
        if self._slot_state(mapping, slot) is not DecodedRGState.READY:
            return False
        refbit_offset = self._layout.refbits_offset + slot
        if not mapping[refbit_offset]:
            return False
        mapping[refbit_offset] = 0
        return True

    def was_evicted(self, slot: int) -> bool:
        """Return whether this dense slot was removed by decoded-cache eviction."""
        self._check_slot(slot)
        return bool(self._get_byte(self._layout.evicted_offset + slot))

    def mark_evicted(self, slot: int) -> None:
        """Remember an eviction while the caller holds this slot's exact lock."""
        self._check_slot(slot)
        self._set_byte(self._layout.evicted_offset + slot, 1)

    def clear_evicted(self, slot: int) -> None:
        """Consume an eviction marker after successfully republishing its RG."""
        self._check_slot(slot)
        self._set_byte(self._layout.evicted_offset + slot, 0)

    def arm_thrash_detection(self) -> None:
        """Start node-wide churn counters after the first successful eviction."""
        self._check_live()
        if self._get_byte(_OFF_THRASH_ARMED):
            return
        self._set_u64(_OFF_POST_FILL_PUBLICATIONS, 0)
        self._set_u64(_OFF_POST_FILL_RELOADS, 0)
        self._set_byte(_OFF_THRASH_ARMED, 1)

    def note_post_fill_publication(self, *, reloaded: bool) -> tuple[int, int]:
        """Record one publication while the caller holds capacity EX."""
        self._check_live()
        if not self._get_byte(_OFF_THRASH_ARMED):
            return 0, 0
        publications = self._get_u64(_OFF_POST_FILL_PUBLICATIONS)
        reloads = self._get_u64(_OFF_POST_FILL_RELOADS)
        if reloads > publications:
            # These counters are advisory. Recover impossible state locally
            # instead of making warning bookkeeping disable a healthy cache.
            publications = 0
            reloads = 0
        if publications < _MAX_U64:
            publications += 1
        if reloaded and reloads < publications:
            reloads += 1
        self._set_u64(_OFF_POST_FILL_PUBLICATIONS, publications)
        self._set_u64(_OFF_POST_FILL_RELOADS, reloads)
        return publications, reloads

    def discard_dirty_slot_locked(self, slot: int) -> bool:
        """Discard the uncertain slot and rebuild byte counters by dense scan."""
        mapping = self._check_live()
        header = self._header_unchecked(mapping)
        if not header.dirty:
            return True
        if header.dirty_slot != slot:
            raise RuntimeError(
                f"Dirty decoded RG slot is {header.dirty_slot}, not locked slot {slot}"
            )
        self._set_slot_unchecked(
            mapping,
            slot,
            state=DecodedRGState.EMPTY,
            refbit=False,
            bytes=0,
        )
        accounted = 0
        inflight = 0
        try:
            for candidate in range(self.slot_count):
                snapshot = self._slot_unchecked(mapping, candidate)
                self._validate_slot_fields(snapshot.state, snapshot.bytes)
                if snapshot.state is not DecodedRGState.EMPTY:
                    accounted += snapshot.bytes
                if snapshot.state is DecodedRGState.BUILDING:
                    inflight += snapshot.bytes
            if inflight > accounted or accounted > self.limit_bytes:
                raise RuntimeError("Decoded RG recount violates configured quota")
        except Exception:
            mapping[_OFF_DISABLED] = 1
            return False
        _U64.pack_into(mapping, _OFF_ACCOUNTED_BYTES, accounted)
        _U64.pack_into(mapping, _OFF_INFLIGHT_BYTES, inflight)
        _U64.pack_into(mapping, _OFF_CAPACITY_EPOCH, header.capacity_epoch + 1)
        mapping[_OFF_DIRTY] = 0
        _I64.pack_into(mapping, _OFF_DIRTY_SLOT, -1)
        return True

    def disable_locked(self) -> None:
        """Disable this live generation after irrecoverable metadata damage."""
        self._set_byte(_OFF_DISABLED, 1)

    def close(self) -> None:
        """Close this process's map and descriptor without unlinking the file."""
        with _FORK_GUARD:
            mapping = self._mapping
            if mapping is None:
                return
            self._mapping = None
            _LIVE_CONTROLS.discard(self)
            with contextlib.suppress(Exception):
                mapping.close()
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = -1

    def _begin_mutation(self, slot: int) -> tuple[RGSlotSnapshot, int, int]:
        header = self.header()
        if header.disabled:
            raise RuntimeError("Decoded RG cache generation is disabled")
        if header.dirty:
            raise RuntimeError("Decoded RG control has an unrecovered dirty slot")
        old_slot = self.slot(slot)
        self._set_i64(_OFF_DIRTY_SLOT, slot)
        self._set_byte(_OFF_DIRTY, 1)
        return old_slot, header.accounted_bytes, header.inflight_bytes

    def _finish_mutation(
        self,
        slot: int,
        *,
        old_slot: RGSlotSnapshot,
        old_accounted: int,
        old_inflight: int,
    ) -> None:
        new_slot = self.slot(slot)
        self._validate_slot_fields(new_slot.state, new_slot.bytes)
        old_charge = 0 if old_slot.state is DecodedRGState.EMPTY else old_slot.bytes
        new_charge = 0 if new_slot.state is DecodedRGState.EMPTY else new_slot.bytes
        old_build = old_slot.bytes if old_slot.state is DecodedRGState.BUILDING else 0
        new_build = new_slot.bytes if new_slot.state is DecodedRGState.BUILDING else 0
        accounted = old_accounted - old_charge + new_charge
        inflight = old_inflight - old_build + new_build
        if inflight < 0 or inflight > accounted or accounted > self.limit_bytes:
            raise RuntimeError("Decoded RG accounting invariant failed")
        self._set_u64(_OFF_ACCOUNTED_BYTES, accounted)
        self._set_u64(_OFF_INFLIGHT_BYTES, inflight)
        self._set_u64(_OFF_CAPACITY_EPOCH, self._get_u64(_OFF_CAPACITY_EPOCH) + 1)
        self._set_byte(_OFF_DIRTY, 0)
        self._set_i64(_OFF_DIRTY_SLOT, -1)

    @staticmethod
    def _validate_slot_fields(state: DecodedRGState, bytes: int) -> None:
        if bytes < 0:
            raise ValueError("Decoded RG slot bytes must be non-negative")
        if state is DecodedRGState.EMPTY and bytes != 0:
            raise ValueError("EMPTY decoded RG slot must have zero bytes")
        if state is not DecodedRGState.EMPTY and bytes == 0:
            raise ValueError("Non-empty decoded RG slot must have positive bytes")

    def _set_slot_unchecked(
        self,
        mapping: mmap.mmap,
        slot: int,
        *,
        state: DecodedRGState,
        refbit: bool,
        bytes: int,
    ) -> None:
        mapping[self._layout.refbits_offset + slot] = int(refbit)
        _U64.pack_into(mapping, self._layout.sizes_offset + slot * _U64.size, bytes)
        mapping[self._layout.states_offset + slot] = int(state)

    def _check_slot(self, slot: int) -> mmap.mmap:
        mapping = self._check_live()
        if slot < 0 or slot >= self.slot_count:
            raise IndexError(slot)
        return mapping

    def _slot_state(self, mapping: mmap.mmap, slot: int) -> DecodedRGState:
        raw_state = mapping[self._layout.states_offset + slot]
        try:
            return DecodedRGState(raw_state)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid decoded RG state {raw_state} at slot {slot}"
            ) from exc

    def _check_live(self) -> mmap.mmap:
        if os.getpid() != self._creator_pid:
            raise RuntimeError("Decoded RG control mapping was inherited across fork")
        if self._mapping is None:
            raise RuntimeError("Decoded RG control mapping is closed")
        return self._mapping

    def _close_after_fork(self) -> None:
        mapping = self._mapping
        self._mapping = None
        if mapping is not None:
            with contextlib.suppress(Exception):
                mapping.close()
        if self._fd >= 0:
            with contextlib.suppress(OSError):
                os.close(self._fd)
        self._fd = -1

    def _get_byte(self, offset: int) -> int:
        return self._check_live()[offset]

    def _set_byte(self, offset: int, value: int) -> None:
        self._check_live()[offset] = value

    def _get_u64(self, offset: int) -> int:
        return _U64.unpack_from(self._check_live(), offset)[0]

    def _set_u64(self, offset: int, value: int) -> None:
        _U64.pack_into(self._check_live(), offset, value)

    def _get_i64(self, offset: int) -> int:
        return _I64.unpack_from(self._check_live(), offset)[0]

    def _set_i64(self, offset: int, value: int) -> None:
        _I64.pack_into(self._check_live(), offset, value)

    def __enter__(self) -> "ParquetRGControl":
        self._check_live()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort at shutdown
        with contextlib.suppress(BaseException):
            self.close()


def _parse_fingerprint(fingerprint: str) -> bytes:
    if not fingerprint.startswith("sha256:"):
        raise ValueError("Decoded RG fingerprint must use sha256")
    try:
        digest = bytes.fromhex(fingerprint[7:])
    except ValueError as exc:
        raise ValueError("Invalid decoded RG fingerprint") from exc
    if len(digest) != 32:
        raise ValueError("Invalid decoded RG fingerprint length")
    return digest


def _validate_uuid_bytes(name: str, value: bytes) -> None:
    if len(value) != 16:
        raise ValueError(f"{name} must contain exactly 16 bytes")


def _allocate_file(fd: int, file_bytes: int) -> None:
    os.ftruncate(fd, file_bytes)
    posix_fallocate = getattr(os, "posix_fallocate", None)
    if posix_fallocate is not None:
        try:
            posix_fallocate(fd, 0, file_bytes)
            return
        except OSError:
            pass
    pwrite = getattr(os, "pwrite", None)
    if pwrite is None:
        os.lseek(fd, 0, os.SEEK_SET)
        remaining = file_bytes
        page = b"\0" * _PAGE_BYTES
        while remaining:
            written = os.write(fd, page[: min(_PAGE_BYTES, remaining)])
            if written <= 0:
                raise OSError("Unable to allocate decoded RG control file")
            remaining -= written
        return
    zero = b"\0"
    for offset in range(_PAGE_BYTES - 1, file_bytes, _PAGE_BYTES):
        pwrite(fd, zero, offset)
    pwrite(fd, zero, file_bytes - 1)


def _validate_mapping(
    mapping: mmap.mmap,
    *,
    layout: RGControlLayout,
    limit_bytes: int,
    catalog_digest: bytes,
    configuration_digest: bytes,
    namespace_uuid: bytes,
) -> None:
    if mapping[_OFF_MAGIC : _OFF_MAGIC + len(_MAGIC)] != _MAGIC:
        raise RuntimeError("Decoded RG control has invalid magic")
    if _U32.unpack_from(mapping, _OFF_CONTROL_VERSION)[0] != CONTROL_FORMAT_VERSION:
        raise RuntimeError("Decoded RG control format version mismatch")
    if _U32.unpack_from(mapping, _OFF_PAYLOAD_VERSION)[0] != PAYLOAD_FORMAT_VERSION:
        raise RuntimeError("Decoded RG payload format version mismatch")
    for offset, expected, label in (
        (_OFF_SLOT_COUNT, layout.slot_count, "slot count"),
        (_OFF_FILE_BYTES, layout.file_bytes, "file length"),
        (_OFF_LIMIT_BYTES, limit_bytes, "limit"),
    ):
        actual = _U64.unpack_from(mapping, offset)[0]
        if actual != expected:
            raise RuntimeError(
                f"Decoded RG control {label} mismatch: expected {expected}, got {actual}"
            )
    for offset, expected, label in (
        (_OFF_CATALOG_DIGEST, catalog_digest, "catalog fingerprint"),
        (_OFF_CONFIGURATION_DIGEST, configuration_digest, "configuration"),
        (_OFF_NAMESPACE_UUID, namespace_uuid, "namespace UUID"),
    ):
        if mapping[offset : offset + len(expected)] != expected:
            raise RuntimeError(f"Decoded RG control {label} mismatch")
    accounted = _U64.unpack_from(mapping, _OFF_ACCOUNTED_BYTES)[0]
    inflight = _U64.unpack_from(mapping, _OFF_INFLIGHT_BYTES)[0]
    if inflight > accounted or accounted > limit_bytes:
        raise RuntimeError("Decoded RG control accounting invariant failed")
    clock_hand = _U64.unpack_from(mapping, _OFF_CLOCK_HAND)[0]
    if clock_hand >= layout.slot_count:
        raise RuntimeError("Decoded RG control CLOCK hand is out of range")


def _before_fork() -> None:
    _FORK_GUARD.acquire()


def _after_fork_parent() -> None:
    _FORK_GUARD.release()


def _after_fork_child() -> None:
    try:
        for control in tuple(_LIVE_CONTROLS):
            control._close_after_fork()
        _LIVE_CONTROLS.clear()
    finally:
        _FORK_GUARD.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_before_fork,
        after_in_parent=_after_fork_parent,
        after_in_child=_after_fork_child,
    )


__all__ = [
    "CONTROL_FORMAT_VERSION",
    "PAYLOAD_FORMAT_VERSION",
    "DecodedRGState",
    "ParquetRGControl",
    "RGControlHeader",
    "RGControlLayout",
    "RGControlMutation",
    "RGSlotSnapshot",
]
