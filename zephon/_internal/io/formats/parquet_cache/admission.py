# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Bounded locking, admission, recovery, and CLOCK transitions for decoded RGs."""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from zephon._internal.io.formats.parquet_cache.control import (
    DecodedRGState,
    ParquetRGControl,
)
from zephon._internal.io.formats.parquet_cache.session import ParquetRGSession
from zephon._internal.io.ofd_lock import OFDLease, OFDLockMode

_CAPACITY_RANGE = 0
_EVICTION_LEADER_RANGE = 1
_FIRST_RG_RANGE = 3


class RGAdmissionOutcome(Enum):
    """Result of a bounded decoded-payload reservation attempt."""

    ADMITTED = "admitted"
    OVERSIZED = "oversized"
    PRESSURE = "pressure"
    CONTENDED = "contended"
    NOT_EMPTY = "not_empty"
    DISABLED = "disabled"


class RGEvictionOutcome(Enum):
    """Result of considering one exact-locked slot during CLOCK scanning."""

    SECOND_CHANCE = "second_chance"
    EVICTING = "evicting"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class RGPublicationResult:
    """Observability outcome of one successfully published payload."""

    reloaded_after_eviction: bool
    node_post_fill_publications: int
    node_post_fill_reloads: int


class ParquetRGAdmission:
    """Decide when a process may read, build, publish, or evict a row group.

    It combines ``ParquetRGSession`` locks with ``ParquetRGControl`` metadata to
    make bounded cross-process state changes. It never decodes or writes row data.
    """

    def __init__(
        self,
        session: ParquetRGSession,
        *,
        discard_paths: Callable[[int], bool],
    ) -> None:
        self._session = session
        self._discard_paths = discard_paths

    def try_exact_exclusive(self, slot: int) -> OFDLease | None:
        self._check_slot(slot)
        return self._session.lock_file.try_acquire(
            start=self._range_for_slot(slot),
            mode=OFDLockMode.EXCLUSIVE,
        )

    def try_ready_shared(self, slot: int) -> OFDLease | None:
        """Acquire a READY reader without touching a global lock."""
        self._check_slot(slot)
        control = self._session.control
        try:
            if not self._is_readable(control, slot):
                return None
        except RuntimeError:
            return None
        lease = self._session.lock_file.try_acquire(
            start=self._range_for_slot(slot),
            mode=OFDLockMode.SHARED,
        )
        if lease is None:
            return None
        try:
            if not self._is_readable(control, slot):
                lease.close()
                return None
            control.touch_ready(slot)
            return lease
        except Exception:
            lease.close()
            return None

    def wait_ready_shared(self, slot: int, *, timeout: float) -> OFDLease | None:
        """Wait only for this exact RG; timeout means direct decode."""
        self._check_slot(slot)
        deadline = time.monotonic() + timeout
        backoff = 0.0005
        while True:
            lease = self.try_ready_shared(slot)
            if lease is not None:
                return lease
            # If the exact range is free but the slot is not READY, there is no
            # publisher to wait for. Briefly taking SH distinguishes that case.
            probe = self._session.lock_file.try_acquire(
                start=self._range_for_slot(slot),
                mode=OFDLockMode.SHARED,
            )
            if probe is not None:
                probe.close()
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(backoff, remaining))
            backoff = min(backoff * 2, 0.01)

    def try_eviction_leader(self) -> OFDLease | None:
        return self._session.lock_file.try_acquire(
            start=_EVICTION_LEADER_RANGE,
            mode=OFDLockMode.EXCLUSIVE,
        )

    def ensure_clean_for_exact(
        self,
        *,
        slot: int,
        exact_lease: OFDLease,
        timeout: float,
    ) -> bool:
        """Repair any dirty slot before an exact owner mutates cache state."""
        self._validate_exact(slot, exact_lease)
        deadline = time.monotonic() + timeout
        capacity = self._capacity_after_recovery(
            exact_slot=slot,
            exact_lease=exact_lease,
            deadline=deadline,
        )
        if capacity is None:
            return False
        with capacity:
            return not self._session.control.header().disabled

    def try_reserve(
        self,
        *,
        slot: int,
        exact_lease: OFDLease,
        payload_bytes: int,
        min_free_bytes: int,
        timeout: float,
    ) -> RGAdmissionOutcome:
        return self._try_reserve(
            slot=slot,
            exact_lease=exact_lease,
            payload_bytes=payload_bytes,
            min_free_bytes=min_free_bytes,
            timeout=timeout,
            leader_lease=None,
        )

    def try_reserve_as_leader(
        self,
        *,
        slot: int,
        exact_lease: OFDLease,
        leader_lease: OFDLease,
        payload_bytes: int,
        min_free_bytes: int,
        timeout: float,
    ) -> RGAdmissionOutcome:
        self._validate_leader(leader_lease)
        return self._try_reserve(
            slot=slot,
            exact_lease=exact_lease,
            payload_bytes=payload_bytes,
            min_free_bytes=min_free_bytes,
            timeout=timeout,
            leader_lease=leader_lease,
        )

    def _try_reserve(
        self,
        *,
        slot: int,
        exact_lease: OFDLease,
        payload_bytes: int,
        min_free_bytes: int,
        timeout: float,
        leader_lease: OFDLease | None,
    ) -> RGAdmissionOutcome:
        self._validate_exact(slot, exact_lease)
        if payload_bytes <= 0:
            raise ValueError("Decoded RG payload bytes must be positive")
        if min_free_bytes < 0:
            raise ValueError("Decoded RG minimum free bytes must be non-negative")
        control = self._session.control
        if payload_bytes > control.limit_bytes:
            return RGAdmissionOutcome.OVERSIZED

        deadline = time.monotonic() + timeout
        for _ in range(2):
            capacity = self._capacity_after_recovery(
                exact_slot=slot,
                exact_lease=exact_lease,
                deadline=deadline,
            )
            if capacity is None:
                return RGAdmissionOutcome.CONTENDED
            with capacity:
                first = control.header()
                if first.disabled:
                    return RGAdmissionOutcome.DISABLED
                if first.eviction_pending and leader_lease is None:
                    return RGAdmissionOutcome.PRESSURE
                if control.slot(slot).state is not DecodedRGState.EMPTY:
                    return RGAdmissionOutcome.NOT_EMPTY
                epoch = first.capacity_epoch
                inflight = first.inflight_bytes
                if first.accounted_bytes + payload_bytes > first.limit_bytes:
                    return RGAdmissionOutcome.PRESSURE

            try:
                available_bytes = shutil.disk_usage(self._session.entries_dir).free
            except OSError:
                return RGAdmissionOutcome.CONTENDED

            capacity = self._capacity_after_recovery(
                exact_slot=slot,
                exact_lease=exact_lease,
                deadline=deadline,
            )
            if capacity is None:
                return RGAdmissionOutcome.CONTENDED
            with capacity:
                current = control.header()
                if current.capacity_epoch != epoch:
                    continue
                if current.disabled:
                    return RGAdmissionOutcome.DISABLED
                if current.eviction_pending and leader_lease is None:
                    return RGAdmissionOutcome.PRESSURE
                if control.slot(slot).state is not DecodedRGState.EMPTY:
                    return RGAdmissionOutcome.NOT_EMPTY
                logical_fit = (
                    current.accounted_bytes + payload_bytes <= current.limit_bytes
                )
                free_fit = available_bytes - inflight - payload_bytes >= min_free_bytes
                if not logical_fit or not free_fit:
                    return RGAdmissionOutcome.PRESSURE
                with control.mutation(slot) as mutation:
                    mutation.set_slot(
                        state=DecodedRGState.BUILDING,
                        refbit=False,
                        bytes=payload_bytes,
                    )
                if leader_lease is not None:
                    control.set_replacement_state(eviction_pending=False)
                return RGAdmissionOutcome.ADMITTED
        return RGAdmissionOutcome.CONTENDED

    def publish_ready(
        self,
        *,
        slot: int,
        exact_lease: OFDLease,
        timeout: float,
    ) -> RGPublicationResult | None:
        """Publish one reservation and identify cross-process eviction reloads."""
        self._validate_exact(slot, exact_lease)
        capacity = self._capacity_after_recovery(
            exact_slot=slot,
            exact_lease=exact_lease,
            deadline=time.monotonic() + timeout,
        )
        if capacity is None:
            return None
        with capacity:
            control = self._session.control
            snapshot = control.slot(slot)
            if (
                control.header().disabled
                or snapshot.state is not DecodedRGState.BUILDING
            ):
                return None
            reloaded = control.was_evicted(slot)
            with control.mutation(slot) as mutation:
                mutation.set_slot(
                    state=DecodedRGState.READY,
                    refbit=True,
                    bytes=snapshot.bytes,
                )
            if reloaded:
                control.clear_evicted(slot)
            publications, reloads = control.note_post_fill_publication(
                reloaded=reloaded
            )
            return RGPublicationResult(
                reloaded_after_eviction=reloaded,
                node_post_fill_publications=publications,
                node_post_fill_reloads=reloads,
            )

    def release_reservation(
        self,
        *,
        slot: int,
        exact_lease: OFDLease,
        timeout: float,
    ) -> bool:
        return self._transition(
            slot=slot,
            exact_lease=exact_lease,
            expected=DecodedRGState.BUILDING,
            replacement=DecodedRGState.EMPTY,
            refbit=False,
            timeout=timeout,
        )

    def finish_eviction(
        self,
        *,
        slot: int,
        exact_lease: OFDLease,
        timeout: float,
    ) -> bool:
        self._validate_exact(slot, exact_lease)
        capacity = self._capacity_after_recovery(
            exact_slot=slot,
            exact_lease=exact_lease,
            deadline=time.monotonic() + timeout,
        )
        if capacity is None:
            return False
        with capacity:
            control = self._session.control
            snapshot = control.slot(slot)
            if (
                control.header().disabled
                or snapshot.state is not DecodedRGState.EVICTING
            ):
                return False
            with control.mutation(slot) as mutation:
                mutation.set_slot(
                    state=DecodedRGState.EMPTY,
                    refbit=False,
                    bytes=0,
                )
            # Ghost/counter state is advisory, but updating it under the capacity
            # lease already held here makes the node-wide signal race-free.
            control.mark_evicted(slot)
            control.arm_thrash_detection()
            return True

    def restore_evicting_ready(
        self,
        *,
        slot: int,
        exact_lease: OFDLease,
        timeout: float,
    ) -> bool:
        return self._transition(
            slot=slot,
            exact_lease=exact_lease,
            expected=DecodedRGState.EVICTING,
            replacement=DecodedRGState.READY,
            refbit=True,
            timeout=timeout,
        )

    def _transition(
        self,
        *,
        slot: int,
        exact_lease: OFDLease,
        expected: DecodedRGState,
        replacement: DecodedRGState,
        refbit: bool,
        timeout: float,
    ) -> bool:
        self._validate_exact(slot, exact_lease)
        capacity = self._capacity_after_recovery(
            exact_slot=slot,
            exact_lease=exact_lease,
            deadline=time.monotonic() + timeout,
        )
        if capacity is None:
            return False
        with capacity:
            control = self._session.control
            snapshot = control.slot(slot)
            if control.header().disabled or snapshot.state is not expected:
                return False
            payload_bytes = 0 if replacement is DecodedRGState.EMPTY else snapshot.bytes
            with control.mutation(slot) as mutation:
                mutation.set_slot(
                    state=replacement,
                    refbit=refbit,
                    bytes=payload_bytes,
                )
            return True

    def set_eviction_pending(
        self,
        *,
        leader_lease: OFDLease,
        pending: bool,
        timeout: float,
    ) -> bool:
        """Set the node-wide eviction signal while holding the leader lease.

        A leader that encounters bounded metadata contention may leave the flag
        set. This is safe: the next leader may proceed and clears the stale flag.
        """
        self._validate_leader(leader_lease)
        capacity = self._capacity_after_recovery(
            exact_slot=None,
            exact_lease=None,
            deadline=time.monotonic() + timeout,
        )
        if capacity is None:
            return False
        with capacity:
            if self._session.control.header().disabled:
                return False
            self._session.control.set_replacement_state(eviction_pending=pending)
            return True

    def claim_clock_chunk(
        self,
        *,
        leader_lease: OFDLease,
        max_slots: int,
        timeout: float,
    ) -> tuple[int, ...] | None:
        self._validate_leader(leader_lease)
        if max_slots <= 0:
            raise ValueError("CLOCK chunk size must be positive")
        capacity = self._capacity_after_recovery(
            exact_slot=None,
            exact_lease=None,
            deadline=time.monotonic() + timeout,
        )
        if capacity is None:
            return None
        with capacity:
            control = self._session.control
            header = control.header()
            if header.disabled:
                return None
            count = min(max_slots, control.slot_count)
            slots = tuple(
                (header.clock_hand + offset) % control.slot_count
                for offset in range(count)
            )
            control.set_replacement_state(
                clock_hand=(header.clock_hand + count) % control.slot_count,
                eviction_pending=True,
            )
            return slots

    def try_mark_evicting(
        self,
        *,
        slot: int,
        exact_lease: OFDLease,
        timeout: float,
    ) -> RGEvictionOutcome:
        self._validate_exact(slot, exact_lease)
        control = self._session.control
        snapshot = control.slot(slot)
        if snapshot.state is not DecodedRGState.READY:
            return RGEvictionOutcome.SKIPPED
        if snapshot.refbit:
            control.clear_ready_refbit(slot)
            return RGEvictionOutcome.SECOND_CHANCE
        capacity = self._capacity_after_recovery(
            exact_slot=slot,
            exact_lease=exact_lease,
            deadline=time.monotonic() + timeout,
        )
        if capacity is None:
            return RGEvictionOutcome.SKIPPED
        with capacity:
            snapshot = control.slot(slot)
            if control.header().disabled or snapshot.state is not DecodedRGState.READY:
                return RGEvictionOutcome.SKIPPED
            if snapshot.refbit:
                control.clear_ready_refbit(slot)
                return RGEvictionOutcome.SECOND_CHANCE
            with control.mutation(slot) as mutation:
                mutation.set_slot(
                    state=DecodedRGState.EVICTING,
                    refbit=False,
                    bytes=snapshot.bytes,
                )
            return RGEvictionOutcome.EVICTING

    def _capacity_after_recovery(
        self,
        *,
        exact_slot: int | None,
        exact_lease: OFDLease | None,
        deadline: float,
    ) -> OFDLease | None:
        """Acquire capacity, repairing one dirty slot without filesystem IO under it."""
        while True:
            capacity = self._acquire_until(
                start=_CAPACITY_RANGE,
                mode=OFDLockMode.EXCLUSIVE,
                deadline=deadline,
            )
            if capacity is None:
                return None
            header = self._session.control.header()
            if header.disabled:
                return capacity
            if not header.dirty:
                return capacity
            target = header.dirty_slot
            if (
                target is None
                or target < 0
                or target >= self._session.control.slot_count
            ):
                self._session.control.disable_locked()
                return capacity
            capacity.close()

            owns_target = target == exact_slot and exact_lease is not None
            repair_exact = (
                exact_lease
                if owns_target
                else self._acquire_until(
                    start=self._range_for_slot(target),
                    mode=OFDLockMode.EXCLUSIVE,
                    deadline=deadline,
                )
            )
            if repair_exact is None:
                return None
            try:
                if not self._discard_paths(target):
                    return None
                capacity = self._acquire_until(
                    start=_CAPACITY_RANGE,
                    mode=OFDLockMode.EXCLUSIVE,
                    deadline=deadline,
                )
                if capacity is None:
                    return None
                latest = self._session.control.header()
                if latest.dirty and latest.dirty_slot == target:
                    self._session.control.discard_dirty_slot_locked(target)
                    return capacity
                capacity.close()
            finally:
                if not owns_target:
                    repair_exact.close()

    def _acquire_until(
        self,
        *,
        start: int,
        mode: OFDLockMode,
        deadline: float,
    ) -> OFDLease | None:
        remaining = deadline - time.monotonic()
        if remaining < 0:
            return None
        return self._session.lock_file.acquire(
            start=start,
            mode=mode,
            timeout=remaining,
        )

    def _validate_exact(self, slot: int, lease: OFDLease) -> None:
        self._check_slot(slot)
        if (
            lease.closed
            or lease.mode is not OFDLockMode.EXCLUSIVE
            or lease.start != self._range_for_slot(slot)
            or lease.length != 1
        ):
            raise ValueError(f"Expected live exact exclusive lease for RG slot {slot}")

    @staticmethod
    def _validate_leader(lease: OFDLease) -> None:
        if (
            lease.closed
            or lease.mode is not OFDLockMode.EXCLUSIVE
            or lease.start != _EVICTION_LEADER_RANGE
            or lease.length != 1
        ):
            raise ValueError("Expected live decoded RG eviction-leader lease")

    def _check_slot(self, slot: int) -> None:
        if slot < 0 or slot >= self._session.control.slot_count:
            raise IndexError(slot)

    @staticmethod
    def _is_readable(control: ParquetRGControl, slot: int) -> bool:
        header = control.header()
        if header.disabled or (
            header.dirty and (header.dirty_slot is None or header.dirty_slot == slot)
        ):
            return False
        return control.slot(slot).state is DecodedRGState.READY

    @staticmethod
    def _range_for_slot(slot: int) -> int:
        return _FIRST_RG_RANGE + slot


__all__ = [
    "ParquetRGAdmission",
    "RGAdmissionOutcome",
    "RGEvictionOutcome",
    "RGPublicationResult",
]
