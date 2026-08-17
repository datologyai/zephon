# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import multiprocessing
import threading
import time
from pathlib import Path

import pytest

from zephon._internal.io.formats.parquet_cache.admission import (
    ParquetRGAdmission,
    RGAdmissionOutcome,
    RGEvictionOutcome,
)
from zephon._internal.io.formats.parquet_cache.control import DecodedRGState
from zephon._internal.io.formats.parquet_cache.session import ParquetRGSession
from zephon._internal.io.ofd_lock import OFDLockMode

_CATALOG_FINGERPRINT = "sha256:" + "78" * 32
_CONFIGURATION_FINGERPRINT = "sha256:" + "9a" * 32


def _discard_paths(_slot: int) -> bool:
    return True


def _admission(session: ParquetRGSession) -> ParquetRGAdmission:
    return ParquetRGAdmission(session, discard_paths=_discard_paths)


def _session(root: Path, *, slots: int = 32, limit: int = 1_000):
    return ParquetRGSession(
        root,
        slot_count=slots,
        limit_bytes=limit,
        catalog_fingerprint=_CATALOG_FINGERPRINT,
        configuration_fingerprint=_CONFIGURATION_FINGERPRINT,
    )


def _hold_unique_exact_range(
    root: str,
    slot: int,
    acquired: multiprocessing.Queue,
    release: multiprocessing.synchronize.Event,
) -> None:
    with _session(Path(root)) as session:
        admission = _admission(session)
        lease = admission.try_exact_exclusive(slot)
        acquired.put(lease is not None)
        release.wait(timeout=10)
        if lease is not None:
            lease.close()


def test_different_rg_build_elections_do_not_conflict(tmp_path: Path) -> None:
    with _session(tmp_path / "decoded") as session:
        admission = _admission(session)
        leases = [admission.try_exact_exclusive(slot) for slot in range(32)]
        try:
            assert all(lease is not None for lease in leases)
            assert admission.try_exact_exclusive(0) is None
        finally:
            for lease in leases:
                if lease is not None:
                    lease.close()


def test_spawned_process_can_hold_unrelated_build_election(tmp_path: Path) -> None:
    # Range conflicts are pairwise; the preceding test covers all slot numbers.
    root = tmp_path / "decoded"
    parent = _session(root)
    parent_admission = _admission(parent)
    parent_lease = parent_admission.try_exact_exclusive(0)
    assert parent_lease is not None

    context = multiprocessing.get_context("spawn")
    acquired = context.Queue()
    release = context.Event()
    process = context.Process(
        target=_hold_unique_exact_range,
        args=(str(root), 1, acquired, release),
    )
    process.start()
    try:
        assert acquired.get(timeout=15)
        assert parent_admission.try_exact_exclusive(1) is None
        third = parent_admission.try_exact_exclusive(2)
        assert third is not None
        third.close()
    finally:
        release.set()
        process.join(timeout=15)
        parent_lease.close()
        parent.close()
    assert process.exitcode == 0


def test_reserve_publish_and_hit_never_needs_capacity_on_ready_path(
    tmp_path: Path,
) -> None:
    with _session(tmp_path / "decoded") as session:
        admission = _admission(session)
        exact = admission.try_exact_exclusive(3)
        assert exact is not None
        with exact:
            assert (
                admission.try_reserve(
                    slot=3,
                    exact_lease=exact,
                    payload_bytes=100,
                    min_free_bytes=0,
                    timeout=0.1,
                )
                is RGAdmissionOutcome.ADMITTED
            )
            assert admission.publish_ready(slot=3, exact_lease=exact, timeout=0.1)

        capacity = session.lock_file.try_acquire(
            start=0,
            mode=OFDLockMode.EXCLUSIVE,
        )
        assert capacity is not None
        with capacity:
            hit = admission.try_ready_shared(3)
            assert hit is not None
            hit.close()
        assert session.control.slot(3).state is DecodedRGState.READY


def test_ready_hit_bypasses_dirty_same_slot_until_exact_repair(
    tmp_path: Path,
) -> None:
    with _session(tmp_path / "decoded") as session:
        admission = _admission(session)
        exact = admission.try_exact_exclusive(3)
        assert exact is not None
        with exact:
            assert (
                admission.try_reserve(
                    slot=3,
                    exact_lease=exact,
                    payload_bytes=100,
                    min_free_bytes=0,
                    timeout=0.1,
                )
                is RGAdmissionOutcome.ADMITTED
            )
            assert admission.publish_ready(slot=3, exact_lease=exact, timeout=0.1)

        with pytest.raises(RuntimeError, match="simulated death"):
            with session.control.mutation(3) as mutation:
                mutation.set_slot(
                    state=DecodedRGState.READY,
                    refbit=False,
                    bytes=100,
                )
                raise RuntimeError("simulated death")

        assert admission.try_ready_shared(3) is None
        exact = admission.try_exact_exclusive(3)
        assert exact is not None
        with exact:
            assert admission.ensure_clean_for_exact(
                slot=3,
                exact_lease=exact,
                timeout=0.1,
            )
        assert session.control.slot(3).state is DecodedRGState.EMPTY


def test_unrelated_ready_hit_proceeds_while_another_slot_is_dirty(
    tmp_path: Path,
) -> None:
    with _session(tmp_path / "decoded") as session:
        admission = _admission(session)
        for slot in (3, 4):
            exact = admission.try_exact_exclusive(slot)
            assert exact is not None
            with exact:
                assert (
                    admission.try_reserve(
                        slot=slot,
                        exact_lease=exact,
                        payload_bytes=100,
                        min_free_bytes=0,
                        timeout=0.1,
                    )
                    is RGAdmissionOutcome.ADMITTED
                )
                assert admission.publish_ready(
                    slot=slot,
                    exact_lease=exact,
                    timeout=0.1,
                )

        with pytest.raises(RuntimeError, match="simulated death"):
            with session.control.mutation(3) as mutation:
                mutation.set_slot(
                    state=DecodedRGState.READY,
                    refbit=False,
                    bytes=100,
                )
                raise RuntimeError("simulated death")

        unrelated = admission.try_ready_shared(4)
        assert unrelated is not None
        unrelated.close()


def test_global_quota_and_free_space_floor_are_fail_open_pressure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _session(tmp_path / "decoded", limit=1_000) as session:
        admission = _admission(session)
        first = admission.try_exact_exclusive(0)
        assert first is not None
        with first:
            assert (
                admission.try_reserve(
                    slot=0,
                    exact_lease=first,
                    payload_bytes=700,
                    min_free_bytes=0,
                    timeout=0.1,
                )
                is RGAdmissionOutcome.ADMITTED
            )
            assert admission.publish_ready(slot=0, exact_lease=first, timeout=0.1)

        second = admission.try_exact_exclusive(1)
        assert second is not None
        with second:
            with monkeypatch.context() as scoped:
                scoped.setattr(
                    "zephon._internal.io.formats.parquet_cache.admission.shutil.disk_usage",
                    lambda _path: pytest.fail(
                        "logical pressure should not query filesystem capacity"
                    ),
                )
                assert (
                    admission.try_reserve(
                        slot=1,
                        exact_lease=second,
                        payload_bytes=400,
                        min_free_bytes=0,
                        timeout=0.1,
                    )
                    is RGAdmissionOutcome.PRESSURE
                )
            assert (
                admission.try_reserve(
                    slot=1,
                    exact_lease=second,
                    payload_bytes=1,
                    min_free_bytes=2**63,
                    timeout=0.1,
                )
                is RGAdmissionOutcome.PRESSURE
            )


def test_pending_blocks_headroom_and_capacity_wait_is_bounded(tmp_path: Path) -> None:
    with _session(tmp_path / "decoded") as session:
        admission = _admission(session)
        leader = admission.try_eviction_leader()
        assert leader is not None
        with leader:
            assert admission.set_eviction_pending(
                leader_lease=leader,
                pending=True,
                timeout=0.1,
            )
        exact = admission.try_exact_exclusive(0)
        assert exact is not None
        with exact:
            assert (
                admission.try_reserve(
                    slot=0,
                    exact_lease=exact,
                    payload_bytes=1,
                    min_free_bytes=0,
                    timeout=0.1,
                )
                is RGAdmissionOutcome.PRESSURE
            )

        leader = admission.try_eviction_leader()
        assert leader is not None
        with leader:
            assert admission.set_eviction_pending(
                leader_lease=leader,
                pending=False,
                timeout=0.1,
            )

        capacity = session.lock_file.try_acquire(
            start=0,
            mode=OFDLockMode.EXCLUSIVE,
        )
        exact = admission.try_exact_exclusive(0)
        assert capacity is not None and exact is not None
        with capacity, exact:
            started = time.monotonic()
            outcome = admission.try_reserve(
                slot=0,
                exact_lease=exact,
                payload_bytes=1,
                min_free_bytes=0,
                timeout=0.02,
            )
            elapsed = time.monotonic() - started
        assert outcome is RGAdmissionOutcome.CONTENDED
        assert elapsed < 0.1


def test_new_leader_clears_pending_left_by_dead_leader(tmp_path: Path) -> None:
    with _session(tmp_path / "decoded") as session:
        admission = _admission(session)
        old_leader = admission.try_eviction_leader()
        assert old_leader is not None
        assert admission.set_eviction_pending(
            leader_lease=old_leader,
            pending=True,
            timeout=0.1,
        )
        old_leader.close()

        new_leader = admission.try_eviction_leader()
        assert new_leader is not None
        exact = admission.try_exact_exclusive(0)
        assert exact is not None
        with new_leader:
            with exact:
                assert (
                    admission.try_reserve_as_leader(
                        slot=0,
                        exact_lease=exact,
                        leader_lease=new_leader,
                        payload_bytes=1,
                        min_free_bytes=0,
                        timeout=0.1,
                    )
                    is RGAdmissionOutcome.ADMITTED
                )
                assert not session.control.header().eviction_pending


def test_same_rg_wait_observes_exact_builder_without_builder_train(
    tmp_path: Path,
) -> None:
    with _session(tmp_path / "decoded") as session:
        admission = _admission(session)
        builder = admission.try_exact_exclusive(2)
        assert builder is not None
        result: list[tuple[object, float]] = []

        def wait() -> None:
            started = time.monotonic()
            result.append(
                (
                    admission.wait_ready_shared(2, timeout=0.2),
                    time.monotonic() - started,
                )
            )

        waiter = threading.Thread(target=wait)
        waiter.start()
        time.sleep(0.03)
        builder.close()
        waiter.join(timeout=1)

        assert result and result[0][0] is None
        assert result[0][1] >= 0.02
        assert session.control.slot(2).state is DecodedRGState.EMPTY


def test_global_clock_second_chance_evicts_and_reserves_for_leader(
    tmp_path: Path,
) -> None:
    with _session(tmp_path / "decoded", slots=4, limit=1_000) as session:
        admission = _admission(session)
        first = admission.try_exact_exclusive(0)
        assert first is not None
        with first:
            assert (
                admission.try_reserve(
                    slot=0,
                    exact_lease=first,
                    payload_bytes=700,
                    min_free_bytes=0,
                    timeout=0.1,
                )
                is RGAdmissionOutcome.ADMITTED
            )
            assert admission.publish_ready(slot=0, exact_lease=first, timeout=0.1)

        leader = admission.try_eviction_leader()
        assert leader is not None
        with leader:
            chunk = admission.claim_clock_chunk(
                leader_lease=leader,
                max_slots=4,
                timeout=0.1,
            )
            assert chunk == (0, 1, 2, 3)
            assert session.control.header().eviction_pending

            victim = admission.try_exact_exclusive(0)
            assert victim is not None
            with victim:
                assert (
                    admission.try_mark_evicting(
                        slot=0,
                        exact_lease=victim,
                        timeout=0.1,
                    )
                    is RGEvictionOutcome.SECOND_CHANCE
                )
                assert (
                    admission.try_mark_evicting(
                        slot=0,
                        exact_lease=victim,
                        timeout=0.1,
                    )
                    is RGEvictionOutcome.EVICTING
                )
                assert session.control.header().accounted_bytes == 700
                assert admission.finish_eviction(
                    slot=0,
                    exact_lease=victim,
                    timeout=0.1,
                )

            replacement = admission.try_exact_exclusive(1)
            assert replacement is not None
            with replacement:
                assert (
                    admission.try_reserve_as_leader(
                        slot=1,
                        exact_lease=replacement,
                        leader_lease=leader,
                        payload_bytes=600,
                        min_free_bytes=0,
                        timeout=0.1,
                    )
                    is RGAdmissionOutcome.ADMITTED
                )
                assert not session.control.header().eviction_pending
                assert session.control.header().accounted_bytes == 600
                assert admission.release_reservation(
                    slot=1,
                    exact_lease=replacement,
                    timeout=0.1,
                )
