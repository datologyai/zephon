# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import multiprocessing
import os
import uuid
from pathlib import Path

import pytest

from zephon._internal.io.formats.parquet_cache.control import (
    DecodedRGState,
    ParquetRGControl,
    RGControlLayout,
)

_CATALOG = "sha256:" + "ab" * 32
_CONFIGURATION = "sha256:" + "cd" * 32


def _create_control(
    path: Path, *, slots: int = 3, limit: int = 1_000
) -> tuple[ParquetRGControl, bytes]:
    namespace_uuid = uuid.uuid4().bytes
    control = ParquetRGControl.create(
        path,
        slot_count=slots,
        limit_bytes=limit,
        catalog_fingerprint=_CATALOG,
        configuration_fingerprint=_CONFIGURATION,
        session_uuid=uuid.uuid4().bytes,
        namespace_uuid=namespace_uuid,
    )
    return control, namespace_uuid


def _attached_header(path: str, namespace_uuid: bytes, output) -> None:
    with ParquetRGControl.attach(
        path,
        slot_count=3,
        limit_bytes=1_000,
        catalog_fingerprint=_CATALOG,
        configuration_fingerprint=_CONFIGURATION,
        namespace_uuid=namespace_uuid,
    ) as control:
        output.put((control.header(), control.slot(1)))


def test_layout_is_compact_and_page_aligned() -> None:
    layout = RGControlLayout.for_slots(1_000_000)
    assert layout.file_bytes < 11 * 1024**2
    assert layout.file_bytes % 4096 == 0


def test_create_and_attach_observe_shared_state(tmp_path: Path) -> None:
    path = tmp_path / "control.bin"
    control, namespace_uuid = _create_control(path)
    with control.mutation(1) as mutation:
        mutation.set_slot(state=DecodedRGState.READY, refbit=True, bytes=120)

    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=_attached_header,
        args=(os.fspath(path), namespace_uuid, output),
    )
    process.start()
    process.join(timeout=10)
    try:
        assert process.exitcode == 0
        header, slot = output.get(timeout=1)
        assert header.accounted_bytes == 120
        assert header.inflight_bytes == 0
        assert slot.state is DecodedRGState.READY
    finally:
        control.close()


def test_eviction_activity_and_ghost_bits_are_shared_between_attachments(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control.bin"
    control, namespace_uuid = _create_control(path)
    try:
        control.mark_evicted(2)
        assert control.was_evicted(2)

        with ParquetRGControl.attach(
            path,
            slot_count=3,
            limit_bytes=1_000,
            catalog_fingerprint=_CATALOG,
            configuration_fingerprint=_CONFIGURATION,
            namespace_uuid=namespace_uuid,
        ) as attached:
            assert attached.was_evicted(2)
            attached.clear_evicted(2)

        assert not control.was_evicted(2)
    finally:
        control.close()


def test_thrash_counters_are_shared_and_arm_only_after_eviction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control.bin"
    control, namespace_uuid = _create_control(path)
    try:
        assert control.note_post_fill_publication(reloaded=True) == (0, 0)
        control.arm_thrash_detection()
        assert control.note_post_fill_publication(reloaded=False) == (1, 0)

        with ParquetRGControl.attach(
            path,
            slot_count=3,
            limit_bytes=1_000,
            catalog_fingerprint=_CATALOG,
            configuration_fingerprint=_CONFIGURATION,
            namespace_uuid=namespace_uuid,
        ) as attached:
            assert attached.note_post_fill_publication(reloaded=True) == (2, 1)
            header = control.header()
            assert header.thrash_armed
            assert header.post_fill_publications == 2
            assert header.post_fill_reloads == 1
    finally:
        control.close()


def test_dirty_slot_is_discarded_and_dense_counters_recounted(tmp_path: Path) -> None:
    control, _ = _create_control(tmp_path / "control.bin")
    try:
        with control.mutation(0) as mutation:
            mutation.set_slot(state=DecodedRGState.READY, refbit=True, bytes=30)
        with pytest.raises(RuntimeError, match="simulated death"):
            with control.mutation(2) as mutation:
                mutation.set_slot(
                    state=DecodedRGState.BUILDING,
                    refbit=False,
                    bytes=80,
                )
                raise RuntimeError("simulated death")

        assert control.header().dirty
        assert control.header().dirty_slot == 2
        assert control.discard_dirty_slot_locked(2)
        header = control.header()
        assert not header.dirty
        assert header.accounted_bytes == 30
        assert header.inflight_bytes == 0
        assert control.slot(2).state is DecodedRGState.EMPTY
    finally:
        control.close()


def test_dense_slot_scans_check_mapping_liveness_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control, _ = _create_control(tmp_path / "control.bin")
    try:
        with control.mutation(0) as mutation:
            mutation.set_slot(state=DecodedRGState.READY, refbit=True, bytes=30)
        with pytest.raises(RuntimeError, match="simulated death"):
            with control.mutation(2) as mutation:
                mutation.set_slot(
                    state=DecodedRGState.BUILDING,
                    refbit=False,
                    bytes=80,
                )
                raise RuntimeError("simulated death")

        real_getpid = os.getpid
        getpid_calls = 0

        def counted_getpid() -> int:
            nonlocal getpid_calls
            getpid_calls += 1
            return real_getpid()

        monkeypatch.setattr(os, "getpid", counted_getpid)
        assert control.count_slots(DecodedRGState.READY) == 1
        assert getpid_calls == 1

        getpid_calls = 0
        assert control.discard_dirty_slot_locked(2)
        assert getpid_calls == 1
    finally:
        control.close()


def test_replacement_state_rejects_empty_update(tmp_path: Path) -> None:
    control, _ = _create_control(tmp_path / "control.bin")
    try:
        epoch = control.header().capacity_epoch
        with pytest.raises(ValueError, match="must change at least one field"):
            control.set_replacement_state()
        assert control.header().capacity_epoch == epoch
    finally:
        control.close()


def test_inflight_is_derived_from_slot_transitions(tmp_path: Path) -> None:
    control, _ = _create_control(tmp_path / "control.bin")
    try:
        with control.mutation(1) as mutation:
            mutation.set_slot(
                state=DecodedRGState.BUILDING,
                refbit=False,
                bytes=70,
            )
        assert control.header().accounted_bytes == 70
        assert control.header().inflight_bytes == 70

        with control.mutation(1) as mutation:
            mutation.set_slot(state=DecodedRGState.READY, refbit=True, bytes=70)
        assert control.header().accounted_bytes == 70
        assert control.header().inflight_bytes == 0
    finally:
        control.close()


def test_attach_rejects_configuration_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "control.bin"
    control, namespace_uuid = _create_control(path)
    control.close()
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        ParquetRGControl.attach(
            path,
            slot_count=3,
            limit_bytes=1_000,
            catalog_fingerprint=_CATALOG,
            configuration_fingerprint="sha256:" + "ef" * 32,
            namespace_uuid=namespace_uuid,
        )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_fork_child_closes_inherited_control_mapping(tmp_path: Path) -> None:
    control, _ = _create_control(tmp_path / "control.bin")
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - parent observes the result
        os.close(read_fd)
        try:
            control.header()
        except RuntimeError as exc:
            outcome = str(exc).encode()
        else:
            outcome = b"mapping unexpectedly remained usable"
        os.write(write_fd, outcome)
        os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    try:
        outcome = os.read(read_fd, 1024).decode()
        _, status = os.waitpid(child_pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert outcome == "Decoded RG control mapping was inherited across fork"
        assert control.header().slot_count == 3
    finally:
        os.close(read_fd)
        control.close()
