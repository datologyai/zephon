# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from zephon._internal.io import ofd_lock as ofd_lock_module
from zephon._internal.io.ofd_lock import (
    OFDLockFile,
    OFDLockUnavailable,
    _select_backend,
    ofd_backend_info,
    probe_ofd_support,
)


def test_backend_has_expected_native_layout() -> None:
    try:
        info = ofd_backend_info()
    except OFDLockUnavailable as exc:
        pytest.skip(str(exc))

    if info.abi_id == "darwin-64":
        assert info.struct_size == 24
    elif info.abi_id == "linux-64":
        assert info.struct_size == 32
    else:  # pragma: no cover - selecting an unknown ABI is itself a failure
        pytest.fail(f"Unexpected supported ABI: {info.abi_id}")


def test_backend_rejects_unknown_or_32_bit_abi() -> None:
    try:
        ofd_backend_info()
    except OFDLockUnavailable as exc:
        pytest.skip(str(exc))

    with pytest.raises(OFDLockUnavailable, match="Unsupported OFD ABI"):
        _select_backend(
            sys_platform="freebsd",
            architecture="x86_64",
            pointer_size=8,
        )
    with pytest.raises(OFDLockUnavailable, match="Unsupported 32-bit"):
        _select_backend(
            sys_platform="linux",
            architecture="x86_64",
            pointer_size=4,
        )


def test_probe_reports_backend_unavailability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason = "OFD locks are unavailable"

    def unavailable_backend() -> None:
        raise OFDLockUnavailable(reason)

    monkeypatch.setattr(ofd_lock_module, "_select_backend", unavailable_backend)

    result = probe_ofd_support(tmp_path)

    assert not result.supported
    assert result.backend is None
    assert result.reason == reason
    assert not list(tmp_path.iterdir())


def test_probe_cleanup_error_is_reported_instead_of_raised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        ofd_backend_info()
    except OFDLockUnavailable as exc:
        pytest.skip(str(exc))

    real_unlink = ofd_lock_module.os.unlink
    unlink_calls = 0

    def denied_unlink(path: str | bytes) -> None:
        nonlocal unlink_calls
        unlink_calls += 1
        raise PermissionError("probe cleanup denied")

    monkeypatch.setattr(ofd_lock_module.os, "unlink", denied_unlink)
    result = probe_ofd_support(tmp_path)
    monkeypatch.setattr(ofd_lock_module.os, "unlink", real_unlink)

    assert not result.supported
    assert result.reason is not None
    assert result.reason.startswith("PermissionError: probe cleanup denied")
    assert unlink_calls == 2
    for leftover in tmp_path.iterdir():
        leftover.unlink()


def test_anchor_fd_exists_when_backend_selection_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_backend() -> None:
        raise OFDLockUnavailable("backend unavailable")

    monkeypatch.setattr(ofd_lock_module, "_select_backend", unavailable_backend)
    lock_file = OFDLockFile.__new__(OFDLockFile)

    with pytest.raises(OFDLockUnavailable, match="backend unavailable"):
        lock_file.__init__(tmp_path / "locks.ofd")

    assert lock_file.closed
