"""Tests for /dev/shm error classification."""

from __future__ import annotations

import errno
from types import SimpleNamespace

import pytest

from zephon._internal.utils import shm
from zephon._internal.utils.shm import is_shm_error


def test_oserror_enospc() -> None:
    assert is_shm_error(OSError(errno.ENOSPC, "No space left on device"))


def test_torch_write_path_message() -> None:
    assert is_shm_error(
        RuntimeError(
            "unable to write to file </torch_1111090_121816052_484>: "
            "No space left on device (28)"
        )
    )


def test_torch_fallocate_path_message() -> None:
    # posix_fallocate reports via return value, leaving errno at 0 ("Success").
    assert is_shm_error(
        RuntimeError(
            "unable to allocate shared memory(shm) for file "
            "</torch_21651_171144194_0>: Success (0)"
        )
    )


def test_chained_enospc() -> None:
    cause = OSError(errno.ENOSPC, "No space left on device")
    wrapped = RuntimeError("pickling failed")
    wrapped.__cause__ = cause
    assert is_shm_error(wrapped)


def test_unrelated_errors_not_matched() -> None:
    assert not is_shm_error(RuntimeError("CUDA out of memory"))
    assert not is_shm_error(OSError(errno.EMFILE, "Too many open files"))


def test_backoff_can_be_cancelled_before_shared_memory_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now, sleeps = [0.0], []
    cancelled = RuntimeError("queue aborted")

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    def check_cancelled() -> None:
        if len(sleeps) == 2:
            raise cancelled

    monkeypatch.setattr(shm, "shm_has_free_space", lambda threshold: False)
    monkeypatch.setattr(shm, "shm_usage_str", lambda: "full")
    monkeypatch.setattr(shm._random, "uniform", lambda *args: 0)
    monkeypatch.setattr(
        shm, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep)
    )
    with pytest.raises(RuntimeError) as caught:
        shm.wait_for_shm_space("cancelled queue", check_cancelled=check_cancelled)
    assert caught.value is cancelled
    assert sleeps == [0.1, 0.1]


def test_backoff_without_cancellation_retains_retry_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = iter([False, False, True])
    sleeps = []
    monkeypatch.setattr(shm, "shm_has_free_space", lambda threshold: next(space))
    monkeypatch.setattr(shm, "shm_usage_str", lambda: "full")
    monkeypatch.setattr(shm._random, "uniform", lambda *args: 0)
    monkeypatch.setattr(shm, "time", SimpleNamespace(sleep=sleeps.append))
    assert shm.wait_for_shm_space("recovering queue") == 2
    assert sleeps == [shm._SHM_RETRY_BASE_BACKOFF, 2 * shm._SHM_RETRY_BASE_BACKOFF]
