"""Tests for /dev/shm error classification."""

from __future__ import annotations

import errno

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
