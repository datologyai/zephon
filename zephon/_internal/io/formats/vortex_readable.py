# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Bridge Zephon storage ranges to Vortex's Python ``ReadBytesAt`` binding."""

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from zephon._internal.io.storage import StorageBackend


class StorageReadAt:
    """A position-independent reader; the backend must allow concurrent ranges.

    Each read returns the backend's buffer to Vortex without a copy. Vortex
    keeps a read-only, contiguous, aligned buffer as it is, and copies any other
    buffer once. Python ``bytes`` and obstore results are read-only.
    """

    def __init__(
        self,
        storage: StorageBackend,
        path: str,
        size: int,
        *,
        retry_attempts: int = 1,
        retry_initial_backoff: float = 0.1,
        retry_max_backoff: float = 2.0,
    ) -> None:
        if size < 0:
            raise ValueError("Vortex source size must be non-negative")
        self._storage = storage
        self._path = path
        self._size = size
        self._retrying = Retrying(
            stop=stop_after_attempt(max(1, retry_attempts)),
            wait=wait_exponential(
                multiplier=retry_initial_backoff, max=retry_max_backoff
            ),
            retry=retry_if_exception_type(OSError),
            reraise=True,
        )

    def size(self) -> int:
        """Return the size already obtained during discovery or resolution."""
        return self._size

    def read_at(self, offset: int, length: int) -> bytes | memoryview:
        """Return the backend's buffer for a range; Vortex reads a short result again."""
        if offset < 0:
            raise ValueError("Vortex read offset must be non-negative")
        length = min(length, max(0, self._size - offset))
        if not length:
            return b""

        # Each call gets its own retry controller.
        data = self._retrying.copy()(
            self._storage.read_range, self._path, offset, length=length
        )

        if memoryview(data).nbytes > length:
            raise ValueError("Storage backend returned more bytes than requested")
        return data
