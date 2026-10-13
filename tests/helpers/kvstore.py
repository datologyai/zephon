"""In-memory key-value store for checkpoint coordination tests."""

import threading

from zephon._internal.kvstore import KVStore


class InMemoryKVStore(KVStore):
    """Thread-safe fake; the participants in a test share one instance.

    Matches ``TorchKVStore``'s contract, including ``set_if_absent`` returning
    the stored value, but not its network timing.
    """

    poll_interval_s = 0.001

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            return self.data.get(key)

    def set(self, key: str, value: bytes) -> None:
        with self._lock:
            self.data[key] = value

    def set_if_absent(self, key: str, value: bytes) -> bytes:
        with self._lock:
            return self.data.setdefault(key, value)

    def append(self, key: str, value: bytes) -> None:
        with self._lock:
            self.data[key] = self.data.get(key, b"") + value

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self.data

    def delete(self, key: str) -> None:
        with self._lock:
            self.data.pop(key, None)
