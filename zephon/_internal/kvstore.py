"""Key-value stores for checkpoint aggregation.

``RuntimeOptions.aggregate_store`` names a store by URI, and the engine builds
it in its own process because a live client can't be handed to an MTP
subprocess. ``torch://`` connects to torchrun's c10d ``TCPStore`` at
``MASTER_ADDR``/``MASTER_PORT``; ``torch://host:port`` names one explicitly.
"""

from __future__ import annotations

import math
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import timedelta
from typing import Any
from urllib.parse import urlparse


class KVStore(ABC):
    """Byte-valued key-value store shared by every checkpoint participant.

    Writes from one client become visible to other clients in the order that
    client made them, but not necessarily by the time ``set`` returns; readers
    wait for keys rather than assuming a write is already visible.
    """

    poll_interval_s: float = 0.05
    """Delay between checks in the default polling ``wait``."""

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        """Return the value for ``key``, or None if it does not exist."""

    @abstractmethod
    def set(self, key: str, value: bytes) -> None:
        """Write ``value`` to ``key``, replacing any existing value."""

    @abstractmethod
    def set_if_absent(self, key: str, value: bytes) -> bytes:
        """Write ``value`` only if ``key`` does not exist; return the stored value."""

    @abstractmethod
    def append(self, key: str, value: bytes) -> None:
        """Atomically append ``value`` to ``key``, creating it if absent."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key`` if it exists."""

    def get_many(self, keys: list[str]) -> list[bytes | None]:
        """Read several keys; override to batch them."""
        return [self.get(key) for key in keys]

    def wait(self, keys: list[str], timeout: float) -> bool:
        """Block until every key exists; return False on timeout.

        Checks once more when the timeout has run out, so keys that are
        already present are never reported missing.
        """
        deadline = time.monotonic() + timeout
        pending = list(keys)
        while True:
            pending = [key for key in pending if not self.exists(key)]
            if not pending:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(self.poll_interval_s, remaining))


KVStoreFactory = Callable[[str, float], KVStore]
"""Builds a store from its full URI (for example ``torch://host:29500``).

The second argument is the aggregation timeout in seconds, which bounds the
store's blocking operations, including connection setup.
"""

_REGISTRY: dict[str, KVStoreFactory] = {}


def register_kv_store(scheme: str, factory: KVStoreFactory) -> None:
    """Register ``factory`` for ``scheme://`` URIs.

    Registration is per process, so an MTP or spawned worker process must also
    import the module that registers the store.
    """
    _REGISTRY[scheme.lower()] = factory


def open_kv_store(uri: str, *, timeout_s: float) -> KVStore:
    """Build the store that ``uri`` names; raise ValueError for unknown schemes."""
    scheme, sep, _ = uri.partition("://")
    factory = _REGISTRY.get(scheme.lower()) if sep else None
    if factory is None:
        raise ValueError(
            f"Unknown aggregate_store scheme in {uri!r}; "
            f"registered schemes: {sorted(_REGISTRY)}"
        )
    return factory(uri, timeout_s)


class TorchKVStore(KVStore):
    """Client for a c10d ``TCPStore``, such as torchrun's.

    Connects on first use and again after a fork, so a forked DataLoader worker
    never shares its parent's socket.
    """

    poll_interval_s = 0.01

    def __init__(self, host: str, port: int, *, timeout_s: float) -> None:
        self.host = host
        self.port = port
        self._timeout_s = timeout_s
        self._store: Any = None  # TCPStore; torch is an optional import
        self._pid: int | None = None

    @classmethod
    def from_uri(cls, uri: str, timeout_s: float) -> TorchKVStore:
        parsed = urlparse(uri)
        host = parsed.hostname or os.environ.get("MASTER_ADDR")
        port = parsed.port or os.environ.get("MASTER_PORT")
        if not host or not port:
            raise ValueError(
                f"{uri!r} names no host and port, and MASTER_ADDR/MASTER_PORT "
                "are not set; use torch://host:port or launch with torchrun"
            )
        return cls(host, int(port), timeout_s=timeout_s)

    def _client(self) -> Any:
        pid = os.getpid()
        if self._store is None or self._pid != pid:
            import torch.distributed as dist

            self._store = dist.TCPStore(
                self.host,
                self.port,
                is_master=False,
                timeout=timedelta(seconds=self._timeout_s),
                wait_for_workers=False,
            )
            self._pid = pid
        return self._store

    def get(self, key: str) -> bytes | None:
        store = self._client()
        # TCPStore.get blocks until the key exists, so check first.
        if not store.check([key]):
            return None
        return store.get(key)

    def get_many(self, keys: list[str]) -> list[bytes | None]:
        # multi_get blocks on missing keys, so batch only when all exist.
        store = self._client()
        if keys and store.check(keys):
            return list(store.multi_get(keys))
        return [self.get(key) for key in keys]

    def set(self, key: str, value: bytes) -> None:
        self._client().set(key, value)

    def set_if_absent(self, key: str, value: bytes) -> bytes:
        # With an empty expected value, compare_set writes only if the key is
        # absent, and returns the stored value either way.
        return self._client().compare_set(key, "", value)

    def append(self, key: str, value: bytes) -> None:
        self._client().append(key, value)

    def exists(self, key: str) -> bool:
        return self._client().check([key])

    def delete(self, key: str) -> None:
        self._client().delete_key(key)

    def wait(self, keys: list[str], timeout: float) -> bool:
        store = self._client()
        # TCPStore waits in whole milliseconds; with no budget left, only
        # check whether the keys are already there.
        timeout_ms = math.ceil(timeout * 1000)
        if timeout_ms <= 0:
            return store.check(keys)
        import torch.distributed as dist

        try:
            store.wait(keys, timedelta(milliseconds=timeout_ms))
        except dist.DistStoreError:
            return store.check(keys)
        return True


register_kv_store("torch", TorchKVStore.from_uri)
