"""Key-value stores for checkpoint aggregation.

``RuntimeOptions.aggregate_store`` names a store by URI, and the engine builds
it in its own process because a live client can't be handed to an MTP
subprocess. ``torch://`` connects to torchrun's c10d ``TCPStore`` at
``MASTER_ADDR``/``MASTER_PORT``; ``torch://host:port`` names one explicitly.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
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

    max_value_bytes: int | None = None
    """Largest value one write may carry, or None for no limit.

    :func:`stage_value` splits larger values into chunks of this size.
    """

    def check_ready(self) -> None:
        """Raise if the store can't be reached; called before training starts."""
        self.exists("zephon/ready")

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
    """Register ``factory`` for ``scheme://`` URIs; each scheme registers once.

    Registration is per process, so an MTP or spawned worker process must also
    import the module that registers the store.
    """
    scheme = scheme.lower()
    if scheme in _REGISTRY:
        raise ValueError(f"aggregate_store scheme {scheme!r} is already registered")
    _REGISTRY[scheme] = factory


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
    never shares its parent's socket. A failed operation discards the client,
    so the next checkpoint reconnects; the failed operation is not retried.
    """

    poll_interval_s = 0.01
    # The libuv server drops the connection on any value above 8 MiB it
    # receives; replies, including multi_get, have no such limit.
    max_value_bytes = 4 * 1024 * 1024

    def __init__(self, host: str, port: int, *, timeout_s: float) -> None:
        self.host = host
        self.port = port
        self._timeout_s = timeout_s
        self._store: Any = None  # TCPStore; torch is an optional import
        self._pid: int | None = None

    @classmethod
    def from_uri(cls, uri: str, timeout_s: float) -> TorchKVStore:
        """Parse ``torch://host:port``, or ``torch://`` for MASTER_ADDR/MASTER_PORT."""
        parsed = urlparse(uri)
        if parsed.username or parsed.password or parsed.path.strip("/"):
            raise ValueError(f"{uri!r} may name only a host and port")
        if parsed.params or parsed.query or parsed.fragment:
            raise ValueError(f"{uri!r} may name only a host and port")
        if not parsed.netloc:
            host = os.environ.get("MASTER_ADDR")
            port = _parse_port(os.environ.get("MASTER_PORT"), "MASTER_PORT")
            if not host or port is None:
                raise ValueError(
                    f"{uri!r} names no host and port, and MASTER_ADDR/MASTER_PORT "
                    "are not set; use torch://host:port or launch with torchrun"
                )
            return cls(host, port, timeout_s=timeout_s)
        try:
            explicit_port = parsed.port
        except ValueError as exc:
            raise ValueError(f"{uri!r} has an invalid port") from exc
        port = _parse_port(explicit_port, uri)
        if not parsed.hostname or port is None:
            raise ValueError(f"{uri!r} must name both a host and a port")
        return cls(parsed.hostname, port, timeout_s=timeout_s)

    def _call(self, op: str, key: object, fn: Callable[[Any], Any]) -> Any:
        try:
            return fn(self._client())
        except Exception as exc:
            # Don't reuse a client in an unknown state; reconnect next time.
            self._store = None
            raise RuntimeError(
                f"TCPStore {op} of {key!r} at {self.host}:{self.port} failed: {exc}"
            ) from exc

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

    def check_ready(self) -> None:
        try:
            import torch.distributed as dist
        except ImportError as exc:
            raise RuntimeError(
                "aggregate_store='torch://...' needs PyTorch with torch.distributed"
            ) from exc
        if not dist.is_available():
            raise RuntimeError(
                "aggregate_store='torch://...' needs torch.distributed, which this "
                "PyTorch build doesn't include"
            )
        try:
            self.exists("zephon/ready")
        except RuntimeError as exc:
            raise RuntimeError(
                f"Can't reach the checkpoint TCPStore at {self.host}:{self.port} "
                f"within {self._timeout_s}s. It must be running before the Pipeline "
                "starts: torchrun's agent provides one, but with "
                "TORCH_DISABLE_SHARE_RDZV_TCP_STORE=1 or outside torchrun, call "
                "torch.distributed.init_process_group() first."
            ) from exc

    def get(self, key: str) -> bytes | None:
        # TCPStore.get blocks until the key exists, so check first.
        if not self.exists(key):
            return None
        return self._call("get", key, lambda store: store.get(key))

    def get_many(self, keys: list[str]) -> list[bytes | None]:
        # multi_get blocks on missing keys, so batch only when all exist.
        if keys and self._call("check", keys, lambda store: store.check(keys)):
            return list(
                self._call("multi_get", keys, lambda store: store.multi_get(keys))
            )
        return [self.get(key) for key in keys]

    def set(self, key: str, value: bytes) -> None:
        self._call("set", key, lambda store: store.set(key, value))

    def set_if_absent(self, key: str, value: bytes) -> bytes:
        # With an empty expected value, compare_set writes only if the key is
        # absent, and returns the stored value either way.
        return self._call(
            "compare_set", key, lambda store: store.compare_set(key, "", value)
        )

    def append(self, key: str, value: bytes) -> None:
        self._call("append", key, lambda store: store.append(key, value))

    def exists(self, key: str) -> bool:
        return self._call("check", key, lambda store: store.check([key]))

    def delete(self, key: str) -> None:
        self._call("delete", key, lambda store: store.delete_key(key))

    def wait(self, keys: list[str], timeout: float) -> bool:
        # TCPStore waits in whole milliseconds; with no budget left, only
        # check whether the keys are already there.
        timeout_ms = math.ceil(timeout * 1000)
        if timeout_ms <= 0:
            return self._call("check", keys, lambda store: store.check(keys))
        import torch.distributed as dist

        def _wait(store: Any) -> bool:
            try:
                store.wait(keys, timedelta(milliseconds=timeout_ms))
            except dist.DistStoreError:
                # A timeout leaves the connection usable.
                return store.check(keys)
            return True

        return self._call("wait", keys, _wait)


def _parse_port(value: object, source: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        port = int(str(value))
    except ValueError:
        raise ValueError(f"{source} is not a valid port: {value!r}") from None
    if not 0 < port < 65536:
        raise ValueError(f"{source} is not a valid port: {value!r}")
    return port


# ---------- Values larger than one write ----------

_MAGIC = b"ZV1"
_INLINE = b"I"
_CHUNKED = b"C"
_WRITE_ID_BYTES = 16
_HEADER_BYTES = len(_MAGIC) + 1 + _WRITE_ID_BYTES
# Total size, chunk count, and a digest of the whole value.
_CHUNKED_INFO = struct.Struct(">QI16s")


@dataclass(frozen=True)
class StagedValue:
    """A value whose chunks are written; publish ``manifest`` at the key."""

    manifest: bytes
    chunk_keys: list[str]


def stage_value(store: KVStore, key: str, data: bytes) -> StagedValue:
    """Write ``data``'s chunks and return the manifest to publish at ``key``.

    A value that fits in one write travels inside its manifest. A larger one is
    split into chunks under a write ID unique to this publication, so retries
    and competing publishers never mix chunks. Callers publish the manifest
    last, from the same client, so a reader that sees it also sees its chunks.
    """
    write_id = uuid.uuid4().bytes
    limit = store.max_value_bytes
    if limit is None or _HEADER_BYTES + len(data) <= limit:
        return StagedValue(_MAGIC + _INLINE + write_id + data, [])
    count = math.ceil(len(data) / limit)
    chunk_keys = [f"{key}/{write_id.hex()}/chunk/{i}" for i in range(count)]
    # One write per chunk: a batched write would exceed the limit as a whole.
    for i, chunk_key in enumerate(chunk_keys):
        store.set(chunk_key, data[i * limit : (i + 1) * limit])
    info = _CHUNKED_INFO.pack(len(data), count, _digest(data))
    return StagedValue(_MAGIC + _CHUNKED + write_id + info, chunk_keys)


def read_staged_value(store: KVStore, key: str, manifest: bytes) -> bytes:
    """Return the complete value that ``manifest``, read from ``key``, names.

    Raise if the manifest is malformed or the reassembled value doesn't match
    the size and digest it records.
    """
    if not manifest.startswith(_MAGIC) or len(manifest) < _HEADER_BYTES:
        raise RuntimeError(f"Value at {key} has no valid manifest")
    kind = manifest[len(_MAGIC) : len(_MAGIC) + 1]
    write_id = manifest[len(_MAGIC) + 1 : _HEADER_BYTES]
    body = manifest[_HEADER_BYTES:]
    if kind == _INLINE:
        return body
    if kind != _CHUNKED or len(body) != _CHUNKED_INFO.size:
        raise RuntimeError(f"Value at {key} has no valid manifest")
    size, count, digest = _CHUNKED_INFO.unpack(body)
    chunk_keys = [f"{key}/{write_id.hex()}/chunk/{i}" for i in range(count)]
    chunks = store.get_many(chunk_keys)
    missing = [k for k, chunk in zip(chunk_keys, chunks) if chunk is None]
    if missing:
        raise RuntimeError(
            f"Value at {key} is missing {len(missing)} of {count} chunks, e.g. {missing[0]}"
        )
    data = b"".join(chunk for chunk in chunks if chunk is not None)
    if len(data) != size or _digest(data) != digest:
        raise RuntimeError(
            f"Value at {key} doesn't match its manifest: {len(data)} of {size} bytes"
        )
    return data


def _digest(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=16).digest()


register_kv_store("torch", TorchKVStore.from_uri)
