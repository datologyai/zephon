"""Resolver that operates on shards present on the local filesystem."""

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_none,
)

from zephon._internal.io.resolvers.base import ShardResolver
from zephon._internal.io.resolvers.utils import compute_file_hash
from zephon._internal.io.storage import (
    LocalFSBackend,
    StorageBackend,
    is_remote_path,
)
from zephon._internal.io.types import (
    LocalShardFile,
    LocalShardRef,
    RemoteShardRef,
    ShardLocator,
    ShardRef,
)
from zephon._internal.utils.compression import decompress_file, normalize_compression


class DirectResolver(ShardResolver):
    """Resolve shard locators without a shard cache.

    A shard under a local root resolves to its file, which is validated first.
    A shard under a remote root resolves to a :class:`RemoteShardRef` on
    ``remote_storage``, without IO. Its format then reads it in place.
    """

    def __init__(
        self,
        storage: LocalFSBackend,
        *,
        remote_storage: StorageBackend | None = None,
        validate_hash: str | None = None,
    ) -> None:
        self._storage = storage
        self._remote_storage = remote_storage
        self._validate_hash = validate_hash

    def resolve(self, locator: ShardLocator, *, blocking: bool = True) -> ShardRef:
        if is_remote_path(locator.root):
            return self._resolve_remote(locator)

        raw_path = self._filepath(locator.root, locator.raw.basename)
        raw_bytes = self._ensure_raw_ready(locator, raw_path)

        zip_file = None
        if locator.zip is not None:
            zip_path = self._filepath(locator.root, locator.zip.basename)
            if not zip_path.is_file():
                raise FileNotFoundError(
                    f"Zip shard missing: dataset={locator.dataset} shard={locator.shard_id}"
                )
            zip_file = LocalShardFile(path=zip_path, bytes=zip_path.stat().st_size)

        raw_file = LocalShardFile(path=raw_path, bytes=raw_bytes)
        return LocalShardRef(
            raw=raw_file,
            zip=zip_file,
            compression=locator.compression,
            extra=locator.extra,
            cache_hit=True,
        )

    def touch(self, locator: ShardLocator) -> None:
        return None

    def _resolve_remote(self, locator: ShardLocator) -> RemoteShardRef:
        if self._remote_storage is None:
            raise ValueError(
                f"Shard {locator.shard_id} of dataset {locator.dataset!r} is under "
                + f"the remote root {locator.root!r}; set cache.enabled=True"
            )

        if locator.zip is not None or locator.compression:
            raise ValueError(
                f"Compressed shards of dataset {locator.dataset!r} under a remote "
                + "root must be downloaded; set cache.enabled=True"
            )

        # A whole-file hash cannot be checked without reading the whole file.
        if self._validate_hash and self._validate_hash in locator.raw.hashes:
            raise ValueError(
                f"cache.validate_hash={self._validate_hash!r} for dataset "
                + f"{locator.dataset!r} under a remote root needs cache.enabled=True"
            )

        return RemoteShardRef(
            storage=self._remote_storage,
            path=f"{locator.root.rstrip('/')}/{locator.raw.basename}",
            bytes=locator.raw.bytes,
            extra=locator.extra,
            cache_hit=False,
        )

    def _filepath(self, root: str, basename: str) -> Path:
        if os.path.isabs(basename):
            return Path(basename)
        return Path(root) / basename

    def _ensure_raw_ready(self, locator: ShardLocator, raw_path: Path) -> int:
        # First see if we already have a healthy shard on disk; avoids decompression entirely.
        status = self._validate_raw(locator, raw_path)
        if status.valid:
            return status.bytes

        if locator.zip is None or not locator.compression:
            raise status.to_exception()

        # We may need to materialise the raw shard, so ensure its directory exists first.
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        return self._prepare_raw_file(locator, raw_path)

    def _prepare_raw_file(self, locator: ShardLocator, raw_path: Path) -> int:
        zip_meta = locator.zip
        compression_name = locator.compression
        if zip_meta is None or not compression_name:
            raise RuntimeError("Cannot prepare raw shard without compression metadata")

        zip_path = self._filepath(locator.root, zip_meta.basename)
        if not zip_path.is_file():
            raise FileNotFoundError(
                f"Zip shard missing: dataset={locator.dataset} shard={locator.shard_id}"
            )

        compression = normalize_compression(compression_name)
        retrying = Retrying(
            stop=stop_after_attempt(3),
            wait=wait_none(),
            # Decompressors raise library-specific errors, so retry on any.
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )

        def _decompress_once() -> int:
            # Unique temp sibling, so concurrent resolvers never see a partial file.
            # Its length is fixed, so long shard names stay within NAME_MAX.
            tmp = raw_path.with_name(f".{uuid.uuid4().hex}.tmp")
            try:
                decompress_file(zip_path, tmp, compression)
                os.replace(tmp, raw_path)
            finally:
                tmp.unlink(missing_ok=True)

            # Validation here keeps the retry loop focused on integrity failures rather than
            # letting broken output leak to callers.
            status = self._validate_raw(locator, raw_path)
            if not status.valid:
                raise _RetryableValidationError(status.to_exception())
            return status.bytes

        try:
            return retrying(_decompress_once)
        except _RetryableValidationError as exc:
            raise exc.original

    def _validate_raw(
        self, locator: ShardLocator, raw_path: Path
    ) -> "_RawValidationStatus":
        if not raw_path.is_file():
            return _RawValidationStatus(
                valid=False,
                bytes=0,
                error=FileNotFoundError(
                    f"Shard raw file missing: dataset={locator.dataset} shard={locator.shard_id}"
                ),
            )

        actual_bytes = raw_path.stat().st_size
        expected_bytes_raw = locator.raw.bytes
        try:
            expected_bytes = int(expected_bytes_raw)
        except (TypeError, ValueError):
            expected_bytes = 0

        if actual_bytes == 0 and (locator.zip is not None or expected_bytes):
            return _RawValidationStatus(
                valid=False,
                bytes=actual_bytes,
                error=ValueError(
                    f"Raw shard empty: dataset={locator.dataset} shard={locator.shard_id}"
                ),
            )

        if expected_bytes and actual_bytes < expected_bytes:
            return _RawValidationStatus(
                valid=False,
                bytes=actual_bytes,
                error=ValueError(
                    f"Raw shard truncated: expected >= {expected_bytes} bytes, got {actual_bytes}"
                ),
            )

        if self._validate_hash and self._validate_hash in locator.raw.hashes:
            expected = locator.raw.hashes[self._validate_hash]
            actual = compute_file_hash(raw_path, self._validate_hash)
            if actual != expected:
                return _RawValidationStatus(
                    valid=False,
                    bytes=actual_bytes,
                    error=ValueError(
                        f"Raw shard checksum mismatch ({self._validate_hash}): {actual} != {expected}"
                    ),
                )

        return _RawValidationStatus(valid=True, bytes=actual_bytes, error=None)


__all__ = ["DirectResolver"]


@dataclass(slots=True)
class _RawValidationStatus:
    valid: bool
    bytes: int
    error: Exception | None

    def to_exception(self) -> Exception:
        if self.error is not None:
            return self.error
        return RuntimeError("Raw validation succeeded unexpectedly")


class _RetryableValidationError(Exception):
    def __init__(self, original: Exception) -> None:
        self.original = original
        super().__init__(str(original))
