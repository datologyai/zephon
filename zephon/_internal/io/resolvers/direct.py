"""Resolver that operates on shards present on the local filesystem."""

import os
from dataclasses import dataclass
from pathlib import Path

from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_none,
)

from zephon._internal.io.resolvers.base import ShardResolver
from zephon._internal.io.resolvers.utils import compute_file_hash
from zephon._internal.io.storage import LocalFSBackend
from zephon._internal.io.types import LocalShardFile, LocalShardRef, ShardLocator
from zephon._internal.utils.atomic import atomic_write_bytes


class DirectResolver(ShardResolver):
    """Resolve shard locators by reading directly from the filesystem."""

    def __init__(
        self,
        storage: LocalFSBackend,
        *,
        validate_hash: str | None = None,
    ) -> None:
        self._storage = storage
        self._validate_hash = validate_hash

    def resolve(self, locator: ShardLocator, *, blocking: bool = True) -> LocalShardRef:
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

        compression = compression_name.lower()
        if compression.startswith("zstd"):
            try:
                import zstd
            except ImportError as exc:
                raise RuntimeError(
                    "Resolving compressed shards requires the 'zstd' package"
                ) from exc

            retryable_exceptions: tuple[type[BaseException], ...] = (
                _RetryableValidationError,
                OSError,
                IOError,
                Exception,  # zstd raises generic exceptions
            )

            retrying = Retrying(
                stop=stop_after_attempt(3),
                wait=wait_none(),
                retry=retry_if_exception_type(retryable_exceptions),
                reraise=False,
            )

            def _decompress_once() -> int:
                compressed = zip_path.read_bytes()
                decompressed = zstd.decompress(compressed)
                atomic_write_bytes(raw_path, decompressed)

                # Validation here keeps the retry loop focused on integrity failures rather than
                # letting broken output leak to callers.
                status = self._validate_raw(locator, raw_path)
                if not status.valid:
                    raise _RetryableValidationError(status.to_exception())
                return status.bytes

            try:
                return retrying(_decompress_once)
            except RetryError as exc:
                last_exc = exc.last_attempt.exception()
                if isinstance(last_exc, _RetryableValidationError):
                    raise last_exc.original
                raise
            except _RetryableValidationError as exc:
                raise exc.original
        else:
            raise RuntimeError(
                f"Unsupported compression '{locator.compression}' for direct resolver"
            )

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
