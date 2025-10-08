"""Resolver that operates on shards present on the local filesystem."""

import os
from pathlib import Path

from zephon.io.resolvers.base import ShardResolver
from zephon.io.resolvers.utils import compute_file_hash
from zephon.io.storage import LocalFSBackend
from zephon.io.types import LocalShardFile, LocalShardRef, ShardLocator


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
        if not raw_path.is_file():
            raise FileNotFoundError(
                f"Shard raw file missing: dataset={locator.dataset} shard={locator.shard_id}"
            )

        expected_bytes = int(locator.raw.bytes)
        actual_bytes = raw_path.stat().st_size
        if expected_bytes and actual_bytes < expected_bytes:
            raise ValueError(
                f"Raw shard truncated: expected >= {expected_bytes} bytes, got {actual_bytes}"
            )

        if self._validate_hash and self._validate_hash in locator.raw.hashes:
            expected = locator.raw.hashes[self._validate_hash]
            actual = compute_file_hash(raw_path, self._validate_hash)
            if actual != expected:
                raise ValueError(
                    f"Raw shard checksum mismatch ({self._validate_hash}): {actual} != {expected}"
                )

        zip_file = None
        if locator.zip is not None:
            zip_path = self._filepath(locator.root, locator.zip.basename)
            if not zip_path.is_file():
                raise FileNotFoundError(
                    f"Zip shard missing: dataset={locator.dataset} shard={locator.shard_id}"
                )
            zip_file = LocalShardFile(path=zip_path, bytes=zip_path.stat().st_size)

        raw_file = LocalShardFile(path=raw_path, bytes=actual_bytes)
        return LocalShardRef(
            raw=raw_file,
            zip=zip_file,
            compression=locator.compression,
            extra=locator.extra,
        )

    def touch(self, locator: ShardLocator) -> None:
        return None

    def _filepath(self, root: str, basename: str) -> Path:
        if os.path.isabs(basename):
            return Path(basename)
        return Path(root) / basename


__all__ = ["DirectResolver"]
