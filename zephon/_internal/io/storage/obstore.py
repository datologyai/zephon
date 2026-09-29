"""Abstract base class for obstore-backed storage backends."""

from __future__ import annotations

import fnmatch
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Mapping

from ._utils import OpenViaDownloadMixin, split_url


class ObstoreBackend(OpenViaDownloadMixin, ABC):
    """Abstract base class for obstore-backed storage backends.

    Subclasses must define:
    - ``valid_schemes``: frozenset of URL schemes this backend handles
    - ``_get_store(bucket)``: returns an obstore Store for the given bucket

    This base class provides shared implementations of all StorageBackend
    methods that are identical across cloud providers (exists, download,
    listdir, stat, put, delete, glob).
    """

    valid_schemes: frozenset[str]

    @abstractmethod
    def _get_store(self, bucket: str) -> Any:
        """Get or create an obstore Store for the given bucket."""
        ...

    def canonical_root(self, path: str, fmt: str | None = None) -> str:
        """Object-store paths already name fixed locations; return ``path``."""
        return path

    def exists(self, path: str) -> bool:
        """Check if a file exists at the given cloud path."""
        scheme, bucket, key = split_url(path)
        if scheme not in self.valid_schemes or not bucket or not key:
            return False

        import obstore as obs

        store = self._get_store(bucket)
        try:
            obs.head(store, key)
            return True
        except Exception as e:
            err = str(e)
            if "404" in err or "NoSuchKey" in err or "NotFound" in err:
                return False
            if "403" in err or "AccessDenied" in err:
                return False
            raise

    def _download_with_store(self, store: Any, key: str, src: str, dst: str) -> None:
        """Download using a specific store, with error handling."""
        import obstore as obs

        try:
            data = obs.get(store, key).bytes()
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(data)
        except Exception as e:
            err = str(e)
            if "403" in err or "AccessDenied" in err:
                raise FileNotFoundError(f"Access denied: {src}") from e
            if "404" in err or "NoSuchKey" in err or "NotFound" in err:
                raise FileNotFoundError(f"Object not found: {src}") from e
            raise

    def download(self, src: str, dst: str, timeout: float | None = None) -> None:
        """Download a file from cloud storage to local disk."""
        # timeout param kept for StorageBackend protocol compatibility but unused.
        # obstore only supports store-level timeout (set in _get_store via client_options).
        # See: https://developmentseed.org/obstore/latest/api/store/config/
        del timeout
        scheme, bucket, key = split_url(src)
        if scheme not in self.valid_schemes or not bucket or not key:
            raise ValueError(f"Invalid URL: {src}")

        store = self._get_store(bucket)
        self._download_with_store(store, key, src, dst)

    def read_range(
        self,
        path: str,
        start: int,
        *,
        end: int | None = None,
        length: int | None = None,
    ) -> memoryview:
        """Read a byte range from a cloud object."""
        scheme, bucket, key = split_url(path)
        if scheme not in self.valid_schemes or not bucket or not key:
            raise ValueError(f"Invalid URL: {path}")

        import obstore as obs

        store = self._get_store(bucket)
        try:
            data = obs.get_range(store, key, start=start, end=end, length=length)
            return memoryview(data)
        except Exception as e:
            err = str(e)
            if "403" in err or "AccessDenied" in err:
                raise FileNotFoundError(f"Access denied: {path}") from e
            if "404" in err or "NoSuchKey" in err or "NotFound" in err:
                raise FileNotFoundError(f"Object not found: {path}") from e
            raise

    def _walk_with_store(self, store: Any, base: str) -> Iterator[tuple[str, int]]:
        """Yield ``(rel_path, size)`` for every object under ``base``.

        Zero-byte folder markers (keys equal to ``base`` or ending in ``/``)
        are skipped — they aren't downloadable as files.
        """
        import obstore as obs

        for chunk in obs.list(store, prefix=base):
            for obj in chunk:
                obj_path = obj["path"]
                if not obj_path.startswith(base):
                    continue
                rel = obj_path[len(base) :]
                if not rel or rel.endswith("/"):
                    continue
                yield rel, int(obj["size"])

    def walk(self, path: str) -> Iterator[tuple[str, int]]:
        """Recursively yield ``(rel_path, size)`` for every object under ``path``.

        Unlike :meth:`listdir`, this descends into subdirectories and
        returns sizes from the listing (no extra ``stat`` round-trip).
        """
        scheme, bucket, prefix = split_url(path)
        if scheme not in self.valid_schemes or not bucket:
            raise ValueError(f"Invalid URL: {path}")

        store = self._get_store(bucket)
        base = prefix if (not prefix or prefix.endswith("/")) else prefix + "/"
        yield from self._walk_with_store(store, base)

    def listdir(self, path: str) -> list[str]:
        """List files in a cloud directory (prefix)."""
        scheme, bucket, prefix = split_url(path)
        if scheme not in self.valid_schemes or not bucket:
            raise NotADirectoryError(f"Not a valid directory: {path}")

        import obstore as obs

        store = self._get_store(bucket)
        base = (
            (prefix + "/") if (prefix and not prefix.endswith("/")) else (prefix or "")
        )

        entries: list[str] = []
        for chunk in obs.list(store, prefix=base):
            for obj in chunk:
                obj_path = obj["path"]
                if not obj_path.startswith(base):
                    continue
                name = obj_path[len(base) :]
                # Only direct children (no nested paths)
                if name and "/" not in name:
                    entries.append(name)

        return sorted(entries)

    def stat(self, path: str) -> Mapping[str, int | float]:
        """Get file metadata (size, mtime) for a cloud object."""
        scheme, bucket, key = split_url(path)
        if scheme not in self.valid_schemes or not bucket or not key:
            raise FileNotFoundError(f"Invalid path: {path}")

        import obstore as obs

        store = self._get_store(bucket)
        try:
            meta = obs.head(store, key)
            return {
                "size": meta["size"],
                "mtime": meta["last_modified"].timestamp(),
            }
        except Exception as e:
            err = str(e)
            if "404" in err or "NoSuchKey" in err or "NotFound" in err:
                raise FileNotFoundError(f"Object not found: {path}") from e
            if "403" in err or "AccessDenied" in err:
                raise PermissionError(f"Access denied: {path}") from e
            raise

    def put(self, path: str, data: bytes) -> None:
        """Write data to a cloud object. Object storage writes are atomic."""
        scheme, bucket, key = split_url(path)
        if scheme not in self.valid_schemes or not bucket or not key:
            raise ValueError(f"Invalid path: {path}")

        import obstore as obs

        store = self._get_store(bucket)
        obs.put(store, key, data)

    def delete(self, path: str) -> None:
        """Delete a cloud object. Idempotent (no error if not found)."""
        scheme, bucket, key = split_url(path)
        if scheme not in self.valid_schemes or not bucket or not key:
            return  # Invalid path is "already deleted"

        import obstore as obs

        store = self._get_store(bucket)
        try:
            obs.delete(store, key)
        except Exception as e:
            err = str(e)
            if "404" in err or "NoSuchKey" in err or "NotFound" in err:
                return  # Idempotent: already deleted
            raise

    def glob(self, pattern: str) -> list[str]:
        """Match cloud objects against a glob pattern.

        Example: glob("s3://bucket/prefix/state_r*.json")
        """
        import re

        import obstore as obs

        scheme, bucket, key_pattern = split_url(pattern)
        if scheme not in self.valid_schemes or not bucket:
            return []

        # Extract prefix up to first wildcard for efficient listing
        wildcard_match = re.search(r"[*?]", key_pattern)
        if wildcard_match:
            prefix = key_pattern[: wildcard_match.start()]
            # Trim to last slash to get directory prefix
            if "/" in prefix:
                prefix = prefix.rsplit("/", 1)[0] + "/"
            else:
                prefix = ""
        else:
            # No wildcards - check if exact path exists
            if self.exists(pattern):
                return [pattern]
            return []

        store = self._get_store(bucket)
        results = []
        for chunk in obs.list(store, prefix=prefix):
            for obj in chunk:
                if fnmatch.fnmatch(obj["path"], key_pattern):
                    results.append(f"{scheme}://{bucket}/{obj['path']}")

        return sorted(results)

    def mkdir(self, path: str, parents: bool = False, exist_ok: bool = False) -> None:
        """Create a folder marker in cloud storage.

        For cloud storage, directories are virtual. This creates an empty object
        with a trailing slash to serve as a folder marker, which some tools expect.

        Args:
            path: Directory path to create (e.g., "s3://bucket/path/to/dir").
            parents: If True, create parent folder markers as needed.
            exist_ok: If True, don't raise an error if marker already exists.
        """
        import obstore as obs

        scheme, bucket, key = split_url(path)
        if scheme not in self.valid_schemes or not bucket:
            raise ValueError(f"Invalid path: {path}")

        store = self._get_store(bucket)

        # Normalize key to have trailing slash for folder marker
        key = key.rstrip("/") + "/" if key else ""

        if not key:
            return  # Nothing to create for bucket root

        # Check if marker already exists when exist_ok=False
        if not exist_ok:
            try:
                obs.head(store, key)
                raise FileExistsError(f"Directory already exists: {path}")
            except Exception as e:
                err = str(e)
                if (
                    "404" not in err
                    and "NoSuchKey" not in err
                    and "NotFound" not in err
                ):
                    raise

        if parents:
            # Create parent folder markers
            parts = key.rstrip("/").split("/")
            for i in range(1, len(parts) + 1):
                parent_key = "/".join(parts[:i]) + "/"
                obs.put(store, parent_key, b"")
        else:
            obs.put(store, key, b"")


__all__ = ["ObstoreBackend"]
