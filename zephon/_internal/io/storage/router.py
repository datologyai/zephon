"""Router storage backend that delegates by URL scheme."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any, Mapping, cast

from .base import StorageBackend
from .local import LocalFSBackend


def _make_s3_backend() -> StorageBackend:
    from .s3 import S3Backend

    return S3Backend()


def _make_gcs_backend() -> StorageBackend:
    from .gcs import GCSBackend

    return GCSBackend()


def _make_hf_backend() -> StorageBackend:
    from .hf import HFBackend

    return HFBackend()


class RouterStorageBackend(StorageBackend):
    """Delegate storage operations to a backend chosen by path scheme.

    - ``s3://`` -> S3 backend
    - ``gs://`` or ``gcs://`` -> GCS backend
    - ``hf://`` -> HuggingFace backend (a split's uploaded or converted files)
    - otherwise -> Local filesystem
    """

    def __init__(self, *, local_root: Path | None = None) -> None:
        self._local = LocalFSBackend(root=local_root or Path("/"))
        self._s3 = None
        self._gcs = None
        self._hf = None

    def _backend_for(self, path: str) -> StorageBackend:
        import urllib.parse as _url

        scheme = _url.urlparse(path).scheme
        if not scheme:
            return self._local
        if scheme == "s3":
            if self._s3 is None:
                self._s3 = _make_s3_backend()
            return self._s3
        if scheme in {"gs", "gcs"}:
            if self._gcs is None:
                self._gcs = _make_gcs_backend()
            return self._gcs
        if scheme == "hf":
            if self._hf is None:
                self._hf = _make_hf_backend()
            return self._hf
        return self._local

    def open(self, path: str, mode: str = "rb", **kwargs: Any) -> IO[bytes] | IO[str]:
        return self._backend_for(path).open(path, mode, **kwargs)

    def exists(self, path: str) -> bool:
        return self._backend_for(path).exists(path)

    def canonical_root(self, path: str, fmt: str | None = None) -> str:
        return self._backend_for(path).canonical_root(path, fmt)

    def download(self, src: str, dst: str, timeout: float | None = None) -> None:
        return self._backend_for(src).download(src, dst, timeout)

    def read_range(
        self,
        path: str,
        start: int,
        *,
        end: int | None = None,
        length: int | None = None,
    ) -> bytes | memoryview:
        if start < 0:
            raise ValueError("start must be non-negative")
        if end is not None and length is not None:
            raise ValueError("Specify at most one of end or length")

        backend = self._backend_for(path)
        try:
            return backend.read_range(path, start, end=end, length=length)
        except (AttributeError, NotImplementedError):
            with backend.open(path, "rb") as handle:
                fh = cast(IO[bytes], handle)
                fh.seek(start)
                if end is not None:
                    return fh.read(max(0, end - start))
                if length is not None:
                    return fh.read(length)
                return fh.read()

    def listdir(self, path: str) -> list[str]:
        return self._backend_for(path).listdir(path)

    def stat(self, path: str) -> Mapping[str, int | float]:
        return self._backend_for(path).stat(path)

    def put(self, path: str, data: bytes) -> None:
        return self._backend_for(path).put(path, data)

    def delete(self, path: str) -> None:
        return self._backend_for(path).delete(path)

    def glob(self, pattern: str) -> list[str]:
        return self._backend_for(pattern).glob(pattern)

    def mkdir(self, path: str, parents: bool = False, exist_ok: bool = False) -> None:
        return self._backend_for(path).mkdir(path, parents, exist_ok)

    def is_cloud_path(self, path: str) -> bool:
        """Return True if path uses a cloud storage scheme."""
        return self._backend_for(path) is not self._local

    def walk(self, path: str) -> Iterator[tuple[str, int]]:
        """Recursively yield ``(rel_path, size)`` under ``path``.

        Delegates to whichever backend matches ``path``'s scheme; local
        filesystems use :class:`pathlib.Path.rglob` under the hood.
        """
        yield from self._backend_for(path).walk(path)


__all__ = ["RouterStorageBackend"]
