"""Router storage backend that delegates by URL scheme."""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Mapping

from .base import StorageBackend
from .local import LocalFSBackend


def _make_s3_backend() -> StorageBackend:
    from .s3 import S3Backend

    return S3Backend()


def _make_gcs_backend() -> StorageBackend:
    from .gcs import GCSBackend

    return GCSBackend()


class RouterStorageBackend(StorageBackend):
    """Delegate storage operations to a backend chosen by path scheme.

    - ``s3://`` -> S3 backend
    - ``gs://`` or ``gcs://`` -> GCS backend
    - otherwise -> Local filesystem
    """

    def __init__(self, *, local_root: Path | None = None) -> None:
        self._local = LocalFSBackend(root=local_root or Path("/"))
        self._s3 = None
        self._gcs = None

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
        return self._local

    def open(self, path: str, mode: str = "rb", **kwargs: Any) -> IO[bytes] | IO[str]:
        return self._backend_for(path).open(path, mode, **kwargs)

    def exists(self, path: str) -> bool:
        return self._backend_for(path).exists(path)

    def download(self, src: str, dst: str, timeout: float | None = None) -> None:
        return self._backend_for(src).download(src, dst, timeout)

    def listdir(self, path: str) -> list[str]:
        return self._backend_for(path).listdir(path)

    def stat(self, path: str) -> Mapping[str, int]:
        return self._backend_for(path).stat(path)


__all__ = ["RouterStorageBackend"]
