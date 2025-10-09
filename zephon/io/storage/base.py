"""Protocols for IO storage backends."""

from typing import IO, Any, Mapping, Protocol


class StorageBackend(Protocol):
    """Protocol implemented by all storage backends.

    Notes on semantics:
    - ``open`` is intended for small control files used during discovery
      (e.g., ``index.json``) and light text reads. Implementations may
      internally materialize to a temporary file or use streaming file
      objects. Large data transfers should use ``download`` and be managed
      by resolvers/caches.
    """

    def open(self, path: str, mode: str = "rb", **kwargs: Any) -> IO[bytes] | IO[str]:
        """Open ``path`` and return a file-like object.

        Intended for small metadata files; backends may use temporary files
        or streamed file objects under the hood.
        """
        ...

    def exists(self, path: str) -> bool:
        """Return ``True`` if ``path`` exists on the backend."""
        ...

    def download(self, src: str, dst: str, timeout: float | None = None) -> None:
        """Copy ``src`` from the backend onto the local filesystem at ``dst``."""
        ...

    def listdir(self, path: str) -> list[str]:
        """List entries directly contained in ``path``."""
        ...

    def stat(self, path: str) -> Mapping[str, int]:
        """Return basic metadata (at least ``size``) for ``path``."""
        ...


__all__ = ["StorageBackend"]
