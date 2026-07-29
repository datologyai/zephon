"""Protocols for IO storage backends."""

from collections.abc import Iterator
from typing import IO, Any, Mapping, Protocol


class StorageBackend(Protocol):
    """Protocol implemented by all storage backends.

    Notes on semantics:
    - ``open`` is intended for small control files used during discovery
      (e.g., ``index.json``) and light text reads. Implementations may
      internally materialize to a temporary file or use streaming file
      objects. Large data transfers should use ``download`` and be managed
      by resolvers/caches.
    - ``put`` is the write counterpart to ``open``/``download``, intended for
      small control files. Writes are atomic.
    - ``delete`` is idempotent: deleting a non-existent path is not an error.
    - ``glob`` supports ``*`` and ``?`` wildcards; cloud backends list with a
      prefix and filter client-side for efficiency.
    - ``mkdir`` creates directories (local) or folder markers (cloud). The
      ``parents`` and ``exist_ok`` parameters mirror ``pathlib.Path.mkdir``.
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

    def read_range(
        self,
        path: str,
        start: int,
        *,
        end: int | None = None,
        length: int | None = None,
    ) -> bytes | memoryview:
        """Read a byte range from ``path``.

        Implementations should return binary data (``bytes`` or
        ``memoryview``) in the half-open interval ``[start, end)`` when
        ``end`` is provided, or ``length`` bytes starting at ``start`` when
        ``length`` is provided.
        """
        raise NotImplementedError

    def listdir(self, path: str) -> list[str]:
        """List entries directly contained in ``path``."""
        ...

    def stat(self, path: str) -> Mapping[str, int | float]:
        """Return metadata for ``path``.

        Returns a mapping with at least:
        - ``size``: file size in bytes (int)
        - ``mtime``: last modification time as seconds since epoch (float)
        """
        ...

    def put(self, path: str, data: bytes) -> None:
        """Write ``data`` to ``path`` atomically.

        Creates parent directories as needed. On local filesystems, uses
        atomic rename for crash safety. On object storage, writes are
        inherently atomic.
        """
        ...

    def delete(self, path: str) -> None:
        """Delete ``path``.

        Idempotent: no error is raised if ``path`` does not exist.
        """
        ...

    def glob(self, pattern: str) -> list[str]:
        """Return paths matching the glob ``pattern``.

        Supports ``*`` and ``?`` wildcards. For cloud backends, efficiently
        lists with a prefix and filters client-side.

        Example: ``glob("s3://bucket/dir/state_r*.json")``
        """
        ...

    def mkdir(self, path: str, parents: bool = False, exist_ok: bool = False) -> None:
        """Create a directory at ``path``.

        Args:
            path: Directory path to create.
            parents: If True, create parent directories as needed (like mkdir -p).
            exist_ok: If True, don't raise an error if directory already exists.

        For cloud backends, creates a folder marker (empty object with trailing /).
        """
        ...

    def walk(self, path: str) -> Iterator[tuple[str, int]]:
        """Recursively yield ``(rel_path, size)`` for every file under ``path``.

        ``rel_path`` is relative to ``path``, POSIX-style.  Non-existent
        roots yield nothing rather than raise.
        """
        raise NotImplementedError


__all__ = ["StorageBackend"]
