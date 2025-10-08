"""Local filesystem storage backend implementation."""

import contextlib
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Mapping

from .base import StorageBackend

_COPY_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass
class LocalFSBackend(StorageBackend):
    """Backend that operates on the local filesystem."""

    root: Path

    def _abspath(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        return p

    def open(self, path: str, mode: str = "rb", **kwargs: Any) -> IO[bytes] | IO[str]:
        abspath = self._abspath(path)
        return abspath.open(mode, **kwargs)

    def exists(self, path: str) -> bool:
        return self._abspath(path).exists()

    def download(self, src: str, dst: str, timeout: float | None = None) -> None:
        source = self._abspath(src)
        target = Path(dst)
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            if timeout is None or timeout <= 0:
                shutil.copyfile(source, target)
                return

            deadline = time.monotonic() + timeout
            with source.open("rb") as in_f, target.open("wb") as out_f:
                while True:
                    chunk = in_f.read(_COPY_CHUNK_SIZE)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"Download timeout after {timeout} seconds: {src}"
                        )
        except Exception:
            with contextlib.suppress(Exception):
                target.unlink()
            raise

    def listdir(self, path: str) -> list[str]:
        abspath = self._abspath(path)
        if not abspath.is_dir():
            raise NotADirectoryError(f"Not a directory: {abspath}")
        return [entry.name for entry in sorted(abspath.iterdir())]

    def stat(self, path: str) -> Mapping[str, int]:
        abspath = self._abspath(path)
        info = abspath.stat()
        return {"size": info.st_size}


__all__ = ["LocalFSBackend"]
