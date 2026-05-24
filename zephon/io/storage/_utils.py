"""Internal helpers shared across storage backends."""

import os
import tempfile
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Optional, cast


def split_url(path: str) -> tuple[str, str, str]:
    """Return (scheme, bucket_or_host, key) for URL-like paths.

    The key component has no leading slash. For non-URL strings, the scheme
    is an empty string and the other parts are empty.
    """
    obj = urllib.parse.urlparse(path)
    return obj.scheme, obj.netloc, obj.path.lstrip("/")


@dataclass
class TempLocalFile:
    """Context-managed local file that is deleted on close.

    Used by cloud backends to implement ``open`` by first downloading to a
    temporary path and then returning a file handle that cleans itself up.
    """

    path: Path
    mode: str
    _fh: IO[bytes] | IO[str] | None = field(default=None, init=False)
    _open_kwargs: dict[str, Any] = field(default_factory=dict, init=False)

    def open(self, **kwargs: Any) -> IO[bytes] | IO[str]:
        if kwargs:
            self._open_kwargs = dict(kwargs)
        self._fh = self.path.open(self.mode, **kwargs)
        return self._fh

    def close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
        finally:
            try:
                self.path.unlink(missing_ok=True)
            except Exception:
                pass

    def __enter__(self) -> IO[bytes] | IO[str]:
        return self._fh if self._fh is not None else self.open(**self._open_kwargs)

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()


class OpenViaDownloadMixin:
    """Mixin providing ``open`` by delegating to ``download`` into a temp file."""

    def open(self, path: str, mode: str = "rb", **kwargs: Any) -> IO[bytes] | IO[str]:  # type: ignore[override]
        # Use NamedTemporaryFile pattern that works on Windows: create and close
        # the file, then reopen by path.
        fd, tmpname = tempfile.mkstemp(prefix="zephon_tmp_", suffix=".bin")
        os.close(fd)
        tmp = Path(tmpname)
        try:
            # Full read of entire file!
            self.download(path, str(tmp))  # type: ignore[attr-defined].
        except Exception:
            tmp.unlink(missing_ok=True)
            # Some backends like HF stage partial bytes in a sibling ``.incomplete`` file.
            Path(str(tmp) + ".incomplete").unlink(missing_ok=True)
            raise
        # Reads from local temp
        wrapper = TempLocalFile(tmp, mode)
        if kwargs:
            wrapper.open(**kwargs)
        return cast(IO[bytes] | IO[str], wrapper)


__all__ = ["split_url", "TempLocalFile", "OpenViaDownloadMixin"]
