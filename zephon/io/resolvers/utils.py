"""Utility helpers shared across resolver implementations."""

import hashlib
from pathlib import Path


def compute_file_hash(path: Path, algo: str) -> str:
    """Compute the hexadecimal digest of ``path`` using ``algo``."""
    try:
        h = hashlib.new(algo)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algo}") from exc
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = ["compute_file_hash"]
