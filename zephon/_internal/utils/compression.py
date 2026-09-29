# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Compression codecs shared by shard I/O and checkpoint aggregation."""

from __future__ import annotations

import bz2
import gzip
import lzma
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import BinaryIO, cast

if sys.version_info >= (3, 14):
    try:  # optional stdlib module: CPython can be built without it
        from compression import zstd
    except ImportError:
        zstd = None
else:
    from backports import zstd

try:  # LZ4 is optional
    import lz4.frame as lz4frame
except ImportError:
    lz4frame = None

# File-name suffix -> algorithm of compressed data files.
COMPRESSION_SUFFIXES: dict[str, str] = {
    ".gz": "gzip",
    ".zst": "zstd",
    ".bz2": "bz2",
    ".xz": "xz",
}

_ALIASES: dict[str, str] = {
    "gz": "gzip",
    "gzip": "gzip",
    "bz2": "bz2",
    "bzip2": "bz2",
    "xz": "xz",
    "lzma": "xz",
    "zst": "zstd",
    "zstd": "zstd",
    "zstandard": "zstd",
    "lz4": "lz4",
}

_COPY_CHUNK_BYTES = 8 * 1024 * 1024
_MISSING_ZSTD = (
    "zstd needs the compression.zstd module, which this Python build was "
    "compiled without"
)


def normalize_compression(name: str) -> str:
    """Return the canonical algorithm for a compression label.

    Accepts aliases (``gz``, ``zst``, ``bzip2``) and the ``algo:level`` form that
    LitData/MDS record (``zstd:7``).

    Raises:
        ValueError: if the algorithm is not supported.
    """
    algo = name.lower().split(":", 1)[0]
    try:
        return _ALIASES[algo]
    except KeyError:
        raise ValueError(f"Unsupported compression: {name}") from None


def compression_for_name(name: str) -> str | None:
    """Return the algorithm implied by ``name``'s compression suffix, if any."""
    for suffix, algo in COMPRESSION_SUFFIXES.items():
        if name.endswith(suffix):
            return algo
    return None


def require_zstd() -> ModuleType:
    """Return the zstd codec module (stdlib ``compression.zstd`` or its backport).

    Raises:
        RuntimeError: when this Python build has no zstd codec module.
    """
    if zstd is None:
        raise RuntimeError(_MISSING_ZSTD)
    return zstd


def open_decompressed(
    src: str | os.PathLike[str] | BinaryIO, compression: str
) -> BinaryIO:
    """Open ``src`` through a streaming decompressor.

    ``src`` is a path (owned and closed with the returned stream) or a binary
    file object (left open). Multi-frame / multi-member input and zstd frames
    without a content-size header are supported, and input is streamed rather
    than read whole.

    Raises:
        ValueError: if the algorithm is not supported.
        RuntimeError: when the codec module for ``zstd`` or ``lz4`` is missing.
    """
    algo = normalize_compression(compression)
    if algo == "gzip":
        return cast(BinaryIO, gzip.open(src, "rb"))
    if algo == "bz2":
        return cast(BinaryIO, bz2.open(src, "rb"))
    if algo == "xz":
        return cast(BinaryIO, lzma.open(src, "rb"))
    if algo == "zstd":
        return cast(BinaryIO, require_zstd().open(src, "rb"))
    if lz4frame is None:
        raise RuntimeError("lz4 compression requires the 'lz4' package")
    return cast(BinaryIO, lz4frame.open(src, "rb"))


def decompress_file(src: Path, dst: Path, compression: str) -> None:
    """Stream-decompress ``src`` into ``dst``, overwriting it."""
    with open_decompressed(src, compression) as stream, dst.open("wb") as out:
        shutil.copyfileobj(stream, out, _COPY_CHUNK_BYTES)


__all__ = [
    "COMPRESSION_SUFFIXES",
    "compression_for_name",
    "decompress_file",
    "normalize_compression",
    "open_decompressed",
    "require_zstd",
    "zstd",
]
