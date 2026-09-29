# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""File-name suffixes identifying each built-in format's data files.

The single source for suffix checks: format auto-detection, index-less
discovery, and the index builders all match file names through this module.
"""

from __future__ import annotations

from collections.abc import Iterable

from zephon._internal.utils.compression import COMPRESSION_SUFFIXES

JSONL_SUFFIXES: tuple[str, ...] = (
    ".jsonl",
    *(f".jsonl{suffix}" for suffix in COMPRESSION_SUFFIXES),
)
PARQUET_SUFFIXES: tuple[str, ...] = (".parquet",)
VORTEX_SUFFIXES: tuple[str, ...] = (".vortex",)
LITDATA_CHUNK_SUFFIX = ".bin"
LITDATA_COMPRESSED_CHUNK_SUFFIX = ".bin.zst"

# Formats recognized from a directory listing, in precedence order for
# directories that mix them.
AUTO_DETECT_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("jsonl", JSONL_SUFFIXES),
    ("vortex", VORTEX_SUFFIXES),
    ("parquet", PARQUET_SUFFIXES),
)


def detect_format(names: Iterable[str]) -> str | None:
    """Return the first :data:`AUTO_DETECT_ORDER` format matching any of ``names``."""
    names = list(names)
    for kind, suffixes in AUTO_DETECT_ORDER:
        if any(name.endswith(suffixes) for name in names):
            return kind
    return None


__all__ = [
    "AUTO_DETECT_ORDER",
    "JSONL_SUFFIXES",
    "LITDATA_CHUNK_SUFFIX",
    "LITDATA_COMPRESSED_CHUNK_SUFFIX",
    "PARQUET_SUFFIXES",
    "VORTEX_SUFFIXES",
    "detect_format",
]
