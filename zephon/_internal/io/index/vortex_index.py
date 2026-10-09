# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex index builder.

Registers the ``vortex`` ``IndexBuilder`` on import; build an index via the
``zephon.build_index`` CLI.
"""

from pathlib import Path
from typing import Any

from zephon._internal.io.formats.vortex import _read_vortex_row_count
from zephon._internal.io.index.index_builder import (
    IndexBuilder,
    ShardInfo,
    register_builder,
)
from zephon._internal.io.storage.local import LocalFSBackend
from zephon._internal.io.suffixes import VORTEX_SUFFIXES


class VortexIndexBuilder(IndexBuilder):
    """Index builder for Vortex datasets."""

    suffixes = VORTEX_SUFFIXES

    def extract_shard_info(self, path: str, file_size: int) -> ShardInfo:
        """Extract row count and metadata from a Vortex file."""
        count = _read_vortex_row_count(path, LocalFSBackend(Path.cwd()), file_size)

        extra: dict[str, Any] = {"length": count}

        return ShardInfo(
            basename=path.rsplit("/", 1)[-1],
            bytes=file_size,
            num_rows=count,
            extra=extra,
        )


register_builder("vortex", VortexIndexBuilder)


__all__ = ["VortexIndexBuilder"]
