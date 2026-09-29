# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex index builder.

Registers the ``vortex`` ``IndexBuilder`` on import; build an index via the
``zephon.build_index`` CLI.
"""

from typing import Any

from zephon._internal.io.index.index_builder import (
    IndexBuilder,
    ShardInfo,
    register_builder,
)
from zephon._internal.io.suffixes import VORTEX_SUFFIXES

try:
    import vortex as _vortex
except ImportError:
    _vortex = None


class VortexIndexBuilder(IndexBuilder):
    """Index builder for Vortex datasets."""

    suffixes = VORTEX_SUFFIXES

    def extract_shard_info(self, path: str, file_size: int) -> ShardInfo:
        """Extract row count and metadata from a Vortex file."""
        if _vortex is None:
            raise ImportError(
                "vortex-data is required for Vortex index building. "
                "Install with: pip install zephon[vortex]"
            )

        url = f"file://{path}"
        reader = _vortex.io.read_url(url)
        count = len(reader)

        # Extract schema info if available
        extra: dict[str, Any] = {"length": count}

        return ShardInfo(
            basename=path.rsplit("/", 1)[-1],
            bytes=file_size,
            num_rows=count,
            extra=extra,
        )


register_builder("vortex", VortexIndexBuilder)


__all__ = ["VortexIndexBuilder"]
