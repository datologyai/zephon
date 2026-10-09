# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""JSONL index builder.

Registers the ``jsonl`` ``IndexBuilder`` on import; build an index via the
``zephon.build_index`` CLI.
"""

import io
from pathlib import Path

from zephon._internal.io.index.index_builder import (
    IndexBuilder,
    ShardInfo,
    register_builder,
)
from zephon._internal.io.suffixes import JSONL_SUFFIXES
from zephon._internal.utils.compression import compression_for_name, open_decompressed


class JsonlIndexBuilder(IndexBuilder):
    """Index builder for JSON Lines datasets."""

    suffixes = JSONL_SUFFIXES

    def extract_shard_info(self, path: str, file_size: int) -> ShardInfo:
        """Count non-empty JSON Lines records in one shard."""
        compression = compression_for_name(path)
        extra: dict[str, int] = {}
        stream = (
            open(path, "rb")
            if compression is None
            else open_decompressed(path, compression)
        )
        with stream, io.TextIOWrapper(stream, encoding="utf-8") as handle:
            count = sum(1 for line in handle if not line.isspace())
            if compression is not None:
                extra["raw_bytes"] = stream.tell()

        return ShardInfo(
            basename=Path(path).name,
            bytes=file_size,
            num_rows=count,
            extra=extra,
        )


register_builder("jsonl", JsonlIndexBuilder)


__all__ = ["JsonlIndexBuilder"]
