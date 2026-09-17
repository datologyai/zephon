# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""JSONL index builder.

Registers the ``jsonl`` ``IndexBuilder`` on import; build an index via the
``zephon.build_index`` CLI.
"""

from zephon._internal.io.index.index_builder import (
    IndexBuilder,
    ShardInfo,
    register_builder,
)


class JsonlIndexBuilder(IndexBuilder):
    """Index builder for JSON Lines datasets."""

    suffixes = (".jsonl",)

    def extract_shard_info(self, path: str, file_size: int) -> ShardInfo:
        """Count non-empty JSON Lines records in one shard."""
        count = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1

        return ShardInfo(
            basename=path.rsplit("/", 1)[-1],
            bytes=file_size,
            num_rows=count,
        )


register_builder("jsonl", JsonlIndexBuilder)


__all__ = ["JsonlIndexBuilder"]
