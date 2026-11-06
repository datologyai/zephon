# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""ArrayRecord index builder.

Registers the ``array_record`` ``IndexBuilder`` on import; build an index via
the ``zephon.build_index`` CLI.
"""

from zephon._internal.io.formats.array_record import _read_array_record_row_count
from zephon._internal.io.index.index_builder import (
    IndexBuilder,
    ShardInfo,
    register_builder,
)
from zephon._internal.io.suffixes import ARRAY_RECORD_SUFFIXES


class ArrayRecordIndexBuilder(IndexBuilder):
    """Index builder for ArrayRecord datasets."""

    suffixes = ARRAY_RECORD_SUFFIXES

    def extract_shard_info(self, path: str, file_size: int) -> ShardInfo:
        """Extract the record count from an ArrayRecord file."""
        count = _read_array_record_row_count(path, self._storage, file_size)
        return ShardInfo(
            basename=path.rsplit("/", 1)[-1],
            bytes=file_size,
            num_rows=count,
            extra={"length": count},
        )


register_builder("array_record", ArrayRecordIndexBuilder)


__all__ = ["ArrayRecordIndexBuilder"]
