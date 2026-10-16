# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Lance deletion vectors: tombstone files and the row mapping they imply.

A Lance fragment keeps its rows in place when they are deleted and records the
deleted *physical* offsets in a tombstone file under the dataset's
``_deletions/``, either an Arrow IPC file of ``uint32`` offsets or a portable
Roaring bitmap. A shard therefore exposes ``physical_rows - num_deleted_rows``
rows, and reading logical row ``i`` means reading the ``i``-th surviving
physical row.

The Roaring reader here is deliberately self-contained: the format is the
stable portable serialization every Roaring implementation writes, and decoding
it in-process avoids reaching into ``pylance``'s private bitmap bindings.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from zephon._internal.io.storage import StorageBackend

# Portable Roaring serialization (see the format specification shared by
# CRoaring and roaring-rs). The two cookies differ in whether the stream may
# contain run containers, which changes the header that follows.
_COOKIE_NO_RUN = 12346
_COOKIE_WITH_RUN = 12347
_OFFSET_HEADER_THRESHOLD = 4
_ARRAY_CONTAINER_MAX = 4096
_BITSET_CONTAINER_BYTES = 8192


def _parse_roaring(buf: bytes) -> np.ndarray:
    """Decode a portable Roaring bitmap into sorted ``int64`` values."""
    if len(buf) < 8:
        raise ValueError("Truncated Roaring bitmap")
    (cookie_word,) = struct.unpack_from("<I", buf, 0)
    pos = 4
    run_flags = b""
    if cookie_word & 0xFFFF == _COOKIE_WITH_RUN:
        size = (cookie_word >> 16) + 1
        flag_bytes = (size + 7) // 8
        run_flags = buf[pos : pos + flag_bytes]
        pos += flag_bytes
        has_runs = True
    elif cookie_word == _COOKIE_NO_RUN:
        (size,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        has_runs = False
    else:
        raise ValueError(f"Not a portable Roaring bitmap (cookie {cookie_word:#x})")

    descriptors = np.frombuffer(buf, dtype="<u2", count=size * 2, offset=pos)
    pos += size * 4
    if not has_runs or size >= _OFFSET_HEADER_THRESHOLD:
        # The offset header only repeats where the containers already sit.
        pos += size * 4

    parts: list[np.ndarray] = []
    for index in range(size):
        key = int(descriptors[index * 2]) << 16
        cardinality = int(descriptors[index * 2 + 1]) + 1
        is_run = has_runs and bool(run_flags[index // 8] >> (index % 8) & 1)
        if is_run:
            (run_count,) = struct.unpack_from("<H", buf, pos)
            pos += 2
            runs = np.frombuffer(buf, dtype="<u2", count=run_count * 2, offset=pos)
            pos += run_count * 4
            starts = runs[0::2].astype(np.int64)
            lengths = runs[1::2].astype(np.int64) + 1
            offsets = np.arange(int(lengths.sum()), dtype=np.int64)
            offsets -= np.repeat(np.cumsum(lengths) - lengths, lengths)
            values = np.repeat(starts, lengths) + offsets
        elif cardinality <= _ARRAY_CONTAINER_MAX:
            values = np.frombuffer(
                buf, dtype="<u2", count=cardinality, offset=pos
            ).astype(np.int64)
            pos += cardinality * 2
        else:
            words = np.frombuffer(
                buf, dtype=np.uint8, count=_BITSET_CONTAINER_BYTES, offset=pos
            )
            pos += _BITSET_CONTAINER_BYTES
            values = np.flatnonzero(np.unpackbits(words, bitorder="little")).astype(
                np.int64
            )
        parts.append(values + key)
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def _parse_arrow_offsets(buf: bytes) -> np.ndarray:
    """Decode an Arrow IPC deletion file: one batch of ``uint32`` row offsets."""
    import io

    import pyarrow.ipc as ipc

    table = ipc.open_file(io.BytesIO(buf)).read_all()
    if table.num_columns != 1:
        raise ValueError(
            f"Lance deletion file has {table.num_columns} columns, expected 1"
        )
    return table.column(0).to_numpy(zero_copy_only=False).astype(np.int64)


@dataclass(frozen=True)
class DeletionVector:
    """Maps a fragment's logical row offsets onto its physical ones.

    ``surplus[j] = deleted[j] - j`` is non-decreasing, so the number of deleted
    rows at or before a logical offset is a single binary search: logical ``i``
    lives at physical ``i + searchsorted(surplus, i, "right")``.
    """

    deleted: np.ndarray
    surplus: np.ndarray

    @classmethod
    def from_offsets(cls, offsets: np.ndarray) -> "DeletionVector":
        deleted = np.unique(np.asarray(offsets, dtype=np.int64))
        return cls(deleted=deleted, surplus=deleted - np.arange(deleted.size))

    def to_physical(self, indices: list[int]) -> list[int]:
        """Translate logical row offsets to physical ones, preserving order."""
        logical = np.asarray(indices, dtype=np.int64)
        shift = np.searchsorted(self.surplus, logical, side="right")
        return (logical + shift).tolist()


def read_deletion_vector(
    storage: StorageBackend, path: str, *, expected_deleted: int, physical_rows: int
) -> DeletionVector:
    """Load and validate the tombstone file at ``path``.

    The shard's row count is derived from the manifest's ``num_deleted_rows``,
    so a tombstone file that disagrees would shift every row behind it.
    """
    with storage.open(path, "rb") as handle:
        buf = handle.read()
    assert isinstance(buf, bytes)
    offsets = (
        _parse_roaring(buf) if path.endswith(".bin") else _parse_arrow_offsets(buf)
    )
    vector = DeletionVector.from_offsets(offsets)
    if vector.deleted.size != expected_deleted:
        raise ValueError(
            f"Lance deletion file {path} lists {vector.deleted.size} rows, but the "
            f"manifest records {expected_deleted} deleted rows"
        )
    if vector.deleted.size and int(vector.deleted[-1]) >= physical_rows:
        raise ValueError(
            f"Lance deletion file {path} deletes row {int(vector.deleted[-1])} "
            f"beyond the fragment's {physical_rows} physical rows"
        )
    return vector


__all__ = ["DeletionVector", "read_deletion_vector"]
