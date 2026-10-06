"""Arrow layout detection and payload decoding for LitData chunks.

Starting with LitData 0.2.74 (upstream PR #897), eligible dictionary rows can
be stored as Arrow IPC. LitData's own ``PyTreeLoader`` reads both the older
binary layout and this Arrow layout. The ``item_loader="PyTreeLoader"`` field
in ``index.json`` names that reader class; it does not distinguish the layouts.

Zephon uses ``PyTreeLoader`` for binary pytree chunks and ``ArrowLoader`` for
Arrow chunks. The format handler selects the decoder by checking the file's
Arrow footer, independently of the installed LitData version.

The on-disk contract comes from LitData's ``streaming/item_loader.py`` helpers
``append_arrow_row_footer``, ``_arrow_footer_span``, and
``open_arrow_footer_reader``: an Arrow IPC payload is followed by its
little-endian uint32 byte length and the eight-byte magic ``LDARW01`` plus NUL.
IPC files use 256-row batches, except for a possibly shorter final batch;
LitData's writer and reader both use ``_DEFAULT_BATCH_ROWS`` for this layout.
Upstream references:

* https://github.com/Lightning-AI/litData/pull/897
* https://github.com/Lightning-AI/litData/blob/v0.2.75/src/litdata/streaming/item_loader.py

This module is Zephon's implementation of that contract using PyArrow's public
IPC APIs and the shared ``BaseItemLoader`` interface. Keeping it local lets
Zephon read Arrow chunks with older LitData installations that lack those
upstream helpers. IPC files and streams both use the shared ``_LitDataShard``
wrapper; binary pytree and token decoders live in ``litdata_support.pytree``.
"""

from __future__ import annotations

import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

import optree

from zephon._internal.io.formats.arrow_rows import require_pyarrow
from zephon._internal.io.formats.litdata_support.support import (
    BaseItemLoader,
    FlatPyTree,
)

_ARROW_FOOTER_MAGIC = b"LDARW01\0"
_ARROW_FOOTER_SIZE = 12  # uint32 IPC byte length followed by the eight-byte magic
_ARROW_BATCH_ROWS = 256  # LitData's _DEFAULT_BATCH_ROWS on-disk contract


def arrow_footer_span(path: Path, filesize_bytes: int = 0) -> tuple[int, int] | None:
    """Identify Arrow chunks, rejecting incomplete files before layout dispatch."""
    with path.open("rb") as source:
        size = source.seek(0, 2)
        if size < filesize_bytes:
            raise FileNotFoundError(f"Chunk file not found or incomplete: {path}")
        if size < _ARROW_FOOTER_SIZE:
            return None
        source.seek(-_ARROW_FOOTER_SIZE, 2)
        footer = source.read(_ARROW_FOOTER_SIZE)
    if footer[4:] != _ARROW_FOOTER_MAGIC:
        return None
    length = struct.unpack_from("<I", footer)[0]
    start = size - _ARROW_FOOTER_SIZE - length
    if length == 0 or start < 0:
        raise ValueError(f"Invalid LitData Arrow footer length in {path}")
    return start, length


class ArrowLoader(BaseItemLoader):
    """Read Arrow rows through the common LitData item-loader interface.

    IPC files use LitData's fixed batch size to decode only requested batches
    and retain one decoded batch. The chunk header and IPC batch count are
    checked on open; each requested batch's row count is checked on decode.
    IPC streams are read into a table because they lack random batch access.
    This reader does not depend on the binary pytree reader or its serializers.
    """

    def __init__(self) -> None:
        self._return_flat_leaves = False
        self._chunk_filepath: str | None = None
        self._length = 0
        self._source: Any = None
        self._ipc: Any = None
        self._reader: Any = None
        self._table: Any = None
        self._batch: Any = None
        self._batch_index = -1

    def _ensure_file_open(
        self, chunk_filepath: str, filesize_bytes: int, chunk_size: int
    ) -> None:
        if chunk_filepath == self._chunk_filepath:
            return
        path = Path(chunk_filepath)
        self.close(0)
        span = arrow_footer_span(path, filesize_bytes)
        if span is None:
            raise ValueError(f"Missing LitData Arrow footer in {path}")
        try:
            pa = require_pyarrow()
        except ImportError as exc:
            raise ImportError(
                "Arrow-backed LitData chunks require pyarrow. "
                + 'Install with: pip install "zephon[parquet]"'
            ) from exc
        self._length = chunk_size
        try:
            self._source = pa.memory_map(str(path), "r")
            start, size = span
            if start < 4:
                raise ValueError(f"Missing LitData chunk header in {path}")
            self._validate_length(struct.unpack("<I", self._source.read(4))[0])
            self._source.seek(start)
            self._ipc = self._source.read_buffer(size)
            if self._ipc.slice(0, 6).to_pybytes() == b"ARROW1":
                self._reader = pa.ipc.open_file(self._ipc)
                expected_batches = (
                    self._length + _ARROW_BATCH_ROWS - 1
                ) // _ARROW_BATCH_ROWS
                if self._reader.num_record_batches != expected_batches:
                    raise ValueError(
                        f"LitData Arrow batch count {self._reader.num_record_batches} "
                        + f"does not match expected {expected_batches} "
                        + f"for chunk_size={self._length}"
                    )
            else:
                with pa.ipc.open_stream(self._ipc) as reader:
                    self._table = reader.read_all()
                self._validate_length(self._table.num_rows)
            self._chunk_filepath = chunk_filepath
        except Exception:
            self.close(0)
            raise

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
    ) -> Any:
        return self.load_items_from_chunk(
            [index], chunk_index, chunk_filepath, begin, filesize_bytes
        )[0]

    def load_items_from_chunk(
        self,
        indices: list[int],
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
    ) -> list[Any]:
        if not indices:
            return []
        self._ensure_file_open(
            chunk_filepath, filesize_bytes, int(self._chunks[chunk_index]["chunk_size"])
        )
        rows = self._read_rows([index - begin for index in indices])
        if self._return_flat_leaves:
            # FlatPyTree is a shared output representation, not a storage decoder.
            rows = [
                FlatPyTree(*optree.tree_flatten(row, none_is_leaf=True)) for row in rows
            ]
        return rows

    def _validate_length(self, length: int) -> None:
        if length != self._length:
            raise ValueError(
                f"LitData Arrow row count {length} does not match "
                + f"chunk_size={self._length}"
            )

    def _get_batch(self, index: int) -> Any:
        if index != self._batch_index:
            batch = self._reader.get_batch(index)
            expected_rows = min(
                _ARROW_BATCH_ROWS, self._length - index * _ARROW_BATCH_ROWS
            )
            if batch.num_rows != expected_rows:
                raise ValueError(
                    f"LitData Arrow batch {index} has {batch.num_rows} rows; "
                    + f"expected {expected_rows} for {_ARROW_BATCH_ROWS}-row batches"
                )
            self._batch = batch
            self._batch_index = index
        return self._batch

    def _read_rows(self, indices: list[int]) -> list[Any]:
        """Return caller-owned rows, preserving request order and duplicates."""
        if self._table is not None:
            rows = self._table.take(indices).to_pylist()
        else:
            groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for position, index in enumerate(indices):
                batch_index, relative_index = divmod(index, _ARROW_BATCH_ROWS)
                groups[batch_index].append((position, relative_index))
            rows: list[Any] = [None] * len(indices)
            for batch_index, positions in groups.items():
                batch = self._get_batch(batch_index)
                selected = batch.take([index for _, index in positions]).to_pylist()
                for (position, _), row in zip(positions, selected, strict=True):
                    rows[position] = row
        return rows

    def close(self, chunk_index: int) -> None:
        self._batch = None
        self._table = None
        self._reader = None
        self._ipc = None
        if self._source is not None:
            self._source.close()
            self._source = None
        self._chunk_filepath = None
        self._batch_index = -1
