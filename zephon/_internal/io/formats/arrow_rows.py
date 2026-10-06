# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Turn selected Arrow table rows into caller-owned Zephon row dictionaries."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from zephon._internal.utils.thread_utils import cap_arrow_threads

ArrowColumns = dict[str, np.ndarray]
_pa: Any | None = None
_pa_lock = threading.Lock()


def require_pyarrow() -> Any:
    """Return PyArrow after applying Zephon's thread cap once."""
    global _pa
    if _pa is None:
        with _pa_lock:
            if _pa is None:
                try:
                    import pyarrow as pa
                except ImportError as exc:
                    raise ImportError(
                        "pyarrow is required for Parquet format support. "
                        + 'Install with: pip install "zephon[parquet]"'
                    ) from exc
                cap_arrow_threads()
                _pa = pa
    return _pa


def arrow_table_to_numpy(table: Any) -> tuple[ArrowColumns, int]:
    """Convert an Arrow table to per-column NumPy arrays."""
    pa = require_pyarrow()
    decoded_bytes = table.nbytes
    result: ArrowColumns = {}
    for name in table.column_names:
        column = table.column(name)
        array = column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()
        column_type = array.type

        if isinstance(column_type, pa.lib.FixedSizeListType):
            flat = array.values.to_numpy(zero_copy_only=False, writable=True)
            values = flat.reshape(len(array), column_type.list_size)
            if array.null_count == 0:
                result[name] = values
            else:
                valid = array.is_valid().to_numpy(zero_copy_only=False)
                rows = np.empty(len(array), dtype=object)
                for index in range(len(array)):
                    rows[index] = values[index] if valid[index] else None
                result[name] = rows
        elif isinstance(column_type, pa.lib.ListType):
            offsets = array.offsets.to_numpy(zero_copy_only=False)
            values = array.values.to_numpy(zero_copy_only=False, writable=True)
            valid = (
                array.is_valid().to_numpy(zero_copy_only=False)
                if array.null_count
                else None
            )
            rows = np.empty(len(array), dtype=object)
            for index in range(len(array)):
                rows[index] = (
                    values[offsets[index] : offsets[index + 1]]
                    if valid is None or valid[index]
                    else None
                )
            result[name] = rows
        else:
            try:
                result[name] = array.to_numpy(zero_copy_only=False)
            except Exception:
                result[name] = np.array(array.to_pylist(), dtype=object)
    return result, decoded_bytes


def extract_row(columns: ArrowColumns, index: int) -> dict[str, object]:
    """Materialize one row as scalars or views into call-owned buffers."""
    return {name: array[index] for name, array in columns.items()}


def take_and_materialize(
    table: Any,
    local_indices: list[int],
) -> list[dict[str, object]]:
    """Select requested Arrow rows and return values independent of the table."""
    pa = require_pyarrow()
    selected = table.take(pa.array(local_indices, type=pa.int64()))
    columns, _ = arrow_table_to_numpy(selected)
    return [extract_row(columns, index) for index in range(len(local_indices))]


__all__ = [
    "ArrowColumns",
    "arrow_table_to_numpy",
    "extract_row",
    "require_pyarrow",
    "take_and_materialize",
]
