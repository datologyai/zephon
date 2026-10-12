# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Convert Arrow tables to Zephon sample payloads.

Conversion decisions, applied to each column chunk independently::

    Arrow value
    +-- Null -> None
    +-- Integer / float scalar -> NumPy scalar with the same dtype
    +-- List / ListView / fixed-size list
    |   +-- Integer / float / bool elements (including fixed-size nesting)
    |   |   +-- No null elements -> read-only NumPy array
    |   |   `-- Null elements -> Python list, preserving None
    |   `-- Other elements / variable-size nesting -> Python list
    `-- Other types -> Arrow's Python values (str, bytes, bool, dict, ...)

    Assemble each requested row
    +-- One column -> value directly
    `-- Multiple columns -> {column_name: value}

Chunks are converted separately to avoid concatenating numeric buffers. Numeric
arrays use read-only NumPy views where possible, avoiding Python objects for each
element. Views retain their backing buffers after reader closure; mutation needs
an explicit copy. Boolean unpacking, nullable scalar conversion and Python
fallbacks allocate. Parquet's row gather copies selected values before conversion;
Vortex supplies an indexed selection. Process transport may make a further SHM
copy. Duplicate rows share read-only arrays but own their mutable Python containers.
"""

from __future__ import annotations

import threading
from typing import Any, Iterable, Mapping, cast

import numpy as np

from zephon._internal.utils.thread_utils import cap_arrow_threads
from zephon.types import SamplePayload

ArrowColumns = Mapping[str, list[Any] | np.ndarray]
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
                        "pyarrow is required for Arrow row conversion. "
                        + "Install the matching zephon[parquet] or zephon[vortex] extra."
                    ) from exc
                cap_arrow_threads()
                _pa = pa
    return _pa


def arrow_table_to_samples(
    table: Any, positions: Iterable[int] | None = None
) -> list[SamplePayload]:
    """Convert selected table rows; omitted positions return all rows.

    Columns may have different chunk boundaries.
    """
    pa = require_pyarrow()
    columns: dict[str, list[Any] | np.ndarray] = {}
    for name in table.column_names:
        column = table.column(name)
        if column.num_chunks == 1:
            columns[name] = _column_values(column.chunk(0), pa)
        else:
            columns[name] = [
                value for chunk in column.chunks for value in _column_values(chunk, pa)
            ]
    return samples_from_columns(
        columns, range(table.num_rows) if positions is None else positions
    )


def samples_from_columns(
    columns: ArrowColumns, positions: Iterable[int]
) -> list[SamplePayload]:
    """Assemble payloads in position order with independent mutable containers."""
    # Ranges cannot repeat positions.
    if not isinstance(positions, range):
        positions = list(positions)
    if len(columns) == 1:
        values = next(iter(columns.values()))
        samples = [values[pos] for pos in positions]
    else:
        samples = [
            {name: values[pos] for name, values in columns.items()} for pos in positions
        ]
    if not isinstance(positions, range):
        seen: set[int] = set()
        for output, pos in enumerate(positions):
            if pos in seen:
                samples[output] = _copy_containers(samples[output])
            seen.add(pos)
    return cast("list[SamplePayload]", samples)


def _copy_containers(value: Any) -> Any:
    """Copy container structure, retaining array views and scalar values."""
    if isinstance(value, dict):
        return {key: _copy_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_containers(item) for item in value)
    return value


def _is_numeric_type(kind: Any, pa: Any) -> bool:
    return (
        pa.types.is_integer(kind)
        or pa.types.is_floating(kind)
        or pa.types.is_boolean(kind)
        or (pa.types.is_fixed_size_list(kind) and _is_numeric_type(kind.value_type, pa))
    )


def _numeric_values(array: Any, pa: Any) -> np.ndarray | None:
    """Return a read-only numeric array, or None for nulls or unsupported types."""
    if array.null_count:
        return None
    kind = array.type
    if (
        pa.types.is_integer(kind)
        or pa.types.is_floating(kind)
        or pa.types.is_boolean(kind)
    ):
        values = array.to_numpy(zero_copy_only=False)
        # Unpacked boolean buffers must also be read-only.
        values.setflags(write=False)
        return values
    if pa.types.is_fixed_size_list(kind):
        size = kind.list_size
        child = array.values.slice(array.offset * size, len(array) * size)
        values = _numeric_values(child, pa)
        if values is not None:
            return values.reshape((len(array), size, *values.shape[1:]))
    return None


def _column_values(array: Any, pa: Any) -> list[Any] | np.ndarray:
    """Convert one Arrow chunk to per-row values."""
    kind = array.type
    is_fixed = pa.types.is_fixed_size_list(kind)
    is_view = pa.types.is_list_view(kind) or pa.types.is_large_list_view(kind)
    is_list = (
        is_fixed or is_view or pa.types.is_list(kind) or pa.types.is_large_list(kind)
    )
    if not is_list:
        if pa.types.is_integer(kind) or pa.types.is_floating(kind):
            if not array.null_count:
                return array.to_numpy(zero_copy_only=False)
            # Fill before conversion to preserve dtype; restore nulls per row.
            numbers = array.fill_null(0).to_numpy(zero_copy_only=False)
            valid = array.is_valid().to_numpy(zero_copy_only=False)
            return [
                number if present else None for number, present in zip(numbers, valid)
            ]
        return array.to_pylist()

    if not _is_numeric_type(kind.value_type, pa):
        return array.to_pylist()

    numbers = _numeric_values(array.values, pa)
    if numbers is None:
        # Nulls elsewhere in the chunk need not force Python lists for every row.
        rows: list[Any] = []
        for scalar in array:
            if not scalar.is_valid:
                rows.append(None)
                continue
            values = _numeric_values(scalar.values, pa)
            rows.append(values if values is not None else scalar.as_py())
        return rows

    if is_fixed and not array.null_count:
        start = array.offset * kind.list_size
        return numbers[start : start + len(array) * kind.list_size].reshape(
            (len(array), kind.list_size, *numbers.shape[1:])
        )
    valid = (
        array.is_valid().to_numpy(zero_copy_only=False) if array.null_count else None
    )
    if is_fixed:
        starts = (np.arange(len(array)) + array.offset) * kind.list_size
        ends = starts + kind.list_size
    else:
        offsets = array.offsets.to_numpy(zero_copy_only=False)
        starts = offsets if is_view else offsets[:-1]
        ends = (
            starts + array.sizes.to_numpy(zero_copy_only=False)
            if is_view
            else offsets[1:]
        )
    return [
        numbers[start:end] if valid is None or valid[row] else None
        for row, (start, end) in enumerate(zip(starts, ends))
    ]


def take_and_materialize(
    table: Any,
    local_indices: list[int],
) -> list[SamplePayload]:
    """Gather requested rows, then convert them to sample payloads."""
    pa = require_pyarrow()
    selected = table.take(pa.array(local_indices, type=pa.int64()))
    return arrow_table_to_samples(selected)


__all__ = [
    "arrow_table_to_samples",
    "require_pyarrow",
    "samples_from_columns",
    "take_and_materialize",
]
