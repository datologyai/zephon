# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Convert Arrow columns into typed, read-only Zephon sample payloads."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Iterable, Mapping, cast

import numpy as np

from zephon._internal.utils.thread_utils import cap_arrow_threads

if TYPE_CHECKING:
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
    """Convert rows, sharing numeric buffers without concatenating chunks.

    A single column becomes the sample payload directly; multiple columns
    become dictionaries. Numeric arrays are read-only and keep their backing
    buffers alive. Each column may have different chunk boundaries.
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
    """Build payloads with the same shape for Arrow and native scalar reads."""
    # A range has no duplicates. Only reordered/repeated requests need tracking.
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
                # Numeric views may alias. Mutable Python fallbacks must not.
                samples[output] = _copy_containers(samples[output])
            seen.add(pos)
    return cast("list[SamplePayload]", samples)


def _copy_containers(value: Any) -> Any:
    """Copy fallback container structure while retaining immutable leaves/views."""
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
    """Expose numeric buffers and nested fixed-size shapes without Python boxing."""
    if array.null_count:
        return None
    kind = array.type
    if (
        pa.types.is_integer(kind)
        or pa.types.is_floating(kind)
        or pa.types.is_boolean(kind)
    ):
        values = array.to_numpy(zero_copy_only=False)
        # Boolean buffers need unpacking. Keep their mutability consistent with
        # zero-copy numeric views, including when duplicate rows share a view.
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
    """Return numeric rows as typed NumPy arrays where possible.

    Null-free numeric lists preserve dtype and fixed-size nested shapes, using
    views of Arrow buffers when possible; boolean buffers require unpacking.
    Views remain valid after the shard closes. Rows containing null elements
    or nested variable-length lists use Python lists to preserve None and
    nesting without coercing numeric values or creating NumPy object arrays.
    A null row stays None. Strings, bytes, and scalar booleans stay Python values.
    """
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
            # Filling only nullable scalar columns preserves dtype without NumPy's
            # usual integer-to-float/null coercion. Validity restores None per row.
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
        # Null parents can have null child slots even when every valid row is
        # numeric. Convert those rows independently rather than boxing them all.
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
    """Gather Parquet rows before exposing read-only views of selected buffers."""
    pa = require_pyarrow()
    selected = table.take(pa.array(local_indices, type=pa.int64()))
    return arrow_table_to_samples(selected)


__all__ = [
    "arrow_table_to_samples",
    "require_pyarrow",
    "samples_from_columns",
    "take_and_materialize",
]
