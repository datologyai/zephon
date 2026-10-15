# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Convert Arrow tables and arrays to Zephon sample payloads.

Conversion decisions, applied to each column chunk independently::

    Arrow value
    +-- Null -> None
    +-- Integer / float scalar -> NumPy scalar with the same dtype
    +-- Timestamp / date / duration scalar -> NumPy datetime64 / timedelta64
    +-- List / ListView / fixed-size list
    |   +-- Integer / float / bool elements (including fixed-size nesting)
    |   |   +-- No null elements -> read-only NumPy array
    |   |   `-- Null elements -> Python list, preserving None
    |   `-- Other elements / variable-size nesting -> Python list
    `-- Other types -> Arrow's Python values (str, bytes, bool, dict, ...)

    Assemble each requested row -> {column_name: value}
    Non-struct root arrays -> bare values with the same conversion rules

Chunks are converted separately to avoid concatenating numeric buffers. Numeric
arrays use read-only NumPy views where possible, avoiding Python objects for each
element. Views retain their backing buffers after reader closure; mutation needs
an explicit copy. Boolean unpacking, nullable scalar conversion and Python
fallbacks allocate. Parquet's row gather copies selected values before conversion;
Vortex supplies an indexed selection. Process transport may make a further SHM
copy. Duplicate rows share read-only arrays but own their mutable Python containers.
"""

from __future__ import annotations

import sys
import threading
from typing import Any, Iterable, Mapping

import numpy as np

from zephon._internal.utils.thread_utils import configure_arrow_threads
from zephon.io.options import ArrowOptions
from zephon.types import SamplePayload

ArrowColumns = Mapping[str, list[Any] | np.ndarray]
_pa: Any | None = None
_pa_lock = threading.Lock()
_arrow_options: ArrowOptions | None = None


def configure_pyarrow(options: ArrowOptions) -> None:
    """Apply store pool settings now, or on first use if Arrow is not loaded.

    LitData's binary layouts do not need Arrow. Retaining the process-wide
    policy here lets Arrow layouts initialize lazily without importing an
    optional dependency just to configure it. Later stores replace the policy.
    """
    global _arrow_options, _pa
    with _pa_lock:
        pa = _pa if _pa is not None else sys.modules.get("pyarrow")
        if pa is not None:
            configure_arrow_threads(pa, options)
            _pa = pa
        _arrow_options = options


def require_pyarrow() -> Any:
    """Return PyArrow, applying a reader's pending pool configuration on import."""
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
                if _arrow_options is not None:
                    configure_arrow_threads(pa, _arrow_options)
                _pa = pa
    return _pa


def arrow_table_to_samples(
    table: Any, positions: Iterable[int] | None = None
) -> list[SamplePayload]:
    """Convert selected table rows; omitted positions return all rows.

    Columns may have different chunk boundaries.
    """
    pa = require_pyarrow()
    columns = {
        name: _column_values(table.column(name), pa) for name in table.column_names
    }
    return samples_from_columns(
        columns, range(table.num_rows) if positions is None else positions
    )


def arrow_array_to_samples(
    array: Any, positions: Iterable[int] | None = None
) -> list[SamplePayload]:
    """Return struct rows as dictionaries and other root values as bare payloads."""
    pa = require_pyarrow()
    if pa.types.is_struct(array.type):
        return arrow_table_to_samples(pa.Table.from_struct_array(array), positions)
    values = _column_values(array, pa)
    if positions is None:
        positions = range(len(array))
    elif not isinstance(positions, range):
        positions = list(positions)
    return _copy_repeated_containers([values[pos] for pos in positions], positions)


def samples_from_columns(
    columns: ArrowColumns, positions: Iterable[int]
) -> list[SamplePayload]:
    """Assemble payloads in position order with independent mutable containers."""
    # Ranges cannot repeat positions.
    if not isinstance(positions, range):
        positions = list(positions)
    # TODO(MaxiBoether): Consider unwrapping single-column payloads once packing
    # and other downstream ops support bare values.
    samples: list[SamplePayload] = [
        {name: values[pos] for name, values in columns.items()} for pos in positions
    ]
    return _copy_repeated_containers(samples, positions)


def _copy_repeated_containers(
    samples: list[SamplePayload], positions: Iterable[int]
) -> list[SamplePayload]:
    """Keep duplicate samples' mutable containers independent without copying arrays."""
    if not isinstance(positions, range):
        seen: set[int] = set()
        for output, pos in enumerate(positions):
            if pos in seen:
                samples[output] = _copy_containers(samples[output])
            seen.add(pos)
    return samples


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


def _numpy_dtype(kind: Any, pa: Any) -> np.dtype:
    """Return the NumPy dtype of an Arrow integer or floating-point type."""
    if pa.types.is_floating(kind):
        code = "f"
    elif pa.types.is_signed_integer(kind):
        code = "i"
    else:
        code = "u"
    return np.dtype(f"{code}{kind.bit_width // 8}")


def _numeric_values(array: Any, pa: Any) -> np.ndarray | None:
    """Return a read-only numeric array, or None for nulls or unsupported types."""
    if array.null_count:
        return None
    kind = array.type
    if pa.types.is_integer(kind) or pa.types.is_floating(kind):
        # View the data buffer directly. ``to_numpy`` goes through pyarrow's
        # pandas writer, which releases the GIL; with many reading threads,
        # taking it back costs more than the conversion.
        values = np.frombuffer(
            array.buffers()[1] or b"",
            dtype=_numpy_dtype(kind, pa),
            count=len(array),
            offset=array.offset * kind.bit_width // 8,
        )
        values.setflags(write=False)
        return values
    if pa.types.is_boolean(kind):
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


def _lists_for_python(array: Any, pa: Any) -> Any:
    """Compact list views into ordinary lists for Arrow's Python conversion."""
    kind = array.type
    mask = array.is_null() if array.null_count else None
    if pa.types.is_struct(kind):
        children = [_lists_for_python(array.field(i), pa) for i in range(len(kind))]
        if all(child.type == field.type for child, field in zip(children, kind)):
            return array
        return pa.StructArray.from_arrays(
            children,
            fields=[
                field.with_type(child.type) for field, child in zip(kind, children)
            ],
            mask=mask,
        )
    if pa.types.is_list_view(kind) or pa.types.is_large_list_view(kind):
        # Do not cast to list: PyArrow 25.0.1 corrupts nonmonotonic views.
        # Offsets [2, 0, 1], sizes [2, 2, 2] over [a,b,c,d] cast to [[],[a],[]].
        # Flatten follows sizes and offsets, including gaps, overlaps and nulls.
        values = _lists_for_python(array.flatten(), pa)
        sizes = array.sizes.to_numpy(zero_copy_only=False)
        if mask is not None:
            sizes = np.where(mask.to_numpy(zero_copy_only=False), 0, sizes)
        offsets = np.empty(len(array) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(sizes, dtype=np.int64, out=offsets[1:])
        return pa.LargeListArray.from_arrays(
            offsets,
            values,
            type=pa.large_list(kind.value_field.with_type(values.type)),
            mask=mask,
        )
    if pa.types.is_list(kind) or pa.types.is_large_list(kind):
        values = _lists_for_python(array.values, pa)
        if values.type == kind.value_type:
            return array
        large = pa.types.is_large_list(kind)
        return (pa.LargeListArray if large else pa.ListArray).from_arrays(
            array.offsets,
            values,
            type=(pa.large_list if large else pa.list_)(
                kind.value_field.with_type(values.type)
            ),
            mask=mask,
        )
    if pa.types.is_fixed_size_list(kind):
        values = _lists_for_python(
            array.values.slice(
                array.offset * kind.list_size, len(array) * kind.list_size
            ),
            pa,
        )
        if values.type == kind.value_type:
            return array
        return pa.FixedSizeListArray.from_arrays(
            values,
            type=pa.list_(kind.value_field.with_type(values.type), kind.list_size),
            mask=mask,
        )
    return array


def _column_values(array: Any, pa: Any) -> list[Any] | np.ndarray:
    """Convert Arrow chunks independently to avoid concatenating their buffers."""
    if isinstance(array, pa.ChunkedArray):
        if array.num_chunks == 1:
            array = array.chunk(0)
        else:
            return [
                value for chunk in array.chunks for value in _column_values(chunk, pa)
            ]
    kind = array.type
    is_fixed = pa.types.is_fixed_size_list(kind)
    is_view = pa.types.is_list_view(kind) or pa.types.is_large_list_view(kind)
    is_list = (
        is_fixed or is_view or pa.types.is_list(kind) or pa.types.is_large_list(kind)
    )
    if not is_list:
        if (
            pa.types.is_timestamp(kind)
            or pa.types.is_date(kind)
            or pa.types.is_duration(kind)
        ):
            # Python conversion can require pandas or lose nanosecond precision.
            values = array.to_numpy(zero_copy_only=False)
            if not array.null_count:
                return values
            valid = array.is_valid().to_numpy(zero_copy_only=False)
            return [value if present else None for value, present in zip(values, valid)]
        if pa.types.is_integer(kind) or pa.types.is_floating(kind):
            if not array.null_count:
                return array.to_numpy(zero_copy_only=False)
            # Fill before conversion to preserve dtype; restore nulls per row.
            numbers = array.fill_null(0).to_numpy(zero_copy_only=False)
            valid = array.is_valid().to_numpy(zero_copy_only=False)
            return [
                number if present else None for number, present in zip(numbers, valid)
            ]
        if pa.types.is_struct(kind):
            array = _lists_for_python(array, pa)
        return array.to_pylist()

    if not _is_numeric_type(kind.value_type, pa):
        # Arrow converts ordinary lists to Python much faster than list views.
        return _lists_for_python(array, pa).to_pylist()

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
