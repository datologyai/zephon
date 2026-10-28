"""Payload sizing for observability and data-volume estimates.

:func:`content_bytes` measures decoded payload bytes only: no Python object
overhead, no mapping keys, and no sample metadata, so totals line up with
on-disk bytes. It backs per-node produced/consumed byte metrics,
:meth:`Dataset.raw_bytes`, and the bytes-per-token ratios in token estimation.

Sizing runs on the hot path of every runner when byte metrics are enabled, so
it dispatches on exact type through a per-type handler table and sizes
homogeneous containers with C-level ``map`` passes instead of recursing into
every element. Lazy SHM payloads are sized from their descriptors without
being materialized.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

import numpy as np

from zephon._internal.stream import LazyPayload
from zephon.types import SampleBatch, SampleRecord

logger = logging.getLogger(__name__)

_SCALAR_BYTES = 8
_SCALAR_TYPES = frozenset({int, float, bool})

_Handler = Callable[[Any], int]


def _str_bytes(s: str) -> int:
    # ``isascii`` is O(1) on CPython's compact strings, so ASCII text (the common
    # case) skips the UTF-8 encode copy. Lone surrogates are dropped, not raised.
    if s.isascii():
        return len(s)
    return len(s.encode("utf-8", errors="ignore"))


def _values_bytes(values: Collection[Any]) -> int:
    """Total bytes of a re-iterable collection of values."""
    if not values:
        return 0
    types = set(map(type, values))
    if len(types) == 1:
        (cls,) = types
        if cls in _SCALAR_TYPES:
            return _SCALAR_BYTES * len(values)
        if cls is str:
            if all(map(str.isascii, values)):
                return sum(map(len, values))
            return sum(map(_str_bytes, values))
        if cls is bytes:
            return sum(map(len, values))
        handler = _HANDLERS.get(cls)
        if handler is not None:
            return sum(map(handler, values))
    elif types <= _SCALAR_TYPES:
        return _SCALAR_BYTES * len(values)
    return sum(map(content_bytes, values))


def _mapping_bytes(obj: Mapping[Any, Any]) -> int:
    return _values_bytes(obj.values())


def _record_bytes(record: SampleRecord) -> int:
    return content_bytes(record.payload)


def _batch_bytes(batch: SampleBatch) -> int:
    return sum(content_bytes(r.payload) for r in batch.records)


def _scalar_bytes(_obj: Any) -> int:
    return _SCALAR_BYTES


def _zero_bytes(_obj: Any) -> int:
    return 0


_HANDLERS: dict[type, _Handler] = {
    str: _str_bytes,
    bytes: len,
    bytearray: len,
    memoryview: lambda mv: mv.nbytes,
    np.ndarray: lambda arr: int(arr.nbytes),
    dict: _mapping_bytes,
    list: _values_bytes,
    tuple: _values_bytes,
    SampleRecord: _record_bytes,
    SampleBatch: _batch_bytes,
    # None carries no payload; this also matches lazy SHM payloads, whose
    # pytree flattening drops None leaves.
    type(None): _zero_bytes,
    **dict.fromkeys(_SCALAR_TYPES, _scalar_bytes),
}


def _struct_field_names(cls: type) -> tuple[str, ...] | None:
    """Return field names for dataclass/pydantic/attrs classes, else ``None``."""
    if dataclasses.is_dataclass(cls):
        # Match SHM transport (``shm_coalesce._struct_info``), which ships only
        # init fields, so lazy and eager sizing agree.
        return tuple(f.name for f in dataclasses.fields(cls) if f.init)
    if hasattr(cls, "model_fields") and hasattr(cls, "model_construct"):
        return tuple(cls.model_fields.keys())  # pydantic
    if hasattr(cls, "__attrs_attrs__"):
        return tuple(a.name for a in cls.__attrs_attrs__)  # attrs
    return None


def _struct_handler(names: tuple[str, ...]) -> _Handler:
    def handler(obj: Any) -> int:
        return sum(content_bytes(getattr(obj, name, None)) for name in names)

    return handler


def _resolve_handler(cls: type, obj: Any) -> _Handler:
    """Pick a handler for a type outside the exact-type table."""
    if issubclass(cls, LazyPayload):
        return lambda payload: payload.payload_nbytes()
    if issubclass(cls, str):
        return _str_bytes
    if issubclass(cls, (bytes, bytearray)):
        return len
    if issubclass(cls, (np.ndarray, np.generic)):
        return lambda arr: int(arr.nbytes)
    torch = sys.modules.get("torch")
    if torch is not None and issubclass(cls, torch.Tensor):
        return lambda t: int(t.element_size() * t.nelement())
    if issubclass(cls, Mapping):
        return _mapping_bytes
    if issubclass(cls, Sequence):
        return lambda seq: sum(map(content_bytes, seq))
    names = _struct_field_names(cls)
    if names is not None:
        return _struct_handler(names)
    if issubclass(cls, (int, float)):
        return _scalar_bytes
    pil_image = sys.modules.get("PIL.Image")
    if pil_image is not None and issubclass(cls, pil_image.Image):
        return _pil_image_bytes
    if _has_int_nbytes(obj):
        return lambda arr: int(arr.nbytes)  # pyarrow, cupy, jax, ... arrays
    if hasattr(obj, "__bytes__") and hasattr(obj, "__len__"):
        return len  # SHM-backed bytes views
    return _zero_bytes


def _has_int_nbytes(obj: Any) -> bool:
    try:
        nbytes = obj.nbytes
    except Exception:
        return False
    return isinstance(nbytes, int) and not isinstance(nbytes, bool)


@functools.lru_cache(maxsize=None)
def _pil_pixel_bytes(mode: str) -> int:
    from PIL import ImageMode

    desc = ImageMode.getmode(mode)
    return np.dtype(desc.typestr).itemsize * len(desc.bands)


def _pil_image_bytes(img: Any) -> int:
    # Decoded pixel bytes, matching ``content_bytes(np.asarray(img))``.
    width, height = img.size
    return width * height * _pil_pixel_bytes(img.mode)


def content_bytes(obj: Any) -> int:
    """Content-only payload bytes, excluding object overhead and mapping keys.

    Records and batches count only their payloads. No cycle guard: payloads
    are expected to be acyclic. Runner metrics use :func:`metric_bytes`, which
    never raises.
    """
    cls = type(obj)
    handler = _HANDLERS.get(cls)
    if handler is None:
        handler = _HANDLERS[cls] = _resolve_handler(cls, obj)
    return handler(obj)


# Runner byte metrics size one in ``_SAMPLE_STRIDE`` records per batch and scale
# up. Sampling inside every batch keeps the estimate unbiased when record sizes
# drift over a run, and per-record variance averages out over the many batches
# a metrics total covers. Batches under ``_SAMPLE_MIN_RECORDS`` are sized exactly.
_SAMPLE_STRIDE = 4
_SAMPLE_MIN_RECORDS = 2 * _SAMPLE_STRIDE
# Rotate the first sampled index so size patterns aligned with the stride (e.g.
# alternating short/long records) do not bias the estimate. Races between
# threads only perturb the rotation, which is harmless.
_sample_offset = 0


def sampled_content_bytes(items: Sequence[Any]) -> int:
    """Estimate :func:`content_bytes` of a runner microbatch by stride sampling.

    ``SampleBatch`` items are flattened into their records so packed batches
    sample at record granularity.
    """
    global _sample_offset
    units: Sequence[Any] = items
    if any(type(item) is SampleBatch for item in items):
        units = [
            rec
            for item in items
            for rec in (item.records if type(item) is SampleBatch else (item,))
        ]
    n = len(units)
    if n < _SAMPLE_MIN_RECORDS:
        return content_bytes(units)
    offset = _sample_offset % _SAMPLE_STRIDE
    _sample_offset = offset + 1
    sample = units[offset::_SAMPLE_STRIDE]
    return round(sum(map(content_bytes, sample)) * n / len(sample))


_metric_failure_logged = False


def metric_bytes(items: Sequence[Any]) -> int:
    """:func:`sampled_content_bytes` for runner byte metrics; never raises.

    Byte metrics are best-effort, so a batch that cannot be sized (cyclic,
    too deeply nested, or a type whose size accessor fails) counts as 0
    instead of failing the batch. The first failure is logged.
    """
    try:
        return sampled_content_bytes(items)
    except Exception:
        global _metric_failure_logged
        if not _metric_failure_logged:
            _metric_failure_logged = True
            logger.warning(
                "Could not size a payload for byte metrics; counting it as 0 bytes",
                exc_info=True,
            )
        return 0
