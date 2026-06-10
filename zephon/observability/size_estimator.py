"""Utility helpers for estimating the size of data structures."""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Mapping, Sequence
from typing import Any

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy may be optional
    _np = None  # type: ignore[assignment]

_torch_mod = None
_torch_checked = False


def _lazy_torch():  # pragma: no cover - torch may or may not be installed
    global _torch_mod, _torch_checked
    if not _torch_checked:
        try:
            import torch

            _torch_mod = torch
        except Exception:
            pass
        _torch_checked = True
    return _torch_mod


def _estimate_sequence_bytes(obj: Sequence[Any], visited: set[int]) -> int:
    size = sys.getsizeof(obj)
    for item in obj:
        size += estimate_bytes(item, visited)
    return size


_struct_field_names_cache: dict[type, tuple[str, ...] | None] = {}


def _struct_field_names(cls: type) -> tuple[str, ...] | None:
    """Return field names for dataclass/pydantic/attrs classes, else ``None``."""
    try:
        return _struct_field_names_cache[cls]
    except KeyError:
        pass
    names: tuple[str, ...] | None = None
    if dataclasses.is_dataclass(cls):
        names = tuple(f.name for f in dataclasses.fields(cls))
    elif hasattr(cls, "model_fields") and hasattr(cls, "model_construct"):
        names = tuple(cls.model_fields.keys())  # pydantic
    elif hasattr(cls, "__attrs_attrs__"):
        names = tuple(a.name for a in cls.__attrs_attrs__)  # attrs
    _struct_field_names_cache[cls] = names
    return names


def _estimate_struct_bytes(
    obj: Any, field_names: tuple[str, ...], visited: set[int]
) -> int:
    size = sys.getsizeof(obj)
    for name in field_names:
        size += estimate_bytes(getattr(obj, name, None), visited)
    return size


def _estimate_mapping_bytes(obj: Mapping[Any, Any], visited: set[int]) -> int:
    size = sys.getsizeof(obj)
    for key, value in obj.items():
        size += estimate_bytes(key, visited)
        size += estimate_bytes(value, visited)
    return size


def estimate_bytes(obj: Any, visited: set[int] | None = None) -> int:
    """Best-effort estimate of allocated bytes for ``obj``.

    The estimate is intentionally coarse but stable across processes, which is
    suitable for relative comparisons in observability metrics.
    """
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return 0
    visited.add(obj_id)

    if obj is None:
        return 0
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return len(obj)
    if isinstance(obj, str):
        return len(obj.encode("utf-8"))
    if _np is not None and isinstance(obj, _np.ndarray):
        return int(obj.nbytes)
    _t = _lazy_torch()
    if _t is not None and isinstance(obj, _t.Tensor):
        return int(obj.element_size() * obj.nelement())
    if isinstance(obj, Mapping):
        return _estimate_mapping_bytes(obj, visited)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return _estimate_sequence_bytes(obj, visited)
    field_names = _struct_field_names(type(obj))
    if field_names is not None:
        return _estimate_struct_bytes(obj, field_names, visited)

    try:
        return sys.getsizeof(obj)
    except TypeError:  # pragma: no cover - fallback for objects without __sizeof__
        return 0
