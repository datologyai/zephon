"""Utility helpers for estimating the size of data structures.

Two entry points with different semantics:

- :func:`estimate_bytes` approximates a value's in-memory footprint (object
  overhead + mapping keys, cycle-safe). Use it for observability metrics such
  as per-invocation consumed/produced bytes.
- :func:`content_bytes` measures decoded payload bytes only (no object
  overhead, no mapping keys, no cycle guard). Use it for data-volume sizing
  that should line up with on-disk bytes, such as :meth:`Dataset.raw_bytes`.
"""

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


def _leaf_bytes(obj: Any) -> int | None:
    """Leaf byte size shared by estimate_bytes and content_bytes."""
    if isinstance(obj, str):
        return len(obj.encode("utf-8", errors="ignore"))
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return len(obj)
    if _np is not None and isinstance(obj, _np.ndarray):
        return int(obj.nbytes)
    _t = _lazy_torch()
    if _t is not None and isinstance(obj, _t.Tensor):
        return int(obj.element_size() * obj.nelement())
    return None


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
    leaf = _leaf_bytes(obj)
    if leaf is not None:
        return leaf
    if isinstance(obj, Mapping):
        return _estimate_mapping_bytes(obj, visited)
    if isinstance(obj, Sequence):  # str/bytes/bytearray handled as leaves above
        return _estimate_sequence_bytes(obj, visited)
    field_names = _struct_field_names(type(obj))
    if field_names is not None:
        return _estimate_struct_bytes(obj, field_names, visited)

    try:
        return sys.getsizeof(obj)
    except TypeError:  # pragma: no cover - fallback for objects without __sizeof__
        return 0


def content_bytes(obj: Any) -> int:
    """Content-only payload bytes, excluding object overhead and mapping keys.

    No cycle guard: payloads are expected to be acyclic.
    """
    leaf = _leaf_bytes(obj)
    if leaf is not None:
        return leaf
    if isinstance(obj, Mapping):
        return sum(content_bytes(v) for v in obj.values())
    if isinstance(obj, Sequence):  # str/bytes handled as leaves above
        return sum(content_bytes(item) for item in obj)
    field_names = _struct_field_names(type(obj))
    if field_names is not None:
        return sum(content_bytes(getattr(obj, name, None)) for name in field_names)
    if isinstance(obj, (int, float, bool)) or obj is None:
        return 8
    return 0
