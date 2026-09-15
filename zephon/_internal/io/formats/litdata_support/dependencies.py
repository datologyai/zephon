"""Lazy access to optional LitData and Torch dependencies.

This module owns LitData's serializer registry, dtype mappings, and thread-safe
initialization. Binary pytree and token readers use these dependencies; token
estimation also initializes them before starting concurrent reads. Arrow payloads
use PyArrow independently and do not require this initialization.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional

# ---------------------------------------------------------------------------
# Lazy imports — torch and litdata internals
#
# litdata.constants (third-party) does ``import torch`` at module level, and
# litdata.streaming.serializers transitively imports it too.  Importing *any*
# litdata submodule therefore pulls in torch (~700 MB RSS).  In worker
# processes that register the litdata format but never open a shard, this is
# pure waste.
#
# We defer ALL litdata (and torch) imports to ``ensure_litdata_deps()``,
# which runs on first call to ``_get_serializers()`` — i.e. at shard-open
# time, not at format-registration time.
#
# ``_ensure_torch()`` provides standalone lazy access to the torch module
# for ``TokensLoader``, which needs ``torch.frombuffer`` / ``torch.empty``.
# Once ``ensure_litdata_deps()`` has run torch is already in sys.modules,
# so ``_ensure_torch()`` is effectively a dict lookup at that point.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from litdata.streaming.serializers import (
        NoHeaderNumpySerializer,
        NoHeaderTensorSerializer,
        PILSerializer,
        Serializer,
    )

# -- Lazy torch (used only by TokensLoader) --------------------------------

_torch_mod = None


def _ensure_torch():  # pragma: no cover - optional dependency
    """Import torch on first use.  Returns the torch module."""
    global _torch_mod
    if _torch_mod is None:
        try:
            import torch

            _torch_mod = torch
        except ImportError:
            raise ImportError("PyTorch is required for the TokensLoader")
    return _torch_mod


# -- Lazy litdata deps (serializers + dtype mappings) -----------------------

_litdata_deps_ready = False
_litdata_deps_lock = threading.Lock()

_SERIALIZERS: OrderedDict[str, Any] = OrderedDict()
_NUMPY_DTYPES_REVERSE: dict[Any, int] = {}
_TORCH_DTYPES_MAPPING: dict[int, Any] = {}
_NUMPY_DTYPES_MAPPING: dict[int, Any] = {}


def _load_litdata_deps() -> None:
    global _litdata_deps_ready

    from litdata.constants import _NUMPY_DTYPES_MAPPING as _ndm
    from litdata.constants import _TORCH_DTYPES_MAPPING as _tdm
    from litdata.streaming.serializers import _SERIALIZERS as _litdata_serializers
    from litdata.streaming.serializers import (
        NoHeaderNumpySerializer,
        NoHeaderTensorSerializer,
        PILSerializer,
        Serializer,
    )

    _SERIALIZERS.update(_litdata_serializers)
    _NUMPY_DTYPES_MAPPING.update(_ndm)
    _TORCH_DTYPES_MAPPING.update(_tdm)
    _NUMPY_DTYPES_REVERSE.update({dtype: idx for idx, dtype in _ndm.items()})
    globals().update(
        {
            "NoHeaderNumpySerializer": NoHeaderNumpySerializer,
            "NoHeaderTensorSerializer": NoHeaderTensorSerializer,
            "PILSerializer": PILSerializer,
            "Serializer": Serializer,
        }
    )
    _litdata_deps_ready = True


def ensure_litdata_deps() -> None:
    """Load LitData's serializer and dtype modules exactly once."""
    if _litdata_deps_ready:
        return
    with _litdata_deps_lock:
        if not _litdata_deps_ready:
            _load_litdata_deps()


def __getattr__(name: str) -> Any:
    """Module-level __getattr__ (PEP 562) for lazy litdata type access.

    Triggers ``ensure_litdata_deps()`` so that ``from litdata_support.dependencies
    import Serializer`` works without eagerly importing litdata/torch.
    """
    ensure_litdata_deps()
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def _get_serializers(
    overrides: Optional[Mapping[str, Serializer]] = None,
) -> dict[str, Serializer]:
    """Return serializer instances, allowing overrides for testing."""
    ensure_litdata_deps()
    serializers: OrderedDict[str, Serializer] = OrderedDict(_SERIALIZERS)
    if overrides:
        for key, value in overrides.items():
            serializers[key] = value
    return serializers


__all__ = [
    "ensure_litdata_deps",
    "_ensure_torch",
    "_get_serializers",
    "NoHeaderNumpySerializer",
    "NoHeaderTensorSerializer",
    "PILSerializer",
    "Serializer",
]
