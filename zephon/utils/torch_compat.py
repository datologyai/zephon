"""Helper module for PyTorch compatibility utilities.

Includes:
- Dataloader detection (torch vs torchdata)
- Free-threaded Python tensor lock utilities (PyTorch allocator race workaround)
"""

from __future__ import annotations

import contextlib
import inspect
import sys
import sysconfig
import threading

_TORCHDATA_HINTS = (
    "torchdata.stateful_dataloader",  # e.g. 'torchdata.stateful_dataloader.worker'
    "torchdata._stateful_dataloader",  # defensive (alt/private layouts)
)
_TORCH_VANILLA_HINT = "torch.utils.data"  # e.g. 'torch.utils.data._utils.worker'


def detect_loader_kind() -> str:
    """Decides which dataloader we are using.

    Options:
      - 'torchdata'  (StatefulDataLoader worker)
      - 'vanilla'    (torch.utils.data.DataLoader worker or main-thread)
      - 'unknown'
    Heuristic: scan the whole call stack; torchdata wins if present anywhere.
    """
    saw_torchdata = False
    saw_vanilla = False

    try:
        for fi in inspect.stack():
            fn = (fi.filename or "").replace("\\", "/")
            fr = getattr(fi, "frame", None)
            mod = fr.f_globals.get("__name__", "") if fr else ""
            pkg = fr.f_globals.get("__package__", "") if fr else ""

            # torchdata evidence
            if (
                any(h in mod for h in _TORCHDATA_HINTS)
                or any(h in pkg for h in _TORCHDATA_HINTS)
                or any(f"/{h.replace('.', '/')}/" in fn for h in _TORCHDATA_HINTS)
            ):
                saw_torchdata = True

            # vanilla torch evidence
            if (
                _TORCH_VANILLA_HINT in mod
                or _TORCH_VANILLA_HINT in pkg
                or "/torch/utils/data/" in fn
            ):
                saw_vanilla = True
    except Exception:
        # fall back below
        pass

    if saw_torchdata:
        return "torchdata"
    if saw_vanilla:
        return "vanilla"
    return "unknown"


# =============================================================================
# Free-threaded Python tensor lock utilities
# =============================================================================
#
# PyTorch < 2.10 has a race condition in PyType_GenericAlloc when multiple
# threads create tensor wrapper objects concurrently without the GIL.
# See: https://github.com/pytorch/pytorch/issues/171992
#
# These utilities provide a lock to serialize tensor operations on free-threaded
# Python builds (e.g., CPython 3.13t/3.14t) when using PyTorch < 2.10.


def _gil_disabled() -> bool:
    """Check if running on free-threaded Python (GIL disabled)."""
    check = getattr(sys, "_is_gil_enabled", None)
    if callable(check):
        try:
            return not bool(check())
        except Exception:
            pass

    try:
        gil_disabled = sysconfig.get_config_var("Py_GIL_DISABLED")
        if gil_disabled is not None:
            return bool(gil_disabled)
    except Exception:
        pass

    return False


def _torch_has_allocator_fix() -> bool:
    """Check if PyTorch version >= 2.10 (has the allocator race fix).

    PyTorch < 2.10 has a race condition in PyType_GenericAlloc when multiple
    threads create tensor wrapper objects concurrently without the GIL.
    See: https://github.com/pytorch/pytorch/issues/171992
    """
    try:
        import torch

        version = torch.__version__.split("+")[0]  # Strip build metadata like +cu124
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor) >= (2, 10)
    except Exception:
        return False  # Assume unfixed if can't determine


def _should_use_tensor_lock() -> bool:
    """Determine if tensor operations need lock protection."""
    if not _gil_disabled():
        return False  # GIL already serializes
    if _torch_has_allocator_fix():
        return False  # PyTorch >= 2.10 has the fix
    return True


# Lock to serialize tensor creation on free-threaded Python with PyTorch < 2.10.
_TENSOR_ITER_LOCK: threading.Lock | None = (
    threading.Lock() if _should_use_tensor_lock() else None
)


def _tensor_lock_ctx() -> threading.Lock | contextlib.nullcontext[None]:
    """Return context manager for tensor operations on free-threaded Python."""
    if _TENSOR_ITER_LOCK is not None:
        return _TENSOR_ITER_LOCK
    return contextlib.nullcontext()


# Warn once at module load if running on problematic configuration
if _should_use_tensor_lock():
    import warnings

    warnings.warn(
        (
            "PyTorch < 2.10 on free-threaded Python: tensor operations will be "
            "serialized to avoid allocator race condition. Upgrade to PyTorch 2.10+ "
            "for better performance. See: https://github.com/pytorch/pytorch/issues/171992"
        ),
        UserWarning,
        stacklevel=1,
    )
