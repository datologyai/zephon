# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Format registry helpers and built-in registrations."""

from zephon.io.formats.base import get_format

_INITIALIZED = False


def ensure_builtin_formats() -> None:
    """Ensure that the built-in format handlers are registered."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    # Importing these modules registers their handlers via side effects.
    import importlib

    importlib.import_module("zephon.io.formats.jsonl")
    importlib.import_module("zephon.io.formats.mds")

    # Optional formats - parquet and vortex handle missing deps internally
    importlib.import_module("zephon.io.formats.parquet")
    importlib.import_module("zephon.io.formats.vortex")

    # litdata requires optree and litdata packages at import time
    try:
        importlib.import_module("zephon.io.formats.litdata")
    except ImportError:
        pass

    _INITIALIZED = True


__all__ = ["ensure_builtin_formats", "get_format"]
