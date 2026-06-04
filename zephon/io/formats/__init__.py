# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Format registry helpers and built-in registrations."""

from zephon.io.formats.base import get_format

_INITIALIZED_FORMATS: set[str] = set()

_FORMAT_MODULES: dict[str, str] = {
    "jsonl": "zephon.io.formats.jsonl",
    "mds": "zephon.io.formats.mds",
    "parquet": "zephon.io.formats.parquet",
    "vortex": "zephon.io.formats.vortex",
    "litdata": "zephon.io.formats.litdata",
}


def ensure_builtin_formats(required: set[str], *, missing_ok: bool = False) -> None:
    """Register format handlers for the given format kinds.

    Only the requested formats are loaded.  Formats that have already been
    registered in a previous call are skipped (idempotent).  Each format
    module is expected to handle its own optional dependencies internally
    (e.g. parquet defers pyarrow, litdata defers optree/litdata).

    With ``missing_ok``, kinds outside the builtin table are skipped instead of
    raising — custom formats register themselves (and their codec) at import.
    """
    import importlib

    needed = required - _INITIALIZED_FORMATS
    for kind in needed:
        mod = _FORMAT_MODULES.get(kind)
        if mod is None:
            if missing_ok:
                continue
            raise ValueError(f"Unknown format kind: {kind!r}")
        importlib.import_module(mod)
        _INITIALIZED_FORMATS.add(kind)


__all__ = ["ensure_builtin_formats", "get_format"]
