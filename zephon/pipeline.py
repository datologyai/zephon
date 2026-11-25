# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""User-facing pipeline wrapper that layers ergonomics atop core planning."""

from zephon._internal.pipeline import (
    Pipeline,
)
from zephon._internal.record_ops import Branch

__all__ = [
    "Branch",
    "Pipeline",
]

# Re-home objects defined in the implementation modules onto this public path
# so runtime introspection (help/repr/tracebacks/get_type_hints) and pickle
# report ``zephon.pipeline``. Every name is importable here, so
# by-reference serialization keeps working.
for _name in __all__:
    _obj = globals()[_name]
    if getattr(_obj, "__module__", None) in (
        "zephon._internal.pipeline",
        "zephon._internal.record_ops",
    ):
        try:
            _obj.__module__ = __name__
        except (AttributeError, TypeError):  # aliases, Literals, constants
            pass
del _name, _obj
