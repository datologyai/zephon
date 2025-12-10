"""
GC helpers used as a defensive guard around CPython GC quirks.

See https://github.com/python/cpython/issues/142531 for background on why
we sometimes disable the collector in tight loops or startup paths.
"""

import gc
from contextlib import contextmanager


@contextmanager
def disable_gc():
    """
    Context manager that temporarily disables the Garbage Collector.

    It checks the current state upon entry and only re-enables the GC
    on exit if it was originally enabled. This is safe to nest or use
    in environments where GC might already be disabled.
    """
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
