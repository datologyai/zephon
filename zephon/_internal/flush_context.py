"""Marks accumulator flushes that drain a pipeline being shut down early.

When the consumer stops iterating, runners still force-flush accumulators to
drain cleanly, but the output is never consumed. Ops use
``in_shutdown_flush()`` to keep tail-drop warnings for genuine end-of-data
flushes and log shutdown drains at DEBUG instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_shutdown_flush: ContextVar[bool] = ContextVar("zephon_shutdown_flush", default=False)


@contextmanager
def shutdown_flush(*, enabled: bool = True) -> Iterator[None]:
    """Mark shutdown drains; when disabled, leave the current context unchanged."""
    if not enabled:
        yield
        return

    token = _shutdown_flush.set(True)
    try:
        yield
    finally:
        _shutdown_flush.reset(token)


def in_shutdown_flush() -> bool:
    """Return whether the current flush is draining a pipeline being shut down."""
    return _shutdown_flush.get()


__all__ = ["in_shutdown_flush", "shutdown_flush"]
