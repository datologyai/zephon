"""Cooperative time budgets for explicit resource cleanup."""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_GRACEFUL_TIMEOUT = 1.0
_HARD_TIMEOUT = 0.1
_deadline: ContextVar[float | None] = ContextVar("cleanup_deadline", default=None)


@contextmanager
def cleanup_budget(*, hard: bool = False) -> Iterator[None]:
    """Share a deadline across nested cleanup calls on the current thread.

    Resource implementations consult this budget before waiting. It cannot
    interrupt arbitrary user teardown code or a blocking filesystem operation.
    """
    deadline = time.monotonic() + (_HARD_TIMEOUT if hard else _GRACEFUL_TIMEOUT)
    parent = _deadline.get()
    token = _deadline.set(min(parent, deadline) if parent is not None else deadline)
    try:
        yield
    finally:
        _deadline.reset(token)


def remaining_cleanup_timeout() -> float:
    """Return the remaining budget, or the default for a standalone close."""
    deadline = _deadline.get()
    return (
        _GRACEFUL_TIMEOUT if deadline is None else max(0.0, deadline - time.monotonic())
    )
