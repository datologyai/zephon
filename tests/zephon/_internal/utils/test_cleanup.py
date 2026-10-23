"""Tests for shared cleanup deadlines."""

import pytest

from zephon._internal.utils import cleanup


def test_nested_cleanup_shares_the_earlier_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 10.0
    monkeypatch.setattr(cleanup.time, "monotonic", lambda: now)
    with cleanup.cleanup_budget(hard=True):
        now += 0.04
        with cleanup.cleanup_budget():
            assert cleanup.remaining_cleanup_timeout() == pytest.approx(0.06)
        now += 0.1
        assert cleanup.remaining_cleanup_timeout() == 0
    assert cleanup.remaining_cleanup_timeout() == cleanup._GRACEFUL_TIMEOUT


def test_cleanup_budget_restored_after_failure() -> None:
    with pytest.raises(ValueError):
        with cleanup.cleanup_budget(hard=True):
            raise ValueError("cleanup failed")
    assert cleanup.remaining_cleanup_timeout() == cleanup._GRACEFUL_TIMEOUT
