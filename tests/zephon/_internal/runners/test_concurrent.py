# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for concurrent runner building blocks."""

import threading

import pytest

from zephon._internal.runners.concurrent import _InflightCounter


class TestInflightCounter:
    """Tests for _InflightCounter atomic operations."""

    def test_try_decrement_returns_false_when_zero(self) -> None:
        """try_decrement should return False without raising when count is 0."""
        counter = _InflightCounter()
        assert counter.is_zero()
        assert counter.try_decrement() is False
        assert counter.is_zero()

    def test_try_decrement_succeeds_when_nonzero(self) -> None:
        """try_decrement should return True and decrement when count > 0."""
        counter = _InflightCounter()
        counter.increment()
        counter.increment()
        assert counter.try_decrement() is True
        assert not counter.is_zero()
        assert counter.try_decrement() is True
        assert counter.is_zero()

    def test_force_zero_clears_count(self) -> None:
        """force_zero should atomically set count to 0 and return previous."""
        counter = _InflightCounter()
        counter.increment()
        counter.increment()
        counter.increment()
        old = counter.force_zero()
        assert old == 3
        assert counter.is_zero()

    def test_force_zero_on_zero_is_safe(self) -> None:
        """force_zero should be safe to call when already zero."""
        counter = _InflightCounter()
        old = counter.force_zero()
        assert old == 0
        assert counter.is_zero()

    def test_concurrent_try_decrements_no_underflow(self) -> None:
        """Many threads racing to decrement should never underflow."""
        counter = _InflightCounter()
        num_increments = 100

        for _ in range(num_increments):
            counter.increment()

        results: list[int] = []
        lock = threading.Lock()

        def try_decrement_loop() -> None:
            count = 0
            while counter.try_decrement():
                count += 1
            with lock:
                results.append(count)

        threads = [threading.Thread(target=try_decrement_loop) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(results) == num_increments
        assert counter.is_zero()

    def test_decrement_still_raises_on_underflow(self) -> None:
        """Regular decrement should still raise on underflow."""
        counter = _InflightCounter()
        with pytest.raises(RuntimeError, match="underflowed"):
            counter.decrement()
