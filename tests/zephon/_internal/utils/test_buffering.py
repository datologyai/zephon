# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import itertools
import threading
import time
from typing import Iterator

import pytest

from zephon._internal.utils.buffering import buffered_iterable


def _gen_n(n: int) -> Iterator[int]:
    for i in range(n):
        yield i


def test_buffering_capacity_zero_passthrough_and_no_on_stop() -> None:
    called = {"count": 0}

    def on_stop() -> None:
        called["count"] += 1

    # capacity <= 0 returns iter(source) directly, so on_stop is never invoked
    out = list(buffered_iterable(range(5), 0, on_stop=on_stop))
    assert out == [0, 1, 2, 3, 4]
    assert called["count"] == 0


def test_buffering_yields_all_in_order_and_calls_on_stop_on_completion() -> None:
    called = {"count": 0}

    def on_stop() -> None:
        called["count"] += 1

    src = range(100)
    out = list(buffered_iterable(src, capacity=8, on_stop=on_stop))
    assert out == list(range(100))
    # Normal exhaustion should call on_stop once
    assert called["count"] == 1


def test_buffering_early_close_invokes_on_stop_once() -> None:
    called = {"count": 0}

    def on_stop() -> None:
        called["count"] += 1

    gen = buffered_iterable(itertools.count(), capacity=4, on_stop=on_stop)
    # Pull a few items, then close early
    first = [next(gen) for _ in range(5)]
    assert first == [0, 1, 2, 3, 4]
    # Closing the generator triggers the finally block (on_stop + join)
    gen.close()
    assert called["count"] == 1


def test_buffering_producer_exception_propagates_on_close() -> None:
    called = {"count": 0}

    def on_stop() -> None:
        called["count"] += 1

    def bad_source() -> Iterator[int]:
        yield 1
        yield 2
        # Background producer should record the exception
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        # Consuming to completion will close the consumer generator,
        # which raises the producer exception from its finally block.
        _ = list(buffered_iterable(bad_source(), capacity=2, on_stop=on_stop))

    # Even on error, on_stop should have been invoked exactly once
    assert called["count"] == 1


def test_buffering_close_unblocks_producer_waiting_on_full_queue() -> None:
    called = {"count": 0}

    def on_stop() -> None:
        called["count"] += 1

    # Source that outpaces the consumer; capacity=1 ensures quick backpressure
    src = (i for i in range(10_000))
    it = buffered_iterable(src, capacity=1, on_stop=on_stop)

    # Consume a single item then close immediately; if the producer were blocked
    # on a full queue and not unblocked by on_stop + draining, this could hang.
    _ = next(it)
    t0 = time.time()
    it.close()
    dt = time.time() - t0

    assert called["count"] == 1
    # Sanity: ensure close() returned promptly (no deadlock). Allow up to
    # 2 seconds to account for the internal join(timeout=1.0) and scheduler jitter.
    assert dt < 2.0


def test_buffering_break_triggers_on_stop_and_source_cleanup() -> None:
    called = {"count": 0}
    finished = threading.Event()

    def source() -> Iterator[int]:
        try:
            for i in itertools.count():
                yield i
        finally:
            finished.set()

    src_gen = source()

    def on_stop() -> None:
        called["count"] += 1
        # Honor the contract: cancel upstream promptly.
        src_gen.close()

    gen = buffered_iterable(src_gen, capacity=3, on_stop=on_stop)

    for idx, value in enumerate(gen):
        assert value == idx
        if idx == 5:
            break

    with contextlib.suppress(Exception):
        gen.close()

    assert called["count"] == 1
    assert finished.wait(timeout=2.0)


def test_buffering_consumer_exception_still_cleans_up() -> None:
    called = {"count": 0}
    finished = threading.Event()

    def source() -> Iterator[int]:
        try:
            for i in itertools.count():
                yield i
        finally:
            finished.set()

    src_gen = source()

    def on_stop() -> None:
        called["count"] += 1
        src_gen.close()

    gen = buffered_iterable(src_gen, capacity=2, on_stop=on_stop)

    def consume() -> None:
        for idx, _ in enumerate(gen):
            if idx == 3:
                raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        consume()
    gen.close()

    assert called["count"] == 1
    assert finished.wait(timeout=2.0)


def test_buffering_on_stop_exception_bubbles_up() -> None:
    called = {"count": 0}

    def on_stop() -> None:
        called["count"] += 1
        raise RuntimeError("stop failure")

    with pytest.raises(RuntimeError, match="stop failure"):
        list(buffered_iterable(range(3), capacity=2, on_stop=on_stop))

    assert called["count"] == 1
