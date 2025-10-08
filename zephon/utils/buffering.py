# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Bounded buffering helpers for iterables."""

import queue
import threading
from collections.abc import Iterable, Iterator
from typing import Generator, TypeVar, cast

T = TypeVar("T")


def buffered_iterable(source: Iterable[T], capacity: int) -> Iterator[T]:
    """Materialise *source* behind a bounded queue of size *capacity*."""
    if capacity <= 0:
        return iter(source)

    class _Sentinel:
        pass

    sentinel: _Sentinel = _Sentinel()
    q: "queue.Queue[T | _Sentinel]" = queue.Queue(maxsize=capacity)
    stop_event = threading.Event()
    exc_holder: list[BaseException] = []

    source_iter = iter(source)

    def producer() -> None:
        try:
            for item in source_iter:
                while not stop_event.is_set():
                    try:
                        q.put(item, timeout=1)
                        break
                    except queue.Full:
                        if stop_event.is_set():
                            return
        except BaseException as err:  # noqa: BLE001
            exc_holder.append(err)
        finally:
            while True:
                try:
                    q.put(sentinel, timeout=1)
                    break
                except queue.Full:
                    if stop_event.is_set():
                        return

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    def consumer() -> Generator[T, None, None]:
        try:
            while True:
                item = q.get()
                if item is sentinel:
                    break
                yield cast(T, item)
        finally:
            stop_event.set()
            thread.join()
            if exc_holder:
                raise exc_holder[0]

    return consumer()
