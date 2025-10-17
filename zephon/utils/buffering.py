# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Bounded buffering helpers for iterables."""

import queue
import threading
from collections.abc import Iterable, Iterator
from typing import Callable, Generator, TypeVar, cast

T = TypeVar("T")


def buffered_iterable(
    source: Iterable[T], capacity: int, on_stop: Callable[[], None] | None = None
) -> Iterator[T]:
    """
    Materialise ``source`` behind a bounded, FIFO queue fed by a background thread.

    This helper decouples production and consumption: a daemon producer thread
    pulls items from ``source`` and enqueues them (up to ``capacity``), while the
    returned iterator dequeues and yields them in order. Backpressure is applied
    when the queue is full.

    Parameters
    ----------
    source
        Any iterable (finite or infinite). **Important for unbounded / blocking
        sources**: see the *Cancellation contract* section below.
    capacity
        Maximum number of items buffered between producer and consumer. If
        ``capacity <= 0`` the function returns ``iter(source)`` directly (no
        background thread, no buffering), and ``on_stop`` is never called.
    on_stop
        Optional callback invoked **exactly once** when the consumer stops, for
        any reason: normal exhaustion, explicit ``close()``, early ``break``, or
        an exception in the consumer. Use this hook to cancel upstream work and
        trigger any cleanup associated with ``source`` (closing generators,
        signalling events, cancelling I/O, etc.).

    Returns:
    -------
    Iterator[T]
        A generator that yields items from ``source`` in order. Calling
        ``close()`` on it will run shutdown logic as described below.

    Behaviour
    ---------
    - **Ordering:** Items are yielded in the same order they are produced.
    - **Backpressure:** If the consumer is slower than the producer, the producer
      blocks on ``q.put`` until space is available.
    - **Shutdown (normal):** When ``source`` is exhausted, the consumer finishes
      and ``on_stop`` (if given) is called once.
    - **Shutdown (early / exceptional):** If the consumer breaks, closes the
      iterator, or raises, ``on_stop`` is called once; the queue is drained to
      unblock the producer; and the producer thread is joined with a short
      timeout to avoid deadlocks.

    Cancellation contract (read this if your source is unbounded or side-effecting)
    -------------------------------------------------------------------------------
    ``buffered_iterable`` **does not** forcibly interrupt a running ``next()`` on
    ``source`` from another thread (Python cannot preempt that safely). Instead,
    when the consumer stops it will:
      1) call ``on_stop()`` (if provided), then
      2) signal an internal stop event.

    **Contract:** If ``source`` can be unbounded, may block, or owns resources
    that must be released (e.g., a generator with a ``finally:`` block, network
    reads, subprocess output), your ``on_stop`` **must synchronously cancel the
    upstream** so the producer thread stops pulling and any upstream cleanup can
    run. Typical patterns include:

    - Closing a generator you created around the source:
        ``src = make_generator(); on_stop = src.close``
    - Flipping a threading.Event that the source polls and exits on.
    - Cancelling/closing an I/O object the source reads from.

    If you pass a raw infinite iterator that cannot be cancelled (e.g.
    ``itertools.count()``) **and** your ``on_stop`` is a no-op, the producer
    thread may continue to advance the source after the consumer has stopped,
    and upstream ``finally:`` blocks might not run promptly. Wrap such sources
    in a stoppable generator if you need deterministic cleanup.

    Error propagation
    -----------------
    - If the producer thread raises while iterating ``source``, the first
      exception is re-raised from the consumer when iteration ends (e.g., when
      you exhaust the iterator or close it).
    - If ``on_stop`` raises, that exception is propagated to the caller. (Note:
      because ``on_stop`` is executed in the consumer’s ``finally`` block,
      subsequent cleanup steps may not run if it raises.)
    - With ``capacity <= 0`` the call is a simple passthrough to ``iter(source)``
      and no background thread or ``on_stop`` semantics apply.

    Threading notes
    ---------------
    - The producer runs on a **daemon** thread and consumes ``source`` exclusively.
      Do not read from ``source`` elsewhere.
    - The queue is single-producer / single-consumer; items are only moved via
      the buffer.

    Examples:
    --------
    A stoppable infinite source that honours the contract:

    >>> stop = threading.Event()
    >>> def source():
    ...     try:
    ...         i = 0
    ...         while not stop.is_set():
    ...             yield i; i += 1
    ...     finally:
    ...         print("cleanup ran")
    >>> def on_stop():
    ...     stop.set()
    >>> it = buffered_iterable(source(), capacity=8, on_stop=on_stop)
    >>> for x in it:
    ...     if x == 100:
    ...         break
    >>> it.close()  # triggers on_stop -> cleanup

    For a generator with ``finally:``, close it in ``on_stop``:

    >>> def source():
    ...     try:
    ...         for i in range(1_000_000_000):
    ...             yield i
    ...     finally:
    ...         print("generator finally ran")
    >>> src = source()
    >>> it = buffered_iterable(src, capacity=8, on_stop=src.close)
    >>> next(it); next(it); it.close()
    """
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
                try:
                    item = q.get(timeout=0.1)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue
                if item is sentinel:
                    break
                yield cast(T, item)
        finally:
            # 1) Cancel upstream *before* we try to join the producer thread.
            if on_stop is not None:
                on_stop()
            stop_event.set()

            # 2) Free space (if producer was in q.put) and don’t block forever.
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

            thread.join(timeout=1.0)
            if exc_holder:
                raise exc_holder[0]

    return consumer()
