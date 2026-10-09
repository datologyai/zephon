# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Bounded, process-local metadata reuse across format readers."""

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Generic, TypeVar

_Key = TypeVar("_Key", bound=Hashable)
_Value = TypeVar("_Value")


class MetadataCache(Generic[_Key, _Value]):
    """Thread-safe LRU for immutable metadata, bounded by entry count.

    Loading happens outside the lock, so unrelated files can load concurrently.
    Concurrent misses may load the same key more than once; the first retained
    result wins. Failed loads are not cached. A zero limit disables retention.
    """

    def __init__(self, max_entries: int) -> None:
        if max_entries < 0:
            raise ValueError("max_entries cannot be negative")
        self._max_entries = max_entries
        self._entries: OrderedDict[_Key, _Value] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_load(self, key: _Key, load: Callable[[], _Value]) -> _Value:
        """Return retained metadata, or load and retain it on a miss."""
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]

        value = load()

        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
            self._entries[key] = value
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            return value

    def clear(self) -> None:
        """Release retained metadata."""
        with self._lock:
            self._entries.clear()
