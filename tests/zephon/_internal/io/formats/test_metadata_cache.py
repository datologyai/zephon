# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Metadata reuse, bounded retention, and concurrent cache misses."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from zephon._internal.io.formats.metadata_cache import MetadataCache


def test_lru_eviction_and_clear() -> None:
    cache = MetadataCache[str, object](2)
    first = cache.get_or_load("first", object)
    second = cache.get_or_load("second", object)
    assert cache.get_or_load("first", object) is first
    cache.get_or_load("third", object)
    assert cache.get_or_load("first", object) is first
    assert cache.get_or_load("second", object) is not second
    cache.clear()
    assert cache.get_or_load("first", object) is not first


def test_disabled_cache_and_invalid_limit() -> None:
    cache = MetadataCache[str, object](0)
    assert cache.get_or_load("key", object) is not cache.get_or_load("key", object)
    with pytest.raises(ValueError, match="negative"):
        MetadataCache(-1)


def test_failed_load_can_be_retried() -> None:
    cache = MetadataCache[str, object](1)

    def fail() -> object:
        raise OSError("missing")

    with pytest.raises(OSError, match="missing"):
        cache.get_or_load("key", fail)
    value = cache.get_or_load("key", object)
    assert cache.get_or_load("key", fail) is value


def test_concurrent_misses_return_first_retained_value() -> None:
    cache = MetadataCache[str, object](1)
    barrier = Barrier(2)

    def load() -> object:
        barrier.wait(timeout=5)
        return object()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(cache.get_or_load, "key", load) for _ in range(2)]
        first, second = [future.result(timeout=5) for future in futures]
    assert first is second
    assert cache.get_or_load("key", object) is first
