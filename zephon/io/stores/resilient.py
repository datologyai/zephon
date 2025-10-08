"""Resilient shard wrapper that retries around cache evictions."""

import contextlib

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from zephon.io.formats.base import FormatHandler
from zephon.io.protocols import RandomAccessShard
from zephon.io.resolvers.base import ShardResolver
from zephon.io.types import ShardLocator

_RESILIENT_RETRY_EXCEPTIONS = (FileNotFoundError, OSError, IOError)


class ResilientShard(RandomAccessShard):
    """Resolve, open, and read shard samples with eviction-aware retries."""

    def __init__(
        self,
        *,
        locator: ShardLocator,
        resolver: ShardResolver,
        handler: FormatHandler,
        length: int,
        retry_attempts: int,
        retry_initial_backoff: float,
        retry_max_backoff: float,
    ) -> None:
        self._locator = locator
        self._resolver = resolver
        self._handler = handler
        self._length = max(0, int(length))
        attempts = max(1, int(retry_attempts))
        initial_backoff = max(0.0, float(retry_initial_backoff))
        max_backoff = max(initial_backoff, float(retry_max_backoff))
        self._retrying = Retrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(
                multiplier=initial_backoff or 0.01, max=max_backoff or 0.01
            ),
            retry=retry_if_exception_type(_RESILIENT_RETRY_EXCEPTIONS),
            reraise=True,
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            raise IndexError(index)
        if self._length and index >= self._length:
            raise IndexError(index)

        def _load_sample() -> dict[str, object]:
            # Unlike Mosiac, we currently resolve (= touch the cache) on every call.
            # Mosaic's model is: prefetching thread requests sample in advance, and then they optimistically
            # try to open their local ref, and only resolve here in case of exception.
            # Since we currently do not have an explicit prefetcher for the cache, we resolve first instead
            # of optimistically opening the old handle.
            # If we ever want to adapt, we should cache the local ref and only resolve on exception.
            local_ref = self._resolver.resolve(self._locator)

            shard = self._handler.open_shard(self._locator, local_ref)
            try:
                item = shard[index]
            finally:
                with contextlib.suppress(Exception):
                    shard.close()
            self._resolver.touch(self._locator)
            return item

        return self._retrying(_load_sample)

    def close(self) -> None:
        return None


__all__ = ["ResilientShard"]
