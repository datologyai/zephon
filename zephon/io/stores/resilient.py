"""Resilient shard wrapper that retries around cache evictions."""

import contextlib

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from zephon.io.formats.base import FormatHandler
from zephon.io.protocols import RandomAccessShard, SampleLoadStats
from zephon.io.resolvers.base import ShardResolver
from zephon.io.types import ShardLocator
from zephon.observability.stopwatch import Stopwatch

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
        self.timer = Stopwatch(False)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[dict[str, object], SampleLoadStats]:
        if index < 0:
            raise IndexError(index)
        if self._length and index >= self._length:
            raise IndexError(index)

        stats = SampleLoadStats()
        attempts, cache_hits, cache_misses = (0, 0, 0)

        def _load_sample() -> dict[str, object]:
            nonlocal attempts, cache_hits, cache_misses
            attempts += 1
            # Unlike Mosaic, we resolve on every access to keep the cache warm.
            resolve_start = self.timer.start()
            local_ref = self._resolver.resolve(self._locator)
            stats.resolve_ns += self.timer.elapsed(resolve_start)

            # Note that cache hits/misses are a counter as we accumulate over all tries.
            if local_ref.cache_hit:
                cache_hits += 1
            else:
                cache_misses += 1

            open_start = self.timer.start()
            shard = self._handler.open_shard(self._locator, local_ref)
            stats.open_ns += self.timer.elapsed(open_start)

            read_start = self.timer.start()
            try:
                item = shard[index]
                assert isinstance(
                    item, dict
                )  # until we improve typing -- indicates no recursion in resilientshards.
            finally:
                stats.read_ns += self.timer.elapsed(read_start)
                close_start = self.timer.start()
                with contextlib.suppress(Exception):
                    shard.close()
                stats.close_ns += self.timer.elapsed(close_start)
            touch_start = self.timer.start()
            self._resolver.touch(self._locator)
            stats.touch_ns += self.timer.elapsed(touch_start)
            return item

        item = self._retrying(_load_sample)
        stats.retries = max(0, attempts - 1)
        stats.cache_hits = cache_hits
        stats.cache_misses = cache_misses
        self._last_stats = stats
        return item, stats

    def close(self) -> None:
        return None


__all__ = ["ResilientShard", "SampleLoadStats"]
