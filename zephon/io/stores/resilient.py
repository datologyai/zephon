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
from zephon.io.types import LocalShardRef, ShardLocator
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

        # Each FetchOp instance has its own shard store, however, all of the fetch instances share the same underlying local cache
        # Per FetchOp, we cache the ResilientShard instances, and internally, we can re-use the local ref with an optimistic path
        # With internal prefetching enabled, in practice, we hope to run into more cache hits than misses overall
        # If our resilient shard isn't used for quite some time we will most likely run into an exception, but the idea is that
        # the same shard is used within a certain local window. In a global shuffle, always resolving first would be better.
        # This follows Mosaic StreamingDataset, where a prefetching thread requests sample in advance, and then they optimistically
        # try to open their local ref, and only resolve in case of exception.
        self._local_ref: LocalShardRef | None = None

    def __len__(self) -> int:
        return self._length

    def _resolve(self, stats: SampleLoadStats) -> bool:
        """Resolve and store a fresh LocalShardRef; return True if cache hit."""
        resolve_start = self.timer.start()
        self._local_ref = self._resolver.resolve(self._locator)
        stats.resolve_ns += self.timer.elapsed(resolve_start)
        cache_hit = (
            self._local_ref.cache_hit
            if self._local_ref.cache_hit is not None
            else False
        )
        return cache_hit

    def __getitem__(self, index: int) -> tuple[dict[str, object], SampleLoadStats]:
        if index < 0:
            raise IndexError(index)
        if self._length and index >= self._length:
            raise IndexError(index)

        stats = SampleLoadStats()
        attempts, cache_hits, cache_misses, optimistic_reuses = (0, 0, 0, 0)
        # Track whether any resolve happened across all attempts of this __getitem__.
        # Semantics: count optimistic reuse only if we used a previously-held
        # local ref and never had to resolve at any point (including retries).
        resolved_any = False

        def _load_sample() -> dict[str, object]:
            nonlocal attempts, cache_hits, cache_misses, optimistic_reuses, resolved_any
            attempts += 1
            # Optimistic path: only resolve if we don't have a ref yet.
            used_optimistic = self._local_ref is not None
            if self._local_ref is None:
                if self._resolve(stats):
                    cache_hits += 1
                else:
                    cache_misses += 1
                resolved_any = True

            open_start = self.timer.start()
            try:
                # _local_ref is non-None here.
                shard = self._handler.open_shard(self._locator, self._local_ref)  # type: ignore[arg-type]
            except _RESILIENT_RETRY_EXCEPTIONS:
                stats.open_ns += self.timer.elapsed(open_start)
                if self._resolve(stats):
                    cache_hits += 1
                else:
                    cache_misses += 1
                resolved_any = True
                raise
            stats.open_ns += self.timer.elapsed(open_start)

            read_start = self.timer.start()
            try:
                item = shard[index]
                assert not isinstance(
                    item, tuple
                )  # until we improve typing -- indicates no recursion in resilientshards.
            except _RESILIENT_RETRY_EXCEPTIONS:
                if self._resolve(stats):
                    cache_hits += 1
                else:
                    cache_misses += 1
                resolved_any = True
                raise
            finally:
                stats.read_ns += self.timer.elapsed(read_start)
                close_start = self.timer.start()
                with contextlib.suppress(Exception):
                    shard.close()
                stats.close_ns += self.timer.elapsed(close_start)
            # Count optimistic reuse only if no resolve happened in any attempt.
            if used_optimistic and not resolved_any:
                optimistic_reuses += 1

            touch_start = self.timer.start()
            self._resolver.touch(self._locator)
            stats.touch_ns += self.timer.elapsed(touch_start)
            return item

        item = self._retrying(_load_sample)
        stats.retries = max(0, attempts - 1)
        stats.cache_hits = cache_hits
        stats.cache_misses = cache_misses
        self._last_stats = stats
        stats.optimistic_reuses = optimistic_reuses
        return item, stats

    def close(self) -> None:
        return None


__all__ = ["ResilientShard", "SampleLoadStats"]
