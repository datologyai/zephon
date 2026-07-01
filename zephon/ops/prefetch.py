# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shard prefetching operator that warms the local cache ahead of consumption."""

import sys
import traceback
from typing import Any, Callable, Mapping

from zephon.core.accumulators import Accumulator, CountingAccumulator
from zephon.core.constants import EngineSample
from zephon.core.op_base import DefaultSetup, OpContext
from zephon.core.traits import OpTraits
from zephon.io.catalog import set_catalog_dir
from zephon.io.options import StoreOptions
from zephon.io.resolvers.base import ShardResolver
from zephon.io.stores.multi import build_resolver_with_locators
from zephon.io.types import ShardLocator
from zephon.observability.stats import PrefetchTimingDelta
from zephon.utils.rank import rank_ctx


class PrefetchOp(DefaultSetup):
    """Prefetch shards to local cache by looking ahead in the sample stream.

    This operator maintains a large buffer of samples, inspects their shard IDs,
    and downloads those shards to the local disk cache before they're needed by
    FetchOp. This reduces fetch latency when loading from remote storage (S3, GCS).

    The operator is deterministic: it yields samples in the exact order received,
    and only uses the lookahead buffer for triggering prefetch operations.

    Cache Coordination:
        PrefetchOp and FetchOp each create their own ShardResolver/CacheManager.
        Multiple CacheManager instances pointing to the same cache directory work
        correctly due to internal file locking.

    Threading Model:
        PrefetchOp uses Zephon's stage parallelism to download shards concurrently.
        With parallelism=4, up to 4 batches are processed in parallel by 4 worker
        threads, each downloading shards synchronously. This provides 4 concurrent
        downloads without additional thread management.

        The shared CacheManager coordinates concurrent resolve() calls across worker
        threads, ensuring shards are downloaded only once even when multiple workers
        request the same shard.

    Args:
        buffer_size: Number of samples to buffer for lookahead (default: 1024).
            Larger values provide more prefetch opportunities but use more memory.
    """

    def __init__(
        self,
        buffer_size: int = 1024,
    ) -> None:
        DefaultSetup.__init__(self)
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")

        self.buffer_size = int(buffer_size)

        # Runtime state (initialized in setup(), not here, to avoid pickling issues)
        self._resolver: ShardResolver | None = None
        self._locators: Mapping[tuple[int, int], ShardLocator] = {}

        # Callback for emitting metrics
        self._emit_prefetch_metrics: Callable[[PrefetchTimingDelta], None] | None = None

    def setup(
        self,
        ctx: OpContext,
        stage_index: int,
        stage_name: str,
        op_index: int,
        collect_stats: bool,
    ) -> None:
        DefaultSetup.setup(self, ctx, stage_index, stage_name, op_index, collect_stats)

        # Get dataset configurations
        datasets_by_id = ctx.get("datasets_by_id")
        if not datasets_by_id:
            raise RuntimeError(
                "PrefetchOp requires 'datasets_by_id' in context (provided by WorkSource)"
            )

        # Build resolver and locators for all file-backed datasets
        # Multiple CacheManager instances pointing to the same cache directory work
        # correctly due to internal file locking.
        store_options = StoreOptions.from_any(ctx.get("io_options"))
        # Resolve the process-global catalog dir before attaching/building the
        # node-local catalog (see FetchOp.setup for the rationale).
        set_catalog_dir(store_options)
        self._resolver, self._locators = build_resolver_with_locators(
            datasets_by_id, options=store_options, skip_inmem=True
        )

        # Get metrics callback from context
        self._emit_prefetch_metrics = ctx.get("emit_prefetch_metrics")

    def traits(self) -> OpTraits:
        # Preserves order and can be indexed
        # Higher parallelism allows concurrent downloads across batches
        return OpTraits(indexable=True, preserves_cursor_order=True, parallelism=4)

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[EngineSample]:
        # Use a counting accumulator with the configured buffer size
        # This ensures we have enough lookahead for prefetching
        return CountingAccumulator[EngineSample](
            max_batch=self.buffer_size,
            max_latency_ms=None if deterministic else 10,
        )

    def process_one(self, elem: EngineSample) -> list[EngineSample]:
        return self.process_many([elem])

    def process_many(self, elems: list[EngineSample]) -> list[EngineSample]:
        """Process a batch of samples, prefetching all unique shards.

        Scans the batch to identify all unique shards and downloads them
        synchronously. With stage parallelism > 1, multiple batches can
        download concurrently in different worker threads.
        """
        if not elems:
            return []

        # Identify all unique shards in this batch (preserving order for determinism)
        # Using dict to preserve insertion order (Python 3.7+)
        # This prevents test flakiness when cache is insufficient - download order
        # determines which shards get evicted (LRU), affecting cache hit/miss patterns.
        unique_shards = {(elem[0][0], elem[0][1]): None for elem in elems}

        # Prefetch all unique shards, tracking metrics with local counters (thread-safe)
        prefetch_succeeded = 0
        prefetch_failed = 0

        for dataset_id, shard_id in unique_shards:
            if self._prefetch_shard(dataset_id, shard_id):
                prefetch_succeeded += 1
            else:
                prefetch_failed += 1

        # Emit metrics if callback is configured
        if self._emit_prefetch_metrics is not None and self.collect_stats:
            delta = PrefetchTimingDelta(
                stage_index=self.stage_index,
                batch_size=len(elems),
                prefetch_requests=prefetch_succeeded + prefetch_failed,
                prefetch_succeeded=prefetch_succeeded,
                prefetch_failed=prefetch_failed,
            )
            self._emit_prefetch_metrics(delta)

        # Pass through all samples unchanged
        return elems

    def _prefetch_shard(self, dataset_id: int, shard_id: int) -> bool:
        """Download a shard to cache if not already present.

        This method calls CacheManager.resolve() synchronously. CacheManager
        handles concurrent resolve() calls correctly with its own locking,
        so multiple worker threads can safely call this method.

        Returns:
            True if prefetch succeeded, False otherwise (failed or skipped).
        """
        if self._resolver is None:
            return False

        key = (dataset_id, shard_id)

        # Check if we have a locator for this shard
        locator = self._locators.get(key)
        if locator is None:
            # No locator available (e.g., in-memory dataset or unknown format)
            return False

        try:
            # Resolve the shard to trigger download
            # CacheManager handles concurrent calls and deduplication
            self._resolver.resolve(locator, blocking=True)
            return True
        except Exception as exc:
            # Prefetch failures are non-fatal — FetchOp will retry the shard
            # on its own path.  Surface them anyway so underlying issues
            # (access/credentials, rate limits, missing keys) aren't masked
            # by silent retries.
            tb = traceback.format_exc().rstrip()
            print(
                f"[zephon] PrefetchOp: non-fatal shard prefetch failed "
                f"(FetchOp will retry) dataset_id={dataset_id} "
                f"shard_id={shard_id} err_type={type(exc).__name__} "
                f"err={exc!s} {rank_ctx()}\n{tb}",
                file=sys.stderr,
                flush=True,
            )
            return False


__all__ = ["PrefetchOp"]
