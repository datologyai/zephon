# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Operators that pull raw samples from the shard store."""

from typing import Callable, Optional

from zephon.core.constants import (
    EngineSample,
    SampleMeta,
    SampleRecord,
)
from zephon.core.op_base import DefaultFinalize, DefaultSetup, OpContext
from zephon.core.traits import Buffering, OpTraits
from zephon.io import build_multi_dataset_store
from zephon.io.options import StoreOptions
from zephon.io.protocols import MultiDatasetShardStore
from zephon.io.stores.resilient import SampleLoadStats
from zephon.observability import Stopwatch
from zephon.observability.stats import FetchTimingDelta


class FetchOp(DefaultSetup, DefaultFinalize[SampleRecord]):
    """Load sample payloads from a `MultiDatasetShardStore`."""

    def __init__(self, buf: Optional[Buffering] = None) -> None:
        DefaultSetup.__init__(self)
        self._store: MultiDatasetShardStore | None = None
        self._buffering = buf or Buffering(max_batch=64, max_latency_ms=5)
        self._emit_fetch_metrics: Callable[[FetchTimingDelta], None] | None = None
        self._seen_shards: set[tuple[int, int]] = set()
        self._timer = Stopwatch(False)

    def setup(
        self,
        ctx: OpContext,
        stage_index: int,
        stage_name: str,
        op_index: int,
        collect_stats: bool,
    ) -> None:
        DefaultSetup.setup(self, ctx, stage_index, stage_name, op_index, collect_stats)

        datasets_by_id = ctx.get("datasets_by_id")
        if not datasets_by_id:
            raise RuntimeError(
                "FetchOp requires 'datasets_by_id' in context (provided by WorkSource)"
            )
        store_options = StoreOptions.from_any(ctx.get("io_options"))
        self._store = build_multi_dataset_store(datasets_by_id, options=store_options)
        self._emit_fetch_metrics = ctx.get("emit_fetch_metrics")
        self._seen_shards.clear()
        self._timer = Stopwatch(collect_stats)

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, parallelism=16)

    def buffering(self) -> Optional[Buffering]:
        return self._buffering

    def process_one(self, elem: EngineSample) -> list[SampleRecord]:
        assert self._store is not None
        sample_id, lane_id, chunk_id, chunk_offset = elem
        dataset_id, shard_id, sample_idx = sample_id
        view = self._store.for_dataset(dataset_id)
        shard, _ = view.open(shard_id)
        got = shard[sample_idx]
        if isinstance(got, tuple):
            row, _ = got
        else:
            row = got
        assert isinstance(row, dict)
        meta = SampleMeta(
            sample_id=sample_id,
            lane_id=lane_id,
            chunk_id=chunk_id,
            chunk_offset=chunk_offset,
        )
        return [SampleRecord(meta=meta, payload=row)]

    def process_many(self, elems: list[EngineSample]) -> list[SampleRecord]:
        assert self._store is not None
        if not elems:
            return []

        # Prepare output slots to preserve the input order regardless of grouping.
        out: list[SampleRecord | None] = [None] * len(elems)

        # Group by (dataset_id, shard_id) to reuse the opened shard per group.
        groups: dict[
            tuple[int, int], list[tuple[int, int, int, int, int, tuple[int, int, int]]]
        ] = {}
        for pos, elem in enumerate(elems):
            sample_id, lane_id, chunk_id, chunk_offset = elem
            dataset_id, shard_id, sample_idx = sample_id
            key = (int(dataset_id), int(shard_id))
            lst = groups.get(key)
            if lst is None:
                lst = []
                groups[key] = lst
            # Store (position-in-batch, lane_id, chunk_id, chunk_offset, sample_idx, sample_id)
            lst.append(
                (
                    pos,
                    int(lane_id),
                    int(chunk_id),
                    int(chunk_offset),
                    int(sample_idx),
                    sample_id,
                )
            )

        # TODO(MaxiBoether): Further optimize per-shard fetching
        # - Sort indices within each group by sample_idx to improve locality, then
        #   scatter results back to their original positions.
        # - If shards expose a bulk API (e.g., get_many/reads for a list of indices),
        #   use it to avoid per-item resolve/open/close in ResilientShard
        #   (see zephon/io/stores/resilient.py).

        group_start_ns = self._timer.start()
        metrics_deltas: list[FetchTimingDelta] = []

        for (dataset_id, shard_id), items in groups.items():
            view = self._store.for_dataset(dataset_id)
            (shard, reused_flag), resolve_overhead_ns = self._timer.time_call(
                view.open, shard_id
            )
            shard.timer = (
                self._timer
            )  # TODO: improve typing to indicate this always is a ResilientShard.

            (
                resolve_ns,
                open_ns,
                read_ns,
                close_ns,
                retries,
                cache_hits,
                cache_misses,
            ) = (0, 0, 0, 0, 0, 0, 0)
            sample_stats: list[SampleLoadStats] = []

            shard_key = (int(dataset_id), int(shard_id))
            seen_before = shard_key in self._seen_shards
            shard_reopens = 1 if not reused_flag and seen_before else 0

            for pos, lane_id, chunk_id, chunk_offset, sample_idx, sample_id in items:
                # TODO: We should improve the typing/structure a bit: Right now we always return a ResilientShard, which returns stats. For in memory data we don't return stats. This requires us to define a RandomAccessShard to either return stats and the item or just the item since a resilientshard is also a random acccess shard. It works but we could re-think the relationship of RandomAcccessShard, ResilientShard, and what the view returns.
                got = shard[sample_idx]
                if isinstance(got, tuple):
                    row, stats = got
                    sample_stats.append(stats)
                else:
                    row = got
                meta = SampleMeta(
                    sample_id=sample_id,
                    lane_id=lane_id,
                    chunk_id=chunk_id,
                    chunk_offset=chunk_offset,
                )
                out[pos] = SampleRecord(meta=meta, payload=row)
            self._seen_shards.add(shard_key)

            if self.collect_stats:
                for stats in sample_stats:
                    resolve_ns += stats.resolve_ns
                    open_ns += stats.open_ns
                    read_ns += stats.read_ns
                    close_ns += stats.close_ns
                    retries += stats.retries
                    cache_hits += stats.cache_hits
                    cache_misses += stats.cache_misses

                metrics_deltas.append(
                    FetchTimingDelta(
                        stage_index=self.stage_index,
                        shard_id=shard_id,
                        samples=len(items),
                        group_ns=0,  # populated below (rough estimations per group instead of actual measurement tho)
                        resolve_ns=resolve_overhead_ns + resolve_ns,
                        open_ns=open_ns,
                        read_ns=read_ns,
                        close_ns=close_ns,
                        retries=retries,
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        shard_reopens=shard_reopens,
                    )
                )

        if metrics_deltas:  # only contains items if stats collection is enabled.
            assert self._emit_fetch_metrics is not None
            group_total_ns = self._timer.elapsed(group_start_ns)
            group_count = len(metrics_deltas)
            base_share = group_total_ns // group_count if group_count else 0
            remainder = group_total_ns % group_count if group_count else 0

            for idx, delta in enumerate(metrics_deltas):
                delta.group_ns = base_share + (1 if idx < remainder else 0)
                self._emit_fetch_metrics(delta)

        # The type checker: out should now be fully populated.
        return [x for x in out if x is not None]
