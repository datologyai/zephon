# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Operators that pull raw samples from the shard store."""

from typing import Optional

from zephon.core.constants import (
    EngineSample,
    SampleMeta,
    SampleRecord,
)
from zephon.core.op_base import DefaultFinalize, OpContext
from zephon.core.traits import Buffering, OpTraits
from zephon.io import build_multi_dataset_store
from zephon.io.options import StoreOptions
from zephon.io.protocols import MultiDatasetShardStore


class FetchOp(DefaultFinalize[SampleRecord]):
    """Load sample payloads from a `MultiDatasetShardStore`."""

    def __init__(self, buf: Optional[Buffering] = None) -> None:
        self._store: MultiDatasetShardStore | None = None
        self._buffering = buf or Buffering(max_batch=64, max_latency_ms=5)

    def setup(self, ctx: OpContext) -> None:
        datasets_by_id = ctx.get("datasets_by_id")
        if not datasets_by_id:
            raise RuntimeError(
                "FetchOp requires 'datasets_by_id' in context (provided by WorkSource)"
            )
        store_options = StoreOptions.from_any(ctx.get("io_options"))
        self._store = build_multi_dataset_store(datasets_by_id, options=store_options)

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, parallelism=16)

    def buffering(self) -> Optional[Buffering]:
        return self._buffering

    def process_one(self, elem: EngineSample) -> list[SampleRecord]:
        assert self._store is not None
        sample_id, lane_id, chunk_id, chunk_offset = elem
        dataset_id, shard_id, sample_idx = sample_id
        view = self._store.for_dataset(dataset_id)
        shard = view.open(shard_id)
        row = shard[sample_idx]
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

        # Fetch per group and fill outputs.
        for (dataset_id, shard_id), items in groups.items():
            view = self._store.for_dataset(dataset_id)
            shard = view.open(shard_id)
            for pos, lane_id, chunk_id, chunk_offset, sample_idx, sample_id in items:
                row = shard[sample_idx]
                meta = SampleMeta(
                    sample_id=sample_id,
                    lane_id=lane_id,
                    chunk_id=chunk_id,
                    chunk_offset=chunk_offset,
                )
                out[pos] = SampleRecord(meta=meta, payload=row)

        # The type checker: out should now be fully populated.
        return [x for x in out if x is not None]
