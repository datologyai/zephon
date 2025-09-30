# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Operators that pull raw samples from the shard store."""

from typing import Optional

from zephon.core.constants import Element, SampleId, SampleMeta, SampleRecord
from zephon.core.op_base import DefaultFinalize, OpContext
from zephon.core.traits import Buffering, OpTraits
from zephon.io import InMemoryDatasetStore, InMemoryMultiDatasetStore


class FetchOp(DefaultFinalize):
    """Load sample payloads from a `MultiDatasetShardStore`."""

    def __init__(self, buf: Optional[Buffering] = None) -> None:
        self._store: InMemoryMultiDatasetStore | None = None
        self._buffering = buf or Buffering(max_batch=64, max_latency_ms=5)

    def setup(self, ctx: OpContext) -> None:
        datasets_by_id = ctx.get("datasets_by_id")
        if not datasets_by_id:
            raise RuntimeError(
                "FetchOp requires 'datasets_by_id' in context (provided by WorkSource)"
            )
        # Build a multi-dataset store from dataset descriptors.
        # TODO(jwills): Here we would like to build a cache-backed store like in mosaic.
        # Not sure how the mosaic store handles multiple datasets, right now we index sampels by dataset id -> shard id -> sample id
        # So this code is bound to change and only works for in-memory testing ATM.
        views: dict[int, InMemoryDatasetStore] = {}
        for ds_id, ds in dict(datasets_by_id).items():
            backend = ds.backend  # type: ignore[attr-defined]
            kind = backend.get("kind") if isinstance(backend, dict) else None
            if kind == "inmem":
                shards = backend.get("shards", {})
                views[int(ds_id)] = InMemoryDatasetStore(shards)
            elif kind == "mds":
                # Not implemented yet: require mosaicml-streaming or custom backend.
                raise RuntimeError(
                    "MDS backend not available in FetchOp. Install a reader or implement a store."
                )
            else:
                raise RuntimeError(f"Unknown dataset backend kind: {kind}")
        self._store = InMemoryMultiDatasetStore(views)

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, parallelism=16)

    def buffering(self) -> Optional[Buffering]:
        return self._buffering

    def process_one(self, elem: Element) -> list[Element]:
        assert self._store is not None
        sample_id: SampleId = elem
        dataset_id, shard_id, sample_idx = sample_id
        view = self._store.for_dataset(dataset_id)
        shard = view.open(shard_id)
        row = shard[sample_idx]
        meta = SampleMeta(sample_id=sample_id)
        return [SampleRecord(meta=meta, payload=row)]

    def process_many(self, elems: list[Element]) -> list[Element]:
        assert self._store is not None
        outputs: list[Element] = []
        for sample_id in elems:
            dataset_id, shard_id, sample_idx = sample_id
            view = self._store.for_dataset(dataset_id)
            shard = view.open(shard_id)
            row = shard[sample_idx]
            meta = SampleMeta(sample_id=sample_id)
            outputs.append(SampleRecord(meta=meta, payload=row))
        return outputs
