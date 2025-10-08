# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Operators that pull raw samples from the shard store."""

from typing import Optional

from zephon.core.constants import Element, SampleId, SampleMeta, SampleRecord
from zephon.core.op_base import DefaultFinalize, OpContext
from zephon.core.traits import Buffering, OpTraits
from zephon.io import build_multi_dataset_store
from zephon.io.options import StoreOptions
from zephon.io.protocols import MultiDatasetShardStore


class FetchOp(DefaultFinalize):
    """Load sample payloads from a `MultiDatasetShardStore`."""

    def __init__(self, buf: Optional[Buffering] = None) -> None:
        self._store: MultiDatasetShardStore | None = None
        self._file_store = None
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
