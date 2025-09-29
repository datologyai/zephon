# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Operators that pull raw samples from the shard store."""

from typing import Optional

from zephon.core.constants import Element, SampleId, SampleMeta, SampleRecord
from zephon.core.op_base import DefaultFinalize, OpContext
from zephon.core.traits import Buffering, OpTraits
from zephon.io import IndexedShardStore


class FetchOp(DefaultFinalize):
    """Load sample payloads from an `IndexedShardStore`."""

    def __init__(self, buf: Optional[Buffering] = None) -> None:
        self._store: IndexedShardStore | None = None
        self._buffering = buf or Buffering(max_batch=64, max_latency_ms=5)

    def setup(self, ctx: OpContext) -> None:
        store = ctx.get("shard_store")
        if store is None:
            msg = "FetchOp requires 'shard_store' in context"
            raise RuntimeError(msg)
        self._store = store

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, parallelism=16)

    def buffering(self) -> Optional[Buffering]:
        return self._buffering

    def process_one(self, elem: Element) -> list[Element]:
        assert self._store is not None
        sample_id: SampleId = elem
        shard = self._store.open(sample_id[0])
        row = shard[sample_id[1]]
        meta = SampleMeta(sample_id=sample_id)
        return [SampleRecord(meta=meta, payload=row)]

    def process_many(self, elems: list[Element]) -> list[Element]:
        assert self._store is not None
        outputs: list[Element] = []
        for sample_id in elems:
            shard = self._store.open(sample_id[0])
            row = shard[sample_id[1]]
            meta = SampleMeta(sample_id=sample_id)
            outputs.append(SampleRecord(meta=meta, payload=row))
        return outputs
