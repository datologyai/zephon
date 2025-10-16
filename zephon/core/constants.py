# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Canonical data model shared across the core data-loading pipeline."""

from dataclasses import dataclass, field
from typing import Any

DatasetId = int
ShardId = int
LocalSampleId = int
SampleId = tuple[DatasetId, ShardId, LocalSampleId]
LaneId = int
ChunkId = int
EngineSample = tuple[SampleId, LaneId, ChunkId]


@dataclass
class LanePtr:
    """Keeps track at which chunk and item we are per lane."""

    chunk_id: int = -1  # -1 means "nothing delivered yet"
    offset: int = 0  # number of final outputs from 'chunk_id' already delivered


@dataclass(frozen=True)
class SampleMeta:
    """Lightweight metadata that uniquely identifies a sample in a shard."""

    sample_id: SampleId
    lane_id: LaneId
    chunk_id: ChunkId
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleRecord:
    """Sample payload bundled with its metadata for transport through stages."""

    meta: SampleMeta
    payload: dict[str, Any]


@dataclass(frozen=True)
class SampleBatch:
    """A batch of SampleRecord."""

    records: tuple[SampleRecord, ...]

    def __len__(self) -> int:
        return len(self.records)

    @property
    def ids(self) -> tuple[SampleId, ...]:
        return tuple(r.meta.sample_id for r in self.records)

    @property
    def lane_ids(self) -> tuple[LaneId, ...]:
        return tuple(r.meta.lane_id for r in self.records)

    @property
    def chunk_ids(self) -> tuple[ChunkId, ...]:
        return tuple(r.meta.chunk_id for r in self.records)

    def to_training(self) -> dict[str, Any]:
        items = list(self.records)
        if not items:
            return {"ids": [], "texts": []}

        batch: dict[str, Any] = {
            "ids": [r.meta.sample_id for r in items],
            "texts": [r.payload.get("text", "") for r in items],
        }

        # include tensor-like fields only if present across all records
        keys_all = set(items[0].payload.keys())
        for r in items[1:]:
            keys_all &= set(r.payload.keys())

        if "input_ids" in keys_all:
            batch["input_ids"] = [r.payload["input_ids"] for r in items]
        if "attention_mask" in keys_all:
            batch["attention_mask"] = [r.payload["attention_mask"] for r in items]

        return batch
