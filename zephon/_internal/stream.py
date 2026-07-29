# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Runner-boundary wire types (engine samples, lane pointers, lazy payloads)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias

from zephon.types import (
    ChunkId,
    ChunkOffset,
    ComponentId,
    LaneId,
    SampleBatch,
    SampleId,
    SamplePayload,
    SampleRecord,
    StreamItem,
)

EngineSample = tuple[SampleId, LaneId, ChunkId, ChunkOffset, ComponentId]


@dataclass(slots=True)
class LanePtr:
    """Keeps track at which chunk and item we are per lane."""

    chunk_id: int = -1  # -1 means "nothing delivered yet"
    offset: int = 0  # number of final outputs from 'chunk_id' already delivered


def is_bytes_like(value: object) -> bool:
    """Check if *value* is bytes-like (bytes, memoryview, bytearray, or SHM-backed).

    Covers built-in types via isinstance and SHM-backed wrappers (e.g.
    ``_ShmBytes``) via duck-typing (must have ``__bytes__`` and ``decode``).
    """
    if isinstance(value, (bytes, memoryview, bytearray)):
        return True
    return hasattr(value, "__bytes__") and hasattr(value, "decode")


class LazyPayload(ABC):
    """A deferred payload; call ``.resolve_payload()`` to materialize.

    Runtime-only wrapper injected by runners that defer materialization
    across an IPC / plasma-store boundary (SHM for ProcessRunner, ObjectRef
    for the Ray runner).  Consumers never see this type statically:
    :attr:`zephon.types.SampleRecord.payload` is typed as
    :data:`zephon.types.SamplePayload`, and resolution boundaries (accumulator
    gate, stage exit, worker ingress) materialize any ``LazyPayload`` instances
    in place before downstream code runs.
    """

    __slots__ = ()

    @abstractmethod
    def resolve_payload(self) -> "SamplePayload": ...


# Pipeline items and micro-batches travel between operators/stages.
Microbatch = list[StreamItem]
# Inputs that enter a runner are either raw engine samples, previously emitted
# stream items, or micro-batches forwarded across runners.
RunnerStreamIn: TypeAlias = EngineSample | StreamItem
RunnerStageIn: TypeAlias = RunnerStreamIn | Microbatch
# Downstream stages read either micro-batches (preferred) or flattened stream
# items depending on how the runner is configured.
RunnerStageOut: TypeAlias = StreamItem | Microbatch


def lane_of(elem: RunnerStreamIn) -> LaneId:
    """Extract lane_id from any element that flows through the pipeline.

    Works with SampleRecord (.meta.lane_id), SampleBatch (first record's
    lane_id), and EngineSample tuples (index 1).
    """
    if isinstance(elem, tuple):
        return elem[1]  # EngineSample
    if isinstance(elem, SampleBatch):
        return elem.lane_ids[0]
    return elem.meta.lane_id  # SampleRecord


def resolve_lazy_payloads(items: list[Any]) -> None:
    """Resolve any :class:`LazyPayload` instances in *items* in-place.

    Call this in the worker process before ``process_many()``, or at stage
    exit boundaries to prevent lazy payloads from leaking downstream.

    No-op for records whose payloads are already materialized.
    """
    for item in items:
        if isinstance(item, SampleRecord) and isinstance(item.payload, LazyPayload):
            item.payload = item.payload.resolve_payload()  # type: ignore[assignment]
        elif isinstance(item, SampleBatch):
            for rec in item.records:
                if isinstance(rec.payload, LazyPayload):
                    rec.payload = rec.payload.resolve_payload()  # type: ignore[assignment]
