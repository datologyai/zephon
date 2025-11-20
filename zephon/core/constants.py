# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Canonical data model shared across the core data-loading pipeline."""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable, TypeAlias, cast

if TYPE_CHECKING:  # Precise typing when numpy/torch available to the type checker.
    from numpy.typing import NDArray
    from torch import Tensor

    SamplePayloadArray: TypeAlias = NDArray[Any] | Tensor
else:  # Runtime fallback keeps optional dependencies optional.
    SamplePayloadArray = Any  # type: ignore[assignment]

DatasetId = int
ShardId = int
LocalSampleId = int
SampleId = tuple[DatasetId, ShardId, LocalSampleId]
LaneId = int
ChunkId = int
ChunkOffset = int
EngineSample = tuple[SampleId, LaneId, ChunkId, ChunkOffset]
LineageIndex = int
LineagePath = tuple[LineageIndex, ...]
# Cursor order: chunk_id -> chunk_offset -> lineage path -> original sample id.
# ``sample_id`` comes last so it only breaks ties when both physical position
# and lineage are identical (for example across shards) while remaining useful
# for debugging.
SampleCursorKey = tuple[ChunkId, ChunkOffset, LineagePath, SampleId]


def _normalize_lineage(path: Iterable[int] | LineagePath) -> LineagePath:
    """Return an immutable lineage path tuple with input validation."""
    try:
        normalized = tuple(int(v) for v in path)  # type: ignore[arg-type]
    except TypeError as exc:  # noqa: BLE001
        raise TypeError("lineage paths must be iterable sequences of ints") from exc
    return normalized


@dataclass(frozen=True, slots=True)
class SampleCursor:
    """Stable ordering key that survives fan-out across the pipeline."""

    chunk_id: ChunkId
    chunk_offset: ChunkOffset
    sample_id: SampleId
    lineage: LineagePath = field(default_factory=tuple)

    @staticmethod
    def from_key(key: SampleCursorKey) -> "SampleCursor":
        chunk_id, chunk_offset, lineage, sample_id = key
        sid = cast(SampleId, tuple(int(x) for x in sample_id))
        return SampleCursor(
            int(chunk_id), int(chunk_offset), sid, _normalize_lineage(lineage)
        )

    def child(self, index: LineageIndex) -> "SampleCursor":
        """Return the cursor for the ``index``-th child of this element."""
        if index < 0:
            raise ValueError("lineage index must be non-negative")
        return SampleCursor(
            self.chunk_id,
            self.chunk_offset,
            self.sample_id,
            self.lineage + (int(index),),
        )

    def as_key(self) -> SampleCursorKey:
        """Return the canonical tuple key used in checkpoints and comparisons."""
        return (
            int(self.chunk_id),
            int(self.chunk_offset),
            self.lineage,
            self.sample_id,
        )

    def _cmp_key(self) -> tuple[int, int, LineagePath, SampleId]:
        return (
            int(self.chunk_id),
            int(self.chunk_offset),
            self.lineage,
            self.sample_id,
        )

    def __lt__(self, other: "SampleCursor") -> bool:
        if not isinstance(other, SampleCursor):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented
        return self._cmp_key() < other._cmp_key()

    def __le__(self, other: "SampleCursor") -> bool:
        if not isinstance(other, SampleCursor):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented
        return self._cmp_key() <= other._cmp_key()

    def __gt__(self, other: "SampleCursor") -> bool:
        if not isinstance(other, SampleCursor):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented
        return self._cmp_key() > other._cmp_key()

    def __ge__(self, other: "SampleCursor") -> bool:
        if not isinstance(other, SampleCursor):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented
        return self._cmp_key() >= other._cmp_key()


@dataclass(slots=True)
class LanePtr:
    """Keeps track at which chunk and item we are per lane."""

    chunk_id: int = -1  # -1 means "nothing delivered yet"
    offset: int = 0  # number of final outputs from 'chunk_id' already delivered


@dataclass(frozen=True, slots=True)
class SampleMeta:
    """Lightweight metadata that uniquely identifies a sample in a shard.

    ``lineage`` tracks the deterministic position of this record after any fan-out.
    Operators that split inputs must call :meth:`child` in the order elements are
    emitted so downstream consumers observe an ordering identical to the
    single-threaded execution semantics enforced by the runner.
    """

    sample_id: SampleId
    lane_id: LaneId
    chunk_id: ChunkId
    chunk_offset: ChunkOffset = 0
    lineage: LineagePath = field(default_factory=tuple)
    tags: dict[str, Any] = field(default_factory=dict)

    def with_lineage(self, path: Iterable[int] | LineagePath) -> "SampleMeta":
        """Return a new ``SampleMeta`` where the lineage is replaced by ``path``."""
        normalized = _normalize_lineage(path)
        if normalized is self.lineage:
            return self
        return replace(self, lineage=normalized)

    def child(self, index: LineageIndex) -> "SampleMeta":
        """Return metadata for the ``index``-th child emitted from this sample."""
        return replace(self, lineage=self.lineage + (int(index),))

    @property
    def cursor(self) -> SampleCursor:
        """Return a ``SampleCursor`` ordering key for this metadata."""
        return SampleCursor(
            self.chunk_id, self.chunk_offset, self.sample_id, self.lineage
        )

    def as_cursor_key(self) -> SampleCursorKey:
        """Convenience helper returning the tuple form used for persistence."""
        return self.cursor.as_key()


@dataclass(slots=True)
class SampleRecord:
    """Sample payload bundled with its metadata for transport through stages."""

    meta: SampleMeta
    payload: "SamplePayload"


@dataclass(frozen=True, slots=True)
class SampleBatch:
    """A batch of SampleRecord."""

    records: tuple[SampleRecord, ...]

    def __len__(self) -> int:
        return len(self.records)

    @property
    def ids(self) -> tuple[SampleId, ...]:
        return tuple(r.meta.sample_id for r in self.records)

    @property
    def lineage_paths(self) -> tuple[LineagePath, ...]:
        return tuple(r.meta.lineage for r in self.records)

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

        payloads: list[SamplePayloadDict] = []
        texts: list[str] = []
        for record in items:
            payload = record.payload
            if not isinstance(payload, dict):
                raise TypeError(
                    "SampleBatch.to_training expects dict payloads on every record"
                )
            payloads.append(payload)
            texts.append(str(payload.get("text", "")))

        batch: dict[str, Any] = {
            "ids": [r.meta.sample_id for r in items],
            "texts": texts,
        }

        # include tensor-like fields only if present across all records
        keys_all = set(payloads[0].keys())
        for payload in payloads[1:]:
            keys_all &= set(payload.keys())

        if "input_ids" in keys_all:
            batch["input_ids"] = [payload["input_ids"] for payload in payloads]
        if "attention_mask" in keys_all:
            batch["attention_mask"] = [
                payload["attention_mask"] for payload in payloads
            ]

        return batch


# Payload typing --------------------------------------------------------------
SampleNumeric: TypeAlias = int | float | complex
SamplePayloadAtom: TypeAlias = (
    bytes | memoryview | str | SampleNumeric | SamplePayloadArray
)
SamplePayload: TypeAlias = (
    SamplePayloadAtom | list["SamplePayload"] | dict[Any, "SamplePayload"]
)
SamplePayloadDict: TypeAlias = dict[Any, SamplePayload]

# Pipeline items and micro-batches travel between operators/stages.
StreamItem: TypeAlias = SampleRecord | SampleBatch
Microbatch = list[StreamItem]
# Inputs that enter a runner are either raw engine samples, previously emitted
# stream items, or micro-batches forwarded across runners.
RunnerStreamIn: TypeAlias = EngineSample | StreamItem
RunnerStageIn: TypeAlias = RunnerStreamIn | Microbatch
# Downstream stages read either micro-batches (preferred) or flattened stream
# items depending on how the runner is configured.
RunnerStageOut: TypeAlias = StreamItem | Microbatch
