# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Canonical data model shared across the core data-loading pipeline."""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable, Sequence, TypeAlias, cast

from zephon.utils.length_extraction import TOKEN_FIELD_CANDIDATES, detect_length_field
from zephon.utils.tensor_utils import resolve_dtype, slice_last_dim, stack_sequences

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

    @property
    def base_offset(self) -> tuple[int, int]:
        """Return ``(chunk_id, chunk_offset)`` for eviction/base-offset accounting."""
        return (self.chunk_id, self.chunk_offset)

    def _cmp_key(self) -> tuple[int, int, LineagePath, SampleId]:
        return (
            self.chunk_id,
            self.chunk_offset,
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


@dataclass(frozen=True, slots=True)
class ContributorRef:
    """Reference to a contributing child derived from a base sample.

    ``cursor`` pinpoints the base offset and lineage; ``is_last_child=True``
    denotes the sole contributor (or tombstone) that closes that base offset for
    eviction purposes.
    """

    cursor: SampleCursor
    is_last_child: bool = True


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

    Contributor and tombstone flags are stored inside ``tags`` under
    ``"_contributors"`` and ``"_tombstone"`` because in simple pipelines
    (notify_monotone path) that increases the IPC overhead since the
    public schema grows.
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
        return replace(
            self,
            lineage=normalized,
        )

    def child(self, index: LineageIndex) -> "SampleMeta":
        """Return metadata for the ``index``-th child emitted from this sample."""
        return replace(
            self,
            lineage=self.lineage + (int(index),),
        )

    @property
    def cursor(self) -> SampleCursor:
        """Return a ``SampleCursor`` ordering key for this metadata."""
        return SampleCursor(
            self.chunk_id, self.chunk_offset, self.sample_id, self.lineage
        )

    def as_cursor_key(self) -> SampleCursorKey:
        """Convenience helper returning the tuple form used for persistence."""
        return self.cursor.as_key()

    def contribution_refs(self) -> tuple[ContributorRef, ...]:
        """Return contributor references used for eviction/progress.

        Operators that don't set ``contributors`` (1:1 outputs) are treated as emitting
        a single contributor with ``is_last_child=True`` using their own cursor.
        """
        if self.contributors:
            return self.contributors
        return (ContributorRef(self.cursor, True),)

    def contribution_cursors(self) -> tuple[SampleCursor, ...]:
        """Return the cursors for all contributors."""
        return tuple(ref.cursor for ref in self.contribution_refs())

    @property
    def contributors(self) -> tuple[ContributorRef, ...]:
        val = self.tags.get("_contributors")
        if val is None:
            return ()
        if isinstance(val, tuple):
            return val
        return tuple(val)

    @property
    def tombstone(self) -> bool:
        return bool(self.tags.get("_tombstone", False))

    def with_contributors(self, value: Iterable[ContributorRef] | None) -> "SampleMeta":
        """Return a new ``SampleMeta`` with contributors set/cleared in tags."""
        tags = dict(self.tags)
        if value:
            tags["_contributors"] = tuple(value)
        else:
            tags.pop("_contributors", None)
        return replace(self, tags=tags)

    def with_tombstone(self, value: bool = True) -> "SampleMeta":
        """Return a new ``SampleMeta`` with tombstone marker set/cleared in tags."""
        tags = dict(self.tags)
        if value:
            tags["_tombstone"] = True
        else:
            tags.pop("_tombstone", None)
        return replace(self, tags=tags)


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

    def to_training(
        self,
        *,
        tokens_field: str = "auto",
        return_labels: bool = False,
        dtype: Any = "auto",
        extra_fields: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Convert batch to training-ready format with optional LM label generation.

        Args:
            tokens_field: Field name containing token IDs, or "auto" to detect
                from common field names (input_ids, tokens, token_ids, ids).
            return_labels: If True, generates next-token prediction labels by
                shifting tokens. input_ids becomes tokens[:, :-1] and labels
                becomes tokens[:, 1:]. Extra fields are also shifted to match.
            dtype: Tensor dtype for stacking. Use "auto" to detect (prefers
                torch.long if available, else np.int64, else returns lists).
                Use None to explicitly return lists instead of tensors.
            extra_fields: Additional fields to include and stack (e.g.,
                ["attention_mask"]). These are shifted when return_labels=True.

        Returns:
            Dictionary with:
            - "ids": List of sample IDs (always list, not stacked)
            - "texts": List of text strings (always list, not stacked)
            - "input_ids": Stacked token tensor (shifted if return_labels=True)
            - "labels": Shifted labels tensor (only if return_labels=True)
            - Any extra_fields as stacked tensors (shifted if return_labels=True)

        Raises:
            TypeError: If payloads are not dicts.
            ValueError: If tokens_field cannot be auto-detected or is missing.
        """
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

        # Resolve tokens_field if "auto"
        resolved_tokens_field = tokens_field
        if tokens_field == "auto":
            resolved_tokens_field = detect_length_field(payloads[0])
            if resolved_tokens_field is None:
                raise ValueError(
                    f"Cannot auto-detect tokens field. Payload keys: "
                    + f"{list(payloads[0].keys())}. Expected one of: "
                    + f"{', '.join(TOKEN_FIELD_CANDIDATES)}"
                )

        # Validate tokens_field exists in all payloads
        for i, payload in enumerate(payloads):
            if resolved_tokens_field not in payload:
                raise ValueError(
                    f"Field '{resolved_tokens_field}' not found in payload at index {i}"
                )

        # Resolve dtype
        resolved_dtype, framework = resolve_dtype(dtype)

        # Build base result
        result: dict[str, Any] = {
            "ids": [r.meta.sample_id for r in items],
            "texts": texts,
        }

        # Extract and stack tokens
        token_lists = [payload[resolved_tokens_field] for payload in payloads]
        tokens = stack_sequences(token_lists, resolved_dtype, framework)

        if return_labels:
            # Shift for next-token prediction: input = tokens[:-1], labels = tokens[1:]
            result["input_ids"] = slice_last_dim(tokens, slice(None, -1), framework)
            result["labels"] = slice_last_dim(tokens, slice(1, None), framework)
        else:
            result["input_ids"] = tokens

        # Handle extra fields
        for field_name in extra_fields:
            # Check field exists in all payloads
            for i, payload in enumerate(payloads):
                if field_name not in payload:
                    raise ValueError(
                        f"Extra field '{field_name}' not found in payload at index {i}"
                    )

            field_lists = [payload[field_name] for payload in payloads]
            field_tensor = stack_sequences(field_lists, resolved_dtype, framework)

            if return_labels:
                # Shift extra fields to match input_ids shape
                result[field_name] = slice_last_dim(
                    field_tensor, slice(None, -1), framework
                )
            else:
                result[field_name] = field_tensor

        return result


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
