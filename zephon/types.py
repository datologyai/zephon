# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Canonical data model shared across the core data-loading pipeline."""

from collections import Counter
from collections.abc import Sized
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence, TypeAlias, cast

from zephon._internal.utils.length_extraction import (
    TOKEN_FIELD_CANDIDATES as _TOKEN_FIELD_CANDIDATES,
)
from zephon._internal.utils.length_extraction import (
    detect_length_field as _detect_length_field,
)
from zephon._internal.utils.tensor_utils import (
    count_valid_tokens as _count_valid_tokens,
)
from zephon._internal.utils.tensor_utils import (
    flatten_sequences as _flatten_sequences,
)
from zephon._internal.utils.tensor_utils import (
    labels_to_loss_mask as _labels_to_loss_mask,
)
from zephon._internal.utils.tensor_utils import (
    mask_padding_labels as _mask_padding_labels,
)
from zephon._internal.utils.tensor_utils import (
    mask_unsupervised_labels as _mask_unsupervised_labels,
)
from zephon._internal.utils.tensor_utils import (
    positions_to_cu_seqlens as _positions_to_cu_seqlens,
)
from zephon._internal.utils.tensor_utils import (
    resolve_dtype as _resolve_dtype,
)
from zephon._internal.utils.tensor_utils import (
    slice_last_dim as _slice_last_dim,
)
from zephon._internal.utils.tensor_utils import (
    stack_sequences as _stack_sequences,
)

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
ComponentId = int
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

    Component Contribution Tracking
    -------------------------------
    Samples track which mixture components they contain via two fields:

    ``component_sample_counts``: Maps component_id -> number of original samples
    from that component. For a regular sample this is ``{cid: 1}``. For a packed
    sample combining 3 from component 0 and 2 from component 1: ``{0: 3, 1: 2}``.

    ``component_token_counts``: Maps component_id -> token count from that component.
    None until packing occurs after tokenization. When packing tokenized samples,
    captures tokens per component, e.g., ``{0: 300, 1: 200}``.

    This design supports ensure_mixture tracking per-component contributions:

    - ``weight="samples"``: use ``component_sample_counts`` directly
    - ``weight="tokens"``: use ``component_token_counts`` if set, else distribute
      total token count proportionally by ``component_sample_counts``

    The separation allows packing before or after tokenization:

    - Pack before tokenize: only sample counts known at pack time
    - Pack after tokenize: both sample and token counts computed at pack time
    """

    sample_id: SampleId
    lane_id: LaneId
    chunk_id: ChunkId
    chunk_offset: ChunkOffset = 0
    component_sample_counts: dict[int, int] = field(default_factory=lambda: {0: 1})
    component_token_counts: dict[int, int] | None = None
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

    @property
    def is_sentinel(self) -> bool:
        """True if this record is a control signal (tombstone, flush, etc.)."""
        return self.tombstone or self.is_flush_sentinel

    @property
    def is_flush_sentinel(self) -> bool:
        """True if this record is a flush sentinel (triggers accumulator flush)."""
        return bool(self.tags.get("_flush_sentinel", False))

    @property
    def padding_length(self) -> int | None:
        """Trailing pad-token count of a flat ``pack_flat`` record, else ``None``."""
        packing = self.tags.get("_packing_metadata")
        if not packing:
            return None
        return packing.get("padding_length")

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


def _rename_and_exclude_fields(
    result: dict[str, Any],
    rename_fields: Mapping[str, str] | None,
    exclude_fields: Sequence[str],
    *,
    allow_missing_sources: bool = False,
) -> dict[str, Any]:
    """Rename emitted fields, rejecting collisions before applying exclusions."""
    if rename_fields:
        missing = [key for key in rename_fields if key not in result]
        if missing and not allow_missing_sources:
            raise ValueError(
                f"rename_fields source keys not in output: {missing}; "
                + f"emitted keys: {list(result)}"
            )
        # Count final names of emitted fields, so swaps are allowed and missing
        # sources cannot introduce collisions in an empty batch.
        target_counts = Counter(rename_fields.get(key, key) for key in result)
        collisions = sorted(key for key, count in target_counts.items() if count > 1)
        if collisions:
            raise ValueError(f"rename_fields target keys collide: {collisions}")
        result = {rename_fields.get(key, key): value for key, value in result.items()}

    for field_name in exclude_fields:
        result.pop(field_name, None)
    return result


@dataclass(slots=True)
class SampleBatch:
    """A batch of SampleRecord."""

    records: tuple[SampleRecord, ...]

    # Eagerly computed (hot path) — direct slot access, zero cost after init.
    lane_ids: tuple[LaneId, ...] = field(init=False, repr=False, compare=False)
    chunk_ids: tuple[ChunkId, ...] = field(init=False, repr=False, compare=False)

    # Lazily computed (cold path) — only built on first access.
    _ids: "tuple[SampleId, ...] | None" = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )
    _lineage_paths: "tuple[LineagePath, ...] | None" = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self) -> None:
        self.lane_ids = tuple(r.meta.lane_id for r in self.records)
        self.chunk_ids = tuple(r.meta.chunk_id for r in self.records)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def ids(self) -> tuple[SampleId, ...]:
        if self._ids is None:
            self._ids = tuple(r.meta.sample_id for r in self.records)
        return self._ids

    @property
    def lineage_paths(self) -> tuple[LineagePath, ...]:
        if self._lineage_paths is None:
            self._lineage_paths = tuple(r.meta.lineage for r in self.records)
        return self._lineage_paths

    def _extract_from_dicts(
        self,
        items: list[SampleRecord],
        tokens_field: str,
        extra_fields: Sequence[str],
    ) -> tuple[list[Any], list[str], dict[str, list[Any]], list[Any] | None]:
        """Extract tokens, texts, extra fields, and loss masks from dict payloads."""
        payloads: list[SamplePayloadDict] = []
        texts: list[str] = []
        for record in items:
            payload = record.payload
            if not isinstance(payload, dict):
                raise TypeError(f"Expected dict payload, got: {type(payload).__name__}")
            payloads.append(payload)
            texts.append(str(payload.get("text", "")))

        # Resolve tokens_field if "auto"
        resolved_tokens_field = tokens_field
        if tokens_field == "auto":
            resolved_tokens_field = _detect_length_field(payloads[0])
            if resolved_tokens_field is None:
                raise ValueError(
                    f"Cannot auto-detect tokens field. Payload keys: "
                    + f"{list(payloads[0].keys())}. Expected one of: "
                    + f"{', '.join(_TOKEN_FIELD_CANDIDATES)}"
                )

        # Validate tokens_field exists in all payloads
        for i, payload in enumerate(payloads):
            if resolved_tokens_field not in payload:
                raise ValueError(
                    f"Field '{resolved_tokens_field}' not found in payload at index {i}"
                )

        # Extract tokens
        token_lists = [payload[resolved_tokens_field] for payload in payloads]

        # Handle extra fields
        extra_data: dict[str, list[Any]] = {}
        for field_name in extra_fields:
            # Check field exists in all payloads
            for i, payload in enumerate(payloads):
                if field_name not in payload:
                    raise ValueError(
                        f"Extra field '{field_name}' not found in payload at index {i}"
                    )
            extra_data[field_name] = [payload[field_name] for payload in payloads]

        # Auto-surface the packing 'positions' field (document boundaries from
        # pack_flat) so callers don't have to thread it through extra_fields.
        if "positions" in payloads[0] and "positions" not in extra_data:
            for i, payload in enumerate(payloads):
                if "positions" not in payload:
                    raise ValueError(
                        f"'positions' present in some payloads but missing at index {i}"
                    )
            extra_data["positions"] = [payload["positions"] for payload in payloads]

        loss_mask_lists: list[Any] | None = None
        if "loss_mask" in payloads[0]:
            for i, payload in enumerate(payloads):
                if "loss_mask" not in payload:
                    raise ValueError(
                        f"'loss_mask' present in some payloads but missing at index {i}"
                    )
                mask_val = payload["loss_mask"]
                tokens_val = token_lists[i]
                if not isinstance(mask_val, Sized) or not isinstance(tokens_val, Sized):
                    raise ValueError(
                        f"'loss_mask' and '{resolved_tokens_field}' must be "
                        + f"sequences at index {i}"
                    )
                if len(mask_val) != len(tokens_val):
                    raise ValueError(
                        f"'loss_mask' length {len(mask_val)} does not match "
                        + f"'{resolved_tokens_field}' length {len(tokens_val)} "
                        + f"at index {i}"
                    )
            loss_mask_lists = [payload["loss_mask"] for payload in payloads]

        return token_lists, texts, extra_data, loss_mask_lists

    def _extract_from_arrays(
        self,
        items: list[SampleRecord],
    ) -> tuple[list[Any], list[str], dict[str, list[Any]], list[Any] | None]:
        """Extract tokens from array payloads (numpy/torch tensors)."""
        payloads = [r.payload for r in items]

        # Validate all payloads are array-like (have shape attr) not strings/bytes
        for i, p in enumerate(payloads):
            if isinstance(p, dict):
                raise TypeError(
                    "SampleBatch.to_training: mixed payload types at index "
                    f"{i} (expected all arrays, got dict)"
                )
            if not hasattr(p, "shape"):
                raise TypeError(
                    "SampleBatch.to_training expects dict or array-like "
                    "(numpy/torch) payloads"
                )

        return payloads, [""] * len(items), {}, None

    def to_training(
        self,
        *,
        tokens_field: str = "auto",
        return_labels: bool = False,
        dtype: Any = "auto",
        extra_fields: Sequence[str] = (),
        ignore_index: int = -100,
        rename_fields: Mapping[str, str] | None = None,
        flatten: bool = False,
        exclude_fields: Sequence[str] = (),
        return_num_valid_tokens: bool = False,
        return_loss_mask: bool = False,
        return_cu_seqlens: bool = False,
    ) -> dict[str, Any]:
        """Convert batch to training-ready format with optional LM label generation.

        A ``"loss_mask"`` payload field (a per-token supervised-vs-not mask,
        e.g. from chat-template tokenization) is recognized automatically, like
        ``"positions"``. With ``return_labels=True`` it is consumed: labels
        whose mask entry (shifted into label alignment, ``mask[:, 1:]``) is 0
        are set to ``ignore_index``. Set ``return_loss_mask=True`` to emit the
        final label-aligned mask, derived as ``labels != ignore_index`` after
        both supervision and padding masking. Listing ``"loss_mask"`` in
        ``extra_fields`` together with ``return_labels`` is an error: extra
        fields are sliced input-aligned (``[:, :-1]``), the wrong alignment for
        a label mask. Without ``return_labels`` it is surfaced stacked and
        unshifted, like any data field.

        Args:
            tokens_field: Field name containing token IDs, or "auto" to detect
                from common field names (input_ids, tokens, token_ids, ids).
            return_labels: If True, generates next-token prediction labels by
                shifting tokens. input_ids becomes tokens[:, :-1] and labels
                becomes tokens[:, 1:]. Extra fields are sliced like input_ids
                (drop the last token), so they stay aligned with it.
            dtype: Tensor dtype for stacking. Use "auto" to detect (prefers
                torch.long if available, else np.int64, else returns lists).
                Use None to explicitly return lists instead of tensors.
                A requested loss mask always uses float32 (Python floats for
                lists), independently of this dtype. Requested cumulative
                sequence lengths and per-row maxima use int32 (Python ints for
                lists). With flatten=True, the maximum is always a Python int.
            extra_fields: Additional fields to include and stack (e.g.,
                ["attention_mask"]). When return_labels=True they are sliced like
                input_ids (drop the last token) to stay aligned with it. A
                ``"positions"`` field (emitted by ``pack_flat``) is surfaced
                automatically and need not be listed here.
            ignore_index: Loss-ignore sentinel (default ``-100``). With
                ``return_labels``, each record's trailing pad labels
                (``meta.padding_length``) are set to this value.
            rename_fields: Optional ``{emitted_key: new_key}`` mapping applied to
                the output dict before exclusion (e.g. ``{"input_ids":
                "input"}``). Source keys must be present in the output, except
                for empty batches, where missing sources are ignored. All
                emitted fields are renamed, including for empty batches, and
                their final names must be unique. Swaps are allowed.
            flatten: Flatten stacked scalar token fields from ``[B, S]`` to
                ``[B * S]`` after per-sequence shifting and masking. The ids/texts
                fields remain per-record lists.
            exclude_fields: Output names to omit after renaming, e.g.
                ``("ids", "texts")``. Absent names are ignored.
            return_num_valid_tokens: Include a Python int counting labels unequal
                to ignore_index after masking (zero for empty batches). Requires
                return_labels=True.
            return_loss_mask: Include a label-aligned "loss_mask" with 1.0 for
                non-ignored labels and 0.0 otherwise. Requires return_labels=True.
                Uses float32 for tensors/arrays, Python floats for lists, and
                an empty list for empty batches. Follows flatten, rename_fields,
                and exclude_fields like other sequence fields.
            return_cu_seqlens: Include "cu_seqlens" and "max_seqlen" derived
                from zeros in input-aligned "positions". Requires a positions
                sequence matching the tokens in every payload, starting at zero
                in every nonempty input row. With flatten=False, boundaries
                have shape [B, K], where K is the largest boundary count in the
                batch, padded with each row's input length. Maxima have shape [B].
                With flatten=True, boundaries are one compact cumulative sequence
                across the batch, ending at the total input length, and the
                maximum is a Python int. Boundaries and per-row maxima use int32
                for tensors/arrays, Python ints for lists. Empty batches return
                []/[] for boundaries/maxima, or [0]/0 when flattened. Renaming
                and exclusion apply normally. Padding segments are preserved;
                this metadata does not itself enable attention masking or
                change labels. Works with or without return_labels.

        Returns:
            Dictionary with:
            - "ids": List of sample IDs (always list, not stacked)
            - "texts": List of text strings (always list, not stacked)
            - "input_ids": Stacked token tensor (shifted if return_labels=True)
            - "labels": Shifted labels tensor (only if return_labels=True), with
              each record's trailing pad labels and loss-masked positions set to
              ignore_index
            - "positions": Stacked document-position tensor, present iff the
              payloads carry one (sliced to match input_ids if return_labels=True)
            - Any extra_fields as stacked tensors (shifted if return_labels=True)
            - "num_valid_tokens": Number of non-ignored labels (only if
              return_num_valid_tokens=True)
            - "loss_mask": Float mask of non-ignored labels if
              return_loss_mask=True; otherwise the raw payload mask if present
              and return_labels=False
            - "cu_seqlens", "max_seqlen": Cumulative segment boundaries and
              maximum segment lengths (only if return_cu_seqlens=True)

        Raises:
            TypeError: If payloads are neither dicts nor arrays, or exclude_fields
                is a string instead of a sequence of field names.
            ValueError: If tokens_field cannot be auto-detected or is missing,
                if a loss_mask is present in only some payloads or misaligned
                with the token field, if "loss_mask" is listed in extra_fields
                with return_labels=True, or if rename_fields references a
                missing source key in a nonempty batch or produces a key collision.
                Also if return_num_valid_tokens or return_loss_mask is used without
                return_labels, or an extra field would overwrite the requested
                num_valid_tokens, cu_seqlens, or max_seqlen. Also if requested
                boundaries cannot be derived from aligned, zero-start positions
                or represented as int32.
        """
        if isinstance(exclude_fields, str):
            raise TypeError("exclude_fields must be a sequence of names, not a string")
        if return_loss_mask and not return_labels:
            raise ValueError("return_loss_mask requires return_labels=True")
        if return_cu_seqlens and {"cu_seqlens", "max_seqlen"}.intersection(
            extra_fields
        ):
            raise ValueError(
                "extra_fields cannot include 'cu_seqlens' or 'max_seqlen' "
                + "with return_cu_seqlens=True"
            )
        if return_num_valid_tokens:
            if not return_labels:
                raise ValueError("return_num_valid_tokens requires return_labels=True")
            if "num_valid_tokens" in extra_fields:
                raise ValueError(
                    "extra_fields cannot include 'num_valid_tokens' "
                    + "with return_num_valid_tokens=True"
                )
        if return_labels and "loss_mask" in extra_fields:
            raise ValueError(
                "extra_fields cannot include 'loss_mask' with return_labels=True: "
                + "extra fields are input-aligned, not label-aligned. Use "
                + "return_loss_mask=True for the label-aligned mask, or "
                + "return_labels=False for the raw mask."
            )
        items = list(self.records)
        if not items:
            empty: dict[str, Any] = {"ids": [], "texts": []}
            if return_num_valid_tokens:
                empty["num_valid_tokens"] = 0
            if return_loss_mask:
                empty["loss_mask"] = []
            if return_cu_seqlens:
                empty["cu_seqlens"] = [0] if flatten else []
                empty["max_seqlen"] = 0 if flatten else []
            return _rename_and_exclude_fields(
                empty, rename_fields, exclude_fields, allow_missing_sources=True
            )

        # Extract based on payload type
        if isinstance(items[0].payload, dict):
            token_lists, texts, extra_data, loss_mask_lists = self._extract_from_dicts(
                items, tokens_field, extra_fields
            )
        else:
            token_lists, texts, extra_data, loss_mask_lists = self._extract_from_arrays(
                items
            )

        if return_cu_seqlens:
            if "positions" not in extra_data:
                raise ValueError(
                    "return_cu_seqlens requires a 'positions' field in every payload; "
                    + "use pack_flat(emit_positions=True)"
                )
            for i, (tokens_row, positions_row) in enumerate(
                zip(token_lists, extra_data["positions"])
            ):
                if (
                    not isinstance(positions_row, Sized)
                    or not isinstance(tokens_row, Sized)
                    or len(positions_row) != len(tokens_row)
                ):
                    raise ValueError(
                        f"'positions' must match the token sequence length at index {i}"
                    )

        # Resolve dtype
        resolved_dtype, framework = _resolve_dtype(dtype)

        # Build base result
        result: dict[str, Any] = {
            "ids": [r.meta.sample_id for r in items],
            "texts": texts,
        }

        # Stack tokens
        tokens = _stack_sequences(token_lists, resolved_dtype, framework)

        if return_labels:
            # Shift for next-token prediction: input = tokens[:-1], labels = tokens[1:]
            result["input_ids"] = _slice_last_dim(tokens, slice(None, -1), framework)
            labels = _slice_last_dim(tokens, slice(1, None), framework)
            # After the shift, a record's trailing padding_length labels are
            # exactly its right-pad tokens; mask by position, not by id.
            pad_lengths = [r.meta.padding_length or 0 for r in items]
            if any(pad_lengths):
                labels = _mask_padding_labels(
                    labels, pad_lengths, ignore_index, framework
                )
            result["labels"] = labels
        else:
            result["input_ids"] = tokens

        # Handle extra fields
        for field_name, field_lists in extra_data.items():
            field_tensor = _stack_sequences(field_lists, resolved_dtype, framework)

            if return_labels:
                # Shift extra fields to match input_ids shape
                result[field_name] = _slice_last_dim(
                    field_tensor, slice(None, -1), framework
                )
            else:
                result[field_name] = field_tensor

        if loss_mask_lists is not None:
            masks = _stack_sequences(loss_mask_lists, resolved_dtype, framework)
            if return_labels:
                # The label at position i supervises token i+1, so the mask
                # shifts like labels (drop the first entry), not like inputs.
                result["labels"] = _mask_unsupervised_labels(
                    result["labels"],
                    _slice_last_dim(masks, slice(1, None), framework),
                    ignore_index,
                    framework,
                )
            elif "loss_mask" not in result:
                # extra_fields may have already emitted it explicitly.
                result["loss_mask"] = masks

        if return_loss_mask:
            result["loss_mask"] = _labels_to_loss_mask(
                result["labels"], ignore_index, framework
            )

        num_valid_tokens = None
        if return_num_valid_tokens:
            num_valid_tokens = _count_valid_tokens(
                result["labels"], ignore_index, framework
            )

        sequence_metadata: dict[str, Any] = {}
        if return_cu_seqlens:
            if (
                framework is not None
                and result["positions"].shape != result["input_ids"].shape
            ):
                raise ValueError("'positions' must match the input_ids shape")
            cu_seqlens, max_seqlen = _positions_to_cu_seqlens(
                result["positions"], framework, flatten=flatten
            )
            sequence_metadata = {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen}

        if flatten:
            result = {
                k: v if k in ("ids", "texts") else _flatten_sequences(v, framework)
                for k, v in result.items()
            }

        if num_valid_tokens is not None:
            result["num_valid_tokens"] = num_valid_tokens
        result.update(sequence_metadata)

        return _rename_and_exclude_fields(result, rename_fields, exclude_fields)


# Payload typing --------------------------------------------------------------
SampleNumeric: TypeAlias = int | float | complex

SamplePayloadAtom: TypeAlias = (
    bytes | memoryview | str | SampleNumeric | SamplePayloadArray
)
SamplePayload: TypeAlias = (
    SamplePayloadAtom | list["SamplePayload"] | dict[Any, "SamplePayload"]
)
SamplePayloadDict: TypeAlias = dict[Any, SamplePayload]


# Pipeline items travel between operators and stages; op authors annotate
# against this. The runner-boundary wire aliases (EngineSample, Microbatch,
# RunnerStage*, LazyPayload) live in zephon._internal.stream.
StreamItem: TypeAlias = SampleRecord | SampleBatch


__all__ = [
    "ChunkId",
    "ChunkOffset",
    "ComponentId",
    "ContributorRef",
    "DatasetId",
    "LaneId",
    "LineageIndex",
    "LineagePath",
    "LocalSampleId",
    "SampleBatch",
    "SampleCursor",
    "SampleCursorKey",
    "SampleId",
    "SampleMeta",
    "SampleNumeric",
    "SamplePayload",
    "SamplePayloadArray",
    "SamplePayloadAtom",
    "SamplePayloadDict",
    "SampleRecord",
    "ShardId",
    "StreamItem",
]
