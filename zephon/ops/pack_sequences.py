# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Sequence packing operator for grouping variable-length sequences into bins.

Packing is split into two orthogonal concerns:

- **How to pack** (the algorithm): ``first_fit`` / ``best_fit`` pack whole
  records into bins; ``wrap`` slices a continuous token stream into bins of
  exactly ``max_length``. Every algorithm produces an ordered list of
  :class:`Segment` per bin and knows nothing about the output format.
- **How to serialize** a finished bin (a :class:`Segment` list): the
  :class:`_EnvelopeSerializer` keeps the lossless ``{"packed_samples": [...]}``
  list; the :class:`_FlatSerializer` concatenates the token field into a flat,
  fixed-length training record (with optional ``positions``).

This separation keeps the packing algorithms free of output-format logic and
makes ``flat`` a clean serialization of the same bin the envelope preserves.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import chain
from typing import Any, Literal, Optional, Sequence

import numpy as _np

# lazy import for torch, which is an optional dependency in zephon
try:
    import torch as _torch
except ImportError:  # pragma: no cover - torch absent only in slim envs.
    _torch = None  # type: ignore[assignment]

from zephon.core.accumulators import Accumulator, ReadyBatch
from zephon.core.children import (
    pack_meta,
    tombstones_for_record,
)
from zephon.core.constants import ComponentId, ContributorRef, LaneId, SampleRecord
from zephon.core.op_base import DefaultSetup
from zephon.core.traits import OpTraits
from zephon.utils.length_extraction import (
    TOKEN_FIELD_CANDIDATES,
    _get_length,
    detect_length_field,
    extract_length,
)
from zephon.utils.seeding import batch_seed
from zephon.utils.torch_compat import _tensor_lock_ctx

logger = logging.getLogger(__name__)


def _integer_apportion_tokens(length: int, weights: dict[int, int]) -> dict[int, int]:
    """Split ``length`` across components with integer shares proportional to weights.

    Largest-remainder style: last key (by sorted cid) absorbs any rounding gap so
    the parts sum to ``length`` exactly.
    """
    total_w = sum(weights.values())
    if total_w <= 0 or length <= 0:
        return {}
    out: dict[int, int] = {}
    acc = 0
    items = sorted(weights.items(), key=lambda kv: kv[0])
    for i, (cid, w) in enumerate(items):
        if i == len(items) - 1:
            out[cid] = max(0, length - acc)
        else:
            v = length * w // total_w
            out[cid] = v
            acc += v
    return out


# ----------------------------------------------------------------------
# Homogeneous packing (per-mixing-domain packing state)
# ----------------------------------------------------------------------

# A packing group: (lane_id, domain). ``domain`` is ``None`` for mixed packing
# (one group per lane) and the record's mixing-domain key when homogeneous
# packing partitions a lane.
PackKey = tuple[LaneId, Any]


def _component_domain(record: SampleRecord) -> ComponentId:
    """Fully homogeneous domain key: the record's sole component.

    Raises unless the record carries exactly one component.
    """
    counts = record.meta.component_sample_counts
    try:
        (domain,) = counts
    except ValueError:
        raise ValueError(
            "homogeneity='full' packing requires single-component records, but got "
            f"one with components {sorted(counts)}. Place packing before any "
            "operator that merges components."
        ) from None
    return domain


def _make_pack_key(
    domain_fn: Callable[[SampleRecord], Any] | None,
) -> Callable[[SampleRecord], PackKey]:
    """Record group key: ``(lane, None)`` when mixing, else ``(lane, domain)``."""
    if domain_fn is None:
        return lambda elem: (elem.meta.lane_id, None)
    return lambda elem: (elem.meta.lane_id, domain_fn(elem))


# ----------------------------------------------------------------------
# Shared serialization helpers (pure; operate on payload field values)
# ----------------------------------------------------------------------


def _positions_from_segment_lengths(segment_lengths: Sequence[int]) -> _np.ndarray:
    """Build a 1D ``int32`` positions array that resets to 0 at each segment.

    Given segment lengths ``[L0, L1, ..., Lk]`` returns a flat array of length
    ``sum(Li)`` whose contents are
    ``[0, 1, ..., L0-1, 0, 1, ..., L1-1, ..., 0, 1, ..., Lk-1]``.

    Downstream consumers (e.g. torchtitan's positions-based document mask)
    derive document IDs as ``cumsum(positions == 0) - 1``, so each ``0`` marks a
    fresh document boundary inside the packed bin.
    """
    parts = [_np.arange(int(n), dtype=_np.int32) for n in segment_lengths if n > 0]
    if not parts:
        return _np.empty(0, dtype=_np.int32)
    return _np.concatenate(parts)


def _pad_value_oob_msg(pad_value: int, dtype: Any) -> str:
    return (
        f"pack_flat: pad value {pad_value} is not representable in token field dtype "
        f"{dtype}; use a non-negative id within the dtype's range (or widen the dtype)."
    )


def _pad_field(seq: Any, pad_len: int, pad_value: int) -> Any:
    """Append ``pad_len`` copies of ``pad_value`` to ``seq`` along axis 0.

    Preserves the container type (list / tuple / numpy / torch).
    """
    if pad_len <= 0:
        return seq
    if isinstance(seq, list):
        return seq + [pad_value] * pad_len
    if isinstance(seq, tuple):
        return seq + (pad_value,) * pad_len
    if isinstance(seq, _np.ndarray):
        # Check the dtype range explicitly: numpy < 2.0 silently wraps an
        # out-of-range pad_value (e.g. -1 into uint*) instead of raising.
        if seq.dtype.kind in "iu":
            info = _np.iinfo(seq.dtype)
            if not info.min <= pad_value <= info.max:
                raise ValueError(_pad_value_oob_msg(pad_value, seq.dtype))
        tail = _np.full(pad_len, pad_value, dtype=seq.dtype)
        return _np.concatenate([seq, tail], axis=0)
    if _torch is not None and isinstance(seq, _torch.Tensor):
        try:
            tail = _torch.full((pad_len,), pad_value, dtype=seq.dtype)
        except (RuntimeError, OverflowError) as e:
            raise ValueError(_pad_value_oob_msg(pad_value, seq.dtype)) from e
        return _torch.cat([seq, tail], dim=0)
    raise TypeError(
        f"pack_flat: unsupported field type {type(seq).__name__} to pad; "
        "expected list, tuple, numpy.ndarray, or torch.Tensor"
    )


def _concatenate_sequences(sub_arrays: list[Any]) -> Any:
    """Concatenate a list of sequence slices preserving the original type.

    Sequence dimension is axis / dim 0 (1D tokens, ``(seq,)``, ``(seq, hidden)``,
    etc.). Supports list, tuple, ``numpy.ndarray``, and ``torch.Tensor``.
    Heterogeneous bins are not validated here; numpy/torch will raise their own
    clear errors when given mixed types.
    """
    if not sub_arrays:
        raise ValueError("Cannot concatenate an empty list of slices")

    first = sub_arrays[0]
    if isinstance(first, list):
        return list(chain.from_iterable(sub_arrays))
    if isinstance(first, tuple):
        return tuple(chain.from_iterable(sub_arrays))
    if isinstance(first, _np.ndarray):
        return _np.concatenate(sub_arrays, axis=0)
    if _torch is not None and isinstance(first, _torch.Tensor):
        return _torch.cat(sub_arrays, dim=0)

    raise TypeError(
        f"packing: unsupported slice type {type(first).__name__}; "
        "expected list, tuple, numpy.ndarray, or torch.Tensor"
    )


def _sliceable_field_names(
    payload: dict[str, Any], token_field: str, seq_len: int
) -> frozenset[str]:
    """Fields to slice in lockstep: ``token_field`` plus same-length sliceable fields.

    Scalar metadata (``int``, ``float``, ``bool``, ``None``) and other
    non-sequence values are ignored and do not appear in the packed output. Any
    sequence-like field whose length differs from ``seq_len`` raises.
    """
    if token_field not in payload:
        raise ValueError(
            f"packing: field {token_field!r} missing from payload "
            f"(keys: {list(payload.keys())})"
        )
    tf_val = payload[token_field]
    if _get_length(tf_val, token_field) != seq_len:
        raise ValueError(
            f"packing: field {token_field!r} length does not match record "
            f"sequence length {seq_len}"
        )
    # The token field must be sliceable here. An int value is a valid precomputed
    # length for envelope first_fit/best_fit (which never slices), but slicing it
    # would raise a cryptic ``'int' object is not subscriptable`` downstream.
    try:
        tf_val[0:0]
    except TypeError as e:
        raise ValueError(
            f"packing: tokens_field {token_field!r} must be a sliceable sequence to "
            f"slice/concatenate, got {type(tf_val).__name__} (an int precomputed "
            "length is only valid for envelope first_fit/best_fit)."
        ) from e
    names: set[str] = {token_field}
    for k, v in payload.items():
        if k == token_field:
            continue
        if isinstance(v, (bool, int, float, type(None))):
            continue
        try:
            length = _get_length(v, k)
        except TypeError:
            continue
        if length != seq_len:
            raise ValueError(
                f"packing: payload key {k!r} has length {length}, expected "
                f"{seq_len} to match field {token_field!r} for aligned slicing."
            )
        try:
            v[0:0]
        except Exception as e:
            raise ValueError(f"packing: payload key {k!r} is not sliceable.") from e
        names.add(k)
    return frozenset(names)


def _resolve_token_field(payload: Any, tokens_field: str) -> str:
    """Resolve the sliceable token field name from a payload.

    ``tokens_field`` is either an explicit field name or ``"auto"`` (detect via
    :func:`detect_length_field` over :data:`TOKEN_FIELD_CANDIDATES`).
    """
    if not isinstance(payload, dict):
        raise TypeError(
            "packing requires dict payloads to slice/concatenate; got payload "
            f"type {type(payload).__name__}"
        )
    if tokens_field != "auto":
        if tokens_field not in payload:
            raise ValueError(
                f"tokens_field {tokens_field!r} not found in payload "
                f"(keys: {list(payload.keys())})"
            )
        return tokens_field
    detected = detect_length_field(payload)
    if detected is None:
        raise ValueError(
            "cannot auto-detect tokens field. "
            f"Payload keys: {list(payload.keys())}. "
            f"Expected one of: {TOKEN_FIELD_CANDIDATES}"
        )
    return detected


# ----------------------------------------------------------------------
# Bin / Segment data structures
# ----------------------------------------------------------------------


@dataclass(slots=True)
class Bin:
    """A first_fit/best_fit bin: whole-record segments plus remaining capacity.

    Segments are built once at placement time (when the length is already known),
    so emitting the bin never re-measures a record.
    """

    segments: list["Segment"]
    remaining: int


@dataclass(slots=True)
class Segment:
    """One unit of a packed bin: a (possibly partial) span of a single record.

    The packing algorithms emit an ordered list of these per bin; serializers
    turn the list into an output payload. ``first_fit``/``best_fit`` produce
    whole-record segments (``start=0, end=seq_len, is_last=True, is_slice=False``)
    whose envelope output keeps the full payload; ``wrap`` produces slice
    segments (``is_slice=True``) of one contiguous token range.
    """

    record: SampleRecord
    start: int
    end: int
    seq_len: int  # full record length (for proportional token apportionment)
    field: str | None = None  # token field (slice / positions anchor)
    is_last: bool = False  # consumes the record's final token (closes contributors)
    is_slice: bool = False  # True = wrap (extract sliced fields), False = whole record

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(slots=True)
class _DeferredBin:
    """Deferred payload plan materialized by :meth:`PackSequences.process_many`."""

    segments: list[Segment]


# ----------------------------------------------------------------------
# Serializers (output format only; algorithm-agnostic)
# ----------------------------------------------------------------------


class _EnvelopeSerializer:
    """Serialize a bin as the lossless ``{"packed_samples": [...]}`` envelope.

    Whole-record segments contribute their full payload; slice segments (wrap)
    contribute the sliced sliceable-fields dict. The resulting list is passed
    through ``pack_payloads_fn`` (default identity → the list).
    """

    def __init__(self, pack_payloads_fn: Callable[[list[Any]], Any]) -> None:
        self._pack_payloads_fn = pack_payloads_fn

    def build_payload(self, segments: list[Segment]) -> dict[str, Any]:
        if not segments:
            raise ValueError("cannot serialize an empty bin")
        entries: list[Any] = []
        for seg in segments:
            if seg.is_slice:
                payload = seg.record.payload
                if not isinstance(payload, dict):
                    raise TypeError(
                        "algorithm='wrap' requires dict payloads to slice; got "
                        f"payload type {type(payload).__name__}"
                    )
                assert seg.field is not None
                fields = _sliceable_field_names(payload, seg.field, seg.seq_len)
                entries.append(
                    {k: payload[k][seg.start : seg.end] for k in sorted(fields)}
                )
            else:
                entries.append(seg.record.payload)
        return {"packed_samples": self._pack_payloads_fn(entries)}

    def padding_length(self, total_length: int) -> int | None:
        """Envelope packs are lossless and never pad."""
        return None


class _FlatSerializer:
    """Serialize a bin as a flat, fixed-length training record.

    Concatenates the token field (and any length-aligned sliceable fields)
    across segments, pads to ``max_length`` (token field with ``pad_token_id``,
    aligned fields with 0), and — when ``emit_positions`` — adds a ``positions``
    array that resets at each segment boundary (the pad tail forms its own
    trailing document). ``to_training`` consumes the record directly.
    """

    def __init__(
        self,
        max_length: int,
        tokens_field: str,
        pad_token_id: int | None,
        emit_positions: bool,
    ) -> None:
        self.max_length = max_length
        self.tokens_field = tokens_field
        self.pad_token_id = pad_token_id
        self.emit_positions = emit_positions

    def build_payload(self, segments: list[Segment]) -> dict[str, Any]:
        if not segments:
            raise ValueError("cannot serialize an empty bin")
        # wrap segments carry the lane-resolved field; whole-record segments
        # (first/best) leave it None, so resolve from the tokens_field config.
        token_field = segments[0].field or _resolve_token_field(
            segments[0].record.payload, self.tokens_field
        )

        field_sets: list[frozenset[str]] = []
        for seg in segments:
            payload = seg.record.payload
            if not isinstance(payload, dict):
                raise TypeError(
                    "pack_flat requires dict payloads to concatenate; got payload "
                    f"type {type(payload).__name__}"
                )
            field_sets.append(_sliceable_field_names(payload, token_field, seg.seq_len))
        first_fields = field_sets[0]
        for other in field_sets[1:]:
            if other != first_fields:
                raise ValueError(
                    "pack_flat: inconsistent payload keys or sequence lengths across "
                    f"packed records: {sorted(first_fields)!r} vs {sorted(other)!r}"
                )
        field_names = sorted(first_fields)
        # Catch a pre-existing 'positions' even if it is a scalar (and thus not in
        # field_names) — we are about to overwrite that key.
        if self.emit_positions and any(
            isinstance(seg.record.payload, dict) and "positions" in seg.record.payload
            for seg in segments
        ):
            raise ValueError(
                "pack_flat: payload already has a 'positions' field; rename the "
                "upstream field or set emit_positions=False."
            )

        per_field_parts = {
            k: [seg.record.payload[k][seg.start : seg.end] for seg in segments]
            for k in field_names
        }
        payload: dict[str, Any] = {
            k: _concatenate_sequences(per_field_parts[k]) for k in field_names
        }

        segment_lengths = [seg.length for seg in segments]
        pad_len = self.max_length - sum(segment_lengths)
        if pad_len > 0:
            if self.pad_token_id is None:
                raise ValueError(
                    "pack_flat padding requires pad_token_id but none was set."
                )
            for k in field_names:
                pad_value = self.pad_token_id if k == token_field else 0
                payload[k] = _pad_field(payload[k], pad_len, pad_value)
            segment_lengths = segment_lengths + [pad_len]
        if self.emit_positions:
            payload["positions"] = _positions_from_segment_lengths(segment_lengths)
        return payload

    def padding_length(self, total_length: int) -> int:
        """Pad tokens needed to fill the bin to ``max_length`` (0 when full)."""
        return max(0, self.max_length - total_length)


class PackingAccumulator(Accumulator[SampleRecord]):
    """Accumulator that packs variable-length sequences into bins.

    Runs on the pump thread and maintains per-lane bins. Bin assignment is
    deterministic regardless of parallelism. The chosen ``serializer`` turns each
    finished bin (a :class:`Segment` list) into the output record; the algorithm
    itself never touches the output format.
    """

    def __init__(
        self,
        max_length: int,
        num_bins: int,
        length_fn: Callable[[SampleRecord], int],
        algorithm: Literal["first_fit", "best_fit", "wrap"],
        drop_oversized: bool,
        min_sequence_length: int,
        shuffle_strategy: Literal["random", "length", None],
        shuffle_seed: int,
        flush_strategy: Literal["fifo", "fullest"],
        serializer: _EnvelopeSerializer | _FlatSerializer,
        wrap_field: str | None = None,
        domain_fn: Callable[[SampleRecord], Any] | None = None,
    ) -> None:
        self.max_length = max_length
        self.num_bins = num_bins
        self.length_fn = length_fn
        self.algorithm = algorithm
        self.drop_oversized = drop_oversized
        self.min_sequence_length = min_sequence_length
        self.shuffle_strategy = shuffle_strategy
        self.shuffle_seed = shuffle_seed
        self.flush_strategy = flush_strategy
        self._serializer = serializer
        self.wrap_field = wrap_field
        self._pack_key = _make_pack_key(domain_fn)

        # Oversized records dropped by first/best (can't split). Tombstones are
        # emitted at drop time; this only feeds the flush-time summary warning.
        self._dropped_oversized_count = 0
        self._dropped_oversized_tokens = 0

        # Bins/wrap buffers key on PackKey = (lane_id, domain). With no domain_fn
        # every record maps to (lane_id, None), i.e. plain per-lane packing.
        self._bins: defaultdict[PackKey, list[Bin]] = defaultdict(list)

        # wrap buffer per group: a buffered Segment's live tokens are
        # payload[field][start:end], and start advances as it drains into bins.
        self._wrap_segments: defaultdict[PackKey, deque[Segment]] = defaultdict(deque)
        self._wrap_total: defaultdict[PackKey, int] = defaultdict(int)
        # Auto-detected wrap field, keyed by lane: every record in a
        # lane must resolve to the same field, even across domains.
        self._wrap_lane_auto_field: dict[LaneId, str] = {}
        # Token counts already charged on a split record's non-final slices (floor
        # parts); the is_last slice subtracts these so the remainder lands exactly.
        # Keyed by (lane_id, cursor key) — cursor keys alone collide across lanes.
        self._wrap_comp_emitted: dict[tuple[Any, ...], defaultdict[int, int]] = {}

    @property
    def reads_payload(self) -> bool:
        return True

    def _known_keys(self) -> set[PackKey]:
        """All packing groups that currently hold (or held) buffered state."""
        return set(self._bins) | set(self._wrap_total) | set(self._wrap_segments)

    @staticmethod
    def _sorted_keys(keys: Iterable[PackKey]) -> list[PackKey]:
        """Deterministic key order (set iteration is layout-dependent).

        Domains are uniformly ``None`` or valued, so plain tuple sort is safe.
        """
        return sorted(keys)

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        """Return True if any bins or wrap segments hold data (in ``lane_id`` if given)."""
        if lane_id is None:
            if any(bins for bins in self._bins.values()):
                return True
            return any(total > 0 for total in self._wrap_total.values())
        if any(bins for key, bins in self._bins.items() if key[0] == lane_id):
            return True
        return any(
            total > 0 for key, total in self._wrap_total.items() if key[0] == lane_id
        )

    def push_many(
        self, elems: Sequence[SampleRecord]
    ) -> list[ReadyBatch[SampleRecord]]:
        """Accumulate records and emit packed bins when ready."""
        if not elems:
            return []

        # Avoid copying unless random shuffle mutates the batch.
        work_list: Sequence[SampleRecord]
        if self.shuffle_strategy == "random":
            shuffled = list(elems)
            rng = random.Random(batch_seed(self.shuffle_seed, shuffled))
            rng.shuffle(shuffled)
            work_list = shuffled
        elif self.shuffle_strategy == "length":
            work_list = sorted(
                elems,
                key=lambda rec: (-self.length_fn(rec), rec.meta.cursor.as_key()),
            )
        else:
            work_list = elems

        ready: list[ReadyBatch[SampleRecord]] = []

        for elem in work_list:
            key = self._pack_key(elem)

            # The wrap algorithm streams tokens through a per-group buffer and
            # slices into bins of exactly ``max_length``; oversized inputs
            # naturally span multiple bins so the >max_length / drop_oversized
            # branch does not apply here.
            if self.algorithm == "wrap":
                ready.extend(([rec], 0) for rec in self._wrap_pack(key, elem))
                continue

            seq_len = self.length_fn(elem)

            # Handle oversized sequences. first_fit/best_fit cannot split a
            # record, so an oversized one is dropped (with a tombstone so its
            # contributor offsets still close, and counted for the flush-time
            # warning) when drop_oversized is set; else it is a hard error.
            if seq_len > self.max_length:
                if self.drop_oversized:
                    self._dropped_oversized_count += 1
                    self._dropped_oversized_tokens += seq_len
                    ready.extend(([t], 0) for t in tombstones_for_record(elem))
                    continue
                raise ValueError(
                    f"Sequence length {seq_len} exceeds max_length {self.max_length}"
                )

            if self.algorithm == "first_fit":
                packed = self._first_fit_pack(key, elem, seq_len)
            elif self.algorithm == "best_fit":
                packed = self._best_fit_pack(key, elem, seq_len)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            for rec in packed:
                ready.append(([rec], 0))

        return ready

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[SampleRecord]]:
        """Emit any remaining partially-filled bins / wrap tails."""
        if lane_id is None:
            # Stable group order at upstream close (set iteration is layout-dependent).
            keys = self._sorted_keys(self._known_keys())
        else:
            keys = self._sorted_keys(k for k in self._known_keys() if k[0] == lane_id)

        ready: list[ReadyBatch[SampleRecord]] = []
        # Wrap mode: drop any tail that did not fill a full ``max_length``, emit
        # tombstones so dropped records still close contributor offsets, and
        # clear buffered state. ``tombstones_for_record`` only emits for
        # contributors with ``is_last_child=True`` (mirroring ReplayFilter /
        # MapTransform), so non-closing contributors are not falsely advanced.
        # One entry per packing group (lane, domain) with a dropped wrap tail.
        tail_drop_groups: list[tuple[PackKey, int]] = []
        wrap_tombstones: list[ReadyBatch[SampleRecord]] = []
        for key in keys:
            lid = key[0]
            for bin_data in self._bins.pop(key, []):
                if bin_data.segments:
                    ready.append(([self._emit_first_best_bin(bin_data, lid)], 0))

            total = self._wrap_total.pop(key, 0)
            segments = self._wrap_segments.pop(key, None)
            if total > 0 and segments:
                tail_drop_groups.append((key, total))
                for seg in segments:
                    rec = seg.record
                    self._wrap_comp_emitted.pop((lid, rec.meta.cursor.as_key()), None)
                    wrap_tombstones.extend(([t], 0) for t in tombstones_for_record(rec))

        if tail_drop_groups:
            dropped_tokens = sum(t for _, t in tail_drop_groups)
            dropped_lanes = len({k[0] for k, _ in tail_drop_groups})
            logger.warning(
                "PackSequences wrap: dropping %s trailing token(s) across %d packing "
                "group(s) in %d lane(s) that did not fill a full max_length=%d bin; "
                "emitted %d tombstone(s) to close contributor offsets.",
                dropped_tokens,
                len(tail_drop_groups),
                dropped_lanes,
                self.max_length,
                len(wrap_tombstones),
            )

        # The auto-field cache is lane-scoped, so reset it for the flushed lanes
        # (a lane with only empty wrap records has no key above but may hold one).
        if lane_id is None:
            self._wrap_comp_emitted.clear()
            self._wrap_lane_auto_field.clear()
        else:
            self._wrap_lane_auto_field.pop(lane_id, None)
        ready.extend(wrap_tombstones)

        # first_fit/best_fit oversized drops: surface the count so a comparison
        # against wrap (which keeps every token by splitting) is honest.  These
        # are global diagnostic counters (the drop+tombstone already happened in
        # push_many), not per-lane replay state, so report and reset only on the
        # all-lanes flush at upstream close.
        if lane_id is None and self._dropped_oversized_count:
            logger.warning(
                "PackSequences %s: dropped %d document(s) (%d token(s)) longer "
                "than max_length=%d; these algorithms cannot split a record, so "
                "oversized inputs are discarded (wrap would split them instead).",
                self.algorithm,
                self._dropped_oversized_count,
                self._dropped_oversized_tokens,
                self.max_length,
            )
            self._dropped_oversized_count = 0
            self._dropped_oversized_tokens = 0

        return ready

    # ------------------------------------------------------------------
    # first_fit / best_fit
    # ------------------------------------------------------------------

    def _first_fit_pack(
        self, key: PackKey, seq: SampleRecord, seq_len: int
    ) -> list[SampleRecord]:
        """Try to pack sequence using first-fit algorithm."""
        bins = self._bins[key]
        outputs: list[SampleRecord] = []

        for bin_data in bins:
            if bin_data.remaining >= seq_len:
                outputs.extend(
                    self._add_sample_to_bin(bin_data, bins, seq, seq_len, key)
                )
                return outputs

        outputs.extend(self._create_bin_with_sample(bins, seq, seq_len, key))
        return outputs

    def _best_fit_pack(
        self, key: PackKey, seq: SampleRecord, seq_len: int
    ) -> list[SampleRecord]:
        """Try to pack sequence using best-fit algorithm."""
        bins = self._bins[key]
        outputs: list[SampleRecord] = []

        best_bin = None
        best_remaining = self.max_length + 1

        for bin_data in bins:
            remaining = bin_data.remaining
            if remaining >= seq_len and remaining < best_remaining:
                best_bin = bin_data
                best_remaining = remaining

        if best_bin is not None:
            outputs.extend(self._add_sample_to_bin(best_bin, bins, seq, seq_len, key))
            return outputs

        outputs.extend(self._create_bin_with_sample(bins, seq, seq_len, key))
        return outputs

    def _create_bin_with_sample(
        self,
        bins: list[Bin],
        seq: SampleRecord,
        seq_len: int,
        key: PackKey,
    ) -> list[SampleRecord]:
        """Create a new bin, add a sample, and emit if full."""
        new_bin = Bin(
            segments=[self._whole_segment(seq, seq_len)],
            remaining=self.max_length - seq_len,
        )
        if new_bin.remaining < self.min_sequence_length:
            # Self-emits immediately and is never retained, so no eviction needed
            # — don't flush an existing partial bin to make room it won't use.
            return [self._emit_first_best_bin(new_bin, key[0])]
        outputs = self._enforce_max_bins(bins, key)
        bins.append(new_bin)
        return outputs

    def _add_sample_to_bin(
        self,
        bin_data: Bin,
        bins: list[Bin],
        seq: SampleRecord,
        seq_len: int,
        key: PackKey,
    ) -> list[SampleRecord]:
        """Add a sample to an existing bin and emit if full."""
        bin_data.segments.append(self._whole_segment(seq, seq_len))
        bin_data.remaining -= seq_len
        outputs: list[SampleRecord] = []
        if bin_data.remaining < self.min_sequence_length:
            bins.remove(bin_data)
            outputs.append(self._emit_first_best_bin(bin_data, key[0]))
        return outputs

    def _enforce_max_bins(
        self,
        bins: list[Bin],
        key: PackKey,
    ) -> list[SampleRecord]:
        """Flush bins to keep at most ``num_bins`` open per packing group.

        The limit is per group, so homogeneous packing keeps ``num_bins`` open
        bins *per domain* in a lane.
        """
        outputs: list[SampleRecord] = []
        while len(bins) >= self.num_bins and bins:
            if self.flush_strategy == "fifo":
                bin_to_flush = bins.pop(0)
            elif self.flush_strategy == "fullest":
                fullest_idx = min(range(len(bins)), key=lambda i: bins[i].remaining)
                bin_to_flush = bins.pop(fullest_idx)
            else:
                raise ValueError(f"Unknown flush_strategy: {self.flush_strategy}")
            outputs.append(self._emit_first_best_bin(bin_to_flush, key[0]))
        return outputs

    @staticmethod
    def _whole_segment(record: SampleRecord, seq_len: int) -> Segment:
        """A first_fit/best_fit segment: the whole record as one complete document."""
        return Segment(
            record=record, start=0, end=seq_len, seq_len=seq_len, is_last=True
        )

    def _emit_first_best_bin(self, bin_data: Bin, lane_id: int) -> SampleRecord:
        """Emit a finished first_fit/best_fit bin (segments built at placement time)."""
        return self._emit_bin(bin_data.segments, lane_id)

    # ------------------------------------------------------------------
    # Shared emit / lineage
    # ------------------------------------------------------------------

    def _emit_bin(self, segments: list[Segment], lane_id: int) -> SampleRecord:
        """Build the output record for one finished bin.

        Meta is computed serially; payload materialization is deferred.
        """
        if not segments:
            raise ValueError("cannot emit a packed record from an empty bin")
        meta = self._build_meta(segments, lane_id)
        return SampleRecord(meta=meta, payload=_DeferredBin(segments))

    def _record_component_token_targets(
        self, record: SampleRecord, record_len: int
    ) -> dict[int, int]:
        """Per-component token target: explicit counts if present, else apportioned by sample share.

        The returned dict is read-only (the explicit-counts branch aliases the
        record's own dict to avoid a copy); callers must not mutate it.
        """
        if record.meta.component_token_counts is not None:
            return record.meta.component_token_counts
        return _integer_apportion_tokens(
            record_len, record.meta.component_sample_counts
        )

    def _build_meta(self, segments: list[Segment], lane_id: int) -> Any:
        """Aggregate contributors and component counts across a bin's segments.

        Handles whole records (first/best: a single ``is_last`` segment per
        record) and split records (wrap: a record spans multiple bins) uniformly.
        Component sample counts are charged on the closing (``is_last``) segment so
        a split record counts once; component token counts use floor splits on
        intermediate slices and a remainder fixup on the closing slice so totals
        match each record's apportioned target.
        """
        contributors: list[ContributorRef] = []
        component_sample_counts: dict[int, int] = defaultdict(int)
        component_token_counts: dict[int, int] = defaultdict(int)
        for seg in segments:
            record = seg.record
            for ref in record.meta.contribution_refs():
                contributors.append(
                    ContributorRef(
                        cursor=ref.cursor,
                        is_last_child=(ref.is_last_child and seg.is_last),
                    )
                )

            if seg.is_last:
                for cid, count in record.meta.component_sample_counts.items():
                    component_sample_counts[cid] += count

            record_len = seg.seq_len
            targets = self._record_component_token_targets(record, record_len)
            if not targets or record_len <= 0:
                continue

            # Key by (lane_id, cursor): chunk ids restart per lane, so a cursor
            # key alone collides across lanes — two lanes mid-split would then
            # corrupt each other's token tally.
            key = (lane_id, record.meta.cursor.as_key())
            if seg.is_last:
                acc = self._wrap_comp_emitted.pop(key, None)
                if acc is None:
                    acc = defaultdict(int)
                for cid, T in targets.items():
                    component_token_counts[cid] += T - acc[cid]
            else:
                acc = self._wrap_comp_emitted.setdefault(key, defaultdict(int))
                for cid, T in targets.items():
                    contrib = (T * seg.length) // record_len
                    acc[cid] += contrib
                    component_token_counts[cid] += contrib

        total_length = sum(seg.length for seg in segments)
        # Stateless deterministic primary cursor: first segment's record cursor
        # with the segment's ``start`` offset as the lineage index. Unique across
        # consecutive bins because either the first-record cursor differs, or the
        # same record spans bins and its slice ``start`` advances by max_length.
        first = segments[0]
        primary_cursor = first.record.meta.cursor.child(first.start)
        return pack_meta(
            primary_cursor=primary_cursor,
            contributors=contributors,
            lane_id=lane_id,
            component_sample_counts=dict(component_sample_counts),
            component_token_counts=(
                dict(component_token_counts) if component_token_counts else None
            ),
            tags={
                "_packing_metadata": {
                    "num_sequences": len(segments),
                    # Real (non-pad) tokens, so packing_efficiency is a
                    # meaningful cross-algorithm comparison metric.
                    "total_length": total_length,
                    "packing_efficiency": total_length / self.max_length,
                    # Trailing pad tokens (flat); None for envelope.
                    "padding_length": self._serializer.padding_length(total_length),
                }
            },
        )

    # ------------------------------------------------------------------
    # wrap
    # ------------------------------------------------------------------

    def _resolve_wrap_field(self, record: SampleRecord, key: PackKey) -> str:
        """Determine which payload field of ``record`` carries the sliceable sequence.

        Uses the explicit field name set on the operator if provided; otherwise
        auto-detects and caches per lane, so every record in a lane (across
        domains) must resolve to the same field.
        """
        payload = record.payload
        if not isinstance(payload, dict):
            raise TypeError(
                "algorithm='wrap' requires dict payloads to slice; got payload "
                + f"type {type(payload).__name__}"
            )

        if self.wrap_field is not None:
            if self.wrap_field not in payload:
                raise ValueError(
                    f"algorithm='wrap': field {self.wrap_field!r} not found "
                    + f"in payload (keys: {list(payload.keys())})"
                )
            return self.wrap_field

        detected = detect_length_field(payload)
        if detected is None:
            raise ValueError(
                "algorithm='wrap': cannot auto-detect length field. "
                + f"Payload keys: {list(payload.keys())}. "
                + f"Expected one of: {TOKEN_FIELD_CANDIDATES}"
            )

        lane_id = key[0]
        cached = self._wrap_lane_auto_field.get(lane_id)
        if cached is None:
            self._wrap_lane_auto_field[lane_id] = detected
        elif cached != detected:
            raise ValueError(
                "algorithm='wrap': inconsistent auto-detected length field "
                f"for lane {lane_id}: stream started with {cached!r} but this "
                f"record uses {detected!r}. Set tokens_field to an explicit field "
                "or use homogeneous payloads."
            )
        return detected

    def _wrap_pack(self, key: PackKey, elem: SampleRecord) -> list[SampleRecord]:
        """Stream ``elem`` through the per-group wrap buffer and emit full bins.

        Tokens flow into a single FIFO buffer; whenever the buffer holds at least
        ``max_length`` tokens we slice off exactly that many and emit one record.
        Empty records carry no tokens but still emit tombstones so their
        contributor offsets close.
        """
        lane_id = key[0]
        # Measure the resolved wrap field directly; avoids repeated auto-detection.
        wrap_field = self._resolve_wrap_field(elem, key)
        payload = elem.payload
        assert isinstance(payload, dict)  # guaranteed by _resolve_wrap_field
        seq_len = _get_length(payload[wrap_field], wrap_field)
        if seq_len <= 0:
            return tombstones_for_record(elem)

        segments = self._wrap_segments[key]
        segments.append(
            Segment(
                record=elem,
                start=0,
                end=seq_len,
                seq_len=seq_len,
                field=wrap_field,
                is_last=False,
                is_slice=True,
            )
        )
        # Keep the hot buffered-token count local across the drain.
        total = self._wrap_total[key] + seq_len

        outputs: list[SampleRecord] = []
        max_length = self.max_length
        while total >= max_length:
            bin_slices: list[Segment] = []
            remaining = max_length
            while remaining > 0:
                seg = segments[0]
                seg_len = seg.end - seg.start
                if seg_len <= remaining:
                    # Reuse fully drained segments; mark them as closers.
                    seg.is_last = True
                    bin_slices.append(seg)
                    segments.popleft()
                    remaining -= seg_len
                    total -= seg_len
                else:
                    # Split off the consumed prefix and keep the remainder buffered.
                    new_start = seg.start + remaining
                    bin_slices.append(
                        Segment(
                            record=seg.record,
                            start=seg.start,
                            end=new_start,
                            seq_len=seg.seq_len,
                            field=seg.field,
                            is_last=False,
                            is_slice=True,
                        )
                    )
                    seg.start = new_start
                    total -= remaining
                    remaining = 0

            outputs.append(self._emit_bin(bin_slices, lane_id))

        self._wrap_total[key] = total
        return outputs


class PackSequences(DefaultSetup):
    """Pack variable-length sequences into bins.

    Bin assignment is deterministic regardless of parallelism. The ``output``
    mode selects how each finished bin is serialized (envelope list vs flat
    training record); the Pipeline exposes these as ``pack_sequences`` and
    ``pack_flat``. See :class:`PackingAccumulator` for the packing logic.
    """

    def __init__(
        self,
        max_length: int,
        num_bins: int,
        length_fn: Callable[[SampleRecord], int] | None = None,
        algorithm: Literal["first_fit", "best_fit", "wrap"] = "first_fit",
        *,
        output: Literal["envelope", "flat"] = "envelope",
        tokens_field: str = "auto",
        drop_oversized: bool = True,
        min_sequence_length: int = 1,
        shuffle_strategy: Literal["random", "length", None] = None,
        shuffle_seed: Optional[int] = None,
        pack_payloads: str | Callable[[list[Any]], Any] = "keep_list",
        flush_strategy: Literal["fifo", "fullest"] = "fifo",
        emit_positions: bool = True,
        pad_token_id: int | None = None,
        homogeneity: Literal["none", "full"] = "none",
    ) -> None:
        """Initialize the PackSequences operator.

        Args:
            max_length: Maximum length for packed bins.
            num_bins: Number of bins to maintain per lane (first_fit/best_fit).
            length_fn: Optional callable measuring a record's packing length —
                for ``output="envelope"`` with first_fit/best_fit only. Those
                pack whole records, so the length can be anything (e.g. a
                precomputed ``length`` field, with no token field to slice).
                ``None`` (default) measures ``len(payload[tokens_field])``. Not
                allowed with ``output="flat"`` or ``algorithm="wrap"``: those
                slice/concatenate ``tokens_field``, so the packing length is
                necessarily ``len(payload[tokens_field])`` and a separate
                ``length_fn`` could only disagree with it — set ``tokens_field``
                instead. Note this length also seeds per-component token
                apportionment, so for token-weighted mixtures it should reflect
                the true token count; it may be called more than once per record,
                so keep it cheap.
            algorithm: ``"first_fit"`` (default), ``"best_fit"``, or ``"wrap"``.
                wrap slices records to fill bins exactly; first/best pack whole
                records (and, in flat output, pad partial bins).
            output: ``"envelope"`` emits ``{"packed_samples": [...]}`` (lossless
                ordered list; ``pack_payloads`` applies). ``"flat"`` emits a
                fixed-length training record ``{tokens_field: concat[+pad],
                "positions"?}`` that ``to_training`` consumes directly. flat keeps
                only the token field and length-aligned sliceable fields; scalar
                and non-aligned payload is dropped (envelope preserves everything).
            tokens_field: Token field to slice/concatenate. ``"auto"`` detects
                from common candidates; or an explicit field name.
            drop_oversized: If True, drop sequences longer than max_length
                (first/best emit a tombstone per drop). Must be False for wrap.
            min_sequence_length: Minimum remaining capacity below which a bin is
                emitted (first_fit/best_fit). With 0, an exactly-full bin is not
                auto-emitted until num_bins pressure or flush — keep it >=1 for
                eager emission.
            shuffle_strategy: Strategy for ordering sequences before packing.
            shuffle_seed: Seed for random shuffling.
            pack_payloads: Envelope-only merge of the segment list. "keep_list"
                (default), "torch_tensor", "numpy_array", or a callable.
            flush_strategy: Strategy for flushing bins when num_bins is reached.
            emit_positions: Flat output only — include the ``positions`` array
                (document boundaries). Defaults to True.
            pad_token_id: Flat output only — fill value for the token field when
                padding partial first_fit/best_fit bins (aligned fields pad with
                0). Required for those algorithms; unused for wrap. Any embeddable
                id works — pad is masked from the loss by position, not by id.
            homogeneity: If ``"full"``, no packed sample combines records from
                different mixing domains (mixture components); needs
                single-component records, so place packing before any op that
                merges components. ``"none"`` (default) mixes freely. Packing
                keeps per-domain state, so size ``num_bins`` for the domain count.

                With ``algorithm="wrap"`` a split document is counted once, on its
                closing bin, so interior bins carry empty
                ``component_sample_counts``; attribute domains by
                ``component_token_counts`` instead (e.g. a downstream
                ``ensure_mixture(weight="tokens")``).
        """
        DefaultSetup.__init__(self)

        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if min_sequence_length < 0:
            raise ValueError("min_sequence_length must be non-negative")
        if num_bins <= 0:
            raise ValueError("num_bins must be positive")
        if algorithm not in ("first_fit", "best_fit", "wrap"):
            raise ValueError(f"Unknown algorithm: {algorithm}")
        if output not in ("envelope", "flat"):
            raise ValueError(f"Unknown output: {output!r}")
        if homogeneity not in ("none", "full"):
            raise ValueError(
                f"Unknown homogeneity: {homogeneity!r} (expected 'none' or 'full')"
            )
        if length_fn is not None and not callable(length_fn):
            raise ValueError(
                "length_fn must be a callable or None; to select a field by name "
                "use tokens_field=..."
            )
        # In flat/wrap the packing length is the sliced/concatenated field's
        # length, so a custom measure could only disagree with it.
        if callable(length_fn) and (output == "flat" or algorithm == "wrap"):
            raise ValueError(
                "length_fn is only supported for output='envelope' with "
                "first_fit/best_fit (whole-record packing by an arbitrary "
                "length). With output='flat' or algorithm='wrap' the length is "
                "len(payload[tokens_field]); set tokens_field and leave length_fn unset."
            )

        if algorithm == "wrap" and drop_oversized:
            raise ValueError(
                "drop_oversized=True is not allowed with algorithm='wrap'; wrap "
                "slices oversized inputs across multiple bins, so the concept of "
                "'oversized' does not apply."
            )

        # Flat first/best pads partial bins to max_length, so it needs a pad id.
        if output == "flat" and algorithm != "wrap" and pad_token_id is None:
            raise ValueError(
                f"pack_flat with algorithm={algorithm!r} pads partial bins to "
                "max_length, so pad_token_id is required."
            )

        # Reject params that the chosen output silently ignores, so a mis-wired
        # operator fails loudly instead of dropping the setting.
        if output == "envelope" and pad_token_id is not None:
            raise ValueError("pad_token_id only applies to output='flat'.")
        if output == "flat" and pack_payloads != "keep_list":
            raise ValueError("pack_payloads only applies to output='envelope'.")

        self.max_length = max_length
        self.algorithm = algorithm
        self.output = output
        self.tokens_field = tokens_field
        self.drop_oversized = drop_oversized
        self.min_sequence_length = min_sequence_length
        self.shuffle_strategy = shuffle_strategy
        self.shuffle_seed = shuffle_seed if shuffle_seed is not None else 0
        self.num_bins = num_bins
        self.flush_strategy = flush_strategy
        self.emit_positions = emit_positions
        self.pad_token_id = pad_token_id
        self.homogeneity = homogeneity

        # ``"full"`` keys packing state on the record's sole component so a packed
        # sample never spans domains; ``"none"`` (no domain_fn) mixes freely.
        self._domain_fn: Callable[[SampleRecord], Any] | None = (
            _component_domain if homogeneity == "full" else None
        )

        # Resolve the length function. tokens_field identifies the field (for
        # wrap slicing and flat concatenation); length_fn only measures.
        self._wrap_field: str | None = None if tokens_field == "auto" else tokens_field
        if callable(length_fn):
            self.length_fn: Callable[[SampleRecord], int] = length_fn
        elif tokens_field == "auto":
            self.length_fn = lambda r: extract_length(r, None)
        else:
            self.length_fn = lambda r, f=tokens_field: extract_length(r, f)

        # Envelope payload merge (ignored in flat output).
        self._pack_payloads_fn = self._resolve_pack_payloads_fn(pack_payloads)

        self._serializer = self._make_serializer()

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=False,
            batch_shape_sensitive=False,
            # Serial state lives in the accumulator, so the op needs none.
            requires_serial_state=False,
            preserves_cursor_order=False,
        )

    def _make_serializer(self) -> _EnvelopeSerializer | _FlatSerializer:
        """Build a fresh stateless serializer for the configured output format."""
        if self.output == "flat":
            return _FlatSerializer(
                max_length=self.max_length,
                tokens_field=self.tokens_field,
                pad_token_id=self.pad_token_id,
                emit_positions=self.emit_positions,
            )
        return _EnvelopeSerializer(self._pack_payloads_fn)

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return PackingAccumulator(
            max_length=self.max_length,
            num_bins=self.num_bins,
            length_fn=self.length_fn,
            algorithm=self.algorithm,
            drop_oversized=self.drop_oversized,
            min_sequence_length=self.min_sequence_length,
            shuffle_strategy=self.shuffle_strategy,
            shuffle_seed=self.shuffle_seed,
            flush_strategy=self.flush_strategy,
            serializer=self._make_serializer(),
            wrap_field=self._wrap_field,
            domain_fn=self._domain_fn,
        )

    def _resolve_pack_payloads_fn(
        self, pack_payloads: str | Callable[[list[Any]], Any]
    ) -> Callable[[list[Any]], Any]:
        """Resolve pack_payloads parameter to a callable function."""
        if callable(pack_payloads):
            return pack_payloads

        if pack_payloads == "keep_list":
            return lambda payloads: payloads

        if pack_payloads == "torch_tensor":
            return self._pack_torch_tensors

        if pack_payloads == "numpy_array":
            return self._pack_numpy_arrays

        raise ValueError(
            f"Unknown pack_payloads option: {pack_payloads}. "
            "Must be one of: 'keep_list', 'torch_tensor', 'numpy_array', or a callable."
        )

    @staticmethod
    def _pack_torch_tensors(payloads: list[Any]) -> Any:
        """Pack PyTorch tensors by concatenating along first dimension."""
        try:
            import torch
        except ImportError:
            raise ImportError(
                "pack_payloads='torch_tensor' requires PyTorch to be installed"
            )

        if not payloads:
            return payloads

        with _tensor_lock_ctx():
            if isinstance(payloads[0], dict):
                result = {}
                for key in payloads[0].keys():
                    values = [p[key] for p in payloads]
                    if all(isinstance(v, torch.Tensor) for v in values):
                        result[key] = torch.cat(values, dim=0)
                    else:
                        non_tensor_types = {
                            type(v).__name__
                            for v in values
                            if not isinstance(v, torch.Tensor)
                        }
                        raise TypeError(
                            f"pack_payloads='torch_tensor' requires all values to be torch.Tensor, "
                            + f"but found non-tensor types: {non_tensor_types}"
                        )
                return result

            if all(isinstance(p, torch.Tensor) for p in payloads):
                return torch.cat(payloads, dim=0)

            non_tensor_types = {
                type(p).__name__ for p in payloads if not isinstance(p, torch.Tensor)
            }
            raise TypeError(
                f"pack_payloads='torch_tensor' requires all payloads to be torch.Tensor, "
                + f"but found non-tensor types: {non_tensor_types}"
            )

    @staticmethod
    def _pack_numpy_arrays(payloads: list[Any]) -> Any:
        """Pack NumPy arrays by concatenating along first axis."""
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "pack_payloads='numpy_array' requires NumPy to be installed"
            )

        if not payloads:
            return payloads

        if isinstance(payloads[0], dict):
            result = {}
            for key in payloads[0].keys():
                values = [p[key] for p in payloads]
                if all(isinstance(v, np.ndarray) for v in values):
                    result[key] = np.concatenate(values, axis=0)
                else:
                    non_array_types = {
                        type(v).__name__
                        for v in values
                        if not isinstance(v, np.ndarray)
                    }
                    raise TypeError(
                        f"pack_payloads='numpy_array' requires all values to be numpy.ndarray, "
                        + f"but found non-array types: {non_array_types}"
                    )
            return result

        if all(isinstance(p, np.ndarray) for p in payloads):
            return np.concatenate(payloads, axis=0)

        non_array_types = {
            type(p).__name__ for p in payloads if not isinstance(p, np.ndarray)
        }
        raise TypeError(
            f"pack_payloads='numpy_array' requires all payloads to be numpy.ndarray, "
            + f"but found non-array types: {non_array_types}"
        )

    def _materialize(self, elem: SampleRecord) -> SampleRecord:
        """Serialize one :class:`_DeferredBin` into its real payload.

        Records without a deferred payload (e.g. tombstones) pass through.
        """
        payload = elem.payload
        if isinstance(payload, _DeferredBin):
            return SampleRecord(
                meta=elem.meta,
                payload=self._serializer.build_payload(payload.segments),
            )
        return elem

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        """Materialize one packed bin (the accumulator decided its composition)."""
        return [self._materialize(elem)]

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        """Materialize each :class:`_DeferredBin` into its real payload.

        The slice/concatenate/pad runs here rather than in the accumulator, so
        it fans out across ``parallelism`` workers and its GIL-releasing concat
        overlaps the pump thread that runs the accumulator.
        """
        return [self._materialize(elem) for elem in elems]
