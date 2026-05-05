# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Sequence packing operator for grouping variable-length sequences into bins."""

from __future__ import annotations

import logging
import random
from collections import defaultdict, deque
from collections.abc import Callable
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
    collect_pack_contributions,
    pack_meta,
    tombstones_for_record,
)
from zephon.core.constants import ContributorRef, SampleRecord
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


@dataclass(slots=True)
class Bin:
    """Represents a bin for packing sequences."""

    samples: list[SampleRecord]
    remaining: int


@dataclass(slots=True)
class _WrapSlice:
    """One contiguous slice of a record's wrap field used for the 'wrap' algorithm."""

    record: SampleRecord
    start: int
    end: int
    field: str
    seq_len: int
    is_last: bool = False

    @property
    def length(self) -> int:
        return self.end - self.start


class PackingAccumulator(Accumulator[SampleRecord]):
    """Accumulator that packs variable-length sequences into bins.

    This accumulator runs on the pump thread and maintains per-lane bins.
    It handles all the bin management and packing algorithm logic, emitting
    ready bins when they reach capacity or can't fit more sequences.

    The PackSequences operator uses this accumulator to ensure deterministic
    packing regardless of parallelism level.
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
        pack_payloads_fn: Callable[[list[Any]], Any],
        flush_strategy: Literal["fifo", "fullest"],
        wrap_field: str | None = None,
    ) -> None:
        self.max_length = max_length
        self.num_bins = num_bins
        self.length_fn = length_fn
        self.algorithm = algorithm
        self.drop_oversized = drop_oversized
        self.min_sequence_length = min_sequence_length
        self.shuffle_strategy = shuffle_strategy
        self.shuffle_seed = shuffle_seed
        self._pack_payloads_fn = pack_payloads_fn
        self.flush_strategy = flush_strategy
        self.wrap_field = wrap_field

        # Per-lane state for first_fit/best_fit: lane_id -> list[Bin].
        self._bins: defaultdict[int, list[Bin]] = defaultdict(list)

        # Per-lane state for wrap: lane_id -> deque of buffered _WrapSlice
        # entries.  Each entry's tokens still in the buffer are
        # ``record.payload[field][start:end]``; ``start`` advances as the
        # buffer is drained into emitted bins.
        self._wrap_segments: defaultdict[int, deque[_WrapSlice]] = defaultdict(deque)
        # Total buffered token count per lane (sum of segment lengths).
        self._wrap_total: defaultdict[int, int] = defaultdict(int)
        # When length_fn is auto, first resolved field name per lane (must stay consistent).
        self._wrap_lane_auto_field: dict[int, str] = {}
        # Per-base cursor: component token counts already charged on non-final wrap slices
        # (floor parts); closed when is_last so the last slice absorbs rounding remainder.
        self._wrap_comp_emitted: dict[tuple[Any, ...], defaultdict[int, int]] = {}

    @property
    def reads_payload(self) -> bool:
        return True

    def has_pending_data(self) -> bool:
        """Return True if there are any bins or wrap-mode segments with data."""
        if any(bins for bins in self._bins.values()):
            return True
        return any(total > 0 for total in self._wrap_total.values())

    def push_many(
        self, elems: Sequence[SampleRecord]
    ) -> list[ReadyBatch[SampleRecord]]:
        """Accumulate records and emit packed bins when ready."""
        if not elems:
            return []

        work_list = list(elems)

        # Apply shuffle strategy to order sequences
        if self.shuffle_strategy == "random":
            rng = random.Random(batch_seed(self.shuffle_seed, work_list))
            rng.shuffle(work_list)
        elif self.shuffle_strategy == "length":
            work_list = sorted(
                work_list,
                key=lambda rec: (-self.length_fn(rec), rec.meta.cursor.as_key()),
            )

        ready: list[ReadyBatch[SampleRecord]] = []

        for elem in work_list:
            lane_id = elem.meta.lane_id

            # The wrap algorithm streams tokens through a per-lane buffer and
            # slices into bins of exactly ``max_length``; oversized inputs
            # naturally span multiple bins so the >max_length / drop_oversized
            # branch does not apply here.
            if self.algorithm == "wrap":
                ready.extend(([rec], 0) for rec in self._wrap_pack(lane_id, elem))
                continue

            seq_len = self.length_fn(elem)

            # Handle oversized sequences
            if seq_len > self.max_length:
                if self.drop_oversized:
                    continue
                raise ValueError(
                    f"Sequence length {seq_len} exceeds max_length {self.max_length}"
                )

            # Pack using selected algorithm
            if self.algorithm == "first_fit":
                packed = self._first_fit_pack(lane_id, elem, seq_len)
            elif self.algorithm == "best_fit":
                packed = self._best_fit_pack(lane_id, elem, seq_len)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            for rec in packed:
                ready.append(([rec], 0))

        return ready

    def flush(self, *, reset: bool = False) -> list[ReadyBatch[SampleRecord]]:
        """Emit any remaining partially-filled bins."""
        ready: list[ReadyBatch[SampleRecord]] = []
        for lane_id, bins in self._bins.items():
            for bin_data in bins:
                if bin_data.samples:
                    rec = self._create_packed_record(bin_data, lane_id)
                    ready.append(([rec], 0))
        self._bins.clear()

        # Wrap mode: drop any tail that did not fill a full ``max_length``,
        # emit tombstones so dropped records still close contributor offsets,
        # and clear buffered state.  We delegate to ``tombstones_for_record``
        # which mirrors the precedent in ``ReplayFilter`` / ``MapTransform``:
        # only contributors with ``is_last_child=True`` produce a tombstone,
        # so non-closing contributors (whose closing sibling lives elsewhere
        # in the stream) are not falsely advanced to closed.
        tail_drop_lanes: list[tuple[int, int]] = []
        wrap_tombstones: list[ReadyBatch[SampleRecord]] = []
        for lane_id, total in list(self._wrap_total.items()):
            if total <= 0:
                continue
            tail_drop_lanes.append((lane_id, total))
            for seg in self._wrap_segments[lane_id]:
                rec = seg.record
                self._wrap_comp_emitted.pop(rec.meta.cursor.as_key(), None)
                for tomb in tombstones_for_record(rec):
                    wrap_tombstones.append(([tomb], 0))

        if tail_drop_lanes:
            dropped_tokens = sum(t for _, t in tail_drop_lanes)
            logger.warning(
                "PackSequences wrap: dropping %s trailing token(s) across %d lane(s) "
                "that did not fill a full max_length=%d bin; emitted %d tombstone(s) "
                "to close contributor offsets.",
                dropped_tokens,
                len(tail_drop_lanes),
                self.max_length,
                len(wrap_tombstones),
            )

        self._wrap_segments.clear()
        self._wrap_total.clear()
        self._wrap_lane_auto_field.clear()
        ready.extend(wrap_tombstones)

        return ready

    def _first_fit_pack(
        self, lane_id: int, seq: SampleRecord, seq_len: int
    ) -> list[SampleRecord]:
        """Try to pack sequence using first-fit algorithm."""
        bins = self._bins[lane_id]
        outputs: list[SampleRecord] = []

        for bin_data in bins:
            if bin_data.remaining >= seq_len:
                outputs.extend(
                    self._add_sample_to_bin(bin_data, bins, seq, seq_len, lane_id)
                )
                return outputs

        outputs.extend(self._create_bin_with_sample(bins, seq, seq_len, lane_id))
        return outputs

    def _best_fit_pack(
        self, lane_id: int, seq: SampleRecord, seq_len: int
    ) -> list[SampleRecord]:
        """Try to pack sequence using best-fit algorithm."""
        bins = self._bins[lane_id]
        outputs: list[SampleRecord] = []

        best_bin = None
        best_remaining = self.max_length + 1

        for bin_data in bins:
            remaining = bin_data.remaining
            if remaining >= seq_len and remaining < best_remaining:
                best_bin = bin_data
                best_remaining = remaining

        if best_bin is not None:
            outputs.extend(
                self._add_sample_to_bin(best_bin, bins, seq, seq_len, lane_id)
            )
            return outputs

        outputs.extend(self._create_bin_with_sample(bins, seq, seq_len, lane_id))
        return outputs

    def _create_bin_with_sample(
        self,
        bins: list[Bin],
        seq: SampleRecord,
        seq_len: int,
        lane_id: int,
    ) -> list[SampleRecord]:
        """Create a new bin, add a sample, and emit if full."""
        outputs = self._enforce_max_bins(bins, lane_id)

        new_bin = Bin(samples=[seq], remaining=self.max_length - seq_len)
        if new_bin.remaining < self.min_sequence_length:
            outputs.append(self._create_packed_record(new_bin, lane_id))
        else:
            bins.append(new_bin)
        return outputs

    def _add_sample_to_bin(
        self,
        bin_data: Bin,
        bins: list[Bin],
        seq: SampleRecord,
        seq_len: int,
        lane_id: int,
    ) -> list[SampleRecord]:
        """Add a sample to an existing bin and emit if full."""
        bin_data.samples.append(seq)
        bin_data.remaining -= seq_len
        outputs: list[SampleRecord] = []
        if bin_data.remaining < self.min_sequence_length:
            bins.remove(bin_data)
            outputs.append(self._create_packed_record(bin_data, lane_id))
        return outputs

    def _enforce_max_bins(
        self,
        bins: list[Bin],
        lane_id: int,
    ) -> list[SampleRecord]:
        """Enforce num_bins limit by flushing bins if necessary."""
        outputs: list[SampleRecord] = []
        while len(bins) >= self.num_bins and bins:
            if self.flush_strategy == "fifo":
                bin_to_flush = bins.pop(0)
            elif self.flush_strategy == "fullest":
                fullest_idx = min(range(len(bins)), key=lambda i: bins[i].remaining)
                bin_to_flush = bins.pop(fullest_idx)
            else:
                raise ValueError(f"Unknown flush_strategy: {self.flush_strategy}")
            outputs.append(self._create_packed_record(bin_to_flush, lane_id))
        return outputs

    def _create_packed_record(self, bin_data: Bin, lane_id: int) -> SampleRecord:
        """Create a packed SampleRecord from a bin.

        Aggregates component contributions from all packed samples:
        - component_sample_counts: sum of sample counts per component
        - component_token_counts: sum of token counts per component (using length_fn)

        This enables ensure_mixture to correctly track which components are
        represented in a packed sample and by how much.
        """
        samples: list[SampleRecord] = bin_data.samples
        if not samples:
            raise ValueError("Cannot create packed record from empty bin")

        total_length = self.max_length - bin_data.remaining
        num_sequences = len(samples)
        packing_efficiency = total_length / self.max_length

        raw_payloads = [s.payload for s in samples]
        packed_payload_value = self._pack_payloads_fn(raw_payloads)

        packed_payload: dict[str, Any] = {
            "packed_samples": packed_payload_value,
        }

        contributors, component_sample_counts, component_token_counts = (
            collect_pack_contributions(samples, self.length_fn)
        )

        base_meta = samples[0].meta
        primary_cursor = base_meta.cursor.child(0)

        packed_meta = pack_meta(
            primary_cursor=primary_cursor,
            contributors=contributors,
            lane_id=lane_id,
            component_sample_counts=component_sample_counts,
            component_token_counts=component_token_counts,
            tags={
                "_packing_metadata": {
                    "num_sequences": num_sequences,
                    "total_length": total_length,
                    "packing_efficiency": packing_efficiency,
                }
            },
        )

        return SampleRecord(meta=packed_meta, payload=packed_payload)

    # ------------------------------------------------------------------
    # Wrap algorithm
    # ------------------------------------------------------------------

    def _resolve_wrap_field(self, record: SampleRecord, lane_id: int) -> str:
        """Determine which payload field of ``record`` carries the sliceable sequence.

        Uses the explicit field name set on the operator if provided; otherwise
        auto-detects via ``detect_length_field`` over ``TOKEN_FIELD_CANDIDATES``.
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

        cached = self._wrap_lane_auto_field.get(lane_id)
        if cached is None:
            self._wrap_lane_auto_field[lane_id] = detected
        elif cached != detected:
            raise ValueError(
                "algorithm='wrap': inconsistent auto-detected length field "
                f"for lane {lane_id}: stream started with {cached!r} but this "
                f"record uses {detected!r}. Set length_fn to an explicit field "
                "or use homogeneous payloads."
            )
        return detected

    def _wrap_pack(self, lane_id: int, elem: SampleRecord) -> list[SampleRecord]:
        """Stream ``elem`` through the per-lane wrap buffer and emit full bins.

        Tokens from incoming records flow into a single FIFO buffer, and
        whenever the buffer holds at least ``max_length`` tokens we slice off
        exactly that many and emit one packed record.
        """
        seq_len = self.length_fn(elem)
        if seq_len <= 0:
            # Empty record: no tokens to enqueue.
            return []

        wrap_field = self._resolve_wrap_field(elem, lane_id)

        segments = self._wrap_segments[lane_id]
        segments.append(
            _WrapSlice(
                record=elem, start=0, end=seq_len, field=wrap_field, seq_len=seq_len
            )
        )
        self._wrap_total[lane_id] += seq_len

        outputs: list[SampleRecord] = []
        while self._wrap_total[lane_id] >= self.max_length:
            bin_slices: list[_WrapSlice] = []
            remaining = self.max_length
            while remaining > 0:
                seg = segments[0]
                take = min(remaining, seg.length)
                new_start = seg.start + take
                # ``is_last`` is True exactly when this slice consumes the
                # record's final token (i.e. its segment is fully drained).
                consumes_record = new_start == seg.end
                bin_slices.append(
                    _WrapSlice(
                        record=seg.record,
                        start=seg.start,
                        end=new_start,
                        field=seg.field,
                        seq_len=seg.seq_len,
                        is_last=consumes_record,
                    )
                )
                if consumes_record:
                    segments.popleft()
                else:
                    seg.start = new_start
                remaining -= take
                self._wrap_total[lane_id] -= take

            outputs.append(self._create_wrapped_record(bin_slices, lane_id))

        return outputs

    @staticmethod
    def _wrap_sliceable_field_names(
        payload: dict[str, Any], wrap_field: str, seq_len: int
    ) -> frozenset[str]:
        """Fields to slice in lockstep: ``wrap_field`` plus same-length sliceable fields.

        Scalar metadata (``int``, ``float``, ``bool``, ``None``) and other
        non-sequence values are ignored and do not appear in the packed output.
        Any sequence-like field whose length differs from ``seq_len`` raises.
        """
        if wrap_field not in payload:
            raise ValueError(
                f"algorithm='wrap': field {wrap_field!r} missing from payload "
                f"(keys: {list(payload.keys())})"
            )
        wf_val = payload[wrap_field]
        if _get_length(wf_val, wrap_field) != seq_len:
            raise ValueError(
                f"algorithm='wrap': field {wrap_field!r} length does not match record "
                f"sequence length {seq_len}"
            )
        names: set[str] = {wrap_field}
        for k, v in payload.items():
            if k == wrap_field:
                continue
            if isinstance(v, (bool, int, float, type(None))):
                continue
            try:
                length = _get_length(v, k)
            except TypeError:
                continue
            if length != seq_len:
                raise ValueError(
                    f"algorithm='wrap': payload key {k!r} has length {length}, "
                    f"expected {seq_len} to match field {wrap_field!r} for aligned slicing."
                )
            try:
                v[0:0]
            except Exception as e:
                raise ValueError(
                    f"algorithm='wrap': payload key {k!r} is not sliceable."
                ) from e
            names.add(k)
        return frozenset(names)

    def _record_component_token_targets(
        self, record: SampleRecord, record_len: int
    ) -> dict[int, int]:
        if record.meta.component_token_counts is not None:
            return dict(record.meta.component_token_counts)
        return _integer_apportion_tokens(
            record_len, dict(record.meta.component_sample_counts)
        )

    def _create_wrapped_record(
        self,
        bin_slices: list[_WrapSlice],
        lane_id: int,
    ) -> SampleRecord:
        """Build a packed record for one ``max_length``-sized wrap bin.

        ``bin_slices`` is a list of ``_WrapSlice`` entries in emission order
        whose lengths sum to ``max_length``.  ``slice.is_last`` indicates
        whether this slice consumed the record's final token, which controls
        ``ContributorRef.is_last_child`` on every contributor propagated from
        that record.
        """
        if not bin_slices:
            raise ValueError("Cannot build wrapped record from empty bin_slices")

        field_sets: list[frozenset[str]] = []
        for sl in bin_slices:
            payload = sl.record.payload
            assert isinstance(payload, dict)
            field_sets.append(
                self._wrap_sliceable_field_names(payload, sl.field, sl.seq_len)
            )
        first_fields = field_sets[0]
        for other in field_sets[1:]:
            if other != first_fields:
                raise ValueError(
                    "algorithm='wrap': inconsistent payload keys or sequence lengths "
                    "within one bin: "
                    f"{sorted(first_fields)!r} vs {sorted(other)!r}"
                )
        field_names = sorted(first_fields)

        per_field_parts: dict[str, list[Any]] = {k: [] for k in field_names}
        for sl in bin_slices:
            payload = sl.record.payload
            assert isinstance(payload, dict)
            for k in field_names:
                seq = payload[k]
                per_field_parts[k].append(seq[sl.start : sl.end])

        packed_entry = {
            k: self._concatenate_wrap_slices(per_field_parts[k]) for k in field_names
        }
        packed_payload: dict[str, Any] = {"packed_samples": [packed_entry]}

        # Aggregate contributors and component counts across slices.  Component
        # sample counts are charged on the closing slice (so a record split
        # across N bins contributes its sample count exactly once, on the
        # bin where its final token lands).  Component token counts use
        # floor splits on intermediate slices and a remainder fixup on the
        # slice with ``is_last`` so totals match ``component_token_counts`` /
        # apportioned targets per record.
        contributors: list[ContributorRef] = []
        component_sample_counts: dict[int, int] = defaultdict(int)
        component_token_counts: dict[int, int] = defaultdict(int)
        for sl in bin_slices:
            record = sl.record
            slice_len = sl.length
            for ref in record.meta.contribution_refs():
                contributors.append(
                    ContributorRef(
                        cursor=ref.cursor,
                        is_last_child=(ref.is_last_child and sl.is_last),
                    )
                )

            if sl.is_last:
                for cid, count in record.meta.component_sample_counts.items():
                    component_sample_counts[cid] += count

            record_len = sl.seq_len
            targets = self._record_component_token_targets(record, record_len)
            if not targets or record_len <= 0:
                continue

            key = record.meta.cursor.as_key()
            if sl.is_last:
                acc = self._wrap_comp_emitted.pop(key, None)
                if acc is None:
                    acc = defaultdict(int)
                for cid, T in targets.items():
                    component_token_counts[cid] += T - acc[cid]
            else:
                acc = self._wrap_comp_emitted.setdefault(key, defaultdict(int))
                for cid, T in targets.items():
                    contrib = (T * slice_len) // record_len
                    acc[cid] += contrib
                    component_token_counts[cid] += contrib

        # Stateless deterministic primary cursor: first slice's record cursor
        # with the slice's ``start`` offset as the lineage index.  This is
        # unique across consecutive bins because either (a) consecutive bins
        # have different first-record cursors, or (b) the same record spans
        # multiple bins, in which case its slice ``start`` advances by
        # ``max_length`` each bin so the child indices differ.
        first_slice = bin_slices[0]
        primary_cursor = first_slice.record.meta.cursor.child(first_slice.start)

        packed_meta = pack_meta(
            primary_cursor=primary_cursor,
            contributors=contributors,
            lane_id=lane_id,
            component_sample_counts=dict(component_sample_counts),
            component_token_counts=(
                dict(component_token_counts) if component_token_counts else None
            ),
            tags={
                "_packing_metadata": {
                    "num_sequences": len(bin_slices),
                    "total_length": self.max_length,
                    "packing_efficiency": 1.0,
                }
            },
        )

        return SampleRecord(meta=packed_meta, payload=packed_payload)

    @staticmethod
    def _concatenate_wrap_slices(sub_arrays: list[Any]) -> Any:
        """Concatenate wrap slices preserving the original sequence type.

        Sequence dimension is axis / dim 0 (1D tokens, ``(seq,)``, ``(seq, hidden)``, etc.).
        Supports list, tuple, ``numpy.ndarray``, and ``torch.Tensor``.
        Heterogeneous bins are not validated here; numpy/torch will raise
        their own clear errors when given mixed types.
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
            f"algorithm='wrap': unsupported slice type {type(first).__name__}; "
            "expected list, tuple, numpy.ndarray, or torch.Tensor"
        )


class PackSequences(DefaultSetup):
    """Pack variable-length sequences into bins using first-fit or best-fit algorithms.

    This operator uses a PackingAccumulator to maintain per-lane bins on the pump
    thread. The packing decisions are deterministic regardless of parallelism level.

    See PackingAccumulator for the actual packing logic.
    """

    def __init__(
        self,
        max_length: int,
        num_bins: int,
        length_fn: Callable[[SampleRecord], int] | Literal["auto"] | str = "auto",
        algorithm: Literal["first_fit", "best_fit", "wrap"] = "first_fit",
        *,
        drop_oversized: bool = True,
        min_sequence_length: int = 1,
        shuffle_strategy: Literal["random", "length", None] = None,
        shuffle_seed: Optional[int] = None,
        pack_payloads: str | Callable[[list[Any]], Any] = "keep_list",
        flush_strategy: Literal["fifo", "fullest"] = "fifo",
    ) -> None:
        """Initialize the PackSequences operator.

        Args:
            max_length: Maximum length for packed bins.
            num_bins: Number of bins to maintain per lane.
            length_fn: How to extract sequence length. Options:
                - "auto" (default): Auto-detect from common token fields
                  (input_ids, tokens, token_ids, ids).
                - Explicit field name (e.g., "input_ids"): Use that field.
                - Callable: Custom function taking SampleRecord, returning int.
                  Not allowed with ``algorithm="wrap"`` (the operator must
                  know which payload field to slice).
            algorithm: Packing algorithm.
                - ``"first_fit"`` (default): Place each record into the first
                  bin with enough remaining capacity; open a new bin if none
                  fits.  Cheap and order-preserving.
                - ``"best_fit"``: Place each record into the bin that leaves
                  the smallest remaining capacity (still >= seq_len), to
                  reduce wasted space at the cost of scanning all bins.
                - ``"wrap"``: wrap input data into output bins of fix length.
            drop_oversized: If True, drop sequences longer than max_length.
                Must be False when ``algorithm="wrap"`` (slicing handles
                oversized inputs naturally).
            min_sequence_length: Minimum expected sequence length.
            shuffle_strategy: Strategy for ordering sequences before packing.
            shuffle_seed: Seed for random shuffling.
            pack_payloads: How to combine payloads from multiple samples.
            flush_strategy: Strategy for flushing bins when limit is reached.
        """
        DefaultSetup.__init__(self)

        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if min_sequence_length < 0:
            raise ValueError("min_sequence_length must be non-negative")
        if num_bins <= 0:
            raise ValueError("num_bins must be positive")

        if algorithm == "wrap":
            if drop_oversized:
                raise ValueError(
                    "drop_oversized=True is not allowed with algorithm='wrap'; "
                    "wrap slices oversized inputs across multiple bins, so the "
                    "concept of 'oversized' does not apply."
                )
            if callable(length_fn):
                raise ValueError(
                    "algorithm='wrap' requires length_fn='auto' or an explicit "
                    "field name string; callable length_fn is not supported "
                    "because wrap must know which payload field to slice."
                )

        self.max_length = int(max_length)
        self.algorithm = algorithm
        self.drop_oversized = drop_oversized
        self.min_sequence_length = int(min_sequence_length)
        self.shuffle_strategy = shuffle_strategy
        self.shuffle_seed = int(shuffle_seed) if shuffle_seed is not None else 0
        self.num_bins = num_bins
        self.flush_strategy = flush_strategy

        # Set up length extraction function using shared utility.
        # "auto" = auto-detect, other string = explicit field, callable = custom.
        # For wrap mode we also remember the field name (or ``None`` for
        # auto-detect) so the accumulator can slice the right payload field.
        self._wrap_field: str | None = None
        if callable(length_fn):
            self.length_fn: Callable[[SampleRecord], int] = length_fn
        elif length_fn == "auto":
            # Auto-detect token field from common candidates
            self.length_fn = lambda r: extract_length(r, None)
        else:
            # Explicit field name
            _field = length_fn  # Capture for lambda
            self.length_fn = lambda r, f=_field: extract_length(r, f)
            self._wrap_field = _field

        # Set up payload packing function
        self._pack_payloads_fn = self._resolve_pack_payloads_fn(pack_payloads)

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=False,
            batch_shape_sensitive=False,
            # No longer needs serial state - accumulator handles it
            requires_serial_state=False,
            preserves_cursor_order=False,
        )

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
            pack_payloads_fn=self._pack_payloads_fn,
            flush_strategy=self.flush_strategy,
            wrap_field=self._wrap_field,
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
            + f"Must be one of: 'keep_list', 'torch_tensor', 'numpy_array', or a callable."
        )

    def _pack_torch_tensors(self, payloads: list[Any]) -> Any:
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

    def _pack_numpy_arrays(self, payloads: list[Any]) -> Any:
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

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        """Process a single sample record.

        The accumulator handles the actual packing logic. This method just
        passes through the record since the accumulator already packed it.
        """
        return [elem]

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        """Process multiple sample records.

        The accumulator handles the actual packing logic. This method just
        passes through the records since the accumulator already packed them.
        """
        return list(elems)
