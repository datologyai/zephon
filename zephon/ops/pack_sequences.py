# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Sequence packing operator for grouping variable-length sequences into fixed-length bins."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

from zephon.core.accumulators import Accumulator, ReadyBatch
from zephon.core.children import collect_pack_contributions, pack_meta
from zephon.core.constants import SampleRecord
from zephon.core.op_base import DefaultSetup
from zephon.core.traits import OpTraits
from zephon.utils.length_extraction import extract_length
from zephon.utils.seeding import batch_seed
from zephon.utils.torch_compat import _tensor_lock_ctx


@dataclass(slots=True)
class Bin:
    """Represents a bin for packing sequences."""

    samples: list[SampleRecord]
    remaining: int


class PackingAccumulator(Accumulator[SampleRecord]):
    """Accumulator that packs variable-length sequences into fixed-length bins.

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
        algorithm: Literal["first_fit", "best_fit"],
        drop_oversized: bool,
        min_sequence_length: int,
        shuffle_strategy: Literal["random", "length", None],
        shuffle_seed: int,
        pack_payloads_fn: Callable[[list[Any]], Any],
        flush_strategy: Literal["fifo", "fullest"],
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

        # Per-lane state: lane_id -> list[Bin]
        self._bins: defaultdict[int, list[Bin]] = defaultdict(list)

    @property
    def reads_payload(self) -> bool:
        return True

    def has_pending_data(self) -> bool:
        """Return True if there are any bins with samples."""
        return any(bins for bins in self._bins.values())

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


class PackSequences(DefaultSetup):
    """Pack variable-length sequences into fixed-length bins using first-fit or best-fit algorithms.

    This operator uses a PackingAccumulator to maintain per-lane bins on the pump
    thread. The packing decisions are deterministic regardless of parallelism level.

    See PackingAccumulator for the actual packing logic.
    """

    def __init__(
        self,
        max_length: int,
        num_bins: int,
        length_fn: Callable[[SampleRecord], int] | Literal["auto"] | str = "auto",
        algorithm: Literal["first_fit", "best_fit"] = "first_fit",
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
            algorithm: Packing algorithm ("first_fit" or "best_fit").
            drop_oversized: If True, drop sequences longer than max_length.
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
        if callable(length_fn):
            self.length_fn: Callable[[SampleRecord], int] = length_fn
        elif length_fn == "auto":
            # Auto-detect token field from common candidates
            self.length_fn = lambda r: extract_length(r, None)
        else:
            # Explicit field name
            _field = length_fn  # Capture for lambda
            self.length_fn = lambda r, f=_field: extract_length(r, f)

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
