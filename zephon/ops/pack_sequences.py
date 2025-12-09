# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Sequence packing operator for grouping variable-length sequences into fixed-length bins."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sized
from dataclasses import dataclass
from typing import Any, Literal, Optional

from zephon.core.children import pack_meta
from zephon.core.constants import ContributorRef, SampleRecord
from zephon.core.op_base import DefaultFinalize, DefaultSetup
from zephon.core.traits import Buffering, OpTraits
from zephon.utils.seeding import batch_seed


@dataclass(slots=True)
class Bin:
    """Represents a bin for packing sequences."""

    samples: list[SampleRecord]
    remaining: int


class PackSequences(DefaultSetup, DefaultFinalize[SampleRecord]):
    """Pack variable-length sequences into fixed-length bins using first-fit or best-fit algorithms.

    This operator maintains per-lane bins and packs sequences into bins up to max_length.
    It supports two packing algorithms:
    - first_fit: Places each sequence in the first bin that has enough space
    - best_fit: Places each sequence in the bin with the smallest remaining capacity that fits

    The operator maintains cross-invocation state (bins per lane) and requires
    parallelism=1 in deterministic mode to preserve determinism.

    Why `requires_serial_state=True`?
    --------------------------------
    PackSequences maintains partially-filled bins across multiple `process_one()` calls.
    This cross-invocation state enables efficient packing across batch boundaries.
    However, if multiple workers process the same lane, each would maintain separate bins,
    leading to non-deterministic packing. The `requires_serial_state` trait ensures
    parallelism=1 in deterministic mode, guaranteeing one worker per lane and
    deterministic packing decisions.

    Unlike ShuffleBuffer (which is stateless and uses runner-side buffering to process
    batches independently), PackSequences needs to maintain state for cross-batch packing
    efficiency.
    """

    def __init__(
        self,
        max_length: int,
        num_bins: int,
        length_fn: Callable[[SampleRecord], int] | str = "length",
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
            max_length: Maximum length for packed bins. Sequences are packed into bins
                up to this length.
            num_bins: Number of bins to maintain per lane. When the limit is reached, bins
                are flushed according to flush_strategy to make space for new bins.
            length_fn: Function or field name to extract sequence length from a SampleRecord.
                If a string, looks up that field in the payload dict. The field value can be:
                - An int (returns the value directly)
                - A tensor-like object (numpy arrays, torch tensors, JAX arrays, etc.):
                  extracts length from shape[0]
                - A sequence-like object (list, tuple, etc.): extracts length using len()
                If a callable, calls it with the SampleRecord and expects an int return value.
            algorithm: Packing algorithm to use. "first_fit" finds the first bin that fits,
                "best_fit" finds the bin with smallest remaining capacity.
            drop_oversized: If True, drop sequences longer than max_length. If False,
                raise an error when encountering oversized sequences.
            min_sequence_length: Minimum expected sequence length. Bins with remaining capacity
                less than this are flushed immediately. Defaults to 1.
            shuffle_strategy: Strategy for ordering sequences in process_many before packing.
                - "random": Randomly shuffle sequences (helps avoid pathological ordering)
                - "length": Sort by length descending (best-fit-decreasing for maximum efficiency)
                - None: No reordering (preserve input order)
                Defaults to None (preserve input order).
            shuffle_seed: Seed for random shuffling when shuffle_strategy="random". If None, uses 0.
                The actual seed combines this with the data values for deterministic shuffling.
            pack_payloads: How to combine payloads from multiple samples into a single packed payload.
                - "keep_list": Keep as list of payloads (default, backward compatible)
                - "torch_tensor": Concatenate PyTorch tensors along first dimension
                - "numpy_array": Concatenate NumPy arrays along first axis
                - Callable: Custom function that takes list[Any] (payloads) and returns combined payload.
                  For dict payloads, the function will be applied to each field independently.
            flush_strategy: Strategy for flushing bins when num_bins limit is reached.
                - "fifo": Flush oldest bins first (default, ensures fairness)
                - "fullest": Flush bins with smallest remaining capacity first (better packing efficiency)
                Defaults to "fifo".
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

        # Set up length extraction function
        if isinstance(length_fn, str):
            self.length_fn: Callable[[SampleRecord], int] = (
                lambda r: self._get_length_from_field(r, length_fn)
            )
        else:
            self.length_fn = length_fn

        # Set up payload packing function
        self._pack_payloads_fn = self._resolve_pack_payloads_fn(pack_payloads)

        # Per-lane state: lane_id -> list[Bin]
        # Each bin is a Bin instance with:
        #   - samples: list[SampleRecord]
        #   - remaining: int (remaining capacity)
        self._bins: defaultdict[int, list[Bin]] = defaultdict(list)

    def _get_length_from_field(self, record: SampleRecord, field: str) -> int:
        """Extract length from a field in the payload.

        Supports:
        - int: Returns the value directly
        - list/tuple: Returns len()
        - tensor-like objects (numpy arrays, torch tensors, JAX arrays): Returns shape[0]
        - Any sequence-like object: Returns len()
        """
        if not isinstance(record.payload, dict):
            raise TypeError(
                f"length_fn='{field}' requires payload to be a dict, got {type(record.payload)}"
            )
        value = record.payload.get(field)
        if value is None:
            raise ValueError(f"Field '{field}' not found in payload")

        # Direct integer length
        if isinstance(value, int):
            return value

        # Tensor-like objects (numpy arrays, torch tensors, JAX arrays, etc.)
        # Check for shape attribute and try to access first dimension
        # Use getattr to safely access shape without type checker errors
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                if len(shape) > 0:
                    return int(shape[0])
            except (TypeError, AttributeError, IndexError):
                # shape exists but isn't indexable or is empty, fall through to len()
                pass

        # Sequence-like objects (list, tuple, or anything with len())
        # Only call len() on types that support it (Sized protocol)
        if isinstance(value, Sized):
            return len(value)

        raise TypeError(
            f"Field '{field}' must be int, sequence-like (list/tuple), or tensor-like "
            + f"(with .shape attribute) for length extraction, got {type(value)}"
        )

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=False,
            batch_shape_sensitive=False,
            requires_serial_state=True,  # Requires parallelism=1 in deterministic mode
            preserves_cursor_order=False,  # Packing reorders sequences into bins
        )

    def buffering(self) -> Optional[Buffering]:
        return None

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

        # Handle dict payloads
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

        # Handle direct tensor payloads
        if all(isinstance(p, torch.Tensor) for p in payloads):
            return torch.cat(payloads, dim=0)

        # Fail if not all tensors
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

        # Handle dict payloads
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

        # Handle direct array payloads
        if all(isinstance(p, np.ndarray) for p in payloads):
            return np.concatenate(payloads, axis=0)

        # Fail if not all arrays
        non_array_types = {
            type(p).__name__ for p in payloads if not isinstance(p, np.ndarray)
        }
        raise TypeError(
            f"pack_payloads='numpy_array' requires all payloads to be numpy.ndarray, "
            + f"but found non-array types: {non_array_types}"
        )

    def _flush_all_bins(self, bins: list[Bin], lane_id: int) -> list[SampleRecord]:
        """Flush all bins and return packed records.

        Args:
            bins: List of bins to flush (will be emptied).
            lane_id: Lane ID for creating packed records.

        Returns:
            List of packed SampleRecords from all flushed bins.
        """
        outputs: list[SampleRecord] = []
        # Use pop() to efficiently remove all items (O(1) per pop vs O(n) for remove)
        while bins:
            outputs.append(self._create_packed_record(bins.pop(), lane_id))
        return outputs

    def _enforce_max_bins(
        self,
        bins: list[Bin],
        lane_id: int,
    ) -> list[SampleRecord]:
        """Enforce num_bins limit by flushing bins if necessary.

        Uses the configured flush_strategy to determine which bins to flush.

        Returns:
            List of packed SampleRecords emitted from flushed bins.
        """
        outputs: list[SampleRecord] = []
        while len(bins) >= self.num_bins and bins:
            if self.flush_strategy == "fifo":
                # Flush oldest bin (first in list)
                bin_to_flush = bins.pop(0)
            elif self.flush_strategy == "fullest":
                # Flush bin with smallest remaining capacity (most full)
                fullest_idx = min(range(len(bins)), key=lambda i: bins[i].remaining)
                bin_to_flush = bins.pop(fullest_idx)
            else:
                raise ValueError(f"Unknown flush_strategy: {self.flush_strategy}")
            outputs.append(self._create_packed_record(bin_to_flush, lane_id))
        return outputs

    def _create_bin_with_sample(
        self,
        bins: list[Bin],
        seq: SampleRecord,
        seq_len: int,
        lane_id: int,
    ) -> list[SampleRecord]:
        """Create a new bin, add a sample, and emit if full or can't fit min sequence.

        Args:
            bins: List of bins to potentially append the new bin to.
            seq: Sample record to add to the new bin.
            seq_len: Length of the sample sequence.
            lane_id: Lane ID for creating packed records.

        Returns:
            List of packed SampleRecords emitted (empty or one record if bin is full/unusable).
        """
        # Enforce num_bins limit before creating new bin
        outputs = self._enforce_max_bins(bins, lane_id)

        new_bin = Bin(samples=[seq], remaining=self.max_length - seq_len)
        # Emit if new bin can't fit minimum sequence length
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
        """Add a sample to an existing bin and emit if full or can't fit min sequence.

        Args:
            bin_data: Bin to add the sample to.
            bins: List of bins containing this bin.
            seq: Sample record to add.
            seq_len: Length of the sample sequence.
            lane_id: Lane ID for creating packed records.

        Returns:
            List of packed SampleRecords emitted (empty or one record if bin is full/unusable).
        """
        bin_data.samples.append(seq)
        bin_data.remaining -= seq_len
        outputs: list[SampleRecord] = []
        # Emit bin if it can't fit minimum sequence length
        if bin_data.remaining < self.min_sequence_length:
            bins.remove(bin_data)
            outputs.append(self._create_packed_record(bin_data, lane_id))
        return outputs

    def _first_fit_pack(
        self, lane_id: int, seq: SampleRecord, seq_len: int
    ) -> list[SampleRecord]:
        """Try to pack sequence using first-fit algorithm.

        Multiple bins can coexist - we maintain partial bins until they become full.
        When a sequence doesn't fit in any existing bin, we create a new bin without
        flushing existing ones. This allows better packing efficiency as future sequences
        may fit into the partially-filled bins.
        Only bins that can fit at least min_sequence_length are checked.

        Returns:
            List of packed SampleRecords emitted (may be empty or contain one record).
        """
        bins = self._bins[lane_id]
        outputs: list[SampleRecord] = []

        # Try to find first bin that fits
        for bin_data in bins:
            if bin_data.remaining >= seq_len:
                outputs.extend(
                    self._add_sample_to_bin(bin_data, bins, seq, seq_len, lane_id)
                )
                return outputs

        # No bin fits - create a new bin without flushing existing bins.
        # This maintains multiple bins for better packing efficiency.
        outputs.extend(self._create_bin_with_sample(bins, seq, seq_len, lane_id))

        return outputs

    def _best_fit_pack(
        self, lane_id: int, seq: SampleRecord, seq_len: int
    ) -> list[SampleRecord]:
        """Try to pack sequence using best-fit algorithm.

        Multiple bins can coexist - we maintain partial bins until they become full.
        When a sequence doesn't fit in any existing bin, we create a new bin without
        flushing existing ones. This allows better packing efficiency as future sequences
        may fit into the partially-filled bins.
        Only bins that can fit at least min_sequence_length are checked.

        Returns:
            List of packed SampleRecords emitted (may be empty or contain one record).
        """
        bins = self._bins[lane_id]
        outputs: list[SampleRecord] = []

        # Find bin with smallest remaining capacity that fits
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

        # No bin fits - create a new bin without flushing existing bins.
        # This maintains multiple bins for better packing efficiency.
        outputs.extend(self._create_bin_with_sample(bins, seq, seq_len, lane_id))

        return outputs

    def _create_packed_record(self, bin_data: Bin, lane_id: int) -> SampleRecord:
        """Create a packed SampleRecord from a bin."""
        samples: list[SampleRecord] = bin_data.samples
        if not samples:
            raise ValueError("Cannot create packed record from empty bin")

        total_length = self.max_length - bin_data.remaining
        num_sequences = len(samples)
        packing_efficiency = total_length / self.max_length

        # Pack payloads using the configured strategy
        # This combines multiple payloads into a single payload structure
        raw_payloads = [s.payload for s in samples]
        packed_payload_value = self._pack_payloads_fn(raw_payloads)

        # Create packed payload (metadata goes in meta tags, not payload)
        packed_payload: dict[str, Any] = {
            "packed_samples": packed_payload_value,
        }

        # Collect all contributors from all samples in the bin
        # The engine handles duplicates correctly - if any contributor has is_last_child=True,
        # it will close the base offset when iterating over all contributors
        contributors: list[ContributorRef] = []
        for sample in samples:
            contributors.extend(sample.meta.contribution_refs())

        # Use first sample's cursor as primary, with a unique lineage step for this packed bin
        base_meta = samples[0].meta
        primary_cursor = base_meta.cursor.child(0)

        # Build packed metadata with proper contributors and packing metadata in tags
        packed_meta = pack_meta(
            primary_cursor=primary_cursor,
            contributors=contributors,
            lane_id=lane_id,
            tags={
                "_packing_metadata": {
                    "num_sequences": num_sequences,
                    "total_length": total_length,
                    "packing_efficiency": packing_efficiency,
                }
            },
        )

        return SampleRecord(meta=packed_meta, payload=packed_payload)

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        """Process a single sample record."""
        lane_id = elem.meta.lane_id

        # Extract length; surface validation errors rather than silently dropping.
        seq_len = self.length_fn(elem)

        # Check if sequence is oversized BEFORE packing
        # This ensures oversized sequences don't trigger bin emissions
        if seq_len > self.max_length:
            if self.drop_oversized:
                return []  # Drop oversized sequence without affecting bins
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_length {self.max_length}"
            )

        # Pack using selected algorithm
        if self.algorithm == "first_fit":
            outputs = self._first_fit_pack(lane_id, elem, seq_len)
        elif self.algorithm == "best_fit":
            outputs = self._best_fit_pack(lane_id, elem, seq_len)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return outputs

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        """Process multiple sample records with lane-local packing.

        Applies the configured shuffle strategy (random shuffle, length sort, or none) to order
        the sequences, then processes each one using process_one.
        """
        if not elems:
            return []

        # Apply shuffle strategy to order sequences
        if self.shuffle_strategy == "random":
            # Use deterministic seed based on batch contents (like ShuffleBuffer)
            rng = random.Random(batch_seed(self.shuffle_seed, elems))
            rng.shuffle(elems)
        elif self.shuffle_strategy == "length":
            # Sort by length descending, stable by cursor key for deterministic ordering
            elems = sorted(
                elems,
                key=lambda rec: (-self.length_fn(rec), rec.meta.cursor.as_key()),
            )
        elif self.shuffle_strategy is None:
            # No reordering - preserve input order
            pass
        else:
            raise ValueError(f"Unknown shuffle_strategy: {self.shuffle_strategy}")

        # Process each record using process_one
        outputs: list[SampleRecord] = []
        for elem in elems:
            outputs.extend(self.process_one(elem))

        return outputs

    def finalize(self) -> list[SampleRecord]:
        """Emit any remaining partially-filled bins."""
        outputs: list[SampleRecord] = []
        # No need to copy items() - we're not modifying the dict during iteration
        for lane_id, bins in self._bins.items():
            # Emit all bins
            for bin_data in bins:
                if bin_data.samples:  # Only emit non-empty bins
                    outputs.append(self._create_packed_record(bin_data, lane_id))
        self._bins.clear()
        return outputs
