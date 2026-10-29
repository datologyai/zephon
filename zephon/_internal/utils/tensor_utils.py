# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tensor/array utilities for multi-framework support (PyTorch, NumPy, lists).

Conversion helpers select the output framework from the requested dtype.
Packing helpers preserve the input container, or fill an allocated output buffer.
Optional dependencies are lazily imported.
"""

from collections.abc import Sequence
from typing import Any


def resolve_dtype(dtype: Any) -> tuple[Any, str | None]:
    """Resolve dtype to (actual_dtype, framework).

    Args:
        dtype: One of:
            - "auto": Detect framework (prefers torch.long > np.int64 > lists)
            - None: Return lists (no tensor conversion)
            - torch.long, torch.int64, etc.: Use PyTorch
            - np.int64, np.int32, etc.: Use NumPy

    Returns:
        Tuple of (resolved_dtype, framework) where framework is one of:
        "torch", "numpy", or None (for lists).
    """
    if dtype is None:
        return None, None

    if dtype == "auto":
        # Try torch first, then numpy, then fall back to lists
        try:
            import torch

            return torch.long, "torch"
        except ImportError:
            pass
        try:
            import numpy as np

            return np.int64, "numpy"
        except ImportError:
            pass
        return None, None

    # Infer framework from dtype using multiple detection methods
    # Check the dtype's own __module__ (works for numpy types like np.int64)
    dtype_own_module = getattr(dtype, "__module__", "")
    if "torch" in dtype_own_module:
        return dtype, "torch"
    if "numpy" in dtype_own_module:
        return dtype, "numpy"

    # Check type(dtype).__module__ (works for torch.dtype instances)
    dtype_type_module = type(dtype).__module__
    if "torch" in dtype_type_module:
        return dtype, "torch"
    if "numpy" in dtype_type_module:
        return dtype, "numpy"

    # Fallback: check string representation
    dtype_str = str(dtype)
    if "torch" in dtype_str:
        return dtype, "torch"
    if "numpy" in dtype_str:
        return dtype, "numpy"

    return dtype, None


def stack_sequences(
    sequences: list[Any],
    dtype: Any,
    framework: str | None,
    *,
    length: int | None = None,
    pad_value: int = 0,
    tails: Sequence[Any] | None = None,
) -> Any:
    """Stack sequences, optionally padding them to a common length first.

    ``length`` pads unequal merged documents; without it, stack them directly.

    Args:
        sequences: List of sequences (lists, numpy arrays, or tensors).
        dtype: Target dtype (or None for lists).
        framework: "torch", "numpy", or None.
        length: Optional padded scalar sequence width. Allocate the final batch
            once and copy each sequence into its prefix; never truncate.
        pad_value: Fill value for padding, validated against the output dtype.
        tails: Optional padding values per sequence instead of pad_value. Positions
            use a new zero-based range so padding forms a separate segment.

    Returns:
        Stacked tensor or list of lists.
    """
    if length is not None:
        result: Any
        if any(len(seq) > length for seq in sequences):
            raise ValueError(
                f"sequence length exceeds pad_to_length={length}; truncation is not supported"
            )
        if framework == "torch":
            import torch

            device = (
                sequences[0].device
                if sequences and isinstance(sequences[0], torch.Tensor)
                else None
            )
            result = torch.empty((len(sequences), length), dtype=dtype, device=device)
        elif framework == "numpy":
            import numpy as np

            result = np.empty((len(sequences), length), dtype=dtype)
        else:
            result = [None] * len(sequences)
        for i, seq in enumerate(sequences):
            tail = tails[i] if tails is not None else None
            padded = pad_field(
                seq,
                length - len(seq),
                pad_value,
                out=result[i] if framework is not None else None,
                tail=tail,
                scalar=True,
            )
            if framework is None:
                result[i] = list(padded)
        return result
    if framework == "torch":
        import torch

        # If inputs are already tensors, use stack
        if sequences and isinstance(sequences[0], torch.Tensor):
            return torch.stack(sequences).to(dtype=dtype)
        # If inputs are numpy arrays, stack via numpy first to avoid slow
        # list-of-ndarrays path that triggers PyTorch warnings
        if sequences and "numpy" in type(sequences[0]).__module__:
            import numpy as np

            return torch.from_numpy(np.array(sequences)).to(dtype=dtype)
        return torch.tensor(sequences, dtype=dtype)
    elif framework == "numpy":
        import numpy as np

        # Older NumPy cannot coerce a list of array-like objects (e.g. tensors).
        return np.array(
            [np.asarray(seq, dtype=dtype) for seq in sequences], dtype=dtype
        )
    else:
        # Return as list of lists
        return [list(seq) for seq in sequences]


def slice_last_dim(tensor: Any, slc: slice, framework: str | None) -> Any:
    """Slice tensor along the last dimension.

    Args:
        tensor: Tensor or list to slice.
        slc: Slice object for the last dimension.
        framework: "torch", "numpy", or None.

    Returns:
        Sliced tensor or list.
    """
    if framework in ("torch", "numpy"):
        # Both torch and numpy support [..., slc] indexing
        return tensor[..., slc]
    else:
        # List of lists: slice each inner list
        return [row[slc] for row in tensor]


def flatten_sequences(sequences: Any, framework: str | None) -> Any:
    """Flatten batched scalar sequences in row order, preserving their dtype."""
    if framework in ("torch", "numpy"):
        return sequences.reshape(-1)
    return [value for row in sequences for value in row]


def count_valid_tokens(labels: Any, ignore_index: int, framework: str | None) -> int:
    """Count non-ignored labels in a batch of sequences."""
    if framework in ("torch", "numpy"):
        return int((labels != ignore_index).sum())
    return int(sum(value != ignore_index for row in labels for value in row))


def labels_to_loss_mask(labels: Any, ignore_index: int, framework: str | None) -> Any:
    """Return a float32 mask of non-ignored labels (Python floats for lists)."""
    if framework == "torch":
        return (labels != ignore_index).float()
    if framework == "numpy":
        import numpy as np

        return (labels != ignore_index).astype(np.float32)
    return [[float(value != ignore_index) for value in row] for row in labels]


def padding_lengths_to_mask(
    tokens: Any,
    pad_lengths: list[int],
    framework: str | None,
    *,
    shifted: bool,
) -> Any:
    """Return a boolean input-aligned padding mask without inspecting token IDs.

    Lengths describe unshifted rows. Next-token inputs omit the last token,
    reducing their padding by one; their real-token cutoff stays unchanged.
    Array backends broadcast O(B + S) indices only when padding is present.
    Lists may be ragged.
    """
    if framework in ("torch", "numpy"):
        if tokens.ndim != 2:
            raise ValueError("return_padding_mask requires 2-D tokens")
        raw_width = tokens.shape[1]
        lengths = [raw_width] * tokens.shape[0]
    else:
        lengths = [len(row) for row in tokens]
    if len(pad_lengths) != len(lengths) or any(
        not 0 <= pad <= length for pad, length in zip(pad_lengths, lengths)
    ):
        raise ValueError("padding_length must lie within its token row's length")

    shift = int(shifted)
    if framework == "torch":
        import torch

        width = max(0, tokens.shape[1] - shift)
        if not any(n > shift for n in pad_lengths):
            return tokens.new_zeros((tokens.shape[0], width), dtype=torch.bool)
        cutoff = torch.tensor(
            [tokens.shape[1] - n for n in pad_lengths],
            dtype=torch.int64,
            device=tokens.device,
        )
        return torch.arange(width, device=tokens.device)[None, :] >= cutoff[:, None]
    if framework == "numpy":
        import numpy as np

        width = max(0, tokens.shape[1] - shift)
        if not any(n > shift for n in pad_lengths):
            return np.zeros((tokens.shape[0], width), dtype=bool)
        cutoff = tokens.shape[1] - np.asarray(pad_lengths)
        return np.arange(width)[None, :] >= cutoff[:, None]
    return [
        [False] * min(length - pad, max(0, length - shift))
        + [True] * max(0, pad - shift)
        for length, pad in zip(lengths, pad_lengths)
    ]


def mask_eos_labels(
    labels: Any,
    input_ids: Any,
    eos_token_id: int,
    replacement: int,
    framework: str | None,
) -> Any:
    """Mask labels at EOS input positions, rather than matching the target ID."""
    if framework == "torch":
        return labels.masked_fill(input_ids == eos_token_id, replacement)
    if framework == "numpy":
        import numpy as np

        return np.where(input_ids == eos_token_id, replacement, labels)
    return [
        [
            replacement if token == eos_token_id else label
            for token, label in zip(row, labs)
        ]
        for row, labs in zip(input_ids, labels)
    ]


def sequence_positions(input_ids: Any, framework: str | None) -> Any:
    """Number scalar tokens from zero in each input row, matching dtype/device."""
    if framework in ("torch", "numpy"):
        if input_ids.ndim != 2:
            raise ValueError("position_mode='sequence' requires 2-D input_ids")
        if framework == "torch":
            import torch

            return torch.arange(
                input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            ).repeat(input_ids.shape[0], 1)
        import numpy as np

        return np.broadcast_to(
            np.arange(input_ids.shape[1], dtype=input_ids.dtype), input_ids.shape
        ).copy()
    # List stacking does not validate rank, so nested payloads can still arrive.
    # Reject them instead of assigning one position per container as if scalar.
    if any(
        isinstance(value, (list, tuple, dict, str, bytes))
        or getattr(value, "ndim", 0) != 0
        for row in input_ids
        for value in row
    ):
        raise ValueError("position_mode='sequence' requires 2-D input_ids")
    return [list(range(len(row))) for row in input_ids]


def positions_to_document_ids(
    positions: Any, padding_mask: Any, framework: str | None, *, flatten: bool
) -> Any:
    """Convert the packer's scalar position resets to per-token document IDs.

    Positions [0, 1, 2, 0, 1] give IDs [1, 1, 1, 2, 2].
    Inputs/output have shape [batch, sequence]; lists may be ragged. True padding
    entries receive zero, independently of token values or supervision. Positive
    IDs start at one per batch document, or run across the batch for flatten=True.
    Requires positions that reset at subdocument starts; continuous positions
    do not contain the boundaries needed for this conversion.
    """
    if framework in ("torch", "numpy"):
        if positions.ndim != 2 or positions.shape != padding_mask.shape:
            raise ValueError(
                "return_document_ids requires scalar positions matching input_ids"
            )
        if positions.shape[1] and bool((positions[:, 0] != 0).any()):
            raise ValueError("positions must start at zero in every nonempty document")
        starts = positions == 0
        if framework == "torch":
            import torch

            ids = (
                starts.reshape(-1).cumsum(0, dtype=torch.int64).reshape(positions.shape)
                if flatten
                else starts.cumsum(-1, dtype=torch.int64)
            )
            return ids.masked_fill(padding_mask, 0)

        import numpy as np

        ids = np.cumsum(starts, axis=None if flatten else -1, dtype=np.int64)
        ids = ids.reshape(positions.shape)
        return np.where(padding_mask, 0, ids)

    if len(positions) != len(padding_mask):
        raise ValueError(
            "return_document_ids requires scalar positions matching input_ids"
        )
    result = []
    current = 0
    for positions_document, padding_document in zip(positions, padding_mask):
        if any(
            isinstance(value, (list, tuple, dict, str, bytes))
            or getattr(value, "ndim", 0) != 0
            for value in positions_document
        ) or len(positions_document) != len(padding_document):
            raise ValueError(
                "return_document_ids requires scalar positions matching input_ids"
            )
        if len(positions_document) and positions_document[0] != 0:
            raise ValueError("positions must start at zero in every nonempty document")
        if not flatten:
            current = 0
        document = []
        for position, padding in zip(positions_document, padding_document):
            current += int(position == 0)
            document.append(0 if padding else current)
        result.append(document)
    return result


def positions_to_cu_seqlens(
    positions: Any, framework: str | None, *, flatten: bool
) -> tuple[Any, Any]:
    """Derive int32 cumulative segment lengths and maxima from position resets.

    Batched boundaries are padded with each row's length to the largest boundary
    count in the batch; maxima have shape ``[B]``. Flattened boundaries are compact
    offsets into the whole batch, with a Python int maximum. Lists use Python ints.
    """
    if framework in ("torch", "numpy"):
        if positions.ndim != 2:
            raise ValueError("return_cu_seqlens requires 2-D positions")
        batch_size, width = positions.shape
        if width and bool((positions[:, 0] != 0).any()):
            raise ValueError("positions must start at zero in every nonempty row")
        extent = positions.size if framework == "numpy" else positions.numel()
        if (extent if flatten else width) > 2**31 - 1:
            raise ValueError("cumulative sequence lengths exceed the int32 range")

        if framework == "torch":
            import torch

            if flatten:
                starts = (positions.reshape(-1) == 0).nonzero(as_tuple=True)[0]
                cu = torch.cat(
                    [
                        starts.to(torch.int32),
                        torch.tensor(
                            [extent], dtype=torch.int32, device=positions.device
                        ),
                    ]
                )
                maximum = int(cu.diff().max().item()) if extent else 0
                return cu, maximum

            starts = positions == 0
            rows, columns = starts.nonzero(as_tuple=True)
            counts = starts.sum(dim=1)
            boundary_count = int(counts.max().item()) + 1 if batch_size else 1
            cu = torch.full(
                (batch_size, boundary_count),
                width,
                dtype=torch.int32,
                device=positions.device,
            )
            offsets = counts.cumsum(dim=0) - counts
            # nonzero groups starts by row; subtract preceding rows' counts
            # to get each start's slot in its padded boundary row.
            slots = torch.arange(rows.numel(), device=positions.device) - offsets[rows]
            cu[rows, slots] = columns.to(torch.int32)
            maxima = (
                cu.diff(dim=1).amax(dim=1)
                if boundary_count > 1
                else cu.new_zeros(batch_size)
            )
            return cu, maxima

        import numpy as np

        if flatten:
            starts = np.flatnonzero(positions.reshape(-1) == 0)
            cu = np.append(starts, extent).astype(np.int32)
            return cu, int(np.diff(cu).max(initial=0))

        starts = positions == 0
        rows, columns = np.nonzero(starts)
        counts = starts.sum(axis=1)
        boundary_count = int(counts.max(initial=0)) + 1
        cu = np.full((batch_size, boundary_count), width, dtype=np.int32)
        offsets = counts.cumsum() - counts
        cu[rows, np.arange(len(rows)) - offsets[rows]] = columns
        return cu, np.diff(cu, axis=1).max(axis=1, initial=0)

    boundaries, maxima = [], []
    flat_starts: list[int] = []
    offset = 0
    for row in positions:
        if any(
            isinstance(value, (list, tuple, dict, str, bytes))
            or getattr(value, "ndim", 0) != 0
            for value in row
        ):
            raise ValueError("return_cu_seqlens requires 2-D positions")
        if row and row[0] != 0:
            raise ValueError("positions must start at zero in every nonempty row")
        starts = [i for i, value in enumerate(row) if value == 0]
        cu = starts + [len(row)]
        maxima.append(max((b - a for a, b in zip(cu, cu[1:])), default=0))
        if flatten:
            flat_starts.extend(offset + start for start in starts)
            offset += len(row)
        else:
            boundaries.append(cu)
    if flatten:
        return flat_starts + [offset], max(maxima, default=0)
    boundary_count = max(map(len, boundaries), default=0)
    boundaries = [cu + [cu[-1]] * (boundary_count - len(cu)) for cu in boundaries]
    return boundaries, maxima


def mask_padding_labels(
    labels: Any, pad_lengths: list[int], replacement: int, framework: str | None
) -> Any:
    """Set the last ``pad_lengths[i]`` entries of row ``i`` to ``replacement``.

    Masks right-padding out of next-token labels by position rather than by id, so
    a pad token that collides with a real (or eos) id is still handled correctly.
    Returns a new container.
    """
    if framework in ("torch", "numpy"):
        n_rows, width = labels.shape[0], labels.shape[-1]
        assert all(0 <= n <= width for n in pad_lengths), (
            f"pad_lengths must lie in [0, {width}]: {pad_lengths}"
        )
    else:
        n_rows = len(labels)
    assert len(pad_lengths) == n_rows, (
        f"pad_lengths has {len(pad_lengths)} entries for {n_rows} rows"
    )

    if framework == "torch":
        import torch

        cols = torch.arange(width, device=labels.device)
        cutoff = width - torch.as_tensor(pad_lengths, device=labels.device)
        return labels.masked_fill(cols.unsqueeze(0) >= cutoff.unsqueeze(1), replacement)
    elif framework == "numpy":
        import numpy as np

        out = np.array(labels, copy=True)
        for i, n in enumerate(pad_lengths):
            if n:
                out[i, width - n :] = replacement
        return out
    else:
        masked = []
        for row, n in zip(labels, pad_lengths):
            assert 0 <= n <= len(row), f"pad_lengths must lie in [0, {len(row)}]: {n}"
            row = list(row)
            if n:
                row[len(row) - n :] = [replacement] * n
            masked.append(row)
        return masked


def mask_unsupervised_labels(
    labels: Any, masks: Any, replacement: int, framework: str | None
) -> Any:
    """Apply a label-aligned 0/1 supervision mask."""
    if framework == "torch":
        assert labels.shape == masks.shape, (
            f"labels and masks must have the same shape: {labels.shape} != {masks.shape}"
        )
        return labels.masked_fill(masks == 0, replacement)
    elif framework == "numpy":
        import numpy as np

        assert labels.shape == masks.shape, (
            f"labels and masks must have the same shape: {labels.shape} != {masks.shape}"
        )
        return np.where(np.asarray(masks) != 0, labels, replacement)
    else:
        assert len(labels) == len(masks), (
            f"labels and masks must have the same row count: {len(labels)} != {len(masks)}"
        )
        masked = []
        for row, mask_row in zip(labels, masks):
            assert len(row) == len(mask_row), (
                "labels and masks rows must have the same length: "
                + f"{len(row)} != {len(mask_row)}"
            )
            masked.append([lab if m else replacement for lab, m in zip(row, mask_row)])
        return masked


def positions_from_segment_lengths(segment_lengths: Sequence[int]) -> Any:
    """Build the packer's zero-based positions from its ordered segment lengths.

    Lengths [3, 2] give NumPy int32 positions [0, 1, 2, 0, 1]. Flat serialization
    uses this for subdocuments; conversion uses it for a separate padding segment.
    Empty segments contribute nothing. Inputs are not modified.
    """
    import numpy as np

    parts = [np.arange(int(n), dtype=np.int32) for n in segment_lengths if n > 0]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)


def _is_torch_tensor(value: Any) -> bool:
    """Check actual tensor identity without requiring optional Torch to be installed."""
    if value is None:
        return False
    try:
        import torch
    except ImportError:
        return False
    return isinstance(value, torch.Tensor)


def pad_field(
    seq: Any,
    pad_len: int,
    pad_value: int,
    *,
    out: Any = None,
    tail: Any = None,
    scalar: bool = False,
) -> Any:
    """Right-pad a field for flat packing or fixed-length training conversion.

    Packing preserves the input container/dtype. ``stack_sequences`` supplies
    an allocated output row to avoid intermediate padded arrays.

    Args:
        seq: List, tuple, NumPy array or Torch tensor with sequence on axis 0.
        pad_len: Number of positions to append.
        pad_value: Constant padding value, checked against integer output ranges.
        out: Optional owned destination [len(seq) + pad_len, ...]; determines
            the output dtype/backend and is filled in place.
        tail: Explicit padding values, e.g. zero-based padding-segment positions.
        scalar: Require one scalar per token; other layouts need field collators.

    Returns:
        The output buffer or padded input container; separate inputs are unchanged.
        Lists and tuples preserve scalar types from seq and tail, including
        NumPy scalars; only the container is converted when appending tail.
    """
    import numpy as np

    if out is None and pad_len <= 0 and not scalar:
        return seq
    if out is None and isinstance(seq, (list, tuple)):
        if (
            scalar
            and seq
            and (
                isinstance(seq[0], (list, tuple, dict, str, bytes))
                or getattr(seq[0], "ndim", 0) != 0
            )
        ):
            raise ValueError(
                "expected a scalar token sequence; use field_collators for other layouts"
            )
        padding = [pad_value] * pad_len if tail is None else list(tail)
        if isinstance(seq, tuple):
            return seq + tuple(padding)
        return seq + padding
    torch_input = _is_torch_tensor(seq)
    if not torch_input and not isinstance(seq, (list, tuple, np.ndarray)):
        raise TypeError(
            f"unsupported field type {type(seq).__name__} to pad; "
            + "expected list, tuple, numpy.ndarray, or torch.Tensor"
        )
    torch_output = _is_torch_tensor(out)
    values: Any = seq if torch_input else np.asarray(seq)
    if scalar and (
        values.ndim != 1 or (not torch_input and values.dtype.kind in "OUS")
    ):
        raise ValueError(
            "expected a scalar token sequence; use field_collators for other layouts"
        )
    if torch_input or torch_output:
        import torch

        if out is None:
            out = values.new_empty((len(seq) + pad_len, *values.shape[1:]))
        if torch_output or _is_torch_tensor(out):
            dtype = out.dtype
            if (
                not dtype.is_floating_point
                and not dtype.is_complex
                and dtype != torch.bool
            ):
                info = torch.iinfo(dtype)
                if pad_len and not info.min <= pad_value <= info.max:
                    raise ValueError(
                        f"pad value {pad_value} is not representable in dtype {dtype}"
                    )
            out[: len(seq)] = torch.as_tensor(seq, dtype=dtype, device=out.device)
            out[len(seq) :] = (
                pad_value
                if tail is None
                else torch.as_tensor(tail, dtype=dtype, device=out.device)
            )
            return out
        values = values.numpy()
    if out is None:
        out = np.empty((len(seq) + pad_len, *values.shape[1:]), dtype=values.dtype)
    if pad_len and out.dtype.kind in "iu":
        info = np.iinfo(out.dtype)
        if not info.min <= pad_value <= info.max:
            raise ValueError(
                f"pad value {pad_value} is not representable in dtype {out.dtype}"
            )
    out[: len(seq)] = values
    out[len(seq) :] = pad_value if tail is None else tail
    return out


def mask_segment_start_labels(
    labels: Any, positions: Any, replacement: int, framework: str | None
) -> Any:
    """Exclude predictions of the first token of a later packed subdocument.

    This avoids training the last token of one unrelated subdocument to predict
    the next one's first token. For positions [0, 1, 0, 1], labels [a2, b1, b2]
    become [a2, replacement, b2]. ``labels`` has shape [batch, sequence - 1];
    original scalar ``positions`` has shape [batch, sequence]. Lists may be ragged.
    Returns new labels, retaining existing exclusions. The first input token is
    already outside the target slice. Attention, positions and inputs are unchanged.
    """
    if framework in ("torch", "numpy"):
        if (
            positions.ndim != 2
            or positions.shape[:-1] != labels.shape[:-1]
            or max(0, positions.shape[-1] - 1) != labels.shape[-1]
        ):
            raise ValueError(
                "mask_segment_starts requires scalar positions aligned with tokens"
            )
        if positions.shape[-1] and bool((positions[:, 0] != 0).any()):
            raise ValueError("positions must start at zero in every nonempty document")
        return mask_unsupervised_labels(
            labels, positions[:, 1:] != 0, replacement, framework
        )
    for row, targets in zip(positions, labels):
        if any(
            isinstance(item, (list, tuple, dict, str, bytes))
            or getattr(item, "ndim", 0) != 0
            for item in row
        ) or max(0, len(row) - 1) != len(targets):
            raise ValueError(
                "mask_segment_starts requires scalar positions aligned with tokens"
            )
        if row and row[0] != 0:
            raise ValueError("positions must start at zero in every nonempty document")
    masks = [[value != 0 for value in row[1:]] for row in positions]
    return mask_unsupervised_labels(labels, masks, replacement, framework)
