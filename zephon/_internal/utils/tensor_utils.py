# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tensor/array utilities for multi-framework support (PyTorch, NumPy, lists).

This module provides utilities for working with tensors and arrays across
different frameworks. Framework detection is automatic based on dtype,
and optional dependencies are lazily imported.
"""

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


def stack_sequences(sequences: list[Any], dtype: Any, framework: str | None) -> Any:
    """Stack a list of sequences into a tensor or nested list.

    Args:
        sequences: List of sequences (lists, numpy arrays, or tensors).
        dtype: Target dtype (or None for lists).
        framework: "torch", "numpy", or None.

    Returns:
        Stacked tensor or list of lists.
    """
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
