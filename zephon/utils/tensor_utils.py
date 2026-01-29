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

        return np.array(sequences, dtype=dtype)
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
