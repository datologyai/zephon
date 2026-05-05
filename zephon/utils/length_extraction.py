# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shared length extraction for pack_sequences and ensure_mixture.

This module provides unified logic for extracting sequence lengths from
SampleRecord payloads, with support for auto-detection of common token
field names and explicit field specification.
"""

from collections.abc import Sized
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zephon.core.constants import SampleRecord

# Standard token field candidates for auto-detection.
# Checked in order; first match wins.
TOKEN_FIELD_CANDIDATES = ["input_ids", "tokens", "token_ids", "ids"]


def detect_length_field(payload: dict[str, Any]) -> str | None:
    """Auto-detect length field from payload.

    Checks candidates in order: input_ids, tokens, token_ids, ids.

    Args:
        payload: The sample payload dict.

    Returns:
        Field name if found, None otherwise.
    """
    for field_name in TOKEN_FIELD_CANDIDATES:
        if field_name in payload and _has_length(payload[field_name]):
            return field_name
    return None


def extract_length(record: "SampleRecord", field: str | None = None) -> int:
    """Extract length from a record's payload.

    Args:
        record: The sample record.
        field: Explicit field name, or None to auto-detect from
               TOKEN_FIELD_CANDIDATES.

    Returns:
        The length as an integer.

    Raises:
        TypeError: If payload is not a dict.
        ValueError: If field not found or can't extract length.
    """
    payload = record.payload
    if not isinstance(payload, dict):
        field_info = f" for field '{field}'" if field else ""
        raise TypeError(
            f"Length extraction{field_info} requires dict payload, got {type(payload)}"
        )

    # Auto-detect if no explicit field
    if field is None:
        field = detect_length_field(payload)
        if field is None:
            raise ValueError(
                f"Cannot auto-detect length field. Payload keys: {list(payload.keys())}. "
                f"Expected one of: {TOKEN_FIELD_CANDIDATES}"
            )

    if field not in payload:
        raise ValueError(f"Field '{field}' not found in payload")

    value = payload[field]
    return _get_length(value, field)


def _has_length(value: Any) -> bool:
    """Check if value supports length extraction."""
    if isinstance(value, (int, Sized)):
        return True
    if hasattr(value, "item") and callable(value.item):
        return True
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        return True
    return False


def _get_length(value: Any, field: str) -> int:
    """Extract length from a value.

    Supports:
    - int: returned directly
    - Scalar tensor with .item(): converted to int
    - Tensor with .shape: uses shape[0]
    - Sized (list, tuple, etc.): uses len()

    Args:
        value: The value to extract length from.
        field: Field name for error messages.

    Returns:
        The length as an integer.

    Raises:
        TypeError: If value type is not supported.
    """
    if isinstance(value, int):
        return value

    # Scalar tensor (.item())
    if hasattr(value, "item") and callable(value.item):
        try:
            return int(value.item())
        # numpy raises ValueError for non-scalar arrays; fall through to shape.
        except (TypeError, AttributeError, ValueError):
            pass

    # Tensor shape
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            if len(shape) > 0:
                return int(shape[0])
        except (TypeError, AttributeError, IndexError):
            pass

    # Sequence length
    if isinstance(value, Sized):
        return len(value)

    raise TypeError(
        f"Field '{field}' must be int, sequence-like, or tensor-like, got {type(value)}"
    )
