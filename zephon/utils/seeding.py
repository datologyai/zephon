# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Utilities for deterministic seeding based on data contents."""

from collections.abc import Sequence

from zephon.core.constants import SampleRecord


def batch_seed(base_seed: int, elems: Sequence[SampleRecord]) -> int:
    """Generate a deterministic seed from a base seed and batch contents.

    Uses a stable hash function to combine the base seed with cursor keys from
    all elements in the batch. This ensures deterministic shuffling/ordering
    that is reproducible across runs while being sensitive to the actual data.

    Args:
        base_seed: Base seed value to start from.
        elems: List of SampleRecord instances.

    Returns:
        A deterministic seed value derived from the base seed and batch contents.
    """
    acc = base_seed
    for rec in elems:
        c = rec.meta.cursor.as_key()
        acc = (acc * 1315423911) ^ (c[0] * 2654435761) ^ (c[1] << 8)
        acc = acc ^ hash(c[2]) ^ hash(c[3])
    return acc & 0xFFFFFFFF
