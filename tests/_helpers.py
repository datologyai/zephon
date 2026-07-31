# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Test helpers that import only Zephon's public API."""

from zephon.io import InMemoryShard
from zephon.io.dataset import Dataset


def mk_dataset(name: str, shards: dict[int, int]) -> Dataset:
    """Create a Dataset backed by in-memory shards.

    Args:
        name: Dataset name (used as prefix in generated text payloads).
        shards: Mapping of shard ID to row count.

    Returns:
        A ``Dataset`` with ``InMemoryShard`` entries whose rows contain
        ``{"text": "<name>:<shard_id>:<row_index>"}``.
    """
    data: dict[int, InMemoryShard] = {}
    for sid, count in shards.items():
        rows = [{"text": f"{name}:{sid}:{i}"} for i in range(count)]
        data[int(sid)] = InMemoryShard(rows)
    return Dataset.from_dict(name, data)


def counts_dict(dataset: Dataset) -> dict[int, int]:
    """``shard_id -> sample_count`` dict for test assertions."""
    return dict(zip(dataset.ids().tolist(), dataset.counts().tolist(), strict=True))
