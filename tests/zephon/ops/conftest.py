# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers for ops integration tests."""

from zephon.io import InMemoryShard
from zephon.io.dataset import Dataset


def mk_dataset(name: str, shards: dict[int, int]) -> Dataset:
    """Create a dataset with specified shards and counts."""
    data: dict[int, InMemoryShard] = {}
    for sid, count in shards.items():
        rows = [{"text": f"{name}:{sid}:{i}", "length": 3} for i in range(count)]
        data[int(sid)] = InMemoryShard(rows)
    return Dataset.from_dict(name, data)
