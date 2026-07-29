# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Public IO API: datasets, the in-memory shard primitive, and store options.

Configure store behavior via ``Pipeline.options(io_options=StoreOptions(...))``.
"""

from zephon.io.dataset import Dataset
from zephon.io.memory import InMemoryShard
from zephon.io.options import CacheOptions, StoreOptions

__all__ = [
    "CacheOptions",
    "Dataset",
    "InMemoryShard",
    "StoreOptions",
]
