# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Public IO API surfacing core protocols and implementations."""

from zephon.io.dataset import Dataset
from zephon.io.memory import (
    InMemoryDatasetStore,
    InMemoryMultiDatasetStore,
    InMemoryShard,
)
from zephon.io.options import CacheOptions, StoreOptions
from zephon.io.protocols import (
    DatasetShardView,
    MultiDatasetShardStore,
    RandomAccessShard,
)
from zephon.io.resolvers import CacheManager, DirectResolver, ShardResolver
from zephon.io.storage import LocalFSBackend, StorageBackend
from zephon.io.stores import (
    DatasetStoreRegistry,
    ResilientShard,
    build_multi_dataset_store,
    build_resolver,
)
from zephon.io.types import LocalShardFile, LocalShardRef, ShardFile, ShardLocator

__all__ = [
    "CacheManager",
    "CacheOptions",
    "Dataset",
    "DatasetShardView",
    "DatasetStoreRegistry",
    "DirectResolver",
    "InMemoryDatasetStore",
    "InMemoryMultiDatasetStore",
    "InMemoryShard",
    "LocalFSBackend",
    "LocalShardFile",
    "LocalShardRef",
    "MultiDatasetShardStore",
    "RandomAccessShard",
    "ResilientShard",
    "ShardFile",
    "ShardLocator",
    "ShardResolver",
    "StorageBackend",
    "StoreOptions",
    "build_multi_dataset_store",
    "build_resolver",
]
