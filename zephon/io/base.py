# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shard interfaces and in-memory helpers for feeding pipelines."""

from typing import Protocol


class RandomAccessShard(Protocol):
    """A shard that supports len/index style random access."""

    def __getitem__(self, index: int) -> dict[str, object]: ...

    def __len__(self) -> int: ...

    def close(self) -> None: ...


class IndexedShardStore(Protocol):
    """Factory that opens shards by identifier."""

    def open(self, shard_id: int) -> RandomAccessShard: ...


class InMemoryShard:
    """List-backed shard useful for tests and quickstarts."""

    def __init__(self, rows: list[dict[str, object]]):
        self._rows = rows

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._rows[index]

    def __len__(self) -> int:
        return len(self._rows)

    def close(self) -> None:
        return None


class InMemoryShardStore:
    """Dictionary-backed shard store exposing the protocol surface."""

    def __init__(self, shards: dict[int, InMemoryShard]):
        self._shards = shards

    def open(self, shard_id: int) -> RandomAccessShard:
        return self._shards[shard_id]
