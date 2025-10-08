"""Protocols shared by shard resolvers."""

from typing import Protocol

from zephon.io.types import LocalShardRef, ShardLocator


class ShardResolver(Protocol):
    """Resolve a shard locator into a local reference ready for reading."""

    def resolve(
        self, locator: ShardLocator, *, blocking: bool = True
    ) -> LocalShardRef: ...

    def touch(self, locator: ShardLocator) -> None: ...


__all__ = ["ShardResolver"]
