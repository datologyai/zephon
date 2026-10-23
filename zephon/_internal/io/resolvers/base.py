"""Protocols shared by shard resolvers."""

from typing import Protocol

from zephon._internal.io.types import LocalShardRef, ShardLocator


class ShardResolver(Protocol):
    """Resolve a shard locator into a local reference ready for reading."""

    def resolve(
        self, locator: ShardLocator, *, blocking: bool = True
    ) -> LocalShardRef: ...

    def touch(self, locator: ShardLocator) -> None: ...

    def close(self) -> None:
        """Release owned resources; repeated calls must be safe."""
        ...


__all__ = ["ShardResolver"]
