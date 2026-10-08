"""Protocols shared by shard resolvers."""

from typing import Protocol

from zephon._internal.io.types import ShardLocator, ShardRef


class ShardResolver(Protocol):
    """Resolve a shard locator into a reference ready for reading.

    The reference is local, except that the direct resolver gives a
    :class:`~zephon._internal.io.types.RemoteShardRef` for a shard under a
    remote root.
    """

    def resolve(self, locator: ShardLocator, *, blocking: bool = True) -> ShardRef: ...

    def touch(self, locator: ShardLocator) -> None: ...


__all__ = ["ShardResolver"]
