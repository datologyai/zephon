# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Base interfaces and registry for shard formats."""

from typing import TYPE_CHECKING, Mapping, Protocol

from zephon.io.protocols import RandomAccessShard
from zephon.io.storage import StorageBackend
from zephon.io.types import LocalShardRef, ShardLocator

if TYPE_CHECKING:
    from zephon.io.dataset import Dataset


class FormatHandler(Protocol):
    """Format-specific logic for constructing shard locators and readers."""

    kind: str

    def discover(
        self, path: str, storage: StorageBackend
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, object]]]:
        """Inspect ``path`` and return shard counts plus locator metadata.

        The returned tuple is consumed by :class:`zephon.io.dataset.Dataset`
        to populate ``shard_index`` and backend metadata. Implementations may
        perform filesystem or remote IO via ``storage`` but should avoid
        opening shards eagerly beyond what is required to compute counts.
        """
        ...

    def build_locators(self, dataset: "Dataset") -> Mapping[int, ShardLocator]: ...

    def open_shard(
        self, locator: ShardLocator, local_ref: LocalShardRef
    ) -> RandomAccessShard: ...


_REGISTRY: dict[str, FormatHandler] = {}


def register_format(handler: FormatHandler) -> None:
    """Register ``handler`` under its declared ``kind``."""
    _REGISTRY[handler.kind] = handler


def get_format(kind: str) -> FormatHandler:
    """Retrieve the previously-registered format handler for ``kind``."""
    try:
        return _REGISTRY[kind]
    except KeyError as exc:
        raise KeyError(f"Format handler not registered for kind '{kind}'") from exc


__all__ = ["FormatHandler", "get_format", "register_format"]
