# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Base interfaces and registry for shard formats."""

from typing import TYPE_CHECKING, Mapping, Protocol

import numpy as np

from zephon._internal.io.protocols import RandomAccessShard
from zephon._internal.io.storage import StorageBackend
from zephon._internal.io.types import LocalShardRef, ShardLocator

if TYPE_CHECKING:
    from zephon.io.dataset import Dataset


class ShardOpener(Protocol):
    """Create readable shards from resolved local references.

    Stateless format handlers can serve directly. Stateful formats may instead
    use a store-scoped opener carrying resources such as a decoded cache.
    """

    def open_shard(
        self, locator: ShardLocator, local_ref: LocalShardRef
    ) -> RandomAccessShard: ...


class FormatHandler(ShardOpener, Protocol):
    """Format-specific discovery and shard-reading behavior."""

    kind: str

    def discover(
        self, path: str, storage: StorageBackend
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, object]]]:
        """Inspect ``path`` and return shard counts plus locator metadata.

        The returned tuple feeds the catalog build
        (:func:`zephon._internal.io.catalog.builder.build_catalog`), which packs the
        counts and locators into the catalog artifact. Implementations may
        perform filesystem or remote IO via ``storage`` but should avoid
        opening shards eagerly beyond what is required to compute counts.
        """
        ...

    def discover_counts(
        self, path: str, storage: StorageBackend
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-shard counts as ``(shard_ids, num_rows)`` int64 arrays.

        Default implementation runs the full ``discover`` and keeps only the
        counts. Formats with an index override this with a metadata-only fast
        path and defer to ``super()`` otherwise; either way the result must
        agree with ``discover`` on ``(shard_id, num_rows)``.
        """
        shard_index, _ = self.discover(path, storage)
        ids = sorted(shard_index)
        return (
            np.array(ids, dtype=np.int64),
            np.array([shard_index[i] for i in ids], dtype=np.int64),
        )

    def build_locators(self, dataset: "Dataset") -> Mapping[int, ShardLocator]: ...


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


__all__ = ["FormatHandler", "ShardOpener", "get_format", "register_format"]
