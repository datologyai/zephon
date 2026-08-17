# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Dense node-wide Parquet row-group slots over mmap-backed catalogs."""

from __future__ import annotations

import bisect
import functools
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from zephon._internal.io.catalog import CatalogSet, ShardCatalog
from zephon._internal.io.formats.parquet_cache.codec import DecodedRGIdentity
from zephon._internal.io.types import ShardLocator

_IDENTITY_CACHE_SIZE = 4096


@dataclass(frozen=True)
class ParquetRGLocation:
    """Transient identity for one dense decoded row-group slot."""

    global_slot: int
    dataset_name: str
    shard_slot: int
    shard_id: int
    rg_id: int
    locator: ShardLocator


class ParquetRGIndex:
    """Map ``(dataset, shard, row group)`` to a global cache slot ID.

    Built from ``CatalogSet``, it lets readers and ``ParquetRGCache`` agree on
    slot numbers and creates the identity checked by ``ArrowRGFileCodec``.
    """

    def __init__(self, catalog_set: CatalogSet) -> None:
        self._names: list[str] = []
        self._catalogs: list[ShardCatalog] = []
        self._rg_offsets: list[np.ndarray] = []
        self._dataset_bases: list[int] = []
        self._index_of: dict[str, int] = {}

        total = 0
        hasher = hashlib.sha256()
        hasher.update(b"zephon-parquet-rg-index-v1\0")
        for name in catalog_set.cacheable_names:
            catalog = catalog_set.catalog_for(name)
            if catalog.format != "parquet":
                continue
            offsets = self._validate_offsets(
                name,
                catalog,
                catalog.extra_int_column("rg_off"),
            )

            index = len(self._names)
            self._names.append(name)
            self._catalogs.append(catalog)
            self._rg_offsets.append(offsets)
            self._dataset_bases.append(total)
            self._index_of[name] = index

            dataset_rg_count = int(offsets[-1])
            total += dataset_rg_count
            hasher.update(name.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(catalog.fingerprint.encode("ascii"))
            hasher.update(b"\0")
            hasher.update(dataset_rg_count.to_bytes(8, "little", signed=False))

        self.num_row_groups = total
        self.fingerprint = "sha256:" + hasher.hexdigest()
        self._identity_for_cached: Callable[[int], DecodedRGIdentity] = (
            functools.lru_cache(maxsize=_IDENTITY_CACHE_SIZE)(
                self._build_identity,
            )
        )

    @staticmethod
    def _validate_offsets(
        name: str,
        catalog: ShardCatalog,
        offsets: np.ndarray | None,
    ) -> np.ndarray:
        if offsets is None:
            raise ValueError(
                f"Parquet catalog {name!r} has no columnar row-group offsets"
            )
        if offsets.ndim != 1 or len(offsets) != catalog.shard_count + 1:
            raise ValueError(
                f"Parquet catalog {name!r} has invalid row-group offset shape"
            )
        if int(offsets[0]) != 0 or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError(
                f"Parquet catalog {name!r} has non-monotonic row-group offsets"
            )
        return offsets

    def __getstate__(self):  # pragma: no cover - guards a programming error
        raise TypeError(
            "ParquetRGIndex must not be pickled; it holds mmap-backed catalog views"
        )

    def slot_of(self, dataset_name: str, shard_id: int, rg_id: int) -> int | None:
        """Map logical RG identity to its dense global slot."""
        dataset_index = self._index_of.get(dataset_name)
        if dataset_index is None:
            return None
        catalog = self._catalogs[dataset_index]
        try:
            shard_slot = catalog.slot_of(shard_id)
        except KeyError:
            return None
        offsets = self._rg_offsets[dataset_index]
        first = int(offsets[shard_slot])
        rg_count = int(offsets[shard_slot + 1]) - first
        if rg_id < 0 or rg_id >= rg_count:
            return None
        return self._dataset_bases[dataset_index] + first + rg_id

    def locate(self, global_slot: int) -> ParquetRGLocation:
        """Synthesize the logical/source identity for one dense RG slot."""
        if global_slot < 0 or global_slot >= self.num_row_groups:
            raise IndexError(global_slot)
        dataset_index = bisect.bisect_right(self._dataset_bases, global_slot) - 1
        dataset_name = self._names[dataset_index]
        catalog = self._catalogs[dataset_index]
        local_rg_slot = global_slot - self._dataset_bases[dataset_index]
        offsets = self._rg_offsets[dataset_index]
        shard_slot = int(np.searchsorted(offsets, local_rg_slot, side="right")) - 1
        rg_id = local_rg_slot - int(offsets[shard_slot])
        locator = catalog.locator_at(shard_slot, dataset_name=dataset_name)
        return ParquetRGLocation(
            global_slot=global_slot,
            dataset_name=dataset_name,
            shard_slot=shard_slot,
            shard_id=locator.shard_id,
            rg_id=rg_id,
            locator=locator,
        )

    def identity_for(self, global_slot: int) -> DecodedRGIdentity:
        """Return the canonical source/catalog identity for one payload."""
        return self._identity_for_cached(global_slot)

    def _build_identity(self, global_slot: int) -> DecodedRGIdentity:
        location = self.locate(global_slot)
        locator = location.locator
        canonical = {
            "cache_format_version": 1,
            "payload_format_version": 1,
            "decoded_catalog_fingerprint": self.fingerprint,
            "dataset_name": location.dataset_name,
            "shard_id": location.shard_id,
            "rg_id": location.rg_id,
            "global_slot": global_slot,
            "projection_policy": "full-schema-v1",
            "source": {
                "root": locator.root,
                "basename": locator.raw.basename,
                "bytes": locator.raw.bytes,
                "hashes": sorted(locator.raw.hashes.items()),
            },
        }
        encoded = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return DecodedRGIdentity("sha256:" + hashlib.sha256(encoded).hexdigest())


__all__ = ["ParquetRGIndex", "ParquetRGLocation"]
