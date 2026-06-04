# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Zero-copy catalog views: per-dataset ``ShardCatalog`` and global ``CatalogSet``.

``ShardLocator`` stays the public type but is **synthesized per call** from the
mmap-backed columns, never materialized en masse. ``CatalogSet`` composes the
per-dataset catalogs into the global, name-sorted dense slot space and exposes
``slot_of`` / ``locator_at`` / fingerprint / summary.
"""

from __future__ import annotations

import bisect
import collections.abc
import hashlib
from typing import Iterator, Mapping

import numpy as np

from zephon.io.catalog.extra_codec import get_extra_codec, unpackb
from zephon.io.catalog.io import LoadedCatalog
from zephon.io.types import ShardFile, ShardLocator

_UNSET = object()


class _LazyExtra(collections.abc.Mapping):
    """A ``Mapping`` that decodes its contents on first access.

    ``dict(extra)`` works (``open_shard`` needs it) but nothing decodes until
    touched — one small object per ``open()``, not a resident per-shard dict.
    """

    __slots__ = ("_decode", "_cached")

    def __init__(self, decode) -> None:
        self._decode = decode
        self._cached = _UNSET

    def _materialize(self) -> Mapping:
        if self._cached is _UNSET:
            decoded = self._decode()
            self._cached = dict(decoded) if decoded else {}
        return self._cached  # type: ignore[return-value]

    def __getitem__(self, key):
        return self._materialize()[key]

    def __iter__(self) -> Iterator:
        return iter(self._materialize())

    def __len__(self) -> int:
        return len(self._materialize())


class ShardCatalog:
    """Per-dataset, mmap-backed view over the columnar catalog file.

    Name-agnostic on purpose (the fingerprint excludes logical identity), so a
    single mapping is shared by every alias of one physical dataset; the dataset
    name is bound by the caller at ``locator_at``.
    """

    def __init__(self, loaded: LoadedCatalog) -> None:
        self._loaded = loaded
        self._header = loaded.header
        self._cols = loaded.columns
        self._format: str = self._header["format"]
        self._root: str = self._header["root"]
        self._dense: bool = self._header["dense"]
        self._present: dict = dict(self._header.get("present", {}))
        self._extra_flags: dict = dict(self._header.get("extra_flags", {}))
        self._extra_int_columns: list[str] = list(
            self._header.get("extra_int_columns", [])
        )
        self._shard_id = self._cols["shard_id"]
        self._num_rows = self._cols["num_rows"]
        # Codecs register at format-module import, so a cold attach in a fresh
        # worker must force that import — the default-codec fallback would
        # silently drop codec-owned extra keys. missing_ok: non-builtin kinds
        # register themselves. Imported lazily (formats -> storage -> catalog
        # is a module cycle).
        from zephon.io.formats import ensure_builtin_formats

        ensure_builtin_formats(required={self._format}, missing_ok=True)
        self._codec = get_extra_codec(self._format)
        header_blob = None
        if self._present.get("extra_header"):
            header_blob = self._cols["extra_header_blob"].tobytes()
        self._extra_header_obj = self._codec.decode_header(
            header_blob, self._extra_flags
        )
        # Generic "rest" of extra (keys the codec doesn't own): a single constant
        # blob or a per-shard column. Merged under the codec's reconstruction.
        self._rest_constant: bool = bool(self._extra_flags.get("rest_constant"))
        self._rest_header_obj = (
            unpackb(self._cols["extra_rest_header_blob"].tobytes())
            if self._present.get("extra_rest_header")
            else None
        )
        # Lazy basename->slot index for the RESUME reverse lookup.
        self._bn_slots: np.ndarray | None = None
        self._bn_roles: np.ndarray | None = None

    @property
    def fingerprint(self) -> str:
        return self._header["fingerprint"]

    @property
    def shard_count(self) -> int:
        return self._header["shard_count"]

    @property
    def root(self) -> str:
        return self._root

    @property
    def total_raw_bytes(self) -> int:
        return self._header.get("total_raw_bytes", 0)

    def ids(self) -> np.ndarray:
        return self._shard_id

    def num_rows(self) -> np.ndarray:
        return self._num_rows

    def total(self) -> int:
        return int(self._num_rows.sum())

    def max_count(self) -> int:
        return int(self._num_rows.max()) if self.shard_count else 0

    def slot_of(self, shard_id: int) -> int:
        """Map ``shard_id`` to a dense slot (identity when dense, else search)."""
        shard_id = int(shard_id)
        if self._dense:
            if 0 <= shard_id < self.shard_count:
                return shard_id
            raise KeyError(shard_id)
        pos = int(np.searchsorted(self._shard_id, shard_id))
        if pos >= self.shard_count or self._shard_id[pos] != shard_id:
            raise KeyError(shard_id)
        return pos

    def _var_at(self, off_name: str, data_name: str, slot: int) -> bytes:
        off = self._cols[off_name]
        start = off[slot]
        end = off[slot + 1]
        if end <= start:
            return b""
        return self._cols[data_name][start:end].tobytes()

    def _decode_extra(self, slot: int):
        # Full arrays — the codec indexes (litdata Interval) or slices (ragged
        # parquet row_groups) them itself.
        int_cols = {
            name: self._cols[f"extra_int_{name}"] for name in self._extra_int_columns
        }
        owned = self._codec.decode(
            slot,
            self._extra_header_obj,
            int_cols,
            int(self._num_rows[slot]),
            self._extra_flags,
        )
        if self._rest_constant:
            rest = self._rest_header_obj
        elif self._present.get("extra_rest"):
            rest = unpackb(self._var_at("extra_rest_off", "extra_rest_data", slot))
        else:
            rest = None
        if not rest:
            return owned
        merged = dict(rest)
        if owned:
            merged.update(owned)
        return merged

    def locator_at(self, slot: int, *, dataset_name: str) -> ShardLocator:
        """Synthesize one ``ShardLocator`` for ``slot`` (transient, per call)."""
        raw_basename = self._var_at(
            "raw_basename_off", "raw_basename_data", slot
        ).decode("utf-8")
        raw_hashes: Mapping[str, str] = {}
        if self._present.get("raw_hashes"):
            decoded = unpackb(self._var_at("raw_hashes_off", "raw_hashes_data", slot))
            if decoded:
                raw_hashes = decoded
        raw = ShardFile(
            basename=raw_basename,
            bytes=int(self._cols["raw_bytes"][slot]),
            hashes=raw_hashes,
        )

        zip_file: ShardFile | None = None
        if self._present.get("zip"):
            zip_basename = self._var_at(
                "zip_basename_off", "zip_basename_data", slot
            ).decode("utf-8")
            if zip_basename:
                zip_hashes: Mapping[str, str] = {}
                if self._present.get("zip_hashes"):
                    decoded = unpackb(
                        self._var_at("zip_hashes_off", "zip_hashes_data", slot)
                    )
                    if decoded:
                        zip_hashes = decoded
                zip_file = ShardFile(
                    basename=zip_basename,
                    bytes=int(self._cols["zip_bytes"][slot]),
                    hashes=zip_hashes,
                )

        compression: str | None = None
        if self._present.get("compression"):
            comp = self._var_at("compression_off", "compression_data", slot).decode(
                "utf-8"
            )
            compression = comp or None

        return ShardLocator(
            dataset=dataset_name,
            shard_id=int(self._shard_id[slot]),
            format=self._format,
            root=self._root,
            raw=raw,
            zip=zip_file,
            compression=compression,
            extra=_LazyExtra(lambda s=slot: self._decode_extra(s)),
        )

    def _ensure_basename_index(self) -> None:
        if self._bn_slots is not None:
            return
        # Build the basename->slot index for the RESUME reverse lookup. Sort the
        # (raw, plus present zip) basenames via a *transient* numpy fixed-width
        # 'S' array — no list of M (str, int, int) tuples and only one Python
        # bytes object alive at a time. Only the int64 slot + int8 role arrays,
        # sorted by basename, are retained; the 'S' array is discarded and
        # lookup decodes basenames from the columns per probe.
        count = self.shard_count
        has_zip = bool(self._present.get("zip"))
        if count == 0:
            self._bn_slots = np.empty(0, dtype=np.int64)
            self._bn_roles = np.empty(0, dtype=np.int8)
            return

        raw_lens = np.diff(self._cols["raw_basename_off"])
        maxlen = int(raw_lens.max())
        total = count
        zip_lens = None
        if has_zip:
            zip_lens = np.diff(self._cols["zip_basename_off"])
            maxlen = max(maxlen, int(zip_lens.max()))
            total += int((zip_lens > 0).sum())

        names = np.empty(total, dtype=f"S{max(1, maxlen)}")
        slots = np.empty(total, dtype=np.int64)
        roles = np.empty(total, dtype=np.int8)
        k = 0
        for slot in range(count):
            names[k] = self._var_at("raw_basename_off", "raw_basename_data", slot)
            slots[k], roles[k] = slot, 0
            k += 1
        if has_zip:
            assert zip_lens is not None
            for slot in range(count):
                if zip_lens[slot] > 0:
                    names[k] = self._var_at(
                        "zip_basename_off", "zip_basename_data", slot
                    )
                    slots[k], roles[k] = slot, 1
                    k += 1

        order = np.argsort(names, kind="stable")
        self._bn_slots = slots[order]
        self._bn_roles = roles[order]

    def reverse_lookup(self, basename: str) -> tuple[int, str] | None:
        """Resolve ``basename -> (slot, role)`` via a binary search over columns.

        Used only on the RESUME disk-scan reconcile path. Compares raw UTF-8
        bytes so the search order matches the byte-order sort built in
        ``_ensure_basename_index``.
        """
        self._ensure_basename_index()
        assert self._bn_slots is not None and self._bn_roles is not None
        target = basename.encode("utf-8")
        lo, hi = 0, self._bn_slots.size
        while lo < hi:
            mid = (lo + hi) // 2
            slot = int(self._bn_slots[mid])
            role = "zip" if self._bn_roles[mid] else "raw"
            off, data = (
                ("zip_basename_off", "zip_basename_data")
                if role == "zip"
                else ("raw_basename_off", "raw_basename_data")
            )
            name = self._var_at(off, data, slot)
            if name < target:
                lo = mid + 1
            elif name > target:
                hi = mid
            else:
                return slot, role
        return None


class CatalogSet:
    """Global composition over per-dataset catalogs, built at op setup.

    Holds references to mmap-backed catalogs; it **must never be pickled** —
    shipping it would copy buffers back into workers.
    """

    def __init__(self, entries: Mapping[int, tuple[str, ShardCatalog]]) -> None:
        # Global slots are sorted-name then sorted-shard_id (matching the
        # CacheManager numbering); dataset_id is not part of the slot space.
        by_name: dict[str, ShardCatalog] = {}
        for name, catalog in entries.values():
            if name in by_name:
                raise ValueError(
                    f"Duplicate dataset name {name!r} in catalog set; the dense "
                    "slot map, disk layout and fingerprint all key by name."
                )
            by_name[name] = catalog
        self._names: list[str] = sorted(by_name)
        self._by_name = by_name
        self._bases: list[int] = []
        self._base_of: dict[str, int] = {}
        base = 0
        for name in self._names:
            self._bases.append(base)
            self._base_of[name] = base
            base += by_name[name].shard_count
        self._num_shards = base

    def __getstate__(self):  # pragma: no cover - guards a programming error
        raise TypeError(
            "CatalogSet must not be pickled; it holds mmap-backed catalog views. "
            "Ship the per-dataset catalog handle instead."
        )

    @property
    def num_shards(self) -> int:
        return self._num_shards

    @property
    def cacheable_names(self) -> tuple[str, ...]:
        return tuple(self._names)

    def catalog_for(self, name: str) -> ShardCatalog:
        return self._by_name[name]

    def _name_index(self, global_slot: int) -> int:
        return bisect.bisect_right(self._bases, global_slot) - 1

    def slot_of(self, dataset_name: str, shard_id: int) -> int | None:
        catalog = self._by_name.get(dataset_name)
        if catalog is None:
            return None
        try:
            local = catalog.slot_of(shard_id)
        except KeyError:
            return None
        return self._base_of[dataset_name] + local

    def locator_at(self, global_slot: int) -> ShardLocator:
        idx = self._name_index(global_slot)
        name = self._names[idx]
        local = global_slot - self._bases[idx]
        return self._by_name[name].locator_at(local, dataset_name=name)

    def reverse_lookup(self, name: str, basename: str) -> tuple[int, str] | None:
        """Resolve ``(name, basename) -> (global_slot, role)`` (RESUME only).

        Delegates to the per-dataset catalog's binary search over the basename
        columns, so no per-shard Python dict and no synthesized locators are
        materialized.
        """
        catalog = self._by_name.get(name)
        if catalog is None:
            return None
        found = catalog.reverse_lookup(basename)
        if found is None:
            return None
        local_slot, role = found
        return self._base_of[name] + local_slot, role

    @property
    def global_fingerprint(self) -> str:
        hasher = hashlib.sha256()
        for name in self._names:
            hasher.update(name.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(self._by_name[name].fingerprint.encode("utf-8"))
            hasher.update(b"\0")
        return "sha256:" + hasher.hexdigest()

    def summary(self) -> list[dict]:
        return [
            {
                "name": name,
                "path": self._by_name[name].root,
                "shard_count": self._by_name[name].shard_count,
                "total_raw_bytes": self._by_name[name].total_raw_bytes,
            }
            for name in self._names
        ]


__all__ = ["CatalogSet", "ShardCatalog"]
