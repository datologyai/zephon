# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Build a columnar catalog artifact from a dataset's ``discover()`` output.

The builder runs the *full* discovery — ``handler.discover()`` then
``handler.build_locators()`` over a temporary :class:`~zephon.io.dataset.Dataset`
— so the columns are columnarized from the *canonical* :class:`ShardLocator`
objects. That guarantees ``ShardCatalog.locator_at`` synthesizes locators that
match what ``build_locators`` would have produced. This is the heavy step; it
runs at most once per node.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from zephon.io.catalog import io as catalog_io
from zephon.io.catalog.extra_codec import get_extra_codec, packb
from zephon.io.types import ShardLocator


@dataclass
class BuiltCatalog:
    """A freshly built catalog: its content fingerprint and packed file bytes."""

    fingerprint: str
    file_bytes: bytes


@dataclass(frozen=True)
class DatasetHeader:
    """Small, immutable description of what to (re)build and where it points."""

    name: str  # logical alias; never part of the fingerprint/file
    root: str  # what discover() reads; what ShardLocator.root needs
    format: str  # which handler to rebuild with
    path: str | None  # diagnostics only


def _pack_var(items: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    """Pack variable-length byte strings into ``(offsets[M+1], data[])``."""
    offsets = np.zeros(len(items) + 1, dtype=np.int64)
    total = 0
    for i, blob in enumerate(items):
        total += len(blob)
        offsets[i + 1] = total
    data = (
        np.frombuffer(b"".join(items), dtype=np.uint8)
        if total
        else np.empty(0, dtype=np.uint8)
    )
    return offsets, data


def _fingerprint(fmt: str, root: str, columns: dict) -> str:
    """O(M) content hash over physical identity + packed column bytes.

    Covers ``format``/``root`` and every column buffer; excludes logical identity
    (``name``/``dataset_id``), so two aliases of one physical dataset share one
    artifact. ``path`` is diagnostics-only and equals ``root``, so it is not
    hashed. Hashes numpy buffers directly (no per-shard dict graph).
    """
    hasher = hashlib.sha256()
    hasher.update(b"zephon-catalog-v")
    hasher.update(catalog_io.MAGIC)
    for part in (fmt, root):
        encoded = part.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)
    for name in sorted(columns):
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(np.ascontiguousarray(columns[name]))
    return "sha256:" + hasher.hexdigest()


def build_catalog(header: DatasetHeader) -> BuiltCatalog:
    """Run full discovery for ``header`` and pack a single-file catalog artifact.

    Discovers, builds the canonical locators via the format handler, then
    columnarizes them with :func:`pack_locators`. ``header.name`` participates
    only in ``build_locators``, never in the fingerprint or stored file.
    """
    # Lazy imports to avoid an import cycle (dataset/formats -> catalog -> ...).
    from zephon.io.dataset import Dataset
    from zephon.io.formats import ensure_builtin_formats
    from zephon.io.formats.base import get_format
    from zephon.io.storage import RouterStorageBackend

    ensure_builtin_formats(required={header.format})
    handler = get_format(header.format)
    storage = RouterStorageBackend()

    shard_index, shard_meta = handler.discover(header.root, storage)
    tmp_dataset = Dataset(
        name=header.name,
        shard_index=dict(shard_index),
        backend={"kind": header.format, "path": header.root, "shards": shard_meta},
        path=header.path,
    )
    locators = dict(handler.build_locators(tmp_dataset))
    return pack_locators(header, locators, shard_index)


def pack_locators(
    header: DatasetHeader,
    locators: Mapping[int, ShardLocator],
    num_rows_by_shard: Mapping[int, int],
) -> BuiltCatalog:
    """Columnarize ``shard_id -> ShardLocator`` into a single-file catalog artifact.

    The lower half of :func:`build_catalog`, callable directly (e.g. from tests)
    with synthetic locators. ``num_rows_by_shard`` maps ``shard_id -> row count``.
    """
    from zephon.io.catalog.io import SCHEMA_VERSION

    shard_ids = sorted(int(sid) for sid in locators)
    ordered = [locators[sid] for sid in shard_ids]
    count = len(ordered)

    num_rows = np.array([num_rows_by_shard[sid] for sid in shard_ids], dtype=np.int64)
    raw_bytes = np.array([loc.raw.bytes for loc in ordered], dtype=np.int64)
    raw_basename_off, raw_basename_data = _pack_var(
        [loc.raw.basename.encode("utf-8") for loc in ordered]
    )

    columns: dict[str, np.ndarray] = {
        "shard_id": np.array(shard_ids, dtype=np.int64),
        "num_rows": num_rows,
        "raw_bytes": raw_bytes,
        "raw_basename_off": raw_basename_off,
        "raw_basename_data": raw_basename_data,
    }
    present: dict[str, bool] = {}

    if any(loc.raw.hashes for loc in ordered):
        off, data = _pack_var(
            [packb(dict(loc.raw.hashes)) if loc.raw.hashes else b"" for loc in ordered]
        )
        columns["raw_hashes_off"] = off
        columns["raw_hashes_data"] = data
        present["raw_hashes"] = True

    if any(loc.zip is not None for loc in ordered):
        columns["zip_bytes"] = np.array(
            [loc.zip.bytes if loc.zip is not None else 0 for loc in ordered],
            dtype=np.int64,
        )
        zoff, zdata = _pack_var(
            [
                loc.zip.basename.encode("utf-8") if loc.zip is not None else b""
                for loc in ordered
            ]
        )
        columns["zip_basename_off"] = zoff
        columns["zip_basename_data"] = zdata
        present["zip"] = True
        if any(loc.zip is not None and loc.zip.hashes for loc in ordered):
            hoff, hdata = _pack_var(
                [
                    packb(dict(loc.zip.hashes))
                    if loc.zip is not None and loc.zip.hashes
                    else b""
                    for loc in ordered
                ]
            )
            columns["zip_hashes_off"] = hoff
            columns["zip_hashes_data"] = hdata
            present["zip_hashes"] = True

    if any(loc.compression for loc in ordered):
        coff, cdata = _pack_var(
            [(loc.compression or "").encode("utf-8") for loc in ordered]
        )
        columns["compression_off"] = coff
        columns["compression_data"] = cdata
        present["compression"] = True

    extras = [loc.extra for loc in ordered]
    codec = get_extra_codec(header.format)
    encoded = codec.encode(extras, num_rows)

    # The codec stores its owned keys in the header / int columns. Every other
    # key is preserved generically here: msgpack the per-shard "rest", hoisting
    # it to a single header blob when it is byte-identical across shards.
    owned = encoded.owned_keys
    rest_packed = [
        packb({k: v for k, v in e.items() if k not in owned}) if e else b""
        for e in extras
    ]
    rest_constant = bool(count) and all(blob == rest_packed[0] for blob in rest_packed)

    if encoded.header_blob is not None:
        columns["extra_header_blob"] = np.frombuffer(
            encoded.header_blob, dtype=np.uint8
        )
        present["extra_header"] = True
    extra_int_columns: list[str] = []
    for name, arr in encoded.int_columns.items():
        columns[f"extra_int_{name}"] = np.ascontiguousarray(arr, dtype=np.int64)
        extra_int_columns.append(name)

    if rest_constant:
        if rest_packed and rest_packed[0]:
            columns["extra_rest_header_blob"] = np.frombuffer(
                rest_packed[0], dtype=np.uint8
            )
            present["extra_rest_header"] = True
    else:
        rest_off, rest_data = _pack_var(rest_packed)
        columns["extra_rest_off"] = rest_off
        columns["extra_rest_data"] = rest_data
        present["extra_rest"] = True

    dense = shard_ids == list(range(count))
    fingerprint = _fingerprint(header.format, header.root, columns)

    header_fields = {
        "schema_version": SCHEMA_VERSION,
        "format": header.format,
        "root": header.root,
        "path": header.path,
        "shard_count": count,
        "dense": dense,
        "fingerprint": fingerprint,
        "total_raw_bytes": int(raw_bytes.sum()),
        "present": present,
        "extra_flags": {**encoded.flags, "rest_constant": rest_constant},
        "extra_int_columns": extra_int_columns,
    }
    file_bytes = catalog_io.pack_catalog(header_fields, columns)
    return BuiltCatalog(fingerprint=fingerprint, file_bytes=file_bytes)


__all__ = ["BuiltCatalog", "DatasetHeader", "build_catalog", "pack_locators"]
