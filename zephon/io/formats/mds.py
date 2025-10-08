# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""MDS shard format integration via mosaicml-streaming."""

import json
import os
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, cast

from zephon.io.formats.base import FormatHandler, register_format
from zephon.io.protocols import RandomAccessShard
from zephon.io.storage import StorageBackend
from zephon.io.types import LocalShardRef, ShardFile, ShardLocator

if TYPE_CHECKING:
    from zephon.io.dataset import Dataset

try:
    from streaming.base.format.mds.reader import MDSShard as _StreamingMDSShard
except Exception:
    _StreamingMDSShard = None


class _UnderlyingMDSShard(Protocol):
    def __getitem__(self, index: int) -> dict[str, object]: ...

    def __len__(self) -> int: ...


class _PassthroughMDSShard(RandomAccessShard):
    def __init__(self, shard: _UnderlyingMDSShard) -> None:
        self._shard = shard

    def __getitem__(self, index: int):
        return self._shard[index]

    def __len__(self) -> int:
        return len(self._shard)

    def close(self) -> None:
        close_method = cast(
            Callable[[], None] | None, getattr(self._shard, "close", None)
        )
        if close_method is not None:
            close_method()


class MDSFormat(FormatHandler):
    """Format handler for Mosaic MDS datasets."""

    kind = "mds"

    def discover(
        self, path: str, storage: StorageBackend
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, object]]]:
        """Parse ``index.json`` under ``path`` and build shard metadata."""
        index_path = os.path.join(path, "index.json")
        try:
            with storage.open(index_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError as exc:
            raise ValueError(f"Missing MDS index: {index_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse MDS index: {index_path}") from exc

        shards = data.get("shards")
        if not isinstance(shards, list):
            raise ValueError("MDS index missing 'shards' list")

        shard_index: dict[int, int] = {}
        shard_meta: dict[int, dict[str, object]] = {}
        for shard_id, entry in enumerate(shards):
            if not isinstance(entry, Mapping):
                raise ValueError(f"Shard entry {shard_id} must be a mapping")
            samples = entry.get("samples")
            if samples is None:
                raise ValueError(f"Shard {shard_id} missing 'samples' count")
            try:
                count = int(samples)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Shard {shard_id} has invalid sample count: {samples}"
                ) from exc
            shard_index[shard_id] = count
            shard_meta[shard_id] = _normalize_shard(entry, shard_id)

        return shard_index, shard_meta

    def build_locators(self, dataset: "Dataset") -> Mapping[int, ShardLocator]:
        backend = dataset.backend
        path = backend.get("path")
        if not isinstance(path, str):
            raise ValueError("MDS dataset missing 'path' metadata")
        shards = backend.get("shards")
        if not isinstance(shards, Mapping):
            raise ValueError("MDS dataset missing 'shards' metadata")

        locators: dict[int, ShardLocator] = {}
        for shard_id_obj, meta in shards.items():
            shard_id = int(shard_id_obj)
            if not isinstance(meta, Mapping):
                raise ValueError(f"Invalid MDS shard metadata for shard {shard_id}")
            raw_meta = meta.get("raw")
            if not isinstance(raw_meta, Mapping):
                raise ValueError(f"Shard {shard_id} missing raw metadata")
            raw = _build_file(raw_meta, shard_id, "raw")
            zip_meta = meta.get("zip")
            zip_file = None
            if isinstance(zip_meta, Mapping):
                zip_file = _build_file(zip_meta, shard_id, "zip")
            compression = meta.get("compression")
            extra = (
                meta.get("extra") if isinstance(meta.get("extra"), Mapping) else None
            )
            locators[shard_id] = ShardLocator(
                dataset=dataset.name,
                shard_id=shard_id,
                format=self.kind,
                root=path,
                raw=raw,
                zip=zip_file,
                compression=str(compression) if compression else None,
                extra=extra,
            )
        return locators

    def open_shard(
        self, locator: ShardLocator, local_ref: LocalShardRef
    ) -> RandomAccessShard:
        if _StreamingMDSShard is None:
            raise RuntimeError(
                "Opening MDS shards requires the 'mosaicml-streaming' package; install it to proceed"
            )
        kwargs = {}
        if local_ref.extra:
            kwargs = dict(local_ref.extra)
        try:
            shard = _StreamingMDSShard(
                raw=local_ref.raw.path,
                zip_file=local_ref.zip.path if local_ref.zip else None,
                compression=local_ref.compression,
                hashes=locator.raw.hashes,
                **kwargs,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Unsupported mosaicml-streaming version; please upgrade to a recent release"
            ) from exc
        return _PassthroughMDSShard(shard)


def _build_file(meta: Mapping[str, object], shard_id: int, kind: str) -> ShardFile:
    basename = meta.get("basename")
    if not isinstance(basename, str):
        raise ValueError(f"Shard {shard_id} missing basename for {kind} file")
    bytes_value = meta.get("bytes")
    if bytes_value is None:
        raise ValueError(f"Shard {shard_id} missing byte size for {kind} file")
    if not isinstance(bytes_value, (int, str)):
        raise ValueError(
            f"Shard {shard_id} invalid byte size for {kind} file: {bytes_value}"
        )
    try:
        bytes_int = int(bytes_value)
    except ValueError as exc:
        raise ValueError(
            f"Shard {shard_id} invalid byte size for {kind} file: {bytes_value}"
        ) from exc
    hashes_obj = meta.get("hashes") or {}
    if not isinstance(hashes_obj, Mapping):
        raise ValueError(f"Shard {shard_id} invalid hashes for {kind} file")
    hashes = {str(k): str(v) for k, v in hashes_obj.items()}
    return ShardFile(basename=basename, bytes=bytes_int, hashes=hashes)


def _normalize_shard(entry: Mapping[str, object], shard_id: int) -> dict[str, object]:
    shard_meta: dict[str, object] = {}

    raw_meta = entry.get("raw") or entry.get("data")
    if not isinstance(raw_meta, Mapping):
        raise ValueError(f"Shard {shard_id} missing raw file metadata")
    raw_basename = raw_meta.get("basename") or raw_meta.get("path")
    if not isinstance(raw_basename, str):
        raise ValueError(f"Shard {shard_id} missing raw basename")
    raw_size_obj = raw_meta.get("bytes") or raw_meta.get("size")
    if raw_size_obj is None:
        raise ValueError(f"Shard {shard_id} missing raw byte size")
    if isinstance(raw_size_obj, int):
        raw_bytes = raw_size_obj
    elif isinstance(raw_size_obj, str):
        try:
            raw_bytes = int(raw_size_obj)
        except ValueError as exc:
            raise ValueError(
                f"Shard {shard_id} invalid raw byte size: {raw_size_obj}"
            ) from exc
    else:
        raise ValueError(
            f"Shard {shard_id} invalid raw byte size type: {type(raw_size_obj).__name__}"
        )
    hashes_obj = raw_meta.get("hashes")
    if isinstance(hashes_obj, Mapping):
        hashes = {str(k): str(v) for k, v in hashes_obj.items()}
    else:
        hashes = {}
    shard_meta["raw"] = {
        "basename": raw_basename,
        "bytes": raw_bytes,
        "hashes": hashes,
    }

    zip_meta = entry.get("zip")
    if isinstance(zip_meta, Mapping):
        zip_basename = zip_meta.get("basename") or zip_meta.get("path")
        if not isinstance(zip_basename, str):
            raise ValueError(f"Shard {shard_id} missing zip basename")
        zip_size_obj = zip_meta.get("bytes") or zip_meta.get("size")
        if zip_size_obj is None:
            raise ValueError(f"Shard {shard_id} missing zip byte size")
        if isinstance(zip_size_obj, int):
            zip_bytes = zip_size_obj
        elif isinstance(zip_size_obj, str):
            try:
                zip_bytes = int(zip_size_obj)
            except ValueError as exc:
                raise ValueError(
                    f"Shard {shard_id} invalid zip byte size: {zip_size_obj}"
                ) from exc
        else:
            raise ValueError(
                f"Shard {shard_id} invalid zip byte size type: {type(zip_size_obj).__name__}"
            )
        zip_hashes_obj = zip_meta.get("hashes")
        if isinstance(zip_hashes_obj, Mapping):
            zip_hashes = {str(k): str(v) for k, v in zip_hashes_obj.items()}
        else:
            zip_hashes = {}
        shard_meta["zip"] = {
            "basename": zip_basename,
            "bytes": zip_bytes,
            "hashes": zip_hashes,
        }

    compression = entry.get("compression")
    if compression is not None:
        if not isinstance(compression, str):
            raise ValueError(
                f"Shard {shard_id} has invalid compression value: {compression}"
            )
        shard_meta["compression"] = compression

    extras = dict(entry)
    for key in ("samples", "raw", "data", "zip", "compression"):
        extras.pop(key, None)
    if extras:
        shard_meta["extra"] = extras

    return shard_meta


register_format(MDSFormat())

__all__ = ["MDSFormat"]
