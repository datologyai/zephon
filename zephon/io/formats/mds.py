# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""MDS shard format integration via mosaicml-streaming."""

import os
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, TypedDict, cast

from zephon.io.formats.base import FormatHandler, register_format
from zephon.io.index import find_and_load_index
from zephon.io.protocols import RandomAccessShard
from zephon.io.storage import StorageBackend
from zephon.io.types import LocalShardRef, ShardFile, ShardLocator

if TYPE_CHECKING:
    from zephon.io.dataset import Dataset

try:
    from streaming.base.format.mds.reader import MDSReader as _StreamingMDSReader
except Exception:
    _StreamingMDSReader = None


class _UnderlyingMDSShard(Protocol):
    def __getitem__(self, index: int) -> dict[str, object]: ...

    def __len__(self) -> int: ...


class _StreamingTemplate(TypedDict):
    column_encodings: tuple[str, ...]
    column_names: tuple[str, ...]
    column_sizes: tuple[int | None, ...]
    compression: str | None
    hashes: tuple[str, ...]
    samples: int
    size_limit: int | str | None
    format: str
    version: int


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

    def getsamples(self, indices: list[int]) -> list[dict[str, object]]:
        return [self._shard[i] for i in indices]


class MDSFormat(FormatHandler):
    """Format handler for Mosaic MDS datasets."""

    kind = "mds"

    def discover(
        self, path: str, storage: StorageBackend
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, object]]]:
        """Parse ``index.json`` under ``path`` and build shard metadata."""
        result = find_and_load_index(path, storage)
        if result is None:
            raise ValueError("Missing MDS index")

        data = result
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
        extras = dict(local_ref.extra) if local_ref.extra else {}

        if _StreamingMDSReader is None:
            raise RuntimeError(
                "Opening MDS shards requires the 'mosaicml-streaming' package; install it to proceed"
            )

        streaming_template_data = extras.get("_streaming_template")
        if not isinstance(streaming_template_data, dict):
            raise RuntimeError("Missing streaming metadata for MDS shard")

        streaming_template = cast(_StreamingTemplate, streaming_template_data)
        entry = _finalize_streaming_entry(streaming_template, locator, local_ref)
        # Use the concrete local directory for this shard and avoid providing a
        # split to prevent duplicating subdirectories (e.g., "1350/1350/").
        split = None
        dirname = str(local_ref.raw.path.parent)
        try:
            streaming_shard = _StreamingMDSReader.from_json(
                dirname=dirname,
                split=split,
                obj=entry,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Unsupported mosaicml-streaming version; please upgrade to a recent release"
            ) from exc
        listing: set[str] = set()
        raw_path = local_ref.raw.path
        if raw_path.exists():
            listing.add(str(raw_path))
        if local_ref.zip is not None and local_ref.zip.path.exists():
            listing.add(str(local_ref.zip.path))
        try:
            streaming_shard.set_up_local(listing, safe_keep_zip=True)
        except (
            TypeError
        ) as exc:  # pragma: no cover - signature mismatch on unexpected versions
            raise RuntimeError(
                "Unsupported mosaicml-streaming version; please upgrade to a recent release"
            ) from exc
        return _PassthroughMDSShard(cast(_UnderlyingMDSShard, streaming_shard))


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
    """Convert raw streaming shard metadata into the normalized Zephon shape."""
    shard_meta: dict[str, object] = {}
    entry_copy = deepcopy(dict(entry))

    raw_meta = _find_mapping(
        entry, ("raw", "data", "raw_data"), shard_id, "raw file metadata"
    )
    if raw_meta is None:
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

    zip_meta = _find_mapping(entry, ("zip", "zip_data"), shard_id, None)
    if zip_meta is not None:
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
    for key in (
        "samples",
        "raw",
        "raw_data",
        "data",
        "zip",
        "zip_data",
        "compression",
    ):
        extras.pop(key, None)
    try:
        streaming_template = _prepare_streaming_template(entry_copy, shard_id)
    except ValueError:
        streaming_template = None
    if streaming_template is not None:
        extras["_streaming_template"] = streaming_template
    shard_meta["extra"] = extras

    return shard_meta


def _find_mapping(
    entry: Mapping[str, object],
    names: tuple[str, ...],
    shard_id: int,
    error_msg: str | None,
) -> Mapping[str, object] | None:
    """Return the first mapping under any of ``names`` while validating types."""
    for name in names:
        value = entry.get(name)
        if isinstance(value, Mapping):
            return value
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(
                f"Shard {shard_id} invalid metadata under '{name}' (expected mapping)"
            )
    if error_msg is not None:
        raise ValueError(f"Shard {shard_id} missing {error_msg}")
    return None


def _prepare_streaming_template(
    entry: Mapping[str, object], shard_id: int
) -> _StreamingTemplate:
    """Extract a stable template for mosaic streaming's ``from_json`` loader.

    The upstream ``index.json`` emitted by different streaming versions mixes
    ints, floats, strings, and missing values for the same fields. We collapse
    those variants here so later code can treat the metadata as an immutable
    tuple of primitives without re-validating every branch at runtime.
    """

    def _ensure_sequence(name: str) -> tuple[object, ...]:
        value = entry.get(name)
        if value is None:
            raise ValueError(f"Shard {shard_id} missing '{name}' metadata")
        if isinstance(value, list):
            return tuple(value)
        if isinstance(value, tuple):
            return value
        raise ValueError(
            f"Shard {shard_id} invalid '{name}' metadata type: {type(value).__name__}"
        )

    column_encodings = tuple(str(v) for v in _ensure_sequence("column_encodings"))
    column_names = tuple(str(v) for v in _ensure_sequence("column_names"))
    column_sizes_obj = entry.get("column_sizes")
    if column_sizes_obj is None:
        column_sizes: tuple[int | None, ...] = tuple(None for _ in column_names)
    elif isinstance(column_sizes_obj, (list, tuple)):
        column_sizes_list: list[int | None] = []
        for value in column_sizes_obj:
            if value is None:
                column_sizes_list.append(None)
            elif isinstance(value, int):
                column_sizes_list.append(value)
            elif isinstance(value, str):
                try:
                    column_sizes_list.append(int(value))
                except ValueError as exc:
                    raise ValueError(
                        f"Shard {shard_id} invalid column size value: {value}"
                    ) from exc
            else:
                raise ValueError(
                    f"Shard {shard_id} invalid column size type: {type(value).__name__}"
                )
        column_sizes = tuple(column_sizes_list)
    else:
        raise ValueError(
            f"Shard {shard_id} invalid 'column_sizes' metadata type: {type(column_sizes_obj).__name__}"
        )

    compression_obj = entry.get("compression")
    compression_str: str | None
    if compression_obj is None:
        compression_str = None
    elif isinstance(compression_obj, str):
        compression_str = compression_obj
    else:
        compression_str = str(compression_obj)

    hashes_obj = entry.get("hashes")
    if isinstance(hashes_obj, (list, tuple)):
        hash_list = tuple(str(item) for item in hashes_obj)
    elif isinstance(hashes_obj, Mapping):
        hash_list = tuple(str(key) for key in hashes_obj)
    elif hashes_obj is None:
        hash_list = tuple()
    else:
        hash_list = (str(hashes_obj),)

    samples_obj = entry.get("samples")
    if samples_obj is None:
        raise ValueError(f"Shard {shard_id} missing sample count metadata")
    if isinstance(samples_obj, int):
        samples = samples_obj
    elif isinstance(samples_obj, str):
        try:
            samples = int(samples_obj)
        except ValueError as exc:
            raise ValueError(
                f"Shard {shard_id} invalid sample count: {samples_obj}"
            ) from exc
    else:
        raise ValueError(f"Shard {shard_id} invalid sample count: {samples_obj}")

    size_limit_obj = entry.get("size_limit")
    if isinstance(size_limit_obj, (int, str)) or size_limit_obj is None:
        size_limit_value: int | str | None = size_limit_obj
    else:
        raise ValueError(
            f"Shard {shard_id} invalid size limit type: {type(size_limit_obj).__name__}"
        )

    format_obj = entry.get("format") or "mds"
    format_value = str(format_obj)
    version_obj = entry.get("version") or 2
    if isinstance(version_obj, int):
        version_value = version_obj
    elif isinstance(version_obj, str):
        try:
            version_value = int(version_obj)
        except ValueError as exc:
            raise ValueError(
                f"Shard {shard_id} invalid version value: {version_obj}"
            ) from exc
    else:
        raise ValueError(
            f"Shard {shard_id} invalid version value type: {type(version_obj).__name__}"
        )

    return _StreamingTemplate(
        column_encodings=column_encodings,
        column_names=column_names,
        column_sizes=column_sizes,
        compression=compression_str,
        hashes=hash_list,
        samples=samples,
        size_limit=size_limit_value,
        format=format_value,
        version=version_value,
    )


def _finalize_streaming_entry(
    template: _StreamingTemplate,
    locator: ShardLocator,
    local_ref: LocalShardRef,
) -> dict[str, object]:
    """Merge the cached streaming template with local shard paths and hashes."""
    entry = {
        "column_encodings": [str(v) for v in template["column_encodings"]],
        "column_names": [str(v) for v in template["column_names"]],
        "column_sizes": list(template["column_sizes"]),
        "hashes": [str(v) for v in template["hashes"]],
        "samples": template["samples"],
        "size_limit": template["size_limit"],
        "format": template["format"],
        "version": template["version"],
    }

    compression = template["compression"]
    if local_ref.compression:
        compression = local_ref.compression
    elif locator.compression:
        compression = locator.compression
    entry["compression"] = str(compression) if compression else None

    raw_hashes = {str(k): str(v) for k, v in locator.raw.hashes.items()}
    raw_basename = os.path.basename(str(local_ref.raw.path))
    entry["raw_data"] = {
        "basename": raw_basename,
        "bytes": int(locator.raw.bytes),
        "hashes": raw_hashes,
    }
    if locator.zip is not None and local_ref.zip is not None:
        zip_hashes = {str(k): str(v) for k, v in locator.zip.hashes.items()}
        zip_basename = os.path.basename(str(local_ref.zip.path))
        entry["zip_data"] = {
            "basename": zip_basename,
            "bytes": int(locator.zip.bytes),
            "hashes": zip_hashes,
        }
    else:
        entry["zip_data"] = None
    return entry


register_format(MDSFormat())

__all__ = ["MDSFormat"]
