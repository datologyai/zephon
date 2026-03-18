"""LitData shard format integration."""

from __future__ import annotations

import os
import struct
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from zephon.io.formats.base import FormatHandler, register_format
from zephon.io.index import find_and_load_index
from zephon.io.index import find_and_load_index
from zephon.io.protocols import RandomAccessShard
from zephon.io.storage import StorageBackend
from zephon.io.types import LocalShardRef, ShardFile, ShardLocator

# litdata_support requires numpy, optree, and (lazily) litdata+torch.
# Defer the import so that registering the format handler does not pull in
# heavy dependencies — they are only needed when discover/open_shard run.
_litdata_support = None


def _ensure_litdata_support():
    global _litdata_support
    if _litdata_support is None:
        try:
            from zephon.io.formats import litdata_support

            _litdata_support = litdata_support
        except ImportError as exc:
            raise ImportError(
                "LitData format requires numpy, optree, and litdata packages. "
                "Install with: pip install zephon[litdata]"
            ) from exc
    return _litdata_support


if TYPE_CHECKING:
    from zephon.io.dataset import Dataset


class _StreamingTemplateDict(dict[str, Any]):
    """Typed dict-like helper to appease static type checking."""


def _select_item_loader(
    config: Mapping[str, Any], chunks: list[Mapping[str, Any]] | None = None
) -> Any:
    """Choose the appropriate item loader based on the config metadata."""
    support = _ensure_litdata_support()
    loader_spec = config.get("item_loader")
    loader_name: str | None = None
    block_size: int | None = None

    if isinstance(loader_spec, Mapping):
        for key in ("name", "type", "kind", "loader"):
            value = loader_spec.get(key)
            if isinstance(value, str):
                loader_name = value
                break
        block_candidate = loader_spec.get("block_size")
        if isinstance(block_candidate, int):
            block_size = block_candidate
    elif isinstance(loader_spec, str):
        loader_name = loader_spec

    if block_size is None:
        block_candidate = config.get("block_size")
        if isinstance(block_candidate, int):
            block_size = block_candidate

    if loader_name and loader_name.lower() in {"tokens", "tokensloader"}:
        if block_size is None and chunks:
            for chunk in chunks:
                dim_value = chunk.get("dim")
                chunk_size = chunk.get("chunk_size")
                if isinstance(dim_value, (int, float)) and isinstance(
                    chunk_size, (int, float)
                ):
                    chunk_size_int = int(chunk_size)
                    if chunk_size_int > 0:
                        candidate = int(dim_value) // chunk_size_int
                        if candidate > 0:
                            block_size = candidate
                            break
        if block_size is None:
            raise ValueError("LitData tokens loader requires an integer 'block_size'")
        return support.TokensLoader(block_size=block_size)

    flag = config.get("return_flat_leaves")
    return support.PyTreeLoader(
        return_flat_leaves=bool(flag) if isinstance(flag, bool) else False
    )


def _extract_chunk_basename(chunk: Mapping[str, Any], shard_id: int) -> str:
    """Return the file basename for a chunk."""
    for key in (
        "chunk_path",
        "chunk_file",
        "chunk_filename",
        "file_path",
        "filepath",
        "filename",
        "path",
        "file",
    ):
        value = chunk.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"LitData chunk {shard_id} missing file path metadata")


def _extract_chunk_bytes(chunk: Mapping[str, Any], root: str, basename: str) -> int:
    """Return the expected byte size for a chunk."""
    bytes_value = chunk.get("chunk_bytes")
    if isinstance(bytes_value, (int, float)):
        return int(bytes_value)
    file_path = os.path.join(root, basename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"LitData chunk file not found: {file_path}")
    return os.path.getsize(file_path)


def _normalize_hashes(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(k): str(v) for k, v in value.items()}


def _derive_raw_basename(basename: str, compression: str) -> str:
    """Return the expected raw chunk basename for a compressed chunk."""
    if not basename or not compression:
        return basename
    token = f".{compression}"
    prefix, sep, suffix = basename.partition(token)
    if sep:
        return f"{prefix}{suffix}"
    return basename


def _extract_zip_bytes(chunk: Mapping[str, Any], root: str, basename: str) -> int:
    """Return the byte size of the compressed chunk when available."""
    bytes_value = chunk.get("chunk_zip_bytes")
    if isinstance(bytes_value, (int, float)):
        return int(bytes_value)
    file_path = os.path.join(root, basename)
    try:
        return os.path.getsize(file_path)
    except OSError:
        pass
    fallback = chunk.get("chunk_bytes")
    if isinstance(fallback, (int, float)):
        return int(fallback)
    return 0


def _normalize_config(raw: Mapping[str, Any]) -> _StreamingTemplateDict:
    config = _StreamingTemplateDict(raw)
    data_spec = config.get("data_spec")
    if isinstance(data_spec, str):
        config["data_spec"] = _ensure_litdata_support().treespec_loads(data_spec)
    return config


def _normalize_chunk(
    entry: Mapping[str, Any] | object, shard_id: int
) -> Mapping[str, Any]:
    if not isinstance(entry, Mapping):
        raise ValueError(f"LitData chunk metadata {shard_id} must be a mapping")
    chunk = dict(entry)
    size_value = chunk.get("chunk_size")
    if not isinstance(size_value, (int, float)):
        raise ValueError(f"LitData chunk {shard_id} missing 'chunk_size'")
    chunk["chunk_size"] = int(size_value)

    if "column_sizes" not in chunk or not isinstance(chunk["column_sizes"], list):
        chunk["column_sizes"] = []
    return cast(Mapping[str, Any], chunk)


def _read_chunk_metadata_only(
    path: str, basename: str, storage: StorageBackend, size: int
) -> dict[str, Any]:
    """Read LitData chunk metadata via range reads (header only).

    Chunk format: [num_items (4B)] [offset_array (4*(N+1)B)] [item_data]
    Only the header is read; avoids full file download.
    """
    if size < 8:
        raise ValueError(f"Chunk too small to be valid LitData: {path}")

    # Read first 4 bytes for num_items
    num_items_bytes = storage.read_range(path, 0, length=4)
    num_items = struct.unpack("<I", num_items_bytes)[0]

    header_size = 4 + (num_items + 1) * 4
    if header_size > size:
        raise ValueError(f"Chunk header extends past file size for {path}")

    # Read offset array
    offset_bytes = storage.read_range(path, 4, length=(num_items + 1) * 4)
    offsets = np.frombuffer(offset_bytes, dtype=np.uint32)

    # Compute dim (total tokens) for TokensLoader: sum of (item_size - 4) / 4 per item
    shift_idx = 4  # no_header_tensor has 4-byte per-item size header
    elem_size = 4  # uint32
    dim = 0
    for i in range(num_items):
        item_bytes = int(offsets[i + 1] - offsets[i])
        payload = max(item_bytes - shift_idx, 0)
        dim += payload // elem_size

    return {
        "filename": basename,
        "chunk_size": num_items,
        "chunk_bytes": size,
        "dim": dim,
    }


class LitDataFormat(FormatHandler):
    """Format handler for LitData datasets."""

    kind = "litdata"

    def discover(
        self, path: str, storage: StorageBackend
    ) -> tuple[dict[int, int], dict[int, dict[str, Any]]]:
        result = find_and_load_index(path, storage)
        if result is None:
            raise ValueError("Missing LitData index")

        data = result
        if not isinstance(data, Mapping):
            raise ValueError("LitData index must be a mapping")

        raw_config = data.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError("LitData index missing 'config' mapping")
        config = _normalize_config(raw_config)

        raw_chunks = data.get("chunks")
        if not isinstance(raw_chunks, list):
            raise ValueError("LitData index missing 'chunks' list")

        chunks: list[Mapping[str, Any]] = []
        for shard_id, entry in enumerate(raw_chunks):
            chunk = _normalize_chunk(entry, shard_id)
            chunks.append(chunk)

        support = _ensure_litdata_support()
        loader = _select_item_loader(config, chunks)
        serializers = support._get_serializers()
        loader.setup(config, chunks, serializers, None)
        intervals = loader.generate_intervals()
        if len(intervals) != len(chunks):
            raise ValueError("LitData loader returned inconsistent interval counts")

        shard_index: dict[int, int] = {}
        shard_meta: dict[int, dict[str, Any]] = {}
        for shard_id, interval in enumerate(intervals):
            assert isinstance(interval, support.Interval)
            shard_length = int(interval.chunk_end - interval.chunk_start)
            shard_index[shard_id] = shard_length
            shard_meta[shard_id] = {
                "config": config,
                "chunk": chunks[shard_id],
                "interval": interval,
            }

        return shard_index, shard_meta

    def _discover_from_files(
        self, path: str, storage: StorageBackend
    ) -> tuple[dict[int, int], dict[int, dict[str, Any]]]:
        """Fallback: read chunk metadata from each .bin file via range reads.

        Uncompressed .bin files: use range reads (header only, ~40KB per chunk).
        Compressed .bin.zst files: not supported (require full download/decompress);
        if only .bin.zst files exist, raises a clear error.
        """
        all_entries = storage.listdir(path)
        entries = sorted(name for name in all_entries if name.endswith(".bin"))
        zst_entries = [n for n in all_entries if n.endswith(".bin.zst")]

        if not entries:
            if zst_entries:
                raise ValueError(
                    f"Only compressed .bin.zst chunks found under {path}. "
                    "Range-read discovery is not supported for compressed chunks; "
                    "index.json is required for .bin.zst datasets."
                )
            raise ValueError(f"No .bin chunks found under {path}")

        # Default config for token datasets (common case)
        config = _StreamingTemplateDict(
            {
                "data_format": ["no_header_tensor:0"],
                "block_size": 4096,
                "item_loader": {"name": "tokens", "block_size": 4096},
            }
        )

        chunks: list[dict[str, Any]] = []

        def read_metadata(shard_id: int, name: str) -> tuple[int, dict, dict]:
            full_path = os.path.join(path, name)
            stats = storage.stat(full_path)
            size = int(stats.get("size", 0))
            meta = _read_chunk_metadata_only(full_path, name, storage, size)
            chunk = {
                "filename": meta["filename"],
                "chunk_size": meta["chunk_size"],
                "chunk_bytes": meta["chunk_bytes"],
                "dim": meta["dim"],
                "column_sizes": [],
            }
            return shard_id, meta, chunk

        max_workers = min(32, (len(entries) + 4) // 5)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(read_metadata, shard_id, name): shard_id
                for shard_id, name in enumerate(entries)
            }
            results: list[tuple[int, dict, dict]] = []
            for future in as_completed(futures):
                results.append(future.result())

        # Sort by shard_id to preserve order
        results.sort(key=lambda x: x[0])
        chunks = [r[2] for r in results]

        support = _ensure_litdata_support()
        loader = _select_item_loader(config, chunks)
        serializers = support._get_serializers()
        loader.setup(config, chunks, serializers, None)
        intervals = loader.generate_intervals()
        if len(intervals) != len(chunks):
            raise ValueError("LitData loader returned inconsistent interval counts")

        shard_index: dict[int, int] = {}
        shard_meta: dict[int, dict[str, Any]] = {}
        for shard_id, interval in enumerate(intervals):
            assert isinstance(interval, support.Interval)
            shard_length = int(interval.chunk_end - interval.chunk_start)
            shard_index[shard_id] = shard_length
            shard_meta[shard_id] = {
                "config": config,
                "chunk": chunks[shard_id],
                "interval": interval,
            }

        return shard_index, shard_meta

    def build_locators(self, dataset: "Dataset") -> dict[int, ShardLocator]:
        backend = dataset.backend
        path = backend.get("path")
        if not isinstance(path, str):
            raise ValueError("LitData backend missing dataset path")

        shards_meta = backend.get("shards")
        if not isinstance(shards_meta, Mapping):
            raise ValueError("LitData backend missing shard metadata")

        locators: dict[int, ShardLocator] = {}
        for shard_id_obj, meta in shards_meta.items():
            shard_id = int(shard_id_obj)
            if not isinstance(meta, Mapping):
                raise ValueError(f"LitData shard {shard_id} metadata must be mapping")
            config = meta.get("config")
            chunk = meta.get("chunk")
            interval = meta.get("interval")
            if not isinstance(config, Mapping) or not isinstance(chunk, Mapping):
                raise ValueError(
                    f"LitData shard {shard_id} missing config or chunk data"
                )

            basename = _extract_chunk_basename(chunk, shard_id)
            chunk_bytes = _extract_chunk_bytes(chunk, path, basename)
            hashes = _normalize_hashes(chunk.get("hashes"))
            compression: str | None = None
            zip_file: ShardFile | None = None
            raw_basename = basename
            compression_value = config.get("compression")
            if isinstance(compression_value, str) and compression_value:
                compression = compression_value
                raw_basename = _derive_raw_basename(basename, compression)
                zip_bytes = _extract_zip_bytes(chunk, path, basename)
                zip_file = ShardFile(
                    basename=basename,
                    bytes=zip_bytes,
                    hashes=hashes,
                )

            locators[shard_id] = ShardLocator(
                dataset=dataset.name,
                shard_id=shard_id,
                format=self.kind,
                root=path,
                raw=ShardFile(
                    basename=raw_basename,
                    bytes=chunk_bytes,
                    hashes=hashes,
                ),
                zip=zip_file,
                compression=compression,
                extra={
                    "config": config,
                    "chunk": chunk,
                    "chunk_index": shard_id,
                    **(
                        {"interval": interval}
                        if isinstance(interval, _ensure_litdata_support().Interval)
                        else {}
                    ),
                },
            )
        return locators

    def open_shard(
        self, locator: ShardLocator, local_ref: LocalShardRef
    ) -> RandomAccessShard:
        extra = local_ref.extra or locator.extra
        if not isinstance(extra, Mapping):
            raise RuntimeError("LitData shard metadata missing extras")
        config = extra.get("config")
        chunk = extra.get("chunk")
        if not isinstance(config, Mapping) or not isinstance(chunk, Mapping):
            raise RuntimeError("LitData shard extras missing config or chunk metadata")
        interval = extra.get("interval")
        Interval = _ensure_litdata_support().Interval
        cached_interval = interval if isinstance(interval, Interval) else None
        return _LitDataShard(locator.root, config, chunk, local_ref, cached_interval)


class _LitDataShard(RandomAccessShard):
    """Random access view over a single LitData chunk."""

    def __init__(
        self,
        root: str,
        config: Mapping[str, Any],
        chunk: Mapping[str, Any],
        local_ref: LocalShardRef,
        interval: Any | None = None,
    ) -> None:
        support = _ensure_litdata_support()
        self._root = root
        self._raw_path = local_ref.raw.path
        if isinstance(config, _StreamingTemplateDict):
            self._config = config
        else:
            self._config = _normalize_config(config)
        self._chunk = chunk if isinstance(chunk, dict) else dict(chunk)
        self._chunk_bytes = int(self._chunk.get("chunk_bytes", local_ref.raw.bytes))

        loader = _select_item_loader(self._config, [self._chunk])
        serializers = support._get_serializers()
        loader.setup(self._config, [self._chunk], serializers, None)
        if isinstance(interval, support.Interval):
            self._interval = interval
            self._length = int(interval.chunk_end - interval.chunk_start)
        else:
            intervals = loader.generate_intervals()
            if not intervals:
                self._length = 0
                self._interval = support.Interval(0, 0, 0, 0)
            else:
                self._interval = intervals[0]
                self._length = int(
                    self._interval.chunk_end - self._interval.chunk_start
                )

        self._loader = loader

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Any:
        if index < 0 or index >= self._length:
            raise IndexError("LitData shard index out of range")
        absolute_index = self._interval.chunk_start + index
        return self._loader.load_item_from_chunk(
            absolute_index,
            0,
            str(self._raw_path),
            self._interval.chunk_start,
            self._chunk_bytes,
        )

    def getsamples(self, indices: list[int]) -> list[dict[str, object]]:
        if not indices:
            return []
        # Validate indices
        length = len(self)
        for i in indices:
            if i < 0 or i >= length:
                raise IndexError("LitData shard index out of range")
        # Convert relative indices to absolute indices
        absolute_indices = [self._interval.chunk_start + i for i in indices]
        items = self._loader.load_items_from_chunk(
            absolute_indices,
            0,
            str(self._raw_path),
            self._interval.chunk_start,
            self._chunk_bytes,
        )
        return [cast(dict[str, object], item) for item in items]

    def close(self) -> None:
        close = getattr(self._loader, "close", None)
        if callable(close):
            close(0)


register_format(LitDataFormat())

__all__ = ["LitDataFormat"]
