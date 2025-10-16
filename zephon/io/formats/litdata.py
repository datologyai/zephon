"""LitData shard format integration."""

import json
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from zephon.io.formats.base import FormatHandler, register_format
from zephon.io.formats.litdata_support import (
    BaseItemLoader,
    Interval,
    PyTreeLoader,
    TokensLoader,
    _get_serializers,
    treespec_loads,
)
from zephon.io.protocols import RandomAccessShard
from zephon.io.storage import StorageBackend
from zephon.io.types import LocalShardRef, ShardFile, ShardLocator

if TYPE_CHECKING:
    from zephon.io.dataset import Dataset


class _StreamingTemplateDict(dict[str, Any]):
    """Typed dict-like helper to appease static type checking."""


def _select_item_loader(
    config: Mapping[str, Any], chunks: list[Mapping[str, Any]] | None = None
) -> BaseItemLoader:
    """Choose the appropriate item loader based on the config metadata."""
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
        return TokensLoader(block_size=block_size)

    return PyTreeLoader()


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
        config["data_spec"] = treespec_loads(data_spec)
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


class LitDataFormat(FormatHandler):
    """Format handler for LitData datasets."""

    kind = "litdata"

    def discover(
        self, path: str, storage: StorageBackend
    ) -> tuple[dict[int, int], dict[int, dict[str, Any]]]:
        index_path = os.path.join(path, "index.json")
        try:
            with storage.open(index_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError as exc:
            raise ValueError(f"Missing LitData index: {index_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse LitData index: {index_path}") from exc

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

        loader = _select_item_loader(config, chunks)
        serializers = _get_serializers()
        loader.setup(config, chunks, serializers, None)
        intervals = loader.generate_intervals()
        if len(intervals) != len(chunks):
            raise ValueError("LitData loader returned inconsistent interval counts")

        shard_index: dict[int, int] = {}
        shard_meta: dict[int, dict[str, Any]] = {}
        for shard_id, interval in enumerate(intervals):
            assert isinstance(interval, Interval)
            shard_length = int(interval.chunk_end - interval.chunk_start)
            shard_index[shard_id] = shard_length
            shard_meta[shard_id] = {
                "config": config,
                "chunk": chunks[shard_id],
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
        return _LitDataShard(locator.root, config, chunk, local_ref)


class _LitDataShard(RandomAccessShard):
    """Random access view over a single LitData chunk."""

    def __init__(
        self,
        root: str,
        config: Mapping[str, Any],
        chunk: Mapping[str, Any],
        local_ref: LocalShardRef,
    ) -> None:
        self._root = root
        self._raw_path = local_ref.raw.path
        self._config = _normalize_config(config)
        chunk_copy = dict(chunk)
        self._chunk = chunk_copy

        self._chunk_bytes = int(chunk_copy.get("chunk_bytes", local_ref.raw.bytes))

        loader = _select_item_loader(self._config, [chunk_copy])
        serializers = _get_serializers()
        loader.setup(self._config, [chunk_copy], serializers, None)
        intervals = loader.generate_intervals()
        if not intervals:
            self._length = 0
            self._interval = Interval(0, 0, 0, 0)
        else:
            self._interval = intervals[0]
            self._length = int(self._interval.chunk_end - self._interval.chunk_start)

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

    def close(self) -> None:
        close = getattr(self._loader, "close", None)
        if callable(close):
            close(0)


register_format(LitDataFormat())

__all__ = ["LitDataFormat"]
