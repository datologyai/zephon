"""Support utilities for Zephon's LitData format integration."""

from __future__ import annotations

import copy
import functools
import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from io import BytesIO, FileIO
from typing import Any, Mapping, NamedTuple, Optional

import numpy as np
import optree
from optree import treespec

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

from litdata.constants import _NUMPY_DTYPES_MAPPING, _TORCH_DTYPES_MAPPING
from litdata.streaming.serializers import (
    _SERIALIZERS as LITDATA_SERIALIZERS,
)
from litdata.streaming.serializers import (
    NoHeaderNumpySerializer,
    NoHeaderTensorSerializer,
    PILSerializer,
    Serializer,
)

logger = logging.getLogger("zephon.litdata")

# -----------------------------------------------------------------------------
# Serializers
# -----------------------------------------------------------------------------

_SERIALIZERS: OrderedDict[str, Serializer] = OrderedDict(LITDATA_SERIALIZERS)

_NUMPY_DTYPES_REVERSE: dict[np.dtype, int] = {
    dtype: idx for idx, dtype in _NUMPY_DTYPES_MAPPING.items()
}


@dataclass(slots=True)
class FlatPyTree:
    """Lightweight wrapper around flattened pytree leaves with lazy reconstruction."""

    leaves: list[Any]
    spec: optree.PyTreeSpec

    def materialize(self) -> Any:
        """Reconstruct the original pytree structure."""
        return optree.tree_unflatten(self.spec, self.leaves)

    # Alias for ergonomics
    to_tree = materialize


_CONFIGURED_SERIALIZER_CACHE: dict[str, Serializer] = {}
_CONFIGURED_SERIALIZER_LOCK = threading.Lock()


def _clone_serializer(serializer: Serializer) -> Serializer:
    """Return a lightweight clone of ``serializer`` without deep-copy overhead."""
    try:
        return copy.copy(serializer)
    except Exception:
        cls = serializer.__class__
        try:
            return cls()  # type: ignore[call-arg]
        except Exception:
            return copy.deepcopy(serializer)


def _get_serializers(
    overrides: Optional[Mapping[str, Serializer]] = None,
) -> dict[str, Serializer]:
    """Return serializer instances, allowing overrides for testing."""
    serializers: OrderedDict[str, Serializer] = OrderedDict(_SERIALIZERS)
    if overrides:
        for key, value in overrides.items():
            serializers[key] = value
    return serializers


def _configured_serializer_for_format(
    fmt: str, base_key: str, serializer: Serializer, shareable: bool
) -> Serializer:
    """Return a serializer ready for ``fmt``, caching globally when safe."""
    if fmt == base_key:
        return serializer

    if shareable:
        with _CONFIGURED_SERIALIZER_LOCK:
            cached = _CONFIGURED_SERIALIZER_CACHE.get(fmt)
        if cached is not None:
            return cached

    configured = _clone_serializer(serializer)
    configured.setup(fmt)

    if shareable:
        with _CONFIGURED_SERIALIZER_LOCK:
            existing = _CONFIGURED_SERIALIZER_CACHE.setdefault(fmt, configured)
        return existing
    return configured


# -----------------------------------------------------------------------------
# Item loaders
# -----------------------------------------------------------------------------


class Interval(NamedTuple):
    """Represents a half-open interval [chunk_start, chunk_end) for a chunk."""

    chunk_start: int
    roi_start_idx: int
    roi_end_idx: int
    chunk_end: int


class BaseItemLoader(ABC):
    """Base class for loaders that expose LitData chunks as random-access shards."""

    def setup(
        self,
        config: Mapping[str, Any],
        chunks: list[Mapping[str, Any]],
        serializers: Mapping[str, Serializer],
        region_of_interest: Optional[list[tuple[int, int]]] = None,
    ) -> None:
        self._config = dict(config)
        self._chunks = [dict(chunk) for chunk in chunks]
        self._serializers = dict(serializers)
        self._data_format = list(self._config["data_format"])
        self._shift_idx = len(self._data_format) * 4
        self.region_of_interest = region_of_interest

        for fmt in self._data_format:
            if fmt in self._serializers:
                continue
            key = self._data_format_to_key(fmt)
            base_serializer = self._serializers[key]
            shareable = key in _SERIALIZERS and base_serializer is _SERIALIZERS[key]
            configured = _configured_serializer_for_format(
                fmt, key, base_serializer, shareable
            )
            self._serializers[fmt] = configured

    @functools.lru_cache(maxsize=128)
    def _data_format_to_key(self, data_format: str) -> str:
        if ":" in data_format:
            serializer, subtype = data_format.split(":")
            if subtype in self._serializers:
                return subtype
            return serializer
        return data_format

    def state_dict(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def generate_intervals(self) -> list[Interval]: ...

    @abstractmethod
    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
    ) -> Any: ...

    def load_item_from_bytes(
        self, raw_bytes: bytes, chunk_index: int
    ) -> Any:  # pragma: no cover - rarely used
        raise NotImplementedError

    @abstractmethod
    def delete(self, chunk_index: int, chunk_filepath: str) -> None: ...

    @classmethod
    @abstractmethod
    def encode_data(
        cls, data: list[bytes], sizes: list[int], flattened: list[Any]
    ) -> tuple[bytes, Optional[int]]: ...


class PyTreeLoader(BaseItemLoader):
    """Loader that reconstructs arbitrary pytrees from LitData payloads."""

    def __init__(self, *, return_flat_leaves: bool = False) -> None:
        super().__init__()
        self._chunk_filepath: str | None = None
        self._open_handle: FileIO | None = None
        self._return_flat_leaves = return_flat_leaves
        self._tree_spec: optree.PyTreeSpec | None = None
        self._unflatten: Optional[functools.partial] = None

    def setup(
        self,
        config: Mapping[str, Any],
        chunks: list[Mapping[str, Any]],
        serializers: Mapping[str, Serializer],
        region_of_interest: Optional[list[tuple[int, int]]] = None,
    ) -> None:
        super().setup(config, chunks, serializers, region_of_interest)
        flag = self._config.get("return_flat_leaves")
        if isinstance(flag, bool):
            self._return_flat_leaves = flag
        spec = self._config.get("data_spec")
        if isinstance(spec, optree.PyTreeSpec):
            self._tree_spec = spec
            self._unflatten = functools.partial(optree.tree_unflatten, spec)
        else:
            self._tree_spec = None
            self._unflatten = None

    def generate_intervals(self) -> list[Interval]:
        intervals: list[Interval] = []
        begin = 0
        end = 0
        for idx, chunk in enumerate(self._chunks):
            chunk_size = int(chunk["chunk_size"])
            end += chunk_size
            start_idx = begin
            end_idx = end
            if self.region_of_interest is not None:
                roi = self.region_of_interest[idx]
                start_idx = begin + roi[0]
                end_idx = begin + roi[1]
            intervals.append(Interval(begin, start_idx, end_idx, end))
            begin += chunk_size
        return intervals

    def _load_data(self, fp: FileIO | BytesIO, offset: int) -> bytes:
        fp.seek(offset)
        pair = fp.read(8)
        begin, end = np.frombuffer(pair, np.uint32)
        fp.seek(begin)
        return fp.read(int(end - begin))

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
    ) -> Any:
        offset = (1 + (index - begin) if index >= begin else index + 1) * 4

        if chunk_filepath != self._chunk_filepath:
            if (
                not os.path.exists(chunk_filepath)
                or os.stat(chunk_filepath).st_size < filesize_bytes
            ):
                raise FileNotFoundError(
                    f"Chunk file not found or incomplete: {chunk_filepath}"
                )
            self._chunk_filepath = chunk_filepath
            if self._open_handle is not None:
                self._open_handle.close()
            self._open_handle = open(chunk_filepath, "rb", 0)  # noqa: SIM115

        assert self._open_handle is not None
        data = self._load_data(self._open_handle, offset)
        return self.deserialize(data, chunk_index)

    def deserialize(self, raw_item_data: bytes, chunk_index: int) -> Any:
        idx = self._shift_idx
        sizes = np.frombuffer(raw_item_data[:idx], np.uint32)
        data = []
        for size, data_format in zip(sizes, self._data_format, strict=True):
            serializer = self._serializers[data_format]
            data_bytes = raw_item_data[idx : idx + int(size)]
            data.append(serializer.deserialize(data_bytes))
            idx += int(size)
        if self._return_flat_leaves:
            assert self._tree_spec is not None
            return FlatPyTree(list(data), self._tree_spec)
        if self._unflatten is not None:
            return self._unflatten(data)
        return optree.tree_unflatten(self._config["data_spec"], data)

    def close(self, chunk_index: int) -> None:
        if self._open_handle is not None:
            self._open_handle.close()
            self._open_handle = None

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        if os.path.exists(chunk_filepath):
            os.remove(chunk_filepath)

    @classmethod
    def encode_data(
        cls, data: list[bytes], sizes: list[int], flattened: list[Any]
    ) -> tuple[bytes, Optional[int]]:
        head = np.array(sizes, np.uint32).tobytes()
        body = b"".join(data)
        return head + body, None


class TokensLoader(BaseItemLoader):  # pragma: no cover - requires torch tensors
    """Loader specialised for token-block shards produced by LitData."""

    def __init__(self, block_size: int | None = None) -> None:
        if torch is None:
            raise ImportError("Torch is required for the tokens loader")
        super().__init__()
        self._block_size = block_size
        self._mmaps: dict[int, np.memmap] = {}
        self._buffers: dict[int, memoryview] = {}
        self._counter = defaultdict(int)
        self._dtype: Any = None
        self._chunk_filepaths: dict[str, bool] = {}
        self._offsets: dict[int, np.ndarray] = {}
        self._header_bytes: dict[int, int] = {}
        self._blocks_per_item: dict[int, np.ndarray] = {}
        self._elem_size: Optional[int] = None

    def setup(
        self,
        config: Mapping[str, Any],
        chunks: list[Mapping[str, Any]],
        serializers: Mapping[str, Serializer],
        region_of_interest: Optional[list[tuple[int, int]]] = None,
    ) -> None:
        if torch is None:
            raise ImportError("Torch is required for the tokens loader")
        super().setup(config, chunks, serializers, region_of_interest)
        self._shift_idx = 0

        serializer_name, dtype_index = self._data_format[0].split(":")
        if serializer_name not in ["no_header_numpy", "no_header_tensor"]:
            raise ValueError("Unsupported data format for tokens loader")

        if serializer_name == "no_header_tensor":
            self._dtype = _TORCH_DTYPES_MAPPING[int(dtype_index)]
            self._elem_size = int(torch.empty((), dtype=self._dtype).element_size())
        else:
            self._dtype = _NUMPY_DTYPES_MAPPING[int(dtype_index)]
            self._elem_size = int(np.dtype(self._dtype).itemsize)  # type: ignore[arg-type]

        elem_size = self._elem_size
        assert elem_size is not None
        for chunk in self._chunks:
            if chunk.get("dim") is None:
                chunk_bytes = chunk.get("chunk_bytes")
                chunk_size = chunk.get("chunk_size")
                if isinstance(chunk_bytes, int) and isinstance(chunk_size, int):
                    header_bytes = (1 + chunk_size + 1) * 4
                    payload = max(chunk_bytes - header_bytes, 0)
                    payload = max(payload - (chunk_size * self._shift_idx), 0)
                    chunk["dim"] = payload // elem_size

        if all(chunk.get("dim") is None for chunk in self._chunks):
            raise ValueError("Tokens loader requires chunk 'dim' metadata")

    def generate_intervals(self) -> list[Interval]:
        assert self._block_size is not None
        intervals = []
        begin = 0
        end = 0
        for idx, chunk in enumerate(self._chunks):
            dim = int(chunk["dim"])
            num_blocks = dim // self._block_size
            end += num_blocks
            start_idx, end_idx = begin, end
            if self.region_of_interest is not None:
                roi = self.region_of_interest[idx]
                start_idx = begin + roi[0]
                end_idx = begin + roi[1]
            intervals.append(Interval(begin, start_idx, end_idx, end))
            begin += num_blocks
        return intervals

    def _load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        self._counter[chunk_index] += 1
        if chunk_index in self._mmaps:
            return
        chunk = self._chunks[chunk_index]
        header_bytes = (1 + int(chunk["chunk_size"]) + 1) * 4
        mmap = np.memmap(chunk_filepath, mode="r", order="C", offset=header_bytes)
        self._mmaps[chunk_index] = mmap
        self._buffers[chunk_index] = memoryview(mmap)  # type: ignore
        self._header_bytes[chunk_index] = header_bytes
        offsets = np.memmap(
            chunk_filepath,
            mode="r",
            dtype=np.uint32,
            order="C",
            offset=4,
            shape=(int(chunk["chunk_size"]) + 1,),
        )
        self._offsets[chunk_index] = np.array(offsets, copy=True)
        elem_size = self._elem_size
        assert elem_size is not None and self._block_size is not None
        shift_idx = self._shift_idx
        blocks = []
        for i in range(int(chunk["chunk_size"])):
            item_total = int(
                self._offsets[chunk_index][i + 1] - self._offsets[chunk_index][i]
            )
            payload = max(item_total - shift_idx, 0)
            tokens = payload // elem_size
            blocks.append(tokens // self._block_size)
        self._blocks_per_item[chunk_index] = np.array(blocks, dtype=np.int64)

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
    ) -> Any:
        assert self._block_size is not None
        if chunk_filepath in self._chunk_filepaths and not os.path.isfile(
            chunk_filepath
        ):
            del self._chunk_filepaths[chunk_filepath]
        if chunk_filepath not in self._chunk_filepaths:
            if (
                not os.path.exists(chunk_filepath)
                or os.stat(chunk_filepath).st_size < filesize_bytes
            ):
                raise FileNotFoundError(
                    f"Chunk file not found or incomplete: {chunk_filepath}"
                )
            self._chunk_filepaths[chunk_filepath] = True
        self._load_chunk(chunk_index, chunk_filepath)

        buffer_view = self._buffers[chunk_index]
        buffer = buffer_view.tobytes()
        block_idx_in_chunk = index - begin
        blocks_per_item = self._blocks_per_item[chunk_index]
        cumsum = np.cumsum(blocks_per_item)
        item_idx = int(np.searchsorted(cumsum, block_idx_in_chunk, side="right"))
        prev = int(cumsum[item_idx - 1]) if item_idx > 0 else 0
        within_item_block = int(block_idx_in_chunk - prev)
        elem_size = self._elem_size
        assert elem_size is not None and self._block_size is not None
        start_abs = (
            int(self._offsets[chunk_index][item_idx])
            + self._shift_idx
            + within_item_block * self._block_size * elem_size
        )
        rel_offset = start_abs - self._header_bytes[chunk_index]

        if torch is not None and self._dtype in _TORCH_DTYPES_MAPPING.values():
            return torch.frombuffer(
                buffer, dtype=self._dtype, count=self._block_size, offset=rel_offset
            )
        return np.frombuffer(
            buffer, dtype=self._dtype, count=self._block_size, offset=rel_offset
        )

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        if os.path.exists(chunk_filepath):
            if chunk_index in self._buffers:
                del self._buffers[chunk_index]
            mm = self._mmaps.pop(chunk_index, None)
            if mm is not None:
                raw_mmap = getattr(mm, "_mmap", None)
                if raw_mmap is not None and hasattr(raw_mmap, "close"):
                    raw_mmap.close()
                del self._counter[chunk_index]
            os.remove(chunk_filepath)

    def close(self, chunk_index: int) -> None:
        self._counter[chunk_index] -= 1
        if self._counter[chunk_index] == 0:
            if chunk_index in self._buffers:
                del self._buffers[chunk_index]
            mm = self._mmaps.pop(chunk_index, None)
            if mm is not None:
                raw_mmap = getattr(mm, "_mmap", None)
                if raw_mmap is not None and hasattr(raw_mmap, "close"):
                    raw_mmap.close()

    @classmethod
    def encode_data(
        cls, data: list[bytes], _: list[int], flattened: list[Any]
    ) -> tuple[bytes, Optional[int]]:
        array = flattened[0]
        if hasattr(array, "shape"):
            dim = int(array.shape[0])
        else:
            dim = len(array)
        return data[0], dim


# -----------------------------------------------------------------------------
# Treespec conversions
# -----------------------------------------------------------------------------


def treespec_loads(serialized: str) -> optree.PyTreeSpec:
    """Deserialize a PyTreeSpec from the legacy LitData JSON representation."""
    _protocol, json_schema = json.loads(serialized)
    return _convert_legacy_treespec(json_schema)


def _convert_legacy_treespec(schema: Mapping[str, Any]) -> optree.PyTreeSpec:
    if (
        schema.get("type") is None
        and schema.get("context") is None
        and len(schema.get("children_spec", [])) == 0
    ):
        return treespec.leaf()

    children = [
        _convert_legacy_treespec(child) for child in schema.get("children_spec", [])
    ]

    type_name = schema.get("type")
    context = json.loads(schema["context"]) if schema.get("context") else None

    if type_name == "builtins.dict":
        keys = context if isinstance(context, list) else []
        pairs = [(key, child) for key, child in zip(keys, children)]
        return treespec.ordereddict(pairs)
    if type_name == "builtins.list":
        return treespec.list(children)
    if type_name == "builtins.tuple":
        return treespec.tuple(children)
    if type_name == "collections.OrderedDict":
        keys = context if isinstance(context, list) else []
        pairs = [(key, child) for key, child in zip(keys, children)]
        return treespec.ordereddict(pairs)
    return treespec.list(children)


def treespec_dumps(spec: optree.PyTreeSpec) -> str:
    """Serialize an optree PyTreeSpec into the legacy LitData JSON representation."""

    def _encode(node: optree.PyTreeSpec) -> dict[str, Any]:
        if node.is_leaf():
            return {"type": None, "context": None, "children_spec": []}

        children = [_encode(child) for child in node.children()]
        kind: optree.PyTreeKind = node.kind  # type: ignore[assignment]
        context: Any = None

        if kind == optree.PyTreeKind.TUPLE:
            type_name = "builtins.tuple"
        elif kind == optree.PyTreeKind.LIST:
            type_name = "builtins.list"
        elif kind == optree.PyTreeKind.DICT:
            type_name = "builtins.dict"
            context = list(node.entries())
        elif kind == optree.PyTreeKind.ORDEREDDICT:
            type_name = "collections.OrderedDict"
            context = list(node.entries())
        else:
            type_name = "builtins.list"

        return {
            "type": type_name,
            "context": json.dumps(context) if context is not None else None,
            "children_spec": children,
        }

    schema = _encode(spec)
    return json.dumps([0, schema])


__all__ = [
    "_get_serializers",
    "BaseItemLoader",
    "FlatPyTree",
    "Interval",
    "NoHeaderNumpySerializer",
    "NoHeaderTensorSerializer",
    "PILSerializer",
    "PyTreeLoader",
    "TokensLoader",
    "treespec_dumps",
    "treespec_loads",
]
