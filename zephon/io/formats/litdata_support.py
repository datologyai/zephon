"""Support utilities for Zephon's LitData format integration."""

from __future__ import annotations

import functools
import json
import logging
import os
import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from copy import deepcopy
from io import BytesIO, FileIO
from typing import Any, Mapping, NamedTuple, Optional

import numpy as np
import optree
from optree import treespec

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

logger = logging.getLogger("zephon.litdata")

# -----------------------------------------------------------------------------
# Serializers
# -----------------------------------------------------------------------------

_TORCH_DTYPES_MAPPING: dict[int, Any] = {}
if torch is not None:  # pragma: no branch - evaluated at import
    _TORCH_DTYPES_MAPPING = {
        0: torch.float32,
        1: torch.float,
        2: torch.float64,
        3: torch.double,
        4: torch.complex64,
        5: torch.cfloat,
        6: torch.complex128,
        7: torch.cdouble,
        8: torch.float16,
        9: torch.half,
        10: torch.bfloat16,
        11: torch.uint8,
        12: torch.int8,
        13: torch.int16,
        14: torch.short,
        15: torch.int32,
        16: torch.int,
        17: torch.int64,
        18: torch.long,
        19: torch.bool,
        20: torch.uint16,
    }

_NUMPY_SCTYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    bool,
    object,
    bytes,
    str,
    np.void,
]

_NUMPY_DTYPES_MAPPING: dict[int, np.dtype] = {
    idx: np.dtype(value) for idx, value in enumerate(_NUMPY_SCTYPES)
}
_NUMPY_DTYPES_REVERSE: dict[np.dtype, int] = {
    dtype: idx for idx, dtype in _NUMPY_DTYPES_MAPPING.items()
}


class Serializer(ABC):
    """Interface for object serializers."""

    @abstractmethod
    def serialize(self, data: Any) -> tuple[bytes, Optional[str]]: ...

    @abstractmethod
    def deserialize(self, data: bytes) -> Any: ...

    @abstractmethod
    def can_serialize(self, data: Any) -> bool: ...

    def setup(self, metadata: Any) -> None:
        return None


class _NumericSerializer:
    def __init__(self, dtype: Any) -> None:
        self.dtype = dtype
        self.size = np.dtype(dtype).itemsize  # type: ignore[arg-type]

    def serialize(self, obj: Any) -> tuple[bytes, Optional[str]]:
        return self.dtype(obj).tobytes(), None  # type: ignore[arg-type]

    def deserialize(self, data: bytes) -> Any:
        return np.frombuffer(data, self.dtype)[0]  # type: ignore[arg-type]


class StringSerializer(Serializer):
    def serialize(self, obj: str) -> tuple[bytes, Optional[str]]:
        return obj.encode("utf-8"), None

    def deserialize(self, data: bytes) -> str:
        return data.decode("utf-8")

    def can_serialize(self, data: Any) -> bool:
        return isinstance(data, str) and not os.path.isfile(data)


class BooleanSerializer(Serializer):
    def serialize(self, item: bool) -> tuple[bytes, Optional[str]]:
        return np.uint8(item).tobytes(), None

    def deserialize(self, data: bytes) -> bool:
        return bool(np.frombuffer(data, np.uint8)[0])

    def can_serialize(self, data: Any) -> bool:
        return isinstance(data, bool)


class IntegerSerializer(_NumericSerializer, Serializer):
    def __init__(self) -> None:
        super().__init__(np.int64)

    def can_serialize(self, data: Any) -> bool:
        return isinstance(data, int)


class FloatSerializer(_NumericSerializer, Serializer):
    def __init__(self) -> None:
        super().__init__(np.float64)

    def can_serialize(self, data: Any) -> bool:
        return isinstance(data, float)


class BytesSerializer(Serializer):
    def serialize(self, item: bytes) -> tuple[bytes, Optional[str]]:
        return item, None

    def deserialize(self, data: bytes) -> bytes:
        return data

    def can_serialize(self, item: Any) -> bool:
        return isinstance(item, bytes)


class NumpySerializer(Serializer):
    def __init__(self) -> None:
        self._dtype_to_index = {v: k for k, v in _NUMPY_DTYPES_MAPPING.items()}

    def serialize(self, item: np.ndarray) -> tuple[bytes, Optional[str]]:
        dtype_index = self._dtype_to_index[item.dtype]
        parts = [np.uint32(dtype_index).tobytes(), np.uint32(len(item.shape)).tobytes()]
        for dim in item.shape:
            parts.append(np.uint32(dim).tobytes())
        parts.append(item.tobytes(order="C"))
        return b"".join(parts), None

    def deserialize(self, data: bytes) -> np.ndarray:
        dtype_index = np.frombuffer(data[0:4], np.uint32).item()
        dtype = _NUMPY_DTYPES_MAPPING[dtype_index]
        shape_len = np.frombuffer(data[4:8], np.uint32).item()
        shape = []
        for idx in range(shape_len):
            shape.append(
                np.frombuffer(data[8 + 4 * idx : 8 + 4 * (idx + 1)], np.uint32).item()
            )
        tensor = np.frombuffer(data[8 + 4 * shape_len :], dtype=dtype)
        if tuple(shape) == tensor.shape:
            return tensor
        return np.reshape(tensor, shape)

    def can_serialize(self, item: Any) -> bool:
        return isinstance(item, np.ndarray)


class NoHeaderNumpySerializer(Serializer):
    """Serializer for numpy arrays stored without a header in LitData payloads."""

    def __init__(self) -> None:
        self._dtype_to_indices = {v: k for k, v in _NUMPY_DTYPES_MAPPING.items()}
        self._dtype: np.dtype | None = None

    def setup(self, metadata: Any) -> None:
        if isinstance(metadata, str):
            _, _, suffix = metadata.partition(":")
            if suffix:
                try:
                    index = int(suffix)
                except ValueError as exc:  # pragma: no cover - defensive
                    raise ValueError(
                        f"Invalid dtype index for no_header_numpy: {suffix}"
                    ) from exc
                dtype = _NUMPY_DTYPES_MAPPING.get(index)
                if dtype is None:
                    raise ValueError(f"Unsupported numpy dtype index: {index}")
                self._dtype = dtype
        elif isinstance(metadata, np.dtype):
            self._dtype = metadata

    def serialize(self, item: Any) -> tuple[bytes, Optional[str]]:
        array = np.asarray(item)
        if self._dtype is None:
            dtype_index = self._dtype_to_indices.get(array.dtype)
            if dtype_index is None:
                raise ValueError(
                    f"Unsupported numpy dtype for serialization: {array.dtype}"
                )
            self._dtype = array.dtype
        else:
            dtype_index = self._dtype_to_indices[self._dtype]
            if array.dtype != self._dtype:
                array = array.astype(self._dtype, copy=False)
        return array.tobytes(order="C"), f"no_header_numpy:{dtype_index}"

    def deserialize(self, data: bytes) -> np.ndarray:
        if self._dtype is None:
            raise RuntimeError(
                "No dtype configured for no_header_numpy deserialization"
            )
        return np.frombuffer(data, dtype=self._dtype)

    def can_serialize(self, item: Any) -> bool:
        return isinstance(item, np.ndarray) and len(item.shape) == 1


class PickleSerializer(Serializer):
    def serialize(self, item: Any) -> tuple[bytes, Optional[str]]:
        return pickle.dumps(item), None

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)  # noqa: S301

    def can_serialize(self, _: Any) -> bool:
        return True


class TensorSerializer(Serializer):  # pragma: no cover - torch dependent
    def __init__(self) -> None:
        if torch is None:
            raise ImportError("Torch is required for tensor serialization")
        self._dtype_to_indices = {v: k for k, v in _TORCH_DTYPES_MAPPING.items()}

    def serialize(self, item: Any) -> tuple[bytes, Optional[str]]:
        if torch is None:
            raise ImportError("Torch is required for tensor serialization")
        dtype_index = self._dtype_to_indices[item.dtype]
        data = [np.uint32(dtype_index).tobytes()]
        data.append(np.uint32(item.dim()).tobytes())
        data.extend(np.uint32(dim).tobytes() for dim in item.shape)
        data.append(item.cpu().numpy().tobytes(order="C"))
        return b"".join(data), None

    def deserialize(self, data: bytes) -> Any:
        if torch is None:
            raise ImportError("Torch is required for tensor deserialization")
        dtype_index = np.frombuffer(data[0:4], np.uint32).item()
        dtype = _TORCH_DTYPES_MAPPING[dtype_index]
        shape_len = np.frombuffer(data[4:8], np.uint32).item()
        shape = []
        for idx in range(shape_len):
            shape.append(
                np.frombuffer(data[8 + 4 * idx : 8 + 4 * (idx + 1)], np.uint32).item()
            )
        array = np.frombuffer(data[8 + 4 * shape_len :], dtype=np.float32)
        tensor = torch.from_numpy(array).to(dtype=dtype)
        return tensor.reshape(shape)

    def can_serialize(self, item: Any) -> bool:
        if torch is None:
            return False
        return isinstance(item, torch.Tensor)


class NoHeaderTensorSerializer(Serializer):  # pragma: no cover - torch dependent
    """Serializer for tensors stored without a header in LitData payloads."""

    def __init__(self) -> None:
        if torch is None:
            raise ImportError("Torch is required for tensor serialization")
        self._dtype_to_indices = {v: k for k, v in _TORCH_DTYPES_MAPPING.items()}
        self._dtype: Any = None

    def setup(self, metadata: Any) -> None:
        if torch is None:
            raise ImportError("Torch is required for tensor serialization")
        if isinstance(metadata, str):
            _, _, suffix = metadata.partition(":")
            if suffix:
                try:
                    index = int(suffix)
                except ValueError as exc:  # pragma: no cover - defensive
                    raise ValueError(
                        f"Invalid dtype index for no_header_tensor: {suffix}"
                    ) from exc
                dtype = _TORCH_DTYPES_MAPPING.get(index)
                if dtype is None:
                    raise ValueError(f"Unsupported tensor dtype index: {index}")
                self._dtype = dtype

    def serialize(self, item: Any) -> tuple[bytes, Optional[str]]:
        if torch is None:
            raise ImportError("Torch is required for tensor serialization")
        dtype_index = self._dtype_to_indices[item.dtype]
        self._dtype = item.dtype
        return item.numpy().tobytes(order="C"), f"no_header_tensor:{dtype_index}"

    def deserialize(self, data: bytes) -> Any:
        if torch is None:
            raise ImportError("Torch is required for tensor deserialization")
        if self._dtype is None:
            raise RuntimeError(
                "No dtype configured for no_header_tensor deserialization"
            )
        if len(data) == 0:
            return torch.empty((0,), dtype=self._dtype)
        return torch.frombuffer(bytearray(data), dtype=self._dtype)

    def can_serialize(self, item: Any) -> bool:
        if torch is None:
            return False
        return isinstance(item, torch.Tensor) and len(item.shape) == 1


class PILSerializer(Serializer):
    """Serializer for PIL/Pillow images - compatible with LitData format."""

    def serialize(self, item: Any) -> tuple[bytes, Optional[str]]:
        import numpy as np
        from PIL import Image

        if not isinstance(item, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(item)}")

        mode = item.mode.encode("utf-8")
        width, height = item.size
        raw = item.tobytes()
        header = np.array([width, height, len(mode)], np.uint32)
        return header.tobytes() + mode + raw, None

    def deserialize(self, data: bytes) -> Any:
        import numpy as np
        from PIL import Image

        idx = 3 * 4  # 3 uint32 values
        width, height, mode_size = np.frombuffer(data[:idx], np.uint32)
        width_i = int(width)
        height_i = int(height)
        mode_len = int(mode_size)
        mode_bytes = data[idx : idx + mode_len]
        raw = data[idx + mode_len :]
        return Image.frombytes(mode_bytes.decode("utf-8"), (width_i, height_i), raw)

    def can_serialize(self, data: Any) -> bool:
        try:
            from PIL import Image

            return isinstance(data, Image.Image)
        except ImportError:
            return False


_SERIALIZERS: OrderedDict[str, Serializer] = OrderedDict(
    [
        ("str", StringSerializer()),
        ("bool", BooleanSerializer()),
        ("int", IntegerSerializer()),
        ("float", FloatSerializer()),
        ("bytes", BytesSerializer()),
        ("numpy", NumpySerializer()),
        ("pickle", PickleSerializer()),
        ("no_header_numpy", NoHeaderNumpySerializer()),
        ("pil", PILSerializer()),
    ]
)

if torch is not None:  # pragma: no branch
    _SERIALIZERS["tensor"] = TensorSerializer()
    _SERIALIZERS["no_header_tensor"] = NoHeaderTensorSerializer()


def _get_serializers(
    overrides: Optional[Mapping[str, Serializer]] = None,
) -> dict[str, Serializer]:
    """Return serializer instances, allowing overrides for testing."""
    serializers = OrderedDict(_SERIALIZERS)
    if overrides:
        for key, value in overrides.items():
            serializers[key] = value
    return {key: deepcopy(value) for key, value in serializers.items()}


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
        self._serializers = {key: deepcopy(value) for key, value in serializers.items()}
        self._data_format = list(self._config["data_format"])
        self._shift_idx = len(self._data_format) * 4
        self.region_of_interest = region_of_interest

        for fmt in self._data_format:
            serializer = deepcopy(self._serializers[self._data_format_to_key(fmt)])
            serializer.setup(fmt)
            self._serializers[fmt] = serializer

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

    def __init__(self) -> None:
        super().__init__()
        self._chunk_filepath: str | None = None
        self._open_handle: FileIO | None = None

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
    "Interval",
    "PyTreeLoader",
    "TokensLoader",
    "treespec_dumps",
    "treespec_loads",
]
