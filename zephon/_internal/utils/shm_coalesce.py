"""Prepare CPU payloads for process-stage transport with one memory policy.

Type adapters describe values and write them directly into Torch-managed shared
storage. The common policy chooses inline transport, reuse of existing SHM,
view compaction, or bounded coalescing. Copies target the final shared buffers;
Torch owns allocation and lifetime. Unsupported values retain their normal
serialization.

Descriptors keep both inline data and shared references lazy across forwarding
hops. Resolution reconstructs operator payloads; shared bytes remain _ShmBytes
buffer views on Python 3.12+, avoiding a copy into ordinary bytes.
The NumPy reducer forwards only Zephon-owned storage; policy configuration is
scoped to process-stage messages, not installed globally or applied to MTP.
"""

from __future__ import annotations

import dataclasses
import operator
import pickle
import sys
from dataclasses import dataclass
from functools import cache
from multiprocessing.reduction import register
from typing import Any, Literal, cast

import numpy as np
import optree

from zephon._internal.stream import LazyPayload
from zephon._internal.utils.shm import is_shm_error, wait_for_shm_space
from zephon.types import (
    SampleBatch,
    SampleMeta,
    SamplePayload,
    SampleRecord,
    StreamItem,
)

# NumPy moved byte_bounds in 2.0; the supported floor is 1.20.
try:
    from numpy.lib.array_utils import byte_bounds as _byte_bounds
except ImportError:
    from numpy import byte_bounds as _byte_bounds


# ---------------------------------------------------------------------------
# Lazy torch import
# ---------------------------------------------------------------------------
_torch: Any = None
_torch_loaded = False


def _get_torch() -> Any:
    global _torch, _torch_loaded  # noqa: PLW0603
    if not _torch_loaded:
        try:
            import torch as _t

            _torch = _t
        except ImportError:
            pass
        _torch_loaded = True
    return _torch


# Defaults are shared by all supported CPU payload types.
DEFAULT_SHM_MIN_SIZE = 8192
DEFAULT_SHM_MIN_BUFFER_SIZE = 512 * 1024
DEFAULT_SHM_MIN_REUSE_SIZE = 128 * 1024
DEFAULT_SHM_MAX_RETAINED_RATIO = 8.0
DEFAULT_SHM_MIN_RECLAIM_BYTES = 16 * 1024 * 1024
DEFAULT_SHM_COALESCE_MAX_SIZE = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PayloadMemoryPolicy:
    """Choose transport, retention, and coalescing independently of payload type.

    ``shm_min_size`` applies per value, including values already in SHM.
    ``min_buffer_size`` requires enough useful bytes per new allocation;
    ``min_reuse_size`` applies to existing shared allocations. Compaction accounts
    for sibling views in that message; references elsewhere can delay reclamation.
    Compacted views and large private values receive individual allocations;
    they are never coalesced straight back into an oversized slab.
    """

    shm_min_size: int = DEFAULT_SHM_MIN_SIZE
    coalesce: bool = True
    max_retained_ratio: float | None = DEFAULT_SHM_MAX_RETAINED_RATIO
    min_reclaim_bytes: int = DEFAULT_SHM_MIN_RECLAIM_BYTES
    coalesce_max_size: int | None = DEFAULT_SHM_COALESCE_MAX_SIZE
    min_buffer_size: int = DEFAULT_SHM_MIN_BUFFER_SIZE
    min_reuse_size: int = DEFAULT_SHM_MIN_REUSE_SIZE

    def __reduce__(self) -> tuple:
        return type(self), (
            self.shm_min_size,
            self.coalesce,
            self.max_retained_ratio,
            self.min_reclaim_bytes,
            self.coalesce_max_size,
            self.min_buffer_size,
            self.min_reuse_size,
        )

    def __post_init__(self) -> None:
        if (
            min(
                self.shm_min_size,
                self.min_reclaim_bytes,
                self.min_buffer_size,
                self.min_reuse_size,
            )
            < 0
        ):
            raise ValueError("SHM size thresholds must be non-negative")
        if self.max_retained_ratio is not None and not (
            1 <= self.max_retained_ratio < float("inf")
        ):
            raise ValueError("shm_max_retained_ratio must be finite and >= 1, or None")
        if self.coalesce_max_size is not None and self.coalesce_max_size <= 0:
            raise ValueError("shm_coalesce_max_size must be positive, or None")

    def choose(
        self, nbytes: int, shared_nbytes: int | None
    ) -> Literal["inline", "reuse", "separate", "coalesce"]:
        """Decide from useful bytes and backing size, without inspecting data."""
        if nbytes == 0 or nbytes < self.shm_min_size:
            return "inline"
        if shared_nbytes is not None:
            if (
                self.max_retained_ratio is not None
                and shared_nbytes > nbytes * self.max_retained_ratio
                and shared_nbytes - nbytes >= self.min_reclaim_bytes
            ):
                return "separate"
            return "reuse"
        if not self.coalesce or (
            self.coalesce_max_size is not None and nbytes > self.coalesce_max_size
        ):
            return "separate"
        return "coalesce"


# dtype key used for the coalesced raw-bytes buffer
_BYTES_DTYPE_KEY = "_bytes_uint8"

# Prefix for numpy array dtype keys to distinguish from torch tensor keys
_NDARRAY_PREFIX = "np:"


# ---------------------------------------------------------------------------
# Payload descriptors: either one leaf or a flattened tree
# ---------------------------------------------------------------------------
# Private descriptor types distinguish transport state from real user payloads.
# Keep these bases lightweight: leaf classification is on the restoration path.
class _PayloadDescriptor:
    """Description of a payload, separate from its SampleRecord metadata."""

    __slots__ = ()

    def bind(self, buffers: dict[str, Any]) -> LazyPayload:
        """Attach shared buffers without materializing the payload."""
        raise NotImplementedError


class _LeafDescriptor(_PayloadDescriptor):
    """One extracted value, usable as a tree leaf or an entire payload."""

    __slots__ = ()

    def bind(self, buffers: dict[str, Any]) -> LazyPayload:
        """Keep a root leaf lazy without adding a singleton tree around it."""
        key = getattr(self, "buffer_key", None)
        if key is not None:
            return _ShmLeafPayload(self, {key: buffers[key]} if key in buffers else {})
        return _ShmLeafPayload(self, _used_buffers([self], buffers))

    def restore(self, buffers: dict[str, Any]) -> Any:
        """Restore this value using its shared buffers."""
        raise NotImplementedError


@dataclass(slots=True)
class _PreparedPayload(LazyPayload):
    """A payload whose existing reducers already implement the selected policy."""

    value: Any
    _policy: PayloadMemoryPolicy

    def resolve_payload(self) -> SamplePayload:
        return self.value

    def __reduce__(self) -> tuple:
        return type(self), (self.value, self._policy)


@dataclass(slots=True)
class _InlineSlot(_LeafDescriptor):
    """Keep an inline root lazy without a singleton tree specification."""

    value: Any

    def restore(self, buffers: dict[str, Any]) -> Any:
        return self.value

    def __reduce__(self) -> tuple:
        return type(self), (self.value,)


@dataclass(slots=True)
class _RetainedView:
    """Preserve an existing view if optional compaction cannot allocate SHM."""

    value: Any


def _restore_slot(slot: _LeafDescriptor, buffers: dict[str, Any]) -> Any:
    buffer = buffers.get(getattr(slot, "buffer_key", None))
    return buffer.value if isinstance(buffer, _RetainedView) else slot.restore(buffers)


@dataclass(slots=True)
class _TensorSlot(_LeafDescriptor):
    """Lightweight stand-in for a tensor extracted into a coalesced buffer."""

    dtype_key: str  # str(tensor.dtype)
    offset: int  # element offset into the per-dtype buffer
    shape: tuple[int, ...]
    buffer_key: str | None = None
    requires_grad: bool = False

    def __reduce__(self) -> tuple:
        return type(self), (
            self.dtype_key,
            self.offset,
            self.shape,
            self.buffer_key,
            self.requires_grad,
        )

    def restore(self, buffers: dict[str, Any]) -> Any:
        """Restore a tensor view, preserving the dtype of empty tensors too."""
        numel = 1
        for size in self.shape:
            numel *= size
        buf = buffers.get(self.buffer_key or self.dtype_key)
        if numel == 0 or buf is None:
            torch = _get_torch()
            dtype = (
                buf.dtype
                if buf is not None
                else getattr(torch, self.dtype_key.removeprefix("torch."))
            )
            return torch.empty(
                self.shape, dtype=dtype, device="cpu", requires_grad=self.requires_grad
            )
        result = buf.narrow(0, self.offset, numel)
        if len(self.shape) != 1:
            result = result.reshape(self.shape)
        return result.requires_grad_(True) if self.requires_grad else result


@dataclass(slots=True)
class _NdarraySlot(_LeafDescriptor):
    """Lightweight stand-in for a numpy array extracted into a coalesced buffer."""

    dtype_key: str  # _NDARRAY_PREFIX + str(ndarray.dtype)
    offset: int  # element offset into the per-dtype buffer
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    buffer_key: str | None = None
    writable: bool = True

    def __reduce__(self) -> tuple:
        return type(self), (
            self.dtype_key,
            self.offset,
            self.shape,
            self.dtype,
            self.buffer_key,
            self.writable,
        )

    def restore(self, buffers: dict[str, Any]) -> Any:
        """Restore a NumPy view without intermediate torch tensor views."""
        numel = 1
        for size in self.shape:
            numel *= size
        if numel == 0:
            return np.empty(self.shape, dtype=self.dtype)
        array = _shared_numpy_view(buffers[self.buffer_key or self.dtype_key])
        result = array[self.offset : self.offset + numel]
        if len(self.shape) != 1:
            result = result.reshape(self.shape)
        if not self.writable:
            result.flags.writeable = False
        return result


def _shared_numpy_view(buffer: Any) -> np.ndarray[Any, Any]:
    """Expose a Zephon buffer with an owner recognizable by the reducer."""
    cached = getattr(buffer, "_zephon_numpy_view", None)
    if cached is not None:
        return cached
    array = buffer.numpy()
    # numpy() creates a detached Tensor owner; attributes on buffer do not carry
    # over. Tag that owner so ordinary NumPy views retain Zephon's provenance.
    array.base._zephon_shm = True
    # numpy() has a separate Tensor owner, so this cache does not form a cycle.
    buffer._zephon_numpy_view = array
    return array


def _array_owner(array: np.ndarray[Any, Any]) -> Any:
    """Follow ndarray and memoryview bases to the object owning their storage."""
    owner: Any = array.base
    while isinstance(owner, (np.ndarray, memoryview)):
        owner = owner.base if isinstance(owner, np.ndarray) else owner.obj
    return owner


def _shared_numpy_storage(array: np.ndarray[Any, Any]) -> Any | None:
    """Return Zephon-owned shared storage containing every byte of an array view.

    Object arrays, empty arrays, and unfamiliar owners use ordinary NumPy
    serialization. Bounds include negative strides and exclude memory outside
    the owning allocation. A small view retains the full allocation, as it
    does for a shared Torch tensor.
    """
    if array.base is None or array.dtype.hasobject or array.size == 0:
        return None
    owner = _array_owner(array)
    if not getattr(owner, "_zephon_shm", False):
        return None
    torch = _get_torch()
    if torch is None or not isinstance(owner, torch.Tensor):
        return None
    storage = owner.untyped_storage()
    if not storage.is_shared():
        return None
    low, high = _byte_bounds(array)
    start = storage.data_ptr()
    if low < start or high > start + storage.nbytes():
        return None
    return storage


def _rebuild_shared_numpy(
    storage: Any,
    offset: int,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    dtype: np.dtype[Any],
    writable: bool,
) -> np.ndarray[Any, Any]:
    torch = _get_torch()
    buffer = torch.empty(0, dtype=torch.uint8, device="cpu").set_(storage)
    array = np.ndarray(
        shape,
        dtype=dtype,
        buffer=_shared_numpy_view(buffer),
        offset=offset,
        strides=strides,
    )
    if not writable:
        array.flags.writeable = False
    return array


class _NumpyPickleFallback:
    """Delegate private numeric arrays to NumPy with the pickler's actual protocol.

    Registered reducers receive no protocol argument. This wrapper preserves
    NumPy's protocol-5 buffer path without changing ordinary array pickling.
    """

    __slots__ = ("array",)

    def __init__(self, array: np.ndarray[Any, Any]) -> None:
        self.array = array

    def __reduce_ex__(self, protocol: int) -> Any:
        return self.array.__reduce_ex__(protocol)


def _reduce_numpy(array: np.ndarray[Any, Any]) -> str | tuple[Any, ...]:
    """Forward Zephon-owned SHM; delegate other arrays to NumPy's reduction."""
    storage = _shared_numpy_storage(array)
    if storage is None:
        if array.dtype.hasobject:
            # Object arrays use NumPy's legacy reduction even with protocol 5.
            # Keep its memoization order so self-references still round-trip.
            return array.__reduce__()
        return operator.getitem, ((_NumpyPickleFallback(array),), 0)
    offset = array.__array_interface__["data"][0] - storage.data_ptr()
    return (
        _rebuild_shared_numpy,
        (
            storage,
            offset,
            array.shape,
            array.strides,
            array.dtype,
            array.flags.writeable,
        ),
    )


# Registration is process-wide. Only Zephon-owned buffers use shared transport;
# other arrays retain NumPy's copy semantics without Zephon names in the pickle.
# Ordinary pickle is unchanged; this reducer never creates shared allocations.
register(np.ndarray, _reduce_numpy)


@dataclass(slots=True)
class _NumericListSlot(_LeafDescriptor):
    """Stand-in for a ``list[int]`` or ``list[float]`` promoted to a tensor for SHM.

    Same layout as ``_TensorSlot`` but restored via ``.tolist()`` so the
    caller gets back a Python list, not a tensor.
    """

    dtype_key: str
    offset: int
    length: int  # original list length (== numel for 1-D)
    buffer_key: str | None = None

    def __reduce__(self) -> tuple:
        return type(self), (self.dtype_key, self.offset, self.length, self.buffer_key)

    def restore(self, buffers: dict[str, Any]) -> Any:
        """Restore the original Python list contract."""
        buf = buffers.get(self.buffer_key or self.dtype_key)
        if self.length == 0 or buf is None:
            return []
        return buf.narrow(0, self.offset, self.length).tolist()


@dataclass(slots=True)
class _BytesSlot(_LeafDescriptor):
    """Lightweight stand-in for a bytes payload moved to the uint8 SHM buffer."""

    offset: int  # byte offset into the _BYTES_DTYPE_KEY buffer
    length: int
    buffer_key: str | None = None

    def __reduce__(self) -> tuple:
        return type(self), (self.offset, self.length, self.buffer_key)

    def restore(self, buffers: dict[str, Any]) -> Any:
        """Restore a bytes view backed by the shared uint8 buffer."""
        buf = buffers[self.buffer_key or _BYTES_DTYPE_KEY]
        if _ShmBytes is _ShmBytes312:
            return _ShmBytes312._from_buffer(buf, self.offset, self.length)
        return _ShmBytes(
            _shared_numpy_view(buf)[self.offset : self.offset + self.length]
        )


@dataclass(slots=True)
class _StructSlot(_LeafDescriptor):
    """Stand-in for a structured type (dataclass / pydantic) whose fields were extracted.

    Only stores the class and the inner skeleton.  ``kind`` and
    ``field_names`` are derived from the class at extraction time (via
    ``_struct_info``) and at reconstruction time (via ``hasattr`` checks)
    so they never travel through pickle.
    """

    cls: type
    inner_skeleton: "_FlatSkeleton"

    def __reduce__(self) -> tuple:
        return type(self), (self.cls, self.inner_skeleton)

    def restore(self, buffers: dict[str, Any]) -> Any:
        """Restore named fields and reconstruct their enclosing object."""
        inner_leaves = _resolve_slots(self.inner_skeleton.slots, buffers)
        fields_dict = cast(
            dict[str, Any],
            optree.tree_unflatten(self.inner_skeleton.spec, inner_leaves),
        )
        return _reconstruct_struct(self, fields_dict)


# ---------------------------------------------------------------------------
# Structured type detection (dataclasses, pydantic, attrs)
# ---------------------------------------------------------------------------
@cache
def _is_struct_type(cls: type) -> bool:
    """Cache format detection; the same payload classes recur across records."""
    return (
        dataclasses.is_dataclass(cls)
        or (hasattr(cls, "model_fields") and hasattr(cls, "model_construct"))
        or hasattr(cls, "__attrs_attrs__")
    )


def _is_struct(obj: Any) -> bool:
    """Return ``True`` for objects we can safely decompose into named fields."""
    return not isinstance(obj, type) and _is_struct_type(type(obj))


_struct_info_cache: dict[type, tuple[str, ...]] = {}


def _struct_info(cls: type) -> tuple[str, ...]:
    """Return cached field names for a struct class."""
    cached = _struct_info_cache.get(cls)
    if cached is not None:
        return cached
    if dataclasses.is_dataclass(cls):
        names = tuple(f.name for f in dataclasses.fields(cls) if f.init)
    elif hasattr(cls, "model_fields"):
        names = tuple(cls.model_fields.keys())
    else:
        names = tuple(a.name for a in cls.__attrs_attrs__)
    _struct_info_cache[cls] = names
    return names


def _struct_to_dict(obj: Any) -> tuple[type, tuple[str, ...], dict[str, Any]]:
    """Decompose a structured object into ``(cls, field_names, fields_dict)``."""
    cls = type(obj)
    names = _struct_info(cls)
    return cls, names, {n: getattr(obj, n) for n in names}


# ---------------------------------------------------------------------------
# Flattened payload skeleton (stored on records during coalescing)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _FlatSkeleton(_PayloadDescriptor):
    """Flattened payload stored on records between coalescing and pickling.

    Always retains a tree spec, including for a container with only one leaf.
    The enclosing SampleRecord retains its metadata separately.
    """

    # Root leaves use their own descriptor; this specification is always a tree.
    slots: list[Any]  # flat list of slots + inline leaf values
    spec: optree.PyTreeSpec  # structure recipe for tree_unflatten

    def __reduce__(self) -> tuple:
        return type(self), (self.slots, self.spec)

    def bind(self, buffers: dict[str, Any]) -> LazyPayload:
        """Keep the existing tree lazy payload without retaining this wrapper."""
        return ShmLazyPayload(self.slots, self.spec, _used_buffers(self.slots, buffers))


# ---------------------------------------------------------------------------
# SHM-backed bytes view
# ---------------------------------------------------------------------------
class _ShmBytes312:
    """Zero-copy bytes view backed by a SHM uint8 tensor."""

    __slots__ = ("_buffer", "_offset", "_np_view")

    def __init__(self, tensor_view: Any) -> None:
        self._buffer = tensor_view
        self._offset = 0
        self._np_view = _shared_numpy_view(tensor_view)

    @classmethod
    def _from_buffer(cls, buffer: Any, offset: int, length: int) -> _ShmBytes312:
        value = cls.__new__(cls)
        value._buffer = buffer
        value._offset = offset
        value._np_view = _shared_numpy_view(buffer)[offset : offset + length]
        return value

    @property
    def _tensor_view(self) -> Any:
        if self._offset == 0 and len(self) == self._buffer.numel():
            return self._buffer
        return self._buffer.narrow(0, self._offset, len(self))

    def __reduce_ex__(self, protocol: int) -> tuple:
        # ForkingPickler handles the tensor via SHM FD passing (zero copy).
        # Regular pickle falls back to copying tensor data inline.
        return (_rebuild_shm_bytes, (self._buffer, self._offset, len(self)))

    def __len__(self) -> int:
        return self._np_view.size

    def __bytes__(self) -> bytes:
        return bytes(self._np_view)

    def __getitem__(self, key: Any) -> Any:
        return self._np_view[key]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (bytes, memoryview, bytearray)):
            return bytes(self) == bytes(other)
        if isinstance(other, _ShmBytes312):
            return bytes(self) == bytes(other)
        return NotImplemented

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        """Decode SHM bytes as text."""
        return bytes(self._np_view).decode(encoding, errors)

    def __buffer__(self, flags: int) -> memoryview:  # type: ignore[override]
        return self._np_view.__buffer__(flags)


def _rebuild_shm_bytes(buffer: Any, offset: int, length: int) -> _ShmBytes312:
    return _ShmBytes312._from_buffer(buffer, offset, length)


class _ShmBytesLegacy(bytes):
    """Bytes-like fallback for Python < 3.12."""

    def __new__(cls, tensor_view: Any) -> "_ShmBytesLegacy":
        if hasattr(tensor_view, "numpy"):
            return super().__new__(cls, bytes(tensor_view.numpy()))
        return super().__new__(cls, bytes(tensor_view))


def _make_shm_bytes_class(
    version_info: tuple[int, int] | tuple[int, int, int],
) -> type:
    """Build the platform/version-specific ``_ShmBytes`` implementation."""
    return _ShmBytes312 if version_info >= (3, 12) else _ShmBytesLegacy


_ShmBytes = _make_shm_bytes_class(sys.version_info[:3])


# ---------------------------------------------------------------------------
# pytree-based leaf extraction (replaces recursive _extract_payload)
# ---------------------------------------------------------------------------
_PRIMITIVE_TYPES = (int, float, str, bool, type(None))


def _is_leaf(obj: Any) -> bool:
    """Predicate for ``optree.tree_flatten``.

    Structured types are handled in ``_extract_leaf``, which decomposes them
    into a fields dict and recurses via ``_extract_payload_pytree``. Ordinary
    dicts, lists, and tuples use optree's container traversal.

    Lists of primitives (int, float, str, …) are also treated as leaves
    to avoid inflating the slot count.  A ``List[int]`` with 500 elements
    would otherwise become 500 individual int leaves — catastrophic for
    pickle size and restore time at high DOP.  We check the first and
    last element as an O(1) heuristic; typed fields (pydantic, dataclass)
    guarantee homogeneity so this is safe in practice.  If wrong, the
    only consequence is a missed tensor falling back to pickle (perf,
    not correctness).
    """
    if type(obj) in _PRIMITIVE_TYPES or type(obj) is dict:
        return False
    if _is_struct(obj):
        return True
    if (
        isinstance(obj, list)
        and len(obj) > 0
        and isinstance(obj[0], _PRIMITIVE_TYPES)
        and isinstance(obj[-1], _PRIMITIVE_TYPES)
    ):
        return True
    return False


def _memoryview_source(view: memoryview) -> Any:
    """Return a NumPy source whose C-order bytes match ``view.tobytes()``.

    Contiguous inputs become flat uint8 views; supported strided inputs keep
    their dtype, shape, and strides for ``np.copyto`` into SHM. These results
    may alias the source until that copy finishes. Unsupported formats and
    noncontiguous structured views use owned bytes, preserving padding that
    NumPy field assignment could omit.
    """
    if view.c_contiguous:
        return np.frombuffer(view, dtype=np.uint8)
    try:
        array = np.asarray(view)
        if not array.dtype.hasobject and array.dtype.fields is None:
            return array
    except Exception:  # Any unsupported exporter falls back to a byte copy.
        pass
    return np.frombuffer(view.tobytes(), dtype=np.uint8)


@cache
def _torch_dtype_for(dtype: Any) -> Any | None:
    """Resolve supported NumPy dtypes once; unsupported dtypes remain inline."""
    try:
        return _get_torch().from_numpy(np.empty(0, dtype=dtype)).dtype
    except (TypeError, ValueError):
        return None


@cache
def _numpy_dtype_key(dtype: Any) -> str:
    return _NDARRAY_PREFIX + str(dtype)


def _restore_inline_tensor(
    data: Any, dtype: Any, shape: tuple[int, ...], grad: bool
) -> Any:
    torch = _get_torch()
    if 0 in shape:
        return torch.empty(shape, dtype=dtype, device="cpu", requires_grad=grad)
    # Protocol 4 reconstructs bytes; tensors need writable storage.
    if isinstance(data, bytes):
        data = bytearray(data)
    return torch.frombuffer(data, dtype=dtype).reshape(shape).requires_grad_(grad)


class _InlineTensor(_LeafDescriptor):
    """Serialize tensor values without invoking Torch's SHM reducer."""

    def __init__(self, tensor: Any) -> None:
        self.tensor = tensor

    def restore(self, buffers: dict[str, Any]) -> Any:
        return self.tensor

    def __reduce_ex__(self, protocol: int) -> Any:
        tensor = self.tensor.detach().resolve_conj().resolve_neg().contiguous()
        view = tensor.reshape(-1).view(_get_torch().uint8).numpy()
        data = pickle.PickleBuffer(view) if protocol >= 5 else bytearray(view)
        return _InlineTensorData, (
            data,
            tensor.dtype,
            tuple(tensor.shape),
            self.tensor.requires_grad,
        )


@dataclass(slots=True)
class _InlineTensorData(_LeafDescriptor):
    """Keep inline tensor bytes inline across lazy forwarding hops."""

    data: Any
    dtype: Any
    shape: tuple[int, ...]
    requires_grad: bool

    def __reduce__(self) -> tuple:
        return type(self), (self.data, self.dtype, self.shape, self.requires_grad)

    def restore(self, buffers: dict[str, Any]) -> Any:
        return _restore_inline_tensor(
            self.data, self.dtype, self.shape, self.requires_grad
        )


def _restore_inline_numpy(array: Any, writable: bool) -> Any:
    if not writable:
        array.flags.writeable = False
    return array


class _InlineNumpy:
    """Use NumPy's own serialization even for a Zephon-shared source."""

    def __init__(self, array: Any) -> None:
        self.array = array

    def __reduce_ex__(self, protocol: int) -> Any:
        return _restore_inline_numpy, (
            _NumpyPickleFallback(self.array),
            self.array.flags.writeable,
        )


@dataclass(slots=True)
class _BufferPayload(_LeafDescriptor):
    """Type-specific description; policy and allocation stay format-independent."""

    source: Any
    slot: Any
    dtype_key: str
    numel: int
    nbytes: int
    shared_nbytes: int | None = None
    shared_storage: Any = None
    result: Any = None
    group: Any = None

    def restore(self, buffers: dict[str, Any]) -> Any:
        value = self.result
        return (
            _restore_slot(value, buffers)
            if isinstance(value, _LeafDescriptor)
            else value
        )

    def __reduce_ex__(self, protocol: int) -> Any:
        value = self.result
        if isinstance(value, _LeafDescriptor):
            return value.__reduce_ex__(protocol)
        return _InlineSlot, (value,)


class _BufferPlan:
    """Collect writes using a single memory policy and bounded coalescing groups."""

    def __init__(
        self, policy: PayloadMemoryPolicy, prepare_reused: bool = False
    ) -> None:
        self.policy = policy
        self.prepare_reused = prepare_reused
        self.reused: dict[str, Any] = {}
        self.reused_keys: dict[tuple[int, str], str] = {}
        self.collector: dict[str, list[Any]] = {}
        self.offsets: dict[str, int] = {}
        self.dtype_keys: dict[str, str] = {}
        self.current: dict[str, str] = {}
        self.changed = False
        self.saw_buffer = False
        self.fallbacks: dict[str, Any] = {}
        self.pending: list[_BufferPayload] = []
        self.group_bytes: dict[Any, int] = {}
        self.private_groups: dict[str, int] = {}

    def collect(self, value: _BufferPayload, choice: str) -> _BufferPayload:
        if value.shared_storage is not None:
            group = ("shared", value.shared_storage._cdata)
        elif choice == "coalesce":
            index = self.private_groups.get(value.dtype_key, 0)
            group = ("private", value.dtype_key, index)
            limit = self.policy.coalesce_max_size
            if (
                limit is not None
                and self.group_bytes.get(group, 0) + value.nbytes > limit
            ):
                index += 1
                self.private_groups[value.dtype_key] = index
                group = ("private", value.dtype_key, index)
        else:
            group = ("separate", id(value))
        value.group = group
        self.group_bytes[group] = self.group_bytes.get(group, 0) + value.nbytes
        self.pending.append(value)
        return value

    def inline(self, value: _BufferPayload) -> Any:
        leaf = value.source
        if isinstance(value.slot, _TensorSlot):
            self.changed = True
            return _InlineTensor(leaf)
        if isinstance(value.slot, _NdarraySlot) and value.shared_nbytes is not None:
            self.changed = True
            return _InlineNumpy(leaf)
        if isinstance(leaf, (memoryview, _ShmBytes312)):
            self.changed = True
            return leaf.tobytes() if isinstance(leaf, memoryview) else bytes(leaf)
        return leaf

    def finish(self) -> None:
        """Amortize transport per buffer, and account for sibling views in this message."""
        for value in self.pending:
            useful = self.group_bytes[value.group]
            minimum = (
                self.policy.min_reuse_size
                if value.shared_nbytes is not None
                else self.policy.min_buffer_size
            )
            if useful < minimum:
                value.result = self.inline(value)
            elif value.shared_nbytes is not None:
                # Sum logical bytes conservatively. Overlapping views can make
                # us skip a useful compaction, but never justify an extra copy.
                if self.policy.choose(useful, value.shared_nbytes) == "reuse":
                    value.result = (
                        self.reuse(value) if useful > value.nbytes else value.source
                    )
                elif value.nbytes < self.policy.min_buffer_size:
                    # Compacted views get separate destinations. Apply the new
                    # allocation minimum to each one, not their old shared slab.
                    value.result = self.inline(value)
                else:
                    value.result = self.add(value, separate=True, fallback=value.source)
            else:
                choice = self.policy.choose(value.nbytes, None)
                if (
                    self.prepare_reused
                    and useful == value.nbytes
                    and isinstance(value.slot, _TensorSlot)
                    and value.source.is_contiguous()
                    and not value.source.is_conj()
                    and not value.source.is_neg()
                    and value.source.untyped_storage().nbytes() == value.nbytes
                ):
                    # One tensor using its whole storage needs no coalescing.
                    # Torch's reducer does not preserve conjugate/negative bits.
                    # Torch's normal reducer already moves that storage to SHM.
                    value.result = value.source
                else:
                    value.result = self.add(value, separate=choice == "separate")
            # The copy collector or final descriptor now owns what it needs.
            # Do not retain original large crops in the serialized skeleton.
            value.source = None

    def reuse(self, value: _BufferPayload) -> Any:
        """Describe contiguous shared views once instead of reducing each owner again."""
        leaf, storage = value.source, value.shared_storage
        if not self.prepare_reused or storage is None:
            return leaf
        if isinstance(value.slot, _TensorSlot):
            if not leaf.is_contiguous() or leaf.is_conj() or leaf.is_neg():
                return leaf
            offset, dtype = leaf.storage_offset(), leaf.dtype
        elif isinstance(value.slot, _NdarraySlot):
            offset = leaf.__array_interface__["data"][0] - storage.data_ptr()
            if (
                not leaf.flags.c_contiguous
                or offset % leaf.itemsize
                or storage.nbytes() % leaf.itemsize
            ):
                return leaf
            offset //= leaf.itemsize
            dtype = _torch_dtype_for(leaf.dtype)
        elif isinstance(value.slot, _BytesSlot):
            if not leaf._np_view.flags.c_contiguous:
                return leaf
            offset = leaf._buffer.storage_offset() + leaf._offset
            dtype = _get_torch().uint8
        else:
            return leaf
        identity = storage._cdata, value.dtype_key
        key = self.reused_keys.get(identity)
        if key is None:
            key = self.reused_keys[identity] = f"reuse:{len(self.reused)}"
            self.reused[key] = (
                _get_torch().empty(0, dtype=dtype, device="cpu").set_(storage)
            )
        value.slot.buffer_key = key
        value.slot.offset = offset
        self.changed = True
        return value.slot

    def add(
        self, value: _BufferPayload, *, separate: bool, fallback: Any = None
    ) -> Any:
        source = value.source
        if isinstance(value.slot, _NumericListSlot):
            kind = type(source[0])
            if set(map(type, source)) != {kind}:
                return source
            try:
                array = np.array(source, dtype=np.int64 if kind is int else np.float64)
            except OverflowError:
                return source
            source = _get_torch().from_numpy(array)
        elif isinstance(value.slot, _BytesSlot):
            view = source._np_view if isinstance(source, _ShmBytes312) else source
            source = (
                _memoryview_source(view)
                if isinstance(view, memoryview)
                else np.frombuffer(view, dtype=np.uint8)
            )
        elif isinstance(value.slot, _TensorSlot) and source.requires_grad:
            source = source.detach()
        dtype = value.dtype_key
        key = None if separate else self.current.get(dtype)
        limit = self.policy.coalesce_max_size
        if key is not None and limit is not None:
            itemsize = value.nbytes // value.numel
            if self.offsets[key] * itemsize + value.nbytes > limit:
                key = None
        if key is None:
            key = (
                dtype
                if dtype not in self.collector
                else f"{dtype}#{len(self.collector)}"
            )
            self.collector[key] = []
            self.offsets[key] = 0
            self.dtype_keys[key] = dtype
            if not separate:
                self.current[dtype] = key
        if fallback is not None:
            self.fallbacks[key] = fallback
        value.slot.buffer_key = key
        value.slot.offset = self.offsets[key]
        self.offsets[key] += value.numel
        self.collector[key].append(source)
        self.changed = True
        return value.slot


def _describe_buffer(leaf: Any) -> _BufferPayload | None:
    """Adapt supported CPU payloads to descriptors and direct-copy sources."""
    torch = _get_torch()
    if type(leaf) is torch.Tensor:
        if (
            leaf.device.type != "cpu"
            or leaf.layout != torch.strided
            or leaf.is_quantized
        ):
            return None
        if leaf.requires_grad and not leaf.is_leaf:
            return None
        dtype = str(leaf.dtype)
        numel = leaf.numel()
        storage = leaf.untyped_storage()
        shared = storage.is_shared()
        return _BufferPayload(
            leaf,
            _TensorSlot(dtype, 0, tuple(leaf.shape), requires_grad=leaf.requires_grad),
            dtype,
            numel,
            numel * leaf.element_size(),
            storage.nbytes() if shared else None,
            storage if shared else None,
        )
    if type(leaf) is np.ndarray:
        if _torch_dtype_for(leaf.dtype) is None:
            return None
        storage = _shared_numpy_storage(leaf)
        dtype = _numpy_dtype_key(leaf.dtype)
        return _BufferPayload(
            leaf,
            _NdarraySlot(
                dtype, 0, tuple(leaf.shape), leaf.dtype, writable=leaf.flags.writeable
            ),
            dtype,
            leaf.size,
            leaf.nbytes,
            storage.nbytes() if storage is not None else None,
            storage,
        )
    if isinstance(leaf, (bytes, bytearray, memoryview, _ShmBytes312)):
        storage = (
            leaf._buffer.untyped_storage() if isinstance(leaf, _ShmBytes312) else None
        )
        shared = storage.nbytes() if storage is not None else None
        size = leaf.nbytes if isinstance(leaf, memoryview) else len(leaf)
        return _BufferPayload(
            leaf,
            _BytesSlot(0, size),
            _BYTES_DTYPE_KEY,
            size,
            size,
            shared,
            storage,
        )
    if type(leaf) is list and leaf:
        kind = type(leaf[0])
        if kind not in (int, float):
            return None
        dtype = "torch.int64" if kind is int else "torch.float64"
        return _BufferPayload(
            leaf, _NumericListSlot(dtype, 0, len(leaf)), dtype, len(leaf), len(leaf) * 8
        )
    return None


def _extract_leaf(
    leaf: Any,
    collector: dict[str, list[Any]],
    offsets: dict[str, int],
    shm_min_size: int,
    plan: _BufferPlan | None = None,
) -> Any:
    """Apply one policy after adapting a leaf; never copy into a staging slab."""
    if plan is None:
        plan = _BufferPlan(
            PayloadMemoryPolicy(
                shm_min_size=shm_min_size,
                min_buffer_size=shm_min_size,
                min_reuse_size=shm_min_size,
            )
        )
        plan.collector, plan.offsets = collector, offsets
    if type(leaf) in _PRIMITIVE_TYPES:
        return leaf
    if _is_struct(leaf):
        cls, _names, fields = _struct_to_dict(leaf)
        inner = _extract_payload_pytree(fields, collector, offsets, shm_min_size, plan)
        return _StructSlot(cls=cls, inner_skeleton=inner)
    value = _describe_buffer(leaf)
    if value is None:
        return leaf
    plan.saw_buffer = True
    choice = plan.policy.choose(value.nbytes, value.shared_nbytes)
    if choice == "inline":
        return plan.inline(value)
    return plan.collect(value, choice)


def _extract_payload_pytree(
    payload: Any,
    collector: dict[str, list[Any]],
    offsets: dict[str, int],
    shm_min_size: int,
    plan: _BufferPlan | None = None,
) -> _FlatSkeleton:
    """Flatten payload via optree and replace tensor/bytes leaves with slots."""
    own_plan = plan is None
    if plan is None:
        plan = _BufferPlan(
            PayloadMemoryPolicy(
                shm_min_size=shm_min_size,
                min_buffer_size=shm_min_size,
                min_reuse_size=shm_min_size,
            )
        )
        plan.collector, plan.offsets = collector, offsets
    leaves, spec = optree.tree_flatten(payload, is_leaf=_is_leaf)
    for i, leaf in enumerate(leaves):
        leaves[i] = _extract_leaf(leaf, collector, offsets, shm_min_size, plan)
    if own_plan:
        plan.finish()
        leaves = [
            leaf.result if isinstance(leaf, _BufferPayload) else leaf for leaf in leaves
        ]
    return _FlatSkeleton(slots=leaves, spec=spec)


def _extract_record_payload(
    payload: Any,
    collector: dict[str, list[Any]],
    offsets: dict[str, int],
    shm_min_size: int,
    plan: _BufferPlan | None = None,
) -> _PayloadDescriptor | LazyPayload:
    """Describe a root leaf directly; preserve the structure of containers."""
    if isinstance(payload, LazyPayload):
        if not isinstance(
            payload, (_ShmLeafPayload, ShmLazyPayload, _PreparedPayload)
        ) or (plan is not None and payload._policy == plan.policy):
            return payload
        payload = payload.resolve_payload()
        if plan is not None:
            plan.changed = True
    if not isinstance(payload, dict):
        descriptor = _extract_leaf(payload, collector, offsets, shm_min_size, plan)
        if isinstance(descriptor, _LeafDescriptor):
            return descriptor
        if (
            descriptor is not payload
            or not isinstance(payload, (dict, list, tuple))
            or _is_leaf(payload)
        ):
            return _InlineSlot(descriptor)
    return _extract_payload_pytree(payload, collector, offsets, shm_min_size, plan)


def _extract_from_records(
    items: list[StreamItem],
    collector: dict[str, list[Any]],
    offsets: dict[str, int],
    shm_min_size: int,
    plan: _BufferPlan | None = None,
) -> list[tuple[SampleRecord, _PayloadDescriptor | LazyPayload]]:
    """Extract tensors from records without mutating payloads yet.

    Returns (record, descriptor) pairs. The caller commits the descriptors
    only after shared-buffer allocation succeeds.
    """
    descriptors: list[tuple[SampleRecord, _PayloadDescriptor | LazyPayload]] = []
    for item in items:
        if isinstance(item, SampleRecord):
            descriptor = _extract_record_payload(
                item.payload, collector, offsets, shm_min_size, plan
            )
            descriptors.append((item, descriptor))
        elif isinstance(item, SampleBatch):
            for rec in item.records:
                descriptor = _extract_record_payload(
                    rec.payload, collector, offsets, shm_min_size, plan
                )
                descriptors.append((rec, descriptor))
    return descriptors


# ---------------------------------------------------------------------------
# Build coalesced SHM buffers
# ---------------------------------------------------------------------------
def _alloc_shm_buffer(numel: int, dtype: Any, label: str, *, retry: bool = True) -> Any:
    """Allocate a 1-D tensor directly in shared memory, retrying on ENOSPC.

    Transient ``/dev/shm`` exhaustion blocks in :func:`wait_for_shm_space`
    instead of crashing the worker.
    """
    torch = _get_torch()
    buf = torch.empty(0, dtype=dtype, device="cpu")
    nbytes = numel * buf.element_size()
    while True:
        try:
            # Torch's default_collate uses this allocator through typed storage;
            # it honors torch.multiprocessing.get_sharing_strategy().
            storage = torch.UntypedStorage._new_shared(nbytes, device="cpu")
        except Exception as e:
            if not retry or not is_shm_error(e):
                raise
            wait_for_shm_space(label)
        else:
            return buf.set_(storage)


def _build_shm_buffers(
    collector: dict[str, list[Any]],
    dtype_keys: dict[str, str] | None = None,
    fallbacks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Concatenate collected tensors/bytes per dtype into SHM-backed tensors.

    Allocates the target directly in shared memory, then writes tensors,
    strided NumPy arrays, and supported memoryviews into the shared region.
    Numeric lists are converted before allocation.

    If ``/dev/shm`` is exhausted, retries with exponential backoff via
    :func:`_alloc_shm_buffer` instead of propagating the error.
    """
    buffers: dict[str, Any] = {}
    torch = _get_torch()
    for buffer_key, items in collector.items():
        dtype_key = dtype_keys[buffer_key] if dtype_keys is not None else buffer_key
        optional = fallbacks is not None and buffer_key in fallbacks

        def allocate(count: int, dtype: Any) -> Any:
            if optional:
                return _alloc_shm_buffer(
                    count, dtype, f"compact[{dtype_key}]", retry=False
                )
            return _alloc_shm_buffer(count, dtype, f"coalesce[{dtype_key}]")

        try:
            if dtype_key == _BYTES_DTYPE_KEY:
                # Pack raw bytes into a uint8 tensor.
                total = sum(array.nbytes for array in items)
                buf = allocate(total, torch.uint8)
                np_buf = buf.numpy()
                if len(items) > 1 and all(a.dtype == np.uint8 for a in items):
                    np.concatenate(items, axis=None, out=np_buf, casting="no")
                    buffers[buffer_key] = buf
                    continue
                offset = 0
                for array in items:
                    n = array.nbytes
                    np.copyto(
                        np_buf[offset : offset + n]
                        .view(array.dtype)
                        .reshape(array.shape),
                        array,
                        casting="no",
                    )
                    offset += n
                buffers[buffer_key] = buf
            elif dtype_key.startswith(_NDARRAY_PREFIX):
                # NumPy can read strided, reversed, and read-only sources directly
                # into the final contiguous destination. No ascontiguousarray copy
                # or per-array Torch wrapper is needed.
                total_numel = sum(array.size for array in items)
                if total_numel == 0:
                    continue
                dtype = _torch_dtype_for(items[0].dtype)
                buf = allocate(total_numel, dtype)
                np_buf = buf.numpy()
                if len(items) > 1:
                    # concatenate(axis=None, out=...) copies strided sources
                    # directly into the destination in C order. It avoids
                    # creating a Python destination view for every source.
                    np.concatenate(items, axis=None, out=np_buf, casting="no")
                    buffers[buffer_key] = buf
                    continue
                offset = 0
                for array in items:
                    n = array.size
                    if n > 0:
                        np.copyto(
                            np_buf[offset : offset + n].reshape(array.shape),
                            array,
                            casting="no",
                        )
                    offset += n
                buffers[buffer_key] = buf
            else:
                total_numel = sum(t.numel() for t in items)
                if total_numel == 0:
                    continue
                dtype = items[0].dtype
                # Allocate directly in SHM, then write sub-tensors in.
                buf = allocate(total_numel, dtype)
                if len(items) > 1 and all(t.is_contiguous() for t in items):
                    flat = [t if t.ndim == 1 else t.view(-1) for t in items]
                    torch.cat(flat, out=buf)
                    buffers[buffer_key] = buf
                    continue
                offset = 0
                for t in items:
                    n = t.numel()
                    if n > 0:
                        buf.narrow(0, offset, n).view(t.shape).copy_(t)
                    offset += n
                buffers[buffer_key] = buf
        except Exception as exc:
            if not optional or not is_shm_error(exc):
                raise
            assert fallbacks is not None
            buffers[buffer_key] = _RetainedView(fallbacks[buffer_key])
    return buffers


def _used_buffers(slots: list[Any], buffers: dict[str, Any]) -> dict[str, Any]:
    keys: set[str] = set()
    for slot in slots:
        if isinstance(slot, _StructSlot):
            keys.update(_used_buffers(slot.inner_skeleton.slots, buffers))
        elif isinstance(slot, (_TensorSlot, _NdarraySlot, _NumericListSlot)):
            keys.add(slot.buffer_key or slot.dtype_key)
        elif isinstance(slot, _BytesSlot):
            keys.add(slot.buffer_key or _BYTES_DTYPE_KEY)
    if len(keys) == len(buffers) and keys == buffers.keys():
        return buffers
    return {key: buffers[key] for key in keys if key in buffers}


# ---------------------------------------------------------------------------
# Lazy payload (deferred restoration)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _ShmLeafPayload(LazyPayload):
    """One deferred value, with no singleton list or tree specification."""

    _descriptor: _LeafDescriptor
    _buffers: dict[str, Any]
    _policy: PayloadMemoryPolicy | None = None

    def resolve_payload(self) -> SamplePayload:
        """Restore the leaf using its descriptor's existing type knowledge."""
        return cast(SamplePayload, _restore_slot(self._descriptor, self._buffers))

    def __reduce__(self) -> tuple:
        """Forward the descriptor and buffers without resolving the value."""
        return (_ShmLeafPayload, (self._descriptor, self._buffers, self._policy))


@dataclass(slots=True)
class ShmLazyPayload(LazyPayload):
    """SHM-backed deferred payload — flat slots + pytree spec + SHM buffer refs.

    Created on unpickle (main process).  Resolved explicitly via
    :func:`resolve_lazy_payloads` in the worker process before
    ``process_many()``, or on the pump thread for accumulators that
    declare ``reads_payload = True``.
    """

    _slots: list[Any]
    _spec: optree.PyTreeSpec
    _buffers: dict[str, Any]  # shared ref keeps SHM alive via refcounting
    _policy: PayloadMemoryPolicy | None = None

    def resolve_payload(self) -> SamplePayload:
        """Materialize the full payload by replacing slots with SHM views."""
        restored = _resolve_slots(self._slots, self._buffers)
        return cast(SamplePayload, optree.tree_unflatten(self._spec, restored))

    def __reduce__(self) -> tuple:
        """Pickle without resolving — forward (slots, spec, buffers) as-is."""
        return (
            _make_shm_lazy_payload,
            (self._slots, self._spec, self._buffers, self._policy),
        )


def _make_shm_lazy_payload(
    slots: list[Any],
    spec: optree.PyTreeSpec,
    buffers: dict[str, Any],
    policy: PayloadMemoryPolicy | None = None,
) -> ShmLazyPayload:
    """Unpickle constructor for :class:`ShmLazyPayload`."""
    return ShmLazyPayload(slots, spec, buffers, policy)


def _reconstruct_struct(slot: _StructSlot, fields_dict: dict[str, Any]) -> Any:
    """Rebuild a structured object from its ``_StructSlot`` and resolved fields."""
    cls = slot.cls
    # Pydantic: bypass validators via model_construct.
    if hasattr(cls, "model_construct"):
        return cls.model_construct(**fields_dict)
    return cls(**fields_dict)


def _resolve_slots(slots: list[Any], buffers: dict[str, Any]) -> list[Any]:
    """Replace slot placeholders with zero-copy views into SHM buffers.

    Each slot becomes a tensor/ndarray/bytes view backed by the coalesced
    SHM buffer, or a reconstructed struct for ``_StructSlot``.
    """
    return [
        _restore_slot(item, buffers) if isinstance(item, _LeafDescriptor) else item
        for item in slots
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_metadata_values = operator.attrgetter(
    *(field.name for field in dataclasses.fields(SampleMeta))
)


@dataclass(slots=True)
class _MetadataWire:
    """Avoid frozen-dataclass field discovery on every process-stage message."""

    value: SampleMeta

    def __reduce__(self) -> tuple:
        return SampleMeta, _metadata_values(self.value)


def _wire_records(items: list[StreamItem]) -> list[Any]:
    """Encode records as tuples inside the process transport envelope.

    Tags distinguish records (0), batches (1), and untouched stream items (2).
    This avoids a Python pickle reduction for every record and batch.
    """
    metadata: dict[int, _MetadataWire] = {}

    def record(value: SampleRecord) -> tuple[Any, Any]:
        meta = value.meta
        if type(meta) is SampleMeta:
            key = id(meta)
            wrapped = metadata.get(key)
            if wrapped is None:
                wrapped = metadata[key] = _MetadataWire(meta)
            return wrapped, value.payload
        return meta, value.payload

    return [
        (0, record(item))
        if type(item) is SampleRecord
        else (1, tuple(record(rec) for rec in item.records))
        if type(item) is SampleBatch
        else (2, item)
        for item in items
    ]


def _reconstruct_wire_microbatch(
    wire: list[tuple[int, Any]],
    buffers: dict[str, Any],
    policy: PayloadMemoryPolicy | None,
) -> list[StreamItem]:
    skeleton = [
        SampleRecord(*value)
        if tag == 0
        else SampleBatch(tuple(SampleRecord(*record) for record in value))
        if tag == 1
        else value
        for tag, value in wire
    ]
    return _reconstruct_microbatch_lazy(skeleton, buffers, policy)


@dataclass(slots=True)
class CoalescedMicrobatch:
    """Microbatch with prepared inline values and shared buffers.

    On unpickle (``__reduce__``), produces ``list[StreamItem]`` where each
    record's payload is a :class:`LazyPayload`. Call
    :func:`resolve_lazy_payloads` to materialize before use.
    """

    skeleton: list[StreamItem]
    buffers: dict[str, Any]  # coalescing group -> shared tensor or retained view
    policy: PayloadMemoryPolicy | None = None

    def __reduce__(self) -> tuple:
        return (
            _reconstruct_wire_microbatch,
            (_wire_records(self.skeleton), self.buffers, self.policy),
        )

    def __len__(self) -> int:
        return len(self.skeleton)


def coalesce_microbatch(
    items: list[StreamItem],
    shm_min_size: int = DEFAULT_SHM_MIN_SIZE,
    *,
    policy: PayloadMemoryPolicy | None = None,
    copy_records: bool = False,
    ensure_prepared: bool = False,
) -> CoalescedMicrobatch | None:
    """Prepare supported values using a common transport and memory policy.

    The legacy function name is retained, but inline transport and view
    compaction also apply when coalescing is disabled. ``shm_min_size`` is a
    shortcut setting all inline minimums; an explicit ``policy`` takes precedence.

    Returns None when values need no preparation, unless ``ensure_prepared``
    keeps accepted inline values lazy too. ``copy_records`` protects the
    parent's retry records when preparation runs in a queue feeder.

    Extraction leaves records unchanged until allocation succeeds. Optional
    view compaction keeps the source on SHM exhaustion instead of waiting for
    space held by that source. Required allocations retain normal backpressure.
    """
    if _get_torch() is None:
        return None
    plan = _BufferPlan(
        policy
        or PayloadMemoryPolicy(
            shm_min_size=shm_min_size,
            min_buffer_size=shm_min_size,
            min_reuse_size=shm_min_size,
        ),
        prepare_reused=ensure_prepared,
    )
    descriptors = _extract_from_records(
        items, plan.collector, plan.offsets, plan.policy.shm_min_size, plan
    )
    plan.finish()
    if not plan.changed:
        if not ensure_prepared:
            return None
        # Already prepared lazy payloads stay intact. Other inline payloads
        # need no flattened skeleton once the policy has accepted them.
        descriptors = [
            (
                rec,
                rec.payload
                if not plan.saw_buffer or isinstance(rec.payload, LazyPayload)
                else _PreparedPayload(rec.payload, plan.policy),
            )
            for rec, _ in descriptors
        ]
    buffers = _build_shm_buffers(plan.collector, plan.dtype_keys, plan.fallbacks)
    buffers.update(plan.reused)

    # Only allocate replacement records if preparation changed the message.
    if copy_records:
        pending = iter(descriptor for _, descriptor in descriptors)
        items = [
            SampleRecord(item.meta, cast(SamplePayload, next(pending)))
            if isinstance(item, SampleRecord)
            else SampleBatch(
                records=tuple(
                    SampleRecord(r.meta, cast(SamplePayload, next(pending)))
                    for r in item.records
                )
            )
            if isinstance(item, SampleBatch)
            else item
            for item in items
        ]
    else:
        # Commit: metadata stays on each record; only its payload is replaced.
        for rec, descriptor in descriptors:
            rec.payload = descriptor  # type: ignore[assignment]

    return CoalescedMicrobatch(skeleton=items, buffers=buffers, policy=plan.policy)


# ---------------------------------------------------------------------------
# Unpickle reconstruction (lazy)
# ---------------------------------------------------------------------------
def _reconstruct_microbatch_lazy(
    skeleton: list[StreamItem],
    buffers: dict[str, Any],
    policy: PayloadMemoryPolicy | None = None,
) -> list[StreamItem]:
    """Bind each payload descriptor to shared buffers without restoring it."""

    def bind(descriptor: Any) -> Any:
        if not isinstance(descriptor, _PayloadDescriptor):
            return descriptor
        payload = cast(_ShmLeafPayload | ShmLazyPayload, descriptor.bind(buffers))
        payload._policy = policy
        return payload

    for item in skeleton:
        if isinstance(item, SampleRecord):
            item.payload = bind(item.payload)
        elif isinstance(item, SampleBatch):
            for rec in item.records:
                rec.payload = bind(rec.payload)
    return skeleton


@dataclass(slots=True)
class TransportMicrobatch:
    """Apply a process-stage policy during serialization, preserving retry inputs."""

    items: list[Any]
    policy: PayloadMemoryPolicy

    def __reduce__(self) -> tuple:
        prepared = coalesce_microbatch(
            self.items, policy=self.policy, copy_records=True, ensure_prepared=True
        )
        if prepared is not None:
            return prepared.__reduce__()
        return list, (self.items,)
