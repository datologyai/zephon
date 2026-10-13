"""Coalesce per-microbatch tensors, numpy arrays, and large bytes into SHM buffers.

Reduces the number of POSIX SHM segments (and file descriptors) from
N-per-microbatch to K-per-microbatch, grouping by distinct dtypes
(usually 1–2) and splitting groups at the coalescing limit.
Large ``bytes``/``memoryview``/``bytearray`` payloads are packed
into a ``torch.uint8`` buffer so they travel through the same SHM path.

Supported payload types:

- **torch.Tensor** (CPU only) — coalesced by dtype, restored as zero-copy
  views into the SHM buffer.
- **numpy.ndarray** — private arrays are coalesced into shared buffers.
  Arrays backed by Zephon's shared buffers can be forwarded by the
  multiprocessing reducer, preserving their dtype, strides, and writeability.
- **bytes / memoryview / bytearray** — payloads above a size threshold are packed
  into a uint8 SHM buffer and restored as ``_ShmBytes`` wrappers. Fresh copies
  support index and same-length slice assignment; wrappers cannot resize.
  Python 3.12+ exposes the shared buffer directly. Python 3.10/3.11 materializes
  a bytearray subclass.
- **homogeneous numeric lists** — coalesced as int64/float64 and restored
  as ordinary lists. Mixed lists and integers outside int64 use pickle.

Worker-output decisions::

    shm_enabled? -- no --> ordinary multiprocessing serialization
      |
      yes
      |
    Supported, nonempty value
      |
    Already in SHM? -- no ---------------------------------------------+
      |                                                               |
      yes                                                             |
      |                                                               |
    Group views by existing allocation                                |
      |                                                               |
    useful < shm_min_forward_bytes? -- yes ----------------------------+
      |                                                               |
      no                                                              |
      |                                                               |
    Compact? Both must hold:                                           |
      backing > useful * shm_compact_above_ratio                       |
      backing - useful >= shm_compact_min_savings_bytes                |
      (ratio=None disables compaction)                                |
      |                                                               |
      +-- yes --------------------------------------------------------+
      |                                                               |
      no                                                              |
      |                                                               |
    Keep existing SHM                                                 |
                                                                      |
                                     Fresh destination <--------------+
                                                  |
                         item < shm_min_item_bytes? -- yes --> inline
                                                  |
                                                  no
                                                  |
                           shm_coalesce? -- yes --> Group compatible values,
                                |                   up to shm_max_coalesced_bytes
                                no                  (larger items stand alone)
                                |                         |
                         One group per item               |
                                +-------------------------+
                                                  |
                         group < shm_min_new_allocation_bytes?
                              yes --> inline      no --> fresh SHM

A view exposes both its own size and its backing storage. Useful bytes are the
sum of view sizes referencing one allocation in this microbatch, not the size
of that allocation or a count across other microbatches. The forwarding minimum
controls per-allocation transport overhead; compaction controls retained memory.
Forwarding can be worthwhile below the minimum for creating a new allocation,
because it avoids allocating and filling a new buffer.

Values below the forwarding minimum and values selected for compaction use the
same fresh-destination path as private values. Disabling compaction leaves the
forwarding minimum active. Eligible values can coalesce across old allocations
and with private values.
The item minimum applies only on this path, not when keeping existing SHM; the
allocation minimum applies to each resulting group. Copies go directly into the
final destination. Other references can keep an old allocation alive.
Unsupported values use their normal serialization.
Parent-to-worker payloads use ordinary multiprocessing reducers, which forward
existing shared buffers without reapplying this policy. The final MTP queue
does not use this policy at the moment.

Strategy (single memcpy):

1. Describe a leaf payload directly, or flatten a container via
   ``optree.tree_flatten``. Extract tensors, arrays, and large bytes into
   lightweight descriptors, retaining a tree spec only for containers.
2. For each planned group (plus ``torch.uint8`` for raw bytes), allocate
   storage directly in shared memory and attach a single 1-D ``torch.Tensor``
   to it, then write the values into the shared buffer. Each byte of
   real data selected for a fresh allocation is copied once — straight into SHM.
3. Wrap the skeleton + shared buffers in a ``MicrobatchEnvelope`` whose
   ``__reduce__`` produces ``list[StreamItem]`` with ``LazyPayload``
   wrappers.  Payloads are restored only when explicitly resolved
   (typically in the worker process before ``process_many``).

.. rubric:: Future: torch-free support

This module currently requires torch for SHM lifecycle management.  A
future refactor can replace this with ``shm_open`` / ``mmap`` / ``DupFd``
to remove the torch dependency entirely and support Python 3.10+ without
the ``__buffer__`` protocol.
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
DEFAULT_SHM_MIN_ITEM_BYTES = 4096
DEFAULT_SHM_MIN_NEW_ALLOCATION_BYTES = 1024 * 1024
DEFAULT_SHM_MIN_FORWARD_BYTES = 128 * 1024
DEFAULT_SHM_COMPACT_ABOVE_RATIO = 8.0
DEFAULT_SHM_COMPACT_MIN_SAVINGS_BYTES = 16 * 1024 * 1024
DEFAULT_SHM_MAX_COALESCED_BYTES = 64 * 1024 * 1024

_TORCH_COPY_LOOP_MIN_MEAN_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True, kw_only=True)
class PayloadMemoryPolicy:
    """Choose transport, retention, and coalescing independently of payload type.

    ``min_forward_bytes`` applies to the sum of view sizes per existing
    allocation in this microbatch. Views below it or selected for compaction
    join private values on the fresh-destination path. ``min_item_bytes``
    selects eligible values there, and ``min_new_allocation_bytes`` applies
    to each planned destination after grouping compatible values.
    References outside this message can delay reclamation after compaction.
    """

    min_item_bytes: int = DEFAULT_SHM_MIN_ITEM_BYTES
    min_new_allocation_bytes: int = DEFAULT_SHM_MIN_NEW_ALLOCATION_BYTES
    min_forward_bytes: int = DEFAULT_SHM_MIN_FORWARD_BYTES
    coalesce: bool = True
    max_coalesced_bytes: int | None = DEFAULT_SHM_MAX_COALESCED_BYTES
    compact_above_ratio: float | None = DEFAULT_SHM_COMPACT_ABOVE_RATIO
    compact_min_savings_bytes: int = DEFAULT_SHM_COMPACT_MIN_SAVINGS_BYTES

    def __post_init__(self) -> None:
        for name in (
            "min_item_bytes",
            "compact_min_savings_bytes",
            "min_new_allocation_bytes",
            "min_forward_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"shm_{name} must be a non-negative integer")
        if type(self.coalesce) is not bool:
            raise ValueError("shm_coalesce must be a boolean")
        ratio = self.compact_above_ratio
        if ratio is not None and (
            type(ratio) not in (int, float) or not 1 <= ratio < float("inf")
        ):
            raise ValueError("shm_compact_above_ratio must be finite and >= 1, or None")
        cap = self.max_coalesced_bytes
        if cap is not None and (type(cap) is not int or cap <= 0):
            raise ValueError(
                "shm_max_coalesced_bytes must be a positive integer, or None"
            )

    def shared_action(
        self, useful_bytes: int, backing_bytes: int
    ) -> Literal["keep", "fresh"]:
        """Keep this backing or pass its views through the fresh-destination rules."""
        if useful_bytes < self.min_forward_bytes:
            return "fresh"
        if (
            self.compact_above_ratio is not None
            and backing_bytes > useful_bytes * self.compact_above_ratio
            and backing_bytes - useful_bytes >= self.compact_min_savings_bytes
        ):
            return "fresh"
        return "keep"


DEFAULT_PAYLOAD_MEMORY_POLICY = PayloadMemoryPolicy()


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

    def bind(self, buffers: dict[str, Any]) -> Any:
        """Attach shared buffers without materializing the payload."""
        raise NotImplementedError


class _LeafDescriptor(_PayloadDescriptor):
    """One extracted value, usable as a tree leaf or an entire payload."""

    __slots__ = ()

    def buffer_keys(self) -> tuple[str, ...]:
        """Shared allocations this value needs; inline values need none."""
        return ()

    def bind(self, buffers: dict[str, Any]) -> LazyPayload:
        """Keep a root leaf lazy without retaining unrelated buffers."""
        return _ShmLeafPayload(self, _referenced_buffers([self], buffers))

    def restore(self, buffers: dict[str, Any]) -> Any:
        """Restore this value using its shared buffers."""
        raise NotImplementedError


@dataclass(slots=True)
class _TensorSlot(_LeafDescriptor):
    """Lightweight stand-in for a tensor view into a new or existing SHM buffer."""

    dtype_key: str  # str(tensor.dtype)
    offset: int  # element offset into the per-dtype buffer
    shape: tuple[int, ...]
    buffer_key: str | None = None
    requires_grad: bool = False

    def buffer_keys(self) -> tuple[str, ...]:
        return (self.buffer_key or self.dtype_key,)

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
    """Lightweight stand-in for a NumPy view into a new or existing SHM buffer."""

    dtype_key: str  # _NDARRAY_PREFIX + str(ndarray.dtype)
    offset: int  # element offset into the per-dtype buffer
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    buffer_key: str | None = None
    writable: bool = True

    def buffer_keys(self) -> tuple[str, ...]:
        return (self.buffer_key or self.dtype_key,)

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
    array = buffer.numpy()
    # numpy() creates a detached Tensor owner; attributes on buffer do not carry
    # over. Tag that owner so ordinary NumPy views retain Zephon's provenance.
    array.base._zephon_shm = True
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
        # Unwrap the value after the wrapper's protocol-aware reduction runs.
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

    def buffer_keys(self) -> tuple[str, ...]:
        return (self.buffer_key or self.dtype_key,)

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
    """Lightweight stand-in for a bytes view into a new or existing uint8 buffer."""

    offset: int  # byte offset into the _BYTES_DTYPE_KEY buffer
    length: int
    buffer_key: str | None = None

    def buffer_keys(self) -> tuple[str, ...]:
        return (self.buffer_key or _BYTES_DTYPE_KEY,)

    def __reduce__(self) -> tuple:
        return type(self), (self.offset, self.length, self.buffer_key)

    def restore(self, buffers: dict[str, Any]) -> Any:
        """Restore a bytes view backed by the shared uint8 buffer."""
        buf = buffers[self.buffer_key or _BYTES_DTYPE_KEY]
        if _ShmBytes is _ShmBytes312:
            return _ShmBytes312(buf, self.offset, self.length)
        return _ShmBytesLegacy(
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

    def buffer_keys(self) -> tuple[str, ...]:
        return tuple(
            key
            for slot in self.inner_skeleton.slots
            if isinstance(slot, _LeafDescriptor)
            for key in slot.buffer_keys()
        )

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
        # Serialize final values directly, without reducing each planning wrapper.
        slots = []
        for slot in self.slots:
            if isinstance(slot, _BufferPayload):
                if slot.transport_value is None:
                    raise RuntimeError(
                        "Buffer payload was serialized before preparation finished"
                    )
                slot = slot.transport_value
            slots.append(slot)
        return type(self), (slots, self.spec)

    def bind(self, buffers: dict[str, Any]) -> LazyPayload:
        """Keep the existing tree lazy payload without retaining this wrapper."""
        return ShmLazyPayload(
            self.slots, self.spec, _referenced_buffers(self.slots, buffers)
        )


# ---------------------------------------------------------------------------
# SHM-backed bytes view
# ---------------------------------------------------------------------------
def _assign_byte_slice(view: memoryview, key: slice, value: Any) -> None:
    """Validate a slice replacement with bytearray before writing into the buffer."""
    replacement = bytearray()
    # Slice assignment rejects scalar lengths, unlike bytearray(value).
    replacement[:] = value
    view[key] = replacement


class _ShmBytes312:
    """Zero-copy bytes view backed by a SHM uint8 tensor.

    Supports index and same-length slice assignment, without resizing.
    """

    __slots__ = ("_buffer", "_offset", "_np_view", "_write_view")

    def __init__(
        self,
        buffer: Any,
        offset: int = 0,
        length: int | None = None,
    ) -> None:
        self._buffer = buffer
        self._offset = offset
        end = None if length is None else offset + length
        self._np_view = _shared_numpy_view(buffer)[offset:end]
        self._write_view: memoryview | None = None

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

    def __setitem__(self, key: Any, value: Any) -> None:
        view = self._write_view
        if view is None:
            # Memoryview validates byte ranges and slice lengths without NumPy
            # broadcasting or integer coercion. Cache it only when used.
            self._write_view = view = memoryview(self._np_view)
        try:
            view[key] = value
        except (TypeError, ValueError):
            if not isinstance(key, slice):
                raise
            _assign_byte_slice(view, key, value)

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
    return _ShmBytes312(buffer, offset, length)


class _ShmBytesLegacy(bytearray):
    """Writable, fixed-size fallback for Python < 3.12."""

    __slots__ = ()

    def __init__(self, tensor_view: Any) -> None:
        if hasattr(tensor_view, "numpy"):
            tensor_view = tensor_view.numpy()
        super().__init__(tensor_view)

    def __setitem__(self, key: Any, value: Any) -> None:
        view = memoryview(self)
        try:
            view[key] = value
        except (TypeError, ValueError):
            if not isinstance(key, slice):
                raise
            _assign_byte_slice(view, key, value)

    def _reject_resize(self, *args: Any, **kwargs: Any) -> Any:
        raise BufferError("Cannot resize a fixed-size bytes view")

    append = clear = extend = insert = pop = remove = _reject_resize
    __delitem__ = __iadd__ = __imul__ = _reject_resize


def _make_shm_bytes_class(
    version_info: tuple[int, int] | tuple[int, int, int],
) -> type:
    """Build the platform/version-specific ``_ShmBytes`` implementation."""
    return _ShmBytes312 if version_info >= (3, 12) else _ShmBytesLegacy


_ShmBytes = _make_shm_bytes_class(sys.version_info[:3])


# ---------------------------------------------------------------------------
# pytree-based leaf collection
# ---------------------------------------------------------------------------
_PRIMITIVE_TYPES = (int, float, str, bool, type(None))


def _is_leaf(obj: Any) -> bool:
    """Predicate for ``optree.tree_flatten``.

    Treats structured types (dataclass, pydantic) as leaves. These are handled
    in ``_PayloadPreparation._collect_leaf``, which decomposes them into a fields
    dict and recurses via ``_collect_tree``. Ordinary dicts, lists, and tuples
    use optree's container traversal.

    Lists and tuples of primitives (int, float, str, …) are also treated as leaves
    to avoid inflating the slot count.  A ``List[int]`` with 500 elements
    would otherwise become 500 individual int leaves — catastrophic for
    pickle size and restore time at high DOP. We check the first and
    last element as an O(1) heuristic. Numeric lists selected for SHM are
    checked for homogeneity before conversion. Other sequences use ordinary
    pickle, including any buffer values between primitive endpoints.
    """
    if type(obj) in _PRIMITIVE_TYPES or type(obj) is dict:
        return False
    if _is_struct(obj):
        return True
    # Tuple subclasses can customize construction and pickling. Keep their
    # existing reduction rather than reconstructing them through optree.
    if isinstance(obj, tuple) and type(obj) is not tuple:
        return True
    if (
        isinstance(obj, (list, tuple))
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
    tensor = torch.frombuffer(data, dtype=dtype)
    if len(shape) != 1:
        tensor = tensor.reshape(shape)
    return tensor.requires_grad_(True) if grad else tensor


class _InlineTensor(_LeafDescriptor):
    """Serialize tensor values without invoking Torch's SHM reducer."""

    __slots__ = ("tensor",)

    def __init__(self, tensor: Any) -> None:
        self.tensor = tensor

    def restore(self, buffers: dict[str, Any]) -> Any:
        return self.tensor

    def __reduce_ex__(self, protocol: int) -> Any:
        tensor = self.tensor
        if tensor.requires_grad:
            tensor = tensor.detach()
        tensor = tensor.resolve_conj().resolve_neg().contiguous()
        if tensor.ndim != 1:
            tensor = tensor.reshape(-1)
        if tensor.stride(0) != 1:
            # Empty/singleton tensors can be contiguous with a non-unit stride,
            # but dtype reinterpretation still requires a unit last stride.
            tensor = tensor.as_strided((tensor.numel(),), (1,))
        view = tensor.view(_get_torch().uint8).numpy()
        # Protocol 4 reduces bytearray through another bytes copy. Send bytes
        # directly; restoration gives the tensor writable storage.
        data = pickle.PickleBuffer(view) if protocol >= 5 else view.tobytes()
        return _InlineTensorData, (
            data,
            tensor.dtype,
            tuple(self.tensor.shape),
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


class _InlineNumpy(_LeafDescriptor):
    """Keep NumPy's inline representation and writeability across forwarding hops."""

    __slots__ = ("array", "writable")

    def __init__(self, array: Any, writable: bool | None = None) -> None:
        self.array = array
        self.writable = array.flags.writeable if writable is None else writable

    def __reduce_ex__(self, protocol: int) -> Any:
        return type(self), (_NumpyPickleFallback(self.array), self.writable)

    def restore(self, buffers: dict[str, Any]) -> Any:
        if not self.writable:
            self.array.flags.writeable = False
        return self.array


@dataclass(slots=True)
class _BufferPayload:
    """Buffer candidate shared by a record description and its preparation.

    ``source`` holds the original value; ``slot`` describes its shape and dtype
    and receives an offset if SHM is selected. ``copy_source`` optionally holds
    a numeric list converted for the final write. Destination preparation fills
    ``transport_value`` with the original value, an inline descriptor, or the
    shared slot, then releases the source references. Record replacement and
    skeleton reduction substitute that value for this temporary candidate.

    ``source_is_shared`` records where the source lives. ``forwardable_storage``
    is its backing only when forwarding preserves its values; shared Torch
    conjugate/negative views need copying and have no forwardable storage.
    """

    source: Any
    slot: Any
    dtype_key: str
    numel: int
    nbytes: int
    source_is_shared: bool = False
    forwardable_storage: Any = None
    transport_value: Any = None
    copy_source: Any = None

    def __reduce_ex__(self, protocol: int) -> Any:
        raise RuntimeError("Buffer payload was serialized before preparation finished")


def _inline_buffer(source: Any, slot: Any, shared: bool) -> Any:
    if isinstance(slot, _TensorSlot):
        return _InlineTensor(source)
    if isinstance(slot, _NdarraySlot) and (shared or not source.flags.writeable):
        return _InlineNumpy(source)
    if isinstance(source, memoryview):
        return source.tobytes() if source.readonly else bytearray(source)
    if isinstance(source, _ShmBytes312):
        return bytearray(source._np_view)
    return source


def _group_fresh_destinations(
    values: list[_BufferPayload], policy: PayloadMemoryPolicy
) -> list[list[_BufferPayload]]:
    """Group fresh destinations by dtype and cap; small groups later go inline."""
    groups: list[list[_BufferPayload]] = []
    current: dict[str, int] = {}
    sizes: list[int] = []
    limit = policy.max_coalesced_bytes
    for value in values:
        if not policy.coalesce or (limit is not None and value.nbytes > limit):
            groups.append([value])
            sizes.append(value.nbytes)
            continue
        index = current.get(value.dtype_key)
        if index is None or (limit is not None and sizes[index] + value.nbytes > limit):
            index = current[value.dtype_key] = len(groups)
            groups.append([])
            sizes.append(0)
        groups[index].append(value)
        sizes[index] += value.nbytes
    return groups


class _PayloadPreparation:
    """Collect payloads, prepare their destinations, then replace record payloads.

    Collection owns the candidate descriptions used in both the record trees
    and the pending buffer list. Destination preparation fills their results;
    only ``replace_record_payloads`` changes the original records.
    """

    def __init__(self, policy: PayloadMemoryPolicy) -> None:
        self.policy = policy
        self._pending: list[_BufferPayload] = []
        self.buffers: dict[str, Any] = {}
        self.reused_keys: dict[tuple[int, str], str] = {}
        self._needs_payload_replacement = False
        self._record_payloads: list[tuple[SampleRecord, Any]] = []

    @classmethod
    def collect(
        cls, items: list[StreamItem], policy: PayloadMemoryPolicy
    ) -> _PayloadPreparation:
        """Collect candidates and record descriptions without replacing payloads."""
        preparation = cls(policy)
        for item in items:
            if isinstance(item, SampleRecord):
                preparation._record_payloads.append(
                    (item, preparation._collect_record_payload(item.payload))
                )
            elif isinstance(item, SampleBatch):
                for record in item.records:
                    preparation._record_payloads.append(
                        (record, preparation._collect_record_payload(record.payload))
                    )
        return preparation

    def prepare_destinations(self) -> dict[str, Any]:
        """Select backing, validate fresh values, then allocate each final group.

        Private values and views not kept in place take the same grouping path.
        Sources remain available for optional-allocation fallback until all
        results are chosen; they are then released from the descriptions.
        """
        fresh = self._retain_shared_or_collect_fresh()
        fresh = self._convert_eligible_numeric_lists(fresh)
        for values in _group_fresh_destinations(fresh, self.policy):
            useful = sum(value.nbytes for value in values)
            if useful < self.policy.min_new_allocation_bytes:
                for value in values:
                    value.transport_value = self._inline_value(value)
                continue
            value = values[0]
            if (
                len(values) == 1
                and not value.source_is_shared
                and isinstance(value.slot, _TensorSlot)
                and value.source.is_contiguous()
                and not value.source.is_conj()
                and not value.source.is_neg()
                and value.source.untyped_storage().nbytes() == value.nbytes
            ):
                # One whole-storage output tensor needs no coalescing;
                # Torch's reducer moves its storage into SHM.
                value.transport_value = value.source
            else:
                self._allocate_and_copy_group(values)
        for value in self._pending:
            value.source = None
            value.copy_source = None
        return self.buffers

    def replace_record_payloads(self) -> None:
        """Install completed descriptions if needed; keep each record's metadata."""
        if self._needs_payload_replacement:
            for record, payload in self._record_payloads:
                record.payload = (
                    payload.transport_value
                    if isinstance(payload, _BufferPayload)
                    else payload
                )

    def _collect_leaf(self, leaf: Any) -> Any:
        """Replace a single tensor/ndarray/bytes/struct leaf with a descriptor.

        Returns a descriptor if the leaf needs preparation, or the original leaf
        unchanged. Structured types (dataclass, pydantic) are decomposed into a
        fields dict and recursively extracted via ``_collect_tree``.
        """
        if type(leaf) in _PRIMITIVE_TYPES:
            return leaf
        if _is_struct(leaf):
            cls, _names, fields = _struct_to_dict(leaf)
            inner = self._collect_tree(fields)
            return _StructSlot(cls=cls, inner_skeleton=inner)
        value = _describe_buffer(leaf)
        if value is None:
            return leaf
        self._pending.append(value)
        return value

    def _collect_tree(self, payload: Any) -> _FlatSkeleton:
        """Flatten payload via optree and replace tensor/bytes leaves with slots."""
        leaves, spec = optree.tree_flatten(payload, is_leaf=_is_leaf)
        for i, leaf in enumerate(leaves):
            leaves[i] = self._collect_leaf(leaf)
        return _FlatSkeleton(slots=leaves, spec=spec)

    def _collect_record_payload(self, payload: Any) -> Any:
        """Describe a root leaf directly; preserve the structure of containers."""
        if isinstance(payload, LazyPayload):
            return payload
        if not isinstance(payload, (dict, list, tuple)) or _is_leaf(payload):
            return self._collect_leaf(payload)
        return self._collect_tree(payload)

    def _inline_value(self, value: _BufferPayload) -> Any:
        """Choose ordinary serialization or a descriptor that bypasses SHM reduction."""
        result = _inline_buffer(value.source, value.slot, value.source_is_shared)
        if result is not value.source:
            self._needs_payload_replacement = True
        return result

    def _retain_shared_or_collect_fresh(self) -> list[_BufferPayload]:
        """Select existing backing; return private and remaining shared values to group."""
        fresh: list[_BufferPayload] = []
        shared: dict[int, list[_BufferPayload]] = {}
        for value in self._pending:
            if value.nbytes == 0 or (
                value.forwardable_storage is None
                and value.nbytes < self.policy.min_item_bytes
            ):
                value.transport_value = self._inline_value(value)
            elif value.forwardable_storage is None:
                fresh.append(value)
            else:
                shared.setdefault(value.forwardable_storage._cdata, []).append(value)
        for values in shared.values():
            useful = sum(value.nbytes for value in values)
            action = self.policy.shared_action(
                useful, values[0].forwardable_storage.nbytes()
            )
            for value in values:
                if action == "keep":
                    value.transport_value = (
                        self._reference_shared_storage(value)
                        if len(values) > 1
                        else value.source
                    )
                    if (
                        isinstance(value.transport_value, np.ndarray)
                        and type(value.transport_value) is not np.ndarray
                    ):
                        value.transport_value = np.asarray(value.transport_value)
                        self._needs_payload_replacement = True
                elif value.nbytes < self.policy.min_item_bytes:
                    value.transport_value = self._inline_value(value)
                else:
                    fresh.append(value)
        return fresh

    def _convert_eligible_numeric_lists(
        self, values: list[_BufferPayload]
    ) -> list[_BufferPayload]:
        """Exclude lists that cannot use lossless SHM before counting final groups.

        Lists below the allocation minimum need no conversion or homogeneity
        scan. For potential SHM groups, retain the converted source so it is
        not converted a second time during the final write.
        """
        sizes: dict[str, int] = {}
        for value in values:
            sizes[value.dtype_key] = sizes.get(value.dtype_key, 0) + value.nbytes
        eligible: list[_BufferPayload] = []
        for value in values:
            if (
                isinstance(value.slot, _NumericListSlot)
                and sizes[value.dtype_key] >= self.policy.min_new_allocation_bytes
            ):
                source = value.source
                kind = type(source[0])
                if set(map(type, source)) != {kind}:
                    value.transport_value = source
                    continue
                try:
                    array = np.array(
                        source, dtype=np.int64 if kind is int else np.float64
                    )
                except OverflowError:
                    value.transport_value = source
                    continue
                value.copy_source = _get_torch().from_numpy(array)
            eligible.append(value)
        return eligible

    def _reference_shared_storage(self, value: _BufferPayload) -> Any:
        """Share one descriptor buffer across contiguous views of the same storage.

        Strided or unaligned views keep their existing reducer, which preserves
        the view's layout without copying into another allocation.
        """
        leaf, storage = value.source, value.forwardable_storage
        if isinstance(value.slot, _TensorSlot):
            if not leaf.is_contiguous():
                return leaf
            offset, dtype = leaf.storage_offset(), leaf.dtype
        elif isinstance(value.slot, _NdarraySlot):
            offset = leaf.__array_interface__["data"][0] - storage.data_ptr()
            if (
                not leaf.flags.c_contiguous
                or offset % leaf.itemsize
                or storage.nbytes() % leaf.itemsize
            ):
                return np.asarray(leaf)
            offset //= leaf.itemsize
            dtype = _torch_dtype_for(leaf.dtype)
        elif isinstance(value.slot, _BytesSlot):
            # Byte wrappers use contiguous views of whole-storage buffers.
            assert leaf._np_view.flags.c_contiguous
            assert leaf._buffer.storage_offset() == 0
            offset = leaf._offset
            dtype = _get_torch().uint8
        else:
            return leaf
        identity = storage._cdata, value.dtype_key
        key = self.reused_keys.get(identity)
        if key is None:
            key = self.reused_keys[identity] = f"reuse:{len(self.buffers)}"
            self.buffers[key] = (
                _get_torch().empty(0, dtype=dtype, device="cpu").set_(storage)
            )
        value.slot.buffer_key = key
        value.slot.offset = offset
        self._needs_payload_replacement = True
        return value.slot

    def _allocate_and_copy_group(self, values: list[_BufferPayload]) -> None:
        """Write a group once, or retain each source independently on SHM exhaustion."""
        items: list[Any] = []
        for value in values:
            source = value.source
            if value.copy_source is not None:
                source = value.copy_source
            elif isinstance(value.slot, _BytesSlot):
                view = source._np_view if isinstance(source, _ShmBytes312) else source
                source = (
                    _memoryview_source(view)
                    if isinstance(view, memoryview)
                    else np.frombuffer(view, dtype=np.uint8)
                )
            elif isinstance(value.slot, _TensorSlot) and source.requires_grad:
                source = source.detach()
            items.append(source)
        dtype = values[0].dtype_key
        optional = any(value.source_is_shared for value in values)
        try:
            buffer = _build_shm_buffer(items, dtype, retry=not optional)
        except Exception as exc:
            if not optional or not is_shm_error(exc):
                raise
            for value in values:
                value.transport_value = (
                    value.source
                    if value.forwardable_storage is not None
                    else self._inline_value(value)
                )
            return
        key = dtype if dtype not in self.buffers else f"{dtype}#{len(self.buffers)}"
        self.buffers[key] = buffer
        offset = 0
        for value in values:
            value.slot.buffer_key = key
            value.slot.offset = offset
            offset += value.numel
            value.transport_value = value.slot
        self._needs_payload_replacement = True


def _describe_buffer(leaf: Any) -> _BufferPayload | None:
    """Adapt supported CPU payloads to descriptors and direct-copy sources."""
    torch = _get_torch()
    if isinstance(leaf, torch.Tensor):
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
        # Torch's reducer drops the conjugate/negative view bits. These views
        # must copy, including inline fallback if a fresh SHM allocation fails.
        forwardable = shared and not (leaf.is_conj() or leaf.is_neg())
        return _BufferPayload(
            leaf,
            _TensorSlot(dtype, 0, tuple(leaf.shape), requires_grad=leaf.requires_grad),
            dtype,
            numel,
            numel * leaf.element_size(),
            shared,
            storage if forwardable else None,
        )
    if isinstance(leaf, np.ndarray):
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
            storage is not None,
            storage,
        )
    if isinstance(leaf, (bytes, bytearray, memoryview, _ShmBytes312)):
        storage = (
            leaf._buffer.untyped_storage() if isinstance(leaf, _ShmBytes312) else None
        )
        size = leaf.nbytes if isinstance(leaf, memoryview) else len(leaf)
        return _BufferPayload(
            leaf,
            _BytesSlot(0, size),
            _BYTES_DTYPE_KEY,
            size,
            size,
            storage is not None,
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


def _build_shm_buffer(items: list[Any], dtype_key: str, *, retry: bool = True) -> Any:
    """Concatenate collected tensors/bytes of one dtype into a SHM-backed tensor.

    Allocates the target directly in shared memory, then writes tensors,
    strided NumPy arrays, and supported memoryviews into the shared region.
    Numeric lists are converted before allocation.

    Required allocations retry on SHM exhaustion via ``_alloc_shm_buffer``.
    Replacing existing SHM is optional and returns the error for per-value fallback.
    """
    torch = _get_torch()

    def allocate(count: int, dtype: Any) -> Any:
        return _alloc_shm_buffer(count, dtype, f"coalesce[{dtype_key}]", retry=retry)

    if dtype_key == _BYTES_DTYPE_KEY:
        # Pack raw bytes into a uint8 tensor.
        total = sum(array.nbytes for array in items)
        buf = allocate(total, torch.uint8)
        np_buf = buf.numpy()
        if len(items) > 1 and all(a.dtype == np.uint8 for a in items):
            np.concatenate(items, axis=None, out=np_buf, casting="no")
            return buf
        offset = 0
        for array in items:
            n = array.nbytes
            np.copyto(
                np_buf[offset : offset + n].view(array.dtype).reshape(array.shape),
                array,
                casting="no",
            )
            offset += n
        return buf
    elif dtype_key.startswith(_NDARRAY_PREFIX):
        # NumPy can read strided, reversed, and read-only sources directly
        # into the final contiguous destination. No ascontiguousarray copy
        # or per-array Torch wrapper is needed.
        total_numel = sum(array.size for array in items)
        if total_numel == 0:
            return None
        dtype = _torch_dtype_for(items[0].dtype)
        buf = allocate(total_numel, dtype)
        np_buf = buf.numpy()
        if len(items) > 1:
            # concatenate(axis=None, out=...) copies strided sources
            # directly into the destination in C order. It avoids
            # creating a Python destination view for every source.
            np.concatenate(items, axis=None, out=np_buf, casting="no")
            return buf
        array = items[0]
        np.copyto(np_buf.reshape(array.shape), array, casting="no")
        return buf
    else:
        total_numel = sum(t.numel() for t in items)
        if total_numel == 0:
            return None
        dtype = items[0].dtype
        # Allocate directly in SHM, then write sub-tensors in.
        buf = allocate(total_numel, dtype)
        # cat amortizes per-tensor overhead for small copies. The copy_ threshold
        # is tuned for large first writes into fresh SHM on Linux.
        if (
            len(items) > 1
            and total_numel * items[0].element_size()
            < len(items) * _TORCH_COPY_LOOP_MIN_MEAN_BYTES
            and all(t.is_contiguous() for t in items)
        ):
            flat = [t if t.ndim == 1 else t.view(-1) for t in items]
            torch.cat(flat, out=buf)
            return buf
        offset = 0
        for t in items:
            n = t.numel()
            if n > 0:
                buf.narrow(0, offset, n).view(t.shape).copy_(t)
            offset += n
        return buf


def _referenced_buffers(slots: list[Any], buffers: dict[str, Any]) -> dict[str, Any]:
    """Keep only allocations referenced by this payload's descriptors."""
    keys = {
        key
        for slot in slots
        if isinstance(slot, _LeafDescriptor)
        for key in slot.buffer_keys()
    }
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

    def resolve_payload(self) -> SamplePayload:
        """Restore the leaf using its descriptor's existing type knowledge."""
        return cast(SamplePayload, self._descriptor.restore(self._buffers))

    def __reduce__(self) -> tuple:
        """Forward the descriptor and buffers without resolving the value."""
        return (_ShmLeafPayload, (self._descriptor, self._buffers))


@dataclass(slots=True)
class ShmLazyPayload(LazyPayload):
    """Deferred payload — flat slots + pytree spec + referenced SHM buffers.

    Slots hold inline values or shared-buffer descriptors. Inline-only
    payloads have no SHM buffer refs.

    Created on unpickle. Resolved explicitly via
    :func:`resolve_lazy_payloads` in the worker process before
    ``process_many()``, or on the pump thread for accumulators that
    declare ``reads_payload = True``.
    """

    _slots: list[Any]
    _spec: optree.PyTreeSpec
    _buffers: dict[str, Any]  # shared ref keeps SHM alive via refcounting

    def resolve_payload(self) -> SamplePayload:
        """Materialize the full payload from inline values and shared-buffer slots."""
        restored = _resolve_slots(self._slots, self._buffers)
        return cast(SamplePayload, optree.tree_unflatten(self._spec, restored))

    def __reduce__(self) -> tuple:
        """Pickle without resolving — forward (slots, spec, buffers) as-is."""
        return (
            _make_shm_lazy_payload,
            (self._slots, self._spec, self._buffers),
        )


def _make_shm_lazy_payload(
    slots: list[Any],
    spec: optree.PyTreeSpec,
    buffers: dict[str, Any],
) -> ShmLazyPayload:
    """Unpickle constructor for :class:`ShmLazyPayload`."""
    return ShmLazyPayload(slots, spec, buffers)


def _reconstruct_struct(slot: _StructSlot, fields_dict: dict[str, Any]) -> Any:
    """Rebuild a structured object from its ``_StructSlot`` and resolved fields."""
    cls = slot.cls
    # Pydantic: bypass validators via model_construct.
    if hasattr(cls, "model_construct"):
        return cls.model_construct(**fields_dict)
    return cls(**fields_dict)


def _resolve_slots(slots: list[Any], buffers: dict[str, Any]) -> list[Any]:
    """Restore slot placeholders from inline data or shared buffers.

    Shared slots use views where supported; inline slots reconstruct their
    stored values. Structured slots rebuild their fields, and ordinary leaves
    pass through unchanged.
    """
    return [
        item.restore(buffers) if isinstance(item, _LeafDescriptor) else item
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


_WIRE_RECORD = 0
_WIRE_BATCH = 1
_WIRE_PASSTHROUGH = 2


def _wire_records(items: list[StreamItem]) -> list[Any]:
    """Encode records as tuples inside the process transport envelope.

    Tags distinguish records, batches, and untouched stream items.
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

    wire: list[tuple[int, Any]] = []
    for item in items:
        if type(item) is SampleRecord:
            wire.append((_WIRE_RECORD, record(item)))
        elif type(item) is SampleBatch:
            wire.append((_WIRE_BATCH, tuple(record(rec) for rec in item.records)))
        else:
            wire.append((_WIRE_PASSTHROUGH, item))
    return wire


def _reconstruct_wire_microbatch(
    wire: list[tuple[int, Any]],
    buffers: dict[str, Any],
) -> list[StreamItem]:
    items: list[StreamItem] = []
    for tag, value in wire:
        if tag == _WIRE_RECORD:
            items.append(SampleRecord(*value))
        elif tag == _WIRE_BATCH:
            items.append(SampleBatch(tuple(SampleRecord(*record) for record in value)))
        else:
            items.append(value)
    return _reconstruct_microbatch_lazy(items, buffers)


@dataclass(slots=True)
class MicrobatchEnvelope:
    """Process-transport envelope for records and their shared-buffer references.

    On unpickle (``__reduce__``), produces ``list[StreamItem]`` where
    payload descriptions become :class:`LazyPayload` instances. Existing payloads
    pass through unchanged. The envelope also provides compact record metadata
    encoding when no payload preparation is needed.
    """

    items: list[StreamItem]
    buffers: dict[str, Any]  # group/reuse key -> shared tensor covering its storage

    def __reduce__(self) -> tuple:
        return (
            _reconstruct_wire_microbatch,
            (_wire_records(self.items), self.buffers),
        )

    def __len__(self) -> int:
        return len(self.items)


def prepare_microbatch(
    items: list[StreamItem], policy: PayloadMemoryPolicy
) -> MicrobatchEnvelope:
    """Prepare worker outputs in *items* as per-dtype SHM buffers or inline values.

    Shared Torch tensors and NumPy views of Zephon buffers retain their storage
    through multiprocessing reduction, unless the policy selects a fresh
    destination. Existing lazy payloads are already prepared.

    Collection leaves the records intact. Destination preparation chooses
    buffers and inline values; only then are record payloads replaced. Values
    may travel inline, use new SHM allocations, or retain existing storage
    when optional replacement fails.
    """
    if _get_torch() is None:
        return MicrobatchEnvelope(items, {})
    preparation = _PayloadPreparation.collect(items, policy)
    buffers = preparation.prepare_destinations()
    preparation.replace_record_payloads()
    return MicrobatchEnvelope(items, buffers)


# ---------------------------------------------------------------------------
# Unpickle reconstruction (lazy)
# ---------------------------------------------------------------------------
def _reconstruct_microbatch_lazy(
    skeleton: list[StreamItem],
    buffers: dict[str, Any],
) -> list[StreamItem]:
    """Bind each payload descriptor to shared buffers without restoring it."""

    def bind(descriptor: Any) -> Any:
        if not isinstance(descriptor, _PayloadDescriptor):
            return descriptor
        return descriptor.bind(buffers)

    for item in skeleton:
        if isinstance(item, SampleRecord):
            item.payload = bind(item.payload)
        elif isinstance(item, SampleBatch):
            for rec in item.records:
                rec.payload = bind(rec.payload)
    return skeleton
