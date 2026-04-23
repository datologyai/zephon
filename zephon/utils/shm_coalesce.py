"""Coalesce per-microbatch tensors, numpy arrays, and large bytes into SHM buffers.

Reduces the number of POSIX SHM segments (and file descriptors) from
N-per-microbatch to K-per-microbatch where K = number of distinct
dtypes (usually 1–2).  Large ``bytes``/``memoryview`` payloads are packed
into a ``torch.uint8`` buffer so they travel through the same SHM path.

Supported payload types:

- **torch.Tensor** (CPU only) — coalesced by dtype, restored as zero-copy
  views into the SHM buffer.
- **numpy.ndarray** — converted to torch tensors for SHM transport,
  restored as numpy array views (zero-copy via ``tensor.numpy()``).
- **bytes / memoryview** — payloads above a size threshold are packed
  into a uint8 SHM buffer and restored as ``_ShmBytes`` wrappers.

Strategy (single memcpy):

1. Flatten each payload via ``optree.tree_flatten``, extract every torch
   tensor, numpy array, and bytes payload above a size threshold from
   the flat leaf list; replace with a lightweight placeholder.
2. For each distinct dtype (plus ``torch.uint8`` for raw bytes), allocate
   a single 1-D ``torch.Tensor``, call ``share_memory_()`` to place it
   in ``/dev/shm`` **before** any data is written, then ``copy_`` each
   sub-tensor directly into the shared buffer.  This means each byte of
   real data is copied exactly once — straight into SHM.
3. Wrap the skeleton + shared buffers in a ``CoalescedMicrobatch`` whose
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
import sys
from dataclasses import dataclass
from typing import Any, cast

import optree

from zephon.core.constants import SampleBatch, SampleRecord, StreamItem
from zephon.utils.shm import is_shm_error, wait_for_shm_space

# ---------------------------------------------------------------------------
# Lazy torch import
# ---------------------------------------------------------------------------
_torch: Any = None
_torch_loaded = False
_np: Any = None
_np_loaded = False


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


def _get_numpy() -> Any:
    global _np, _np_loaded  # noqa: PLW0603
    if not _np_loaded:
        try:
            import numpy as _n

            _np = _n
        except ImportError:
            pass
        _np_loaded = True
    return _np


# Default minimum payload size (bytes) worth sending through SHM.
# Below this, SHM setup overhead exceeds the copy savings.
DEFAULT_SHM_MIN_SIZE = 4096  # 4 KiB

# dtype key used for the coalesced raw-bytes buffer
_BYTES_DTYPE_KEY = "_bytes_uint8"

# Prefix for numpy array dtype keys to distinguish from torch tensor keys
_NDARRAY_PREFIX = "np:"


# ---------------------------------------------------------------------------
# Placeholders for values that were moved to coalesced storage
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _TensorSlot:
    """Lightweight stand-in for a tensor extracted into a coalesced buffer."""

    dtype_key: str  # str(tensor.dtype)
    offset: int  # element offset into the per-dtype buffer
    shape: tuple[int, ...]


@dataclass(slots=True)
class _NdarraySlot:
    """Lightweight stand-in for a numpy array extracted into a coalesced buffer."""

    dtype_key: str  # _NDARRAY_PREFIX + str(ndarray.dtype)
    offset: int  # element offset into the per-dtype buffer
    shape: tuple[int, ...]
    np_dtype_str: str  # str(ndarray.dtype) for restoring the correct numpy dtype


@dataclass(slots=True)
class _NumericListSlot:
    """Stand-in for a ``list[int]`` or ``list[float]`` promoted to a tensor for SHM.

    Same layout as ``_TensorSlot`` but restored via ``.tolist()`` so the
    caller gets back a Python list, not a tensor.
    """

    dtype_key: str
    offset: int
    length: int  # original list length (== numel for 1-D)


@dataclass(slots=True)
class _BytesSlot:
    """Lightweight stand-in for a bytes payload moved to the uint8 SHM buffer."""

    offset: int  # byte offset into the _BYTES_DTYPE_KEY buffer
    length: int


@dataclass(slots=True)
class _StructSlot:
    """Stand-in for a structured type (dataclass / pydantic) whose fields were extracted.

    Only stores the class and the inner skeleton.  ``kind`` and
    ``field_names`` are derived from the class at extraction time (via
    ``_struct_info``) and at reconstruction time (via ``hasattr`` checks)
    so they never travel through pickle.
    """

    cls: type
    inner_skeleton: "_FlatSkeleton"


# ---------------------------------------------------------------------------
# Structured type detection (dataclasses, pydantic, attrs)
# ---------------------------------------------------------------------------
def _is_pydantic_instance(obj: Any) -> bool:
    """Check if *obj* is an instance of a pydantic ``BaseModel``."""
    cls = type(obj)
    return hasattr(cls, "model_fields") and hasattr(cls, "model_construct")


def _is_attrs_instance(obj: Any) -> bool:
    """Check if *obj* is an instance of an attrs-decorated class."""
    return hasattr(type(obj), "__attrs_attrs__")


def _is_struct(obj: Any) -> bool:
    """Return ``True`` for objects we can safely decompose into named fields."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return True
    if _is_pydantic_instance(obj):
        return True
    return _is_attrs_instance(obj)


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
class _FlatSkeleton:
    """Flattened payload stored on records between coalescing and pickling.

    Distinct type prevents confusion with real user payloads.
    """

    slots: list[Any]  # flat list of slots + inline leaf values
    spec: optree.PyTreeSpec  # structure recipe for tree_unflatten


# ---------------------------------------------------------------------------
# SHM-backed bytes view
# ---------------------------------------------------------------------------
class _ShmBytes312:
    """Zero-copy bytes view backed by a SHM uint8 tensor."""

    __slots__ = ("_tensor_view", "_np_view")

    def __init__(self, tensor_view: Any) -> None:
        self._tensor_view = tensor_view
        self._np_view = tensor_view.numpy()

    def __reduce_ex__(self, protocol: int) -> tuple:
        # ForkingPickler handles the tensor via SHM FD passing (zero copy).
        # Regular pickle falls back to copying tensor data inline.
        return (_ShmBytes312, (self._tensor_view,))

    def __len__(self) -> int:
        return self._tensor_view.numel()

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

    Treats tuples and structured types (dataclass, pydantic) as leaves.
    Tuples: ``optree`` flattens them by default but the walker only
    recurses into dict and list.  Structs: handled in ``_extract_leaf``
    which decomposes them into a fields dict and recurses via
    ``_extract_payload_pytree``.

    Lists of primitives (int, float, str, …) are also treated as leaves
    to avoid inflating the slot count.  A ``List[int]`` with 500 elements
    would otherwise become 500 individual int leaves — catastrophic for
    pickle size and restore time at high DOP.  We check the first and
    last element as an O(1) heuristic; typed fields (pydantic, dataclass)
    guarantee homogeneity so this is safe in practice.  If wrong, the
    only consequence is a missed tensor falling back to pickle (perf,
    not correctness).
    """
    if isinstance(obj, tuple) or _is_struct(obj):
        return True
    if (
        isinstance(obj, list)
        and len(obj) > 0
        and isinstance(obj[0], _PRIMITIVE_TYPES)
        and isinstance(obj[-1], _PRIMITIVE_TYPES)
    ):
        return True
    return False


def _extract_leaf(
    leaf: Any,
    collector: dict[str, list[Any]],
    offsets: dict[str, int],
    shm_min_size: int,
) -> Any:
    """Replace a single tensor/ndarray/bytes/struct leaf with a slot placeholder.

    Returns the slot if the leaf was extracted, or the original leaf unchanged.
    Structured types (dataclass, pydantic) are decomposed into a fields dict
    and recursively extracted via ``_extract_payload_pytree``.
    """
    # Structured types: decompose fields → dict, recurse via optree.
    if _is_struct(leaf):
        cls, _names, fields_dict = _struct_to_dict(leaf)
        inner_skel = _extract_payload_pytree(
            fields_dict, collector, offsets, shm_min_size
        )
        return _StructSlot(cls=cls, inner_skeleton=inner_skel)
    # Numeric lists: promote to tensor so they coalesce into SHM with
    # other same-dtype tensors.  Restored via .tolist() to preserve the
    # original Python list contract.
    torch = _get_torch()
    if (
        torch is not None
        and isinstance(leaf, list)
        and len(leaf) > 0
        and isinstance(leaf[0], (int, float))
        and isinstance(leaf[-1], (int, float))
    ):
        dtype = torch.int64 if isinstance(leaf[0], int) else torch.float64
        dtype_key = str(dtype)
        n = len(leaf)
        offset = offsets.get(dtype_key, 0)
        slot = _NumericListSlot(dtype_key=dtype_key, offset=offset, length=n)
        offsets[dtype_key] = offset + n
        collector.setdefault(dtype_key, []).append(torch.tensor(leaf, dtype=dtype))
        return slot
    if torch is not None and isinstance(leaf, torch.Tensor):
        if leaf.device.type != "cpu":
            return leaf
        if leaf.is_shared():
            return leaf
        dtype_key = str(leaf.dtype)
        numel = leaf.numel()
        offset = offsets.get(dtype_key, 0)
        slot = _TensorSlot(dtype_key=dtype_key, offset=offset, shape=tuple(leaf.shape))
        offsets[dtype_key] = offset + numel
        collector.setdefault(dtype_key, []).append(leaf)
        return slot
    np = _get_numpy()
    if np is not None and torch is not None and isinstance(leaf, np.ndarray):
        if leaf.dtype.hasobject:
            return leaf
        dtype_key = _NDARRAY_PREFIX + str(leaf.dtype)
        numel = leaf.size
        offset = offsets.get(dtype_key, 0)
        slot = _NdarraySlot(
            dtype_key=dtype_key,
            offset=offset,
            shape=tuple(leaf.shape),
            np_dtype_str=str(leaf.dtype),
        )
        offsets[dtype_key] = offset + numel
        collector.setdefault(dtype_key, []).append(
            torch.from_numpy(np.ascontiguousarray(leaf))
        )
        return slot
    if isinstance(leaf, (bytes, memoryview)):
        nbytes = len(leaf)
        if nbytes >= shm_min_size and _get_torch() is not None:
            offset = offsets.get(_BYTES_DTYPE_KEY, 0)
            slot = _BytesSlot(offset=offset, length=nbytes)
            offsets[_BYTES_DTYPE_KEY] = offset + nbytes
            collector.setdefault(_BYTES_DTYPE_KEY, []).append(
                leaf if isinstance(leaf, bytes) else bytes(leaf)
            )
            return slot
        return leaf
    return leaf


def _extract_payload_pytree(
    payload: Any,
    collector: dict[str, list[Any]],
    offsets: dict[str, int],
    shm_min_size: int,
) -> _FlatSkeleton:
    """Flatten payload via optree and replace tensor/bytes leaves with slots."""
    leaves, spec = optree.tree_flatten(payload, is_leaf=_is_leaf)
    for i, leaf in enumerate(leaves):
        leaves[i] = _extract_leaf(leaf, collector, offsets, shm_min_size)
    return _FlatSkeleton(slots=leaves, spec=spec)


def _extract_from_records(
    items: list[StreamItem],
    collector: dict[str, list[Any]],
    offsets: dict[str, int],
    shm_min_size: int,
) -> list[tuple[SampleRecord, _FlatSkeleton]]:
    """Extract tensors from records without mutating payloads yet.

    Returns a list of (record, skeleton) pairs.  The caller must commit
    the skeletons only after confirming that ``collector`` is non-empty.
    """
    skeletons: list[tuple[SampleRecord, _FlatSkeleton]] = []
    for item in items:
        if isinstance(item, SampleRecord):
            skel = _extract_payload_pytree(
                item.payload, collector, offsets, shm_min_size
            )
            skeletons.append((item, skel))
        elif isinstance(item, SampleBatch):
            for rec in item.records:
                skel = _extract_payload_pytree(
                    rec.payload, collector, offsets, shm_min_size
                )
                skeletons.append((rec, skel))
    return skeletons


# ---------------------------------------------------------------------------
# Build coalesced SHM buffers
# ---------------------------------------------------------------------------
def _alloc_shm_buffer(numel: int, dtype: Any, label: str) -> Any:
    """Allocate a 1-D tensor in ``/dev/shm``, retrying with backoff on ENOSPC.

    Wraps ``torch.empty(...).share_memory_()`` with the shared
    :func:`~zephon.utils.shm.wait_for_shm_space` retry so that transient
    ``/dev/shm`` exhaustion blocks instead of crashing the worker.
    """
    torch = _get_torch()
    buf = torch.empty(numel, dtype=dtype)
    while True:
        try:
            buf.share_memory_()
            return buf
        except Exception as e:
            if not is_shm_error(e):
                raise
            wait_for_shm_space(label)


def _build_shm_buffers(collector: dict[str, list[Any]]) -> dict[str, Any]:
    """Concatenate collected tensors/bytes per dtype into SHM-backed tensors.

    Allocates the target in ``/dev/shm`` first (``share_memory_()``), then
    copies each sub-tensor (or bytes chunk) directly into the shared
    region — **one memcpy per item**, no intermediate staging buffer.

    If ``/dev/shm`` is exhausted, retries with exponential backoff via
    :func:`_alloc_shm_buffer` instead of propagating the error.
    """
    buffers: dict[str, Any] = {}
    torch = _get_torch()
    for dtype_key, items in collector.items():
        if dtype_key == _BYTES_DTYPE_KEY:
            # Pack raw bytes into a uint8 tensor.
            total = sum(len(b) for b in items)
            if total == 0:
                continue
            buf = _alloc_shm_buffer(total, torch.uint8, f"coalesce[{dtype_key}]")
            # numpy view for fast memcpy from bytes into SHM
            np_buf = buf.numpy()
            offset = 0
            for b in items:
                n = len(b)
                if n > 0:
                    np_buf[offset : offset + n] = memoryview(b).cast("B")
                offset += n
            buffers[dtype_key] = buf
        else:
            total_numel = sum(t.numel() for t in items)
            if total_numel == 0:
                continue
            dtype = items[0].dtype
            # Allocate directly in SHM, then write sub-tensors in.
            buf = _alloc_shm_buffer(total_numel, dtype, f"coalesce[{dtype_key}]")
            offset = 0
            for t in items:
                n = t.numel()
                if n > 0:
                    buf.narrow(0, offset, n).view(t.shape).copy_(t)
                offset += n
            buffers[dtype_key] = buf
    return buffers


# ---------------------------------------------------------------------------
# Lazy payload (deferred restoration)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class LazyPayload:
    """Deferred payload — holds flat slots + pytree spec + SHM buffer refs.

    Created on unpickle (main process).  Resolved explicitly via
    :func:`resolve_lazy_payloads` in the worker process before
    ``process_many()``, or on the pump thread for accumulators that
    declare ``reads_payload = True``.
    """

    _slots: list[Any]
    _spec: optree.PyTreeSpec
    _buffers: dict[str, Any]  # shared ref keeps SHM alive via refcounting

    def resolve(self) -> Any:
        """Materialize the full payload by replacing slots with SHM views."""
        restored = _resolve_slots(self._slots, self._buffers)
        return optree.tree_unflatten(self._spec, restored)

    def __reduce__(self) -> tuple:
        """Pickle without resolving — forward (slots, spec, buffers) as-is."""
        return (_make_lazy_payload, (self._slots, self._spec, self._buffers))


def _make_lazy_payload(
    slots: list[Any], spec: optree.PyTreeSpec, buffers: dict[str, Any]
) -> LazyPayload:
    """Unpickle constructor for :class:`LazyPayload`."""
    return LazyPayload(slots, spec, buffers)


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
    result: list[Any] = []
    for item in slots:
        if isinstance(item, _StructSlot):
            inner_leaves = _resolve_slots(item.inner_skeleton.slots, buffers)
            fields_dict = cast(
                dict[str, Any],
                optree.tree_unflatten(item.inner_skeleton.spec, inner_leaves),
            )
            result.append(_reconstruct_struct(item, fields_dict))
            continue
        if isinstance(item, _NumericListSlot):
            buf = buffers.get(item.dtype_key)
            if item.length == 0 or buf is None:
                result.append([])
            else:
                result.append(buf.narrow(0, item.offset, item.length).tolist())
            continue
        if isinstance(item, _TensorSlot):
            numel = 1
            for s in item.shape:
                numel *= s
            buf = buffers.get(item.dtype_key)
            if numel == 0 or buf is None:
                torch = _get_torch()
                dtype = buf.dtype if buf is not None else torch.float32
                result.append(torch.empty(item.shape, dtype=dtype))
            else:
                result.append(buf.narrow(0, item.offset, numel).reshape(item.shape))
        elif isinstance(item, _NdarraySlot):
            numel = 1
            for s in item.shape:
                numel *= s
            if numel == 0:
                np = _get_numpy()
                result.append(np.empty(item.shape, dtype=item.np_dtype_str))
            else:
                buf = buffers[item.dtype_key]
                result.append(
                    buf.narrow(0, item.offset, numel).reshape(item.shape).numpy()
                )
        elif isinstance(item, _BytesSlot):
            buf = buffers[_BYTES_DTYPE_KEY]
            result.append(_ShmBytes(buf.narrow(0, item.offset, item.length)))
        else:
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CoalescedMicrobatch:
    """Microbatch with tensors/bytes coalesced into SHM buffers.

    On unpickle (``__reduce__``), produces ``list[StreamItem]`` where each
    record's payload is a :class:`LazyPayload`.  Call
    :func:`resolve_lazy_payloads` to materialize before use.
    """

    skeleton: list[StreamItem]
    buffers: dict[str, Any]  # dtype_key -> torch.Tensor in SHM

    def __reduce__(self) -> tuple:
        return (_reconstruct_microbatch_lazy, (self.skeleton, self.buffers))

    def __len__(self) -> int:
        return len(self.skeleton)


def coalesce_microbatch(
    items: list[StreamItem],
    shm_min_size: int = DEFAULT_SHM_MIN_SIZE,
) -> CoalescedMicrobatch | None:
    """Extract tensors from *items*, coalesce by dtype into SHM buffers.

    Returns ``None`` when no tensors or large bytes are found.
    Payloads smaller than *shm_min_size* bytes are left inline.

    Uses a two-pass approach: extraction first collects skeletons without
    mutating payloads, then commits only after SHM allocation succeeds.
    """
    if _get_torch() is None:
        return None
    collector: dict[str, list[Any]] = {}
    offsets: dict[str, int] = {}
    skeletons = _extract_from_records(items, collector, offsets, shm_min_size)
    if not collector:
        return None

    # Build SHM buffers first.  If a non-SHM error propagates, payloads
    # are still intact.  ENOSPC is retried internally by _build_shm_buffers.
    buffers = _build_shm_buffers(collector)
    if not buffers:
        return None

    # Commit: only now mutate record payloads to flat skeleton form.
    for rec, skel in skeletons:
        rec.payload = skel  # type: ignore[assignment]

    return CoalescedMicrobatch(skeleton=items, buffers=buffers)


# ---------------------------------------------------------------------------
# Unpickle reconstruction (lazy)
# ---------------------------------------------------------------------------
def _reconstruct_microbatch_lazy(
    skeleton: list[StreamItem],
    buffers: dict[str, Any],
) -> list[StreamItem]:
    """Unpickle helper — wrap payloads in LazyPayload instead of restoring."""
    result: list[StreamItem] = []
    for item in skeleton:
        if isinstance(item, SampleRecord):
            skel = item.payload  # _FlatSkeleton from extraction
            result.append(
                SampleRecord(
                    meta=item.meta,
                    payload=LazyPayload(skel.slots, skel.spec, buffers),  # type: ignore[union-attr,assignment]
                )
            )
        elif isinstance(item, SampleBatch):
            records = tuple(
                SampleRecord(
                    meta=rec.meta,
                    payload=LazyPayload(rec.payload.slots, rec.payload.spec, buffers),  # type: ignore[union-attr,assignment]
                )
                for rec in item.records
            )
            result.append(SampleBatch(records=records))
        else:
            result.append(item)
    return result


def resolve_lazy_payloads(items: list[Any]) -> None:
    """Resolve all :class:`LazyPayload` instances in *items* in-place.

    Call this in the worker process before ``process_many()``, or at stage
    exit boundaries to prevent lazy payloads from leaking downstream.

    No-op for records whose payloads are already materialized.
    """
    for item in items:
        if isinstance(item, SampleRecord) and isinstance(item.payload, LazyPayload):
            item.payload = item.payload.resolve()
        elif isinstance(item, SampleBatch):
            for rec in item.records:
                if isinstance(rec.payload, LazyPayload):
                    rec.payload = rec.payload.resolve()
