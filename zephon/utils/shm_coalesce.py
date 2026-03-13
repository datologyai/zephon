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

1. Walk the microbatch, extract every torch tensor, numpy array, and
   bytes payload above a size threshold; replace with a lightweight
   placeholder.
2. For each distinct dtype (plus ``torch.uint8`` for raw bytes), allocate
   a single 1-D ``torch.Tensor``, call ``share_memory_()`` to place it
   in ``/dev/shm`` **before** any data is written, then ``copy_`` each
   sub-tensor directly into the shared buffer.  This means each byte of
   real data is copied exactly once — straight into SHM.
3. Wrap the skeleton + shared buffers in a ``CoalescedMicrobatch`` whose
   ``__reduce__`` transparently reconstructs the original
   ``list[StreamItem]`` on unpickle (the consumer never sees the wrapper).

.. rubric:: Future: torch-free support

This module currently requires torch for SHM lifecycle management.  A
future refactor can replace this with ``shm_open`` / ``mmap`` / ``DupFd``
to remove the torch dependency entirely and support Python 3.10+ without
the ``__buffer__`` protocol.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from zephon.core.constants import SampleBatch, SampleRecord, StreamItem

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
class _BytesSlot:
    """Lightweight stand-in for a bytes payload moved to the uint8 SHM buffer."""

    offset: int  # byte offset into the _BYTES_DTYPE_KEY buffer
    length: int


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
# Walk a SamplePayload tree, extract tensors, replace with _TensorSlot
# ---------------------------------------------------------------------------
def _extract_payload(
    payload: Any,
    collector: dict[str, list[Any]],
    offsets: dict[str, int],
    shm_min_size: int,
) -> Any:
    """Recursively replace tensors and large bytes with slot placeholders.

    Mutates dicts and lists in-place to avoid allocating copies — the caller
    (worker process) discards the originals after coalescing.
    """
    torch = _get_torch()
    if torch is not None and isinstance(payload, torch.Tensor):
        if payload.device.type != "cpu":
            return payload  # leave non-CPU tensors alone
        if payload.is_shared():
            return payload  # already in SHM, skip
        dtype_key = str(payload.dtype)
        numel = payload.numel()
        offset = offsets.get(dtype_key, 0)
        slot = _TensorSlot(
            dtype_key=dtype_key, offset=offset, shape=tuple(payload.shape)
        )
        offsets[dtype_key] = offset + numel
        collector.setdefault(dtype_key, []).append(payload)
        return slot
    np = _get_numpy()
    if np is not None and torch is not None and isinstance(payload, np.ndarray):
        if payload.dtype.hasobject:
            return payload  # object dtypes can't be memcpy'd
        dtype_key = _NDARRAY_PREFIX + str(payload.dtype)
        numel = payload.size
        offset = offsets.get(dtype_key, 0)
        slot = _NdarraySlot(
            dtype_key=dtype_key,
            offset=offset,
            shape=tuple(payload.shape),
            np_dtype_str=str(payload.dtype),
        )
        offsets[dtype_key] = offset + numel
        # Convert to torch tensor for unified SHM buffer building.
        collector.setdefault(dtype_key, []).append(
            torch.from_numpy(np.ascontiguousarray(payload))
        )
        return slot
    if isinstance(payload, (bytes, memoryview)):
        nbytes = len(payload)
        if nbytes >= shm_min_size and _get_torch() is not None:
            offset = offsets.get(_BYTES_DTYPE_KEY, 0)
            slot = _BytesSlot(offset=offset, length=nbytes)
            offsets[_BYTES_DTYPE_KEY] = offset + nbytes
            collector.setdefault(_BYTES_DTYPE_KEY, []).append(
                payload if isinstance(payload, bytes) else bytes(payload)
            )
            return slot
        return payload
    if isinstance(payload, dict):
        for k, v in payload.items():
            payload[k] = _extract_payload(v, collector, offsets, shm_min_size)
        return payload
    if isinstance(payload, list):
        for i, v in enumerate(payload):
            payload[i] = _extract_payload(v, collector, offsets, shm_min_size)
        return payload
    return payload


def _extract_from_records(
    items: list[StreamItem],
    collector: dict[str, list[Any]],
    offsets: dict[str, int],
    shm_min_size: int,
) -> None:
    """Replace tensors in *items* with slot placeholders, mutating in-place."""
    for item in items:
        if isinstance(item, SampleRecord):
            item.payload = _extract_payload(
                item.payload, collector, offsets, shm_min_size
            )
        elif isinstance(item, SampleBatch):
            for rec in item.records:
                rec.payload = _extract_payload(
                    rec.payload, collector, offsets, shm_min_size
                )


# ---------------------------------------------------------------------------
# Build coalesced SHM buffers
# ---------------------------------------------------------------------------
def _build_shm_buffers(collector: dict[str, list[Any]]) -> dict[str, Any]:
    """Concatenate collected tensors/bytes per dtype into SHM-backed tensors.

    Allocates the target in ``/dev/shm`` first (``share_memory_()``), then
    copies each sub-tensor (or bytes chunk) directly into the shared
    region — **one memcpy per item**, no intermediate staging buffer.
    """
    torch = _get_torch()
    buffers: dict[str, Any] = {}
    for dtype_key, items in collector.items():
        if dtype_key == _BYTES_DTYPE_KEY:
            # Pack raw bytes into a uint8 tensor.
            total = sum(len(b) for b in items)
            if total == 0:
                continue
            buf = torch.empty(total, dtype=torch.uint8).share_memory_()
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
            buf = torch.empty(total_numel, dtype=dtype).share_memory_()
            offset = 0
            for t in items:
                n = t.numel()
                if n > 0:
                    buf.narrow(0, offset, n).view(t.shape).copy_(t)
                offset += n
            buffers[dtype_key] = buf
    return buffers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CoalescedMicrobatch:
    """Microbatch with tensors/bytes coalesced into SHM buffers.

    On unpickle (``__reduce__``), eagerly reconstructs to a plain
    ``list[StreamItem]``.  Tensor payloads become zero-copy views into
    the SHM buffers; bytes payloads become version-specific ``_ShmBytes``
    wrappers. The consumer never sees this wrapper; it transparently
    becomes the original list.
    """

    skeleton: list[StreamItem]
    buffers: dict[str, Any]  # dtype_key -> torch.Tensor in SHM

    def __reduce__(self) -> tuple:
        # Eager materialisation: unpickle produces list[StreamItem].
        # Tensors become zero-copy views (SHM-backed); bytes become
        # plain bytes objects so the result can safely be forwarded
        # through accumulators and re-pickled downstream.
        return (_reconstruct_microbatch, (self.skeleton, self.buffers))

    def __len__(self) -> int:
        return len(self.skeleton)


def coalesce_microbatch(
    items: list[StreamItem],
    shm_min_size: int = DEFAULT_SHM_MIN_SIZE,
) -> CoalescedMicrobatch | None:
    """Extract tensors from *items*, coalesce by dtype into SHM buffers.

    Returns ``None`` when no tensors or large bytes are found.
    Payloads smaller than *shm_min_size* bytes are left inline.
    """
    if _get_torch() is None:
        return None
    collector: dict[str, list[Any]] = {}
    offsets: dict[str, int] = {}
    _extract_from_records(items, collector, offsets, shm_min_size)
    if not collector:
        return None

    buffers = _build_shm_buffers(collector)
    if not buffers:
        return None
    # items have been mutated in-place: tensors/bytes replaced with slots.
    return CoalescedMicrobatch(skeleton=items, buffers=buffers)


# ---------------------------------------------------------------------------
# Unpickle reconstruction (called by pickle.loads via __reduce__)
# ---------------------------------------------------------------------------
def _restore_payload(payload: Any, buffers: dict[str, Any]) -> Any:
    """Replace slot placeholders with views into the coalesced SHM buffers.

    Tensor slots become zero-copy tensor views.  Bytes slots become
    ``_ShmBytes`` backed by the SHM tensor. On Python 3.12+ this stays
    zero-copy through the buffer protocol; older versions fall back to a
    copied ``bytes`` subclass for compatibility.
    """
    if isinstance(payload, _TensorSlot):
        numel = 1
        for s in payload.shape:
            numel *= s
        buf = buffers.get(payload.dtype_key)
        if numel == 0 or buf is None:
            torch = _get_torch()
            dtype = buf.dtype if buf is not None else torch.float32
            return torch.empty(payload.shape, dtype=dtype)
        return buf.narrow(0, payload.offset, numel).reshape(payload.shape)
    if isinstance(payload, _NdarraySlot):
        numel = 1
        for s in payload.shape:
            numel *= s
        if numel == 0:
            np = _get_numpy()
            return np.empty(payload.shape, dtype=payload.np_dtype_str)
        buf = buffers[payload.dtype_key]
        # Zero-copy: torch view → numpy view, both backed by SHM.
        return buf.narrow(0, payload.offset, numel).reshape(payload.shape).numpy()
    if isinstance(payload, _BytesSlot):
        buf = buffers[_BYTES_DTYPE_KEY]
        return _ShmBytes(buf.narrow(0, payload.offset, payload.length))
    if isinstance(payload, dict):
        return {k: _restore_payload(v, buffers) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_restore_payload(v, buffers) for v in payload]
    return payload


def _reconstruct_microbatch(
    skeleton: list[StreamItem],
    buffers: dict[str, Any],
) -> list[StreamItem]:
    """Unpickle helper — reconstruct records from skeleton + SHM buffers."""
    result: list[StreamItem] = []
    for item in skeleton:
        if isinstance(item, SampleRecord):
            result.append(
                SampleRecord(
                    meta=item.meta,
                    payload=_restore_payload(item.payload, buffers),
                )
            )
        elif isinstance(item, SampleBatch):
            records = tuple(
                SampleRecord(
                    meta=rec.meta,
                    payload=_restore_payload(rec.payload, buffers),
                )
                for rec in item.records
            )
            result.append(SampleBatch(records=records))
        else:
            result.append(item)
    return result
