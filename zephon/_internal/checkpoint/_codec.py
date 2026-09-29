"""Pluggable serialisation + compression for checkpoint aggregation I/O.

During multi-rank checkpointing each rank writes its local state dict as bytes
to S3 (or shared FS), the leader reads all of them, merges, and writes the
merged result back.  These intermediate files are **ephemeral** — written and
deleted within a single ``state_dict()`` call — so no backward-compatibility
envelope is needed.  Writer and reader are always the same code version.

This module is orthogonal to the final checkpoint format: the DCP integration
layer in torchtitan continues to ``pickle.dumps`` the checkpoint dict.  The
codec only affects the wire format during aggregation.
"""

from __future__ import annotations

import json
from typing import Any, cast

import msgpack

from zephon._internal.utils.compression import require_zstd

# msgpack's int wire format tops out at int64 / uint64. numpy's PCG64
# ``bit_generator.state`` exposes ``state`` and ``inc`` as 128-bit Python
# ints, and those land in the cursor checkpoint via
# ``block_rng_snapshot["rng_state"]``. JSON encodes arbitrary-precision ints
# natively; msgpack raises ``OverflowError`` mid-pack. We side-step this by
# pre-walking the tree and wrapping any int outside [-2^63, 2^64-1] as a
# msgpack ext type, then unwrap via ``ext_hook`` on decode.
_BIGINT_EXT_CODE = 0
_MSGPACK_UINT64_MAX = (1 << 64) - 1
_MSGPACK_INT64_MIN = -(1 << 63)


def _wrap_bigints(obj: Any) -> Any:
    """Recursively wrap ints outside msgpack's native range as ``ExtType``.

    Walks dicts/lists/tuples and replaces any ``int`` that exceeds
    ``[-2^63, 2^64-1]`` with ``msgpack.ExtType`` carrying the value as
    two's-complement big-endian bytes. ``bool`` (a subclass of ``int``) is
    excluded via ``type(...) is int`` so booleans round-trip unchanged.
    """
    obj_type = type(obj)
    if obj_type is dict:
        return {_wrap_bigints(k): _wrap_bigints(v) for k, v in obj.items()}
    if obj_type is list:
        return [_wrap_bigints(v) for v in obj]
    if obj_type is tuple:
        return tuple(_wrap_bigints(v) for v in obj)
    if obj_type is int and (obj > _MSGPACK_UINT64_MAX or obj < _MSGPACK_INT64_MIN):
        # +1 bit for the sign in two's complement, then ceil to bytes.
        n_bytes = (obj.bit_length() + 8) // 8
        return msgpack.ExtType(
            _BIGINT_EXT_CODE, obj.to_bytes(n_bytes, "big", signed=True)
        )
    return obj


def _bigint_ext_hook(code: int, data: bytes) -> Any:
    """Decode our bigint ext type; pass through any unknown codes."""
    if code == _BIGINT_EXT_CODE:
        return int.from_bytes(data, "big", signed=True)
    return msgpack.ExtType(code, data)


class AggregationCodec:
    """Encode / decode checkpoint dicts for aggregation transfer.

    Parameters
    ----------
    serializer:
        ``"msgpack"`` (default, binary, faster) or ``"json"`` (text, human-readable).
    compressor:
        ``"zstd"`` (default) or ``"none"``.

    Example::

        codec = AggregationCodec()          # msgpack + zstd
        payload = codec.encode(state_dict)
        # ... write payload bytes to S3 ...
        state = codec.decode(payload)
    """

    def __init__(
        self,
        serializer: str = "msgpack",
        compressor: str = "zstd",
    ) -> None:
        if serializer not in ("json", "msgpack"):
            raise ValueError(f"Unknown serializer: {serializer!r}")
        if compressor not in ("none", "zstd"):
            raise ValueError(f"Unknown compressor: {compressor!r}")
        self._serializer = serializer
        self._compressor = compressor

    # ------------------------------------------------------------------

    def encode(self, data: dict[str, Any]) -> bytes:
        """Serialise and optionally compress *data* into bytes."""
        serialised = self._serialize(data)
        return self._compress(serialised)

    def decode(self, payload: bytes) -> dict[str, Any]:
        """Decompress and deserialise *payload* back into a dict."""
        decompressed = self._decompress(payload)
        return self._deserialize(decompressed)

    # ------------------------------------------------------------------

    def _serialize(self, data: dict[str, Any]) -> bytes:
        if self._serializer == "msgpack":
            # Pre-walk to wrap arbitrary-precision ints (e.g. numpy PCG64
            # 128-bit state) that overflow msgpack's int64 / uint64 wire
            # format. JSON encodes such ints natively, so json takes the
            # data as-is.
            return cast(bytes, msgpack.packb(_wrap_bigints(data), use_bin_type=True))
        return json.dumps(data).encode("utf-8")

    def _deserialize(self, data: bytes) -> dict[str, Any]:
        # Both msgpack.unpackb and json.loads return Any, but we only ever
        # encode dicts so the return is always dict[str, Any].
        if self._serializer == "msgpack":
            return cast(
                dict[str, Any],
                msgpack.unpackb(
                    data,
                    raw=False,
                    strict_map_key=False,
                    ext_hook=_bigint_ext_hook,
                ),
            )
        return cast(dict[str, Any], json.loads(data))

    def _compress(self, data: bytes) -> bytes:
        if self._compressor == "zstd":
            return require_zstd().compress(data, level=3)  # fast, decent ratio
        return data

    def _decompress(self, data: bytes) -> bytes:
        if self._compressor == "zstd":
            return require_zstd().decompress(data)
        return data
