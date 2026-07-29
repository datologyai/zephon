# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Per-format split of a shard's ``extra`` into header blob / columns / per-shard blob.

``extra`` is read only at ``open_shard`` (cold), so its cost is memory, not CPU.
The default codec msgpack-serializes each shard's ``extra`` whole (sorted keys,
so byte-stable) — exact for jsonl/vortex. Formats register a custom codec to
hoist a per-dataset blob to the header once (mds ``_streaming_template``, litdata
``config``) or to columnarize known fields (litdata ``Interval``, parquet
``row_groups``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import msgpack
import numpy as np


def canonicalize(obj):
    """Recursively sort dict keys so msgpack output is byte-stable (determinism).

    Tuples become lists (msgpack has no tuple type); numpy scalars become Python
    scalars. Leaves other scalars untouched.
    """
    if isinstance(obj, dict):
        return {k: canonicalize(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [canonicalize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def packb(obj) -> bytes:
    """Deterministic msgpack pack (keys sorted recursively)."""
    packed = msgpack.packb(canonicalize(obj), use_bin_type=True)
    assert packed is not None  # packb returns None only when default raises
    return packed


def unpackb(data: bytes):
    """Msgpack unpack (``raw=False``, ``strict_map_key=False``); ``b""`` -> ``None``.

    ``b""`` is the absent-value sentinel the builder writes for missing
    hashes/extra and that empty columns decode to, so callers need not pre-guard.
    """
    if not data:
        return None
    return msgpack.unpackb(data, raw=False, strict_map_key=False)


@dataclass
class EncodedExtra:
    """Result of ``ExtraCodec.encode`` over all ``M`` shards (slot order).

    A codec only handles the keys it *optimizes*; every other key is preserved
    generically by the builder (``rest = extra - owned_keys``), so codecs cannot
    silently drop fields.

    Attributes:
        owned_keys: top-level ``extra`` keys this codec consumes and reconstructs
            (everything else flows through the generic per-shard / constant-hoist
            rest blob). The default codec owns nothing.
        header_blob: per-dataset metadata for the owned keys (msgpack), stored
            once, or ``None``.
        int_columns: name -> ``int64`` array for owned numeric fields — length-``M``
            (one per shard, e.g. litdata ``Interval``) or ragged with an offset
            column (e.g. parquet ``row_groups``).
        flags: small JSON-serializable dict recorded in the file header and
            passed back to ``decode``/``decode_header``. The built-in codecs
            leave it empty (header/column presence already carries their
            state); the builder folds ``rest_constant`` into the stored flags.
    """

    owned_keys: frozenset[str] = frozenset()
    header_blob: bytes | None = None
    int_columns: dict[str, np.ndarray] = field(default_factory=dict)
    flags: dict = field(default_factory=dict)


class ExtraCodec(Protocol):
    """Per-format encode/decode of the *owned* subset of the ``extra`` mapping."""

    def encode(
        self, metas: list[Mapping | None], num_rows: np.ndarray
    ) -> EncodedExtra: ...

    def decode_header(self, header_blob: bytes | None, flags: dict): ...

    def decode(
        self,
        slot: int,
        header_obj,
        int_cols: dict[str, np.ndarray],
        num_rows: int,
        flags: dict,
    ) -> Mapping | None: ...


class _DefaultExtraCodec:
    """Owns nothing: the builder's generic rest path stores the whole ``extra``."""

    def encode(self, metas: list[Mapping | None], num_rows: np.ndarray) -> EncodedExtra:
        return EncodedExtra()

    def decode_header(self, header_blob: bytes | None, flags: dict):
        return None

    def decode(
        self,
        slot: int,
        header_obj,
        int_cols: dict[str, np.ndarray],
        num_rows: int,
        flags: dict,
    ) -> Mapping | None:
        return None


_DEFAULT_CODEC: ExtraCodec = _DefaultExtraCodec()
_CODECS: dict[str, ExtraCodec] = {}


def register_extra_codec(kind: str, codec: ExtraCodec) -> None:
    """Register a per-format ``ExtraCodec`` under its format ``kind``."""
    _CODECS[kind] = codec


def get_extra_codec(kind: str) -> ExtraCodec:
    """Return the codec registered for ``kind`` or the default codec."""
    return _CODECS.get(kind, _DEFAULT_CODEC)


__all__ = [
    "EncodedExtra",
    "ExtraCodec",
    "get_extra_codec",
    "packb",
    "register_extra_codec",
    "unpackb",
]
