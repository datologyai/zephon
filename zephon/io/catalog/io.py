# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Single-file pack/unpack, ``mmap`` load, and atomic write.

The catalog artifact for one dataset is a single, self-describing file:

    MAGIC (8 bytes)
    header_len (uint64 little-endian)
    header (JSON, utf-8, ``header_len`` bytes)
    pad to 8-byte alignment
    column region (concatenated, 8-byte aligned raw column buffers)

The header carries ``schema_version``, the small ``DatasetHeader`` fields, the
content ``fingerprint`` and a column directory mapping each column name to its
dtype, element count and byte offset (relative to the start of the column
region). Loading the file ``mmap``s it once and exposes every column as a
zero-copy ``np.frombuffer`` view, so all processes on a machine share one set of
clean, reclaimable pages.

A single file (rather than a directory of per-column buffers) makes publishing
atomic: write to a ``.tmp`` sibling, ``fsync``, then ``os.replace`` — a reader on
the lockless fast path sees either no file or the whole file, never a torn one.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from zephon.utils.atomic import atomic_write_bytes

MAGIC = b"ZEPHCAT1"
SCHEMA_VERSION = 1  # on-disk catalog format version (read by builder/handle)
_HEADER_LEN_STRUCT = struct.Struct("<Q")
_ALIGN = 8


def _align_up(value: int, alignment: int = _ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass
class LoadedCatalog:
    """Result of ``load_mmap``: the parsed header plus zero-copy column views.

    Holds the ``mmap`` object (and the backing file) alive for as long as the
    column views are referenced, so the mapped pages are not unmapped early.
    """

    header: dict
    columns: dict[str, np.ndarray]
    _file: BinaryIO | None = None
    _mmap: mmap.mmap | None = None


def pack_catalog(header_fields: dict, columns: dict[str, np.ndarray]) -> bytes:
    """Serialize ``columns`` and ``header_fields`` into the single-file layout.

    ``header_fields`` must already contain the small scalar header (format,
    root, fingerprint, ...). ``columns`` maps column name to a 1-D numpy array;
    each is stored contiguously and 8-byte aligned.
    """
    col_dir: list[dict] = []
    data = bytearray()
    for name, arr in columns.items():
        contiguous = np.ascontiguousarray(arr)
        if contiguous.ndim != 1:
            raise ValueError(f"Catalog column {name!r} must be 1-D")
        pad = _align_up(len(data)) - len(data)
        data.extend(b"\x00" * pad)
        offset = len(data)
        raw = contiguous.tobytes()
        data.extend(raw)
        col_dir.append(
            {
                "name": name,
                "dtype": contiguous.dtype.str,
                "count": int(contiguous.size),
                "offset": offset,
                "nbytes": len(raw),
            }
        )

    header = dict(header_fields)
    header["columns"] = col_dir
    header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")

    prefix_len = len(MAGIC) + _HEADER_LEN_STRUCT.size + len(header_bytes)
    pad = _align_up(prefix_len) - prefix_len

    out = bytearray()
    out.extend(MAGIC)
    out.extend(_HEADER_LEN_STRUCT.pack(len(header_bytes)))
    out.extend(header_bytes)
    out.extend(b"\x00" * pad)
    out.extend(data)
    return bytes(out)


def _data_start(header_len: int) -> int:
    return _align_up(len(MAGIC) + _HEADER_LEN_STRUCT.size + header_len)


def _parse_header(buf: mmap.mmap) -> tuple[dict, int]:
    if buf[: len(MAGIC)] != MAGIC:
        raise ValueError("Not a Zephon catalog file (bad magic)")
    start = len(MAGIC)
    (header_len,) = _HEADER_LEN_STRUCT.unpack(
        buf[start : start + _HEADER_LEN_STRUCT.size]
    )
    hstart = start + _HEADER_LEN_STRUCT.size
    header = json.loads(buf[hstart : hstart + header_len].decode("utf-8"))
    return header, _data_start(header_len)


def load_mmap(path: str | Path) -> LoadedCatalog:
    """``mmap`` the catalog file read-only and expose columns as zero-copy views."""
    fobj = open(path, "rb")
    try:
        mm = mmap.mmap(fobj.fileno(), 0, access=mmap.ACCESS_READ)
    except Exception:
        fobj.close()
        raise

    header, data_start = _parse_header(mm)
    columns: dict[str, np.ndarray] = {}
    for col in header["columns"]:
        dtype = np.dtype(col["dtype"])
        count = int(col["count"])
        if count == 0:
            columns[col["name"]] = np.empty(0, dtype=dtype)
            continue
        offset = data_start + int(col["offset"])
        columns[col["name"]] = np.frombuffer(
            mm, dtype=dtype, count=count, offset=offset
        )
    return LoadedCatalog(header=header, columns=columns, _file=fobj, _mmap=mm)


def is_valid(path: str | Path, *, schema_version: int) -> bool:
    """Cheap integrity check: magic, schema version, and column extents fit.

    Does not re-hash contents — atomic publish is the integrity guarantee and
    the fingerprint is the trusted content-address *name*.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fobj:
            head = fobj.read(len(MAGIC) + _HEADER_LEN_STRUCT.size)
            if len(head) < len(MAGIC) + _HEADER_LEN_STRUCT.size:
                return False
            if head[: len(MAGIC)] != MAGIC:
                return False
            (header_len,) = _HEADER_LEN_STRUCT.unpack(head[len(MAGIC) :])
            header = json.loads(fobj.read(header_len).decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False

    if header.get("schema_version") != schema_version:
        return False
    data_start = _data_start(header_len)
    for col in header.get("columns", []):
        off = int(col["offset"])
        nbytes = int(col["nbytes"])
        count = int(col["count"])
        if off < 0 or nbytes < 0 or count < 0 or data_start + off + nbytes > size:
            return False
    return True


def write_atomic(path: str | Path, data: bytes) -> None:
    """Atomically publish ``data`` to ``path``, ``fsync``ed before the rename."""
    atomic_write_bytes(path, data, fsync=True)


def read_pointer(pointer_path: str | Path) -> str | None:
    """Read a ``source_key -> fingerprint`` pointer file, if present."""
    try:
        text = Path(pointer_path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def write_pointer(pointer_path: str | Path, fingerprint: str) -> None:
    """Atomically record a ``source_key -> fingerprint`` pointer."""
    write_atomic(pointer_path, fingerprint.encode("utf-8"))


__all__ = [
    "LoadedCatalog",
    "MAGIC",
    "SCHEMA_VERSION",
    "is_valid",
    "load_mmap",
    "pack_catalog",
    "read_pointer",
    "write_atomic",
    "write_pointer",
]
