# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free reader for Lance table manifests.

A Lance dataset directory holds one immutable manifest per version under
``_versions/``. The manifest lists every fragment, its data files, their
physical row counts and any deletion file, so it already is the dataset's shard
index: discovery needs two range reads and no ``pylance``.

Layout (``protos/table.proto`` in the Lance repository): the file ends with a
16-byte footer ``[manifest_pos: u64][major: u16][minor: u16]["LANC"]``; at
``manifest_pos`` sits a ``u32`` length prefix followed by the ``Manifest``
protobuf. Only the fields Zephon reads are decoded; the rest is skipped, except
where an unknown entry would change how the data is read (see
:data:`KNOWN_READER_FLAGS`).
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from typing import Iterable, Iterator

from zephon._internal.io.storage import StorageBackend

VERSIONS_DIR = "_versions"
DATA_DIR = "data"
DELETIONS_DIR = "_deletions"
MANIFEST_SUFFIX = ".manifest"

# Reader feature flags Lance defines (``Manifest.reader_feature_flags`` in
# table.proto). Lance's rule is that a reader must refuse a dataset setting a
# flag it does not know, because the flag may change how the data decodes.
# Zephon honors that, but only has to *handle* the flags that reach a plain
# positional read of a data file; the rest are recognized and ignored:
#
#   deletion files (1)        handled: rows are remapped around the deletion vector
#   stable row ids (2)        ignored: shards address rows by offset, not row id
#   deprecated v2 marker (4)  ignored: each data file carries its own version
#   table config (8)          ignored: writer/engine configuration
#   multiple base paths (16)  refused at the fragment that uses one
#   no transaction files (32) ignored: a writer-side choice
#   data overlay files (64)   refused: ignoring an overlay returns stale values
#   index covering cols (128) ignored: Zephon never reads the index section
#   mixed file versions (256) ignored: each data file carries its own version
#   fragment reuse index (512) ignored: index-section metadata
#   spilled row lineage (2048) handled: lineage-only data files are skipped
#   fragment tree (4096)      refused: fragments do not live in the manifest
#   independent cover fields (8192) ignored: index-section metadata
KNOWN_READER_FLAGS = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 2048 | 4096 | 8192
FLAG_DATA_OVERLAYS = 64
FLAG_FRAGMENT_TREE = 4096

# Field ids below zero are reserved by Lance for system columns (tombstones and
# the hidden row-lineage columns); none of them names a dataset column.
_SYSTEM_FIELD_ID = 0

_MAGIC = b"LANC"
_FOOTER_BYTES = 16
_PREFETCH_BYTES = 64 * 1024
_U64_MAX = (1 << 64) - 1
# V2 manifest names are ``{u64::MAX - version:020}.manifest`` so the newest
# version sorts first; V1 names are ``{version}.manifest``.
_V2_NAME_DIGITS = 20
_DETACHED_PREFIX = "d"
_VERSION_HINT = "latest_version_hint.json"

_BLOB_METADATA_KEY = "lance-encoding:blob"
_BLOB_EXTENSION = "lance.blob.v2"
_EXTENSION_METADATA_KEY = "ARROW:extension:name"


@dataclass(frozen=True)
class LanceField:
    """One entry of the manifest schema, a flattened pre-order field tree."""

    id: int
    parent_id: int
    name: str
    extension_name: str
    metadata_keys: frozenset[str]

    @property
    def is_top_level(self) -> bool:
        return self.parent_id < 0

    @property
    def is_blob(self) -> bool:
        """Whether this column's values live outside the fragment's data files."""
        return (
            self.extension_name == _BLOB_EXTENSION
            or _BLOB_METADATA_KEY in self.metadata_keys
        )


@dataclass(frozen=True)
class LanceDataFile:
    """A ``.lance`` file holding some of a fragment's columns.

    ``field_ids`` lists the *leaf* fields the file stores, in file order: a
    nested column contributes one entry per leaf, and a dropped column keeps
    its entry because the file itself is never rewritten.
    """

    path: str  # relative to the dataset's ``data/`` directory
    field_ids: tuple[int, ...]
    file_major_version: int
    file_minor_version: int
    file_size_bytes: int
    has_base_id: bool

    @property
    def is_legacy_v1(self) -> bool:
        """Whether this is a pre-2.0 Lance file, which has no column indices."""
        return self.file_major_version == 0 and self.file_minor_version < 3

    def holds_dataset_columns(self) -> bool:
        """Whether this file stores any dataset column rather than only system ones."""
        return any(field_id >= _SYSTEM_FIELD_ID for field_id in self.field_ids)


@dataclass(frozen=True)
class LanceDeletionFile:
    """The tombstone list for a fragment, under the dataset's ``_deletions/``."""

    basename: str
    num_deleted_rows: int
    has_base_id: bool


@dataclass(frozen=True)
class LanceFragment:
    """A horizontal slice of the dataset: physical rows ``[0, physical_rows)``."""

    id: int
    files: tuple[LanceDataFile, ...]
    physical_rows: int
    deletion: LanceDeletionFile | None
    has_overlays: bool

    @property
    def num_rows(self) -> int:
        """Rows a reader sees, with tombstoned rows removed."""
        deleted = self.deletion.num_deleted_rows if self.deletion else 0
        return self.physical_rows - deleted


@dataclass(frozen=True)
class LanceManifest:
    """The subset of a Lance ``Manifest`` that shard discovery and reads need."""

    version: int
    fields: tuple[LanceField, ...]
    fragments: tuple[LanceFragment, ...]
    reader_feature_flags: int
    has_fragment_tree: bool

    def top_level_fields(self) -> list[LanceField]:
        """Top-level columns in schema order."""
        return [field for field in self.fields if field.is_top_level]

    def column_order(self, field_ids: Iterable[int]) -> list[int | None]:
        """Reduce a data file's leaf ids to the columns they belong to, in order.

        A file stores one entry per leaf, so several consecutive leaves make up
        one nested column; a leaf whose field the schema no longer has belongs
        to a dropped column and reduces to ``None``. The result lines up with
        the file's own top-level columns, which is the only way to locate a
        column after a rename (the manifest changes, the file does not) or a
        drop (neither changes, but the schema has one column fewer).
        """
        parents = {field.id: field.parent_id for field in self.fields}
        order: list[int | None] = []
        for field_id in field_ids:
            root: int | None = field_id
            while root is not None and root in parents and parents[root] >= 0:
                root = parents[root]
            if root not in parents:
                root = None
            if not order or order[-1] != root:
                order.append(root)
        return order


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("Truncated varint in Lance manifest")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7


def _int32(value: int) -> int:
    # Negative int32 values are sign-extended to 64 bits on the wire.
    return value - (1 << 64) if value > 0x7FFFFFFF else value


def _iter_fields(buf: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    """Yield ``(field_number, wire_type, value)`` for one protobuf message."""
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        number, wire_type = key >> 3, key & 0x7
        if wire_type == 0:
            value, pos = _varint(buf, pos)
            yield number, wire_type, value
        elif wire_type == 1:
            yield number, wire_type, buf[pos : pos + 8]
            pos += 8
        elif wire_type == 2:
            length, pos = _varint(buf, pos)
            if pos + length > len(buf):
                raise ValueError("Truncated field in Lance manifest")
            yield number, wire_type, buf[pos : pos + length]
            pos += length
        elif wire_type == 5:
            yield number, wire_type, buf[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")


def _int_list(value: int | bytes) -> list[int]:
    """Decode a ``repeated int32`` entry, packed or not."""
    if isinstance(value, int):
        return [_int32(value)]
    out: list[int] = []
    pos = 0
    while pos < len(value):
        item, pos = _varint(value, pos)
        out.append(_int32(item))
    return out


def _parse_field(buf: bytes) -> LanceField:
    # Lance writes -1 for a top-level column, which proto3 always serializes,
    # so an absent parent_id is the elided zero: a child of field 0.
    id_ = parent_id = 0
    name = extension_name = ""
    metadata_keys: set[str] = set()
    metadata: dict[str, bytes] = {}
    for number, _, value in _iter_fields(buf):
        if number == 2 and isinstance(value, bytes):
            name = value.decode("utf-8")
        elif number == 3 and isinstance(value, int):
            id_ = _int32(value)
        elif number == 4 and isinstance(value, int):
            parent_id = _int32(value)
        elif number == 9 and isinstance(value, bytes):
            extension_name = value.decode("utf-8")
        elif number == 10 and isinstance(value, bytes):
            key = b""
            entry_value = b""
            for entry_number, _, entry in _iter_fields(value):
                if entry_number == 1 and isinstance(entry, bytes):
                    key = entry
                elif entry_number == 2 and isinstance(entry, bytes):
                    entry_value = entry
            if key:
                metadata_keys.add(key.decode("utf-8"))
                metadata[key.decode("utf-8")] = entry_value
    # The deprecated extension_name field moved into schema metadata.
    if not extension_name:
        extension_name = metadata.get(_EXTENSION_METADATA_KEY, b"").decode(
            "utf-8", "replace"
        )
    return LanceField(
        id=id_,
        parent_id=parent_id,
        name=name,
        extension_name=extension_name,
        metadata_keys=frozenset(metadata_keys),
    )


def _parse_data_file(buf: bytes) -> LanceDataFile:
    path = ""
    field_ids: list[int] = []
    major = minor = size = 0
    has_base_id = False
    for number, _, value in _iter_fields(buf):
        if number == 1 and isinstance(value, bytes):
            path = value.decode("utf-8")
        elif number == 2:
            field_ids.extend(_int_list(value))
        elif number == 4 and isinstance(value, int):
            major = value
        elif number == 5 and isinstance(value, int):
            minor = value
        elif number == 6 and isinstance(value, int):
            size = value
        elif number == 7:
            has_base_id = True
    return LanceDataFile(
        path=path,
        field_ids=tuple(field_ids),
        file_major_version=major,
        file_minor_version=minor,
        file_size_bytes=size,
        has_base_id=has_base_id,
    )


def _parse_deletion_file(buf: bytes, fragment_id: int) -> LanceDeletionFile:
    file_type = read_version = deletion_id = num_deleted = 0
    has_base_id = False
    for number, _, value in _iter_fields(buf):
        if number == 1 and isinstance(value, int):
            file_type = value
        elif number == 2 and isinstance(value, int):
            read_version = value
        elif number == 3 and isinstance(value, int):
            deletion_id = value
        elif number == 4 and isinstance(value, int):
            num_deleted = value
        elif number == 7:
            has_base_id = True
    # Lance derives the path rather than storing it:
    # ``_deletions/{fragment_id}-{read_version}-{id}.{arrow|bin}``.
    suffix = "bin" if file_type == 1 else "arrow"
    return LanceDeletionFile(
        basename=f"{fragment_id}-{read_version}-{deletion_id}.{suffix}",
        num_deleted_rows=num_deleted,
        has_base_id=has_base_id,
    )


def _parse_fragment(buf: bytes) -> LanceFragment:
    # Proto3 omits zero values, so fragment 0 carries no id field.
    id_ = physical_rows = 0
    files: list[LanceDataFile] = []
    deletion_buf: bytes | None = None
    has_overlays = False
    for number, _, value in _iter_fields(buf):
        if number == 1 and isinstance(value, int):
            id_ = value
        elif number == 2 and isinstance(value, bytes):
            files.append(_parse_data_file(value))
        elif number == 3 and isinstance(value, bytes):
            deletion_buf = value
        elif number == 4 and isinstance(value, int):
            physical_rows = value
        elif number == 11:
            has_overlays = True
    return LanceFragment(
        id=id_,
        files=tuple(files),
        physical_rows=physical_rows,
        deletion=(
            _parse_deletion_file(deletion_buf, id_)
            if deletion_buf is not None
            else None
        ),
        has_overlays=has_overlays,
    )


def parse_manifest(buf: bytes) -> LanceManifest:
    """Decode a ``Manifest`` protobuf message, without the file envelope."""
    version = flags = 0
    fields: list[LanceField] = []
    fragments: list[LanceFragment] = []
    has_fragment_tree = False
    for number, _, value in _iter_fields(buf):
        if number == 1 and isinstance(value, bytes):
            fields.append(_parse_field(value))
        elif number == 2 and isinstance(value, bytes):
            fragments.append(_parse_fragment(value))
        elif number == 3 and isinstance(value, int):
            version = value
        elif number == 9 and isinstance(value, int):
            flags = value
        elif number == 22:
            has_fragment_tree = True
    return LanceManifest(
        version=version,
        fields=tuple(fields),
        fragments=tuple(fragments),
        reader_feature_flags=flags,
        has_fragment_tree=has_fragment_tree,
    )


def read_manifest(storage: StorageBackend, path: str) -> LanceManifest:
    """Read and decode the manifest file at ``path`` through range reads."""
    size = int(storage.stat(path).get("size", 0))
    if size < _FOOTER_BYTES:
        raise ValueError(f"Lance manifest too small ({size} bytes): {path}")
    start = max(size - _PREFETCH_BYTES, 0)
    tail = bytes(storage.read_range(path, start, end=size))
    pos, _major, _minor, magic = struct.unpack("<QHH4s", tail[-_FOOTER_BYTES:])
    if magic != _MAGIC:
        raise ValueError(f"Not a Lance manifest (bad magic {magic!r}): {path}")
    if pos + 4 > size - _FOOTER_BYTES:
        raise ValueError(f"Corrupt Lance manifest footer: {path}")
    if pos < start:
        tail = bytes(storage.read_range(path, pos, end=size))
        start = pos
    offset = pos - start
    (length,) = struct.unpack("<I", tail[offset : offset + 4])
    body = tail[offset + 4 : offset + 4 + length]
    if len(body) != length:
        raise ValueError(f"Truncated Lance manifest: {path}")
    return parse_manifest(body)


def manifest_version(name: str) -> int | None:
    """Return the dataset version a ``_versions/`` entry names, else ``None``.

    Detached versions, staging files and the latest-version hint are skipped:
    none of them is part of the dataset's committed history.
    """
    if not name.endswith(MANIFEST_SUFFIX) or name.startswith(_DETACHED_PREFIX):
        return None
    stem = name[: -len(MANIFEST_SUFFIX)]
    if not stem.isdigit():
        return None
    if len(stem) == _V2_NAME_DIGITS:
        return _U64_MAX - int(stem)
    return int(stem)


def manifest_names(version: int) -> tuple[str, ...]:
    """Both spellings a committed manifest for ``version`` can have, newest first."""
    return (
        f"{_U64_MAX - version:0{_V2_NAME_DIGITS}d}{MANIFEST_SUFFIX}",
        f"{version}{MANIFEST_SUFFIX}",
    )


def _hinted_manifest(storage: StorageBackend, versions_dir: str) -> str | None:
    """Resolve the version hint, which Lance writes best-effort beside the manifests."""
    hint_path = os.path.join(versions_dir, _VERSION_HINT)
    try:
        with storage.open(hint_path, "r", encoding="utf-8") as handle:
            version = int(json.load(handle)["version"])
    except Exception:
        return None
    for name in manifest_names(version):
        candidate = os.path.join(versions_dir, name)
        if storage.exists(candidate):
            return candidate
    return None


def latest_manifest_path(storage: StorageBackend, dataset_root: str) -> str:
    """Return the path of the newest committed manifest under ``dataset_root``.

    Tries the version hint first so a dataset with a long version history costs
    one read rather than a full listing of ``_versions/``; the hint is advisory,
    so a miss falls back to listing.
    """
    versions_dir = os.path.join(dataset_root, VERSIONS_DIR)
    hinted = _hinted_manifest(storage, versions_dir)
    if hinted is not None:
        return hinted
    try:
        names = storage.listdir(versions_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError(f"No Lance {VERSIONS_DIR}/ under {dataset_root}") from exc
    versioned = [(v, n) for n in names if (v := manifest_version(n)) is not None]
    if not versioned:
        raise ValueError(f"No Lance manifests under {versions_dir}")
    return os.path.join(versions_dir, max(versioned)[1])


def dataset_root_of(manifest_path: str) -> str:
    """Return the dataset directory owning ``manifest_path``."""
    versions_dir = os.path.dirname(manifest_path)
    if os.path.basename(versions_dir) != VERSIONS_DIR:
        raise ValueError(
            f"Lance manifest is not under {VERSIONS_DIR}/: {manifest_path}"
        )
    return os.path.dirname(versions_dir)


def is_manifest_path(path: str) -> bool:
    """Whether ``path`` names a committed manifest inside ``_versions/``."""
    return (
        manifest_version(os.path.basename(path)) is not None
        and os.path.basename(os.path.dirname(path)) == VERSIONS_DIR
    )


def is_lance_dataset(storage: StorageBackend, root: str) -> bool:
    """Whether ``root`` is a Lance dataset directory or one of its manifests."""
    if is_manifest_path(root):
        return True
    try:
        names = storage.listdir(os.path.join(root, VERSIONS_DIR))
    except Exception:
        return False
    return any(manifest_version(name) is not None for name in names)


__all__ = [
    "DATA_DIR",
    "DELETIONS_DIR",
    "FLAG_DATA_OVERLAYS",
    "FLAG_FRAGMENT_TREE",
    "KNOWN_READER_FLAGS",
    "LanceDataFile",
    "LanceDeletionFile",
    "LanceField",
    "LanceFragment",
    "LanceManifest",
    "VERSIONS_DIR",
    "dataset_root_of",
    "is_lance_dataset",
    "is_manifest_path",
    "latest_manifest_path",
    "manifest_names",
    "manifest_version",
    "parse_manifest",
    "read_manifest",
]
