# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Lance manifest parsing, version naming and dataset probing without pylance."""

import struct
from pathlib import Path

import pytest

from zephon._internal.io.formats.lance_support.manifest import (
    dataset_root_of,
    is_lance_dataset,
    is_manifest_path,
    latest_manifest_path,
    manifest_names,
    manifest_version,
    parse_manifest,
    read_manifest,
)
from zephon._internal.io.storage.local import LocalFSBackend

_U64_MAX = (1 << 64) - 1


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _tag(number: int, wire: int) -> bytes:
    return _varint(number << 3 | wire)


def _uint(number: int, value: int) -> bytes:
    return _tag(number, 0) + _varint(value)


def _int32(number: int, value: int) -> bytes:
    # Protobuf sign-extends a negative int32 to 64 bits on the wire.
    return _uint(number, value + (1 << 64) if value < 0 else value)


def _bytes(number: int, payload: bytes) -> bytes:
    return _tag(number, 2) + _varint(len(payload)) + payload


def _string(number: int, value: str) -> bytes:
    return _bytes(number, value.encode("utf-8"))


def _packed_int32(number: int, values: list[int]) -> bytes:
    body = b"".join(
        _varint(value + (1 << 64) if value < 0 else value) for value in values
    )
    return _bytes(number, body)


def _field(
    field_id: int,
    name: str,
    *,
    parent_id: int | None = None,
    metadata: dict[str, bytes] | None = None,
    extension: str | None = None,
) -> bytes:
    body = _string(2, name) + _int32(3, field_id)
    if parent_id is not None:
        body += _int32(4, parent_id)
    if extension is not None:
        body += _string(9, extension)
    for key, value in (metadata or {}).items():
        body += _bytes(10, _string(1, key) + _bytes(2, value))
    return _bytes(1, body)


def _data_file(
    path: str,
    field_ids: list[int],
    *,
    major: int = 2,
    minor: int = 2,
    size: int = 0,
    base_id: int | None = None,
) -> bytes:
    body = (
        _string(1, path)
        + _packed_int32(2, field_ids)
        + _uint(4, major)
        + _uint(5, minor)
    )
    if size:
        body += _uint(6, size)
    if base_id is not None:
        body += _uint(7, base_id)
    return _bytes(2, body)


def _fragment(
    fragment_id: int,
    files: bytes,
    physical_rows: int,
    *,
    deletion: tuple[int, int, int, int] | None = None,
    overlay: bool = False,
) -> bytes:
    body = b""
    if fragment_id:
        body += _uint(1, fragment_id)
    body += files
    if deletion is not None:
        file_type, read_version, deletion_id, num_deleted = deletion
        body += _bytes(
            3,
            _uint(1, file_type)
            + _uint(2, read_version)
            + _uint(3, deletion_id)
            + _uint(4, num_deleted),
        )
    body += _uint(4, physical_rows)
    if overlay:
        body += _bytes(11, _uint(3, 1))
    return _bytes(2, body)


def _manifest_body(
    fields: bytes, fragments: bytes, version: int = 1, flags: int = 0
) -> bytes:
    body = fields + fragments + _uint(3, version)
    if flags:
        body += _uint(9, flags)
    return body


def _envelope(body: bytes, *, prefix: bytes = b"") -> bytes:
    out = bytearray(prefix)
    pos = len(out)
    out += struct.pack("<I", len(body)) + body
    out += struct.pack("<QHH4s", pos, 0, 2, b"LANC")
    return bytes(out)


def _simple_manifest() -> bytes:
    fields = _field(0, "id", parent_id=-1) + _field(1, "text", parent_id=-1)
    fragments = _fragment(0, _data_file("a.lance", [0, 1], size=11), 4) + _fragment(
        1, _data_file("b.lance", [0, 1], size=22), 6
    )
    return _manifest_body(fields, fragments, version=7)


def test_parses_fields_fragments_and_counts() -> None:
    manifest = parse_manifest(_simple_manifest())

    assert manifest.version == 7
    assert [field.name for field in manifest.top_level_fields()] == ["id", "text"]
    assert [fragment.id for fragment in manifest.fragments] == [0, 1]
    assert [fragment.num_rows for fragment in manifest.fragments] == [4, 6]
    assert manifest.fragments[0].files[0].path == "a.lance"
    assert manifest.fragments[0].files[0].file_size_bytes == 11


def test_fragment_zero_keeps_its_implicit_id() -> None:
    """Proto3 elides the default, so an absent id must read as fragment 0."""
    manifest = parse_manifest(_simple_manifest())

    assert manifest.fragments[0].id == 0


def test_deletion_file_name_is_derived_from_its_fields() -> None:
    body = _manifest_body(
        _field(0, "id", parent_id=-1),
        _fragment(3, _data_file("a.lance", [0]), 10, deletion=(0, 5, 99, 4))
        + _fragment(4, _data_file("b.lance", [0]), 10, deletion=(1, 5, 7, 2)),
        flags=1,
    )
    first, second = parse_manifest(body).fragments

    assert first.deletion is not None
    assert first.deletion.basename == "3-5-99.arrow"
    assert first.num_rows == 6
    assert second.deletion is not None
    assert second.deletion.basename == "4-5-7.bin"
    assert second.num_rows == 8


def test_an_absent_parent_id_means_a_child_of_field_zero() -> None:
    """Lance always writes -1 for a root, so the elided default is parent 0."""
    fields = _field(0, "list", parent_id=-1) + _field(1, "item")
    manifest = parse_manifest(_manifest_body(fields, b""))

    assert [field.name for field in manifest.top_level_fields()] == ["list"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metadata": {"lance-encoding:blob": b"true"}},
        {"extension": "lance.blob.v2"},
        {"metadata": {"ARROW:extension:name": b"lance.blob.v2"}},
    ],
)
def test_blob_columns_are_recognized(kwargs: dict) -> None:
    manifest = parse_manifest(
        _manifest_body(_field(0, "payload", parent_id=-1, **kwargs), b"")
    )

    assert manifest.fields[0].is_blob


def test_column_order_groups_a_nested_column_s_leaves() -> None:
    """A struct contributes one leaf per child but occupies one file column."""
    manifest = parse_manifest(
        _manifest_body(
            _field(0, "list", parent_id=-1)
            + _field(1, "item")
            + _field(2, "pair", parent_id=-1)
            + _field(3, "a", parent_id=2)
            + _field(4, "b", parent_id=2)
            + _field(5, "id", parent_id=-1),
            b"",
        )
    )

    assert manifest.column_order([1, 3, 4, 5]) == [0, 2, 5]


def test_column_order_keeps_a_slot_for_a_dropped_column() -> None:
    """Dropping a column rewrites the schema, never the files that still hold it."""
    manifest = parse_manifest(
        _manifest_body(
            _field(0, "id", parent_id=-1) + _field(2, "vec", parent_id=-1), b""
        )
    )

    assert manifest.column_order([0, 1, 2]) == [0, None, 2]


def test_column_order_keeps_adjacent_dropped_columns_apart() -> None:
    """Merging two dropped columns into one slot shifts every column after them."""
    manifest = parse_manifest(
        _manifest_body(_field(0, "a", parent_id=-1) + _field(3, "d", parent_id=-1), b"")
    )

    assert manifest.column_order([0, 1, 2, 3]) == [0, None, None, 3]


def test_system_only_file_holds_no_dataset_columns() -> None:
    manifest = parse_manifest(
        _manifest_body(
            _field(0, "id", parent_id=-1),
            _fragment(
                0,
                _data_file("a.lance", [0]) + _data_file("rowids.lance", [-3]),
                2,
            ),
        )
    )
    first, second = manifest.fragments[0].files

    assert first.holds_dataset_columns()
    assert not second.holds_dataset_columns()
    # A system field has no column of the dataset's own to occupy.
    assert manifest.column_order(second.field_ids) == [None]


def test_flags_overlays_and_fragment_tree_are_surfaced() -> None:
    body = _manifest_body(
        _field(0, "id", parent_id=-1),
        _fragment(0, _data_file("a.lance", [0]), 2, overlay=True),
        flags=64,
    )
    body += _bytes(22, _bytes(1, b""))
    manifest = parse_manifest(body)

    assert manifest.reader_feature_flags == 64
    assert manifest.fragments[0].has_overlays
    assert manifest.has_fragment_tree


def test_legacy_v1_files_are_flagged() -> None:
    manifest = parse_manifest(
        _manifest_body(
            _field(0, "id", parent_id=-1),
            _fragment(0, _data_file("a.lance", [0], major=0, minor=2), 2),
        )
    )

    assert manifest.fragments[0].files[0].is_legacy_v1


def test_base_id_marks_a_file_outside_the_dataset() -> None:
    manifest = parse_manifest(
        _manifest_body(
            _field(0, "id", parent_id=-1),
            _fragment(0, _data_file("a.lance", [0], base_id=0), 2),
        )
    )

    assert manifest.fragments[0].files[0].has_base_id


def test_read_manifest_finds_a_body_before_the_prefetch_window(tmp_path: Path) -> None:
    """A manifest larger than the tail read must fall back to a second range read."""
    body = _simple_manifest()
    path = tmp_path / "1.manifest"
    path.write_bytes(_envelope(body, prefix=b"\0" * (128 * 1024)))

    manifest = read_manifest(LocalFSBackend(root=Path("/")), str(path))

    assert [fragment.num_rows for fragment in manifest.fragments] == [4, 6]


def test_read_manifest_rejects_a_foreign_file(tmp_path: Path) -> None:
    path = tmp_path / "1.manifest"
    path.write_bytes(b"x" * 64)

    with pytest.raises(ValueError, match="bad magic"):
        read_manifest(LocalFSBackend(root=Path("/")), str(path))


@pytest.mark.parametrize(
    "name,expected",
    [
        ("1.manifest", 1),
        ("42.manifest", 42),
        (f"{_U64_MAX - 3:020d}.manifest", 3),
        ("d5-abc.manifest", None),
        ("latest_version_hint.json", None),
        ("5.manifest.tmp", None),
    ],
)
def test_manifest_version_naming(name: str, expected: int | None) -> None:
    assert manifest_version(name) == expected


def test_manifest_names_round_trip() -> None:
    for name in manifest_names(9):
        assert manifest_version(name) == 9


def test_latest_manifest_prefers_the_newest_version(tmp_path: Path) -> None:
    versions = tmp_path / "_versions"
    versions.mkdir()
    for version in (1, 2, 3):
        (versions / f"{_U64_MAX - version:020d}.manifest").write_bytes(b"")
    (versions / "d9-detached.manifest").write_bytes(b"")

    found = latest_manifest_path(LocalFSBackend(root=Path("/")), str(tmp_path))

    assert manifest_version(Path(found).name) == 3


def test_latest_manifest_uses_the_version_hint(tmp_path: Path) -> None:
    """The hint saves listing a long version history; it names an existing file."""
    versions = tmp_path / "_versions"
    versions.mkdir()
    (versions / f"{_U64_MAX - 2:020d}.manifest").write_bytes(b"")
    (versions / "latest_version_hint.json").write_text('{"version": 2}')

    found = latest_manifest_path(LocalFSBackend(root=Path("/")), str(tmp_path))

    assert Path(found).name == f"{_U64_MAX - 2:020d}.manifest"


def test_latest_manifest_walks_past_a_lagging_hint(tmp_path: Path) -> None:
    """Lance writes the hint after the manifest, so it can name a stale version."""
    versions = tmp_path / "_versions"
    versions.mkdir()
    for version in (1, 2, 3):
        (versions / f"{_U64_MAX - version:020d}.manifest").write_bytes(b"")
    (versions / "latest_version_hint.json").write_text('{"version": 1}')

    found = latest_manifest_path(LocalFSBackend(root=Path("/")), str(tmp_path))

    assert manifest_version(Path(found).name) == 3


def test_latest_manifest_ignores_a_hint_without_a_manifest(tmp_path: Path) -> None:
    versions = tmp_path / "_versions"
    versions.mkdir()
    (versions / "1.manifest").write_bytes(b"")
    (versions / "latest_version_hint.json").write_text('{"version": 99}')

    found = latest_manifest_path(LocalFSBackend(root=Path("/")), str(tmp_path))

    assert Path(found).name == "1.manifest"


def test_latest_manifest_without_versions_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="_versions"):
        latest_manifest_path(LocalFSBackend(root=Path("/")), str(tmp_path))


def test_manifest_path_and_dataset_root(tmp_path: Path) -> None:
    manifest = tmp_path / "ds.lance" / "_versions" / "1.manifest"

    assert is_manifest_path(str(manifest))
    assert dataset_root_of(str(manifest)) == str(tmp_path / "ds.lance")
    assert not is_manifest_path(str(tmp_path / "1.manifest"))


def test_dataset_root_of_rejects_a_stray_manifest() -> None:
    with pytest.raises(ValueError, match="_versions"):
        dataset_root_of("/tmp/1.manifest")


def test_is_lance_dataset_probes_the_versions_dir(tmp_path: Path) -> None:
    storage = LocalFSBackend(root=Path("/"))
    plain = tmp_path / "plain"
    plain.mkdir()
    dataset = tmp_path / "ds.lance"
    (dataset / "_versions").mkdir(parents=True)
    (dataset / "_versions" / "1.manifest").write_bytes(b"")

    assert is_lance_dataset(storage, str(dataset))
    assert is_lance_dataset(storage, str(dataset / "_versions" / "1.manifest"))
    assert not is_lance_dataset(storage, str(plain))
