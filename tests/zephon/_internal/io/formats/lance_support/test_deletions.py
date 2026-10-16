# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Roaring and Arrow tombstone decoding, and the logical-to-physical mapping."""

import struct
from pathlib import Path

import numpy as np
import pytest

from zephon._internal.io.formats.lance_support.deletions import (
    DeletionVector,
    _parse_roaring,
    read_deletion_vector,
)
from zephon._internal.io.storage.local import LocalFSBackend

_COOKIE_NO_RUN = 12346
_COOKIE_WITH_RUN = 12347


def _array_container(values: list[int]) -> bytes:
    return b"".join(struct.pack("<H", value) for value in values)


def _bitset_container(values: list[int]) -> bytes:
    bits = np.zeros(8192 * 8, dtype=np.uint8)
    bits[values] = 1
    return np.packbits(bits, bitorder="little").tobytes()


def _run_container(runs: list[tuple[int, int]]) -> bytes:
    body = struct.pack("<H", len(runs))
    for start, length in runs:
        body += struct.pack("<HH", start, length - 1)
    return body


def _roaring_no_run(containers: list[tuple[int, int, bytes]]) -> bytes:
    """Serialize containers as ``(key, cardinality, payload)`` without run flags."""
    out = struct.pack("<II", _COOKIE_NO_RUN, len(containers))
    for key, cardinality, _payload in containers:
        out += struct.pack("<HH", key, cardinality - 1)
    offset = len(out) + 4 * len(containers)
    for _key, _cardinality, payload in containers:
        out += struct.pack("<I", offset)
        offset += len(payload)
    return out + b"".join(payload for _key, _cardinality, payload in containers)


def _roaring_with_runs(containers: list[tuple[int, int, bytes, bool]]) -> bytes:
    out = struct.pack("<I", _COOKIE_WITH_RUN | (len(containers) - 1) << 16)
    flags = bytearray((len(containers) + 7) // 8)
    for index, (_key, _card, _payload, is_run) in enumerate(containers):
        if is_run:
            flags[index // 8] |= 1 << (index % 8)
    out += bytes(flags)
    for key, cardinality, _payload, _is_run in containers:
        out += struct.pack("<HH", key, cardinality - 1)
    if len(containers) >= 4:
        offset = len(out) + 4 * len(containers)
        for _key, _card, payload, _is_run in containers:
            out += struct.pack("<I", offset)
            offset += len(payload)
    return out + b"".join(payload for _k, _c, payload, _r in containers)


def test_array_container() -> None:
    values = [1, 5, 9, 4000]
    buf = _roaring_no_run([(0, len(values), _array_container(values))])

    assert _parse_roaring(buf).tolist() == values


def test_bitset_container() -> None:
    values = list(range(0, 12000, 2))
    buf = _roaring_no_run([(0, len(values), _bitset_container(values))])

    assert _parse_roaring(buf).tolist() == values


def test_run_container() -> None:
    runs = [(3, 4), (100, 2)]
    expected = [3, 4, 5, 6, 100, 101]
    buf = _roaring_with_runs([(0, len(expected), _run_container(runs), True)])

    assert _parse_roaring(buf).tolist() == expected


def test_high_keys_offset_container_values() -> None:
    """A container's key supplies the top 16 bits of every value it holds."""
    buf = _roaring_no_run(
        [(0, 1, _array_container([7])), (2, 2, _array_container([1, 3]))]
    )

    assert _parse_roaring(buf).tolist() == [7, (2 << 16) + 1, (2 << 16) + 3]


def test_many_containers_carry_an_offset_header_even_with_runs() -> None:
    containers = [(key, 2, _array_container([1, 2]), False) for key in range(5)]
    expected = [(key << 16) + value for key in range(5) for value in (1, 2)]

    assert _parse_roaring(_roaring_with_runs(containers)).tolist() == expected


def test_rejects_a_foreign_cookie() -> None:
    with pytest.raises(ValueError, match="Roaring"):
        _parse_roaring(struct.pack("<II", 999, 0))


@pytest.mark.parametrize("physical_rows", [1, 7, 40])
def test_mapping_matches_a_brute_force_oracle(physical_rows: int) -> None:
    rng = np.random.default_rng(physical_rows)
    for _ in range(50):
        num_deleted = int(rng.integers(0, physical_rows))
        deleted = np.sort(
            rng.choice(physical_rows, size=num_deleted, replace=False)
        ).astype(np.int64)
        alive = [
            row for row in range(physical_rows) if row not in set(deleted.tolist())
        ]

        vector = DeletionVector.from_offsets(deleted)

        assert vector.to_physical(list(range(len(alive)))) == alive


def test_mapping_preserves_request_order_and_duplicates() -> None:
    vector = DeletionVector.from_offsets(np.array([1, 2], dtype=np.int64))

    assert vector.to_physical([2, 0, 2]) == [4, 0, 4]


def test_reads_an_arrow_tombstone_file(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.ipc as ipc

    path = tmp_path / "0-1-2.arrow"
    table = pa.table({"row_id": pa.array([1, 3], pa.uint32())})
    with ipc.new_file(pa.OSFile(str(path), "wb"), table.schema) as writer:
        writer.write_table(table)

    vector = read_deletion_vector(
        LocalFSBackend(root=Path("/")),
        str(path),
        expected_deleted=2,
        physical_rows=5,
    )

    assert vector.deleted.tolist() == [1, 3]
    assert vector.to_physical([0, 1, 2]) == [0, 2, 4]


def test_rejects_a_tombstone_count_the_manifest_disagrees_with(tmp_path: Path) -> None:
    """Row counts come from the manifest, so a mismatch would shift every row."""
    path = tmp_path / "0-1-2.bin"
    path.write_bytes(_roaring_no_run([(0, 2, _array_container([1, 2]))]))

    with pytest.raises(ValueError, match="manifest records 3 deleted rows"):
        read_deletion_vector(
            LocalFSBackend(root=Path("/")),
            str(path),
            expected_deleted=3,
            physical_rows=9,
        )


def test_rejects_a_tombstone_beyond_the_fragment(tmp_path: Path) -> None:
    path = tmp_path / "0-1-2.bin"
    path.write_bytes(_roaring_no_run([(0, 1, _array_container([9]))]))

    with pytest.raises(ValueError, match="beyond the fragment"):
        read_deletion_vector(
            LocalFSBackend(root=Path("/")),
            str(path),
            expected_deleted=1,
            physical_rows=5,
        )
