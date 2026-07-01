from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zephon.observability.size_estimator import content_bytes, estimate_bytes


def test_content_bytes_leaves() -> None:
    assert content_bytes("héllo") == len("héllo".encode("utf-8"))
    assert content_bytes(b"1234") == 4
    assert content_bytes(bytearray(b"12")) == 2
    assert content_bytes(memoryview(b"123")) == 3
    assert content_bytes(np.zeros(10, dtype=np.int64)) == 80


def test_content_bytes_string_with_surrogate_does_not_raise() -> None:
    assert content_bytes("a\ud800b") == 2  # lone surrogate dropped, not raised


def test_content_bytes_counts_mapping_values_not_keys() -> None:
    # content_bytes excludes container overhead and mapping keys.
    obj = {"a_long_key": b"x" * 100}
    assert content_bytes(obj) == 100
    assert estimate_bytes(obj) > content_bytes(obj)


def test_content_bytes_has_no_container_overhead() -> None:
    payload = b"x" * 100
    assert content_bytes([payload, payload]) == 200
    assert content_bytes((payload,)) == 100


def test_content_bytes_recurses_into_structs() -> None:
    @dataclass
    class Row:
        blob: bytes
        text: str

    assert content_bytes(Row(blob=b"x" * 64, text="ab")) == 66


def test_content_bytes_scalars_and_unknown() -> None:
    assert content_bytes(1) == 8
    assert content_bytes(1.5) == 8
    assert content_bytes(True) == 8
    assert content_bytes(object()) == 0


def test_content_bytes_nested() -> None:
    obj = {"tokens": [b"aa", b"bbbb"], "meta": {"src": "x" * 3}}
    assert content_bytes(obj) == 2 + 4 + 3


def test_estimate_bytes_string_with_surrogate_does_not_raise() -> None:
    assert estimate_bytes("a\ud800b") == 2
