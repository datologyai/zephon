from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from zephon._internal.observability.size_estimator import content_bytes, estimate_bytes
from zephon.types import SampleBatch, SampleMeta, SampleRecord


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


@pytest.mark.parametrize(
    "value,minimum",
    [
        (None, 0),
        (b"bytes", 5),
        ("text", len("text".encode("utf-8"))),
    ],
)
def test_estimate_bytes_handles_simple_types(value: Any, minimum: int) -> None:
    size = estimate_bytes(value)
    assert size >= minimum


def test_estimate_bytes_handles_nested_containers() -> None:
    nested = {"a": [1, 2, 3], "b": {"inner": b"payload"}}
    size = estimate_bytes(nested)
    assert size >= len(b"payload")


def test_estimate_bytes_counts_sample_record_payloads() -> None:
    payload = b"x" * 4096
    meta = SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0)
    record = SampleRecord(meta=meta, payload={"data": payload})
    batch = SampleBatch(records=(record,))

    assert estimate_bytes(record) >= len(payload)
    assert estimate_bytes(batch) >= len(payload)
    assert estimate_bytes([batch]) >= len(payload)


def test_estimate_bytes_does_not_double_count_shared_records() -> None:
    payload = b"x" * 4096
    meta = SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0)
    record = SampleRecord(meta=meta, payload={"data": payload})

    assert estimate_bytes([record, record]) < 2 * len(payload)
