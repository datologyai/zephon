from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from zephon._internal.observability.size_estimator import (
    content_bytes,
    metric_bytes,
)
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
    assert content_bytes(None) == 0
    assert content_bytes(np.float32(1.0)) == 4
    assert content_bytes(object()) == 0


def test_content_bytes_nested() -> None:
    obj = {"tokens": [b"aa", b"bbbb"], "meta": {"src": "x" * 3}}
    assert content_bytes(obj) == 2 + 4 + 3


def test_content_bytes_counts_record_and_batch_payloads_only() -> None:
    payload = {"data": b"x" * 4096, "text": "abc"}
    meta = SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0, tags={"k": "v"})
    record = SampleRecord(meta=meta, payload=payload)
    batch = SampleBatch(records=(record, record))

    assert content_bytes(record) == 4099
    assert content_bytes(batch) == 2 * 4099
    assert content_bytes([batch, record]) == 3 * 4099


def test_content_bytes_homogeneous_and_mixed_sequences() -> None:
    assert content_bytes([1] * 100) == 800
    assert content_bytes([1, 2.0, True]) == 24
    assert content_bytes([1, None, "ab"]) == 10
    assert content_bytes(["ab", "cd"]) == 4
    assert content_bytes(["é", "a"]) == 3
    assert content_bytes([b"ab", b"c"]) == 3
    assert content_bytes([np.zeros(2, dtype=np.int8)] * 3) == 6


def test_content_bytes_subclasses_and_views() -> None:
    class Text(str):
        pass

    class Blob(bytes):
        pass

    assert content_bytes(Text("hé")) == 3
    assert content_bytes(Blob(b"abcd")) == 4
    assert content_bytes(memoryview(np.zeros(4, dtype=np.int64))) == 32


def test_content_bytes_uses_nbytes_attribute() -> None:
    class ArrayLike:
        nbytes = 123

    class BoolNbytes:
        nbytes = True

    assert content_bytes(ArrayLike()) == 123
    assert content_bytes(BoolNbytes()) == 0


def test_content_bytes_pyarrow_array() -> None:
    pa = pytest.importorskip("pyarrow")
    arr = pa.array(np.arange(100, dtype=np.int64))
    assert content_bytes(arr) == arr.nbytes == 800


@pytest.mark.parametrize("mode", ["1", "L", "P", "RGB", "RGBA", "I", "F", "I;16"])
def test_content_bytes_pil_image_matches_decoded_array(mode: str) -> None:
    image_mod = pytest.importorskip("PIL.Image")
    img = image_mod.new(mode, (7, 5))
    assert content_bytes(img) == content_bytes(np.asarray(img))


def test_metric_bytes_never_raises(caplog: pytest.LogCaptureFixture) -> None:
    cyclic: dict[str, Any] = {"x": b"abc"}
    cyclic["self"] = cyclic

    class BrokenNbytes:
        @property
        def nbytes(self) -> int:
            return 1

    broken = BrokenNbytes()
    assert content_bytes(broken) == 1
    type(broken).nbytes = property(lambda self: 1 // 0)  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING):
        assert metric_bytes(cyclic) == 0
        assert metric_bytes(broken) == 0
        assert metric_bytes({"ok": b"abcd"}) == 4
    with pytest.raises(RecursionError):
        content_bytes(cyclic)
