# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from zephon._internal.ops.decode_text import DecodeText
from zephon.types import SampleMeta, SampleRecord


def _rec(payload: dict[str, Any]) -> SampleRecord:
    meta = SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0)
    return SampleRecord(meta=meta, payload=payload)


def _apply(op: DecodeText, payload: dict[str, Any]) -> SampleRecord:
    return op.process_one(_rec(payload))[0]


def test_decode_bytes_to_string_default_utf8() -> None:
    op = DecodeText()
    out = _apply(op, {"text": b"hallo", "other": "keep"})
    assert out.payload["text"] == "hallo"
    assert out.payload["other"] == "keep"


def test_decode_normalize_newlines_toggle() -> None:
    text = "a\r\nb\rc\n"
    op_norm = DecodeText()
    out = _apply(op_norm, {"text": text})
    assert out.payload["text"] == "a\nb\nc\n"

    op_raw = DecodeText(normalize_newlines=False)
    out2 = _apply(op_raw, {"text": text})
    assert out2.payload["text"] == text


def test_decode_lowercase_toggle() -> None:
    op = DecodeText(lowercase=True)
    out = _apply(op, {"text": b"HeLLo WORLD"})
    assert out.payload["text"] == "hello world"


def test_decode_multiple_fields() -> None:
    op = DecodeText(fields=("text", "title"))
    out = _apply(op, {"text": b"T1", "title": b"T2", "ignore": b"X"})
    assert out.payload["text"] == "T1"
    assert out.payload["title"] == "T2"
    assert out.payload["ignore"] == b"X"


def test_decode_process_many_matches_one_by_one() -> None:
    op = DecodeText(fields=("text",))
    items = [
        _rec({"text": b"A"}),
        _rec({"text": b"B"}),
        _rec({"text": b"C"}),
    ]
    bulk = op.process_many(items)
    seq = [op.process_one(x)[0] for x in items]
    assert [r.payload["text"] for r in bulk] == [r.payload["text"] for r in seq]


def test_decode_traits_and_default_accumulator() -> None:
    op = DecodeText(max_batch=64, max_latency_ms=25)
    t = op.traits()

    assert t.indexable is True and t.parallelism == 2

    # Test deterministic mode disables time-based flushing
    acc_det = op.accumulator(deterministic=True, ctx={})
    assert acc_det._max_batch == 64
    assert acc_det._max_latency_ms is None

    # Test non-deterministic mode preserves latency config
    acc_nondet = op.accumulator(deterministic=False, ctx={})
    assert acc_nondet._max_batch == 64
    assert acc_nondet._max_latency_ms == 25
