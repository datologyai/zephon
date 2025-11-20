# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from typing import Any

import pytest

from zephon.core.constants import SampleMeta, SampleRecord
from zephon.core.op_base import OpContext
from zephon.ops.tokenize_text import TokenizeText


def _noop(*args, **kwargs) -> None:  # pragma: no cover - trivial helper
    return None


def _setup(
    op: TokenizeText,
    ctx_data: dict[str, object] | None = None,
    *,
    collect_stats: bool = False,
) -> TokenizeText:
    ctx = {"record_node_metrics": _noop}
    if ctx_data:
        ctx.update(ctx_data)
    op.setup(
        OpContext(ctx),
        stage_index=0,
        stage_name="stage0",
        op_index=0,
        collect_stats=collect_stats,
    )
    return op


def _rec(text: Any, *, field: str = "text") -> SampleRecord:
    meta = SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0)
    return SampleRecord(meta=meta, payload={field: text})


def _payload_dict(record: SampleRecord) -> dict[str, Any]:
    payload = record.payload
    assert isinstance(payload, dict)
    return payload


def test_tokenize_fallback_process_one_and_many() -> None:
    op = _setup(TokenizeText(tokenizer=None, tokenizer_id="__fallback__"))
    r1 = _rec("hello world")
    out1 = op.process_one(r1)[0]
    payload = _payload_dict(out1)
    assert "input_ids" in payload and "attention_mask" in payload
    # many
    r2 = _rec("more words")
    bulk = op.process_many([r1, r2])
    assert len(bulk) == 2
    for rec in bulk:
        payload = _payload_dict(rec)
        assert "input_ids" in payload
        assert "attention_mask" in payload


def test_tokenize_custom_field_and_missing_field() -> None:
    op = _setup(
        TokenizeText(tokenizer=None, tokenizer_id="__fallback__", field="title")
    )
    r = _rec("ignored", field="text")  # text present, but tokenizer uses "title"
    out = op.process_one(r)[0]
    # When field is missing, fallback tokenizer sees empty string
    payload = _payload_dict(out)
    assert payload.get("input_ids", []) == []
    assert payload.get("attention_mask", []) == []


def test_tokenize_disable_attention_mask() -> None:
    op = TokenizeText(
        tokenizer=None, tokenizer_id="__fallback__", add_attention_mask=False
    )
    op = _setup(op)
    out = op.process_one(_rec("hi there"))[0]
    payload = _payload_dict(out)
    assert "input_ids" in payload
    assert "attention_mask" not in payload


def test_tokenize_custom_tokenizer_and_resolved_id() -> None:
    class ToyTok:
        name_or_path = "toy-tokenizer"

        def __call__(
            self, texts, add_special_tokens=True, padding=False, truncation=False
        ):
            if isinstance(texts, str):
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
            return {
                "input_ids": [[1, 2, 3] for _ in texts],
                "attention_mask": [[1, 1, 1] for _ in texts],
            }

    tok = ToyTok()
    op = _setup(TokenizeText(tokenizer=tok))
    out = op.process_many([_rec("x"), _rec("y")])
    assert _payload_dict(out[0])["input_ids"] == [1, 2, 3]
    assert op.resolved_tokenizer_id() == "toy-tokenizer"


def test_tokenizer_id_load_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    # Suppress expected warning from fallback path to keep logs clean
    caplog.set_level(logging.ERROR, logger="zephon.ops.tokenize_text")
    # Provide a fake transformers module to avoid network/HF
    fake = types.ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name: str):  # pragma: no cover - simple stub
            raise RuntimeError("fail")

    fake.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake)

    op = _setup(TokenizeText(tokenizer=None, tokenizer_id="some-model"))
    out = op.process_one(_rec("hello"))[0]
    # Fallback engaged internally, but resolved_tokenizer_id returns configured id
    assert "input_ids" in _payload_dict(out)
    assert op.resolved_tokenizer_id() == "some-model"


def test_tokenizer_raises_if_setup_not_called() -> None:
    op = TokenizeText(tokenizer=None, tokenizer_id=None)
    with pytest.raises(RuntimeError):
        _ = op.process_one(_rec("text"))


def test_tokenizer_traits_and_default_buffering() -> None:
    op = _setup(TokenizeText(tokenizer_id="__fallback__"))
    t = op.traits()
    buf = op.buffering()
    assert t.indexable is True and t.parallelism == 4
    assert buf is not None and buf.max_batch == 64 and buf.max_latency_ms == 3
