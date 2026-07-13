# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import pickle

import numpy as np
import pytest

from zephon.ops.tokenize_text import TokenizeText
from zephon.utils.tokenizer import fallback_tokenizer
from zephon.work.token_counting import (
    TextTokenCountingSpec,
    _count_raw_tokens,
    _pretokenized_field,
    _PretokenizedPlan,
    _TextCounter,
    _TextPlan,
    _token_array_length,
    choose_text_field,
    extract_text,
)

# ---------------------------------------------------------------------------
# Delivered-token arithmetic
# ---------------------------------------------------------------------------


def test_delivered_tokens_bracket_modes_add_specials():
    assert TextTokenCountingSpec(special_tokens="bos_eos").delivered_tokens(100) == 102
    assert TextTokenCountingSpec(special_tokens="bos").delivered_tokens(100) == 101
    assert TextTokenCountingSpec(special_tokens="eos").delivered_tokens(100) == 101
    assert TextTokenCountingSpec(special_tokens="none").delivered_tokens(100) == 100


def test_delivered_tokens_truncation_destroys_mass():
    cfg = TextTokenCountingSpec(
        special_tokens="bos_eos", truncation=True, max_length=50
    )
    # Specials reserve two slots, so content truncates to 48.
    assert cfg.delivered_tokens(100) == 50
    assert cfg.delivered_tokens(10) == 12


def test_delivered_tokens_split_preserves_mass():
    cfg = TextTokenCountingSpec(
        special_tokens="bos_eos", split_long_samples=True, max_length=50
    )
    assert cfg.delivered_tokens(1000) == 1002


def test_delivered_tokens_tokenizer_default_counts_template():
    cfg = TextTokenCountingSpec(special_tokens="tokenizer_default")
    assert cfg.delivered_tokens(100) == 100
    truncating = TextTokenCountingSpec(
        special_tokens="tokenizer_default", truncation=True, max_length=64
    )
    assert truncating.delivered_tokens(100) == 64


@pytest.mark.parametrize(
    "mode,prepend,append",
    [("bos_eos", 1, 1), ("bos", 1, 0), ("eos", 0, 1), ("none", 0, 0)],
)
def test_num_specials_by_mode(mode, prepend, append):
    assert TextTokenCountingSpec(special_tokens=mode).num_specials == prepend + append


def test_text_spec_from_op_captures_count_fields():
    tok = fallback_tokenizer()
    op = TokenizeText(
        tokenizer=tok,
        field="text",
        max_length=128,
        truncation=True,
        special_tokens="bos",
    )
    spec = TextTokenCountingSpec.from_op(op)
    assert spec.tokenizer is tok
    assert spec.field == "text"
    assert spec.max_length == 128
    assert spec.truncation is True
    assert spec.split_long_samples is False
    assert spec.special_tokens == "bos"


def test_spec_from_op_captures_use_fast():
    op = TokenizeText(tokenizer_id="__fallback__", field="text", use_fast=False)
    assert TextTokenCountingSpec.from_op(op).use_fast is False


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def test_extract_text_configured_field_wins():
    payload = {"text": "body words", "title": "a much longer title than the body"}
    assert extract_text(payload, ("text",)) == "body words"


def test_extract_text_nested_field_path():
    payload = {"doc": {"inner": {"text": "nested body"}}}
    assert extract_text(payload, ("doc", "inner", "text")) == "nested body"


def test_extract_text_configured_field_decodes_bytes():
    payload = {"text": "raw bytes body".encode("utf-8")}
    assert extract_text(payload, ("text",)) == "raw bytes body"


def test_extract_text_plain_string_payload():
    assert extract_text("just a string document") == "just a string document"


def test_extract_text_utf8_bytes_payload():
    assert extract_text("bytes document".encode("utf-8")) == "bytes document"


def test_extract_text_binary_bytes_unmeasurable():
    assert extract_text(b"\xff\xfe\x00\x01binary") is None


def test_extract_text_missing_configured_field_unmeasurable():
    # Extraction does not silently cross over to a different field.
    assert extract_text({"content": "some body"}, ("text",)) is None


def test_extract_text_non_text_payload_unmeasurable():
    assert extract_text(12345) is None
    assert extract_text([1, 2, 3]) is None
    assert extract_text({"a": 1, "b": 2}) is None


def test_choose_text_field_prefers_common_keys():
    payload = {"id": "abc-123", "url": "https://x", "text": "short"}
    assert choose_text_field(payload) == "text"


def test_choose_text_field_single_candidate():
    payload = {"n_tokens": 7, "body_field": "the only string"}
    assert choose_text_field(payload) == "body_field"


def test_choose_text_field_multiple_candidates_picks_longest():
    payload = {"id": "x" * 8, "document_body": "y" * 100}
    assert choose_text_field(payload) == "document_body"


def test_choose_text_field_decodable_bytes_count_as_text():
    payload = {"blob": "decodable text".encode("utf-8")}
    assert choose_text_field(payload) == "blob"


def test_choose_text_field_no_text_returns_none():
    assert choose_text_field({"a": 1}) is None
    assert choose_text_field("not a mapping") is None


# ---------------------------------------------------------------------------
# Pretokenized data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64, np.uint16])
def test_token_array_length_counts_elements_regardless_of_dtype(dtype):
    assert _token_array_length(np.arange(20, dtype=dtype)) == 20


def test_token_array_length_rejects_non_token_arrays():
    assert _token_array_length(np.arange(5, dtype=np.float32)) is None
    assert _token_array_length(np.array([True, False])) is None
    assert _token_array_length([1, 2, 3]) == 3
    assert _token_array_length([1.0, 2.0]) is None
    assert _token_array_length([True, False]) is None
    assert _token_array_length("a string") is None
    assert _token_array_length(b"\x00\x01") is None
    assert _token_array_length([]) is None


def test_token_array_length_duck_typed_integer_tensor():
    # torch is optional, so the tensor branch is detected by duck typing.
    class _IntTensor:
        def is_floating_point(self):
            return False

        def is_complex(self):
            return False

        def numel(self):
            return 42

    class _FloatTensor:
        def is_floating_point(self):
            return True

        def numel(self):
            return 42

    assert _token_array_length(_IntTensor()) == 42
    assert _token_array_length(_FloatTensor()) is None


def test_pretokenized_field_resolution_order():
    # () means "the payload itself is the token array".
    assert _pretokenized_field(np.arange(8, dtype=np.int32), None) == ()
    assert _pretokenized_field({"input_ids": np.arange(12, dtype=np.int64)}, None) == (
        "input_ids",
    )
    nested = {"enc": {"ids": np.arange(7, dtype=np.int32)}}
    assert _pretokenized_field(nested, ("enc", "ids")) == ("enc", "ids")
    assert _pretokenized_field({"text": "words go here"}, None) is None
    assert _pretokenized_field({"text": "words"}, ("text",)) is None


# ---------------------------------------------------------------------------
# Counter building + raw token counting
# ---------------------------------------------------------------------------


def test_build_counter_uses_passed_tokenizer_instance():
    tok = fallback_tokenizer()
    counter = TextTokenCountingSpec(tokenizer=tok).build_counter()
    assert isinstance(counter, _TextCounter)
    assert counter.tokenizer is tok


@pytest.mark.parametrize("tokenizer_id", [None, "__fallback__"])
def test_build_counter_falls_back_without_model(tokenizer_id):
    counter = TextTokenCountingSpec(tokenizer_id=tokenizer_id).build_counter()
    assert isinstance(counter, _TextCounter)
    assert counter.tokenizer.bos_token_id == 1  # the in-process fallback stub


def test_build_counter_delegates_to_shared_loader(monkeypatch):
    calls: list[tuple[str | None, bool | None]] = []

    def fake_load(tokenizer_id, *, use_fast):
        calls.append((tokenizer_id, use_fast))
        return fallback_tokenizer()

    monkeypatch.setattr("zephon.work.token_counting.load_hf_tokenizer", fake_load)
    TextTokenCountingSpec(tokenizer_id="hf-model", use_fast=False).build_counter()
    assert calls == [("hf-model", False)]


def test_count_raw_tokens_counts_whitespace_words():
    tok = fallback_tokenizer()
    profile = TextTokenCountingSpec(special_tokens="none")
    assert _count_raw_tokens(tok, "one two three four", profile) == 4


def test_count_raw_tokens_passes_add_special_tokens_by_mode():
    class _SpecialsAware:
        def __call__(self, text, add_special_tokens=False):
            n = len(text.split())
            return {"input_ids": list(range(n + (2 if add_special_tokens else 0)))}

    tok = _SpecialsAware()
    assert (
        _count_raw_tokens(tok, "a b c", TextTokenCountingSpec(special_tokens="bos_eos"))
        == 3
    )
    assert (
        _count_raw_tokens(
            tok, "a b c", TextTokenCountingSpec(special_tokens="tokenizer_default")
        )
        == 5
    )


def test_count_raw_tokens_reads_input_ids_attribute():
    class _AttrResult:
        def __init__(self, ids):
            self.input_ids = ids

    class _AttrTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return _AttrResult(list(range(len(text.split()))))

    profile = TextTokenCountingSpec(special_tokens="none")
    assert _count_raw_tokens(_AttrTokenizer(), "a b c d", profile) == 4


# ---------------------------------------------------------------------------
# Plan selection (voting) + plan counting
# ---------------------------------------------------------------------------


def _plan(payloads, field: str | None = None):
    counter = _TextCounter(fallback_tokenizer(), TextTokenCountingSpec(field=field))
    return counter.plan(payloads)


def test_text_plan_treats_empty_text_as_unmeasurable():
    plan = _TextPlan(
        fallback_tokenizer(), TextTokenCountingSpec(special_tokens="none"), None
    )
    assert plan.count("") is None


def test_pretokenized_plan_treats_empty_array_as_unmeasurable():
    plan = _PretokenizedPlan(("input_ids",))
    assert plan.count({"input_ids": np.zeros(0, dtype=np.int64)}) is None


def test_plan_delivery_detects_pretokenized():
    plan = _plan([{"input_ids": np.arange(5)}])
    assert isinstance(plan, _PretokenizedPlan)
    assert plan.path == ("input_ids",)


def test_plan_delivery_falls_through_to_text():
    plan = _plan([{"body": "hello world"}])
    assert isinstance(plan, _TextPlan)
    assert plan.path == ("body",)


def test_plan_delivery_uses_configured_field_path():
    plan = _plan([{"doc": {"text": "x"}}], field="doc.text")
    assert isinstance(plan, _TextPlan)
    assert plan.path == ("doc", "text")


def test_plan_delivery_prefix_outvotes_anomalous_first_payload():
    payloads = [{"url": "https://example.com/a", "id": "0"}] + [
        {"url": "https://example.com/b", "text": "real document body"}
    ] * 7
    plan = _plan(payloads)
    assert isinstance(plan, _TextPlan)
    assert plan.path == ("text",)


def test_plan_delivery_prefix_detects_pretokenized_past_bad_record():
    payloads = [{"input_ids": None}, {"input_ids": np.arange(4)}]
    plan = _plan(payloads)
    assert isinstance(plan, _PretokenizedPlan)
    assert plan.path == ("input_ids",)


def test_plan_delivery_prefix_outvotes_anomalous_pretokenized_record():
    payloads = [{"text": "x", "input_ids": np.arange(3)}] + [
        {"text": "real document body"}
    ] * 7
    plan = _plan(payloads)
    assert isinstance(plan, _TextPlan)
    assert plan.path == ("text",)


def test_plan_delivery_majority_field_beats_minority_priority_key():
    payloads = [{"text": "t"}] + [{"content": "long body"}] * 7
    plan = _plan(payloads)
    assert isinstance(plan, _TextPlan)
    assert plan.path == ("content",)


def test_plan_delivery_configured_field_mode_follows_majority():
    payloads = [{"f": np.arange(4)}] + [{"f": "words in the field"}] * 7
    plan = _plan(payloads, field="f")
    assert isinstance(plan, _TextPlan)
    assert plan.path == ("f",)


# ---------------------------------------------------------------------------
# Spec pickling
# ---------------------------------------------------------------------------


class _ClosureTokenizer:
    """Whitespace tokenizer whose lambda attribute defeats standard pickle."""

    def __init__(self):
        self.split = lambda text: text.split()

    def __call__(self, text, **kwargs):
        return {"input_ids": self.split(text)}


def test_counting_spec_std_pickles_with_closure_tokenizer():
    spec = TextTokenCountingSpec(tokenizer=_ClosureTokenizer(), special_tokens="none")
    restored = pickle.loads(pickle.dumps(spec))
    assert restored.tokenizer is not None
    assert restored.tokenizer("a b c")["input_ids"] == ["a", "b", "c"]
