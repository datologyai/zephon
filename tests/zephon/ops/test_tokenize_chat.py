# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from zephon.core.constants import SampleMeta, SampleRecord
from zephon.io import Dataset, InMemoryShard
from zephon.io.stores.multi import build_multi_dataset_store
from zephon.observability.size_estimator import content_bytes
from zephon.ops.tokenize_chat import (
    SpanSource,
    TokenizeChat,
    _last_span_only,
    _mask_from_spans,
)
from zephon.work.token_counting import FatalCountError
from zephon.work.token_estimation import (
    TokenEstimation,
    _measure_dataset,
    prime_token_ratios,
)

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")


_WORDS = (
    "hello hi there how are you good morning the answer is what thanks "
    "yes no 1 2 4 12 + ? ! . w"
).split()

#: Both templates render to the identical string; only span provenance differs.
UNTAGGED_TEMPLATE = (
    "{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}"
    "<|end|>{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
TAGGED_TEMPLATE = (
    "{% for message in messages %}{% if message['role'] == 'assistant' %}"
    "<|assistant|>{% generation %}{{ message['content'] }}<|end|>{% endgeneration %}"
    "{% else %}<|{{ message['role'] }}|>{{ message['content'] }}<|end|>{% endif %}"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
#: Assistant turns end without EOS — exercises the conditional EOS finalizer.
TAGGED_TEMPLATE_NO_EOS = (
    "{% for message in messages %}{% if message['role'] == 'assistant' %}"
    "<|assistant|>{% generation %}{{ message['content'] }}{% endgeneration %}"
    "{% else %}<|{{ message['role'] }}|>{{ message['content'] }}<|end|>{% endif %}"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
#: The trailing marker moves with the conversation length, so partial renders
#: are not prefixes of the full render.
POSITION_DEPENDENT_TEMPLATE = (
    "{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}"
    "{% if loop.last %}<|sep|>{% endif %}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
TOOLS_TEMPLATE = (
    "{% if tools %}<|sep|>{% endif %}{% if enable_thinking %}<|sep|><|sep|>{% endif %}"
    + UNTAGGED_TEMPLATE
)


def _messages(*items: tuple[str, Any]) -> list[dict[str, Any]]:
    return [{"role": role, "content": content} for role, content in items]


CONVERSATION = _messages(
    ("user", "hi there"),
    ("assistant", "good morning"),
    ("user", "what is 2 + 2 ?"),
    ("assistant", "the answer is 4"),
)

GOLDEN_TOKENS = (
    "<|user|> hi there <|end|> <|assistant|> good morning <|end|> "
    "<|user|> what is 2 + 2 ? <|end|> <|assistant|> the answer is 4 <|end|>"
).split()
GOLDEN_MASK = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


@pytest.fixture(scope="module")
def fast_tokenizer() -> Any:
    vocab = {"[UNK]": 0}
    for word in _WORDS:
        vocab[word] = len(vocab)
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        eos_token="<|end|>",
        pad_token="<|pad|>",
    )
    fast.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|user|>",
                "<|assistant|>",
                "<|system|>",
                "<|sep|>",
            ]
        }
    )
    return fast


def _rec(payload: Any, sample_id: tuple[int, int, int] = (0, 0, 0)) -> SampleRecord:
    meta = SampleMeta(sample_id=sample_id, lane_id=0, chunk_id=0)
    return SampleRecord(meta=meta, payload=payload)


def _run_one(op: TokenizeChat, payload: Any) -> SampleRecord:
    out = op.process_many([_rec(payload)])
    assert len(out) == 1
    return out[0]


def _payload(record: SampleRecord) -> dict[str, Any]:
    payload = record.payload
    assert isinstance(payload, dict)
    return payload


def _tokens(fast_tokenizer: Any, record: SampleRecord) -> list[str]:
    payload = _payload(record)
    return fast_tokenizer.convert_ids_to_tokens(payload["input_ids"].tolist())


def _mask(record: SampleRecord, field: str = "loss_mask") -> list[int]:
    return _payload(record)[field].tolist()


def test_mask_from_spans_overlap_rule() -> None:
    offsets = [(0, 4), (4, 8), (8, 12), (12, 12)]
    # Straddling tokens are supervised; zero-width offsets never are.
    assert _mask_from_spans(offsets, [(6, 10)]).tolist() == [0, 1, 1, 0]
    assert _mask_from_spans(offsets, []).tolist() == [0, 0, 0, 0]


def test_last_span_only() -> None:
    def run(mask: list[int]) -> list[int]:
        return _last_span_only(np.asarray(mask, dtype=np.uint8)).tolist()

    assert run([0, 1, 1, 0, 1, 1, 0]) == [0, 0, 0, 0, 1, 1, 0]
    assert run([0, 1, 1]) == [0, 1, 1]
    assert run([0, 0, 0]) == [0, 0, 0]


# Golden fixtures (tagged primary)


def test_tagged_golden(fast_tokenizer: Any) -> None:
    op = TokenizeChat(
        fast_tokenizer,
        chat_template=TAGGED_TEMPLATE,
        span_source="generation_tags",
    )
    record = _run_one(op, {"messages": CONVERSATION})
    assert _tokens(fast_tokenizer, record) == GOLDEN_TOKENS
    assert _mask(record) == GOLDEN_MASK
    payload = _payload(record)
    assert payload["input_ids"].dtype == np.int64
    assert payload["loss_mask"].dtype == np.uint8
    # EOS supervised, generation-prompt header not, first token never.
    assert _mask(record)[0] == 0
    eos = fast_tokenizer.eos_token_id
    ids = payload["input_ids"].tolist()
    assert ids[-1] == eos and _mask(record)[-1] == 1
    # Exactly one trailing EOS: no duplicate append on templates that already
    # close the conversation with eos_id.
    assert ids[-2] != eos


def test_mode_equivalence_tagged_vs_prefix_diff(fast_tokenizer: Any) -> None:
    tagged = TokenizeChat(
        fast_tokenizer, chat_template=TAGGED_TEMPLATE, span_source="generation_tags"
    )
    fallback = TokenizeChat(
        fast_tokenizer, chat_template=UNTAGGED_TEMPLATE, span_source="prefix_diff"
    )
    conversations = [
        CONVERSATION,
        _messages(
            ("system", "you are good"),
            ("user", "hello"),
            ("assistant", "hi !"),
        ),
        _messages(
            ("user", "what is 1 + 1 ?"),
            ("assistant", "2"),
            ("user", "thanks !"),
            ("assistant", "yes ."),
        ),
    ]
    for messages in conversations:
        a = _run_one(tagged, {"messages": messages})
        b = _run_one(fallback, {"messages": messages})
        pa, pb = _payload(a), _payload(b)
        assert pa["input_ids"].tolist() == pb["input_ids"].tolist()
        assert pa["loss_mask"].tolist() == pb["loss_mask"].tolist()


def test_auto_selects_tags_and_untagged_warns(
    fast_tokenizer: Any, caplog: pytest.LogCaptureFixture
) -> None:
    tagged = TokenizeChat(fast_tokenizer, chat_template=TAGGED_TEMPLATE)
    _run_one(tagged, {"messages": CONVERSATION})
    assert tagged._resolved_span_source == "generation_tags"

    with caplog.at_level(logging.WARNING, logger="zephon.ops.tokenize_chat"):
        fallback = TokenizeChat(fast_tokenizer, chat_template=UNTAGGED_TEMPLATE)
        _run_one(fallback, {"messages": CONVERSATION})
    assert fallback._resolved_span_source == "prefix_diff"
    assert any("prefix-diff" in rec.message for rec in caplog.records)


def test_position_dependent_template_hard_errors(fast_tokenizer: Any) -> None:
    op = TokenizeChat(
        fast_tokenizer,
        chat_template=POSITION_DEPENDENT_TEMPLATE,
        span_source="prefix_diff",
    )
    messages = _messages(
        ("user", "hi"),
        ("assistant", "hello"),
        ("user", "thanks"),
    )
    with pytest.raises(ValueError, match="position-dependent"):
        op.process_many([_rec({"messages": messages})])


def test_loss_on_last_turn_only(fast_tokenizer: Any) -> None:
    cases: tuple[tuple[str, SpanSource], ...] = (
        (TAGGED_TEMPLATE, "generation_tags"),
        (UNTAGGED_TEMPLATE, "prefix_diff"),
    )
    for template, source in cases:
        op = TokenizeChat(
            fast_tokenizer,
            chat_template=template,
            span_source=source,
            loss_on_last_turn_only=True,
        )
        record = _run_one(op, {"messages": CONVERSATION})
        expected = [0] * 17 + [1, 1, 1, 1, 1]
        assert _mask(record) == expected


# EOS finalizer


def test_eos_finalizer_appends_when_render_lacks_eos(fast_tokenizer: Any) -> None:
    op = TokenizeChat(
        fast_tokenizer,
        chat_template=TAGGED_TEMPLATE_NO_EOS,
        span_source="generation_tags",
    )
    messages = CONVERSATION[:2]
    record = _run_one(op, {"messages": messages})
    tokens = _tokens(fast_tokenizer, record)
    assert (
        tokens == "<|user|> hi there <|end|> <|assistant|> good morning <|end|>".split()
    )
    assert _mask(record) == [0, 0, 0, 0, 0, 1, 1, 1]


def test_eos_finalizer_unsupervised_after_user_final_turn(fast_tokenizer: Any) -> None:
    # No turn terminators at all, so the render never ends with eos_id.
    template = (
        "{% for message in messages %}{% if message['role'] == 'assistant' %}"
        "<|assistant|>{% generation %}{{ message['content'] }}{% endgeneration %}"
        "{% else %}<|{{ message['role'] }}|>{{ message['content'] }}{% endif %}"
        "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )
    op = TokenizeChat(
        fast_tokenizer, chat_template=template, span_source="generation_tags"
    )
    messages = _messages(
        ("user", "hi there"),
        ("assistant", "good morning"),
        ("user", "thanks !"),
    )
    record = _run_one(op, {"messages": messages})
    payload = _payload(record)
    ids = payload["input_ids"].tolist()
    mask = _mask(record)
    # The terminator is appended for packing but not trained: the
    # conversation ends in user content, a transition inference never samples.
    assert ids[-1] == fast_tokenizer.eos_token_id
    assert mask[-1] == 0
    assert sum(mask) == 2  # "good morning" stays supervised


def test_eos_finalizer_skipped_at_max_length(fast_tokenizer: Any) -> None:
    op = TokenizeChat(
        fast_tokenizer,
        chat_template=TAGGED_TEMPLATE_NO_EOS,
        span_source="generation_tags",
        max_length=7,
    )
    record = _run_one(op, {"messages": CONVERSATION[:2]})
    tokens = _tokens(fast_tokenizer, record)
    # Render is exactly 7 tokens; there is no room left, so the lost EOS is
    # not re-appended (truncate-after-append semantics).
    assert tokens == "<|user|> hi there <|end|> <|assistant|> good morning".split()
    assert _mask(record) == [0, 0, 0, 0, 0, 1, 1]


def test_eos_token_override_rebinds_string_and_id(
    fast_tokenizer: Any, tmp_path: Any
) -> None:
    # Loading by id with an eos_token override must rebind both EOS forms the
    # op derives at setup; the finalizer then appends the overridden id.
    fast_tokenizer.save_pretrained(tmp_path)
    op = TokenizeChat(
        tokenizer_id=str(tmp_path),
        eos_token="<|sep|>",
        chat_template=TAGGED_TEMPLATE_NO_EOS,
        span_source="generation_tags",
    )
    record = _run_one(op, {"messages": CONVERSATION[:2]})
    assert op._eos_str == "<|sep|>"
    assert op._eos_id == fast_tokenizer.convert_tokens_to_ids("<|sep|>")
    ids = _payload(record)["input_ids"].tolist()
    assert ids[-1] == op._eos_id
    assert _mask(record)[-1] == 1  # the appended EOS closes an assistant turn


# No-template path


def test_no_template_golden(fast_tokenizer: Any) -> None:
    op = TokenizeChat(fast_tokenizer, apply_chat_template=False)
    messages = _messages(
        ("user", "what is 2 + 2 ?"),
        ("assistant", "the answer is 4"),
    )
    record = _run_one(op, {"messages": messages})
    # Raw concat: no BOS, no separators, per-assistant-turn EOS supervised.
    assert _tokens(fast_tokenizer, record) == (
        "what is 2 + 2 ? the answer is 4 <|end|>".split()
    )
    assert _mask(record) == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


def test_no_template_multi_turn_interleaves_eos(fast_tokenizer: Any) -> None:
    op = TokenizeChat(fast_tokenizer, apply_chat_template=False)
    messages = _messages(
        ("user", "hi ?"),
        ("assistant", "hello"),
        ("user", "how are you ?"),
        ("assistant", "good !"),
    )
    record = _run_one(op, {"messages": messages})
    assert _tokens(fast_tokenizer, record) == (
        "hi ? hello <|end|> how are you ? good ! <|end|>".split()
    )
    assert _mask(record) == [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1]


def test_no_template_boundary_straddle_is_supervised(fast_tokenizer: Any) -> None:
    # Without template separators, a token can merge across the user→assistant
    # content boundary; the overlap rule supervises it.
    op = TokenizeChat(fast_tokenizer, apply_chat_template=False)
    messages = _messages(("user", "? hi"), ("assistant", "hello"))
    record = _run_one(op, {"messages": messages})
    # "hihello" merges into one (unknown) token straddling the boundary.
    assert _tokens(fast_tokenizer, record) == ["?", "[UNK]", "<|end|>"]
    assert _mask(record) == [0, 1, 1]


def test_no_template_multimodal_content_joins_text_parts(fast_tokenizer: Any) -> None:
    op = TokenizeChat(fast_tokenizer, apply_chat_template=False)
    messages = _messages(
        (
            "user",
            [
                {"type": "text", "text": "hi "},
                {"type": "image", "url": "ignored"},
                {"type": "text", "text": "there ?"},
            ],
        ),
        ("assistant", "hello"),
    )
    record = _run_one(op, {"messages": messages})
    assert _tokens(fast_tokenizer, record) == ["hi", "there", "?", "hello", "<|end|>"]
    assert _mask(record) == [0, 0, 0, 1, 1]


def test_no_template_text_part_missing_text_key_errors(fast_tokenizer: Any) -> None:
    op = TokenizeChat(fast_tokenizer, apply_chat_template=False)
    messages = _messages(
        ("user", [{"type": "text"}]),
        ("assistant", "hello"),
    )
    with pytest.raises(ValueError, match="missing its 'text' key"):
        op.process_many([_rec({"messages": messages})])


def test_mask0_guard_hard_errors_on_assistant_first(fast_tokenizer: Any) -> None:
    op = TokenizeChat(fast_tokenizer, apply_chat_template=False)
    with pytest.raises(ValueError, match="mask\\[0\\]"):
        op.process_many([_rec({"messages": _messages(("assistant", "hi"))})])


# Drops and tombstones


def test_zero_supervision_sample_drops_with_tombstone(fast_tokenizer: Any) -> None:
    op = TokenizeChat(
        fast_tokenizer,
        chat_template=TAGGED_TEMPLATE,
        span_source="generation_tags",
        max_length=4,
    )
    # Truncation at 4 keeps only the user turn: all-zero mask.
    out = op.process_many([_rec({"messages": CONVERSATION})])
    assert len(out) == 1
    assert out[0].meta.tombstone


def test_degenerate_conversation_drops_with_tombstone(fast_tokenizer: Any) -> None:
    op = TokenizeChat(fast_tokenizer, apply_chat_template=False)
    out = op.process_many([_rec({"messages": []})])
    assert len(out) == 1
    assert out[0].meta.tombstone


def test_tombstones_pass_through_unchanged(fast_tokenizer: Any) -> None:
    op = TokenizeChat(fast_tokenizer, apply_chat_template=False)
    record = _rec({"messages": []})
    record.meta = record.meta.with_tombstone()
    out = op.process_many([record])
    assert out == [record]


def test_drop_keeps_batch_order(fast_tokenizer: Any) -> None:
    op = TokenizeChat(fast_tokenizer, apply_chat_template=False)
    keep = {"messages": _messages(("user", "hi ?"), ("assistant", "hello"))}
    out = op.process_many(
        [
            _rec(keep, (0, 0, 0)),
            _rec({"messages": []}, (0, 0, 1)),
            _rec(keep, (0, 0, 2)),
        ]
    )
    assert [r.meta.tombstone for r in out] == [False, True, False]
    assert [r.meta.sample_id for r in out] == [(0, 0, 0), (0, 0, 1), (0, 0, 2)]


# Payload contract


def test_mask_field_out_and_preserve_upstream(fast_tokenizer: Any) -> None:
    op = TokenizeChat(
        fast_tokenizer,
        apply_chat_template=False,
        mask_field_out="supervised",
        preserve_upstream_payload=True,
    )
    payload_in = {
        "messages": _messages(("user", "hi ?"), ("assistant", "hello")),
        "source": "unit",
    }
    record = _run_one(op, payload_in)
    payload = _payload(record)
    assert payload["source"] == "unit"
    assert "loss_mask" not in payload
    assert payload["supervised"].tolist() == [0, 0, 1, 1]


def test_default_payload_is_minimal(fast_tokenizer: Any) -> None:
    op = TokenizeChat(fast_tokenizer, apply_chat_template=False)
    record = _run_one(
        op,
        {
            "messages": _messages(("user", "hi ?"), ("assistant", "hello")),
            "source": "unit",
        },
    )
    payload = _payload(record)
    assert set(payload.keys()) == {"input_ids", "loss_mask"}


def test_per_sample_tools_and_enable_thinking_shape_the_render(
    fast_tokenizer: Any,
) -> None:
    op = TokenizeChat(
        fast_tokenizer, chat_template=TOOLS_TEMPLATE, span_source="prefix_diff"
    )
    messages = _messages(("user", "hi"), ("assistant", "hello"))
    plain = _run_one(op, {"messages": messages})
    with_tools = _run_one(op, {"messages": messages, "tools": [{"name": "calculator"}]})
    with_thinking = _run_one(op, {"messages": messages, "enable_thinking": True})
    sep = fast_tokenizer.convert_tokens_to_ids("<|sep|>")
    plain_ids = _payload(plain)["input_ids"].tolist()
    tool_ids = _payload(with_tools)["input_ids"].tolist()
    think_ids = _payload(with_thinking)["input_ids"].tolist()
    assert tool_ids == [sep, *plain_ids]
    assert think_ids == [sep, sep, *plain_ids]


# Setup and validation errors


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"span_source": cast(Any, "bogus")}, "span_source"),
        (
            {"apply_chat_template": False, "span_source": "prefix_diff"},
            "span_source only applies",
        ),
        (
            {"apply_chat_template": False, "chat_template": "t"},
            "chat_template has no effect",
        ),
        ({"max_length": 1}, "max_length"),
        ({"mask_field_out": "input_ids"}, "mask_field_out"),
        ({"chat_template_kwargs": {"tokenize": False}}, "operator-owned"),
    ],
)
def test_init_validation_errors(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TokenizeChat(tokenizer_id="x", **kwargs)


def test_forced_generation_tags_requires_tagged_template(fast_tokenizer: Any) -> None:
    op = TokenizeChat(
        fast_tokenizer,
        chat_template=UNTAGGED_TEMPLATE,
        span_source="generation_tags",
    )
    with pytest.raises(ValueError, match="generation"):
        op.process_many([_rec({"messages": CONVERSATION})])


def test_missing_chat_template_hard_errors(fast_tokenizer: Any) -> None:
    op = TokenizeChat(fast_tokenizer)  # tokenizer defines no template
    with pytest.raises(ValueError, match="no chat template"):
        op.process_many([_rec({"messages": CONVERSATION})])


def test_setup_error_is_cached(fast_tokenizer: Any) -> None:
    op = TokenizeChat(fast_tokenizer)
    for _ in range(2):
        with pytest.raises(ValueError, match="no chat template"):
            op.process_many([_rec({"messages": CONVERSATION})])


def test_fast_tokenizer_required() -> None:
    slow = SimpleNamespace(
        is_fast=False,
        eos_token="<e>",
        eos_token_id=2,
        chat_template="{{ messages }}",
        name_or_path="slow",
    )
    op = TokenizeChat(cast(Any, slow))
    with pytest.raises(ValueError, match="fast tokenizer"):
        op.process_many([_rec({"messages": CONVERSATION})])
    # The no-template path needs the offset mapping just the same.
    no_template = TokenizeChat(cast(Any, slow), apply_chat_template=False)
    with pytest.raises(ValueError, match="fast tokenizer"):
        no_template.process_many([_rec({"messages": CONVERSATION})])


def test_prefix_diff_requires_generation_prompt_support(fast_tokenizer: Any) -> None:
    template = (
        "{% for message in messages %}<|{{ message['role'] }}|>"
        "{{ message['content'] }}<|end|>{% endfor %}"
    )
    op = TokenizeChat(fast_tokenizer, chat_template=template, span_source="prefix_diff")
    with pytest.raises(ValueError, match="add_generation_prompt"):
        op.process_many([_rec({"messages": CONVERSATION})])


def test_no_template_requires_string_eos() -> None:
    no_eos = SimpleNamespace(
        is_fast=True, eos_token=None, eos_token_id=None, name_or_path="x"
    )
    op = TokenizeChat(cast(Any, no_eos), apply_chat_template=False)
    with pytest.raises(ValueError, match="eos_token"):
        op.process_many([_rec({"messages": CONVERSATION})])


def test_malformed_messages_error(fast_tokenizer: Any) -> None:
    op = TokenizeChat(fast_tokenizer, apply_chat_template=False)
    with pytest.raises(ValueError, match="no 'messages'"):
        op.process_many([_rec({"other": 1})])
    with pytest.raises(TypeError, match="expected a list"):
        op.process_many([_rec({"messages": "hi"})])
    with pytest.raises(TypeError, match="without a 'role'"):
        op.process_many([_rec({"messages": [{"content": "hi"}]})])


def test_chat_template_file_override(fast_tokenizer: Any, tmp_path: Any) -> None:
    path = tmp_path / "template.jinja"
    path.write_text(TAGGED_TEMPLATE, encoding="utf-8")
    op = TokenizeChat(fast_tokenizer, chat_template=path)
    record = _run_one(op, {"messages": CONVERSATION})
    assert _mask(record) == GOLDEN_MASK


# ---------------------------------------------------------------------------
# Token-counting spec (priming)
# ---------------------------------------------------------------------------


def test_chat_spec_counts_match_delivery(fast_tokenizer: Any) -> None:
    # max_length truncates the render, so count-equivalence here also proves the
    # rebuilt counter carried the op's max_length and chat_template — a config
    # drift would make the count diverge from what delivery actually emits.
    op = TokenizeChat(
        fast_tokenizer,
        chat_template=TAGGED_TEMPLATE,
        span_source="generation_tags",
        max_length=12,
    )
    delivered = _run_one(op, {"messages": CONVERSATION})
    payload = delivered.payload
    assert isinstance(payload, dict)
    assert len(payload["input_ids"]) == 12  # the render (22 tokens) was truncated

    plan = op.token_counting_spec().build_counter().plan([])
    assert plan.count({"messages": CONVERSATION}) == len(payload["input_ids"])
    assert plan.count({"messages": []}) == 0


def test_chat_spec_carries_eos_override_into_calibration(
    fast_tokenizer: Any, tmp_path: Any
) -> None:
    # The eos_token override moves the id the finalizer appends, so the
    # calibration counter must rebuild against it; without this the census
    # tokenizes with the tokenizer's default EOS and its counts drift from
    # delivery on the tokenizer_id load path.
    fast_tokenizer.save_pretrained(tmp_path)
    op = TokenizeChat(
        tokenizer_id=str(tmp_path),
        eos_token="<|sep|>",
        chat_template=TAGGED_TEMPLATE_NO_EOS,
        span_source="generation_tags",
    )
    spec = op.token_counting_spec()
    assert spec.eos_token == "<|sep|>"

    counter = spec.build_counter()
    assert counter.op.eos_token == "<|sep|>"
    # Counting triggers the rebuilt op's lazy load; the override must rebind
    # the EOS the counter tokenizes (and appends) against.
    counter.plan([]).count({"messages": CONVERSATION[:2]})
    assert counter.op._eos_id == fast_tokenizer.convert_tokens_to_ids("<|sep|>")


def test_count_delivered_tokens_raises_on_supervised_first_token(
    fast_tokenizer: Any,
) -> None:
    op = TokenizeChat(fast_tokenizer, apply_chat_template=False)
    with pytest.raises(ValueError, match="mask\\[0\\]"):
        op.count_delivered_tokens({"messages": _messages(("assistant", "hi"))})


def test_chat_plan_marks_delivery_error_fatal(fast_tokenizer: Any) -> None:
    # The plan wraps the op's real delivery error so priming treats it as fatal.
    op = TokenizeChat(fast_tokenizer, apply_chat_template=False)
    plan = op.token_counting_spec().build_counter().plan([])
    with pytest.raises(FatalCountError, match="mask\\[0\\]"):
        plan.count({"messages": _messages(("assistant", "hi"))})


def test_census_surfaces_structural_chat_error_past_coverage_gate(
    fast_tokenizer: Any,
) -> None:
    op = TokenizeChat(fast_tokenizer, apply_chat_template=False)
    good = {"messages": _messages(("user", "hi there"), ("assistant", "good morning"))}
    bad = {"messages": _messages(("assistant", "hi"))}  # supervised first token
    # One bad row in five: skipping them (the old behavior) leaves coverage far
    # above the 50% gate, so priming would have quietly accepted a ratio. It must
    # fail instead — _process_record crashes on these same rows at run time.
    rows = [bad if i % 5 == 0 else good for i in range(40)]
    ds = Dataset.from_dict(
        "sft", {0: InMemoryShard(rows[:20]), 1: InMemoryShard(rows[20:])}
    )
    store = build_multi_dataset_store({0: ds})
    counter = op.token_counting_spec().build_counter()
    with pytest.raises(FatalCountError, match="mask\\[0\\]"):
        _measure_dataset(
            ds, 0, store, TokenEstimation(calibration_samples=100), counter, seed=1
        )


def test_prime_token_ratios_hard_fails_on_structural_chat_error(
    fast_tokenizer: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # End-to-end through the spawn pool: the structural error must propagate out
    # of priming, not get re-swallowed into a fallback by the census catches
    # (both the per-future and the outer pool-level handler).
    monkeypatch.setenv("ZEPHON_CATALOG_DIR", str(tmp_path / "cat"))
    root = tmp_path / "sft"
    root.mkdir()
    bad = [{"role": "assistant", "content": "hi"}]  # supervised first token
    for s in range(2):
        lines = [json.dumps({"messages": bad, "id": f"{s}-{i}"}) for i in range(30)]
        (root / f"shard_{s:05d}.jsonl").write_text("\n".join(lines) + "\n")
    ds = Dataset.from_path("sft", str(root))
    op = TokenizeChat(fast_tokenizer, apply_chat_template=False)
    with pytest.raises(FatalCountError):
        prime_token_ratios(
            datasets=[ds],
            dataset_ids={"sft": 0},
            estimation=TokenEstimation(
                calibration_samples=50,
                calibration_shards_min=1,
                calibration_shards_max=2,
            ),
            counting_spec=op.token_counting_spec(),
            seed=1,
        )


def test_census_zero_yield_chat_drops_depress_the_ratio(fast_tokenizer: Any) -> None:
    op = TokenizeChat(fast_tokenizer, chat_template=TAGGED_TEMPLATE)
    supervised = {"messages": CONVERSATION}
    unsupervised = {"messages": _messages(("user", "no assistant turn here"))}
    rows = [supervised if i % 2 else unsupervised for i in range(20)]
    ds = Dataset.from_dict(
        "sft", {0: InMemoryShard(rows[:10]), 1: InMemoryShard(rows[10:])}
    )
    store = build_multi_dataset_store({0: ds})
    counter = op.token_counting_spec().build_counter()
    m = _measure_dataset(
        ds, 0, store, TokenEstimation(calibration_samples=100), counter, seed=1
    )
    # Zero-yield drops count toward coverage and lower the measured ratio.
    assert m.ratio.source == "measured"
    survivors_only = op.count_delivered_tokens(supervised) / content_bytes(supervised)
    assert 0 < m.ratio.tokens_per_byte < 0.9 * survivors_only
