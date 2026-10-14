# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from tests.helpers.chat import (
    CONVERSATION,
    CYCLE,
    GOLDEN_MASK,
    GOLDEN_TOKENS,
    POSITION_DEPENDENT_TEMPLATE,
    SPLIT,
    TAGGED_TEMPLATE,
    TAGGED_TEMPLATE_NO_EOS,
    TOOLS_TEMPLATE,
    UNTAGGED_TEMPLATE,
    _messages,
    make_fast_tokenizer,
)
from zephon._internal.io.stores.multi import build_multi_dataset_store
from zephon._internal.observability.size_estimator import content_bytes
from zephon._internal.ops.tokenize_chat import (
    TokenizeChat,
    _last_span_only,
    _mask_from_spans,
    _thread_salted_template,
)
from zephon._internal.token_counting import FatalCountError
from zephon.io import Dataset, InMemoryShard
from zephon.ops.config import SpanSource
from zephon.types import SampleMeta, SampleRecord
from zephon.validation import ValidationError, verify_chat_template
from zephon.work.token_estimation import (
    TokenEstimation,
    _measure_dataset,
    prime_token_ratios,
)


@pytest.fixture(scope="module")
def fast_tokenizer() -> Any:
    return make_fast_tokenizer()


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

    with caplog.at_level(logging.WARNING, logger="zephon._internal.ops.tokenize_chat"):
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


def test_last_span_scope(fast_tokenizer: Any) -> None:
    cases: tuple[tuple[str, SpanSource], ...] = (
        (TAGGED_TEMPLATE, "generation_tags"),
        (UNTAGGED_TEMPLATE, "prefix_diff"),
    )
    for template, source in cases:
        op = TokenizeChat(
            fast_tokenizer,
            chat_template=template,
            span_source=source,
            loss_scope="last_span",
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


# ---------------------------------------------------------------------------
# Thread safety: transformers' process-wide compiled-template cache
# ---------------------------------------------------------------------------


def _multi_turn(n_pairs: int = 12) -> list[dict[str, Any]]:
    pair = (("user", "what is 2 + 2 ?"), ("assistant", "the answer is 4"))
    return _messages(*(pair * n_pairs))


@pytest.mark.parametrize(
    "tail",
    [
        "",
        " done",
        " done   ",
        " done\n",
        " done\n   ",
        " done\n\t",
        " done\r\n",
        " done\r",
        " done\n\n",
    ],
    ids=[
        "tag-end",
        "text-end",
        "spaces-same-line",
        "trailing-newline",
        "newline-spaces",
        "newline-tab",
        "crlf",
        "cr",
        "double-newline",
    ],
)
def test_thread_salt_render_is_byte_identical(fast_tokenizer: Any, tail: str) -> None:
    # Whitespace tails probe lstrip_blocks and jinja's trailing-newline strip,
    # which a naive template suffix would disturb.
    template = TAGGED_TEMPLATE + tail
    base = fast_tokenizer.apply_chat_template(
        CONVERSATION, chat_template=template, tokenize=False
    )
    rendered = fast_tokenizer.apply_chat_template(
        CONVERSATION, chat_template=_thread_salted_template(template), tokenize=False
    )
    assert rendered == base


def test_concurrent_masked_renders_share_no_tracker_state(fast_tokenizer: Any) -> None:
    """Deepcopied ops rendering one template in parallel must not collide.

    transformers compiles a chat template into one process-wide cached object,
    so without the per-thread salt overlapping assistant-mask renders raise
    "AssistantTracker should not be reused before closed" — or complete with
    corrupted all-zero masks that silently drop the sample.
    """
    n_threads, iterations = 4, 60
    messages = _multi_turn()
    proto = TokenizeChat(
        fast_tokenizer, chat_template=TAGGED_TEMPLATE, span_source="generation_tags"
    )
    reference = _payload(_run_one(copy.deepcopy(proto), {"messages": messages}))

    # The thread runner deepcopies the op proto per worker.
    ops = [copy.deepcopy(proto) for _ in range(n_threads)]
    barrier = threading.Barrier(n_threads)
    failures: list[BaseException] = []
    outputs: list[SampleRecord] = []
    lock = threading.Lock()

    def worker(op: TokenizeChat) -> None:
        barrier.wait()
        for i in range(iterations):
            rec = _rec({"messages": messages}, sample_id=(0, 0, i))
            try:
                (out,) = op.process_many([rec])
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    failures.append(exc)
                return
            with lock:
                outputs.append(out)

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)  # frequent handoffs land mid-render
    try:
        threads = [threading.Thread(target=worker, args=(op,)) for op in ops]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old_interval)

    assert not failures, failures[:3]
    assert len(outputs) == n_threads * iterations
    for out in outputs:
        payload = _payload(out)
        assert np.array_equal(payload["input_ids"], reference["input_ids"])
        assert np.array_equal(payload["loss_mask"], reference["loss_mask"])
        assert int(payload["loss_mask"].sum()) > 0


def run(op: TokenizeChat, messages: list[dict[str, Any]]) -> SampleRecord:
    return op.process_one(
        SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
            payload={"messages": copy.deepcopy(messages)},
        )
    )[0]


@pytest.mark.parametrize("mode", ["generation_tags", "prefix_diff", "literal"])
@pytest.mark.parametrize("scope", ["all_assistant", "last_span", "after_last_user"])
def test_scope_targets_by_source(fast_tokenizer: Any, mode: str, scope: str) -> None:
    kwargs = {
        "chat_template": TAGGED_TEMPLATE
        if mode == "generation_tags"
        else UNTAGGED_TEMPLATE,
        "span_source": mode,
    }
    if mode == "literal":
        kwargs = {"apply_chat_template": False}
    actual = run(TokenizeChat(fast_tokenizer, loss_scope=scope, **kwargs), CONVERSATION)
    baseline = run(TokenizeChat(fast_tokenizer, **kwargs), CONVERSATION)
    assert np.array_equal(actual.payload["input_ids"], baseline.payload["input_ids"])
    assert np.all(actual.payload["loss_mask"] <= baseline.payload["loss_mask"])
    if scope != "all_assistant":
        ids = actual.payload["input_ids"][actual.payload["loss_mask"].astype(bool)]
        assert fast_tokenizer.decode(ids) == "the answer is 4 <|end|>"


def test_user_cycle_includes_calls_reasoning_and_answers_excludes_tools(
    fast_tokenizer: Any,
) -> None:
    after = run(
        TokenizeChat(fast_tokenizer, chat_template=SPLIT, loss_scope="after_last_user"),
        CYCLE,
    )
    last = run(
        TokenizeChat(fast_tokenizer, chat_template=SPLIT, loss_scope="last_span"),
        CYCLE,
    )
    ids, mask = after.payload["input_ids"], after.payload["loss_mask"]
    assert not mask[ids == fast_tokenizer.convert_tokens_to_ids("hi")].any()
    assert not mask[ids == fast_tokenizer.convert_tokens_to_ids("12")].any()
    assert mask[ids == fast_tokenizer.convert_tokens_to_ids("how")].all()
    assert mask[ids == fast_tokenizer.convert_tokens_to_ids("yes")].all()
    assert mask.sum() > last.payload["loss_mask"].sum()
    assert not last.payload["loss_mask"][
        ids == fast_tokenizer.convert_tokens_to_ids("how")
    ].any()


@pytest.mark.parametrize(
    "template",
    [
        TAGGED_TEMPLATE + "{% generation %}{{eos_token}}{% endgeneration %}",
        "{% for m in messages %}{% set x %}{% generation %}{{m.content}}{% endgeneration %}{% endset %}{{x}}{% endfor %}",
        "{% for m in messages %}{% generation %}{% generation %}{{m.content}}{% endgeneration %}{% endgeneration %}{% endfor %}",
        "{% include 'other.jinja' %}" + TAGGED_TEMPLATE,
    ],
)
def test_invalid_structure_fails_without_preflight(
    fast_tokenizer: Any, template: str
) -> None:
    op = TokenizeChat(
        fast_tokenizer, chat_template=template, loss_scope="after_last_user"
    )
    with pytest.raises((ValueError, ValidationError)):
        run(op, CONVERSATION)
    assert op._setup_error is not None
    with pytest.raises(FatalCountError):
        op.token_counting_spec().build_counter().plan([]).count(
            {"messages": CONVERSATION}
        )


def test_profile_detects_omitted_calls_before_emission(fast_tokenizer: Any) -> None:
    op = TokenizeChat(
        fast_tokenizer, chat_template=TAGGED_TEMPLATE, loss_scope="after_last_user"
    )
    assert not run(op, CONVERSATION).meta.tombstone
    with pytest.raises(ValidationError, match="CHAT_REQUIRED_CONTENT"):
        run(op, CYCLE)


def test_tool_content_contamination_detected(fast_tokenizer: Any) -> None:
    template = SPLIT.replace(
        "{{m.content}}<|end|>{% endgeneration %}",
        '{{m.content}}{% if loop.previtem is defined and loop.previtem.role == "tool" %}{{loop.previtem.content}}{% endif %}<|end|>{% endgeneration %}',
    )
    with pytest.raises(ValidationError, match="CHAT_CONTENT_OWNERSHIP"):
        run(
            TokenizeChat(
                fast_tokenizer, chat_template=template, loss_scope="after_last_user"
            ),
            CYCLE,
        )


def test_truncation_never_retargets_earlier_assistant(fast_tokenizer: Any) -> None:
    op = TokenizeChat(
        fast_tokenizer,
        chat_template=TAGGED_TEMPLATE,
        loss_scope="after_last_user",
        max_length=10,
    )
    assert run(op, CONVERSATION).meta.tombstone
    assert run(
        TokenizeChat(
            fast_tokenizer, chat_template=TAGGED_TEMPLATE, loss_scope="after_last_user"
        ),
        CONVERSATION + [{"role": "user", "content": "hello"}],
    ).meta.tombstone


def test_default_and_last_span_use_hf_without_ownership_verifier(
    fast_tokenizer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zephon._internal.ops import chat_template

    monkeypatch.setattr(
        chat_template,
        "TemplateVerifier",
        lambda *a, **k: pytest.fail("unexpected verifier"),
    )
    for scope in ("all_assistant", "last_span"):
        output = run(
            TokenizeChat(
                fast_tokenizer, chat_template=TAGGED_TEMPLATE, loss_scope=scope
            ),
            CONVERSATION,
        )
        assert output.payload["loss_mask"].any()


def test_deprecated_flag_and_conflict(fast_tokenizer: Any) -> None:
    with pytest.warns(FutureWarning):
        old = TokenizeChat(
            fast_tokenizer, chat_template=TAGGED_TEMPLATE, loss_on_last_turn_only=True
        )
    new = TokenizeChat(
        fast_tokenizer, chat_template=TAGGED_TEMPLATE, loss_scope="last_span"
    )
    assert np.array_equal(
        run(old, CONVERSATION).payload["loss_mask"],
        run(new, CONVERSATION).payload["loss_mask"],
    )
    with pytest.raises(ValueError, match="not both"):
        TokenizeChat(
            fast_tokenizer, loss_scope="last_span", loss_on_last_turn_only=False
        )
    with pytest.raises(ValueError, match="explicit"):
        run(
            TokenizeChat(
                fast_tokenizer,
                chat_template=UNTAGGED_TEMPLATE,
                loss_scope="after_last_user",
            ),
            CONVERSATION,
        )


def test_shared_instance_rendering_and_copy(fast_tokenizer: Any) -> None:
    op = TokenizeChat(fast_tokenizer, chat_template=SPLIT, loss_scope="after_last_user")
    expected = run(op, CYCLE).payload["loss_mask"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(
            pool.map(lambda _: run(op, CYCLE).payload["loss_mask"], range(24))
        )
    assert all(np.array_equal(mask, expected) for mask in actual)
    assert np.array_equal(run(copy.deepcopy(op), CYCLE).payload["loss_mask"], expected)


def test_input_shapes_checked_automatically(fast_tokenizer: Any) -> None:
    from zephon._internal.ops.chat_template import OwnedTemplate

    nulls = [{**m, "tool_calls": None, "reasoning_content": None} for m in CONVERSATION]
    op = TokenizeChat(fast_tokenizer, chat_template=SPLIT, loss_scope="after_last_user")
    assert not run(op, nulls).meta.tombstone
    bad_presence = TAGGED_TEMPLATE.replace(
        "message['role'] == 'assistant'",
        "message['role'] == 'assistant' or 'tool_calls' in message",
    )
    with pytest.raises(ValidationError, match="not assistant"):
        run(
            TokenizeChat(
                fast_tokenizer, chat_template=bad_presence, loss_scope="after_last_user"
            ),
            nulls,
        )
    lists = [
        {**m, "content": [{"type": "text", "text": m["content"]}]} for m in CONVERSATION
    ]
    drops_lists = TAGGED_TEMPLATE.replace(
        "{{ message['content'] }}",
        "{{ message['content'] if message['content'] is string else '' }}",
    )
    with pytest.raises(ValidationError, match="CHAT_REQUIRED_CONTENT"):
        run(
            TokenizeChat(
                fast_tokenizer, chat_template=drops_lists, loss_scope="after_last_user"
            ),
            lists,
        )
    messages = [CONVERSATION[0], CONVERSATION[1], CONVERSATION[0]]
    text, spans = OwnedTemplate(TAGGED_TEMPLATE).render(messages, {})
    assert text.count("hi there") == 2
    assert {owner for owner, *_ in spans} == {1}


def test_profile_key_covers_reasoning_flags_and_json_argument_shape(
    fast_tokenizer: Any,
) -> None:
    from zephon._internal.ops.chat_template import _profile

    assert _profile({"arguments": '{"a":"x"}'}) != _profile({"arguments": '{"b":"x"}'})
    template = (
        "{% if enable_thinking %}"
        + SPLIT
        + "{% else %}"
        + TAGGED_TEMPLATE
        + "{% endif %}"
    )
    op = TokenizeChat(
        fast_tokenizer, chat_template=template, loss_scope="after_last_user"
    )
    op._ids_and_mask({"messages": CYCLE, "enable_thinking": True}, "fixture")
    with pytest.raises(ValidationError, match="CHAT_REQUIRED_CONTENT"):
        op._ids_and_mask({"messages": CYCLE, "enable_thinking": False}, "fixture")


def test_real_rows_check_missing_owners_after_profile_cache(
    fast_tokenizer: Any,
) -> None:
    template = TAGGED_TEMPLATE.replace(
        "message['role'] == 'assistant'",
        "message['role'] == 'assistant' and message.content != 'skip'",
    )
    op = TokenizeChat(
        fast_tokenizer, chat_template=template, loss_scope="after_last_user"
    )
    run(op, CONVERSATION)
    changed = copy.deepcopy(CONVERSATION)
    changed[-1]["content"] = "skip"
    with pytest.raises(ValueError, match="emitted no generation range"):
        run(op, changed)


def test_role_and_multimodal_validation_precedes_cached_profile(
    fast_tokenizer: Any,
) -> None:
    op = TokenizeChat(fast_tokenizer, chat_template=SPLIT, loss_scope="after_last_user")
    run(op, CONVERSATION)
    wrapped = copy.deepcopy(CONVERSATION)
    wrapped[-2]["content"] = "<tool_response>result</tool_response>"
    assert not run(op, wrapped).meta.tombstone
    multimodal = copy.deepcopy(CONVERSATION)
    multimodal[0]["content"] = [{"type": "image", "image": "unused"}]
    with pytest.raises(ValueError, match="text-only"):
        run(op, multimodal)


def test_user_cycle_includes_call_before_empty_final_assistant(
    fast_tokenizer: Any,
) -> None:
    messages = copy.deepcopy(CYCLE)
    messages[-1] = {"role": "assistant", "content": ""}
    assert not run(
        TokenizeChat(fast_tokenizer, chat_template=SPLIT, loss_scope="after_last_user"),
        messages,
    ).meta.tombstone


def test_rebinding_message_loop_is_rejected(fast_tokenizer: Any) -> None:
    template = TAGGED_TEMPLATE.replace(
        "{% generation %}", "{% set message = messages[-1] %}{% generation %}"
    )
    with pytest.raises(ValueError, match="rebinding"):
        run(
            TokenizeChat(
                fast_tokenizer, chat_template=template, loss_scope="after_last_user"
            ),
            CONVERSATION,
        )


def test_untagged_history_is_allowed(fast_tokenizer: Any) -> None:
    template = (
        "{% set ns=namespace(user=0) %}{% for m in messages %}{% if m.role == 'user' %}{% set ns.user=loop.index0 %}{% endif %}{% endfor %}"
        + TAGGED_TEMPLATE.replace(
            "message['role'] == 'assistant'",
            "message['role'] == 'assistant' and loop.index0 > ns.user",
        )
    )
    report = verify_chat_template(fast_tokenizer, template)
    assert report.ok, report.format()
    result = run(
        TokenizeChat(
            fast_tokenizer, chat_template=template, loss_scope="after_last_user"
        ),
        CONVERSATION,
    )
    assert (
        fast_tokenizer.decode(
            result.payload["input_ids"][result.payload["loss_mask"].astype(bool)]
        )
        == "the answer is 4 <|end|>"
    )


def test_empty_generated_range_for_nonempty_selected_message_raises(
    fast_tokenizer: Any,
) -> None:
    template = TAGGED_TEMPLATE.replace(
        "{{ message['content'] }}<|end|>",
        "{{ message.content if message.content != 'skip' else '' }}",
    )
    op = TokenizeChat(
        fast_tokenizer, chat_template=template, loss_scope="after_last_user"
    )
    run(op, CONVERSATION)
    changed = copy.deepcopy(CONVERSATION)
    changed[-1]["content"] = "skip"
    with pytest.raises(ValueError, match="emitted no generation range"):
        run(op, changed)


def test_rendered_prefix_guard_does_not_prove_message_selection(
    fast_tokenizer: Any,
) -> None:
    template = (
        "{% for m in messages if m.role == 'user' %}<|user|>{{m.content}}<|end|>{% endfor %}"
        "{% if messages[-1].role == 'assistant' %}{% for m in messages if m.role == 'assistant' %}"
        "<|assistant|>{% generation %}{{m.content}}<|end|>{% endgeneration %}{% endfor %}{% endif %}"
    )
    full = fast_tokenizer.apply_chat_template(
        CONVERSATION, chat_template=template, tokenize=False
    )
    prefix = fast_tokenizer.apply_chat_template(
        CONVERSATION[:3], chat_template=template, tokenize=False
    )
    assert full.startswith(prefix)
    hf = fast_tokenizer.apply_chat_template(
        CONVERSATION,
        chat_template=template,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    boundary_mask = np.asarray(hf["assistant_masks"])
    boundary_mask[: hf.char_to_token(0, len(prefix))] = 0
    assert "good morning" in fast_tokenizer.decode(
        np.asarray(hf["input_ids"])[boundary_mask.astype(bool)]
    )
    result = run(
        TokenizeChat(
            fast_tokenizer, chat_template=template, loss_scope="after_last_user"
        ),
        CONVERSATION,
    )
    assert (
        fast_tokenizer.decode(
            result.payload["input_ids"][result.payload["loss_mask"].astype(bool)]
        )
        == "the answer is 4 <|end|>"
    )


@pytest.mark.parametrize(
    ("scope", "key"),
    [
        ("all_assistant", "tokenizer_kwargs"),
        ("last_span", "padding"),
        ("after_last_user", "continue_final_message"),
        ("after_last_user", "messages"),
        ("after_last_user", "return_offsets_mapping"),
    ],
)
def test_reserved_controls_apply_to_every_scope(
    fast_tokenizer: Any, scope: str, key: str
) -> None:
    with pytest.raises(ValueError, match="operator-owned"):
        TokenizeChat(fast_tokenizer, loss_scope=scope, chat_template_kwargs={key: None})


def test_tool_loop_rebinding_inside_generation_retains_message_owner(
    fast_tokenizer: Any,
) -> None:
    template = SPLIT.replace(
        "{{m.tool_calls|tojson}}",
        "{% for call in m.tool_calls %}{% set call = call.function %}{{call|tojson}}{% endfor %}",
    )
    result = run(
        TokenizeChat(
            fast_tokenizer, chat_template=template, loss_scope="after_last_user"
        ),
        CYCLE,
    )
    assert result.payload["loss_mask"].any()
