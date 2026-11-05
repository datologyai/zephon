# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import pickle
import threading
from typing import Any, Callable

import numpy as np
import pytest

from tests.helpers.chat import (
    CONVERSATION,
    TAGGED_TEMPLATE,
    UNTAGGED_TEMPLATE,
    _messages,
    make_fast_tokenizer,
)
from zephon import Pipeline
from zephon._internal.ops.tokenize_chat import TokenizeChat
from zephon._internal.ops.tokenize_mixed import MixedTokenCountingSpec, TokenizeMixed
from zephon._internal.ops.tokenize_text import TokenizeText
from zephon.io import Dataset, InMemoryShard
from zephon.ops.base import OpContext
from zephon.types import SampleMeta, SampleRecord
from zephon.work import StaticMixtureWorkSource

TEXT = {"text": "hello there how are you good morning"}
LONG_TEXT = {"text": "the answer is what thanks yes no 1 2 4 12 + ? ! . w hello"}
CHAT = {"messages": CONVERSATION}
# No assistant turn: chat delivery drops it with a tombstone.
UNSUPERVISED_CHAT = {"messages": _messages(("user", "hi there"))}


@pytest.fixture(scope="module")
def tok() -> Any:
    # A BOS lets text use tokenize's default bos_eos brackets.
    tokenizer = make_fast_tokenizer()
    tokenizer.add_special_tokens({"bos_token": "<|start|>"})
    return tokenizer


def _rec(payload: Any, offset: int = 0) -> SampleRecord:
    meta = SampleMeta(
        sample_id=(0, 0, offset), lane_id=0, chunk_id=0, chunk_offset=offset
    )
    return SampleRecord(meta=meta, payload=payload)


def _setup(op: Any) -> Any:
    op.setup(OpContext({}))
    return op


def _mixed(tok: Any, **kwargs: Any) -> TokenizeMixed:
    kwargs.setdefault("chat_template", TAGGED_TEMPLATE)
    return _setup(TokenizeMixed(tok, **kwargs))


def _text_op(tok: Any, **kwargs: Any) -> TokenizeText:
    return _setup(TokenizeText(tok, field="text", add_attention_mask=False, **kwargs))


def _chat_op(tok: Any, **kwargs: Any) -> TokenizeChat:
    kwargs.setdefault("chat_template", TAGGED_TEMPLATE)
    return _setup(TokenizeChat(tok, **kwargs))


def _ids(record: SampleRecord) -> list[int]:
    return list(record.payload["input_ids"])


def _mask(record: SampleRecord) -> list[int]:
    return record.payload["loss_mask"].tolist()


def _all_but_first(record: SampleRecord) -> list[int]:
    return [0] + [1] * (len(_ids(record)) - 1)


# Each path matches its standalone operator


def test_defaults_match_the_standalone_operators(tok: Any) -> None:
    mixed = _mixed(tok)
    text, chat = mixed.process_many([_rec(dict(TEXT), 0), _rec(dict(CHAT), 1)])
    [text_expected] = _text_op(tok).process_many([_rec(dict(TEXT))])
    [chat_expected] = _chat_op(tok).process_many([_rec(dict(CHAT))])
    assert _ids(text) == _ids(text_expected)
    assert _ids(text)[0] == tok.bos_token_id and _ids(text)[-1] == tok.eos_token_id
    assert _mask(text) == _all_but_first(text)
    assert _ids(chat) == _ids(chat_expected)
    assert _mask(chat) == _mask(chat_expected)
    for record in (text, chat):
        assert set(record.payload) == {"input_ids", "loss_mask"}
        assert record.payload["input_ids"].dtype == np.int64
        assert record.payload["loss_mask"].dtype == np.uint8


@pytest.mark.parametrize(
    "special_tokens", ["bos_eos", "bos", "eos", "none", "tokenizer_default"]
)
def test_text_special_tokens_match_tokenize(tok: Any, special_tokens: Any) -> None:
    mixed = _mixed(tok, text_special_tokens=special_tokens)
    [out] = mixed.process_many([_rec(dict(TEXT))])
    [expected] = _text_op(tok, special_tokens=special_tokens).process_many(
        [_rec(dict(TEXT))]
    )
    assert _ids(out) == _ids(expected)
    assert _mask(out) == _all_but_first(out)


@pytest.mark.parametrize("template", [TAGGED_TEMPLATE, UNTAGGED_TEMPLATE])
@pytest.mark.parametrize("special_tokens", [None, "eos", "bos_eos", "none"])
@pytest.mark.parametrize("loss_scope", ["all_assistant", "last_span", "all_tokens"])
def test_chat_records_match_tokenize_chat(
    tok: Any, template: str, special_tokens: Any, loss_scope: Any
) -> None:
    options = {
        "chat_template": template,
        "loss_scope": loss_scope,
    }
    mixed = _mixed(tok, chat_special_tokens=special_tokens, **options)
    chat = _chat_op(tok, special_tokens=special_tokens, **options)
    [out] = mixed.process_many([_rec(dict(CHAT))])
    [expected] = chat.process_many([_rec(dict(CHAT))])
    assert _ids(out) == _ids(expected)
    assert _mask(out) == _mask(expected)


def test_per_path_special_token_ids(tok: Any) -> None:
    sep = tok.convert_tokens_to_ids("<|sep|>")
    mixed = _mixed(
        tok,
        text_special_tokens="eos",
        text_eos_token_id=sep,
        chat_special_tokens="bos_eos",
        chat_bos_token_id=sep,
    )
    text, chat = mixed.process_many([_rec(dict(TEXT), 0), _rec(dict(CHAT), 1)])
    assert _ids(text)[-1] == sep and _ids(text)[0] != sep
    assert _ids(chat)[0] == sep and _ids(chat)[-1] == tok.eos_token_id


def test_mask_field_out_applies_to_both_paths(tok: Any) -> None:
    mixed = _mixed(tok, mask_field_out="supervised")
    out = mixed.process_many([_rec(dict(TEXT), 0), _rec(dict(CHAT), 1)])
    assert all(set(r.payload) == {"input_ids", "supervised"} for r in out)


# max_length: each path keeps its standalone overflow handling


def test_max_length_follows_each_operator(tok: Any) -> None:
    # Like tokenize, text ignores max_length without truncation or splitting;
    # like tokenize_chat, chat truncates.
    mixed = _mixed(tok, max_length=6)
    text, chat = mixed.process_many([_rec(dict(LONG_TEXT), 0), _rec(dict(CHAT), 1)])
    [text_expected] = _text_op(tok, max_length=6).process_many([_rec(dict(LONG_TEXT))])
    [chat_expected] = _chat_op(tok, max_length=6).process_many([_rec(dict(CHAT))])
    assert _ids(text) == _ids(text_expected) and len(_ids(text)) > 6
    assert _ids(chat) == _ids(chat_expected) and len(_ids(chat)) <= 6


def test_text_truncation_matches_tokenize(tok: Any) -> None:
    mixed = _mixed(tok, max_length=6, truncation=True)
    [out] = mixed.process_many([_rec(dict(LONG_TEXT))])
    [expected] = _text_op(tok, max_length=6, truncation=True).process_many(
        [_rec(dict(LONG_TEXT))]
    )
    assert _ids(out) == _ids(expected) and len(_ids(out)) == 6


@pytest.mark.parametrize("path", ["text", "chat"])
def test_split_matches_the_standalone_operator(tok: Any, path: str) -> None:
    if path == "text":
        payload = dict(LONG_TEXT)
        mixed = _mixed(tok, max_length=5, split_long_samples=True)
        standalone: Any = _text_op(tok, max_length=5, split_long_samples=True)
    else:
        payload = dict(CHAT)
        mixed = _mixed(tok, max_length=8, split_long_samples=True)
        standalone = _chat_op(tok, max_length=8, split_long_samples=True)
    out = mixed.process_many([_rec(dict(payload))])
    expected = standalone.process_many([_rec(dict(payload))])
    assert len(out) > 1
    assert [r.meta.lineage for r in out] == [r.meta.lineage for r in expected]
    assert [_ids(r) for r in out] == [_ids(r) for r in expected]
    assert all(_mask(r)[0] == 0 for r in out)
    if path == "chat":
        assert [_mask(r) for r in out] == [_mask(r) for r in expected]


# Routing


def test_null_chat_field_routes_to_text(tok: Any) -> None:
    mixed = _mixed(tok)
    [merged_schema] = mixed.process_many([_rec({**TEXT, "messages": None})])
    [plain] = mixed.process_many([_rec(dict(TEXT))])
    assert _ids(merged_schema) == _ids(plain)
    [chat_wins] = mixed.process_many([_rec({**TEXT, **CHAT})])
    assert tok.convert_tokens_to_ids("<|assistant|>") in _ids(chat_wins)


def test_nested_chat_field(tok: Any) -> None:
    mixed = _mixed(tok, chat_field="conv.messages")
    [out] = mixed.process_many([_rec({"conv": dict(CHAT)})])
    assert 1 in _mask(out) and _mask(out)[0] == 0
    [text] = mixed.process_many([_rec({"conv": {}, **TEXT})])
    assert _mask(text) == _all_but_first(text)


def test_non_mapping_payload_is_text(tok: Any) -> None:
    mixed = _mixed(tok)
    [out] = mixed.process_many([_rec(TEXT["text"])])
    [expected] = mixed.process_many([_rec(dict(TEXT))])
    assert _ids(out) == _ids(expected)


# Ordering, splitting, drops


def _interleaved() -> list[SampleRecord]:
    tombstone = _rec({"text": "unused"}, 2)
    tombstone.meta = tombstone.meta.with_tombstone()
    payloads: list[Any] = [
        dict(LONG_TEXT),  # splits
        dict(CHAT),  # splits
        None,  # tombstone slot
        dict(LONG_TEXT),  # adjacent to the first in the text sub-batch
        dict(TEXT),
        dict(UNSUPERVISED_CHAT),  # dropped
        dict(LONG_TEXT),
    ]
    return [
        tombstone if payload is None else _rec(payload, offset)
        for offset, payload in enumerate(payloads)
    ]


def _split_mixed(tok: Any) -> TokenizeMixed:
    return _mixed(tok, max_length=8, split_long_samples=True)


def test_batch_matches_record_by_record_and_keeps_order(tok: Any) -> None:
    batched = _split_mixed(tok).process_many(_interleaved())
    one_by_one = [
        out
        for record in _interleaved()
        for out in _split_mixed(tok).process_one(record)
    ]
    assert [(r.meta.chunk_offset, r.meta.lineage) for r in batched] == [
        (r.meta.chunk_offset, r.meta.lineage) for r in one_by_one
    ]
    for got, want in zip(batched, one_by_one):
        assert got.meta.tombstone == want.meta.tombstone
        if not got.meta.tombstone:
            assert _ids(got) == _ids(want)
            assert _mask(got) == _mask(want)
    offsets = [r.meta.chunk_offset for r in batched]
    assert offsets == sorted(offsets)
    text_children = [r for r in batched if r.meta.chunk_offset == 0]
    assert [r.meta.lineage for r in text_children] == [
        (i,) for i in range(len(text_children))
    ]
    assert len(text_children) > 1
    assert all(_mask(r)[0] == 0 for r in batched if not r.meta.tombstone)
    # Tombstone passes through; the unsupervised chat becomes one.
    assert [r.meta.tombstone for r in batched if r.meta.chunk_offset in (2, 5)] == [
        True,
        True,
    ]


# Validation


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"text_field": "x", "chat_field": "x"}, "must differ"),
        ({"split_long_samples": True}, "requires max_length"),
        (
            {"max_length": 8, "truncation": True, "split_long_samples": True},
            "mutually exclusive",
        ),
        ({"chat_bos_token_id": 3}, "bos_token_id only applies"),
        ({"text_special_tokens": "none", "text_bos_token_id": 3}, "bos_token_id"),
    ],
)
def test_invalid_configuration(tok: Any, kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TokenizeMixed(tok, **kwargs)


# Tokenizer sharing, lazy chat setup, pickling, token counting


def test_sub_operators_share_the_tokenizer_after_pickling(tok: Any) -> None:
    op = pickle.loads(pickle.dumps(TokenizeMixed(tok, chat_template=TAGGED_TEMPLATE)))
    _setup(op).process_many([_rec(dict(TEXT), 0), _rec(dict(CHAT), 1)])
    assert op.text_op.tok is op.tok and op.chat_op.tok is op.tok


def test_setup_does_not_reenter_the_free_threaded_setup_lock(
    tok: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zephon._internal.ops.tokenize_base as tokenize_base

    # Free-threaded builds serialize setup with a non-reentrant module lock.
    monkeypatch.setattr(tokenize_base, "_SETUP_LOCK", threading.Lock())
    op = _mixed(tok)

    def run(fn: Callable[[], Any]) -> None:
        worker = threading.Thread(target=fn, daemon=True)
        worker.start()
        worker.join(timeout=10)
        assert not worker.is_alive(), "setup deadlocked on the setup lock"

    run(lambda: op.process_many([_rec(dict(TEXT))]))
    assert op.text_op._tokenizer_instantiated
    assert not op.chat_op._tokenizer_instantiated
    run(lambda: op.process_many([_rec(dict(CHAT))]))
    assert op.chat_op._tokenizer_instantiated


def test_chat_setup_waits_for_the_first_chat_record(tok: Any) -> None:
    # The tokenizer has no chat template, so chat setup would fail.
    op = _setup(TokenizeMixed(tok))
    [text] = op.process_many([_rec(dict(TEXT))])
    assert _mask(text) == _all_but_first(text)
    for _ in range(2):  # The setup failure is cached and raised again.
        with pytest.raises(ValueError, match="no chat template"):
            op.process_many([_rec(dict(TEXT), 0), _rec(dict(CHAT), 1)])
    assert op.chat_op.tok is op.tok


def test_preflight_checks_the_unused_chat_path(tok: Any) -> None:
    rows = [dict(TEXT) for _ in range(4)]
    broken = Pipeline(_work_source(rows)).tokenize_mixed(tok)
    report = broken.preflight_tokenizers(strict=False)
    assert not report.ok
    assert "no chat template" in report.errors[0].message
    ok = Pipeline(_work_source(rows)).tokenize_mixed(tok, chat_template=TAGGED_TEMPLATE)
    assert ok.preflight_tokenizers().ok


@pytest.mark.parametrize("split", [False, True])
def test_token_counting_matches_delivery(tok: Any, split: bool) -> None:
    op = _mixed(
        tok,
        max_length=8,
        truncation=not split,
        split_long_samples=split,
    )
    payloads = [dict(TEXT), dict(LONG_TEXT), dict(CHAT), dict(UNSUPERVISED_CHAT)]
    spec = pickle.loads(pickle.dumps(op.token_counting_spec()))
    assert isinstance(spec, MixedTokenCountingSpec)
    plan = spec.build_counter().plan(payloads)
    for offset, payload in enumerate(payloads):
        delivered = [
            r for r in op.process_many([_rec(dict(payload), offset)]) if r.payload
        ]
        # Chat drops count as zero; every path counts its delivered tokens.
        assert plan.count(payload) == sum(len(_ids(r)) for r in delivered)


def test_token_counting_carries_eos_override() -> None:
    spec = TokenizeMixed(
        tokenizer_id="some/tokenizer", eos_token="<|end|>"
    ).token_counting_spec()
    assert spec.chat is not None and spec.text is not None
    assert spec.chat.eos_token == "<|end|>"
    assert spec.text.eos_token == "<|end|>"


# Pipeline


def _work_source(rows: list[dict[str, Any]], **kwargs: Any) -> StaticMixtureWorkSource:
    half = len(rows) // 2
    ds = Dataset.from_dict(
        "mixed", {0: InMemoryShard(rows[:half]), 1: InMemoryShard(rows[half:])}
    )
    return StaticMixtureWorkSource([ds], {"mixed": 1.0}, chunk_size=4, **kwargs)


def _rows() -> list[dict[str, Any]]:
    return [dict(TEXT) if i % 2 else dict(CHAT) for i in range(8)]


def _run(pipe: Callable[[], Pipeline]) -> list[Any]:
    return list(
        pipe().options(deterministic=True, max_workers=1, default_stage_prefetch=0)
    )


def test_pipeline_packs_text_and_chat_with_masked_labels(tok: Any) -> None:
    tokenized = _run(
        lambda: Pipeline(_work_source(_rows())).tokenize_mixed(
            tok, chat_template=TAGGED_TEMPLATE
        )
    )
    assert len(tokenized) == 8
    supervised = sum(int(r.payload["loss_mask"].sum()) for r in tokenized)

    batches = _run(
        lambda: (
            Pipeline(_work_source(_rows()))
            .tokenize_mixed(tok, chat_template=TAGGED_TEMPLATE)
            .pack_flat(
                max_length=64,
                algorithm="first_fit",
                num_bins=4,
                pad_token_id=tok.pad_token_id,
            )
            .batch(microbatch_size=1, drop_last=False)
        )
    )
    # Bins mix several text and chat documents.
    assert len(batches) < len(tokenized)
    valid = 0
    for batch in batches:
        result = batch.to_training(return_labels=True)
        valid += int((np.asarray(result["labels"]) != -100).sum())
    # Each document's first token is unsupervised, so shifting the packed mask
    # keeps exactly the per-document supervised tokens.
    assert valid == supervised


def test_pipeline_token_priming_uses_the_mixed_spec(
    tok: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zephon.work.static_mixture as sm
    from zephon.work.token_estimation import TokenEstimation

    captured: dict[str, Any] = {}
    real = sm.prime_token_ratios

    def capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(sm, "prime_token_ratios", capture)
    pipe = Pipeline(
        _work_source(_rows() * 8, token_estimation=TokenEstimation(primer=0.13))
    ).tokenize_mixed(tok, chat_template=TAGGED_TEMPLATE)
    assert pipe.preflight_tokenizers().ok
    pipe._prime_worksource()
    assert isinstance(captured["counting_spec"], MixedTokenCountingSpec)
