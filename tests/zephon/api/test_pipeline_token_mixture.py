# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Pipeline wiring for token-aware mixtures: prime hook + unit validation."""

import pytest

import zephon.work.static_mixture as sm
from zephon.api.pipeline import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.ops.tokenize_chat import ChatTokenCountingSpec
from zephon.work.static_mixture import StaticMixtureWorkSource
from zephon.work.token_counting import TextTokenCountingSpec
from zephon.work.token_estimation import TokenEstimation


def make_dataset(name: str, words_per_doc: int, n_docs: int = 200) -> Dataset:
    text = " ".join(f"{name}w{j}" for j in range(words_per_doc))
    return Dataset.from_dict(
        name,
        {
            0: InMemoryShard([{"text": text} for _ in range(n_docs // 2)]),
            1: InMemoryShard([{"text": text} for _ in range(n_docs // 2)]),
        },
    )


def make_token_ws() -> StaticMixtureWorkSource:
    short = make_dataset("short", 10)
    long = make_dataset("long", 40)
    # Pinned primer: the wiring under test is the prime hook, not the census,
    # and a pinned ratio keeps prime() free of the calibration process pool.
    return StaticMixtureWorkSource(
        [short, long],
        {"short": 0.5, "long": 0.5},
        chunk_size=64,
        token_estimation=TokenEstimation(primer=0.13),
    )


def test_iteration_primes_token_worksource() -> None:
    ws = make_token_ws()
    pipe = (
        Pipeline(ws)
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
        .options(deterministic=True, max_workers=1, default_stage_prefetch=0)
    )
    assert ws.requires_token_priming
    # Drain fully: closing a mid-stream engine waits out graceful worker joins.
    records = list(pipe)
    assert not ws.requires_token_priming
    assert records and "input_ids" in records[0].payload


def test_priming_passes_counting_spec_from_tokenize_op(monkeypatch) -> None:
    captured = {}
    real = sm.prime_token_ratios

    def capture(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(sm, "prime_token_ratios", capture)
    pipe = Pipeline(make_token_ws()).tokenize(
        tokenizer_id="__fallback__",
        field="text",
        special_tokens="none",
        truncation=True,
        max_length=5,
    )
    pipe._prime_worksource()
    spec = captured["counting_spec"]
    assert spec.tokenizer_id == "__fallback__"
    assert spec.field == "text"
    assert spec.special_tokens == "none"
    assert spec.truncation
    assert spec.max_length == 5


def test_priming_passes_chat_counting_spec_from_chat_op(monkeypatch) -> None:
    captured = {}
    real = sm.prime_token_ratios

    def capture(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(sm, "prime_token_ratios", capture)
    pipe = Pipeline(make_token_ws()).tokenize_chat(
        tokenizer_id="__fallback__", max_length=7
    )
    pipe._prime_worksource()
    spec = captured["counting_spec"]
    assert isinstance(spec, ChatTokenCountingSpec)
    assert spec.tokenizer_id == "__fallback__"
    assert spec.max_length == 7


def test_prime_skipped_when_restore_pending() -> None:
    ws = make_token_ws()
    pipe = Pipeline(ws).tokenize(
        tokenizer_id="__fallback__", field="text", special_tokens="none"
    )
    pipe._pending_restore = {"not": "consulted here"}
    pipe._prime_worksource()
    # Pending restores use checkpointed ratios.
    assert ws.requires_token_priming


def test_sample_mode_pipeline_never_primes() -> None:
    ds = make_dataset("a", 5)
    ws = StaticMixtureWorkSource([ds], {"a": 1.0}, chunk_size=32)
    pipe = Pipeline(ws).tokenize(
        tokenizer_id="__fallback__", field="text", special_tokens="none"
    )
    pipe._prime_worksource()


def test_token_mode_rejects_sample_weighted_ensure_mixture() -> None:
    pipe = (
        Pipeline(make_token_ws())
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
        .ensure_mixture(weight_by="samples")
    )
    with pytest.raises(ValueError, match="weight_by='samples'"):
        pipe._prime_worksource()


def test_token_mode_rejects_ensure_mixture_before_tokenize() -> None:
    pipe = (
        Pipeline(make_token_ws())
        .ensure_mixture()
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
    )
    with pytest.raises(ValueError, match="after tokenize"):
        pipe._prime_worksource()


def test_token_mode_rejects_ensure_mixture_before_chat_tokenize() -> None:
    pipe = (
        Pipeline(make_token_ws())
        .ensure_mixture()
        .tokenize_chat(tokenizer_id="__fallback__")
    )
    with pytest.raises(ValueError, match="after tokenize"):
        pipe._prime_worksource()


def test_token_mode_allows_token_weighted_ensure_mixture_after_tokenize() -> None:
    pipe = (
        Pipeline(make_token_ws())
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
        .ensure_mixture(max_buffer_size=128, weight_by="input_ids")
    )
    pipe._prime_worksource()


def test_multi_tokenize_priming_warns_and_uses_first() -> None:
    pipe = (
        Pipeline(make_token_ws())
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
    )
    with pytest.warns(RuntimeWarning, match="first of 2 tokenize ops"):
        pipe._prime_worksource()


def test_mixed_tokenize_ops_warn_and_prime_with_first(monkeypatch) -> None:
    captured = {}
    real = sm.prime_token_ratios

    def capture(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(sm, "prime_token_ratios", capture)
    pipe = (
        Pipeline(make_token_ws())
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
        .tokenize_chat(tokenizer_id="__fallback__")
    )
    with pytest.warns(RuntimeWarning, match="first of 2 tokenize ops"):
        pipe._prime_worksource()
    assert isinstance(captured["counting_spec"], TextTokenCountingSpec)


def test_checkpoint_before_iteration_primes_and_restores() -> None:
    """A pre-iteration checkpoint must not serialize unprimed token lanes."""

    def build() -> Pipeline:
        return (
            Pipeline(make_token_ws())
            .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
            .options(deterministic=True, max_workers=1, default_stage_prefetch=0)
        )

    pipe = build()
    assert pipe.ws.requires_token_priming
    ckpt = pipe.checkpoint()
    assert not pipe.ws.requires_token_priming
    if pipe._engine is not None:
        pipe._engine.close()

    restored = build()
    restored.restore(ckpt)
    records = list(restored)
    assert records


def test_to_torch_dataset_primes_in_driver() -> None:
    """to_torch_dataset primes before DataLoader worker pickling."""
    pytest.importorskip("torch.utils.data")
    ws = make_token_ws()
    pipe = Pipeline(ws).tokenize(
        tokenizer_id="__fallback__", field="text", special_tokens="none"
    )
    assert ws.requires_token_priming
    pipe.to_torch_dataset(stateful=False)
    assert not ws.requires_token_priming
