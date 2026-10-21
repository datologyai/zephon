# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Pipeline wiring for token-aware mixture priming."""

from typing import Any

import pytest

import zephon.work.static_mixture as sm
from zephon import Pipeline as PublicPipeline
from zephon._internal.ops.tokenize_chat import ChatTokenCountingSpec
from zephon._internal.token_counting import TextTokenCountingSpec
from zephon.io import Dataset, InMemoryShard
from zephon.ops import BaseOp, OpTraits
from zephon.types import SampleRecord
from zephon.work import StaticMixtureWorkSource
from zephon.work.token_estimation import (
    TokenEstimation,
    _PreTokenizeReplay,
    _UnreplayableOp,
)


def _make_token_dataset(name: str, words_per_doc: int, n_docs: int = 200) -> Dataset:
    text = " ".join(f"{name}w{j}" for j in range(words_per_doc))
    return Dataset.from_dict(
        name,
        {
            0: InMemoryShard([{"text": text} for _ in range(n_docs // 2)]),
            1: InMemoryShard([{"text": text} for _ in range(n_docs // 2)]),
        },
    )


def _make_token_work_source() -> StaticMixtureWorkSource:
    short = _make_token_dataset("short", 10)
    long = _make_token_dataset("long", 40)
    return StaticMixtureWorkSource(
        [short, long],
        {"short": 0.5, "long": 0.5},
        chunk_size=64,
        token_estimation=TokenEstimation(primer=0.13),
    )


def _capture_token_prime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    real = sm.prime_token_ratios

    def capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(sm, "prime_token_ratios", capture)
    return captured


def test_iteration_primes_token_worksource() -> None:
    ws = _make_token_work_source()
    pipe = (
        PublicPipeline(ws)
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
        .options(deterministic=True, max_workers=1, default_stage_prefetch=0)
    )
    assert ws.requires_token_priming
    records = list(pipe)
    assert not ws.requires_token_priming
    assert records and "input_ids" in records[0].payload


def test_priming_passes_counting_spec_from_tokenize_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_token_prime(monkeypatch)
    pipe = PublicPipeline(_make_token_work_source()).tokenize(
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


def test_priming_passes_chat_counting_spec_from_chat_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_token_prime(monkeypatch)
    pipe = PublicPipeline(_make_token_work_source()).tokenize_chat(
        tokenizer_id="__fallback__", max_length=7
    )
    pipe._prime_worksource()
    spec = captured["counting_spec"]
    assert isinstance(spec, ChatTokenCountingSpec)
    assert spec.tokenizer_id == "__fallback__"
    assert spec.max_length == 7


def test_prime_skipped_when_restore_pending() -> None:
    ws = _make_token_work_source()
    pipe = PublicPipeline(ws).tokenize(
        tokenizer_id="__fallback__", field="text", special_tokens="none"
    )
    pipe._pending_restore = {"not": "consulted here"}
    pipe._prime_worksource()
    assert ws.requires_token_priming


def test_sample_mode_pipeline_never_primes() -> None:
    ds = _make_token_dataset("a", 5)
    ws = StaticMixtureWorkSource([ds], {"a": 1.0}, chunk_size=32)
    pipe = PublicPipeline(ws).tokenize(
        tokenizer_id="__fallback__", field="text", special_tokens="none"
    )
    pipe._prime_worksource()


def test_token_mode_rejects_sample_weighted_ensure_mixture() -> None:
    pipe = (
        PublicPipeline(_make_token_work_source())
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
        .ensure_mixture(weight_by="samples")
    )
    with pytest.raises(ValueError, match="weight_by='samples'"):
        pipe._prime_worksource()


def test_token_mode_rejects_ensure_mixture_before_tokenize() -> None:
    pipe = (
        PublicPipeline(_make_token_work_source())
        .ensure_mixture()
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
    )
    with pytest.raises(ValueError, match="after tokenize"):
        pipe._prime_worksource()


def test_token_mode_rejects_ensure_mixture_before_chat_tokenize() -> None:
    pipe = (
        PublicPipeline(_make_token_work_source())
        .ensure_mixture()
        .tokenize_chat(tokenizer_id="__fallback__")
    )
    with pytest.raises(ValueError, match="after tokenize"):
        pipe._prime_worksource()


def test_token_mode_allows_token_weighted_ensure_mixture_after_tokenize() -> None:
    pipe = (
        PublicPipeline(_make_token_work_source())
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
        .ensure_mixture(max_buffer_size=128, weight_by="input_ids")
    )
    pipe._prime_worksource()


def test_multi_tokenize_priming_warns_and_uses_first() -> None:
    pipe = (
        PublicPipeline(_make_token_work_source())
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
    )
    with pytest.warns(RuntimeWarning, match="first of 2 tokenize ops"):
        pipe._prime_worksource()


def test_mixed_tokenize_ops_warn_and_prime_with_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_token_prime(monkeypatch)
    pipe = (
        PublicPipeline(_make_token_work_source())
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
        .tokenize_chat(tokenizer_id="__fallback__")
    )
    with pytest.warns(RuntimeWarning, match="first of 2 tokenize ops"):
        pipe._prime_worksource()
    assert isinstance(captured["counting_spec"], TextTokenCountingSpec)


def test_checkpoint_before_iteration_primes_and_restores() -> None:
    def build() -> PublicPipeline:
        return (
            PublicPipeline(_make_token_work_source())
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
    assert list(restored)


def test_to_torch_dataset_primes_in_driver() -> None:
    pytest.importorskip("torch.utils.data")
    ws = _make_token_work_source()
    pipe = PublicPipeline(ws).tokenize(
        tokenizer_id="__fallback__", field="text", special_tokens="none"
    )
    assert ws.requires_token_priming
    pipe.to_torch_dataset(stateful=False)
    assert not ws.requires_token_priming


def _lift_text(payload: dict[str, Any]) -> dict[str, Any]:
    return {"text": payload["text"]}


def test_priming_replays_pre_tokenize_map_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_token_prime(monkeypatch)
    pipe = (
        PublicPipeline(_make_token_work_source())
        .map_transform(_lift_text)
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
    )
    pipe._prime_worksource()
    replay = captured["pre_tokenize_replay"]
    assert isinstance(replay, _PreTokenizeReplay)
    assert [type(op).__name__ for op in replay.ops] == ["MapTransform"]


def test_priming_replays_decode_text_and_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_token_prime(monkeypatch)
    pipe = (
        PublicPipeline(_make_token_work_source())
        .decode_text()
        .map_transform(_lift_text)
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
    )
    pipe._prime_worksource()
    replay = captured["pre_tokenize_replay"]
    assert isinstance(replay, _PreTokenizeReplay)
    assert [type(op).__name__ for op in replay.ops] == ["DecodeText", "MapTransform"]


def test_priming_flags_unreplayable_pre_tokenize_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_token_prime(monkeypatch)
    pipe = (
        PublicPipeline(_make_token_work_source())
        .shuffle(buffer_size=4)
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
    )
    pipe._prime_worksource()
    marker = captured["pre_tokenize_replay"]
    assert isinstance(marker, _UnreplayableOp)
    assert marker.op_name == "ShuffleBuffer"


def test_priming_without_pre_tokenize_ops_passes_no_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_token_prime(monkeypatch)
    pipe = PublicPipeline(_make_token_work_source()).tokenize(
        tokenizer_id="__fallback__", field="text", special_tokens="none"
    )
    pipe._prime_worksource()
    assert captured["pre_tokenize_replay"] is None


def test_priming_without_tokenize_op_passes_no_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_token_prime(monkeypatch)
    pipe = PublicPipeline(_make_token_work_source()).map_transform(_lift_text)
    pipe._prime_worksource()
    assert captured["pre_tokenize_replay"] is None


class _NonReplayableOp(BaseOp):
    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True, calibration_replayable=False)

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        raise AssertionError("calibration must not execute this op")


def test_calibration_replays_materialize_and_stops_before_packing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_token_prime(monkeypatch)
    work = StaticMixtureWorkSource(
        [_make_token_dataset("text", 3, n_docs=2)],
        {"text": 1.0},
        chunk_size=1,
        exhausted_policy="stop",
        token_estimation=TokenEstimation(),
    )
    pipe = (
        PublicPipeline(work)
        .materialize()
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
        .pack_flat(max_length=8, num_bins=2, pad_token_id=0)
        .batch(2)
    )
    pipe._prime_worksource()
    replay = captured["pre_tokenize_replay"]
    assert isinstance(replay, _PreTokenizeReplay)
    assert [type(op).__name__ for op in replay.ops] == ["Materialize"]
    ratio, source = work.state_dict()["token_ratios"]["text"]
    assert source == "measured"
    assert ratio > 0


@pytest.mark.parametrize("form", ["instance", "kwargs"])
def test_custom_op_opt_out_rejects_automatic_calibration(form: str) -> None:
    work = StaticMixtureWorkSource(
        [_make_token_dataset("text", 3, n_docs=2)],
        {"text": 1.0},
        token_estimation=TokenEstimation(),
    )
    pipe = PublicPipeline(work)
    op = _NonReplayableOp()
    if form == "instance":
        pipe.add_op(op)
    else:
        pipe.add_op(
            "custom",
            process_many=op.process_many,
            preserves_cursor_order=True,
            calibration_replayable=False,
        )
    pipe.tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
    with pytest.raises(ValueError, match="cannot be replayed outside the engine"):
        pipe._prime_worksource()
