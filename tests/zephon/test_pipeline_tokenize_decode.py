from __future__ import annotations

from typing import Any

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon import Pipeline as PublicPipeline
from zephon.types import SampleBatch, SampleRecord


def _payload_dict(record: SampleRecord) -> dict[str, Any]:
    payload = record.payload
    assert isinstance(payload, dict)
    return payload


def test_decode_and_tokenize_single_records() -> None:
    ds = make_inmem_dataset(
        "tiny",
        [
            {"text": "Hello world"},
            {"text": "Another sample"},
        ],
    )
    ws = FakeIndexableWorkSource(ds, chunk_size=8)
    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .tokenize(tokenizer_id="__fallback__", field="text")
        .options(deterministic=True, max_workers=1, default_stage_prefetch=0)
    )

    it = iter(pipe)
    try:
        items = [next(it), next(it)]
    finally:
        it.close()

    for rec in items:
        assert isinstance(rec, SampleRecord)
        payload = _payload_dict(rec)
        assert isinstance(payload.get("input_ids"), list)
        assert payload.get("input_ids")
        assert isinstance(payload.get("attention_mask"), list)


def test_tokenize_with_batch_outputs_masks() -> None:
    rows = [{"text": f"t {i}"} for i in range(5)]
    ds = make_inmem_dataset("tiny", rows)
    ws = FakeIndexableWorkSource(ds, chunk_size=5)

    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .tokenize(tokenizer_id="__fallback__", field="text")
        .batch(microbatch_size=2, drop_last=False)
        .options(deterministic=True, max_workers=1, default_stage_prefetch=0)
    )

    it = iter(pipe)
    try:
        batches: list[SampleBatch] = []
        for item in it:
            assert isinstance(item, SampleBatch)
            batches.append(item)
    finally:
        it.close()

    # Verify lane purity and presence of token fields on every record
    for b in batches:
        lids = set(b.lane_ids)
        assert len(lids) == 1
        for rec in b.records:
            payload = _payload_dict(rec)
            assert isinstance(payload.get("input_ids"), list)
            mask = payload.get("attention_mask")
            if mask is not None:
                assert isinstance(mask, list)


# ---------------------------------------------------------------------------
# Pipeline.tokenize forwards the special_tokens API to TokenizeText, and
# carries the ``SpecialTokensMode`` Literal so SDK users get type narrowing
# and autocomplete instead of a plain ``str``.
# ---------------------------------------------------------------------------


def test_pipeline_tokenize_forwards_special_tokens_params() -> None:
    """``Pipeline.tokenize`` must accept and forward
    ``special_tokens`` / ``bos_token_id`` / ``eos_token_id`` so the API is
    reachable from the high-level wrapper, not just by constructing
    ``TokenizeText`` directly."""
    import inspect

    from zephon._internal.ops.tokenize_text import TokenizeText
    from zephon.pipeline import Pipeline

    sig = inspect.signature(Pipeline.tokenize)
    assert "special_tokens" in sig.parameters
    assert "bos_token_id" in sig.parameters
    assert "eos_token_id" in sig.parameters

    # And the kwargs round-trip into the op constructor.
    op = TokenizeText(
        tokenizer_id="__fallback__",
        field="text",
        special_tokens="bos",
        bos_token_id=42,
    )
    assert op.special_tokens == "bos"
    assert op._bos_id_override == 42


def test_pipeline_tokenize_forwards_missing_field_policy() -> None:
    from zephon._internal.ops.tokenize_text import TokenizeText

    ds = make_inmem_dataset("tiny", [{"text": "hello"}])
    pipe = PublicPipeline(FakeIndexableWorkSource(ds)).tokenize(
        tokenizer_id="__fallback__", field="text", missing_field="empty"
    )
    op = pipe._tail.op
    assert isinstance(op, TokenizeText)
    assert op.missing_field == "empty"


def test_pipeline_tokenize_signature_uses_missing_field_mode_literal() -> None:
    import inspect

    from zephon.ops.config import MissingFieldMode
    from zephon.pipeline import Pipeline

    parameter = inspect.signature(Pipeline.tokenize).parameters["missing_field"]
    assert parameter.annotation is MissingFieldMode
    assert parameter.default == "error"


def test_pipeline_tokenize_signature_uses_special_tokens_mode_literal() -> None:
    """The Literal alias on ``Pipeline.tokenize`` lets SDK users see the
    five valid mode strings in autocomplete and have static checkers narrow
    the parameter type. A plain ``str`` annotation would lose both."""
    import inspect

    from zephon.ops.config import SpecialTokensMode
    from zephon.pipeline import Pipeline

    sig = inspect.signature(Pipeline.tokenize)
    annotation = sig.parameters["special_tokens"].annotation
    assert annotation is SpecialTokensMode
