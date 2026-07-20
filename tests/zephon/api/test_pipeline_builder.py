from __future__ import annotations

import re

import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon.api import Pipeline as PublicPipeline
from zephon.ops import PackSequences, TokenizeText
from zephon.ops.shuffle_buffer import ShuffleBuffer


def test_pipeline_explain_includes_plan_and_runtime() -> None:
    ds = make_inmem_dataset("tiny", [{"text": "a"}, {"text": "b"}, {"text": "c"}])
    ws = FakeIndexableWorkSource(ds, chunk_size=2)

    pipe = (
        PublicPipeline(ws)
        .decode_text(parallelism=3)  # override to check parallelism propagation
        .tokenize(tokenizer_id="__fallback__", field="text")
        .materialize()
        .batch(microbatch_size=2, drop_last=False)
        .options(
            deterministic=True,
            default_stage_prefetch=0,
            prefetch_batches=0,
            max_workers=4,
        )
    )

    exp = pipe.explain()
    # Plan section: single stage fused (place=local), ops list with p-values
    assert "Stage[0] place=local" in exp
    assert "ops=[" in exp
    assert "fetch@p" in exp
    assert "decode_text@p3" in exp  # overridden parallelism
    assert "tokenize@p4" in exp  # default from TokenizeText.traits()
    assert "materialize@p1" in exp
    assert "batch@p1" in exp

    # Engine section: thread runner with caps and final prefetch marker
    assert "runner=threads" in exp
    assert re.search(r"cap=\d+", exp)
    assert "pipeline_end" in exp


def test_pipeline_fetch_parallelism_override() -> None:
    ds = make_inmem_dataset("tiny", [{"text": "x"}])
    ws = FakeIndexableWorkSource(ds)

    pipe = (
        PublicPipeline(ws)
        .fetch_parallelism(5)
        .decode_text()
        .options(deterministic=True, prefetch_batches=0, default_stage_prefetch=0)
    )

    exp = pipe.explain()
    assert "fetch@p5" in exp


def test_pipeline_fetch_alias_configures_parallelism() -> None:
    ds = make_inmem_dataset("tiny", [{"text": "x"}])
    ws = FakeIndexableWorkSource(ds)

    pipe = (
        PublicPipeline(ws)
        .fetch(parallelism=4)
        .decode_text()
        .options(deterministic=True, prefetch_batches=0, default_stage_prefetch=0)
    )

    exp = pipe.explain()
    assert "fetch@p4" in exp


def test_pipeline_options_merge_and_final_prefetch_marker(tmp_path) -> None:
    ds = make_inmem_dataset("tiny", [{"text": "x"}])
    ws = FakeIndexableWorkSource(ds)

    cache_root = tmp_path / "cache"
    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .options(
            io_options={"cache": {"enabled": True}},
            prefetch_batches=2,
        )
    )

    # subsequent merge should keep enabled=True and change root
    pipe = pipe.options(io_options={"cache": {"root": cache_root}})

    exp = pipe.explain()
    assert "final_prefetch=2" in exp


def test_pipeline_tokenize_preserve_flag_forwarded() -> None:
    ds = make_inmem_dataset("tiny", [{"text": "x"}])
    ws = FakeIndexableWorkSource(ds)

    pipe = (
        PublicPipeline(ws)
        .tokenize(
            tokenizer_id="__fallback__",
            field="text",
            preserve_upstream_payload=True,
        )
        .options(deterministic=True, prefetch_batches=0, default_stage_prefetch=0)
    )

    tail_op = pipe._tail.op  # type: ignore[attr-defined]
    assert isinstance(tail_op, TokenizeText)
    assert tail_op.preserve_upstream_payload is True


# ---------------------------------------------------------------------------
# Pipeline.pack_sequences() / pack_flat() output selection
# ---------------------------------------------------------------------------


def test_pipeline_pack_sequences_is_envelope() -> None:
    """pack_sequences builds the envelope (list) output."""
    ds = make_inmem_dataset("tiny", [{"text": "x"}])
    ws = FakeIndexableWorkSource(ds)

    pipe = PublicPipeline(ws).pack_sequences(max_length=8, num_bins=4)

    tail_op = pipe._tail.op  # type: ignore[attr-defined]
    assert isinstance(tail_op, PackSequences)
    assert tail_op.output == "envelope"


def test_pipeline_pack_sequences_requires_num_bins_for_first_fit() -> None:
    ds = make_inmem_dataset("tiny", [{"text": "x"}])
    ws = FakeIndexableWorkSource(ds)

    with pytest.raises(ValueError, match="num_bins is required"):
        PublicPipeline(ws).pack_sequences(max_length=8)


def test_pipeline_pack_flat_wrap() -> None:
    """pack_flat builds the flat output; wrap needs no pad_token_id."""
    ds = make_inmem_dataset("tiny", [{"text": "x"}])
    ws = FakeIndexableWorkSource(ds)

    pipe = PublicPipeline(ws).pack_flat(max_length=8, algorithm="wrap")

    tail_op = pipe._tail.op  # type: ignore[attr-defined]
    assert isinstance(tail_op, PackSequences)
    assert tail_op.output == "flat"
    assert tail_op.emit_positions is True  # positions on by default
    assert tail_op.drop_oversized is False


def test_pipeline_best_fit_wrap_forwards_candidate_pool_options() -> None:
    ds = make_inmem_dataset("tiny", [{"input_ids": [1, 2]}])
    ws = FakeIndexableWorkSource(ds)

    pipe = PublicPipeline(ws).pack_sequences(
        max_length=8,
        algorithm="best_fit_wrap",
        candidate_pool_size=17,
        max_candidate_age=23,
    )

    tail_op = pipe._tail.op  # type: ignore[attr-defined]
    assert isinstance(tail_op, PackSequences)
    assert tail_op.algorithm == "best_fit_wrap"
    assert tail_op.drop_oversized is False
    assert tail_op.candidate_pool_size == 17
    assert tail_op.max_candidate_age == 23


def test_pipeline_pack_flat_first_fit_forwards_pad_token_id() -> None:
    """first_fit/best_fit pad partial bins, so pad_token_id flows through."""
    ds = make_inmem_dataset("tiny", [{"text": "x"}])
    ws = FakeIndexableWorkSource(ds)

    pipe = PublicPipeline(ws).pack_flat(
        max_length=8,
        num_bins=4,
        tokens_field="input_ids",
        algorithm="first_fit",
        pad_token_id=0,
        emit_positions=False,
    )

    tail_op = pipe._tail.op  # type: ignore[attr-defined]
    assert isinstance(tail_op, PackSequences)
    assert tail_op.output == "flat"
    assert tail_op.algorithm == "first_fit"
    assert tail_op.pad_token_id == 0
    assert tail_op.emit_positions is False


# ---------------------------------------------------------------------------
# Pipeline.shuffle() default buffer size
# ---------------------------------------------------------------------------


def _shuffle_node_op(pipe: PublicPipeline) -> ShuffleBuffer:
    """Find the inserted shuffle_buffer node and return its op."""
    for node in pipe._graph.nodes:  # type: ignore[attr-defined]
        if node.name == "shuffle_buffer":
            assert isinstance(node.op, ShuffleBuffer)
            return node.op
    raise AssertionError("Pipeline does not contain a shuffle_buffer node")


def test_pipeline_shuffle_default_buffer_size() -> None:
    """``pipeline.shuffle()`` with no buffer_size uses the default of 8192."""
    ds = make_inmem_dataset("tiny", [{"text": "x"}])
    ws = FakeIndexableWorkSource(ds)

    pipe = PublicPipeline(ws).shuffle().batch(microbatch_size=8)
    pipe.compile()

    op = _shuffle_node_op(pipe)
    assert op.buffer_size == 8192
    assert op.algorithm == "streaming"


def test_pipeline_shuffle_explicit_buffer_size() -> None:
    """An explicit ``buffer_size=`` is preserved on the op."""
    ds = make_inmem_dataset("tiny", [{"text": "x"}])
    ws = FakeIndexableWorkSource(ds)

    pipe = PublicPipeline(ws).shuffle(buffer_size=42).batch(microbatch_size=8)
    pipe.compile()

    op = _shuffle_node_op(pipe)
    assert op.buffer_size == 42


def test_pipeline_shuffle_explicit_algorithm() -> None:
    ds = make_inmem_dataset("tiny", [{"text": "x"}])
    ws = FakeIndexableWorkSource(ds)

    pipe = (
        PublicPipeline(ws)
        .shuffle(buffer_size=42, algorithm="block")
        .batch(microbatch_size=8)
    )
    pipe.compile()

    op = _shuffle_node_op(pipe)
    assert op.algorithm == "block"
