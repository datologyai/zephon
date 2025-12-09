from __future__ import annotations

import re

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon.api import Pipeline as PublicPipeline
from zephon.ops import TokenizeText


def test_pipeline_explain_includes_plan_and_runtime() -> None:
    ds = make_inmem_dataset("tiny", [{"text": "a"}, {"text": "b"}, {"text": "c"}])
    ws = FakeIndexableWorkSource(ds, chunk_size=2)

    pipe = (
        PublicPipeline(ws)
        .decode_text(parallelism=3)  # override to check parallelism propagation
        .tokenize(tokenizer_id="__fallback__")
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
        .tokenize(tokenizer_id="__fallback__", preserve_upstream_payload=True)
        .options(deterministic=True, prefetch_batches=0, default_stage_prefetch=0)
    )

    tail_op = pipe._tail.op  # type: ignore[attr-defined]
    assert isinstance(tail_op, TokenizeText)
    assert tail_op.preserve_upstream_payload is True
