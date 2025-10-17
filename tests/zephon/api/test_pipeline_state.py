from __future__ import annotations

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon.api import Pipeline as PublicPipeline


def _collect_texts(pipe: PublicPipeline) -> list[str]:
    out: list[str] = []
    it = iter(pipe)
    try:
        for item in it:
            out.append(item.payload.get("text", ""))
    finally:
        it.close()
    return out


def test_pipeline_checkpoint_restore_resumes_from_saved_position() -> None:
    rows = [{"text": f"sample-{i}"} for i in range(8)]
    ds1 = make_inmem_dataset("tiny", rows)
    ws1 = FakeIndexableWorkSource(ds1, chunk_size=3)

    pipe1 = (
        PublicPipeline(ws1)
        .decode_text()
        .tokenize(tokenizer_id="__fallback__")
        .options(deterministic=True, max_workers=1, default_stage_prefetch=0)
    )

    it = iter(pipe1)
    seen: list[str] = []
    try:
        for _ in range(3):
            rec = next(it)
            seen.append(rec.payload.get("text", ""))
        ckpt = pipe1.checkpoint()
        remainder: list[str] = []
        for rec in it:
            remainder.append(rec.payload.get("text", ""))
    finally:
        it.close()

    ds2 = make_inmem_dataset("tiny", rows)
    ws2 = FakeIndexableWorkSource(ds2, chunk_size=3)
    pipe2 = (
        PublicPipeline(ws2)
        .decode_text()
        .tokenize(tokenizer_id="__fallback__")
        .options(deterministic=True, max_workers=1, default_stage_prefetch=0)
    )

    pipe2.restore(ckpt)

    restored = _collect_texts(pipe2)
    assert restored == remainder
    assert len(seen) == 3 and len(restored) + len(seen) == len(rows)


def test_is_indexable_reflects_work_source() -> None:
    rows = [{"text": "x"}, {"text": "y"}]
    ds = make_inmem_dataset("tiny", rows)
    ws = FakeIndexableWorkSource(ds, chunk_size=2)
    pipe = PublicPipeline(ws).decode_text()
    assert pipe.is_indexable

    non_indexable_ws = FakeIndexableWorkSource(ds, chunk_size=2)
    non_indexable_pipe = PublicPipeline(non_indexable_ws).decode_text().batch(2)
    assert not non_indexable_pipe.is_indexable
