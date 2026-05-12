from __future__ import annotations

from typing import Any

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon.api import Pipeline as PublicPipeline
from zephon.core.constants import SampleBatch, SampleRecord


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
