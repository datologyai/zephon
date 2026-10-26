"""Mapping-valued structured pack handoff into the ordinary training converter."""

import copy
from typing import Any

import pytest

from zephon.types import SampleBatch, SampleMeta, SampleRecord


def make_batch(payload: dict[str, Any]) -> SampleBatch:
    return SampleBatch(
        records=(
            SampleRecord(
                meta=SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0),
                payload=payload,
            ),
        )
    )


def test_mapping_pack_matches_direct_conversion_and_preserves_envelope() -> None:
    merged = {
        "input_ids": [1, 2, 3, 4, 5],
        "positions": [0, 1, 2, 0, 1],
        "loss_mask": [0, 1, 1, 0, 1],
        "media": ["image_a", "image_b"],
    }
    envelope = {"packed_samples": merged}
    original = copy.deepcopy(envelope)
    packed = make_batch(envelope)
    meta = packed.records[0].meta
    options = dict(
        dtype=None,
        return_labels=True,
        pad_to_length=7,
        pad_token_id=0,
        return_padding_mask=True,
        return_cu_seqlens=True,
        return_loss_mask=True,
        mask_segment_starts=True,
        field_collators={"media": lambda items, length, shift: sum(items, [])},
    )
    assert packed.to_training(**options) == make_batch(merged).to_training(**options)
    assert packed.records[0].payload == original
    assert packed.records[0].meta is meta


@pytest.mark.parametrize("merged", [[], [{"input_ids": [1, 2]}], [1, 2], None, 3])
def test_default_or_invalid_pack_requires_explicit_merge(merged: Any) -> None:
    with pytest.raises(ValueError, match="pack_payloads"):
        make_batch({"packed_samples": merged}).to_training(dtype=None)


def test_merged_mapping_still_validates_required_fields() -> None:
    with pytest.raises(ValueError, match="Cannot auto-detect"):
        make_batch({"packed_samples": {"media": []}}).to_training(dtype=None)
    with pytest.raises(ValueError, match="loss_mask.*length"):
        make_batch(
            {"packed_samples": {"input_ids": [1, 2], "loss_mask": [1]}}
        ).to_training(dtype=None)


def test_named_token_field_in_mapping_and_regular_payload_extra_are_preserved() -> None:
    merged = {"encoded": [8, 9, 10]}
    output = make_batch({"packed_samples": merged}).to_training(
        dtype=None, tokens_field="encoded", return_labels=True
    )
    assert output["input_ids"] == [[8, 9]]
    assert output["labels"] == [[9, 10]]
    # A normal producer may use the same key as an unrelated extra field; it is
    # only an envelope when packed_samples is its sole field.
    output = make_batch({"input_ids": [1, 2], "packed_samples": []}).to_training(
        dtype=None
    )
    assert output["input_ids"] == [[1, 2]]
