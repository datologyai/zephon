import dataclasses

import pytest

from zephon.core.constants import (
    LanePtr,
    SampleBatch,
    SampleMeta,
    SampleRecord,
)


def _rec(sample_id: tuple[int, int, int], lane_id: int, chunk_id: int, payload: dict):
    return SampleRecord(
        meta=SampleMeta(sample_id=sample_id, lane_id=lane_id, chunk_id=chunk_id),
        payload=payload,
    )


def test_lane_ptr_defaults() -> None:
    ptr = LanePtr()
    assert ptr.chunk_id == -1
    assert ptr.offset == 0


def test_sample_meta_frozen() -> None:
    meta = SampleMeta(sample_id=(1, 2, 3), lane_id=0, chunk_id=0)
    # Frozen dataclass should prevent assignment via normal attribute set
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        meta.lane_id = 1  # type: ignore[misc]


def test_sample_batch_basic_and_ids() -> None:
    r0 = _rec((0, 0, 0), 7, 10, {"text": "a"})
    r1 = _rec((0, 0, 1), 7, 10, {"text": "b"})
    batch = SampleBatch(records=(r0, r1))
    assert len(batch) == 2
    assert batch.ids == ((0, 0, 0), (0, 0, 1))
    assert batch.lane_ids == (7, 7)
    assert batch.chunk_ids == (10, 10)


def test_sample_batch_to_training_text_only_and_empty() -> None:
    empty = SampleBatch(records=())
    assert empty.to_training() == {"ids": [], "texts": []}

    r0 = _rec((1, 2, 3), 0, 0, {"text": "hello"})
    r1 = _rec((1, 2, 4), 0, 0, {"text": "world"})
    out = SampleBatch(records=(r0, r1)).to_training()
    assert out["ids"] == [(1, 2, 3), (1, 2, 4)]
    assert out["texts"] == ["hello", "world"]


def test_sample_batch_to_training_tensor_fields_only_if_all_present() -> None:
    r0 = _rec((1, 2, 3), 0, 0, {"text": "a", "input_ids": [1], "attention_mask": [1]})
    r1 = _rec((1, 2, 4), 0, 0, {"text": "b", "input_ids": [2], "attention_mask": [1]})
    out = SampleBatch(records=(r0, r1)).to_training()
    assert out["input_ids"] == [[1], [2]]
    assert out["attention_mask"] == [[1], [1]]

    r_bad = _rec((1, 2, 5), 0, 0, {"text": "c"})
    out2 = SampleBatch(records=(r0, r_bad)).to_training()
    assert "input_ids" not in out2 and "attention_mask" not in out2
