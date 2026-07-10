import dataclasses

import pytest

from zephon.core.constants import (
    LanePtr,
    SampleBatch,
    SampleCursor,
    SampleMeta,
    SampleRecord,
)


def _rec(
    sample_id: tuple[int, int, int],
    lane_id: int,
    chunk_id: int,
    payload: dict,
    padding_length: int | None = None,
):
    tags: dict = {}
    if padding_length is not None:
        tags["_packing_metadata"] = {"padding_length": padding_length}
    return SampleRecord(
        meta=SampleMeta(
            sample_id=sample_id, lane_id=lane_id, chunk_id=chunk_id, tags=tags
        ),
        payload=payload,
    )


def test_lane_ptr_defaults() -> None:
    ptr = LanePtr()
    assert ptr.chunk_id == -1
    assert ptr.offset == 0


def test_flush_sentinel_properties() -> None:
    """is_flush_sentinel and is_sentinel work for flush sentinels."""
    meta = SampleMeta(
        sample_id=(0, 0, 0), lane_id=0, chunk_id=0, tags={"_flush_sentinel": True}
    )
    assert meta.is_flush_sentinel is True
    assert meta.is_sentinel is True
    assert meta.tombstone is False

    # Regular record
    plain = SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0)
    assert plain.is_flush_sentinel is False
    assert plain.is_sentinel is False


def test_tombstone_is_sentinel_but_not_flush() -> None:
    """Tombstones are sentinels but not flush sentinels."""
    meta = SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0).with_tombstone(True)
    assert meta.is_sentinel is True
    assert meta.is_flush_sentinel is False
    assert meta.tombstone is True


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
    assert batch.lineage_paths == ((), ())


def test_sample_meta_lineage_helpers() -> None:
    base = SampleMeta(sample_id=(1, 2, 3), lane_id=0, chunk_id=4)
    assert base.lineage == ()
    child0 = base.child(0)
    assert child0.lineage == (0,)
    assert child0.sample_id == base.sample_id
    child1 = child0.child(1)
    assert child1.lineage == (0, 1)
    assert base.lineage == ()
    assert child0.cursor < child1.cursor


def test_sample_cursor_from_key_roundtrip() -> None:
    meta = SampleMeta(
        sample_id=(9, 9, 9), lane_id=3, chunk_id=2, chunk_offset=5
    ).with_lineage((2, 4))
    key = meta.as_cursor_key()
    cursor = SampleCursor.from_key(key)
    assert cursor.sample_id == meta.sample_id
    assert cursor.chunk_id == meta.chunk_id
    assert cursor.chunk_offset == meta.chunk_offset
    assert cursor.lineage == (2, 4)
    assert cursor.child(1).lineage == (2, 4, 1)


def test_sample_batch_to_training_text_only_and_empty() -> None:
    """Test empty batch and basic text+tokens extraction."""
    empty = SampleBatch(records=())
    assert empty.to_training() == {"ids": [], "texts": []}

    # Now requires tokens (use dtype=None to get lists for comparison)
    r0 = _rec((1, 2, 3), 0, 0, {"text": "hello", "input_ids": [1, 2, 3]})
    r1 = _rec((1, 2, 4), 0, 0, {"text": "world", "input_ids": [4, 5, 6]})
    out = SampleBatch(records=(r0, r1)).to_training(dtype=None)
    assert out["ids"] == [(1, 2, 3), (1, 2, 4)]
    assert out["texts"] == ["hello", "world"]
    assert out["input_ids"] == [[1, 2, 3], [4, 5, 6]]


def test_sample_batch_to_training_legacy_behavior() -> None:
    """Test that default behavior (no labels, auto dtype) works."""
    r0 = _rec((1, 2, 3), 0, 0, {"text": "a", "input_ids": [1, 2, 3]})
    r1 = _rec((1, 2, 4), 0, 0, {"text": "b", "input_ids": [4, 5, 6]})
    batch = SampleBatch(records=(r0, r1))

    # With dtype=None, should return lists (backward compatible behavior)
    out = batch.to_training(dtype=None)
    assert out["ids"] == [(1, 2, 3), (1, 2, 4)]
    assert out["texts"] == ["a", "b"]
    assert out["input_ids"] == [[1, 2, 3], [4, 5, 6]]
    assert "labels" not in out


# ---------------------------------------------------------------------------
# Tests for to_training with new parameters
# ---------------------------------------------------------------------------


class TestToTrainingTokensFieldAuto:
    """Tests for tokens_field="auto" detection."""

    def test_detects_input_ids(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "input_ids": [1, 2, 3]})
        r1 = _rec((0, 0, 1), 0, 0, {"text": "b", "input_ids": [4, 5, 6]})
        out = SampleBatch(records=(r0, r1)).to_training(dtype=None)
        assert out["input_ids"] == [[1, 2, 3], [4, 5, 6]]

    def test_detects_tokens(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "tokens": [1, 2, 3]})
        r1 = _rec((0, 0, 1), 0, 0, {"text": "b", "tokens": [4, 5, 6]})
        out = SampleBatch(records=(r0, r1)).to_training(dtype=None)
        assert out["input_ids"] == [[1, 2, 3], [4, 5, 6]]

    def test_detects_token_ids(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "token_ids": [1, 2, 3]})
        r1 = _rec((0, 0, 1), 0, 0, {"text": "b", "token_ids": [4, 5, 6]})
        out = SampleBatch(records=(r0, r1)).to_training(dtype=None)
        assert out["input_ids"] == [[1, 2, 3], [4, 5, 6]]

    def test_detects_ids(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "ids": [1, 2, 3]})
        r1 = _rec((0, 0, 1), 0, 0, {"text": "b", "ids": [4, 5, 6]})
        out = SampleBatch(records=(r0, r1)).to_training(dtype=None)
        assert out["input_ids"] == [[1, 2, 3], [4, 5, 6]]

    def test_prefers_input_ids_over_others(self) -> None:
        """input_ids takes precedence over tokens, token_ids, ids."""
        r0 = _rec(
            (0, 0, 0),
            0,
            0,
            {"text": "a", "input_ids": [1, 2], "tokens": [10, 20], "ids": [100, 200]},
        )
        r1 = _rec(
            (0, 0, 1),
            0,
            0,
            {"text": "b", "input_ids": [3, 4], "tokens": [30, 40], "ids": [300, 400]},
        )
        out = SampleBatch(records=(r0, r1)).to_training(dtype=None)
        assert out["input_ids"] == [[1, 2], [3, 4]]

    def test_raises_if_no_token_field_found(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "other_field": [1, 2, 3]})
        batch = SampleBatch(records=(r0,))
        with pytest.raises(ValueError, match="Cannot auto-detect tokens field"):
            batch.to_training(dtype=None)


class TestToTrainingExplicitTokensField:
    """Tests for explicit tokens_field parameter."""

    def test_explicit_field_name(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "my_tokens": [1, 2, 3]})
        r1 = _rec((0, 0, 1), 0, 0, {"text": "b", "my_tokens": [4, 5, 6]})
        out = SampleBatch(records=(r0, r1)).to_training(
            tokens_field="my_tokens", dtype=None
        )
        assert out["input_ids"] == [[1, 2, 3], [4, 5, 6]]

    def test_raises_if_field_missing(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "input_ids": [1, 2, 3]})
        r1 = _rec((0, 0, 1), 0, 0, {"text": "b"})  # Missing input_ids
        batch = SampleBatch(records=(r0, r1))
        with pytest.raises(ValueError, match="not found in payload at index 1"):
            batch.to_training(dtype=None)


class TestToTrainingReturnLabels:
    """Tests for return_labels parameter."""

    def test_return_labels_true_shifts_tokens(self) -> None:
        # tokens: [1, 2, 3, 4] -> input_ids: [1, 2, 3], labels: [2, 3, 4]
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "input_ids": [1, 2, 3, 4]})
        r1 = _rec((0, 0, 1), 0, 0, {"text": "b", "input_ids": [5, 6, 7, 8]})
        out = SampleBatch(records=(r0, r1)).to_training(return_labels=True, dtype=None)

        assert out["input_ids"] == [[1, 2, 3], [5, 6, 7]]
        assert out["labels"] == [[2, 3, 4], [6, 7, 8]]

    def test_return_labels_false_no_labels(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "input_ids": [1, 2, 3, 4]})
        out = SampleBatch(records=(r0,)).to_training(return_labels=False, dtype=None)

        assert out["input_ids"] == [[1, 2, 3, 4]]
        assert "labels" not in out


class TestToTrainingExtraFields:
    """Tests for extra_fields parameter."""

    def test_extra_fields_included(self) -> None:
        r0 = _rec(
            (0, 0, 0),
            0,
            0,
            {"text": "a", "input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        )
        r1 = _rec(
            (0, 0, 1),
            0,
            0,
            {"text": "b", "input_ids": [4, 5, 6], "attention_mask": [1, 1, 0]},
        )
        out = SampleBatch(records=(r0, r1)).to_training(
            extra_fields=["attention_mask"], dtype=None
        )

        assert out["input_ids"] == [[1, 2, 3], [4, 5, 6]]
        assert out["attention_mask"] == [[1, 1, 1], [1, 1, 0]]

    def test_extra_fields_shifted_with_labels(self) -> None:
        # When return_labels=True, extra fields should also be shifted
        r0 = _rec(
            (0, 0, 0),
            0,
            0,
            {"text": "a", "input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
        )
        out = SampleBatch(records=(r0,)).to_training(
            return_labels=True, extra_fields=["attention_mask"], dtype=None
        )

        # input_ids: [1, 2, 3, 4] -> [1, 2, 3]
        # attention_mask: [1, 1, 1, 1] -> [1, 1, 1]
        assert out["input_ids"] == [[1, 2, 3]]
        assert out["labels"] == [[2, 3, 4]]
        assert out["attention_mask"] == [[1, 1, 1]]

    def test_extra_fields_not_shifted_without_labels(self) -> None:
        r0 = _rec(
            (0, 0, 0),
            0,
            0,
            {"text": "a", "input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
        )
        out = SampleBatch(records=(r0,)).to_training(
            return_labels=False, extra_fields=["attention_mask"], dtype=None
        )

        assert out["input_ids"] == [[1, 2, 3, 4]]
        assert out["attention_mask"] == [[1, 1, 1, 1]]
        assert "labels" not in out

    def test_multiple_extra_fields(self) -> None:
        r0 = _rec(
            (0, 0, 0),
            0,
            0,
            {
                "text": "a",
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "token_type_ids": [0, 0, 1],
            },
        )
        out = SampleBatch(records=(r0,)).to_training(
            extra_fields=["attention_mask", "token_type_ids"], dtype=None
        )

        assert out["input_ids"] == [[1, 2, 3]]
        assert out["attention_mask"] == [[1, 1, 1]]
        assert out["token_type_ids"] == [[0, 0, 1]]

    def test_raises_if_extra_field_missing(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "input_ids": [1, 2, 3]})
        batch = SampleBatch(records=(r0,))
        with pytest.raises(ValueError, match="Extra field 'attention_mask' not found"):
            batch.to_training(extra_fields=["attention_mask"], dtype=None)


class TestToTrainingPositions:
    """The packing 'positions' field is surfaced automatically."""

    def test_positions_auto_surfaced(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2, 3], "positions": [0, 1, 0]})
        r1 = _rec((0, 0, 1), 0, 0, {"input_ids": [4, 5, 6], "positions": [0, 0, 1]})
        out = SampleBatch(records=(r0, r1)).to_training(dtype=None)
        # No extra_fields passed, yet positions rides through.
        assert out["positions"] == [[0, 1, 0], [0, 0, 1]]

    def test_positions_shifted_with_labels(self) -> None:
        r0 = _rec(
            (0, 0, 0), 0, 0, {"input_ids": [1, 2, 3, 4], "positions": [0, 1, 0, 1]}
        )
        out = SampleBatch(records=(r0,)).to_training(return_labels=True, dtype=None)
        assert out["input_ids"] == [[1, 2, 3]]
        assert out["labels"] == [[2, 3, 4]]
        # positions is sliced like input_ids (drop last), staying aligned with it.
        assert out["positions"] == [[0, 1, 0]]

    def test_absent_positions_not_added(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2, 3]})
        out = SampleBatch(records=(r0,)).to_training(dtype=None)
        assert "positions" not in out

    def test_explicit_positions_not_duplicated(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2, 3], "positions": [0, 1, 2]})
        out = SampleBatch(records=(r0,)).to_training(
            extra_fields=["positions"], dtype=None
        )
        assert out["positions"] == [[0, 1, 2]]

    def test_inconsistent_positions_raises(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2, 3], "positions": [0, 1, 2]})
        r1 = _rec((0, 0, 1), 0, 0, {"input_ids": [4, 5, 6]})  # missing positions
        with pytest.raises(ValueError, match="'positions' present in some payloads"):
            SampleBatch(records=(r0, r1)).to_training(dtype=None)


class TestToTrainingPadMasking:
    """to_training masks each record's trailing padding_length labels (by position)."""

    def test_pad_labels_become_ignore_index(self) -> None:
        # tokens [10,11,12,999,999] -> labels [11,12,999,999]; 2 pad -> last 2 masked.
        r0 = _rec(
            (0, 0, 0), 0, 0, {"input_ids": [10, 11, 12, 999, 999]}, padding_length=2
        )
        out = SampleBatch(records=(r0,)).to_training(return_labels=True, dtype=None)
        assert out["input_ids"] == [[10, 11, 12, 999]]  # input keeps the real pad id
        assert out["labels"] == [[11, 12, -100, -100]]  # trailing pad masked

    def test_custom_ignore_index(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [10, 11, 999]}, padding_length=1)
        out = SampleBatch(records=(r0,)).to_training(
            return_labels=True, dtype=None, ignore_index=-1
        )
        assert out["labels"] == [[11, -1]]

    def test_no_padding_leaves_labels_untouched(self) -> None:
        # padding_length absent (None) -> nothing masked.
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [10, 11, 999]})
        out = SampleBatch(records=(r0,)).to_training(return_labels=True, dtype=None)
        assert out["labels"] == [[11, 999]]

    def test_zero_padding_leaves_labels_untouched(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [10, 11, 999]}, padding_length=0)
        out = SampleBatch(records=(r0,)).to_training(return_labels=True, dtype=None)
        assert out["labels"] == [[11, 999]]

    def test_pad_id_colliding_with_content_only_masks_trailing(self) -> None:
        # The pad id (0) also appears mid-sequence; only the trailing pad is masked,
        # so the real in-content 0 stays in the loss — the point of position masking.
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [10, 0, 12, 0, 0]}, padding_length=2)
        out = SampleBatch(records=(r0,)).to_training(return_labels=True, dtype=None)
        assert out["labels"] == [[0, 12, -100, -100]]  # leading 0 kept, trailing masked

    def test_padding_without_labels_is_noop(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [10, 11, 999]}, padding_length=1)
        out = SampleBatch(records=(r0,)).to_training(dtype=None)
        assert "labels" not in out
        assert out["input_ids"] == [[10, 11, 999]]

    def test_per_record_padding_lengths(self) -> None:
        # Different pad counts per row are masked independently.
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [10, 11, 12, 13]}, padding_length=1)
        r1 = _rec((0, 0, 1), 0, 0, {"input_ids": [20, 21, 22, 23]}, padding_length=3)
        out = SampleBatch(records=(r0, r1)).to_training(return_labels=True, dtype=None)
        assert out["labels"] == [[11, 12, -100], [-100, -100, -100]]

    def test_mask_padding_labels_list(self) -> None:
        from zephon.utils.tensor_utils import mask_padding_labels

        out = mask_padding_labels([[1, 2, 3], [4, 5, 6]], [1, 2], -100, None)
        assert out == [[1, 2, -100], [4, -100, -100]]

    def test_mask_padding_labels_numpy_does_not_mutate(self) -> None:
        import numpy as np

        from zephon.utils.tensor_utils import mask_padding_labels

        labels = np.array([[1, 2, 3], [4, 5, 6]])
        out = mask_padding_labels(labels, [1, 2], -100, "numpy")
        assert out.tolist() == [[1, 2, -100], [4, -100, -100]]
        assert labels.tolist() == [[1, 2, 3], [4, 5, 6]]  # input untouched

    def test_mask_padding_labels_torch(self) -> None:
        torch = pytest.importorskip("torch")
        from zephon.utils.tensor_utils import mask_padding_labels

        labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
        out = mask_padding_labels(labels, [1, 2], -100, "torch")
        assert out.tolist() == [[1, 2, -100], [4, -100, -100]]

    def test_mask_padding_labels_zero_is_noop(self) -> None:
        from zephon.utils.tensor_utils import mask_padding_labels

        out = mask_padding_labels([[1, 2, 3], [4, 5, 6]], [0, 0], -100, None)
        assert out == [[1, 2, 3], [4, 5, 6]]


class TestToTrainingLossMask:
    def test_loss_mask_masks_labels_and_is_consumed(self) -> None:
        # Two packed conversations + pad tail. The first token of each doc has
        # mask 0, so the cross-document boundary label is masked by construction.
        r0 = _rec(
            (0, 0, 0),
            0,
            0,
            {
                "input_ids": [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 0, 0],
                "loss_mask": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0],
            },
            padding_length=2,
        )
        out = SampleBatch(records=(r0,)).to_training(return_labels=True, dtype=None)
        assert out["input_ids"] == [[11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 0]]
        assert out["labels"] == [
            [-100, -100, 14, 15, 16, -100, -100, 23, 24, -100, -100]
        ]
        assert "loss_mask" not in out

    def test_loss_mask_shift_alignment(self) -> None:
        # mask[i] gates token i as a label, i.e. output position i-1 — the mask
        # shifts like labels ([1:]), not like inputs ([:-1]).
        r0 = _rec(
            (0, 0, 0), 0, 0, {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 0, 1]}
        )
        out = SampleBatch(records=(r0,)).to_training(return_labels=True, dtype=None)
        assert out["labels"] == [[2, -100, 4]]

    def test_loss_mask_surfaced_without_labels(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]})
        out = SampleBatch(records=(r0,)).to_training(dtype=None)
        assert out["loss_mask"] == [[0, 1, 1]]
        assert out["input_ids"] == [[1, 2, 3]]

    def test_loss_mask_inconsistent_presence_raises(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2], "loss_mask": [1, 1]})
        r1 = _rec((0, 0, 1), 0, 0, {"input_ids": [3, 4]})
        with pytest.raises(ValueError, match="'loss_mask' present in some payloads"):
            SampleBatch(records=(r0, r1)).to_training(dtype=None)

    def test_loss_mask_length_mismatch_raises(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2, 3], "loss_mask": [1, 1]})
        with pytest.raises(ValueError, match="'loss_mask' length"):
            SampleBatch(records=(r0,)).to_training(return_labels=True, dtype=None)

    def test_loss_mask_composes_with_pad_masking(self) -> None:
        # A mis-set mask of 1 over the pad tail must not resurrect pad labels.
        r0 = _rec(
            (0, 0, 0),
            0,
            0,
            {"input_ids": [1, 2, 3, 0], "loss_mask": [1, 1, 1, 1]},
            padding_length=1,
        )
        out = SampleBatch(records=(r0,)).to_training(return_labels=True, dtype=None)
        assert out["labels"] == [[2, 3, -100]]

    def test_loss_mask_in_extra_fields_with_labels_raises(self) -> None:
        r0 = _rec(
            (0, 0, 0), 0, 0, {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 0]}
        )
        with pytest.raises(ValueError, match="input-aligned, not label-aligned"):
            SampleBatch(records=(r0,)).to_training(
                return_labels=True, extra_fields=["loss_mask"], dtype=None
            )

    def test_loss_mask_explicit_extra_field_without_labels_not_duplicated(
        self,
    ) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]})
        out = SampleBatch(records=(r0,)).to_training(
            extra_fields=["loss_mask"], dtype=None
        )
        assert out["loss_mask"] == [[0, 1, 1]]

    def test_all_zero_mask_masks_everything(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2, 3], "loss_mask": [0, 0, 0]})
        out = SampleBatch(records=(r0,)).to_training(return_labels=True, dtype=None)
        assert out["labels"] == [[-100, -100]]

    def test_loss_mask_torch(self) -> None:
        torch = pytest.importorskip("torch")
        r0 = _rec(
            (0, 0, 0), 0, 0, {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 0, 1]}
        )
        out = SampleBatch(records=(r0,)).to_training(
            return_labels=True, dtype=torch.long
        )
        assert isinstance(out["labels"], torch.Tensor)
        assert out["labels"].tolist() == [[2, -100, 4]]

    def test_loss_mask_numpy(self) -> None:
        np = pytest.importorskip("numpy")
        r0 = _rec(
            (0, 0, 0), 0, 0, {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 0, 1]}
        )
        out = SampleBatch(records=(r0,)).to_training(return_labels=True, dtype=np.int64)
        assert out["labels"].tolist() == [[2, -100, 4]]


class TestToTrainingRenameFields:
    def test_rename_input_ids_to_input(self) -> None:
        r0 = _rec(
            (0, 0, 0), 0, 0, {"input_ids": [1, 2, 3, 4], "positions": [0, 1, 2, 3]}
        )
        out = SampleBatch(records=(r0,)).to_training(
            return_labels=True, dtype=None, rename_fields={"input_ids": "input"}
        )
        assert out["input"] == [[1, 2, 3]]
        assert "input_ids" not in out
        assert out["labels"] == [[2, 3, 4]]
        assert out["positions"] == [[0, 1, 2]]

    def test_rename_missing_source_raises(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2]})
        with pytest.raises(ValueError, match="source keys not in output"):
            SampleBatch(records=(r0,)).to_training(
                dtype=None, rename_fields={"labels": "label"}
            )

    def test_rename_target_collision_raises(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2]})
        with pytest.raises(ValueError, match="target keys collide"):
            SampleBatch(records=(r0,)).to_training(
                dtype=None, rename_fields={"input_ids": "texts"}
            )

    def test_rename_duplicate_targets_raise(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2]})
        with pytest.raises(ValueError, match="target keys collide"):
            SampleBatch(records=(r0,)).to_training(
                dtype=None, rename_fields={"ids": "x", "texts": "x"}
            )

    def test_rename_swap_is_allowed(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2]})
        out = SampleBatch(records=(r0,)).to_training(
            dtype=None, rename_fields={"ids": "texts", "texts": "ids"}
        )
        assert out["texts"] == [(0, 0, 0)]
        assert out["ids"] == [""]

    def test_rename_skipped_for_empty_batch(self) -> None:
        out = SampleBatch(records=()).to_training(rename_fields={"input_ids": "input"})
        assert out == {"ids": [], "texts": []}


class TestToTrainingDtype:
    """Tests for dtype parameter and framework detection."""

    def test_dtype_none_returns_lists(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "input_ids": [1, 2, 3]})
        out = SampleBatch(records=(r0,)).to_training(dtype=None)

        assert isinstance(out["input_ids"], list)
        assert isinstance(out["input_ids"][0], list)

    def test_dtype_torch_returns_tensor(self) -> None:
        torch = pytest.importorskip("torch")
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "input_ids": [1, 2, 3]})
        r1 = _rec((0, 0, 1), 0, 0, {"text": "b", "input_ids": [4, 5, 6]})
        out = SampleBatch(records=(r0, r1)).to_training(dtype=torch.long)

        assert isinstance(out["input_ids"], torch.Tensor)
        assert out["input_ids"].dtype == torch.long
        assert out["input_ids"].shape == (2, 3)
        assert out["input_ids"].tolist() == [[1, 2, 3], [4, 5, 6]]

    def test_dtype_torch_with_labels(self) -> None:
        torch = pytest.importorskip("torch")
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "input_ids": [1, 2, 3, 4]})
        r1 = _rec((0, 0, 1), 0, 0, {"text": "b", "input_ids": [5, 6, 7, 8]})
        out = SampleBatch(records=(r0, r1)).to_training(
            return_labels=True, dtype=torch.long
        )

        assert isinstance(out["input_ids"], torch.Tensor)
        assert isinstance(out["labels"], torch.Tensor)
        assert out["input_ids"].shape == (2, 3)
        assert out["labels"].shape == (2, 3)
        assert out["input_ids"].tolist() == [[1, 2, 3], [5, 6, 7]]
        assert out["labels"].tolist() == [[2, 3, 4], [6, 7, 8]]

    def test_dtype_numpy_returns_ndarray(self) -> None:
        np = pytest.importorskip("numpy")
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "input_ids": [1, 2, 3]})
        r1 = _rec((0, 0, 1), 0, 0, {"text": "b", "input_ids": [4, 5, 6]})
        out = SampleBatch(records=(r0, r1)).to_training(dtype=np.int64)

        assert isinstance(out["input_ids"], np.ndarray)
        assert out["input_ids"].dtype == np.int64
        assert out["input_ids"].shape == (2, 3)
        assert out["input_ids"].tolist() == [[1, 2, 3], [4, 5, 6]]

    def test_dtype_numpy_with_labels(self) -> None:
        np = pytest.importorskip("numpy")
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "input_ids": [1, 2, 3, 4]})
        out = SampleBatch(records=(r0,)).to_training(return_labels=True, dtype=np.int64)

        assert isinstance(out["input_ids"], np.ndarray)
        assert isinstance(out["labels"], np.ndarray)
        assert out["input_ids"].tolist() == [[1, 2, 3]]
        assert out["labels"].tolist() == [[2, 3, 4]]

    def test_dtype_auto_prefers_torch(self) -> None:
        """When dtype='auto', torch.long should be preferred if available."""
        torch = pytest.importorskip("torch")
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "input_ids": [1, 2, 3]})
        out = SampleBatch(records=(r0,)).to_training(dtype="auto")

        assert isinstance(out["input_ids"], torch.Tensor)
        assert out["input_ids"].dtype == torch.long

    def test_extra_fields_with_torch(self) -> None:
        torch = pytest.importorskip("torch")
        r0 = _rec(
            (0, 0, 0),
            0,
            0,
            {"text": "a", "input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        )
        out = SampleBatch(records=(r0,)).to_training(
            extra_fields=["attention_mask"], dtype=torch.long
        )

        assert isinstance(out["input_ids"], torch.Tensor)
        assert isinstance(out["attention_mask"], torch.Tensor)
        assert out["attention_mask"].tolist() == [[1, 1, 1]]


class TestToTrainingEmptyBatch:
    """Tests for empty batch handling."""

    def test_empty_batch_returns_empty_lists(self) -> None:
        batch = SampleBatch(records=())
        out = batch.to_training()
        assert out == {"ids": [], "texts": []}

    def test_empty_batch_with_labels(self) -> None:
        batch = SampleBatch(records=())
        out = batch.to_training(return_labels=True)
        assert out == {"ids": [], "texts": []}


class TestToTrainingEdgeCases:
    """Edge case tests."""

    def test_single_token_sequence(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "input_ids": [42]})
        out = SampleBatch(records=(r0,)).to_training(dtype=None)
        assert out["input_ids"] == [[42]]

    def test_single_token_with_labels(self) -> None:
        # Single token -> empty input_ids and labels after shift
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "input_ids": [42]})
        out = SampleBatch(records=(r0,)).to_training(return_labels=True, dtype=None)
        assert out["input_ids"] == [[]]
        assert out["labels"] == [[]]

    def test_two_tokens_with_labels(self) -> None:
        # [1, 2] -> input_ids: [1], labels: [2]
        r0 = _rec((0, 0, 0), 0, 0, {"text": "a", "input_ids": [1, 2]})
        out = SampleBatch(records=(r0,)).to_training(return_labels=True, dtype=None)
        assert out["input_ids"] == [[1]]
        assert out["labels"] == [[2]]

    def test_non_dict_non_array_payload_raises(self) -> None:
        r0 = SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
            payload="not a dict or array",  # type: ignore
        )
        batch = SampleBatch(records=(r0,))
        with pytest.raises(TypeError, match="expects dict or array-like"):
            batch.to_training()

    def test_preserves_ids_and_texts(self) -> None:
        """ids and texts should always be lists, never tensors."""
        torch = pytest.importorskip("torch")
        r0 = _rec((1, 2, 3), 0, 0, {"text": "hello", "input_ids": [1, 2, 3]})
        r1 = _rec((4, 5, 6), 0, 0, {"text": "world", "input_ids": [4, 5, 6]})
        out = SampleBatch(records=(r0, r1)).to_training(dtype=torch.long)

        assert isinstance(out["ids"], list)
        assert isinstance(out["texts"], list)
        assert out["ids"] == [(1, 2, 3), (4, 5, 6)]
        assert out["texts"] == ["hello", "world"]

    def test_missing_text_field_returns_empty_string(self) -> None:
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2, 3]})  # No text field
        out = SampleBatch(records=(r0,)).to_training(dtype=None)
        assert out["texts"] == [""]


class TestToTrainingIntegration:
    """Integration tests combining multiple features."""

    def test_full_lm_training_setup(self) -> None:
        """Test a realistic LM training scenario."""
        torch = pytest.importorskip("torch")

        # Simulate a batch with seq_len+1 tokens (e.g., 5 tokens for seq_len=4)
        r0 = _rec(
            (0, 0, 0),
            0,
            0,
            {
                "text": "The quick brown",
                "input_ids": [100, 200, 300, 400, 500],
                "attention_mask": [1, 1, 1, 1, 1],
            },
        )
        r1 = _rec(
            (0, 0, 1),
            0,
            0,
            {
                "text": "fox jumps over",
                "input_ids": [101, 201, 301, 401, 501],
                "attention_mask": [1, 1, 1, 1, 1],
            },
        )
        batch = SampleBatch(records=(r0, r1))

        out = batch.to_training(
            return_labels=True, extra_fields=["attention_mask"], dtype=torch.long
        )

        # Verify shapes
        assert out["input_ids"].shape == (2, 4)  # batch=2, seq_len=4
        assert out["labels"].shape == (2, 4)
        assert out["attention_mask"].shape == (2, 4)

        # Verify values
        assert out["input_ids"].tolist() == [[100, 200, 300, 400], [101, 201, 301, 401]]
        assert out["labels"].tolist() == [[200, 300, 400, 500], [201, 301, 401, 501]]
        assert out["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 1]]

        # Verify metadata preserved
        assert out["ids"] == [(0, 0, 0), (0, 0, 1)]
        assert out["texts"] == ["The quick brown", "fox jumps over"]

    def test_custom_tokens_field_with_numpy(self) -> None:
        """Test custom field name with numpy backend."""
        np = pytest.importorskip("numpy")

        r0 = _rec(
            (0, 0, 0),
            0,
            0,
            {"text": "a", "my_custom_tokens": [1, 2, 3, 4, 5]},
        )
        out = SampleBatch(records=(r0,)).to_training(
            tokens_field="my_custom_tokens", return_labels=True, dtype=np.int32
        )

        assert isinstance(out["input_ids"], np.ndarray)
        assert out["input_ids"].dtype == np.int32
        assert out["input_ids"].tolist() == [[1, 2, 3, 4]]
        assert out["labels"].tolist() == [[2, 3, 4, 5]]


# ---------------------------------------------------------------------------
# Array payload tests (for LitData TokensLoader and similar formats)
# ---------------------------------------------------------------------------


class TestToTrainingArrayPayloads:
    """Tests for array payloads (numpy arrays, torch tensors)."""

    def test_numpy_array_payloads(self) -> None:
        np = pytest.importorskip("numpy")
        arr0 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        arr1 = np.array([6, 7, 8, 9, 10], dtype=np.int64)
        r0 = SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0), payload=arr0
        )
        r1 = SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0), payload=arr1
        )
        batch = SampleBatch(records=(r0, r1))
        out = batch.to_training(dtype=np.int64)

        assert isinstance(out["input_ids"], np.ndarray)
        assert out["input_ids"].tolist() == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        assert out["ids"] == [(0, 0, 0), (0, 0, 1)]
        assert out["texts"] == ["", ""]

    def test_numpy_array_payloads_with_labels(self) -> None:
        np = pytest.importorskip("numpy")
        arr0 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        arr1 = np.array([6, 7, 8, 9, 10], dtype=np.int64)
        r0 = SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0), payload=arr0
        )
        r1 = SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0), payload=arr1
        )
        batch = SampleBatch(records=(r0, r1))
        out = batch.to_training(return_labels=True, dtype=np.int64)

        assert out["input_ids"].tolist() == [[1, 2, 3, 4], [6, 7, 8, 9]]
        assert out["labels"].tolist() == [[2, 3, 4, 5], [7, 8, 9, 10]]

    def test_torch_tensor_payloads(self) -> None:
        torch = pytest.importorskip("torch")
        t0 = torch.tensor([1, 2, 3, 4, 5])
        t1 = torch.tensor([6, 7, 8, 9, 10])
        r0 = SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0), payload=t0
        )
        r1 = SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0), payload=t1
        )
        batch = SampleBatch(records=(r0, r1))
        out = batch.to_training(dtype=torch.long)

        assert isinstance(out["input_ids"], torch.Tensor)
        assert out["input_ids"].tolist() == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]

    def test_mixed_dict_array_raises(self) -> None:
        np = pytest.importorskip("numpy")
        r0 = _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2, 3]})
        r1 = SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0),
            payload=np.array([4, 5, 6]),
        )
        batch = SampleBatch(records=(r0, r1))
        with pytest.raises(TypeError, match="Expected dict payload, got: ndarray"):
            batch.to_training()

    def test_mixed_array_dict_raises(self) -> None:
        np = pytest.importorskip("numpy")
        r0 = SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
            payload=np.array([1, 2, 3]),
        )
        r1 = _rec((0, 0, 1), 0, 0, {"input_ids": [4, 5, 6]})
        batch = SampleBatch(records=(r0, r1))
        with pytest.raises(TypeError, match="mixed payload types"):
            batch.to_training()

    def test_array_payloads_ignore_tokens_field(self) -> None:
        """tokens_field parameter is ignored for array payloads."""
        np = pytest.importorskip("numpy")
        arr = np.array([1, 2, 3], dtype=np.int64)
        r0 = SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0), payload=arr
        )
        batch = SampleBatch(records=(r0,))
        # Should work even with explicit tokens_field (it's ignored)
        out = batch.to_training(tokens_field="nonexistent", dtype=np.int64)
        assert out["input_ids"].tolist() == [[1, 2, 3]]

    def test_numpy_to_torch_no_warning(self) -> None:
        """Converting numpy array payloads to torch tensors should not warn."""
        import warnings

        np = pytest.importorskip("numpy")
        torch = pytest.importorskip("torch")
        arr0 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        arr1 = np.array([6, 7, 8, 9, 10], dtype=np.int64)
        r0 = SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0), payload=arr0
        )
        r1 = SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0), payload=arr1
        )
        batch = SampleBatch(records=(r0, r1))

        # Convert warnings to errors so the test fails if any warning is raised
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = batch.to_training(dtype=torch.long)

        assert isinstance(out["input_ids"], torch.Tensor)
        assert out["input_ids"].tolist() == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]


# ---------------------------------------------------------------------------
# Cross-product parametrized tests for comprehensive coverage
# ---------------------------------------------------------------------------


def _get_dtype_and_framework(dtype_name: str) -> tuple:
    """Helper to get dtype and framework info for parametrized tests."""
    if dtype_name == "list":
        return None, "list", None
    elif dtype_name == "numpy":
        np = pytest.importorskip("numpy")
        return np.int64, "numpy", np
    elif dtype_name == "torch":
        torch = pytest.importorskip("torch")
        return torch.long, "torch", torch
    raise ValueError(f"Unknown dtype_name: {dtype_name}")


def _make_test_batch() -> SampleBatch:
    """Create a standard test batch for parametrized tests."""
    r0 = _rec(
        (0, 0, 0),
        0,
        0,
        {
            "text": "hello",
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1],
        },
    )
    r1 = _rec(
        (0, 0, 1),
        0,
        0,
        {
            "text": "world",
            "input_ids": [6, 7, 8, 9, 10],
            "attention_mask": [1, 1, 1, 0, 0],
        },
    )
    return SampleBatch(records=(r0, r1))


@pytest.mark.parametrize("dtype_name", ["list", "numpy", "torch"])
@pytest.mark.parametrize("return_labels", [False, True])
@pytest.mark.parametrize("use_extra_fields", [False, True])
class TestToTrainingCrossProduct:
    """Cross-product parametrized tests for dtype × return_labels × extra_fields."""

    def test_output_structure(
        self, dtype_name: str, return_labels: bool, use_extra_fields: bool
    ) -> None:
        """Test that output has correct structure for all parameter combinations."""
        dtype, framework, module = _get_dtype_and_framework(dtype_name)
        batch = _make_test_batch()
        extra_fields = ["attention_mask"] if use_extra_fields else []

        out = batch.to_training(
            return_labels=return_labels,
            dtype=dtype,
            extra_fields=extra_fields,
        )

        # Always present
        assert "ids" in out
        assert "texts" in out
        assert "input_ids" in out
        assert isinstance(out["ids"], list)
        assert isinstance(out["texts"], list)

        # Labels only when requested
        if return_labels:
            assert "labels" in out
        else:
            assert "labels" not in out

        # Extra fields only when requested
        if use_extra_fields:
            assert "attention_mask" in out
        else:
            assert "attention_mask" not in out

    def test_output_types(
        self, dtype_name: str, return_labels: bool, use_extra_fields: bool
    ) -> None:
        """Test that output types match the requested dtype."""
        dtype, framework, module = _get_dtype_and_framework(dtype_name)
        batch = _make_test_batch()
        extra_fields = ["attention_mask"] if use_extra_fields else []

        out = batch.to_training(
            return_labels=return_labels,
            dtype=dtype,
            extra_fields=extra_fields,
        )

        if framework == "list":
            assert isinstance(out["input_ids"], list)
            assert isinstance(out["input_ids"][0], list)
            if return_labels:
                assert isinstance(out["labels"], list)
            if use_extra_fields:
                assert isinstance(out["attention_mask"], list)
        elif framework == "numpy":
            assert isinstance(out["input_ids"], module.ndarray)
            if return_labels:
                assert isinstance(out["labels"], module.ndarray)
            if use_extra_fields:
                assert isinstance(out["attention_mask"], module.ndarray)
        elif framework == "torch":
            assert isinstance(out["input_ids"], module.Tensor)
            if return_labels:
                assert isinstance(out["labels"], module.Tensor)
            if use_extra_fields:
                assert isinstance(out["attention_mask"], module.Tensor)

    def test_output_shapes(
        self, dtype_name: str, return_labels: bool, use_extra_fields: bool
    ) -> None:
        """Test that output shapes are correct for all parameter combinations."""
        dtype, framework, module = _get_dtype_and_framework(dtype_name)
        batch = _make_test_batch()
        extra_fields = ["attention_mask"] if use_extra_fields else []

        out = batch.to_training(
            return_labels=return_labels,
            dtype=dtype,
            extra_fields=extra_fields,
        )

        # Expected shapes: batch_size=2, original seq_len=5
        # With labels: seq_len becomes 4 (shifted)
        # Without labels: seq_len stays 5
        expected_seq_len = 4 if return_labels else 5

        def get_shape(tensor):
            if framework == "list":
                return (len(tensor), len(tensor[0]))
            return tuple(tensor.shape)

        assert get_shape(out["input_ids"]) == (2, expected_seq_len)
        if return_labels:
            assert get_shape(out["labels"]) == (2, expected_seq_len)
        if use_extra_fields:
            assert get_shape(out["attention_mask"]) == (2, expected_seq_len)

    def test_output_values(
        self, dtype_name: str, return_labels: bool, use_extra_fields: bool
    ) -> None:
        """Test that output values are correct for all parameter combinations."""
        dtype, framework, module = _get_dtype_and_framework(dtype_name)
        batch = _make_test_batch()
        extra_fields = ["attention_mask"] if use_extra_fields else []

        out = batch.to_training(
            return_labels=return_labels,
            dtype=dtype,
            extra_fields=extra_fields,
        )

        def to_list(tensor):
            if framework == "list":
                return tensor
            return tensor.tolist()

        if return_labels:
            # Shifted: input = tokens[:-1], labels = tokens[1:]
            assert to_list(out["input_ids"]) == [[1, 2, 3, 4], [6, 7, 8, 9]]
            assert to_list(out["labels"]) == [[2, 3, 4, 5], [7, 8, 9, 10]]
            if use_extra_fields:
                assert to_list(out["attention_mask"]) == [[1, 1, 1, 1], [1, 1, 1, 0]]
        else:
            # Not shifted: full tokens
            assert to_list(out["input_ids"]) == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
            if use_extra_fields:
                assert to_list(out["attention_mask"]) == [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0],
                ]

        # Metadata always preserved
        assert out["ids"] == [(0, 0, 0), (0, 0, 1)]
        assert out["texts"] == ["hello", "world"]
