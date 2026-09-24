import copy
import dataclasses
import warnings
from typing import Any

import pytest

from zephon.types import SampleBatch, SampleCursor, SampleMeta, SampleRecord


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

    @pytest.mark.parametrize(
        ("field", "return_labels"),
        [
            ("input_ids", False),
            ("input_ids", True),
            ("labels", True),
            ("ids", False),
            ("texts", False),
        ],
    )
    def test_extra_fields_cannot_overwrite_generated_outputs(
        self, field: str, return_labels: bool
    ) -> None:
        record = _rec((0, 0, 0), 0, 0, {"encoded": [1, 7, 2, 1], field: [99] * 4})
        batch = SampleBatch(records=(record,))
        with pytest.raises(
            ValueError, match=f"cannot overwrite generated field '{field}'"
        ):
            batch.to_training(
                dtype=None,
                tokens_field="encoded",
                return_labels=return_labels,
                eos_mask_loss=return_labels,
                eos_token_id=2 if return_labels else None,
                extra_fields=(field,),
                rename_fields={"input_ids": "tokens"},
            )

    def test_extra_labels_allowed_when_labels_are_not_generated(self) -> None:
        record = _rec((0, 0, 0), 0, 0, {"tokens": [1, 2, 3], "labels": [4, 5, 6]})
        out = SampleBatch(records=(record,)).to_training(
            dtype=None, extra_fields=("labels",)
        )
        assert out["input_ids"] == [[1, 2, 3]]
        assert out["labels"] == [[4, 5, 6]]


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

    @pytest.mark.parametrize("return_loss_mask", [False, True])
    def test_loss_mask_in_extra_fields_with_labels_raises(
        self, return_loss_mask: bool
    ) -> None:
        r0 = _rec(
            (0, 0, 0), 0, 0, {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 0]}
        )
        with pytest.raises(ValueError, match="input-aligned, not label-aligned"):
            SampleBatch(records=(r0,)).to_training(
                return_labels=True,
                return_loss_mask=return_loss_mask,
                extra_fields=["loss_mask"],
                dtype=None,
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

    def test_rename_missing_source_ignored_for_empty_batch(self) -> None:
        out = SampleBatch(records=()).to_training(rename_fields={"input_ids": "input"})
        assert out == {"ids": [], "texts": []}

    @pytest.mark.parametrize(
        ("renames", "expected"),
        [
            (
                {"ids": "sample_ids", "input_ids": "tokens"},
                {"sample_ids": [], "texts": []},
            ),
            (
                {"ids": "sample_ids", "input_ids": "sample_ids"},
                {"sample_ids": [], "texts": []},
            ),
            ({"input_ids": "texts"}, {"ids": [], "texts": []}),
            (
                {"ids": "input_ids", "input_ids": "tokens"},
                {"input_ids": [], "texts": []},
            ),
        ],
    )
    def test_empty_batch_renames_only_emitted_fields(
        self, renames: dict[str, str], expected: dict[str, Any]
    ) -> None:
        assert SampleBatch(records=()).to_training(rename_fields=renames) == expected

    def test_empty_batch_rename_swap_is_allowed(self) -> None:
        out = SampleBatch(records=()).to_training(
            return_labels=True,
            return_num_valid_tokens=True,
            rename_fields={"ids": "num_valid_tokens", "num_valid_tokens": "ids"},
        )
        assert out == {"num_valid_tokens": [], "ids": 0, "texts": []}

    @pytest.mark.parametrize(
        ("empty", "renames"),
        [
            (True, {"ids": "texts"}),
            (True, {"loss_mask": "weights", "cu_seqlens": "weights"}),
            (False, {"loss_mask": "max_seqlen"}),
            (False, {"num_valid_tokens": "ids"}),
        ],
    )
    def test_rename_collisions_raise_before_exclusions(
        self, empty: bool, renames: dict[str, str]
    ) -> None:
        records = (
            ()
            if empty
            else (_rec((0, 0, 0), 0, 0, {"input_ids": [1, 2], "positions": [0, 1]}),)
        )
        with pytest.raises(ValueError, match="target keys collide"):
            SampleBatch(records=records).to_training(
                dtype=None,
                return_labels=True,
                return_loss_mask=True,
                return_cu_seqlens=True,
                return_num_valid_tokens=True,
                rename_fields=renames,
                exclude_fields=tuple(renames.values()),
            )


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


@pytest.fixture(params=["list", "numpy", "torch"])
def conversion_dtype(request: pytest.FixtureRequest) -> Any:
    return _get_dtype_and_framework(request.param)[0]


def _values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else value.tolist()


@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize("return_labels", [False, True])
def test_padding_mask_is_input_aligned_and_independent_of_supervision(
    conversion_dtype: Any, flatten: bool, return_labels: bool
) -> None:
    records = (
        _rec(
            (0, 0, 0),
            0,
            0,
            {
                "input_ids": [0, 11, 12, 0, 0],
                "positions": [0, 1, 2, 0, 1],
                "loss_mask": [0, 0, 1, 0, 0],
            },
            padding_length=2,
        ),
        _rec(
            (0, 0, 1),
            0,
            0,
            {
                "input_ids": [0, 21, 22, 23, 0],
                "positions": [0, 1, 2, 3, 0],
                "loss_mask": [0, 1, 1, 1, 0],
            },
            padding_length=1,
        ),
    )
    output = SampleBatch(records=records).to_training(
        dtype=conversion_dtype,
        return_labels=return_labels,
        return_loss_mask=return_labels,
        return_num_valid_tokens=return_labels,
        return_padding_mask=True,
        return_cu_seqlens=True,
        flatten=flatten,
    )
    expected = [[False, False, False, True, True], [False, False, False, False, True]]
    if return_labels:
        expected = [row[:-1] for row in expected]
        assert output["num_valid_tokens"] == 4
        expected_loss = [[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]]
        assert _values(output["loss_mask"]) == (
            [v for row in expected_loss for v in row] if flatten else expected_loss
        )
    mask = output["padding_mask"]
    assert _values(mask) == (
        [v for row in expected for v in row] if flatten else expected
    )
    if conversion_dtype is None:
        assert all(
            type(v) is bool
            for v in (mask if flatten else [v for row in mask for v in row])
        )
    else:
        assert str(mask.dtype) in ("bool", "torch.bool")
        assert mask.shape == output["input_ids"].shape


@pytest.mark.parametrize("length", [0, 1, 5])
@pytest.mark.parametrize("return_labels", [False, True])
def test_padding_mask_empty_and_all_padding_rows(
    conversion_dtype: Any, length: int, return_labels: bool
) -> None:
    record = _rec((0, 0, 0), 0, 0, {"input_ids": [0] * length}, padding_length=length)
    result = SampleBatch(records=(record,)).to_training(
        dtype=conversion_dtype,
        return_labels=return_labels,
        return_padding_mask=True,
        return_loss_mask=return_labels,
        return_num_valid_tokens=return_labels,
    )
    width = max(0, length - int(return_labels))
    assert _values(result["padding_mask"]) == [[True] * width]
    if return_labels:
        assert _values(result["labels"]) == [[-100] * width]
        assert result["num_valid_tokens"] == 0


def test_padding_mask_ragged_lists_and_missing_metadata() -> None:
    batch = SampleBatch(
        records=(
            _rec((0, 0, 0), 0, 0, {"input_ids": [0, 0]}),
            _rec((0, 0, 1), 0, 0, {"input_ids": [0, 0, 0]}, padding_length=2),
        )
    )
    out = batch.to_training(dtype=None, return_padding_mask=True, flatten=True)
    assert out["padding_mask"] == [False, False, False, True, True]
    assert "padding_mask" not in batch.to_training(dtype=None)
    assert (
        SampleBatch(records=()).to_training(return_padding_mask=True)["padding_mask"]
        == []
    )


@pytest.mark.parametrize("flatten", [False, True])
def test_padding_and_loss_masks_for_ragged_rows(flatten: bool) -> None:
    batch = SampleBatch(
        records=(
            _rec((0, 0, 0), 0, 0, {"input_ids": []}),
            _rec((0, 0, 1), 0, 0, {"input_ids": [10, 11]}),
            _rec((0, 0, 2), 0, 0, {"input_ids": [20, 21, 0, 0, 0]}, padding_length=3),
            _rec((0, 0, 3), 0, 0, {"input_ids": [0, 0, 0]}, padding_length=3),
        )
    )
    out = batch.to_training(
        dtype=None,
        return_labels=True,
        return_padding_mask=True,
        return_loss_mask=True,
        return_num_valid_tokens=True,
        flatten=flatten,
    )
    expected = {
        "input_ids": [[], [10], [20, 21, 0, 0], [0, 0]],
        "labels": [[], [11], [21, -100, -100, -100], [-100, -100]],
        "padding_mask": [[], [False], [False, False, True, True], [True, True]],
        "loss_mask": [[], [1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0]],
    }
    for name, rows in expected.items():
        assert out[name] == ([v for row in rows for v in row] if flatten else rows)
    assert out["num_valid_tokens"] == 2


def test_padding_mask_renaming_exclusion_and_collision() -> None:
    batch = SampleBatch(
        records=(_rec((0, 0, 0), 0, 0, {"input_ids": [1, 0]}, padding_length=1),)
    )
    out = batch.to_training(
        dtype=None, return_padding_mask=True, rename_fields={"padding_mask": "pad"}
    )
    assert out["pad"] == [[False, True]]
    out = batch.to_training(
        dtype=None, return_padding_mask=True, exclude_fields=("padding_mask",)
    )
    assert "padding_mask" not in out
    with pytest.raises(ValueError, match="extra_fields cannot include 'padding_mask'"):
        batch.to_training(return_padding_mask=True, extra_fields=("padding_mask",))
    with pytest.raises(ValueError, match="collide"):
        batch.to_training(
            return_padding_mask=True, rename_fields={"input_ids": "padding_mask"}
        )


@pytest.mark.parametrize("padding", [-1, 4])
def test_padding_mask_rejects_invalid_lengths(padding: int) -> None:
    batch = SampleBatch(
        records=(
            _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2, 3]}, padding_length=padding),
        )
    )
    with pytest.raises(ValueError, match="padding_length"):
        batch.to_training(dtype=None, return_padding_mask=True)


def _assert_loss_mask_dtype(mask: Any, conversion_dtype: Any) -> None:
    if conversion_dtype is None:
        values = (
            [value for row in mask for value in row]
            if mask and isinstance(mask[0], list)
            else mask
        )
        assert all(type(value) is float for value in values)
    elif type(mask).__module__.startswith("torch"):
        assert mask.dtype == pytest.importorskip("torch").float32
    else:
        assert mask.dtype == pytest.importorskip("numpy").float32


@pytest.mark.parametrize("flatten", [False, True])
def test_training_conversion_masks_before_counting_and_flattening(
    conversion_dtype: Any, flatten: bool
) -> None:
    payloads = [
        {
            "tokens": [10, 11, -7, 13, 14],
            "positions": [0, 1, 0, 1, 2],
            "loss_mask": [0, 1, 1, 0, 1],
            "attention_mask": [1, 2, 3, 4, 5],
        },
        {
            "tokens": [20, 21, 22, 23, 24],
            "positions": [0, 1, 2, 0, 1],
            "loss_mask": [1, 0, 1, 1, 1],
            "attention_mask": [6, 7, 8, 9, 10],
        },
    ]
    original = copy.deepcopy(payloads)
    batch = SampleBatch(
        records=(
            _rec((0, 0, 0), 0, 0, payloads[0], 1),
            _rec((0, 0, 1), 0, 0, payloads[1]),
        )
    )
    out = batch.to_training(
        tokens_field="tokens",
        return_labels=True,
        dtype=conversion_dtype,
        extra_fields=("attention_mask",),
        ignore_index=-7,
        rename_fields={"input_ids": "input"},
        flatten=flatten,
        exclude_fields=("ids", "texts"),
        return_num_valid_tokens=True,
        return_loss_mask=True,
    )
    expected = (
        {
            "input": [10, 11, -7, 13, 20, 21, 22, 23],
            "labels": [11, -7, -7, -7, -7, 22, 23, 24],
            "positions": [0, 1, 0, 1, 0, 1, 2, 0],
            "attention_mask": [1, 2, 3, 4, 6, 7, 8, 9],
        }
        if flatten
        else {
            "input": [[10, 11, -7, 13], [20, 21, 22, 23]],
            "labels": [[11, -7, -7, -7], [-7, 22, 23, 24]],
            "positions": [[0, 1, 0, 1], [0, 1, 2, 0]],
            "attention_mask": [[1, 2, 3, 4], [6, 7, 8, 9]],
        }
    )
    expected["loss_mask"] = (
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        if flatten
        else [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]]
    )
    _assert_loss_mask_dtype(out["loss_mask"], conversion_dtype)
    assert set(out) == {*expected, "num_valid_tokens"}
    assert out["num_valid_tokens"] == 4
    assert type(out["num_valid_tokens"]) is int
    for name, values in expected.items():
        assert _values(out[name]) == values
        if conversion_dtype is not None and name != "loss_mask":
            assert out[name].dtype == conversion_dtype
    assert payloads == original


def test_flatten_without_labels_preserves_record_fields(conversion_dtype: Any) -> None:
    batch = SampleBatch(
        records=(
            _rec(
                (0, 0, 0), 0, 0, {"input_ids": [1, 2], "loss_mask": [0, 1], "text": "a"}
            ),
            _rec(
                (0, 0, 1), 0, 0, {"input_ids": [3, 4], "loss_mask": [1, 0], "text": "b"}
            ),
        )
    )
    out = batch.to_training(dtype=conversion_dtype, flatten=True)
    assert set(out) == {"ids", "texts", "input_ids", "loss_mask"}
    assert out["ids"] == [(0, 0, 0), (0, 0, 1)]
    assert out["texts"] == ["a", "b"]
    assert _values(out["input_ids"]) == [1, 2, 3, 4]
    assert _values(out["loss_mask"]) == [0, 1, 1, 0]


def test_exclusion_uses_renamed_keys_and_keeps_count(conversion_dtype: Any) -> None:
    batch = SampleBatch(records=(_rec((0, 0, 0), 0, 0, {"input_ids": [1, 2, 3]}),))
    out = batch.to_training(
        dtype=conversion_dtype,
        return_labels=True,
        return_num_valid_tokens=True,
        rename_fields={"input_ids": "input", "num_valid_tokens": "count"},
        exclude_fields=("ids", "texts", "labels", "input", "absent"),
    )
    assert out == {"count": 2}
    out = batch.to_training(
        dtype=None,
        rename_fields={"input_ids": "input"},
        exclude_fields=("ids", "texts", "input_ids"),
    )
    assert out == {"input": [[1, 2, 3]]}


@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize("tokens", [[], [1], [1, -100, -100]])
def test_zero_valid_tokens(
    conversion_dtype: Any, flatten: bool, tokens: list[int]
) -> None:
    batch = SampleBatch(records=(_rec((0, 0, 0), 0, 0, {"input_ids": tokens}),))
    out = batch.to_training(
        dtype=conversion_dtype,
        return_labels=True,
        return_num_valid_tokens=True,
        return_loss_mask=True,
        flatten=flatten,
    )
    assert out["num_valid_tokens"] == 0
    assert type(out["num_valid_tokens"]) is int
    expected = tokens[1:] if flatten else [tokens[1:]]
    assert _values(out["labels"]) == expected
    expected_mask = [0.0] * len(tokens[1:])
    assert _values(out["loss_mask"]) == (expected_mask if flatten else [expected_mask])
    _assert_loss_mask_dtype(out["loss_mask"], conversion_dtype)


def test_empty_batch_preserves_defaults_and_applies_exclusions() -> None:
    batch = SampleBatch(records=())
    assert batch.to_training() == {"ids": [], "texts": []}
    assert batch.to_training(
        return_labels=True,
        return_num_valid_tokens=True,
        flatten=True,
        rename_fields={"input_ids": "input"},
        exclude_fields=("ids", "texts", "positions"),
    ) == {"num_valid_tokens": 0}
    assert batch.to_training(exclude_fields=("ids", "texts")) == {}


@pytest.mark.parametrize("empty", [False, True])
def test_token_count_requires_labels(empty: bool) -> None:
    records = () if empty else (_rec((0, 0, 0), 0, 0, {"input_ids": [1, 2]}),)
    with pytest.raises(ValueError, match="requires return_labels=True"):
        SampleBatch(records=records).to_training(return_num_valid_tokens=True)


def test_exclusion_rejects_bare_string() -> None:
    with pytest.raises(TypeError, match="sequence of names, not a string"):
        SampleBatch(records=()).to_training(exclude_fields="ids")


def test_requested_count_cannot_overwrite_extra_field() -> None:
    batch = SampleBatch(
        records=(
            _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2], "num_valid_tokens": [3, 4]}),
        )
    )
    with pytest.raises(
        ValueError, match="extra_fields cannot include 'num_valid_tokens'"
    ):
        batch.to_training(
            dtype=None,
            return_labels=True,
            return_num_valid_tokens=True,
            extra_fields=("num_valid_tokens",),
        )
    # Without a requested count, this is an ordinary extra field.
    out = batch.to_training(
        dtype=None, flatten=True, extra_fields=("num_valid_tokens",)
    )
    assert out["num_valid_tokens"] == [3, 4]


def test_rename_collision_is_checked_before_exclusion() -> None:
    batch = SampleBatch(records=(_rec((0, 0, 0), 0, 0, {"input_ids": [1, 2]}),))
    with pytest.raises(ValueError, match="target keys collide"):
        batch.to_training(
            dtype=None,
            return_labels=True,
            return_num_valid_tokens=True,
            rename_fields={"labels": "num_valid_tokens"},
            exclude_fields=("num_valid_tokens",),
        )


@pytest.mark.parametrize("array_framework", ["numpy", "torch"])
def test_flatten_array_payloads(conversion_dtype: Any, array_framework: str) -> None:
    framework = pytest.importorskip(array_framework)
    as_array = framework.array if array_framework == "numpy" else framework.tensor
    batch = SampleBatch(
        records=(
            _rec((0, 0, 0), 0, 0, as_array([1, 2, 3])),
            _rec((0, 0, 1), 0, 0, as_array([4, 5, 6])),
        )
    )
    out = batch.to_training(
        dtype=conversion_dtype,
        return_labels=True,
        return_num_valid_tokens=True,
        return_loss_mask=True,
        flatten=True,
        exclude_fields=("ids", "texts"),
    )
    assert _values(out["input_ids"]) == [1, 2, 4, 5]
    assert _values(out["labels"]) == [2, 3, 5, 6]
    assert _values(out["loss_mask"]) == [1.0, 1.0, 1.0, 1.0]
    _assert_loss_mask_dtype(out["loss_mask"], conversion_dtype)
    assert out["num_valid_tokens"] == 4
    assert type(out["num_valid_tokens"]) is int


def test_flatten_ragged_lists_shifts_within_each_row() -> None:
    batch = SampleBatch(
        records=(
            _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2]}),
            _rec((0, 0, 1), 0, 0, {"input_ids": [3, 4, 5]}),
        )
    )
    out = batch.to_training(
        dtype=None,
        return_labels=True,
        flatten=True,
        return_num_valid_tokens=True,
        return_loss_mask=True,
    )
    assert out["input_ids"] == [1, 3, 4]
    assert out["labels"] == [2, 4, 5]
    assert out["num_valid_tokens"] == 3
    assert out["loss_mask"] == [1.0, 1.0, 1.0]


def test_return_loss_mask_without_payload_mask(conversion_dtype: Any) -> None:
    # Token ID 0 is real content as well as the pad ID; only pad positions vanish.
    batch = SampleBatch(
        records=(
            _rec((0, 0, 0), 0, 0, {"input_ids": [10, 0, 12, 0]}, padding_length=1),
            _rec((0, 0, 1), 0, 0, {"input_ids": [20, 21, 22, 23]}),
        )
    )
    out = batch.to_training(
        dtype=conversion_dtype, return_labels=True, return_loss_mask=True
    )
    assert _values(out["labels"]) == [[0, 12, -100], [21, 22, 23]]
    assert _values(out["loss_mask"]) == [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
    _assert_loss_mask_dtype(out["loss_mask"], conversion_dtype)


@pytest.mark.parametrize("empty", [False, True])
def test_return_loss_mask_requires_labels(empty: bool) -> None:
    records = (
        ()
        if empty
        else (_rec((0, 0, 0), 0, 0, {"input_ids": [1, 2], "loss_mask": [0, 1]}),)
    )
    with pytest.raises(
        ValueError, match="return_loss_mask requires return_labels=True"
    ):
        SampleBatch(records=records).to_training(return_loss_mask=True)


def test_return_loss_mask_empty_batch() -> None:
    batch = SampleBatch(records=())
    out = batch.to_training(
        dtype=None,
        return_labels=True,
        return_loss_mask=True,
        return_num_valid_tokens=True,
        flatten=True,
        rename_fields={"loss_mask": "weights"},
        exclude_fields=("ids", "texts"),
    )
    assert out == {"num_valid_tokens": 0, "weights": []}
    assert (
        batch.to_training(
            return_labels=True,
            return_loss_mask=True,
            rename_fields={"loss_mask": "weights"},
            exclude_fields=("ids", "texts", "weights"),
        )
        == {}
    )


def test_return_loss_mask_rename_and_exclude() -> None:
    batch = SampleBatch(
        records=(
            _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 0]}),
        )
    )
    out = batch.to_training(
        dtype=None,
        return_labels=True,
        return_loss_mask=True,
        flatten=True,
        rename_fields={"loss_mask": "weights"},
        exclude_fields=("ids", "texts", "input_ids", "labels", "loss_mask"),
    )
    assert set(out) == {"weights"}
    assert _values(out["weights"]) == [1.0, 0.0]
    assert (
        batch.to_training(
            dtype=None,
            return_labels=True,
            return_loss_mask=True,
            rename_fields={"loss_mask": "weights"},
            exclude_fields=("ids", "texts", "input_ids", "labels", "weights"),
        )
        == {}
    )
    with pytest.raises(ValueError, match="target keys collide"):
        batch.to_training(
            dtype=None,
            return_labels=True,
            return_loss_mask=True,
            rename_fields={"labels": "loss_mask"},
        )


@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize("return_labels", [False, True])
def test_return_cu_seqlens_alignment_and_batch_offsets(
    conversion_dtype: Any, flatten: bool, return_labels: bool
) -> None:
    payloads = [
        {"tokens": [10, 11, 12, 13, 14, 15, 16], "positions": [0, 1, 0, 1, 2, 0, 0]},
        {"tokens": [20, 21, 22, 23, 24, 25, 26], "positions": [0, 1, 2, 3, 0, 1, 2]},
    ]
    original = copy.deepcopy(payloads)
    batch = SampleBatch(
        records=tuple(
            _rec((0, 0, i), 0, 0, payload) for i, payload in enumerate(payloads)
        )
    )
    out = batch.to_training(
        dtype=conversion_dtype,
        return_labels=return_labels,
        return_loss_mask=return_labels,
        return_num_valid_tokens=return_labels,
        return_cu_seqlens=True,
        flatten=flatten,
    )
    if flatten:
        expected = [0, 2, 5, 6, 10, 12] if return_labels else [0, 2, 5, 6, 7, 11, 14]
        assert _values(out["cu_seqlens"]) == expected
        assert out["max_seqlen"] == 4
        assert type(out["max_seqlen"]) is int
    else:
        expected = (
            [[0, 2, 5, 6], [0, 4, 6, 6]]
            if return_labels
            else [[0, 2, 5, 6, 7], [0, 4, 7, 7, 7]]
        )
        assert _values(out["cu_seqlens"]) == expected
        assert _values(out["max_seqlen"]) == [3, 4]
    if conversion_dtype is not None:
        framework = (
            "torch" if type(out["cu_seqlens"]).__module__ == "torch" else "numpy"
        )
        assert out["cu_seqlens"].dtype == pytest.importorskip(framework).int32
        if not flatten:
            assert out["max_seqlen"].dtype == pytest.importorskip(framework).int32
            assert out["max_seqlen"].shape == (2,)
        if framework == "torch":
            assert out["cu_seqlens"].device == out["input_ids"].device

    # Requesting attention metadata changes neither token data nor loss policy.
    baseline = batch.to_training(
        dtype=conversion_dtype,
        return_labels=return_labels,
        return_loss_mask=return_labels,
        return_num_valid_tokens=return_labels,
        flatten=flatten,
    )
    assert set(out) == {*baseline, "cu_seqlens", "max_seqlen"}
    for key, value in baseline.items():
        if key == "num_valid_tokens":
            assert out[key] == value
        else:
            assert _values(out[key]) == _values(value)
    assert payloads == original


@pytest.mark.parametrize("flatten", [False, True])
def test_return_cu_seqlens_empty_input_after_label_shift(
    conversion_dtype: Any, flatten: bool
) -> None:
    batch = SampleBatch(
        records=(
            _rec(
                (0, 0, 0),
                0,
                0,
                {"input_ids": [1], "positions": [0]},
            ),
        )
    )
    out = batch.to_training(
        dtype=conversion_dtype,
        return_labels=True,
        return_cu_seqlens=True,
        flatten=flatten,
    )
    assert _values(out["cu_seqlens"]) == ([0] if flatten else [[0]])
    if flatten:
        assert out["max_seqlen"] == 0
        assert type(out["max_seqlen"]) is int
    else:
        assert _values(out["max_seqlen"]) == [0]


@pytest.mark.parametrize("flatten", [False, True])
def test_return_cu_seqlens_empty_batch(flatten: bool) -> None:
    out = SampleBatch(records=()).to_training(
        dtype=None,
        return_cu_seqlens=True,
        flatten=flatten,
        exclude_fields=("ids", "texts"),
    )
    assert out == {
        "cu_seqlens": [0] if flatten else [],
        "max_seqlen": 0 if flatten else [],
    }


@pytest.mark.parametrize(
    ("flatten", "exclude_metadata"), [(False, False), (True, True)]
)
def test_empty_training_metadata_rename_and_exclude(
    flatten: bool, exclude_metadata: bool
) -> None:
    expected = {
        "weights": [],
        "padding": [],
        "count": 0,
        "cu": [0] if flatten else [],
        "maximum": 0 if flatten else [],
    }
    excluded = ("sample_ids", "texts")
    if exclude_metadata:
        excluded += tuple(expected)
    out = SampleBatch(records=()).to_training(
        dtype=None,
        return_labels=True,
        return_loss_mask=True,
        return_padding_mask=True,
        return_cu_seqlens=True,
        return_num_valid_tokens=True,
        flatten=flatten,
        rename_fields={
            "ids": "sample_ids",
            "input_ids": "tokens",
            "positions": "position_ids",
            "loss_mask": "weights",
            "padding_mask": "padding",
            "num_valid_tokens": "count",
            "cu_seqlens": "cu",
            "max_seqlen": "maximum",
        },
        exclude_fields=excluded,
    )
    assert out == ({} if exclude_metadata else expected)


@pytest.mark.parametrize(
    ("conversion_dtype", "positions", "message"),
    [
        ("list", [1, 2, 3], "start at zero"),
        ("list", [0, 1], "sequence length"),
        ("list", 0, "sequence length"),
        ("list", [[0], [1], [2]], "2-D"),
        ("numpy", [[0], [1], [2]], "shape"),
    ],
    indirect=["conversion_dtype"],
)
def test_return_cu_seqlens_rejects_invalid_positions(
    conversion_dtype: Any, positions: Any, message: str
) -> None:
    batch = SampleBatch(
        records=(
            _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2, 3], "positions": positions}),
        )
    )
    with pytest.raises(ValueError, match=message):
        batch.to_training(
            dtype=conversion_dtype, return_labels=True, return_cu_seqlens=True
        )


@pytest.mark.parametrize("missing_index", [0, 1])
def test_return_cu_seqlens_requires_positions_in_every_row(missing_index: int) -> None:
    payloads = [{"input_ids": [1, 2], "positions": [0, 1]} for _ in range(2)]
    del payloads[missing_index]["positions"]
    batch = SampleBatch(
        records=tuple(
            _rec((0, 0, i), 0, 0, payload) for i, payload in enumerate(payloads)
        )
    )
    with pytest.raises(ValueError, match="positions"):
        batch.to_training(dtype=None, return_cu_seqlens=True)


@pytest.mark.parametrize("field", ["cu_seqlens", "max_seqlen"])
def test_return_cu_seqlens_rejects_extra_field_collision(field: str) -> None:
    with pytest.raises(ValueError, match="extra_fields cannot include"):
        SampleBatch(records=()).to_training(
            return_cu_seqlens=True, extra_fields=(field,)
        )


def test_return_cu_seqlens_rename_and_exclude() -> None:
    batch = SampleBatch(
        records=(
            _rec((0, 0, 0), 0, 0, {"input_ids": [1, 2, 3], "positions": [0, 1, 0]}),
        )
    )
    out = batch.to_training(
        dtype=None,
        return_cu_seqlens=True,
        flatten=True,
        rename_fields={"cu_seqlens": "cu_seq_q", "max_seqlen": "max_q"},
        exclude_fields=("ids", "texts", "input_ids", "positions", "cu_seqlens"),
    )
    assert set(out) == {"cu_seq_q", "max_q"}
    assert _values(out["cu_seq_q"]) == [0, 2, 3]
    assert out["max_q"] == 2
    assert (
        batch.to_training(
            dtype=None,
            return_cu_seqlens=True,
            rename_fields={"cu_seqlens": "cu"},
            exclude_fields=(
                "ids",
                "texts",
                "input_ids",
                "positions",
                "cu",
                "max_seqlen",
            ),
        )
        == {}
    )
    with pytest.raises(ValueError, match="target keys collide"):
        batch.to_training(
            dtype=None,
            return_cu_seqlens=True,
            rename_fields={"positions": "cu_seqlens"},
        )


def test_return_cu_seqlens_ragged_lists_and_empty_rows() -> None:
    batch = SampleBatch(
        records=tuple(
            _rec((0, 0, i), 0, 0, {"input_ids": [1] * len(row), "positions": row})
            for i, row in enumerate([[], [0, 1, 0], [], [0, 1]])
        )
    )
    out = batch.to_training(dtype=None, return_cu_seqlens=True, flatten=True)
    assert out["cu_seqlens"] == [0, 2, 3, 5]
    assert out["max_seqlen"] == 2
    out = batch.to_training(dtype=None, return_cu_seqlens=True)
    assert out["cu_seqlens"] == [[0, 0, 0], [0, 2, 3], [0, 0, 0], [0, 2, 2]]
    assert out["max_seqlen"] == [0, 2, 0, 2]


@pytest.mark.parametrize("flatten", [False, True])
def test_eos_masking_and_sequence_positions_preserve_document_boundaries(
    conversion_dtype: Any, flatten: bool
) -> None:
    # Distinct BOS=1, EOS=2, PAD=0. Row 0 also masks a prompt token and pads.
    payloads = [
        {
            "tokens": [1, 10, 2, 1, 20, 2, 0, 0],
            "positions": [0, 1, 2, 0, 1, 2, 0, 1],
            "loss_mask": [0, 0, 1, 1, 1, 1, 1, 1],
        },
        {
            "tokens": [1, 30, 31, 2, 1, 40, 41, 2],
            "positions": [0, 1, 2, 3, 0, 1, 2, 3],
            "loss_mask": [1] * 8,
        },
    ]
    original = copy.deepcopy(payloads)
    batch = SampleBatch(
        records=(
            _rec((0, 0, 0), 0, 0, payloads[0], padding_length=2),
            _rec((0, 0, 1), 0, 0, payloads[1]),
        )
    )
    options: dict[str, Any] = dict(
        dtype=conversion_dtype,
        return_labels=True,
        ignore_index=-7,
        return_loss_mask=True,
        return_num_valid_tokens=True,
        return_cu_seqlens=True,
        flatten=flatten,
        rename_fields={"input_ids": "tokens", "positions": "position_ids"},
        exclude_fields=("ids", "texts"),
    )
    baseline = batch.to_training(**options)
    with pytest.warns(UserWarning, match="eos_token_id is ignored") as caught:
        disabled = batch.to_training(**options, eos_token_id=2)
    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert {
        k: _values(v) if hasattr(v, "tolist") else v for k, v in disabled.items()
    } == {k: _values(v) if hasattr(v, "tolist") else v for k, v in baseline.items()}
    out = batch.to_training(
        **options, eos_mask_loss=True, eos_token_id=2, position_mode="sequence"
    )
    expected = {
        "tokens": [[1, 10, 2, 1, 20, 2, 0], [1, 30, 31, 2, 1, 40, 41]],
        "labels": [[-7, 2, -7, 20, 2, -7, -7], [30, 31, 2, -7, 40, 41, 2]],
        "loss_mask": [
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        ],
        "position_ids": [list(range(7)), list(range(7))],
    }
    assert set(out) == {*expected, "cu_seqlens", "max_seqlen", "num_valid_tokens"}
    for name, rows in expected.items():
        assert _values(out[name]) == (
            [value for row in rows for value in row] if flatten else rows
        )
    assert out["num_valid_tokens"] == 9
    assert baseline["num_valid_tokens"] == 11
    original_positions = [payload["positions"][:-1] for payload in original]
    assert _values(baseline["position_ids"]) == (
        [value for row in original_positions for value in row]
        if flatten
        else original_positions
    )
    assert _values(out["cu_seqlens"]) == (
        [0, 3, 6, 7, 11, 14] if flatten else [[0, 3, 6, 7], [0, 4, 7, 7]]
    )
    assert _values(out["cu_seqlens"]) == _values(baseline["cu_seqlens"])
    assert (out["max_seqlen"] if flatten else _values(out["max_seqlen"])) == (
        4 if flatten else [3, 4]
    )
    _assert_loss_mask_dtype(out["loss_mask"], conversion_dtype)
    if conversion_dtype is not None:
        assert out["labels"].dtype == conversion_dtype
        assert out["position_ids"].dtype == conversion_dtype
        if type(out["tokens"]).__module__ == "torch":
            assert out["position_ids"].device == out["tokens"].device
    assert payloads == original


def test_eos_masking_matches_every_occurrence_of_shared_bos_eos_id() -> None:
    # ID zero is valid. Literal ID matching also masks first-content targets.
    batch = SampleBatch(
        records=(_rec((0, 0, 0), 0, 0, {"tokens": [0, 11, 0, 0, 22, 0]}),)
    )
    out = batch.to_training(
        dtype=None, return_labels=True, eos_mask_loss=True, eos_token_id=0
    )
    assert out["input_ids"] == [[0, 11, 0, 0, 22]]
    assert out["labels"] == [[-100, 0, -100, -100, 0]]
    assert "loss_mask" not in out
    assert "positions" not in out


def test_sequence_positions_without_payload_positions(conversion_dtype: Any) -> None:
    batch = SampleBatch(
        records=tuple(_rec((0, 0, i), 0, 0, {"tokens": [10, 11, 12]}) for i in range(2))
    )
    out = batch.to_training(dtype=conversion_dtype, position_mode="sequence")
    assert _values(out["positions"]) == [[0, 1, 2], [0, 1, 2]]
    assert _values(out["input_ids"]) == [[10, 11, 12], [10, 11, 12]]
    if conversion_dtype is not None:
        if type(out["positions"]).__module__ == "torch":
            # Tensor-parallel broadcast helpers commonly flatten with view().
            assert out["positions"].view(-1).tolist() == [0, 1, 2, 0, 1, 2]
        out["positions"][0, 0] = 7
        assert out["positions"][1, 0] == 0
    # Synthetic positions cannot stand in for missing document boundaries.
    with pytest.raises(ValueError, match="requires a 'positions' field"):
        batch.to_training(
            dtype=conversion_dtype, position_mode="sequence", return_cu_seqlens=True
        )


@pytest.mark.parametrize("tokens", [[], [2]])
def test_training_policies_with_empty_rows(
    conversion_dtype: Any, tokens: list[int]
) -> None:
    batch = SampleBatch(records=(_rec((0, 0, 0), 0, 0, {"tokens": tokens}),))
    out = batch.to_training(
        dtype=conversion_dtype,
        return_labels=True,
        return_loss_mask=True,
        return_num_valid_tokens=True,
        eos_mask_loss=True,
        eos_token_id=2,
        position_mode="sequence",
    )
    for name in ("input_ids", "labels", "loss_mask", "positions"):
        assert _values(out[name]) == [[]]
    assert out["num_valid_tokens"] == 0


def test_training_policies_with_ragged_lists() -> None:
    batch = SampleBatch(
        records=tuple(
            _rec((0, 0, i), 0, 0, {"tokens": row})
            for i, row in enumerate([[], [2, 10, 2], [11, 2, 12, 13]])
        )
    )
    out = batch.to_training(
        dtype=None,
        return_labels=True,
        flatten=True,
        eos_mask_loss=True,
        eos_token_id=2,
        position_mode="sequence",
    )
    assert out["input_ids"] == [2, 10, 11, 2, 12]
    assert out["labels"] == [-100, 2, 2, -100, 13]
    assert out["positions"] == [0, 1, 0, 1, 2]


def test_training_policies_with_empty_batch() -> None:
    out = SampleBatch(records=()).to_training(
        return_labels=True,
        return_loss_mask=True,
        return_num_valid_tokens=True,
        return_cu_seqlens=True,
        eos_mask_loss=True,
        eos_token_id=2,
        position_mode="sequence",
        flatten=True,
        rename_fields={"positions": "position_ids"},
        exclude_fields=("ids", "texts"),
    )
    assert out == {
        "position_ids": [],
        "loss_mask": [],
        "num_valid_tokens": 0,
        "cu_seqlens": [0],
        "max_seqlen": 0,
    }


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"position_mode": "document"}, "position_mode must be"),
        ({"eos_mask_loss": True, "eos_token_id": 2}, "requires return_labels=True"),
        *[
            ({"eos_token_id": value}, "nonnegative integer")
            for value in (-1, 1.5, "2", True)
        ],
    ],
)
def test_training_policies_reject_invalid_options(
    options: dict[str, Any], message: str
) -> None:
    # Validation precedes the empty-batch fast path and framework selection.
    with pytest.raises(ValueError, match=message):
        SampleBatch(records=()).to_training(dtype=None, **options)


def test_sequence_positions_rejects_nested_tokens(conversion_dtype: Any) -> None:
    batch = SampleBatch(records=(_rec((0, 0, 0), 0, 0, {"tokens": [[1, 2], [3, 4]]}),))
    with pytest.raises(ValueError, match="requires 2-D input_ids"):
        batch.to_training(dtype=conversion_dtype, position_mode="sequence")


@pytest.mark.parametrize("empty", [False, True])
def test_eos_masking_requires_explicit_id(empty: bool) -> None:
    records = () if empty else (_rec((0, 0, 0), 0, 0, {"tokens": [1, 2]}),)
    batch = SampleBatch(records=records)
    with pytest.raises(ValueError, match="requires an explicit eos_token_id"):
        batch.to_training(dtype=None, return_labels=True, eos_mask_loss=True)


def test_eos_masking_accepts_numpy_integer_id() -> None:
    np = pytest.importorskip("numpy")
    batch = SampleBatch(records=(_rec((0, 0, 0), 0, 0, {"tokens": [7, 2, 8]}),))
    out = batch.to_training(
        dtype=None, return_labels=True, eos_mask_loss=True, eos_token_id=np.int64(2)
    )
    assert out["labels"] == [[2, -100]]


@pytest.mark.parametrize(
    ("options", "expected_labels"),
    [
        ({}, [7, 2, 1, 8, 2]),
        ({"eos_mask_loss": True, "eos_token_id": 2}, [7, 2, -100, 8, 2]),
        ({"eos_mask_loss": True, "eos_token_id": 1}, [-100, 2, 1, -100, 2]),
    ],
)
def test_explicit_token_field_is_independent_of_eos_policy(
    options: dict[str, Any], expected_labels: list[int]
) -> None:
    record = _rec(
        (0, 0, 0),
        0,
        0,
        {
            "input_ids": [9] * 6,
            "encoded": [1, 7, 2, 1, 8, 2],
            "positions": [0, 1, 2, 0, 1, 2],
        },
    )
    batch = SampleBatch(records=(record,))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = batch.to_training(
            dtype=None,
            tokens_field="encoded",
            return_labels=True,
            return_loss_mask=True,
            return_cu_seqlens=True,
            position_mode="sequence",
            rename_fields={"input_ids": "tokens", "labels": "targets"},
            **options,
        )
    assert result["tokens"] == [[1, 7, 2, 1, 8]]
    assert result["targets"] == [expected_labels]
    assert result["loss_mask"] == [[float(label != -100) for label in expected_labels]]
    assert result["positions"] == [[0, 1, 2, 3, 4]]
    assert result["cu_seqlens"] == [[0, 3, 5]]
    assert "input_ids" not in result and "labels" not in result


@pytest.mark.parametrize("missing_index", [0, 1])
def test_missing_explicit_token_field_does_not_fall_back(missing_index: int) -> None:
    payloads = [{"encoded": [1, 2], "input_ids": [9, 9]} for _ in range(2)]
    del payloads[missing_index]["encoded"]
    batch = SampleBatch(
        records=tuple(
            _rec((0, 0, i), 0, 0, payload) for i, payload in enumerate(payloads)
        ),
    )
    with pytest.raises(
        ValueError,
        match=f"Field 'encoded' not found in payload at index {missing_index}",
    ):
        batch.to_training(dtype=None, tokens_field="encoded", return_labels=True)


@pytest.mark.parametrize("boundaries", [False, True])
@pytest.mark.parametrize("explicit_positions", [False, True])
def test_sequence_positions_only_stack_originals_for_attention_boundaries(
    conversion_dtype: Any,
    boundaries: bool,
    explicit_positions: bool,
) -> None:
    from unittest.mock import patch

    import zephon.types as types_module

    batch = SampleBatch(
        records=(
            _rec(
                (0, 0, 0),
                0,
                0,
                {"tokens": [1, 10, 2, 1, 20, 2], "positions": [0, 1, 2, 0, 1, 2]},
            ),
        )
    )
    with patch.object(
        types_module, "_stack_sequences", wraps=types_module._stack_sequences
    ) as stack:
        out = batch.to_training(
            dtype=conversion_dtype,
            return_labels=True,
            position_mode="sequence",
            return_cu_seqlens=boundaries,
            extra_fields=("positions",) if explicit_positions else (),
        )
    assert stack.call_count == (2 if boundaries else 1)
    assert _values(out["positions"]) == [[0, 1, 2, 3, 4]]
    if boundaries:
        assert _values(out["cu_seqlens"]) == [[0, 3, 5]]
