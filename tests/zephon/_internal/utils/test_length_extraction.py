# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for length_extraction module."""

from __future__ import annotations

import pytest

from zephon._internal.utils.length_extraction import (
    TOKEN_FIELD_CANDIDATES,
    _get_length,
    _has_length,
    detect_length_field,
    extract_length,
)
from zephon.types import SampleMeta, SampleRecord

# ---------------------------------------------------------------------------
# Mock classes for testing tensor-like behavior
# ---------------------------------------------------------------------------


class MockScalarTensor:
    """Mock scalar tensor with .item() method."""

    def __init__(self, value: int) -> None:
        self._value = value

    def item(self) -> int:
        return self._value


class MockTensor:
    """Mock tensor with .shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class MockTensorWithItem:
    """Mock tensor with both .shape and .item() (like 0-d tensor)."""

    def __init__(self, value: int) -> None:
        self._value = value
        self.shape = ()  # 0-dimensional

    def item(self) -> int:
        return self._value


class MockFailingItem:
    """Mock object with .item() that raises TypeError."""

    def item(self) -> int:
        raise TypeError("Cannot convert to int")


class MockFailingItemWithShape:
    """Mock object with failing .item() but valid .shape."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def item(self) -> int:
        raise TypeError("Cannot convert to int")


class MockEmptyShape:
    """Mock object with empty shape (like scalar tensor)."""

    def __init__(self) -> None:
        self.shape = ()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _make_record(payload: dict | object) -> SampleRecord:
    """Create a SampleRecord with the given payload."""
    meta = SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0)
    return SampleRecord(meta=meta, payload=payload)


# ---------------------------------------------------------------------------
# Tests for TOKEN_FIELD_CANDIDATES constant
# ---------------------------------------------------------------------------


def test_token_field_candidates_order() -> None:
    """Test that TOKEN_FIELD_CANDIDATES has expected order."""
    assert TOKEN_FIELD_CANDIDATES == ["input_ids", "tokens", "token_ids", "ids"]


# ---------------------------------------------------------------------------
# Tests for detect_length_field()
# ---------------------------------------------------------------------------


class TestDetectLengthField:
    """Tests for detect_length_field function."""

    def test_returns_first_matching_candidate(self) -> None:
        """Test that first matching candidate is returned."""
        payload = {"input_ids": [1, 2, 3]}
        assert detect_length_field(payload) == "input_ids"

    def test_returns_none_when_no_candidates(self) -> None:
        """Test that None is returned when no candidate field exists."""
        payload = {"text": "hello", "label": 1}
        assert detect_length_field(payload) is None

    def test_returns_none_for_empty_payload(self) -> None:
        """Test that None is returned for empty payload."""
        assert detect_length_field({}) is None

    def test_priority_order_input_ids_first(self) -> None:
        """Test that input_ids has highest priority."""
        payload = {
            "ids": [1],
            "token_ids": [1, 2],
            "tokens": [1, 2, 3],
            "input_ids": [1, 2, 3, 4],
        }
        assert detect_length_field(payload) == "input_ids"

    def test_priority_order_tokens_second(self) -> None:
        """Test that tokens has second priority when input_ids absent."""
        payload = {
            "ids": [1],
            "token_ids": [1, 2],
            "tokens": [1, 2, 3],
        }
        assert detect_length_field(payload) == "tokens"

    def test_priority_order_token_ids_third(self) -> None:
        """Test that token_ids has third priority."""
        payload = {
            "ids": [1],
            "token_ids": [1, 2],
        }
        assert detect_length_field(payload) == "token_ids"

    def test_priority_order_ids_last(self) -> None:
        """Test that ids has lowest priority."""
        payload = {"ids": [1]}
        assert detect_length_field(payload) == "ids"

    def test_skips_field_without_length_support(self) -> None:
        """Test that fields without length support are skipped."""
        # None doesn't support length extraction
        payload = {"input_ids": None, "tokens": [1, 2, 3]}
        assert detect_length_field(payload) == "tokens"

    def test_works_with_int_values(self) -> None:
        """Test detection with int values (used for pre-computed lengths)."""
        payload = {"input_ids": 42}
        assert detect_length_field(payload) == "input_ids"

    def test_works_with_list(self) -> None:
        """Test detection with list values."""
        payload = {"tokens": [1, 2, 3, 4, 5]}
        assert detect_length_field(payload) == "tokens"

    def test_works_with_tuple(self) -> None:
        """Test detection with tuple values."""
        payload = {"tokens": (1, 2, 3)}
        assert detect_length_field(payload) == "tokens"

    def test_works_with_string(self) -> None:
        """Test detection with string values (Sized)."""
        payload = {"tokens": "hello"}
        assert detect_length_field(payload) == "tokens"

    def test_works_with_tensor_shape(self) -> None:
        """Test detection with tensor-like objects with .shape."""
        payload = {"input_ids": MockTensor((10, 512))}
        assert detect_length_field(payload) == "input_ids"

    def test_works_with_scalar_tensor_item(self) -> None:
        """Test detection with scalar tensor-like objects with .item()."""
        payload = {"input_ids": MockScalarTensor(100)}
        assert detect_length_field(payload) == "input_ids"


# ---------------------------------------------------------------------------
# Tests for extract_length()
# ---------------------------------------------------------------------------


class TestExtractLength:
    """Tests for extract_length function."""

    def test_auto_detects_field(self) -> None:
        """Test auto-detection when field=None."""
        record = _make_record({"input_ids": [1, 2, 3, 4, 5]})
        assert extract_length(record, None) == 5

    def test_uses_explicit_field(self) -> None:
        """Test explicit field name is used."""
        record = _make_record({"my_custom_field": [1, 2, 3]})
        assert extract_length(record, "my_custom_field") == 3

    def test_explicit_field_overrides_auto(self) -> None:
        """Test explicit field takes precedence over auto-detection."""
        record = _make_record(
            {
                "input_ids": [1, 2, 3, 4, 5],  # would be auto-detected
                "other_field": [1, 2],  # explicit
            }
        )
        assert extract_length(record, "other_field") == 2

    def test_raises_type_error_for_non_dict_payload(self) -> None:
        """Test TypeError is raised for non-dict payload."""
        record = _make_record("not a dict")
        with pytest.raises(TypeError, match="requires dict payload"):
            extract_length(record, None)

    def test_raises_type_error_for_list_payload(self) -> None:
        """Test TypeError is raised for list payload."""
        record = _make_record([1, 2, 3])
        with pytest.raises(TypeError, match="requires dict payload"):
            extract_length(record, None)

    def test_raises_value_error_when_auto_detect_fails(self) -> None:
        """Test ValueError when no candidate field found."""
        record = _make_record({"text": "hello", "label": 1})
        with pytest.raises(ValueError, match="Cannot auto-detect length field"):
            extract_length(record, None)

    def test_raises_value_error_when_explicit_field_missing(self) -> None:
        """Test ValueError when explicit field not in payload."""
        record = _make_record({"input_ids": [1, 2, 3]})
        with pytest.raises(ValueError, match="Field 'missing' not found"):
            extract_length(record, "missing")

    def test_extracts_from_int(self) -> None:
        """Test extraction from int value."""
        record = _make_record({"input_ids": 42})
        assert extract_length(record, None) == 42

    def test_extracts_from_list(self) -> None:
        """Test extraction from list."""
        record = _make_record({"tokens": [1, 2, 3, 4]})
        assert extract_length(record, None) == 4

    def test_extracts_from_tuple(self) -> None:
        """Test extraction from tuple."""
        record = _make_record({"tokens": (1, 2, 3)})
        assert extract_length(record, None) == 3

    def test_extracts_from_tensor_shape(self) -> None:
        """Test extraction from tensor-like .shape[0]."""
        record = _make_record({"input_ids": MockTensor((128, 512))})
        assert extract_length(record, None) == 128

    def test_extracts_from_scalar_tensor_item(self) -> None:
        """Test extraction from scalar tensor .item()."""
        record = _make_record({"input_ids": MockScalarTensor(256)})
        assert extract_length(record, None) == 256

    def test_raises_type_error_for_unsupported_value(self) -> None:
        """Test TypeError for unsupported value type."""
        # Use explicit field to bypass auto-detect (which would fail first)
        record = _make_record({"custom_field": object()})
        with pytest.raises(
            TypeError, match="must be int, sequence-like, or tensor-like"
        ):
            extract_length(record, "custom_field")

    def test_error_message_includes_payload_keys(self) -> None:
        """Test that auto-detect error message includes payload keys."""
        record = _make_record({"foo": [1], "bar": [2]})
        with pytest.raises(ValueError) as exc_info:
            extract_length(record, None)
        assert "foo" in str(exc_info.value)
        assert "bar" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests for _has_length() (internal function)
# ---------------------------------------------------------------------------


class TestHasLength:
    """Tests for _has_length internal function."""

    def test_returns_true_for_int(self) -> None:
        """Test returns True for int."""
        assert _has_length(42) is True
        assert _has_length(0) is True
        assert _has_length(-1) is True

    def test_returns_true_for_list(self) -> None:
        """Test returns True for list."""
        assert _has_length([1, 2, 3]) is True
        assert _has_length([]) is True

    def test_returns_true_for_tuple(self) -> None:
        """Test returns True for tuple."""
        assert _has_length((1, 2)) is True
        assert _has_length(()) is True

    def test_returns_true_for_string(self) -> None:
        """Test returns True for string."""
        assert _has_length("hello") is True
        assert _has_length("") is True

    def test_returns_true_for_dict(self) -> None:
        """Test returns True for dict (Sized)."""
        assert _has_length({"a": 1}) is True

    def test_returns_true_for_set(self) -> None:
        """Test returns True for set (Sized)."""
        assert _has_length({1, 2, 3}) is True

    def test_returns_true_for_object_with_item(self) -> None:
        """Test returns True for object with .item() method."""
        assert _has_length(MockScalarTensor(10)) is True

    def test_returns_true_for_object_with_shape(self) -> None:
        """Test returns True for object with .shape attribute."""
        assert _has_length(MockTensor((10, 20))) is True

    def test_returns_false_for_none(self) -> None:
        """Test returns False for None."""
        assert _has_length(None) is False

    def test_returns_false_for_plain_object(self) -> None:
        """Test returns False for plain object without length support."""
        assert _has_length(object()) is False

    def test_returns_false_for_empty_shape(self) -> None:
        """Test returns False for object with empty .shape."""
        assert _has_length(MockEmptyShape()) is False

    def test_returns_true_for_failing_item_with_shape(self) -> None:
        """Test returns True when .item() fails but .shape works."""
        obj = MockFailingItemWithShape((5,))
        assert _has_length(obj) is True


# ---------------------------------------------------------------------------
# Tests for _get_length() (internal function)
# ---------------------------------------------------------------------------


class TestGetLength:
    """Tests for _get_length internal function."""

    def test_returns_int_directly(self) -> None:
        """Test int is returned directly."""
        assert _get_length(42, "field") == 42
        assert _get_length(0, "field") == 0
        assert _get_length(-5, "field") == -5

    def test_extracts_via_item(self) -> None:
        """Test extraction via .item() for scalar tensor."""
        assert _get_length(MockScalarTensor(100), "field") == 100

    def test_extracts_via_shape(self) -> None:
        """Test extraction via .shape[0] for tensor."""
        assert _get_length(MockTensor((64, 512)), "field") == 64

    def test_extracts_via_len_for_list(self) -> None:
        """Test extraction via len() for list."""
        assert _get_length([1, 2, 3, 4, 5], "field") == 5

    def test_extracts_via_len_for_tuple(self) -> None:
        """Test extraction via len() for tuple."""
        assert _get_length((1, 2, 3), "field") == 3

    def test_extracts_via_len_for_string(self) -> None:
        """Test extraction via len() for string."""
        assert _get_length("hello", "field") == 5

    def test_prefers_item_over_shape_when_item_works(self) -> None:
        """Test that .item() is tried before .shape."""
        # MockTensorWithItem has both; .item() should be used
        obj = MockTensorWithItem(99)
        assert _get_length(obj, "field") == 99

    def test_falls_back_to_shape_when_item_fails(self) -> None:
        """Test fallback to .shape when .item() fails."""
        obj = MockFailingItemWithShape((42,))
        assert _get_length(obj, "field") == 42

    def test_raises_type_error_for_unsupported_type(self) -> None:
        """Test TypeError for unsupported type."""
        with pytest.raises(
            TypeError, match="must be int, sequence-like, or tensor-like"
        ):
            _get_length(object(), "my_field")

    def test_error_includes_field_name(self) -> None:
        """Test that error message includes field name."""
        with pytest.raises(TypeError) as exc_info:
            _get_length(object(), "custom_field")
        assert "custom_field" in str(exc_info.value)

    def test_raises_for_none(self) -> None:
        """Test TypeError for None value."""
        with pytest.raises(TypeError):
            _get_length(None, "field")

    def test_empty_list_returns_zero(self) -> None:
        """Test empty list returns 0."""
        assert _get_length([], "field") == 0

    def test_single_element_shape(self) -> None:
        """Test tensor with 1D shape."""
        assert _get_length(MockTensor((100,)), "field") == 100


# ---------------------------------------------------------------------------
# End-to-end usage tests
# ---------------------------------------------------------------------------


class TestLengthExtractionUsage:
    """End-to-end usage tests combining multiple functions."""

    def test_end_to_end_with_list_tokens(self) -> None:
        """Test full extraction flow with list tokens."""
        record = _make_record(
            {
                "text": "Hello world",
                "tokens": [101, 7592, 2088, 102],
            }
        )
        assert extract_length(record) == 4

    def test_end_to_end_with_input_ids_tensor(self) -> None:
        """Test full extraction flow with tensor-like input_ids."""
        record = _make_record(
            {
                "input_ids": MockTensor((512,)),
                "attention_mask": MockTensor((512,)),
            }
        )
        assert extract_length(record) == 512

    def test_end_to_end_with_precomputed_length(self) -> None:
        """Test full extraction flow with pre-computed int length."""
        record = _make_record(
            {
                "tokens": 1024,  # pre-computed length stored as int
            }
        )
        assert extract_length(record) == 1024

    def test_multiple_records_consistent(self) -> None:
        """Test extraction is consistent across multiple records."""
        records = [
            _make_record({"input_ids": [1, 2, 3]}),
            _make_record({"input_ids": [1, 2, 3, 4, 5]}),
            _make_record({"input_ids": [1]}),
        ]
        lengths = [extract_length(r) for r in records]
        assert lengths == [3, 5, 1]
