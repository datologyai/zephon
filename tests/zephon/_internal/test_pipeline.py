"""Tests for pipeline option validation."""

import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon._internal.pipeline import Pipeline


def test_removed_payload_memory_options_fail_before_updating_pipeline() -> None:
    pipe = Pipeline(FakeIndexableWorkSource(make_inmem_dataset("tiny", [{"x": 1}])))
    pipe.options(shm_min_item_bytes=456, shm_coalesce=True)
    assert pipe._options.shm_min_item_bytes == 456
    assert pipe._options.shm_coalesce
    for old, new, value in (
        ("shm_min_size", "shm_min_item_bytes", 123),
        ("coalesce_tensors", "shm_coalesce", False),
    ):
        for supplied in (value, None):
            with pytest.raises(
                TypeError,
                match=f"'{old}' is no longer supported.*Use '{new}' instead",
            ):
                pipe.options(max_workers=99, **{old: supplied, new: value})
    assert pipe._options.max_workers is None
    assert pipe._options.shm_min_item_bytes == 456
    assert pipe._options.shm_coalesce
    with pytest.raises(ValueError, match="shm_coalesce cannot be None"):
        pipe.options(shm_coalesce=None)
