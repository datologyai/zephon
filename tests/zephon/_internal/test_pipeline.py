"""Tests for pipeline option validation."""

import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon._internal.pipeline import Pipeline


def test_removed_payload_memory_options_fail_before_updating_pipeline() -> None:
    pipe = Pipeline(FakeIndexableWorkSource(make_inmem_dataset("tiny", [{"x": 1}])))
    pipe.options(shm_min_item_bytes=456, shm_coalesce=True)
    with pytest.raises(TypeError, match="shm_min_size.*Use 'shm_min_item_bytes'"):
        pipe.options(max_workers=99, shm_min_size=123, shm_min_item_bytes=123)
    assert pipe._options.max_workers is None
    assert pipe._options.shm_min_item_bytes == 456
    assert pipe._options.shm_coalesce
