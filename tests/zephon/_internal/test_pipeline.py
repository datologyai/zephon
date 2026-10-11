"""Tests for pipeline option normalization."""

import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon._internal.pipeline import Pipeline


def test_payload_memory_option_aliases() -> None:
    pipe = Pipeline(FakeIndexableWorkSource(make_inmem_dataset("tiny", [{"x": 1}])))
    with pytest.deprecated_call():
        pipe.options(shm_min_size=123, coalesce_tensors=False)
    assert pipe._options.shm_min_item_bytes == 123
    assert not pipe._options.shm_coalesce
    pipe.options(shm_min_item_bytes=456, shm_coalesce=True)
    assert pipe._options.shm_min_item_bytes == 456
    assert pipe._options.shm_coalesce
    with pytest.raises(ValueError, match="Conflicting pipeline options"):
        pipe.options(shm_min_item_bytes=456, shm_min_size=123)
    with pytest.raises(ValueError, match="shm_coalesce cannot be None"):
        pipe.options(shm_coalesce=None)
