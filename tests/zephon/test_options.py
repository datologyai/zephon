"""Tests for runtime option normalization."""

import pytest

from zephon.options import RuntimeOptions


def test_shm_aliases_warn_and_reject_conflicting_canonical_values() -> None:
    with pytest.warns(DeprecationWarning, match="shm_min_item_bytes"):
        options = RuntimeOptions(shm_min_size=123)
    assert options.shm_min_item_bytes == 123
    assert options.shm_min_size is None
    with pytest.warns(DeprecationWarning, match="shm_coalesce"):
        options = RuntimeOptions(coalesce_tensors=False)
    assert options.shm_coalesce is False
    assert options.coalesce_tensors is None
    # An explicitly supplied default is still a conflicting value.
    with pytest.raises(ValueError, match="Conflicting pipeline options"):
        RuntimeOptions(shm_min_size=123, shm_min_item_bytes=4096)
    with pytest.raises(ValueError, match="Conflicting pipeline options"):
        RuntimeOptions(coalesce_tensors=False, shm_coalesce=True)
