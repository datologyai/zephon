"""Tests for runtime option normalization."""

from dataclasses import fields, replace

import pytest

from zephon.options import RuntimeOptions


def test_shm_aliases_warn_and_reject_conflicting_canonical_values() -> None:
    with pytest.warns(DeprecationWarning, match="shm_min_item_bytes") as caught:
        options = RuntimeOptions(shm_min_size=123)
    assert caught[0].filename == __file__
    assert options.shm_min_item_bytes == 123
    assert replace(options).shm_min_item_bytes == 123
    assert {"shm_min_size", "coalesce_tensors"}.isdisjoint(
        f.name for f in fields(options)
    )
    with pytest.warns(DeprecationWarning, match="shm_coalesce"):
        options = RuntimeOptions(coalesce_tensors=False)
    assert options.shm_coalesce is False
    # Aliases can override canonical defaults without Optional public fields.
    with pytest.deprecated_call():
        assert (
            RuntimeOptions(shm_min_size=123, shm_min_item_bytes=4096).shm_min_item_bytes
            == 123
        )
    with pytest.deprecated_call():
        assert not RuntimeOptions(
            coalesce_tensors=False, shm_coalesce=True
        ).shm_coalesce
    with pytest.raises(ValueError, match="Conflicting pipeline options"):
        RuntimeOptions(shm_min_size=123, shm_min_item_bytes=456)
    with pytest.raises(ValueError, match="Conflicting pipeline options"):
        RuntimeOptions(coalesce_tensors=True, shm_coalesce=False)
    for name in ("shm_min_item_bytes", "shm_coalesce"):
        with pytest.raises(ValueError, match=f"{name} cannot be None"):
            RuntimeOptions(**{name: None})
