"""Tests for runtime option validation."""

from dataclasses import fields, replace
from inspect import signature

import pytest

from zephon.options import RuntimeOptions


def test_removed_shm_options_are_unexpected_keywords() -> None:
    for name in ("shm_min_size", "coalesce_tensors"):
        with pytest.raises(TypeError, match=f"unexpected keyword argument '{name}'"):
            RuntimeOptions(**{name: None})


def test_shm_options_store_only_canonical_fields() -> None:
    options = RuntimeOptions(shm_min_item_bytes=123, shm_coalesce=False)
    assert options.shm_min_item_bytes == 123
    assert options.shm_coalesce is False
    assert replace(options) == options
    removed = {"shm_min_size", "coalesce_tensors"}
    assert removed.isdisjoint(f.name for f in fields(options))
    assert removed.isdisjoint(signature(RuntimeOptions).parameters)


@pytest.mark.parametrize("value", [None, False, True])
def test_gc_freeze_is_a_runtime_option(value: bool | None) -> None:
    options = RuntimeOptions(gc_freeze=value)
    assert options.gc_freeze is value
    assert replace(options).gc_freeze is value
