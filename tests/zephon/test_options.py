"""Tests for runtime option validation."""

from dataclasses import fields, replace

import pytest

from zephon.options import RuntimeOptions


def test_removed_shm_options_report_replacements() -> None:
    for old, new, value in (
        ("shm_min_size", "shm_min_item_bytes", 123),
        ("coalesce_tensors", "shm_coalesce", False),
    ):
        for supplied in (value, None):
            with pytest.raises(
                TypeError,
                match=f"'{old}' is no longer supported.*Use '{new}' instead",
            ):
                RuntimeOptions(**{old: supplied, new: value})


def test_shm_options_store_only_canonical_fields() -> None:
    options = RuntimeOptions(shm_min_item_bytes=123, shm_coalesce=False)
    assert options.shm_min_item_bytes == 123
    assert options.shm_coalesce is False
    assert replace(options).shm_min_item_bytes == 123
    assert replace(options).shm_coalesce is False
    assert {"shm_min_size", "coalesce_tensors"}.isdisjoint(
        f.name for f in fields(options)
    )
