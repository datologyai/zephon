import pytest

from zephon.io.options import CacheOptions, StoreOptions


def test_cache_options_from_any_and_merge(tmp_path) -> None:
    a = CacheOptions()
    assert a.enabled is False

    b = CacheOptions.from_any({"enabled": True, "root": tmp_path, "download_retry": 3})
    assert b.enabled is True
    assert str(b.root) == str(tmp_path)
    assert b.download_retry == 3

    # None/False
    assert CacheOptions.from_any(None).enabled is False
    assert CacheOptions.from_any(False).enabled is False

    # Instance passthrough
    assert CacheOptions.from_any(b) is b

    merged = a.merge(b)
    assert merged.enabled is True
    assert str(merged.root) == str(tmp_path)

    with pytest.raises(TypeError):
        _ = CacheOptions.from_any(123)


def test_store_options_from_any_and_merge(tmp_path) -> None:
    s0 = StoreOptions()
    assert s0.cache.enabled is False

    s1 = StoreOptions.from_any({"cache": {"enabled": True, "root": str(tmp_path)}})
    assert s1.cache.enabled is True
    assert str(s1.cache.root) == str(tmp_path)

    s2 = s0.merge(s1)
    assert s2.cache.enabled is True

    with pytest.raises(TypeError):
        _ = StoreOptions.from_any(3.14)
