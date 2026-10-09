import pytest

from zephon.io.options import (
    CacheOptions,
    ParquetRGCacheOptions,
    StoreOptions,
    VortexOptions,
)

_CACHE_FIELDS = (
    "enabled",
    "root",
    "limit_bytes",
    "rg_cache_bytes",
    "keep_zip",
    "validate_hash",
    "download_retry",
    "download_timeout",
    "open_retry_attempts",
    "open_retry_initial_backoff",
    "open_retry_max_backoff",
    "min_slack_bytes",
    "max_slack_bytes",
)
_PARQUET_RG_CACHE_FIELDS = (
    "enabled",
    "root",
    "limit_bytes",
    "min_free_bytes",
)


def test_vortex_options_merge_disable_and_reset() -> None:
    base = StoreOptions.from_any(
        {"vortex": {"segment_cache_bytes": "8mb", "metadata_cache_entries": 12}}
    )
    assert base.vortex.segment_cache_bytes == 8 * 1024**2
    assert base.merge(StoreOptions()).vortex == base.vortex
    disabled = base.merge(StoreOptions.from_any({"vortex": {"segment_cache_bytes": 0}}))
    assert disabled.vortex.segment_cache_bytes == 0
    assert disabled.vortex.metadata_cache_entries == 12
    reset = base.merge(StoreOptions.from_any({"vortex": None}))
    assert reset.vortex == VortexOptions()
    assert base.merge(StoreOptions.from_any(None)).vortex == VortexOptions()
    assert VortexOptions.from_any(base.vortex) is base.vortex


@pytest.mark.parametrize(
    "options",
    [
        {"segment_cache_bytes": -1},
        {"segment_cache_bytes": 1.5},
        {"segment_cache_bytes": True},
        {"segment_cache_bytes": ""},
        {"segment_cache_bytes": 2**64},
        {"metadata_cache_entries": -1},
        {"metadata_cache_entries": 1.5},
    ],
)
def test_vortex_options_reject_invalid_limits(options: dict) -> None:
    with pytest.raises(ValueError):
        StoreOptions.from_any({"vortex": options})


def _configured_cache(
    tmp_path,
    *,
    rg_cache_bytes: int | None = 20,
) -> CacheOptions:
    return CacheOptions(
        enabled=True,
        root=tmp_path,
        limit_bytes=10,
        rg_cache_bytes=rg_cache_bytes,
        keep_zip=True,
        validate_hash="sha256",
        download_retry=3,
        download_timeout=4.0,
        open_retry_attempts=6,
        open_retry_initial_backoff=0.2,
        open_retry_max_backoff=3.0,
        min_slack_bytes=30,
        max_slack_bytes=40,
    )


def _configured_parquet_rg_cache(tmp_path) -> ParquetRGCacheOptions:
    return ParquetRGCacheOptions(
        enabled=True,
        root=tmp_path / "decoded",
        limit_bytes=20 * 1024**3,
        min_free_bytes=2 * 1024**3,
    )


def test_cache_options_from_any_and_merge(tmp_path) -> None:
    a = CacheOptions()
    assert a.enabled is False

    b = CacheOptions.from_any({"enabled": True, "root": tmp_path, "download_retry": 3})
    assert b.enabled is True
    assert str(b.root) == str(tmp_path)
    assert b.download_retry == 3

    assert CacheOptions.from_any(None).enabled is False
    assert CacheOptions.from_any(False).enabled is False

    assert CacheOptions.from_any(b) is b

    merged = a.merge(b)
    assert merged.enabled is True
    assert str(merged.root) == str(tmp_path)

    with pytest.raises(TypeError):
        _ = CacheOptions.from_any(123)
    legacy = CacheOptions.from_any({"rg_cache_bytes": "4gb"})
    assert legacy.rg_cache_bytes == 4 * 1024**3


def test_store_options_from_any_and_merge(tmp_path) -> None:
    s0 = StoreOptions()
    assert s0.parquet_rg_cache.enabled is None
    assert s0.parquet_rg_cache.limit_bytes is None
    resolved0 = s0.resolved_parquet_rg_cache()
    assert resolved0.enabled is False
    assert resolved0.limit_bytes == 4 * 1024**3

    s1 = StoreOptions.from_any(
        {
            "cache": {"enabled": True, "root": str(tmp_path)},
            "parquet_rg_cache": {
                "root": str(tmp_path / "rg"),
                "limit_bytes": "8gb",
                "min_free_bytes": "2gb",
            },
        }
    )
    assert s1.cache.enabled is True
    assert str(s1.cache.root) == str(tmp_path)
    assert str(s1.parquet_rg_cache.root) == str(tmp_path / "rg")
    assert s1.parquet_rg_cache.limit_bytes == 8 * 1024**3
    assert s1.parquet_rg_cache.min_free_bytes == 2 * 1024**3
    assert s1.resolved_parquet_rg_cache().enabled is True

    s2 = s0.merge(s1)
    assert s2.cache.enabled is True
    assert s2.parquet_rg_cache.limit_bytes == 8 * 1024**3

    with pytest.raises(TypeError):
        _ = StoreOptions.from_any(3.14)


def test_store_merge_preserves_explicit_rg_limit_when_later_call_omits_it(
    tmp_path,
) -> None:
    first = StoreOptions.from_any({"parquet_rg_cache": {"limit_bytes": "20gb"}})
    second = StoreOptions.from_any(
        {"cache": {"enabled": True, "root": tmp_path / "cache"}}
    )

    merged = first.merge(second)

    assert merged.parquet_rg_cache.limit_bytes == 20 * 1024**3
    resolved = merged.resolved_parquet_rg_cache()
    assert resolved.enabled is True
    assert resolved.limit_bytes == 20 * 1024**3


def test_parquet_rg_cache_default_tracks_enabled_shard_cache_limit(tmp_path) -> None:
    large = StoreOptions(
        cache=CacheOptions(enabled=True, limit_bytes=100 * 1024**3),
    )
    small = StoreOptions(
        cache=CacheOptions(enabled=True, limit_bytes=10 * 1024**3),
    )
    unlimited = StoreOptions(cache=CacheOptions(enabled=True))
    rg_only = StoreOptions(
        cache=CacheOptions(enabled=False, limit_bytes=100 * 1024**3),
        parquet_rg_cache=ParquetRGCacheOptions(root=tmp_path / "decoded"),
    )

    large_rg = large.resolved_parquet_rg_cache()
    small_rg = small.resolved_parquet_rg_cache()
    unlimited_rg = unlimited.resolved_parquet_rg_cache()
    rg_only_resolved = rg_only.resolved_parquet_rg_cache()
    assert large_rg.enabled is True
    assert large_rg.limit_bytes == 10 * 1024**3
    assert small_rg.limit_bytes == 4 * 1024**3
    assert unlimited_rg.limit_bytes == 4 * 1024**3
    assert rg_only_resolved.enabled is True
    assert rg_only_resolved.limit_bytes == 4 * 1024**3


def test_legacy_rg_cache_limit_migrates_unless_new_limit_is_set() -> None:
    with pytest.warns(DeprecationWarning, match="rg_cache_bytes is deprecated"):
        migrated = StoreOptions.from_any(
            {
                "cache": {
                    "enabled": True,
                    "limit_bytes": "100gb",
                    "rg_cache_bytes": "2gb",
                }
            }
        )
    with pytest.warns(DeprecationWarning, match="rg_cache_bytes is deprecated"):
        overridden = StoreOptions.from_any(
            {
                "cache": {"enabled": True, "rg_cache_bytes": "6gb"},
                "parquet_rg_cache": {"limit_bytes": "8gb"},
            }
        )

    assert migrated.resolved_parquet_rg_cache().limit_bytes == 2 * 1024**3
    assert overridden.resolved_parquet_rg_cache().limit_bytes == 8 * 1024**3


def test_legacy_zero_disables_parquet_rg_cache() -> None:
    with pytest.warns(DeprecationWarning, match="rg_cache_bytes is deprecated"):
        options = StoreOptions(cache=CacheOptions(enabled=True, rg_cache_bytes=0))

    resolved = options.resolved_parquet_rg_cache()
    assert resolved.enabled is False
    assert resolved.limit_bytes == 4 * 1024**3


def test_parquet_rg_cache_options_normalization_and_validation(tmp_path) -> None:
    disabled = ParquetRGCacheOptions.from_any(False)
    assert disabled.enabled is False
    assert ParquetRGCacheOptions.from_any(True).enabled is True
    assert ParquetRGCacheOptions.from_any(disabled) is disabled

    configured = ParquetRGCacheOptions.from_any(
        {
            "root": tmp_path,
            "limit_bytes": "512mb",
            "min_free_bytes": "64mb",
        }
    )
    assert configured.root == tmp_path
    assert configured.limit_bytes == 512 * 1024**2
    assert configured.min_free_bytes == 64 * 1024**2
    assert ParquetRGCacheOptions().enabled is None
    assert ParquetRGCacheOptions().limit_bytes is None

    with pytest.raises(ValueError, match="limit_bytes"):
        ParquetRGCacheOptions(limit_bytes=0)
    with pytest.raises(ValueError, match="min_free_bytes"):
        ParquetRGCacheOptions(min_free_bytes=-1)
    with pytest.raises(ValueError, match="must not be empty"):
        ParquetRGCacheOptions.from_any({"limit_bytes": " "})
    with pytest.raises(TypeError):
        ParquetRGCacheOptions.from_any(123)


def test_explicit_rg_enable_requires_a_root_without_shard_cache() -> None:
    options = StoreOptions(
        cache=CacheOptions(enabled=False),
        parquet_rg_cache=ParquetRGCacheOptions(enabled=True),
    )

    with pytest.raises(
        ValueError,
        match="enabled=True requires parquet_rg_cache.root",
    ):
        options.resolved_parquet_rg_cache()


def test_parquet_rg_cache_merge_distinguishes_omission_from_explicit_none(
    tmp_path,
) -> None:
    configured = _configured_parquet_rg_cache(tmp_path)
    defaults = ParquetRGCacheOptions()

    partial = configured.merge(ParquetRGCacheOptions.from_any({"limit_bytes": "30gb"}))
    assert partial.limit_bytes == 30 * 1024**3
    for name in _PARQUET_RG_CACHE_FIELDS:
        if name != "limit_bytes":
            assert getattr(partial, name) == getattr(configured, name)

    for reset_name in _PARQUET_RG_CACHE_FIELDS:
        reset = configured.merge(ParquetRGCacheOptions.from_any({reset_name: None}))
        assert getattr(reset, reset_name) == getattr(defaults, reset_name)
        for preserved_name in _PARQUET_RG_CACHE_FIELDS:
            if preserved_name != reset_name:
                assert getattr(reset, preserved_name) == getattr(
                    configured,
                    preserved_name,
                )


def test_later_limit_only_store_options_preserve_other_rg_settings(tmp_path) -> None:
    first = StoreOptions.from_any(
        {
            "cache": {"enabled": True, "root": tmp_path / "cache"},
            "parquet_rg_cache": {
                "enabled": True,
                "root": tmp_path / "decoded",
                "min_free_bytes": "2gb",
            },
        }
    )
    second = StoreOptions.from_any({"parquet_rg_cache": {"limit_bytes": "20gb"}})

    merged = first.merge(second)

    assert merged.cache == first.cache
    assert merged.parquet_rg_cache.enabled is True
    assert merged.parquet_rg_cache.root == tmp_path / "decoded"
    assert merged.parquet_rg_cache.limit_bytes == 20 * 1024**3
    assert merged.parquet_rg_cache.min_free_bytes == 2 * 1024**3


def test_cache_merge_distinguishes_omitted_fields_from_explicit_none(
    tmp_path,
) -> None:
    configured = _configured_cache(tmp_path)
    defaults = CacheOptions()

    partial = configured.merge(CacheOptions.from_any({"download_retry": 7}))
    assert partial.download_retry == 7
    for name in _CACHE_FIELDS:
        if name != "download_retry":
            assert getattr(partial, name) == getattr(configured, name)

    for reset_name in _CACHE_FIELDS:
        reset = configured.merge(CacheOptions.from_any({reset_name: None}))
        assert getattr(reset, reset_name) == getattr(defaults, reset_name)
        for preserved_name in _CACHE_FIELDS:
            if preserved_name != reset_name:
                assert getattr(reset, preserved_name) == getattr(
                    configured,
                    preserved_name,
                )


def test_store_merge_treats_missing_cache_as_omitted_and_none_as_reset(
    tmp_path,
) -> None:
    configured = StoreOptions(
        cache=_configured_cache(tmp_path, rg_cache_bytes=None),
    )

    assert configured.merge(StoreOptions.from_any({})).cache == configured.cache
    assert configured.merge(StoreOptions.from_any(None)).cache == CacheOptions()
