# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Configuration helpers for IO store construction."""

import warnings
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any

_SIZE_SUFFIXES = {
    "tb": 1024**4,
    "gb": 1024**3,
    "mb": 1024**2,
    "kb": 1024**1,
    "b": 1024**0,
}
_PARQUET_RG_CACHE_MIN_BYTES = 4 * 1024**3
_OPTION_DEFAULT = "zephon_default"


class _NotSupplied(Enum):
    TOKEN = 0


_NOT_SUPPLIED = _NotSupplied.TOKEN


def _option(default: object) -> Any:
    return field(default=_NOT_SUPPLIED, metadata={_OPTION_DEFAULT: default})


def _normalize_options(options: Any) -> None:
    provided = set()
    for option in fields(options):
        if _OPTION_DEFAULT not in option.metadata:
            continue
        value = getattr(options, option.name)
        if value is not _NOT_SUPPLIED:
            provided.add(option.name)
        if value is _NOT_SUPPLIED or value is None:
            setattr(options, option.name, option.metadata[_OPTION_DEFAULT])
    options._provided_fields = frozenset(provided)


def _merged_options(base: Any, override: Any) -> dict[str, Any]:
    merged = {}
    for option in fields(base):
        if _OPTION_DEFAULT not in option.metadata:
            continue
        if option.name in override._provided_fields:
            merged[option.name] = getattr(override, option.name)
        elif option.name in base._provided_fields:
            merged[option.name] = getattr(base, option.name)
    return merged


def _default_options(option_type: Any) -> Any:
    values = {
        option.name: None
        for option in fields(option_type)
        if _OPTION_DEFAULT in option.metadata
    }
    return option_type(**values)


def parse_size_bytes(value: str | int | None) -> int | None:
    """Parse a human-readable size like ``"750gb"`` into a byte count.

    Accepts ``None`` (returns ``None``), an int (returned as-is), or a string
    with an optional suffix in {tb, gb, mb, kb, b}. Empty or whitespace-only
    strings return ``None``. Raises ``ValueError`` on unparseable input.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Cannot parse size from {value!r}")
    s = value.strip().lower()
    if not s:
        return None
    for suffix, multiplier in _SIZE_SUFFIXES.items():
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * multiplier)
    return int(s)


@dataclass
class CacheOptions:
    """User-configurable knobs that control cache behaviour."""

    enabled: bool = _option(False)
    root: str | Path = _option(Path("~/.cache/zephon").expanduser())
    # Bound on the on-disk shard cache only, not in-memory cache.
    limit_bytes: int | None = _option(None)
    # Deprecated alias migrated by StoreOptions to parquet_rg_cache.limit_bytes.
    rg_cache_bytes: int | None = _option(None)
    keep_zip: bool = _option(False)
    validate_hash: str | None = _option(None)
    download_retry: int = _option(12)
    download_timeout: float = _option(180.0)
    open_retry_attempts: int = _option(5)
    open_retry_initial_backoff: float = _option(0.1)
    open_retry_max_backoff: float = _option(2.0)
    min_slack_bytes: int = _option(512 * 1024)  # 512 KiB minimum slack
    max_slack_bytes: int = _option(64 * 1024 * 1024)  # 64 MiB maximum slack
    _provided_fields: frozenset[str] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _normalize_options(self)

    @classmethod
    def from_any(cls, obj: Any) -> "CacheOptions":
        if obj is None or obj is False:
            return _default_options(cls)
        if isinstance(obj, CacheOptions):
            return obj
        if isinstance(obj, dict):
            data = dict(obj)
            if "limit_bytes" in data and isinstance(data["limit_bytes"], str):
                data["limit_bytes"] = parse_size_bytes(data["limit_bytes"])
            if "rg_cache_bytes" in data and isinstance(data["rg_cache_bytes"], str):
                data["rg_cache_bytes"] = parse_size_bytes(data["rg_cache_bytes"])
            return cls(**data)
        raise TypeError(f"Cannot interpret cache options from {obj!r}")

    def merge(self, other: "CacheOptions") -> "CacheOptions":
        """Overlay fields supplied by ``other`` onto ``self``."""
        return CacheOptions(**_merged_options(self, other))


@dataclass
class ParquetRGCacheOptions:
    """Node-shared decoded Parquet row-group cache configuration.

    ``enabled=None`` enables the cache with the shard cache or an explicit
    ``root``. An omitted root uses the shard cache's reserved
    ``.parquet-rg-cache`` child. An omitted limit uses the deprecated shard
    option when present, otherwise the larger of 4 GiB and 10% of the shard
    cache limit. These defaults resolve when the store is built so option merges
    retain the distinction between omitted and explicit values. Explicit
    ``None`` resets a previously supplied field to its automatic default.
    """

    enabled: bool | None = _option(None)
    root: str | Path | None = _option(None)
    limit_bytes: int | None = _option(None)
    min_free_bytes: int = _option(1 * 1024**3)
    _provided_fields: frozenset[str] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _normalize_options(self)
        if (
            self.enabled is not False
            and self.limit_bytes is not None
            and self.limit_bytes <= 0
        ):
            raise ValueError("parquet_rg_cache.limit_bytes must be positive")
        if self.min_free_bytes < 0:
            raise ValueError("parquet_rg_cache.min_free_bytes cannot be negative")

    @classmethod
    def from_any(cls, obj: Any) -> "ParquetRGCacheOptions":
        """Normalize a decoded row-group cache configuration."""
        if obj is None:
            return _default_options(cls)
        if obj is False:
            return cls(enabled=False)
        if obj is True:
            return cls(enabled=True)
        if isinstance(obj, ParquetRGCacheOptions):
            return obj
        if isinstance(obj, dict):
            data = dict(obj)
            for field_name in ("limit_bytes", "min_free_bytes"):
                value = data.get(field_name)
                if isinstance(value, str):
                    parsed = parse_size_bytes(value)
                    if parsed is None:
                        raise ValueError(
                            f"parquet_rg_cache.{field_name} must not be empty"
                        )
                    data[field_name] = parsed
            return cls(**data)
        raise TypeError(f"Cannot interpret Parquet RG cache options from {obj!r}")

    def merge(self, other: "ParquetRGCacheOptions") -> "ParquetRGCacheOptions":
        """Overlay fields supplied by ``other`` onto ``self``."""
        return ParquetRGCacheOptions(**_merged_options(self, other))


@dataclass
class VortexOptions:
    """Process-local Vortex caches shared by the shards in one store.

    ``segment_cache_bytes`` budgets retained encoded segment bytes, not total
    memory: decoded outputs, metadata and in-flight reads are additional.
    Each worker/store owns its budget; it is not shared across processes.
    ``metadata_cache_entries`` bounds the number of parsed footers retained.
    Set either limit to zero to disable that cache.
    """

    segment_cache_bytes: int = _option(64 * 1024**2)
    metadata_cache_entries: int = _option(256)
    _provided_fields: frozenset[str] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _normalize_options(self)
        for name in ("segment_cache_bytes", "metadata_cache_entries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"vortex.{name} must be a non-negative integer")
        if self.segment_cache_bytes > 2**64 - 1:
            raise ValueError("vortex.segment_cache_bytes exceeds Vortex's byte limit")

    @classmethod
    def from_any(cls, obj: Any) -> "VortexOptions":
        """Normalize Vortex options, accepting byte sizes such as ``'64mb'``."""
        if obj is None:
            return _default_options(cls)
        if isinstance(obj, VortexOptions):
            return obj
        if isinstance(obj, dict):
            data = dict(obj)
            if isinstance(data.get("segment_cache_bytes"), str):
                parsed = parse_size_bytes(data["segment_cache_bytes"])
                if parsed is None:
                    raise ValueError("vortex.segment_cache_bytes must not be empty")
                data["segment_cache_bytes"] = parsed
            return cls(**data)
        raise TypeError(f"Cannot interpret Vortex options from {obj!r}")

    def merge(self, other: "VortexOptions") -> "VortexOptions":
        """Overlay fields supplied by ``other`` onto ``self``."""
        return VortexOptions(**_merged_options(self, other))


@dataclass
class StoreOptions:
    """Top-level IO store options passed to FetchOp."""

    cache: CacheOptions = field(default_factory=CacheOptions)
    parquet_rg_cache: ParquetRGCacheOptions = field(
        default_factory=ParquetRGCacheOptions
    )
    vortex: VortexOptions = field(default_factory=VortexOptions)

    def __post_init__(self) -> None:
        legacy_limit = self.cache.rg_cache_bytes
        if legacy_limit is not None:
            warnings.warn(
                "cache.rg_cache_bytes is deprecated; use "
                + "parquet_rg_cache.limit_bytes",
                DeprecationWarning,
                stacklevel=2,
            )

    def resolved_parquet_rg_cache(self) -> ParquetRGCacheOptions:
        """Resolve decoded-cache defaults against the shard-cache options."""
        configured = self.parquet_rg_cache
        if (
            configured.enabled is True
            and not self.cache.enabled
            and configured.root is None
        ):
            raise ValueError(
                "parquet_rg_cache.enabled=True requires parquet_rg_cache.root "
                + "when cache.enabled=False"
            )
        legacy_limit = self.cache.rg_cache_bytes
        limit = configured.limit_bytes
        enabled = (
            configured.enabled
            if configured.enabled is not None
            else self.cache.enabled or configured.root is not None
        )
        if limit is None:
            if legacy_limit is not None:
                enabled = enabled and legacy_limit > 0
                limit = (
                    legacy_limit if legacy_limit > 0 else _PARQUET_RG_CACHE_MIN_BYTES
                )
            else:
                shard_limit = self.cache.limit_bytes if self.cache.enabled else None
                limit = max(
                    _PARQUET_RG_CACHE_MIN_BYTES,
                    shard_limit // 10 if shard_limit is not None else 0,
                )
        return ParquetRGCacheOptions(
            enabled=enabled,
            root=configured.root,
            limit_bytes=limit,
            min_free_bytes=configured.min_free_bytes,
        )

    @classmethod
    def from_any(cls, obj: Any) -> "StoreOptions":
        if obj is None:
            return cls(
                cache=CacheOptions.from_any(None),
                parquet_rg_cache=ParquetRGCacheOptions.from_any(None),
                vortex=VortexOptions.from_any(None),
            )
        if isinstance(obj, StoreOptions):
            return obj
        if isinstance(obj, dict):
            cache = (
                CacheOptions.from_any(obj["cache"])
                if "cache" in obj
                else CacheOptions()
            )
            parquet_rg_cache = (
                ParquetRGCacheOptions.from_any(obj["parquet_rg_cache"])
                if "parquet_rg_cache" in obj
                else ParquetRGCacheOptions()
            )
            vortex = (
                VortexOptions.from_any(obj["vortex"])
                if "vortex" in obj
                else VortexOptions()
            )
            return cls(cache=cache, parquet_rg_cache=parquet_rg_cache, vortex=vortex)
        raise TypeError(f"Cannot interpret store options from {obj!r}")

    def merge(self, other: "StoreOptions") -> "StoreOptions":
        """Return a new options object with ``other`` overriding ``self``."""
        return StoreOptions(
            cache=self.cache.merge(other.cache),
            parquet_rg_cache=self.parquet_rg_cache.merge(other.parquet_rg_cache),
            vortex=self.vortex.merge(other.vortex),
        )


__all__ = [
    "CacheOptions",
    "ParquetRGCacheOptions",
    "StoreOptions",
    "VortexOptions",
    "parse_size_bytes",
]
