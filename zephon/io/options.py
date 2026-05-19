# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Configuration helpers for IO store construction."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SIZE_SUFFIXES = {
    "tb": 1024**4,
    "gb": 1024**3,
    "mb": 1024**2,
    "kb": 1024**1,
    "b": 1024**0,
}


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

    enabled: bool = False
    root: str | Path = Path("~/.cache/zephon").expanduser()
    # Bound on the on-disk shard cache only, not in-memory cache.
    limit_bytes: int | None = None
    keep_zip: bool = False
    validate_hash: str | None = None
    download_retry: int = 12
    download_timeout: float = 180.0
    open_retry_attempts: int = 5
    open_retry_initial_backoff: float = 0.1
    open_retry_max_backoff: float = 2.0
    min_slack_bytes: int = 512 * 1024  # 512 KiB minimum slack
    max_slack_bytes: int = 64 * 1024 * 1024  # 64 MiB maximum slack

    @classmethod
    def from_any(cls, obj: Any) -> "CacheOptions":
        if obj is None or obj is False:
            return cls()
        if isinstance(obj, CacheOptions):
            return obj
        if isinstance(obj, dict):
            data = dict(obj)
            if "limit_bytes" in data and isinstance(data["limit_bytes"], str):
                data["limit_bytes"] = parse_size_bytes(data["limit_bytes"])
            return cls(**data)
        raise TypeError(f"Cannot interpret cache options from {obj!r}")

    def merge(self, other: "CacheOptions") -> "CacheOptions":
        """Return a new options object with ``other`` overriding ``self``."""
        merged = {**self.__dict__, **other.__dict__}
        return CacheOptions(**merged)


@dataclass
class StoreOptions:
    """Top-level IO store options passed to FetchOp."""

    cache: CacheOptions = field(default_factory=CacheOptions)

    @classmethod
    def from_any(cls, obj: Any) -> "StoreOptions":
        if obj is None:
            return cls()
        if isinstance(obj, StoreOptions):
            return obj
        if isinstance(obj, dict):
            cache_cfg = obj.get("cache")
            return cls(cache=CacheOptions.from_any(cache_cfg))
        raise TypeError(f"Cannot interpret store options from {obj!r}")

    def merge(self, other: "StoreOptions") -> "StoreOptions":
        """Return a new options object with ``other`` overriding ``self``."""
        return StoreOptions(cache=self.cache.merge(other.cache))


__all__ = ["CacheOptions", "StoreOptions", "parse_size_bytes"]
