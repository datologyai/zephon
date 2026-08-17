# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Store-level construction and ownership for the decoded Parquet cache."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from zephon._internal.io.catalog import CatalogSet
from zephon._internal.io.formats.parquet import ParquetShardOpener
from zephon._internal.io.formats.parquet_cache.cache import ParquetRGCache
from zephon._internal.io.formats.parquet_cache.index import ParquetRGIndex
from zephon._internal.io.formats.parquet_cache.session import (
    ParquetRGSessionError,
    ParquetRGSessionInUseError,
)
from zephon._internal.io.resolvers.cache.layout import PARQUET_RG_CACHE_SUBDIR
from zephon._internal.utils.disk import check_cache_disk_budgets
from zephon.io.options import StoreOptions

logger = logging.getLogger(__name__)

_legacy_environment_warned = False


@dataclass
class ParquetCacheRuntime:
    """Resources shared by every Parquet reader in one dataset store.

    The runtime owns the optional node-shared decoded cache and supplies the
    single ``ParquetShardOpener`` that binds it to all Parquet shards. Closing
    the store closes this runtime and releases the cache session.
    """

    opener: ParquetShardOpener
    cache: ParquetRGCache | None

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()


def _resolved_cache_root(options: StoreOptions) -> tuple[Path, bool]:
    configured = options.parquet_rg_cache.root
    if configured is None:
        raw_root = Path(options.cache.root).expanduser().resolve()
        return raw_root / PARQUET_RG_CACHE_SUBDIR, False
    decoded_root = Path(configured).expanduser().resolve()
    if options.cache.enabled:
        raw_root = Path(options.cache.root).expanduser().resolve()
        if (
            decoded_root == raw_root
            or decoded_root in raw_root.parents
            or raw_root in decoded_root.parents
        ):
            raise ValueError(
                "Custom parquet_rg_cache.root must not equal, contain, or be contained "
                + f"by the raw cache root ({raw_root}); omit it to use the reserved "
                + f"{PARQUET_RG_CACHE_SUBDIR} child"
            )
    return decoded_root, True


def validate_parquet_cache_disk_space(options: StoreOptions) -> None:
    """Validate the combined raw and decoded cache budgets for a Parquet store."""
    budgets: list[tuple[Path, int]] = []
    raw_limit = options.cache.limit_bytes if options.cache.enabled else None
    if raw_limit is not None:
        budgets.append((Path(options.cache.root), raw_limit))

    decoded = options.resolved_parquet_rg_cache()
    if decoded.enabled is True:
        decoded_root, _is_custom = _resolved_cache_root(options)
        assert decoded.limit_bytes is not None
        budgets.append((decoded_root, decoded.limit_bytes))

    check_cache_disk_budgets(budgets)


def build_parquet_cache_runtime(
    catalog_set: CatalogSet | None,
    options: StoreOptions,
) -> ParquetCacheRuntime:
    """Build the cache and shared opener used by one multi-dataset store."""
    global _legacy_environment_warned

    decoded = options.resolved_parquet_rg_cache()
    if decoded.enabled is not True or catalog_set is None:
        return ParquetCacheRuntime(
            opener=ParquetShardOpener(decoded_cache=None, index=None),
            cache=None,
        )

    index = ParquetRGIndex(catalog_set)
    if index.num_row_groups == 0:
        return ParquetCacheRuntime(
            opener=ParquetShardOpener(decoded_cache=None, index=None),
            cache=None,
        )

    assert decoded.limit_bytes is not None
    if "ZEPHON_PARQUET_RG_CACHE_BYTES" in os.environ and not _legacy_environment_warned:
        _legacy_environment_warned = True
        logger.warning(
            "ZEPHON_PARQUET_RG_CACHE_BYTES is obsolete and ignored; configure "
            "parquet_rg_cache.limit_bytes for the node-wide on-disk cache."
        )

    root, is_custom = _resolved_cache_root(options)
    validate_parquet_cache_disk_space(options)
    if is_custom:
        logger.warning(
            "Using custom Parquet decoded-RG cache root %s. Changing this path "
            "later leaves the old marker-owned cache for explicit operator cleanup.",
            root,
        )
    try:
        cache = ParquetRGCache(
            root=root,
            index=index,
            limit_bytes=decoded.limit_bytes,
            min_free_bytes=decoded.min_free_bytes,
        )
    except ParquetRGSessionInUseError:
        raise
    except ParquetRGSessionError:
        if is_custom:
            raise
        logger.warning(
            "Decoded Parquet RG cache is unavailable at %s; using direct decode.",
            root,
            exc_info=True,
        )
        cache = None
        index = None
    except Exception:
        logger.warning(
            "Decoded Parquet RG cache initialization failed at %s; using direct decode.",
            root,
            exc_info=True,
        )
        cache = None
        index = None

    return ParquetCacheRuntime(
        opener=ParquetShardOpener(decoded_cache=cache, index=index),
        cache=cache,
    )


__all__ = [
    "ParquetCacheRuntime",
    "build_parquet_cache_runtime",
    "validate_parquet_cache_disk_space",
]
