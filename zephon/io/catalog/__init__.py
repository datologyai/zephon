# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Node-local, columnar, mmap-shared per-shard metadata catalog.

One dataset -> one content-addressed file, ``mmap``ped read-only and shared
across every rank/worker on a machine; locators are synthesized on demand from
the columns rather than materialized en masse.
"""

from zephon.io.catalog.builder import BuiltCatalog, DatasetHeader, build_catalog
from zephon.io.catalog.catalog import CatalogSet, ShardCatalog
from zephon.io.catalog.extra_codec import (
    EncodedExtra,
    ExtraCodec,
    get_extra_codec,
    register_extra_codec,
)
from zephon.io.catalog.handle import (
    CATALOG_CACHE_SUBDIR,
    CatalogFingerprintMismatch,
    ShardCatalogHandle,
    attach,
    clear_registry,
    finalize,
    resolve_catalog_dir,
    set_catalog_dir,
)
from zephon.io.catalog.io import SCHEMA_VERSION

__all__ = [
    "CATALOG_CACHE_SUBDIR",
    "SCHEMA_VERSION",
    "BuiltCatalog",
    "CatalogFingerprintMismatch",
    "CatalogSet",
    "DatasetHeader",
    "EncodedExtra",
    "ExtraCodec",
    "ShardCatalog",
    "ShardCatalogHandle",
    "attach",
    "build_catalog",
    "clear_registry",
    "finalize",
    "get_extra_codec",
    "register_extra_codec",
    "resolve_catalog_dir",
    "set_catalog_dir",
]
