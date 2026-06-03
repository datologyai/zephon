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
from zephon.io.catalog.io import SCHEMA_VERSION

__all__ = [
    "SCHEMA_VERSION",
    "BuiltCatalog",
    "CatalogSet",
    "DatasetHeader",
    "EncodedExtra",
    "ExtraCodec",
    "ShardCatalog",
    "build_catalog",
    "get_extra_codec",
    "register_extra_codec",
]
