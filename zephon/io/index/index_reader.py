# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shared utility for finding and loading dataset index JSON files.

This module owns the list of candidate filenames (e.g., index.json, _index.json)
and provides a single function to find and load the first existing index file
in a dataset directory.
"""

from __future__ import annotations

import json
from typing import Any

from zephon.io.storage.base import StorageBackend

# TODO @danielzayas: add _index.json to the list of index files
_INDEX_FILENAMES: list[str] = ["index.json", "_index.json"]


def find_and_load_index(dir_path: str, storage: StorageBackend) -> Any | None:
    """Find and load the first existing index file in a dataset directory.

    Tries each candidate in _INDEX_FILENAMES in order. Returns parsed JSON data
    for the first existing file, or None if none exist.
    JSON parse errors propagate to the caller.

    Args:
        dir_path: Dataset directory path (local or remote, e.g. s3://bucket/dir).
        storage: Storage backend for file access.

    Returns:
        Parsed index data for the first existing file, or None.
    """
    base = dir_path.rstrip("/") or dir_path
    for filename in _INDEX_FILENAMES:
        path = f"{base}/{filename}"
        if storage.exists(path):
            with storage.open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


__all__ = ["find_and_load_index"]
