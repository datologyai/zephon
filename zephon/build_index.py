# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Build an ``index.json`` for a dataset directory (fast shard discovery).

CLI: ``python -m zephon.build_index <format> <dataset_dir>``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from zephon._internal.io.index.index_builder import create_index as _create_index

# A format's index module registers its builder on import but pulls an optional
# dep (pyarrow / vortex), so import it on demand to keep this module dep-free.
_FORMAT_INDEX_MODULES = {
    "jsonl": "zephon._internal.io.index.jsonl_index",
    "parquet": "zephon._internal.io.index.parquet_index",
    "vortex": "zephon._internal.io.index.vortex_index",
}


def build_index(
    format_name: str,
    dataset_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    progress: bool = True,
) -> Path:
    """Build an ``index.json`` for a JSONL, Parquet, or Vortex dataset.

    Writes to ``dataset_dir/index.json`` unless *output_path* is given, printing
    per-file progress when *progress* is true. Returns the written index path.

    JSONL indexing reads each shard once, including compressed shards. Rebuild
    the index after adding, removing, or changing shards. The index speeds up
    dataset discovery; it does not contain offsets for seeking within a shard.
    """
    module = _FORMAT_INDEX_MODULES.get(format_name)
    if module is not None:
        importlib.import_module(module)
    return _create_index(
        format_name, dataset_dir, output_path=output_path, progress=progress
    )


def _cli() -> None:
    if len(sys.argv) != 3:
        formats = ", ".join(sorted(_FORMAT_INDEX_MODULES))
        print("Usage: python -m zephon.build_index <format> <dataset_dir>")
        print()
        print("Creates index.json for fast dataset discovery.")
        print(f"Indexable formats: {formats}")
        sys.exit(1)
    try:
        build_index(sys.argv[1], sys.argv[2])
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli()


__all__ = ["build_index"]
