# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Locators from normalized metadata, independently of native file formats."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from zephon._internal.io.formats.locators import build_file_locators, shard_length
from zephon._internal.io.types import LocalShardFile, LocalShardRef
from zephon.io.dataset import Dataset


def _dataset(tmp_path: Path, shards: Any) -> Dataset:
    return Dataset(
        name="ds",
        path=str(tmp_path),
        backend={"kind": "custom", "path": str(tmp_path), "shards": shards},
    )


@pytest.mark.parametrize(
    "bad_shards",
    [
        {0: {}},
        {0: {"raw": {"bytes": 1}}},
        {0: {"raw": {"basename": "part"}}},
        {0: {"raw": {"basename": "part", "bytes": object()}}},
        {0: {"raw": {"basename": "part", "bytes": -1}}},
        {0: {"raw": {"basename": "part", "bytes": True}}},
        {0: {"raw": {"basename": "part", "bytes": 1 << 63}}},
        {0: {"raw": {"basename": "part", "bytes": "bad"}}},
    ],
)
def test_rejects_bad_metadata(tmp_path: Path, bad_shards: Any) -> None:
    with pytest.raises(ValueError):
        build_file_locators(_dataset(tmp_path, bad_shards), format_kind="custom")


def test_preserves_plain_file_metadata(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        {
            4: {
                "raw": {
                    "basename": "part",
                    "bytes": np.int64(100),
                    "hashes": {"sha256": "abc"},
                },
                "extra": {"length": 3, "row_groups": [{"num_rows": 3}]},
            }
        },
    )
    locator = build_file_locators(dataset, format_kind="custom")[4]
    assert locator.dataset == "ds"
    assert locator.shard_id == 4
    assert locator.format == "custom"
    assert locator.root == str(tmp_path)
    assert locator.raw.basename == "part"
    assert locator.raw.bytes == 100
    assert locator.raw.hashes == {"sha256": "abc"}
    assert locator.extra == {"length": 3, "row_groups": [{"num_rows": 3}]}
    assert locator.zip is None
    assert locator.compression is None


def test_preserves_compressed_file_and_explicit_root(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        {
            "2": {
                "raw": {"basename": "part.raw", "bytes": "100"},
                "zip": {
                    "basename": "part.zstd",
                    "bytes": 40,
                    "hashes": {"sha256": "abc"},
                },
                "compression": "zstd",
                "extra": {"source": "test"},
            }
        },
    )
    locator = build_file_locators(
        dataset, format_kind="custom", root="s3://bucket/data"
    )[2]
    assert locator.root == "s3://bucket/data"
    assert locator.raw.bytes == 100
    assert locator.zip is not None
    assert locator.zip.basename == "part.zstd"
    assert locator.zip.bytes == 40
    assert locator.zip.hashes == {"sha256": "abc"}
    assert locator.compression == "zstd"
    assert locator.extra == {"source": "test"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("4", 4), (4, 4), ("bad", None), (float("inf"), None)],
)
def test_optional_length_hint(tmp_path: Path, value: Any, expected: int | None) -> None:
    ref = LocalShardRef(
        raw=LocalShardFile(path=tmp_path / "part", bytes=1), extra={"length": value}
    )
    assert shard_length(ref) == expected
