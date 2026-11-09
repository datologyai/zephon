# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Common shard index validation and projections, without format dependencies."""

from typing import Any

import numpy as np
import pytest

from zephon._internal.io.index.shard_index import (
    shard_index_counts,
    shard_index_metadata,
    validate_shard_index,
)


def _index() -> Any:
    return {
        "format_version": 1,
        "shards": [
            {
                "basename": "a.custom",
                "bytes": 10,
                "num_rows": 3,
                "hashes": {"sha256": "abc"},
                "extra": {"source": "test"},
            },
            {"basename": "b.custom", "bytes": 20, "num_rows": 0},
        ],
    }


@pytest.mark.parametrize(
    "update",
    [
        {"num_rows": None},
        {"num_rows": -1},
        {"num_rows": True},
        {"num_rows": 1 << 63},
        {"num_rows": "3"},
        {"bytes": -1},
        {"bytes": False},
        {"bytes": 1 << 63},
        {"basename": None},
        {"hashes": []},
        {"hashes": {"sha256": 123}},
        {"extra": None},
    ],
)
def test_rejects_invalid_shard_fields(update: dict[str, Any]) -> None:
    index = _index()
    index["shards"][1].update(update)
    with pytest.raises(ValueError, match="index.*shard 1"):
        validate_shard_index(index)


@pytest.mark.parametrize(
    "update",
    [
        {"format_version": 2},
        {"format_version": True},
        {"shards": []},
        {"shards": None},
        {"shards": ["bad"]},
    ],
)
def test_rejects_invalid_index_header_or_container(update: dict[str, Any]) -> None:
    index = _index()
    index.update(update)
    with pytest.raises(ValueError, match="index"):
        validate_shard_index(index)


def test_counts_and_metadata_agree_and_preserve_extras() -> None:
    index = _index()
    validate_shard_index(index)
    ids, values = shard_index_counts(index)
    counts, metadata = shard_index_metadata(index)
    assert ids.dtype == values.dtype == np.dtype("int64")
    assert ids.tolist() == [0, 1]
    assert values.tolist() == list(counts.values()) == [3, 0]
    assert metadata[0] == {
        "raw": {"basename": "a.custom", "bytes": 10, "hashes": {"sha256": "abc"}},
        "extra": {"source": "test", "length": 3},
    }
    assert metadata[1]["extra"] == {"length": 0}
    assert index["shards"][0]["extra"] == {"source": "test"}
