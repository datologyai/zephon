# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""JSONL record counts and decompressed size from backend-owned streams."""

import bz2
import gzip
import lzma

import pytest

from tests.helpers.storage import MemoryStorage
from zephon._internal.io.index.jsonl_index import JsonlIndexBuilder
from zephon._internal.utils.compression import require_zstd


@pytest.mark.parametrize("suffix", ["gz", "bz2", "xz", "zst"])
def test_compressed_jsonl_records_raw_size(suffix: str) -> None:
    raw = '{"text":"café"}\n\nnull'.encode("utf-8")
    compress = {
        "gz": gzip.compress,
        "bz2": bz2.compress,
        "xz": lzma.compress,
        "zst": lambda data: require_zstd().compress(data),
    }[suffix]
    name = f"data.jsonl.{suffix}"
    data = compress(raw)
    storage = MemoryStorage("custom://data", {name: data})
    info = JsonlIndexBuilder(storage).extract_shard_info(
        f"{storage.root}/{name}", len(data)
    )
    assert info.basename == name
    assert info.bytes == len(data)
    assert info.num_rows == 2
    assert info.extra == {"raw_bytes": len(raw)}
