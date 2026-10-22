# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Test helpers that import only Zephon's public API."""

import sys

import pytest

from zephon.io import InMemoryShard
from zephon.io.dataset import Dataset

# The zstd codec Zephon decodes with, for writing compressed fixtures.
try:
    from compression import zstd
except ImportError:
    from backports import zstd

__all__ = [
    "DEEPSEEK_LIKE_NORMALIZED",
    "DEEPSEEK_LIKE_SPECIALS",
    "LLAMA3_LIKE_SPECIALS",
    "counts_dict",
    "mk_dataset",
    "requires_rust_tokenize",
    "zstd",
]

# Special and ``normalized: true`` added tokens of the tokenizer-folder fixtures,
# as in DeepSeek V4's and Llama-3.2-1B's tokenizer.json (bos, eos, pad first).
DEEPSEEK_LIKE_SPECIALS = (
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜▁pad▁｜>",
)
DEEPSEEK_LIKE_NORMALIZED = (
    "<｜fim▁hole｜>",
    "<｜fim▁begin｜>",
    "<｜fim▁end｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
)
LLAMA3_LIKE_SPECIALS = (
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|finetune_right_pad_id|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token_0|>",
)

# Version-gated, never importorskip: the arrow-tokenize 2.x wheels cover
# CPython 3.11-3.13 only, and a missing package there must fail loudly.
requires_rust_tokenize = pytest.mark.skipif(
    not (3, 11) <= sys.version_info[:2] <= (3, 13),
    reason="arrow-tokenize 2.x ships wheels for Python 3.11-3.13 only",
)


def mk_dataset(name: str, shards: dict[int, int]) -> Dataset:
    """Create a Dataset backed by in-memory shards.

    Args:
        name: Dataset name (used as prefix in generated text payloads).
        shards: Mapping of shard ID to row count.

    Returns:
        A ``Dataset`` with ``InMemoryShard`` entries whose rows contain
        ``{"text": "<name>:<shard_id>:<row_index>"}``.
    """
    data: dict[int, InMemoryShard] = {}
    for sid, count in shards.items():
        rows = [{"text": f"{name}:{sid}:{i}"} for i in range(count)]
        data[int(sid)] = InMemoryShard(rows)
    return Dataset.from_dict(name, data)


def counts_dict(dataset: Dataset) -> dict[int, int]:
    """``shard_id -> sample_count`` dict for test assertions."""
    return dict(zip(dataset.ids().tolist(), dataset.counts().tolist(), strict=True))
