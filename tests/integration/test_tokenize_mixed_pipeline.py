# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Mixed text/chat tokenization through process workers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from zephon import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.work.static_mixture import StaticMixtureWorkSource

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")


def _work() -> StaticMixtureWorkSource:
    from tests.helpers.chat import CONVERSATION

    rows: list[dict[str, Any]] = [
        {"text": "hello there how are you good morning"}
        if i % 3
        else {"messages": CONVERSATION}
        for i in range(12)
    ]
    data = Dataset.from_dict(
        "mixed", {0: InMemoryShard(rows[:6]), 1: InMemoryShard(rows[6:])}
    )
    return StaticMixtureWorkSource([data], {"mixed": 1.0}, chunk_size=3)


def _tokenized(runner: str) -> list[tuple[list[int], list[int]]]:
    from tests.helpers.chat import TAGGED_TEMPLATE, make_fast_tokenizer

    pipe = (
        Pipeline(_work())
        .tokenize_mixed(
            make_fast_tokenizer(),
            text_special_tokens="eos",
            chat_template=TAGGED_TEMPLATE,
            max_length=8,
            split_long_samples=True,
        )
        .options(runner=runner, max_workers=1, deterministic=True)
    )
    assert pipe.preflight_tokenizers().ok
    return [
        (
            record.payload["input_ids"].tolist(),
            record.payload["loss_mask"].tolist(),
        )
        for record in pipe
    ]


def test_process_workers_match_inline_tokenization() -> None:
    inline = _tokenized("inline")
    assert _tokenized("process") == inline
    # Each 22-token conversation keeps two of its three chunks; each 8-token
    # text document fits in one record.
    assert len(inline) == 4 * 2 + 8
    assert all(len(ids) <= 8 for ids, _ in inline)
    assert all(mask[0] == 0 and np.asarray(mask).sum() > 0 for _, mask in inline)
