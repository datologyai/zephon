# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Saved tokenizer fixtures without optional tokenizer dependencies."""

from __future__ import annotations

import json
from pathlib import Path


def write_wordlevel_tokenizer(
    folder: Path, *, model_max_length: int | None = None
) -> Path:
    """Write a complete tokenizer with BOS, EOS, PAD and a small vocabulary."""
    vocabulary = {
        "<bos>": 0,
        "<eos>": 1,
        "<pad>": 2,
        "<unk>": 3,
        "hello": 4,
        "world": 5,
        "data": 6,
        "loader": 7,
        "!": 8,
    }
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "normalizer": None,
        "decoder": None,
        "added_tokens": [
            {
                "id": vocabulary[token],
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for token in ("<bos>", "<eos>", "<pad>")
        ],
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<eos>", "type_id": 0}},
            ],
            "pair": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {
                token: {"id": token, "ids": [vocabulary[token]], "tokens": [token]}
                for token in ("<bos>", "<eos>")
            },
        },
        "model": {"type": "WordLevel", "vocab": vocabulary, "unk_token": "<unk>"},
    }
    config: dict[str, str | int] = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    }
    if model_max_length is not None:
        config["model_max_length"] = model_max_length
    files = {"tokenizer.json": tokenizer, "tokenizer_config.json": config}
    for name, contents in files.items():
        (folder / name).write_text(json.dumps(contents), encoding="utf-8")
    return folder
