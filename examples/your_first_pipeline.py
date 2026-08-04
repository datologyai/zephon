#!/usr/bin/env python3
# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Your First Pipeline - Run this after reading the Quick Start guide.

This script uses checked-in JSONL shards and a real Hugging Face tokenizer when
Transformers is installed. Otherwise, it warns and uses Zephon's built-in
fallback tokenizer so the example remains runnable with the base installation.

Run with: uv run python examples/your_first_pipeline.py
"""

import warnings
from importlib.util import find_spec
from pathlib import Path

from zephon import Pipeline
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource


def default_tokenizer_id() -> str:
    """Choose GPT-2 when Transformers is installed, otherwise the fallback."""
    if find_spec("transformers") is not None:
        return "gpt2"

    warnings.warn(
        "Transformers is not installed; using Zephon's built-in fallback "
        "tokenizer. Install transformers to run this example with GPT-2.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "__fallback__"


def build_pipeline(tokenizer_id: str | None = None) -> Pipeline:
    """Build the file-backed text pipeline shown in the Quick Start."""
    if tokenizer_id is None:
        tokenizer_id = default_tokenizer_id()

    root = Path(__file__).parent / "data" / "jsonl_demo"
    dataset = Dataset.from_path("jsonl_demo", str(root))

    ws = StaticMixtureWorkSource(
        datasets=[dataset],
        mixture=MixtureSpec({"jsonl_demo": 1.0}),
        chunk_size=1,  # The checked-in demo has only five samples
        seed=42,
    )

    pipeline = (
        Pipeline(ws)
        .decode_text()
        .tokenize(
            tokenizer_id=tokenizer_id,
            field="text",
            padding=True,
            parallelism=1,
            preserve_upstream_payload=True,  # Keep "text" field for display
        )
        .batch(microbatch_size=2, drop_last=False)
    )

    return pipeline


def main(tokenizer_id: str | None = None) -> None:
    """Run the Your First Pipeline demo."""
    print("Your First Zephon Pipeline")
    pipeline = build_pipeline(tokenizer_id)

    print("PLAN:\n" + pipeline.explain())
    for batch_number, batch in enumerate(pipeline):
        training_batch = batch.to_training()
        shape = tuple(training_batch["input_ids"].shape)
        texts = training_batch["texts"]
        print(
            f"Batch {batch_number}: {len(training_batch['ids'])} samples, "
            f"shape={shape}, texts={texts}"
        )


if __name__ == "__main__":
    main()
