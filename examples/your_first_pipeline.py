#!/usr/bin/env python3
# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Your First Pipeline - Run this after reading the Quick Start guide.

This script mirrors the "Your First Pipeline" section in docs/source/quickstart.md
with extra print statements to help you understand what happens at each step.

We add tokenization because batch.to_training() requires tokenized payloads
(input_ids) for the training format. The __fallback__ tokenizer is a simple
character-level tokenizer for demos that does not need HuggingFace.

Run with: uv run python examples/your_first_pipeline.py
"""

from zephon import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.work import MixtureSpec, StaticMixtureWorkSource


def main() -> None:
    """Run the Your First Pipeline demo."""
    print("=" * 60)
    print("Your First Zephon Pipeline")
    print("=" * 60)

    # 1. Create some sample data
    print("\n1. Creating sample data (3 records in a single shard)...")
    shards = {
        0: InMemoryShard(
            [
                {"text": "Hello world"},
                {"text": "Zephon is fast"},
                {"text": "Data loading made easy"},
            ]
        )
    }
    print(f"   Shard 0 contains {len(shards[0])} samples")

    # 2. Create a dataset from the shards
    print("\n2. Creating a Dataset from the shards...")
    ds = Dataset.from_dict("demo", shards)
    print(f"   Dataset '{ds.name}' has {len(ds)} total samples")

    # 3. Create a work source that controls sample distribution
    print("\n3. Creating a StaticMixtureWorkSource...")
    ws = StaticMixtureWorkSource(
        datasets=[ds],
        mixture=MixtureSpec({"demo": 1.0}),
        chunk_size=1,
        seed=42,
    )
    print("   Work source configured with mixture={'demo': 1.0}, chunk_size=1, seed=42")

    # 4. Build the pipeline
    print(
        "\n4. Building the pipeline (decode_text -> tokenize -> batch with microbatch_size=2)..."
    )
    pipeline = (
        Pipeline(ws)
        .decode_text()  # Extract text from samples
        .tokenize(
            tokenizer_id="__fallback__",
            field="text",
            parallelism=1,
            preserve_upstream_payload=True,  # Keep "text" field for display
        )
        .batch(
            microbatch_size=2, drop_last=False
        )  # Group into batches; keep last partial batch
    )
    print(
        "   Pipeline ready. decode_text() extracts the 'text' field; "
        "tokenize() adds input_ids; batch() groups 2 samples."
    )

    # 5. Iterate over batches
    print("\n5. Iterating over batches:")
    print("-" * 60)
    batch_count = 0
    for batch_num, batch in enumerate(pipeline):
        training_batch = batch.to_training(dtype=None)  # Use lists for variable-length
        texts = training_batch["texts"]
        batch_count += 1
        print(f"   Batch {batch_num}: {texts}")
    print("-" * 60)
    print(f"\nDone! You processed 3 samples in {batch_count} batch(es).")


if __name__ == "__main__":
    main()
