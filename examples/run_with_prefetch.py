#!/usr/bin/env python3
# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Example: Using PrefetchOp to optimize data loading.

This example demonstrates how to add prefetching to a Zephon pipeline
to reduce latency when loading data from multiple shards.

The prefetch operator looks ahead in the sample stream and downloads
shards to the local cache before they're needed by FetchOp.
"""

from collections import Counter
from pathlib import Path

from zephon.api import Pipeline
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource


def build_pipeline_with_prefetch() -> Pipeline:
    """Construct a pipeline with prefetching enabled."""
    root = Path(__file__).parent / "data" / "jsonl_prefetch_demo"
    cache_root = root / ".cache"
    dataset = Dataset.from_path("jsonl_prefetch_demo", str(root))

    work_source = StaticMixtureWorkSource(
        [dataset],
        mixture=MixtureSpec({"jsonl_prefetch_demo": 1.0}),
        chunk_size=8,  # Larger chunks to benefit from prefetch
        seed=42,
        shuffle_shards=False,
    )

    pipe = (
        Pipeline(work_source)
        .prefetch(buffer_size=64)  # Enable prefetch with default parallelism=4
        .decode_text()
    )
    pipe = pipe.options(io_options={"cache": {"enabled": True, "root": cache_root}})
    pipe = pipe.batch(microbatch_size=8, drop_last=False)
    return pipe


def build_pipeline_without_prefetch() -> Pipeline:
    """Construct a pipeline without prefetching for comparison."""
    root = Path(__file__).parent / "data" / "jsonl_prefetch_demo"
    cache_root = root / ".cache"
    dataset = Dataset.from_path("jsonl_prefetch_demo", str(root))

    work_source = StaticMixtureWorkSource(
        [dataset],
        mixture=MixtureSpec({"jsonl_prefetch_demo": 1.0}),
        chunk_size=8,
        seed=42,
        shuffle_shards=False,
    )

    pipe = Pipeline(work_source).decode_text()  # No prefetch
    pipe = pipe.options(io_options={"cache": {"enabled": True, "root": cache_root}})
    pipe = pipe.batch(microbatch_size=8, drop_last=False)
    return pipe


def run_pipeline(pipe: Pipeline, label: str) -> Counter:
    """Run a pipeline and collect statistics."""
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print("\nPLAN:")
    print(pipe.explain())
    print()

    counts = Counter()
    for i, batch in enumerate(pipe):
        batch_data = batch.to_training()
        counts["batches"] += 1
        counts["samples"] += len(batch_data["texts"])

        # Print first few batches
        if i < 3:
            texts = batch_data["texts"]
            print(f"Batch {i}: {len(texts)} samples")
            for j, text in enumerate(texts):
                print(f"  [{j}] {text[:60]}...")

    return counts


def main() -> None:
    """Compare pipeline with and without prefetch."""
    print("Zephon Prefetch Example")
    print("=" * 60)
    print("\nThis example demonstrates the prefetch operator with a")
    print("dataset containing 4 shards × 20 samples = 80 total samples.")
    print("\nPrefetch looks ahead in the sample stream and downloads")
    print("shards before they're needed, reducing fetch latency.")

    # Run with prefetch
    pipe_with = build_pipeline_with_prefetch()
    counts_with = run_pipeline(pipe_with, "PIPELINE WITH PREFETCH")
    print(f"\nSUMMARY: {counts_with}")

    # Run without prefetch
    pipe_without = build_pipeline_without_prefetch()
    counts_without = run_pipeline(pipe_without, "PIPELINE WITHOUT PREFETCH")
    print(f"\nSUMMARY: {counts_without}")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Both pipelines processed {counts_with['samples']} samples")
    print(f"  With prefetch:    {counts_with['batches']} batches")
    print(f"  Without prefetch: {counts_without['batches']} batches")
    print("\nNote: Performance benefits are most visible with:")
    print("  - Remote storage (S3, GCS) with high download latency")
    print("  - Large datasets with many shards")
    print("  - Sequential or predictable access patterns")


def demonstrate_configuration():
    """Show different prefetch configurations for various scenarios."""
    print("\n" + "=" * 60)
    print("PREFETCH CONFIGURATION EXAMPLES")
    print("=" * 60)

    configs = {
        "High-latency storage (S3/GCS)": {
            "buffer_size": 2048,
            "parallelism": 8,
        },
        "Sequential access": {
            "buffer_size": 1024,
            "parallelism": 4,
        },
        "Mixture datasets": {
            "buffer_size": 1536,
            "parallelism": 6,
        },
        "Local storage (minimal benefit)": {
            "buffer_size": 512,
            "parallelism": 2,
        },
    }

    for scenario, config in configs.items():
        print(f"\n{scenario}:")
        print(
            f"  .prefetch(buffer_size={config['buffer_size']}, "
            f"parallelism={config['parallelism']})"
        )


if __name__ == "__main__":
    main()
    demonstrate_configuration()
