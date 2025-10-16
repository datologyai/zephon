# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Example pipeline reading local JSONL shards."""

from collections import Counter
from pathlib import Path

from zephon.api import Pipeline as PublicPipeline
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource


def build_pipeline() -> PublicPipeline:
    """Construct a pipeline that reads from local JSONL shards."""
    root = Path(__file__).parent / "data" / "jsonl_demo"
    cache_root = root / ".cache"
    dataset = Dataset.from_path("jsonl_demo", str(root))

    work_source = StaticMixtureWorkSource(
        [dataset],
        mixture=MixtureSpec({"jsonl_demo": 1.0}),
        chunk_size=2,
        seed=7,
        shuffle_shards=False,
    )

    pipe = PublicPipeline(work_source).decode_text()
    pipe = pipe.options(io_options={"cache": {"enabled": True, "root": cache_root}})
    pipe = pipe.batch(microbatch_size=2, drop_last=False)
    return pipe


def main() -> None:
    """Print the execution plan and iterate the JSONL example pipeline."""
    pipe = build_pipeline()
    print("PLAN:\n" + pipe.explain())
    counts = Counter()
    for i, batch in enumerate(pipe):
        batch = batch.to_training()
        counts["batches"] += 1
        counts["samples"] += len(batch["ids"])
        print(f"Batch {i}: ids={batch['ids']}, texts={batch['texts']}")
    print("SUMMARY:", counts)


if __name__ == "__main__":
    main()
