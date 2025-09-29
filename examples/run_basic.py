# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end example of building and iterating a Zephon pipeline."""

from collections import Counter

from zephon.api import Pipeline as PublicPipeline
from zephon.io import InMemoryShard, InMemoryShardStore
from zephon.work import StaticWorkSource


def build_pipeline() -> PublicPipeline:
    """Construct a small text pipeline with decode/tokenize/batch stages."""
    shards = {
        0: InMemoryShard(
            [
                {"text": "short 0"},
                {"text": "short 1"},
                *({"text": f"lorem sample 0-{i}"} for i in range(2, 12)),
            ]
        ),
        1: InMemoryShard(
            [
                {"text": "tiny 0"},
                {"text": "tiny 1"},
                *({"text": f"ipsum sample 1-{i}"} for i in range(2, 12)),
            ]
        ),
    }
    store = InMemoryShardStore(shards)
    index = {shard_id: len(shard) for shard_id, shard in shards.items()}
    ws = StaticWorkSource(index, chunk_size=1, seed=42, shuffle=False)

    pipe = PublicPipeline(ws, store).decode_text()

    pipe = pipe.tokenize(tokenizer_id="__fallback__", parallelism=1)
    pipe = pipe.batch(global_batch=10, dp_world=1, drop_last=False)
    pipe = pipe.options(default_stage_prefetch=2, prefetch_batches=3)

    return pipe


def main() -> None:
    """Print the execution plan and iterate the example pipeline."""
    pipe = build_pipeline()
    print("PLAN:\n" + pipe.explain())
    counts = Counter()
    for i, batch in enumerate(pipe):
        counts["batches"] += 1
        counts["samples"] += len(batch["ids"])
        print(f"Batch {i}: ids={batch['ids']}, texts={batch['texts'][:2]} ...")
    print("SUMMARY:", counts)


if __name__ == "__main__":
    main()
