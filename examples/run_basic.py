# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end example of building and iterating a Zephon pipeline."""

from collections import Counter

from zephon.api import Pipeline as PublicPipeline
from zephon.io import Dataset, InMemoryShard
from zephon.work import MixtureSpec, StaticMixtureWorkSource


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
    ds = Dataset.from_dict("demo", shards)
    ws = StaticMixtureWorkSource(
        [ds],
        mixture=MixtureSpec({"demo": 1.0}),
        chunk_size=1,
        seed=42,
        shuffle_shards=False,
    )

    pipe = PublicPipeline(ws).decode_text()

    pipe = pipe.tokenize(tokenizer_id="__fallback__", parallelism=1)
    pipe = pipe.batch(microbatch_size=10, drop_last=False)
    pipe = pipe.options(default_stage_prefetch=2, prefetch_batches=3)

    return pipe


def main() -> None:
    """Print the execution plan and iterate the example pipeline."""
    pipe = build_pipeline()
    print("PLAN:\n" + pipe.explain())
    counts = Counter()
    for i, batch in enumerate(pipe):
        batch = batch.to_training()
        counts["batches"] += 1
        counts["samples"] += len(batch["ids"])
        print(f"Batch {i}: ids={batch['ids']}, texts={batch['texts'][:2]} ...")
    print("SUMMARY:", counts)


if __name__ == "__main__":
    main()
