"""Example pipeline demonstrating StaticMixtureWorkSource usage."""

from collections import Counter

from zephon.api import Pipeline as PublicPipeline
from zephon.io import Dataset, InMemoryShard
from zephon.work import MixtureSpec, StaticMixtureWorkSource


def build_pipeline() -> PublicPipeline:
    """Build a simple pipeline mixing two in-memory datasets."""
    shards = {
        0: InMemoryShard([{"text": f"alpha sample {i}"} for i in range(10)]),
        1: InMemoryShard([{"text": f"beta sample {i}"} for i in range(6)]),
        2: InMemoryShard([{"text": f"gamma sample {i}"} for i in range(8)]),
    }
    dataset_a = Dataset.from_dict("alpha", {0: shards[0], 2: shards[2]})
    dataset_b = Dataset.from_dict("beta", {1: shards[1]})

    # TODO(MaxiBoether): remove need to call .weights
    work_source = StaticMixtureWorkSource(
        datasets=[dataset_a, dataset_b],
        mixture=MixtureSpec({"alpha": 0.25, "beta": 0.75}).weights,
        chunk_size=4,
        seed=99,
    )

    pipe = PublicPipeline(work_source).decode_text()
    pipe = pipe.tokenize(tokenizer_id="__fallback__", parallelism=1)
    pipe = pipe.batch(global_batch=4, dp_world=1, drop_last=False)
    return pipe


def main() -> None:
    """Build and run a simple two-dataset mixture pipeline.

    This example demonstrates the high-level workflow:
    1) Construct two in-memory datasets via ``Dataset.from_dict``.
    2) Create a ``StaticMixtureWorkSource`` with an explicit mixture.
    3) Build a ``Pipeline`` that decodes, tokenizes and batches.
    4) Print the execution plan and iterate the pipeline.
    """
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
