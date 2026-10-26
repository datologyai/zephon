"""Close a running process pipeline and resume its explicit checkpoint."""

import pytest

from zephon import Dataset, InMemoryShard, Pipeline, StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def test_close_process_pipeline_preserves_checkpoint_resume() -> None:
    documents = [{"input_ids": [i, i + 1]} for i in range(8)]
    source = StaticMixtureWorkSource(
        [Dataset.from_dict("tiny", {0: InMemoryShard(documents)})],
        {"tiny": 1.0},
        chunk_size=2,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipeline = (
        Pipeline(source)
        .batch(1)
        .options(
            runner="process",
            mtp_mode=False,
            shm_enabled=False,
            deterministic=True,
            max_workers=1,
            prefetch_batches=0,
        )
    )
    try:
        # Checkpoint destination setup also owns a runtime before iteration.
        initial = pipeline.checkpoint()
        pipeline.close()
        pipeline.close()
        assert pipeline.checkpoint() == initial
        pipeline.restore(initial)
        iterator = iter(pipeline)
        prefix = [next(iterator).to_training(dtype=None)["input_ids"]]
        checkpoint = pipeline.checkpoint()
        pipeline.close()
        pipeline.close()
        with pytest.raises(StopIteration):
            next(iterator)
        pipeline.restore(checkpoint)
        remaining = [batch.to_training(dtype=None)["input_ids"] for batch in pipeline]
        assert prefix + remaining == [[document["input_ids"]] for document in documents]
    finally:
        pipeline.close()
