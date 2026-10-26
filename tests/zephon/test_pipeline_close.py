"""Public cleanup covers iteration and checkpoint-destination initialization."""

import gc
import pickle
from typing import Any

import pytest

from zephon import Dataset, InMemoryShard, Pipeline, StaticMixtureWorkSource


def make_pipeline(runner: str = "inline", *, mtp: bool = False) -> Pipeline:
    dataset = Dataset.from_dict(
        "tiny", {0: InMemoryShard([{"input_ids": [i, i + 1]} for i in range(8)])}
    )
    source = StaticMixtureWorkSource(
        [dataset],
        {"tiny": 1.0},
        chunk_size=2,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    return (
        Pipeline(source)
        .batch(1)
        .options(
            runner=runner,
            mtp_mode=mtp,
            shm_enabled=False,
            deterministic=True,
            max_workers=1,
            prefetch_batches=0,
        )
    )


def payloads(pipeline: Pipeline) -> list[Any]:
    return [batch.to_training(dtype=None)["input_ids"] for batch in pipeline]


def test_close_fresh_or_unstarted_pipeline_does_not_initialize_runtime() -> None:
    pipeline = make_pipeline()
    iterator = iter(pipeline)
    pipeline.close()
    pipeline.close()
    assert pipeline._engine is None
    with pytest.raises(StopIteration):
        next(iterator)
    assert len(payloads(pipeline)) == 8


@pytest.mark.parametrize("runner", ["inline", "threads", "process"])
def test_close_checkpoint_destination_before_iteration(runner: str) -> None:
    pipeline = make_pipeline(runner)
    checkpoint = pipeline.checkpoint()
    engine = pipeline._engine
    assert engine is not None and not engine._closed
    pipeline.close()
    assert engine._closed
    pipeline.close()
    assert pipeline.checkpoint() == checkpoint
    pipeline.restore(checkpoint)
    assert len(payloads(pipeline)) == 8


def test_close_after_invalid_restore_still_releases_destination_runtime() -> None:
    pipeline = make_pipeline()
    checkpoint = pipeline.checkpoint()
    with pytest.raises((ValueError, TypeError, KeyError)):
        pipeline.restore({})
    pipeline.close()
    assert pipeline._engine is not None and pipeline._engine._closed
    assert pipeline.checkpoint() == checkpoint


@pytest.mark.parametrize("runner", ["inline", "threads", "process"])
def test_close_active_iterator_preserves_explicit_resume(runner: str) -> None:
    baseline = payloads(make_pipeline(runner))
    pipeline = make_pipeline(runner)
    iterator = iter(pipeline)
    prefix = [next(iterator).to_training(dtype=None)["input_ids"]]
    checkpoint = pipeline.checkpoint()
    pipeline.close()
    pipeline.close()
    assert pipeline._engine is not None and pipeline._engine._closed
    with pytest.raises(StopIteration):
        next(iterator)
    assert not pipeline._iterating
    pipeline.restore(checkpoint)
    assert prefix + payloads(pipeline) == baseline


def test_close_does_not_discard_pending_restore_or_pickle_runtime_refs() -> None:
    source = make_pipeline()
    iterator = iter(source)
    next(iterator)
    checkpoint = source.checkpoint()
    source.close()
    restored = make_pipeline()
    restored.restore(checkpoint)
    unstarted = iter(restored)
    restored.close()
    assert restored._engine is None
    assert restored.checkpoint() == checkpoint
    transferred = pickle.loads(pickle.dumps(restored))
    assert len(transferred._iterators) == 0
    assert transferred.checkpoint() == checkpoint
    assert len(payloads(transferred)) == 7
    with pytest.raises(StopIteration):
        next(unstarted)


def test_iterator_tracking_does_not_prevent_existing_gc_cleanup() -> None:
    pipeline = make_pipeline()
    iterator = iter(pipeline)
    next(iterator)
    engine = pipeline._engine
    assert engine is not None and not engine._closed
    del iterator
    gc.collect()
    assert engine._closed
    assert not pipeline._iterating
    assert len(pipeline._iterators) == 0


def test_close_after_full_exhaustion_is_idempotent() -> None:
    pipeline = make_pipeline()
    assert len(payloads(pipeline)) == 8
    pipeline.close()
    checkpoint = pipeline.checkpoint()
    pipeline.close()
    assert pipeline.checkpoint() == checkpoint


@pytest.mark.integration
def test_close_active_mtp_iterator_releases_subprocess_and_restores() -> None:
    baseline = payloads(make_pipeline(mtp=True))
    pipeline = make_pipeline(mtp=True)
    iterator = iter(pipeline)
    prefix = [next(iterator).to_training(dtype=None)["input_ids"]]
    checkpoint = pipeline.checkpoint()
    pipeline.close()
    pipeline.close()
    assert pipeline._sp is None
    assert not pipeline._iterating
    with pytest.raises(StopIteration):
        next(iterator)
    pipeline.restore(checkpoint)
    assert prefix + payloads(pipeline) == baseline


@pytest.mark.integration
def test_mtp_terminal_state_replaces_checkpoint_destination_engine() -> None:
    pipeline = make_pipeline(mtp=True)
    initial = pipeline.checkpoint()
    assert len(payloads(pipeline)) == 8
    pipeline.close()
    terminal = pipeline.checkpoint()
    assert terminal != initial
    restored = make_pipeline(mtp=True)
    restored.restore(terminal)
    assert payloads(restored) == []
    restored.close()
