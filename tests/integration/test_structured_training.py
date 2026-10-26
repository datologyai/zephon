"""Whole-document packing retains media association and checkpoint lineage."""

import json
from collections.abc import Sequence
from typing import Any

import pytest

from zephon import Dataset, InMemoryShard, Pipeline, StaticMixtureWorkSource
from zephon.types import SampleBatch

pytestmark = pytest.mark.integration


def merge_documents(documents: list[Any]) -> dict[str, Any]:
    """Assemble unshifted values in the exact order chosen by the packer."""
    return {
        "input_ids": [
            token for document in documents for token in document["input_ids"]
        ],
        "loss_mask": [mask for document in documents for mask in document["loss_mask"]],
        "positions": [
            i for document in documents for i in range(len(document["input_ids"]))
        ],
        "media": [image for document in documents for image in document["media"]],
        "document_lengths": [len(document["input_ids"]) for document in documents],
    }


def keep_documents(values: Sequence[Any], length: int, shift: bool) -> list[Any]:
    return list(values)


def make_pipeline(algorithm: str, runner: str) -> Pipeline:
    documents = []
    for i in range(12):
        # Expanded image slots already count toward capacity. Final text token
        # guarantees shifting does not remove the image's last slot.
        tokens = [10 + i, 90, 90] + ([90] if i % 2 else []) + [2]
        documents.append(
            {
                "input_ids": tokens,
                "loss_mask": [0] * (len(tokens) - 1) + [1],
                "media": [f"image-{i}"],
            }
        )
    dataset = Dataset.from_dict("images", {0: InMemoryShard(documents)})
    source = StaticMixtureWorkSource(
        [dataset],
        {"images": 1.0},
        chunk_size=3,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipeline = Pipeline(source).pack_sequences(
        max_length=10, algorithm=algorithm, num_bins=2, pack_payloads=merge_documents
    )
    pipeline.batch(2, drop_last=False)
    pipeline.options(
        deterministic=True,
        max_workers=2,
        runner=runner,
        mtp_mode=False,
        shm_enabled=False,
    )
    return pipeline


def convert(batch: SampleBatch) -> dict[str, Any]:
    output = batch.to_training(
        dtype=None,
        return_labels=True,
        return_loss_mask=True,
        return_cu_seqlens=True,
        return_padding_mask=True,
        pad_to_length=10,
        pad_token_id=2,
        field_collators={"media": keep_documents, "document_lengths": keep_documents},
    )
    # Contribution refs remain present on the envelope, irrespective of the
    # tensor conversion. Include them in the replay comparison.
    output["contributors"] = [
        [
            (ref.cursor.as_key(), ref.is_last_child)
            for ref in record.meta.contribution_refs()
        ]
        for record in batch.records
    ]
    return output


@pytest.mark.parametrize("algorithm", ["first_fit", "best_fit"])
@pytest.mark.parametrize("runner", ["inline", "threads", "process"])
def test_structured_training_capacity_media_order_and_resume(
    algorithm: str, runner: str
) -> None:
    baseline_pipe = make_pipeline(algorithm, runner)
    baseline = [convert(batch) for batch in baseline_pipe]
    assert baseline
    media = []
    for output in baseline:
        for ids, labels, masks, images, lengths in zip(
            output["input_ids"],
            output["labels"],
            output["padding_mask"],
            output["media"],
            output["document_lengths"],
        ):
            assert len(ids) == len(labels) == 9
            assert sum(lengths) <= 10
            assert len(images) == len(lengths)
            offset = 0
            for image, length in zip(images, lengths):
                assert ids[offset] == 10 + int(image.removeprefix("image-"))
                assert ids[offset + 1 : offset + length - 1] == [90] * (length - 2)
                offset += length
            assert masks == [False] * min(9, offset) + [True] * max(0, 9 - offset)
            media.extend(images)
    assert sorted(media) == sorted(f"image-{i}" for i in range(12))

    first = make_pipeline(algorithm, runner)
    iterator = iter(first)
    try:
        prefix = [convert(next(iterator))]
        checkpoint = json.loads(json.dumps(first.checkpoint()))
    finally:
        iterator.close()
    resumed = make_pipeline(algorithm, runner)
    resumed.restore(checkpoint)
    suffix = [convert(batch) for batch in resumed]
    assert prefix + suffix == baseline


def resume_in_fresh_driver(
    checkpoint: dict[str, Any], algorithm: str, output: Any
) -> None:
    """Restore public checkpoint data in a fresh Python process."""
    pipeline = make_pipeline(algorithm, "inline")
    pipeline.restore(checkpoint)
    output.put([convert(batch) for batch in pipeline])


@pytest.mark.parametrize("algorithm", ["first_fit", "best_fit"])
def test_structured_training_fresh_driver_replay(algorithm: str) -> None:
    import multiprocessing

    baseline = [convert(batch) for batch in make_pipeline(algorithm, "inline")]
    first = make_pipeline(algorithm, "inline")
    iterator = iter(first)
    try:
        prefix = [convert(next(iterator))]
        checkpoint = json.loads(json.dumps(first.checkpoint()))
    finally:
        iterator.close()
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    child = context.Process(
        target=resume_in_fresh_driver, args=(checkpoint, algorithm, output)
    )
    child.start()
    try:
        suffix = output.get(timeout=30)
        child.join(timeout=10)
        assert child.exitcode == 0
        assert prefix + suffix == baseline
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
        output.close()
        output.join_thread()
