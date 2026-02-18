# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for MapTransform and MapBatchTransform operators in pipeline context."""

import json
from pathlib import Path

import pytest

from zephon.api import Pipeline as PublicPipeline
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def test_pipeline_map_transform_full_payload(tmp_path: Path) -> None:
    """Test MapTransform in pipeline with full payload transformation."""
    # Build a small JSONL dataset
    shard0 = tmp_path / "shard0.jsonl"
    shard0.write_text(
        "\n".join(json.dumps({"text": f"sample {i}"}) for i in range(3)),
        encoding="utf-8",
    )
    jsonl_dataset = Dataset.from_path("demo", str(tmp_path))

    def transform(payload: dict) -> dict:
        return {"text": payload.get("text", "").upper(), "transformed": True}

    work_source = StaticMixtureWorkSource(
        [jsonl_dataset],
        mixture=MixtureSpec({jsonl_dataset.name: 1.0}).weights,
        chunk_size=1,
        seed=11,
        shuffle_shards=False,
    )
    pipe = (
        PublicPipeline(work_source)
        .decode_text()
        .map_transform(transform)
        .batch(microbatch_size=2, drop_last=False)
    )

    iterator = iter(pipe)
    try:
        batch = next(iterator)
        assert len(batch.records) == 2
        # Check that transformation was applied
        for record in batch.records:
            payload = record.payload
            assert isinstance(payload, dict)
            assert payload.get("transformed") is True
            assert "SAMPLE" in payload.get("text", "")
    finally:
        iterator.close()


def test_pipeline_map_transform_filtering(tmp_path: Path) -> None:
    """Test MapTransform filtering in pipeline."""
    shard0 = tmp_path / "shard0.jsonl"
    shard0.write_text(
        "\n".join(
            json.dumps({"text": f"sample {i}", "keep": i % 2 == 0}) for i in range(5)
        ),
        encoding="utf-8",
    )
    jsonl_dataset = Dataset.from_path("demo", str(tmp_path))

    def transform(payload: dict) -> dict | None:
        if not payload.get("keep", False):
            return None  # Drop samples where keep=False
        return {"text": payload["text"], "kept": True}

    work_source = StaticMixtureWorkSource(
        [jsonl_dataset],
        mixture=MixtureSpec({jsonl_dataset.name: 1.0}).weights,
        chunk_size=1,
        seed=11,
        shuffle_shards=False,
    )
    pipe = (
        PublicPipeline(work_source)
        .decode_text()
        .map_transform(transform, drop_none=True)
        .batch(microbatch_size=2, drop_last=False)
    )

    iterator = iter(pipe)
    try:
        # Should only get batches with kept samples (even indices: 0, 2, 4)
        batches = list(iterator)
        # Verify filtering worked - should have fewer batches than without filtering
        total_samples = sum(len(batch.records) for batch in batches)
        # With 5 samples, 3 kept (0, 2, 4), so should have at least 1 batch
        assert total_samples >= 1
        # All batches should have kept=True
        for batch in batches:
            for record in batch.records:
                payload = record.payload
                assert isinstance(payload, dict)
                assert payload.get("kept") is True
    finally:
        iterator.close()


def test_pipeline_map_transform_chaining(tmp_path: Path) -> None:
    """Test chaining multiple MapTransform operations."""
    shard0 = tmp_path / "shard0.jsonl"
    shard0.write_text(
        "\n".join(json.dumps({"text": f"sample {i}"}) for i in range(3)),
        encoding="utf-8",
    )
    jsonl_dataset = Dataset.from_path("demo", str(tmp_path))

    def transform1(payload: dict) -> dict:
        return {"text": payload.get("text", "").upper(), "step": 1}

    def transform2(payload: dict) -> dict:
        return {**payload, "step": 2}

    work_source = StaticMixtureWorkSource(
        [jsonl_dataset],
        mixture=MixtureSpec({jsonl_dataset.name: 1.0}).weights,
        chunk_size=1,
        seed=11,
        shuffle_shards=False,
    )
    pipe = (
        PublicPipeline(work_source)
        .decode_text()
        .map_transform(transform1)
        .map_transform(transform2)
        .batch(microbatch_size=2, drop_last=False)
    )

    iterator = iter(pipe)
    try:
        batch = next(iterator)
        assert len(batch.records) == 2
        # Check that both transformations were applied
        for record in batch.records:
            payload = record.payload
            assert isinstance(payload, dict)
            assert "SAMPLE" in payload.get("text", "")
            assert payload.get("step") == 2  # Last transform wins
    finally:
        iterator.close()


# Process runner tests - these verify that map_transform works correctly
# when pickled and sent to worker processes


def test_pipeline_map_transform_with_process_runner_full_payload(
    tmp_path: Path,
) -> None:
    """Test MapTransform with process runner - full payload transformation."""
    shard0 = tmp_path / "shard0.jsonl"
    shard0.write_text(
        "\n".join(json.dumps({"text": f"sample {i}"}) for i in range(5)),
        encoding="utf-8",
    )
    jsonl_dataset = Dataset.from_path("demo", str(tmp_path))

    def transform(payload: dict) -> dict:
        return {"text": payload.get("text", "").upper(), "transformed": True}

    work_source = StaticMixtureWorkSource(
        [jsonl_dataset],
        mixture=MixtureSpec({jsonl_dataset.name: 1.0}).weights,
        chunk_size=1,
        seed=11,
        shuffle_shards=False,
    )
    pipe = (
        PublicPipeline(work_source)
        .decode_text()
        .map_transform(transform)
        .batch(microbatch_size=2, drop_last=False)
        .options(runner="process", max_workers=2)
    )

    iterator = iter(pipe)
    try:
        batches = list(iterator)
        # Verify transformation was applied across all batches
        total_samples = sum(len(batch.records) for batch in batches)
        assert total_samples >= 3  # Should have at least 3 samples
        for batch in batches:
            for record in batch.records:
                payload = record.payload
                assert isinstance(payload, dict)
                assert payload.get("transformed") is True
                assert "SAMPLE" in payload.get("text", "")
    finally:
        iterator.close()


def test_pipeline_map_transform_with_process_runner_filtering(tmp_path: Path) -> None:
    """Test MapTransform filtering with process runner."""
    shard0 = tmp_path / "shard0.jsonl"
    shard0.write_text(
        "\n".join(
            json.dumps({"text": f"sample {i}", "keep": i % 2 == 0}) for i in range(10)
        ),
        encoding="utf-8",
    )
    jsonl_dataset = Dataset.from_path("demo", str(tmp_path))

    def transform(payload: dict) -> dict | None:
        if not payload.get("keep", False):
            return None  # Drop samples where keep=False
        return {"text": payload["text"], "kept": True}

    work_source = StaticMixtureWorkSource(
        [jsonl_dataset],
        mixture=MixtureSpec({jsonl_dataset.name: 1.0}).weights,
        chunk_size=1,
        seed=11,
        shuffle_shards=False,
    )
    pipe = (
        PublicPipeline(work_source)
        .decode_text()
        .map_transform(transform, drop_none=True)
        .batch(microbatch_size=2, drop_last=False)
        .options(runner="process", max_workers=2)
    )

    iterator = iter(pipe)
    try:
        batches = list(iterator)
        # Verify filtering worked - should have fewer samples than input
        total_samples = sum(len(batch.records) for batch in batches)
        # With 10 samples, 5 kept (even indices), so should have at least 2 batches
        assert total_samples >= 2
        # All batches should have kept=True
        for batch in batches:
            for record in batch.records:
                payload = record.payload
                assert isinstance(payload, dict)
                assert payload.get("kept") is True
    finally:
        iterator.close()


def test_pipeline_map_transform_with_process_runner_chaining(tmp_path: Path) -> None:
    """Test chaining multiple MapTransform operations with process runner."""
    shard0 = tmp_path / "shard0.jsonl"
    shard0.write_text(
        "\n".join(json.dumps({"text": f"sample {i}"}) for i in range(5)),
        encoding="utf-8",
    )
    jsonl_dataset = Dataset.from_path("demo", str(tmp_path))

    def transform1(payload: dict) -> dict:
        return {"text": payload.get("text", "").upper(), "step": 1}

    def transform2(payload: dict) -> dict:
        return {**payload, "step": 2}

    work_source = StaticMixtureWorkSource(
        [jsonl_dataset],
        mixture=MixtureSpec({jsonl_dataset.name: 1.0}).weights,
        chunk_size=1,
        seed=11,
        shuffle_shards=False,
    )
    pipe = (
        PublicPipeline(work_source)
        .decode_text()
        .map_transform(transform1)
        .map_transform(transform2)
        .batch(microbatch_size=2, drop_last=False)
        .options(runner="process", max_workers=2)
    )

    iterator = iter(pipe)
    try:
        batches = list(iterator)
        total_samples = sum(len(batch.records) for batch in batches)
        assert total_samples >= 3
        # Check that both transformations were applied
        for batch in batches:
            for record in batch.records:
                payload = record.payload
                assert isinstance(payload, dict)
                assert "SAMPLE" in payload.get("text", "")
                assert payload.get("step") == 2  # Last transform wins
    finally:
        iterator.close()


def test_pipeline_map_transform_with_process_runner_complex_transform(
    tmp_path: Path,
) -> None:
    """Test MapTransform with process runner using a more complex transform function.

    This test uses a transform function that captures variables from outer scope,
    which can sometimes cause pickling issues. This helps verify that the process
    runner can properly pickle and unpickle the transform function.
    """
    shard0 = tmp_path / "shard0.jsonl"
    shard0.write_text(
        "\n".join(json.dumps({"text": f"sample {i}", "value": i}) for i in range(5)),
        encoding="utf-8",
    )
    jsonl_dataset = Dataset.from_path("demo", str(tmp_path))

    # Use a transform function that captures variables from outer scope
    prefix = "PROCESSED"
    multiplier = 2

    def transform(payload: dict) -> dict:
        text = payload.get("text", "")
        value = payload.get("value", 0)
        return {
            "text": f"{prefix}: {text.upper()}",
            "value": value * multiplier,
            "transformed": True,
        }

    work_source = StaticMixtureWorkSource(
        [jsonl_dataset],
        mixture=MixtureSpec({jsonl_dataset.name: 1.0}).weights,
        chunk_size=1,
        seed=11,
        shuffle_shards=False,
    )
    pipe = (
        PublicPipeline(work_source)
        .decode_text()
        .map_transform(transform)
        .batch(microbatch_size=2, drop_last=False)
        .options(runner="process", max_workers=2)
    )

    iterator = iter(pipe)
    try:
        batches = list(iterator)
        total_samples = sum(len(batch.records) for batch in batches)
        assert total_samples >= 3
        # Verify complex transformation was applied correctly
        for batch in batches:
            for record in batch.records:
                payload = record.payload
                assert isinstance(payload, dict)
                assert payload.get("transformed") is True
                assert "PROCESSED:" in payload.get("text", "")
                assert "SAMPLE" in payload.get("text", "")
                # Verify multiplier was applied
                assert payload.get("value") % 2 == 0
    finally:
        iterator.close()


def test_pipeline_map_transform_with_process_runner_lambda(tmp_path: Path) -> None:
    """Test MapTransform with lambda functions in process runner.

    This explicitly tests that lambda functions work with ProcessRunner.
    Without cloudpickle, this would fail with: Can't pickle <function <lambda>>.
    """
    shard0 = tmp_path / "shard0.jsonl"
    shard0.write_text(
        "\n".join(json.dumps({"text": f"sample {i}", "value": i}) for i in range(5)),
        encoding="utf-8",
    )
    jsonl_dataset = Dataset.from_path("demo", str(tmp_path))

    work_source = StaticMixtureWorkSource(
        [jsonl_dataset],
        mixture=MixtureSpec({jsonl_dataset.name: 1.0}).weights,
        chunk_size=1,
        seed=11,
        shuffle_shards=False,
    )

    # Use lambdas directly - this is the key test for cloudpickle support
    pipe = (
        PublicPipeline(work_source)
        .decode_text()
        .map_transform(lambda p: {**p, "value": p["value"] * 2})  # Lambda!
        .map_transform(lambda p: {**p, "doubled": True})  # Another lambda!
        .batch(microbatch_size=2, drop_last=False)
        .options(runner="process", max_workers=2)
    )

    iterator = iter(pipe)
    try:
        batches = list(iterator)
        total_samples = sum(len(batch.records) for batch in batches)
        assert total_samples >= 3
        # Verify lambdas were applied
        for batch in batches:
            for record in batch.records:
                payload = record.payload
                assert isinstance(payload, dict)
                assert payload.get("doubled") is True
                # Value should be doubled
                assert payload.get("value") % 2 == 0
    finally:
        iterator.close()


def test_pipeline_map_batch_after_batch(tmp_path: Path) -> None:
    """Test map_batch() applied after batch() transforms entire batches."""
    shard0 = tmp_path / "shard0.jsonl"
    shard0.write_text(
        "\n".join(json.dumps({"text": f"sample {i}"}) for i in range(4)),
        encoding="utf-8",
    )
    jsonl_dataset = Dataset.from_path("demo", str(tmp_path))

    from zephon.core.constants import SampleBatch, SampleRecord

    def transform_batch(batch: SampleBatch) -> SampleBatch:
        new_records = []
        for record in batch.records:
            payload = record.payload
            assert isinstance(payload, dict)
            new_payload = {"text": payload["text"].upper(), "batch_transformed": True}
            new_records.append(SampleRecord(meta=record.meta, payload=new_payload))
        return SampleBatch(records=tuple(new_records))

    work_source = StaticMixtureWorkSource(
        [jsonl_dataset],
        mixture=MixtureSpec({jsonl_dataset.name: 1.0}).weights,
        chunk_size=1,
        seed=11,
        shuffle_shards=False,
    )
    pipe = (
        PublicPipeline(work_source)
        .decode_text()
        .batch(microbatch_size=2, drop_last=False)
        .map_batch(transform_batch)
    )

    iterator = iter(pipe)
    try:
        batches = list(iterator)
        total_samples = sum(len(batch.records) for batch in batches)
        assert total_samples >= 2
        for batch in batches:
            for record in batch.records:
                payload = record.payload
                assert isinstance(payload, dict)
                assert payload.get("batch_transformed") is True
                assert "SAMPLE" in payload.get("text", "")
    finally:
        iterator.close()
