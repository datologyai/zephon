# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for StatefulTransform operator with parallel workers."""

import json
from pathlib import Path

import pytest

from zephon.api import Pipeline as PublicPipeline
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("runner_kind", ["threads", "process"])
def test_stateful_transform_parallel_dedup_encode(
    tmp_path: Path, runner_kind: str
) -> None:
    """Test stateful_transform with parallelism=4 for dedup + batch encoding.

    This test verifies that:
    1. The accumulator (dedup) runs serially on the pump thread
    2. The transform_fn runs in parallel workers (parallelism=4)
    3. Results are correct across both thread and process runners
    """
    # Create dataset with some duplicates
    shard0 = tmp_path / "shard0.jsonl"
    # 20 unique items, each appearing twice = 40 records total
    records = []
    for _ in range(2):  # Two passes to create duplicates
        for i in range(20):
            records.append(json.dumps({"id": i, "text": f"sample {i}"}))
    shard0.write_text("\n".join(records), encoding="utf-8")

    dataset = Dataset.from_path("demo", str(tmp_path))

    # Dedup state: track seen IDs
    def init_state() -> set:
        return set()

    def push_dedup(state: set, items: list) -> tuple[set, list]:
        output = []
        for item in items:
            payload = item.payload
            item_id = payload.get("id") if isinstance(payload, dict) else None
            if item_id is not None and item_id not in state:
                state.add(item_id)
                output.append(item)
        return state, output

    # Transform: simulate encoding by uppercasing text
    def encode_transform(batch: list) -> list:
        for item in batch:
            if isinstance(item.payload, dict):
                item.payload["text"] = item.payload["text"].upper()
                item.payload["encoded"] = True
        return batch

    work_source = StaticMixtureWorkSource(
        [dataset],
        mixture=MixtureSpec({dataset.name: 1.0}).weights,
        chunk_size=8,
        seed=42,
        shuffle_shards=False,
    )

    pipe = (
        PublicPipeline(work_source)
        .decode_text()
        .stateful_transform(
            name="dedup_encode",
            init_state=init_state,
            push=push_dedup,
            transform=encode_transform,
            parallelism=4,
        )
        .batch(microbatch_size=10, drop_last=False)
        .options(runner=runner_kind, max_workers=4, deterministic=True)
    )

    iterator = iter(pipe)
    try:
        all_records = []
        for batch in iterator:
            all_records.extend(batch.records)

        # Should have exactly 20 unique items (40 - 20 duplicates)
        assert len(all_records) == 20, f"Expected 20, got {len(all_records)}"

        # Verify dedup worked: each ID should appear only once
        seen_ids = set()
        for record in all_records:
            payload = record.payload
            assert isinstance(payload, dict)
            item_id = payload.get("id")
            assert item_id not in seen_ids, f"Duplicate ID found: {item_id}"
            seen_ids.add(item_id)

        # Verify transform was applied
        for record in all_records:
            payload = record.payload
            assert payload.get("encoded") is True
            assert payload.get("text", "").isupper()

    finally:
        iterator.close()


@pytest.mark.parametrize("runner_kind", ["threads", "process"])
def test_stateful_transform_parallel_batching(tmp_path: Path, runner_kind: str) -> None:
    """Test stateful_transform with accumulator that buffers items.

    Uses accumulator to buffer items until we have 5, then emits them.
    Verifies flush emits remaining buffered items at end-of-stream.
    """
    shard0 = tmp_path / "shard0.jsonl"
    # 32 records - not divisible by 5, so flush will emit remainder
    records = [json.dumps({"idx": i, "value": i * 10}) for i in range(32)]
    shard0.write_text("\n".join(records), encoding="utf-8")

    dataset = Dataset.from_path("demo", str(tmp_path))

    # Accumulator: buffer until we have 5 items
    def init_state() -> dict:
        return {"buffer": []}

    def push_buffered(state: dict, items: list) -> tuple[dict, list]:
        buffer = state["buffer"] + items
        outputs = []
        while len(buffer) >= 5:
            outputs.extend(buffer[:5])
            buffer = buffer[5:]
        return {"buffer": buffer}, outputs

    def flush_buffer(state: dict) -> list:
        return state["buffer"]

    # Transform: mark items as processed
    def add_metadata(batch: list) -> list:
        for item in batch:
            if isinstance(item.payload, dict):
                item.payload["processed"] = True
                item.payload["value_doubled"] = item.payload.get("value", 0) * 2
        return batch

    work_source = StaticMixtureWorkSource(
        [dataset],
        mixture=MixtureSpec({dataset.name: 1.0}).weights,
        chunk_size=4,
        seed=42,
        shuffle_shards=False,
    )

    pipe = (
        PublicPipeline(work_source)
        .decode_text()
        .stateful_transform(
            name="batch_transform",
            init_state=init_state,
            push=push_buffered,
            flush=flush_buffer,
            transform=add_metadata,
            parallelism=4,
        )
        .batch(microbatch_size=10, drop_last=False)
        .options(runner=runner_kind, max_workers=4, deterministic=True)
    )

    iterator = iter(pipe)
    try:
        all_records = []
        for batch in iterator:
            all_records.extend(batch.records)

        # Should have all 32 items (30 from push batches + 2 from flush)
        assert len(all_records) == 32, f"Expected 32, got {len(all_records)}"

        # Verify transform was applied to all
        for record in all_records:
            payload = record.payload
            assert isinstance(payload, dict)
            assert payload.get("processed") is True
            expected_doubled = payload.get("idx", 0) * 10 * 2
            assert payload.get("value_doubled") == expected_doubled

    finally:
        iterator.close()


@pytest.mark.parametrize("runner_kind", ["threads", "process"])
def test_stateful_transform_parallel_with_lambda(
    tmp_path: Path, runner_kind: str
) -> None:
    """Test that lambda functions work with stateful_transform in process runner.

    This verifies cloudpickle correctly serializes lambdas for worker processes.
    """
    shard0 = tmp_path / "shard0.jsonl"
    records = [json.dumps({"x": i}) for i in range(20)]
    shard0.write_text("\n".join(records), encoding="utf-8")

    dataset = Dataset.from_path("demo", str(tmp_path))

    work_source = StaticMixtureWorkSource(
        [dataset],
        mixture=MixtureSpec({dataset.name: 1.0}).weights,
        chunk_size=4,
        seed=42,
        shuffle_shards=False,
    )

    # Helper to add squared field (lambdas can't have statements)
    def add_squared(r):
        r.payload["squared"] = r.payload["x"] ** 2
        return r

    # Use lambdas everywhere - tests cloudpickle serialization
    pipe = (
        PublicPipeline(work_source)
        .decode_text()
        .stateful_transform(
            name="counter_square",
            init_state=lambda: 0,
            push=lambda count, items: (count + len(items), items),
            transform=lambda batch: [add_squared(r) for r in batch],
            parallelism=4,
        )
        .batch(microbatch_size=10, drop_last=False)
        .options(runner=runner_kind, max_workers=4, deterministic=True)
    )

    iterator = iter(pipe)
    try:
        all_records = []
        for batch in iterator:
            all_records.extend(batch.records)

        assert len(all_records) == 20

        # Verify transform was applied
        for record in all_records:
            payload = record.payload
            assert isinstance(payload, dict)
            x = payload.get("x")
            assert payload.get("squared") == x * x

    finally:
        iterator.close()
