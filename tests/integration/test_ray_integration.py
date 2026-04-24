# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for the Ray-backed stage runner.

These exercise :class:`zephon.runners.ray.RemoteStageRunner` through the
public :class:`zephon.api.Pipeline` API (rather than constructing the
runner directly) to validate that a Pipeline configured to use the Ray
runner produces the same output as an equivalent non-Ray Pipeline.

Tests here are marked :pytest:mark:`integration` and :pytest:mark:`requires_ray`
— they are skipped by default and only execute when Ray is installed and
integration tests are enabled (``pytest --run-integration``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_ray,
    pytest.mark.usefixtures("_ray_cluster"),
]


@pytest.fixture(scope="module")
def _ray_cluster():
    """Spin up a small Ray cluster for the module.

    Mirrors the ``ray_init`` fixture in tests/zephon/runners/conftest.py;
    duplicated here so this file can sit under tests/integration/ without
    pulling in the unit-test conftest.
    """
    ray = pytest.importorskip("ray")
    session_scope = os.environ.get("ZEPHON_RAY_SESSION_SCOPE") == "session"
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_cpus=2,
            object_store_memory=100_000_000,
            include_dashboard=False,
            runtime_env={"working_dir": None},
        )
    yield
    if not session_scope and ray.is_initialized():
        ray.shutdown()


def _make_jsonl_dataset(tmp_path: Path, num_samples: int):
    """Create a tiny JSONL dataset used across the tests."""
    from zephon.io import Dataset

    shard = tmp_path / "shard0.jsonl"
    shard.write_text(
        "\n".join(json.dumps({"text": f"sample {i}"}) for i in range(num_samples)),
        encoding="utf-8",
    )
    return Dataset.from_path("ray_demo", str(tmp_path))


def _build_pipeline(dataset, runner: str):
    """Build a multi-op pipeline with the given runner set via options()."""
    from zephon.api import Pipeline as PublicPipeline
    from zephon.work import MixtureSpec, StaticMixtureWorkSource

    work_source = StaticMixtureWorkSource(
        [dataset],
        mixture=MixtureSpec({dataset.name: 1.0}).weights,
        chunk_size=1,
        seed=11,
        shuffle_shards=False,
    )

    def uppercase(payload: dict) -> dict:
        return {"text": payload.get("text", "").upper()}

    return (
        PublicPipeline(work_source)
        .decode_text()
        .map_transform(uppercase)
        .batch(microbatch_size=2, drop_last=False)
        .options(runner=runner)
    )


def _collect_texts(pipe) -> list[str]:
    """Drain a pipeline and flatten the transformed texts into a list."""
    out: list[str] = []
    iterator = iter(pipe)
    try:
        for batch in iterator:
            for record in batch.records:
                payload = record.payload
                assert isinstance(payload, dict)
                out.append(payload["text"])
    finally:
        iterator.close()
    return out


def test_ray_pipeline_produces_same_output_as_threads(tmp_path: Path) -> None:
    """A Pipeline configured with runner='remote' produces the same output
    as the same Pipeline configured with runner='threads'."""
    num_samples = 16
    dataset = _make_jsonl_dataset(tmp_path, num_samples)

    threads_out = _collect_texts(_build_pipeline(dataset, runner="threads"))
    ray_out = _collect_texts(_build_pipeline(dataset, runner="remote"))

    expected = [f"SAMPLE {i}" for i in range(num_samples)]
    assert sorted(threads_out) == sorted(expected)
    assert sorted(ray_out) == sorted(expected)
    # In deterministic mode (the default) the exact order should match too.
    assert ray_out == threads_out


def test_ray_pipeline_multi_op_stage(tmp_path: Path) -> None:
    """A Ray-backed Pipeline with a multi-op stage (decode → map → batch)
    drains cleanly and yields the expected number of records."""
    num_samples = 10
    dataset = _make_jsonl_dataset(tmp_path, num_samples)

    pipe = _build_pipeline(dataset, runner="remote")

    records_seen = 0
    iterator = iter(pipe)
    try:
        for batch in iterator:
            records_seen += len(batch.records)
    finally:
        iterator.close()

    assert records_seen == num_samples
