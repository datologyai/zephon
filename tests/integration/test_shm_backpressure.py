# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Integration test: SHM backpressure retry through a full zephon pipeline.

Verifies that transient ``/dev/shm`` exhaustion (``ENOSPC``) during IPC queue
serialization does **not** crash the pipeline.  Instead, the
``NamedQueue._on_queue_feeder_error`` retry loop re-queues the item and
the pipeline delivers all records.

Strategy
--------
A custom ``_ShmPressureValue`` class defines ``__reduce__`` so that pickling
raises ``OSError(ENOSPC)`` for the first *fail_count* attempts, then succeeds
by reducing to a plain ``int``.  A ``map_transform`` Op injects these objects
into every Nth record's payload.  Because ``__reduce__`` is part of the pickle
protocol, it fires inside the real ``Queue._feed`` thread — no mock needed for
the failure path.

The pipeline uses ``mp_context="fork"`` so that monkey-patched retry constants
(near-zero backoff) are inherited by worker processes, keeping the test fast.
"""

from __future__ import annotations

import errno
import json
import multiprocessing as mp
from collections.abc import Callable
from pathlib import Path

import pytest

import zephon.runners.queue as _queue_mod
from zephon.api import Pipeline
from zephon.core.constants import SamplePayload, SampleRecord
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Simulated SHM pressure via __reduce__
# ---------------------------------------------------------------------------


class _ShmPressureValue:
    """A value whose pickling raises ENOSPC for the first *fail_count* attempts.

    On eventual success, ``__reduce__`` returns ``(int, (self.value,))`` so
    the deserialized result is a plain ``int`` — no cross-process class
    lookup issues.
    """

    __slots__ = ("value", "fail_count", "_attempts")

    def __init__(self, value: int, fail_count: int = 0):
        self.value = value
        self.fail_count = fail_count
        self._attempts = 0

    def __reduce__(self) -> tuple:
        self._attempts += 1
        if self._attempts <= self.fail_count:
            raise OSError(errno.ENOSPC, "No space left on device")
        return (int, (self.value,))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_retry():
    """Speed up SHM retry constants so the test doesn't sleep for seconds.

    Uses ``fork`` mp_context so workers inherit these patches.
    """
    orig_base = _queue_mod._SHM_RETRY_BASE_BACKOFF
    orig_max = _queue_mod._SHM_RETRY_MAX_BACKOFF
    orig_jitter = _queue_mod._SHM_RETRY_MAX_JITTER

    _queue_mod._SHM_RETRY_BASE_BACKOFF = 0.001
    _queue_mod._SHM_RETRY_MAX_BACKOFF = 0.01
    _queue_mod._SHM_RETRY_MAX_JITTER = 0

    yield

    _queue_mod._SHM_RETRY_BASE_BACKOFF = orig_base
    _queue_mod._SHM_RETRY_MAX_BACKOFF = orig_max
    _queue_mod._SHM_RETRY_MAX_JITTER = orig_jitter


def _create_dataset(tmp_path: Path, n_records: int = 40) -> Dataset:
    """Write a simple JSONL shard and return a Dataset."""
    shard = tmp_path / "shard0.jsonl"
    lines = [json.dumps({"id": i, "text": f"sample {i}"}) for i in range(n_records)]
    shard.write_text("\n".join(lines), encoding="utf-8")
    return Dataset.from_path("pressure_test", str(tmp_path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _run_pipeline(
    tmp_path: Path,
    n_records: int,
    transform_fn: Callable[[SamplePayload], SamplePayload],
    max_workers: int = 2,
) -> list[SampleRecord]:
    """Build and run a pipeline, returning all collected SampleRecords."""
    dataset = _create_dataset(tmp_path, n_records)
    work = StaticMixtureWorkSource(
        [dataset],
        mixture=MixtureSpec({dataset.name: 1.0}).weights,
        chunk_size=n_records,  # single chunk so all records are emitted
        seed=42,
        shuffle_shards=False,
    )
    pipe = (
        Pipeline(work)
        .decode_text()
        .map_transform(transform_fn)
        .options(
            runner="process",
            max_workers=max_workers,
            deterministic=True,
            mp_context=mp.get_context("fork"),
        )
    )
    iterator = iter(pipe)
    try:
        return [item for item in iterator if isinstance(item, SampleRecord)]
    finally:
        iterator.close()


def test_pipeline_survives_transient_shm_pressure(tmp_path: Path) -> None:
    """Full pipeline with transient SHM pressure — all records delivered."""
    n_records = 40

    def inject_pressure(payload: SamplePayload) -> SamplePayload:
        """Inject _ShmPressureValue into every 5th record's payload."""
        assert isinstance(payload, dict)
        item_id = payload["id"]
        if isinstance(item_id, int) and item_id % 5 == 0:
            payload["pressure"] = _ShmPressureValue(item_id, fail_count=1)
        return payload

    # Baseline: run without pressure to get the natural record count.
    baseline = _run_pipeline(tmp_path, n_records, lambda p: p)

    # With pressure: should deliver the same number of records.
    collected = _run_pipeline(tmp_path, n_records, inject_pressure)

    assert len(collected) == len(baseline), (
        f"SHM pressure caused record loss: got {len(collected)}, "
        f"baseline (no pressure) was {len(baseline)}."
    )

    # Every record with a "pressure" field should have it deserialized as int.
    pressure_ids = []
    for record in collected:
        payload = record.payload
        assert isinstance(payload, dict)
        if "pressure" in payload:
            assert isinstance(payload["pressure"], int), (
                f"Expected int after deserialization, got {type(payload['pressure'])}"
            )
            pressure_ids.append(payload["pressure"])

    # At least some pressure items should have come through.
    assert len(pressure_ids) > 0, "No pressure items found — test may be misconfigured."


def test_pipeline_mixed_pressure_intensities(tmp_path: Path) -> None:
    """Records with varying failure counts — all eventually delivered."""
    n_records = 30

    def inject_varying_pressure(payload: SamplePayload) -> SamplePayload:
        """Inject pressure with varying failure counts."""
        assert isinstance(payload, dict)
        item_id = payload["id"]
        if isinstance(item_id, int):
            if item_id % 10 == 0:
                payload["pressure"] = _ShmPressureValue(item_id, fail_count=3)
            elif item_id % 5 == 0:
                payload["pressure"] = _ShmPressureValue(item_id, fail_count=1)
        return payload

    baseline = _run_pipeline(tmp_path, n_records, lambda p: p)
    collected = _run_pipeline(tmp_path, n_records, inject_varying_pressure)

    assert len(collected) == len(baseline), (
        f"SHM pressure caused record loss: got {len(collected)}, "
        f"baseline (no pressure) was {len(baseline)}."
    )

    # Verify all IDs present
    ids = {r.payload["id"] for r in collected if isinstance(r.payload, dict)}
    baseline_ids = {r.payload["id"] for r in baseline if isinstance(r.payload, dict)}
    assert ids == baseline_ids
