# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for accumulator mid-stream flush + checkpoint/restore.

These tests verify that flush sentinels correctly reset accumulator state
at epoch boundaries, ensuring that checkpoint/restore produces identical
output to an uninterrupted run.  The key invariant: after
``flush(reset=True)``, the accumulator is indistinguishable from a freshly
constructed instance, so replaying only the remaining inflight chunks
after eviction produces the same stream.
"""

from typing import Any, Callable

import pytest

from tests._helpers import mk_dataset
from zephon.api.pipeline import Pipeline
from zephon.core.constants import SampleBatch, SampleRecord
from zephon.io.dataset import Dataset
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drain(pipeline: Pipeline) -> list[str]:
    """Drain a pipeline and return text payloads."""
    texts: list[str] = []
    for rec in pipeline:
        if isinstance(rec, SampleBatch):
            for r in rec.records:
                texts.append(r.payload["text"])
        else:
            assert isinstance(rec, SampleRecord)
            texts.append(rec.payload["text"])
    return texts


def _assert_checkpoint_restore(
    make_pipeline: Callable[[], Pipeline],
    *,
    consume_fraction: float = 0.5,
) -> None:
    """Generic checkpoint/restore validator: prefix + suffix must equal baseline."""
    baseline = _drain(make_pipeline())
    assert len(baseline) > 0, "Baseline must produce records"

    # Run 1: consume a fraction of records, checkpoint.
    pipe1 = make_pipeline()
    it = iter(pipe1)
    prefix: list[str] = []
    target = int(len(baseline) * consume_fraction)
    try:
        while len(prefix) < target:
            rec = next(it)
            if isinstance(rec, SampleBatch):
                for r in rec.records:
                    prefix.append(r.payload["text"])
            else:
                assert isinstance(rec, SampleRecord)
                prefix.append(rec.payload["text"])
        ckpt: dict[str, Any] = pipe1.checkpoint()
    finally:
        it.close()

    # Run 2: restore and drain remainder.
    pipe2 = make_pipeline()
    pipe2.restore(ckpt)
    suffix = _drain(pipe2)

    combined = prefix + suffix
    assert combined == baseline, (
        f"prefix + suffix should reconstruct baseline.\n"
        f"  prefix ({len(prefix)}):   {prefix[:10]}...\n"
        f"  suffix ({len(suffix)}):   {suffix[:10]}...\n"
        f"  baseline ({len(baseline)}): {baseline[:10]}..."
    )


# ---------------------------------------------------------------------------
# EnsureMixture: checkpoint/restore divergence from stale SWRR state
# ---------------------------------------------------------------------------


def _make_ensure_mixture_pipeline(
    ds_a: Dataset,
    ds_b: Dataset,
    *,
    flush_k: int = 2,
    chunk_size: int = 4,
) -> Pipeline:
    work = StaticMixtureWorkSource(
        [ds_a, ds_b],
        {ds_a.name: 0.5, ds_b.name: 0.5},
        chunk_size=chunk_size,
        seed=42,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipeline = Pipeline(work)
    pipeline.ensure_mixture(weight_by="samples", max_buffer_size=50)
    pipeline.options(
        deterministic=True,
        max_workers=1,
        default_stage_prefetch=16,
        mtp_mode=False,
        flush_every_k_chunks=flush_k,
    )
    return pipeline


def test_ensure_mixture_checkpoint_restore_determinism() -> None:
    """Checkpoint/restore must produce identical output to an uninterrupted run."""
    ds_a = mk_dataset("compA", {0: 16})
    ds_b = mk_dataset("compB", {0: 16})
    _assert_checkpoint_restore(lambda: _make_ensure_mixture_pipeline(ds_a, ds_b))


# ---------------------------------------------------------------------------
# StatefulTransform: crash on push_many after mid-stream flush
# ---------------------------------------------------------------------------


def _make_stateful_transform_pipeline(
    ds: Dataset,
    *,
    flush_k: int = 2,
    chunk_size: int = 4,
) -> Pipeline:
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=chunk_size,
        seed=42,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipeline = Pipeline(work)

    def push(seen: set, items: list[SampleRecord]) -> tuple[set, list[SampleRecord]]:
        outputs = []
        for item in items:
            text = item.payload["text"]
            if text not in seen:
                seen.add(text)
                outputs.append(item)
        return seen, outputs

    pipeline.stateful_transform(
        "dedup",
        init_state=lambda: set(),
        push=push,
        flush=lambda s: [],
        preserves_cursor_order=False,
    )
    pipeline.options(
        deterministic=True,
        max_workers=1,
        default_stage_prefetch=16,
        mtp_mode=False,
        flush_every_k_chunks=flush_k,
    )
    return pipeline


def test_stateful_transform_survives_flush_sentinel() -> None:
    """Pipeline with StatefulTransform must not crash at flush sentinel boundaries."""
    ds = mk_dataset("stf", {0: 16})
    records = _drain(_make_stateful_transform_pipeline(ds))
    assert len(records) > 0, "Pipeline must produce records"


def test_stateful_transform_checkpoint_restore() -> None:
    """Checkpoint/restore with StatefulTransform must work end-to-end."""
    ds = mk_dataset("stfckpt", {0: 16})
    _assert_checkpoint_restore(
        lambda: _make_stateful_transform_pipeline(ds),
        consume_fraction=1 / 3,
    )


# ---------------------------------------------------------------------------
# Batch(drop_last=True) stalling: checkpoint/restore with epoch eviction
# ---------------------------------------------------------------------------


def _make_batch_stall_pipeline(
    ds_a: Dataset,
    ds_b: Dataset,
    *,
    runner: str = "inline",
    mixture: dict[str, float] | None = None,
    flush_k: int = 2,
    chunk_size: int = 4,
    microbatch_size: int = 3,
) -> Pipeline:
    """Pipeline with non-stalling EnsureMixture upstream of stalling Batch."""
    work = StaticMixtureWorkSource(
        [ds_a, ds_b],
        {ds_a.name: 0.5, ds_b.name: 0.5},
        chunk_size=chunk_size,
        seed=42,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipeline = Pipeline(work)
    mixture_kw: dict[str, Any] = {}
    if mixture is not None:
        mixture_kw["mixture"] = mixture
    pipeline.ensure_mixture(
        weight_by="samples",
        max_buffer_size=50,
        **mixture_kw,
    )
    pipeline.batch(microbatch_size=microbatch_size, drop_last=True)
    pipeline.options(
        deterministic=True,
        runner=runner,
        max_workers=1,
        default_stage_prefetch=16,
        mtp_mode=False,
        flush_every_k_chunks=flush_k,
    )
    return pipeline


def _collect_output(item: object) -> list[str]:
    """Extract text payloads from a SampleBatch or SampleRecord."""
    if isinstance(item, SampleBatch):
        return [r.payload["text"] for r in item.records]
    assert isinstance(item, SampleRecord)
    return [item.payload["text"]]


@pytest.mark.parametrize(
    "runner_kind", ["inline", "threads"], ids=["inline", "threads"]
)
@pytest.mark.parametrize(
    "mixture_label",
    ["skewed", "balanced"],
    ids=["skewed", "balanced"],
)
def test_batch_stall_checkpoint_restore(
    runner_kind: str,
    mixture_label: str,
) -> None:
    """Checkpoint/restore with stalling Batch(drop_last=True).

    Non-monotone upstream (EnsureMixture) + stalling Batch downstream.
    Parametrized over runner kind (inline vs threads) and mixture
    (skewed 90/10 vs balanced 50/50).

    The checkpoint is taken AFTER early-epoch eviction has been observed,
    proving that restore works even when old-epoch chunks are gone.
    """
    ds_a = mk_dataset("bsA", {0: 64})
    ds_b = mk_dataset("bsB", {0: 64})

    mixture = (
        {ds_a.name: 0.9, ds_b.name: 0.1}
        if mixture_label == "skewed"
        else None  # None = use chunk mixture (effectively 50/50)
    )

    def make_pipeline() -> Pipeline:
        return _make_batch_stall_pipeline(
            ds_a, ds_b, runner=runner_kind, mixture=mixture
        )

    # Baseline: uninterrupted run.
    baseline = _drain(make_pipeline())
    assert len(baseline) > 0, "Baseline must produce records"

    # Run 1: consume until eviction is observed, then checkpoint.
    pipe1 = make_pipeline()
    it = iter(pipe1)
    prefix: list[str] = []
    eviction_observed = False
    ckpt: dict[str, Any] | None = None

    try:
        for item in it:
            prefix.extend(_collect_output(item))

            # Check for eviction: chunk 0 no longer in inflight.
            if not eviction_observed and pipe1._engine is not None:
                inflight = pipe1._engine.inflight_chunks_per_lane.get(0, {})
                if inflight and min(inflight) > 0:
                    eviction_observed = True

            # Checkpoint after eviction and consuming enough records.
            if eviction_observed and len(prefix) >= len(baseline) // 3:
                ckpt = pipe1.checkpoint()
                break
    finally:
        it.close()

    assert eviction_observed, (
        f"[{runner_kind}/{mixture_label}] Expected epoch eviction before checkpoint"
    )
    assert ckpt is not None

    # Run 2: restore and drain remainder.
    pipe2 = make_pipeline()
    pipe2.restore(ckpt)
    suffix = _drain(pipe2)

    combined = prefix + suffix
    assert combined == baseline, (
        f"[{runner_kind}/{mixture_label}] prefix + suffix should match baseline.\n"
        f"  prefix ({len(prefix)}):   {prefix[:10]}...\n"
        f"  suffix ({len(suffix)}):   {suffix[:10]}...\n"
        f"  baseline ({len(baseline)}): {baseline[:10]}..."
    )
