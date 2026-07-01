# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Integration test: per-lane flush sentinels with multi-lane engines.

The engine injects one flush sentinel per lane at each lane's epoch
boundary.  A single engine that owns several lanes (``canonical_replicas``
larger than the rank's lane count) routes them all through one accumulator
instance, so sentinel handling must be lane-scoped: lane L's sentinel may
only flush lane L's accumulator state.

If a sentinel flushes *every* lane, the bug stays hidden while lanes run in
lockstep — each lane's sentinel sits adjacent to the others' in the source
stream, so the spurious cross-lane flush coincides with each lane's own
boundary.  It surfaces on **checkpoint/restore**: resume replays only the
inflight chunks, and when the two lanes have evicted a different number of
epochs their inflight suffixes are misaligned.  Lockstep breaks, a lane is
flushed mid-epoch at another lane's re-injected sentinel, the packer
produces different bins (hence different primary cursors), and the
ReplayFilter can no longer reproduce each lane's stream.

The test checkpoints once the two lanes hold inflight suffixes that start at
different chunk ids — exactly the misaligned regime — and asserts that each
lane's delivered samples are identical to an uninterrupted run.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pytest

from tests._helpers import mk_dataset
from zephon.api.pipeline import Pipeline
from zephon.core.constants import SampleBatch, SampleRecord
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration

CANONICAL_REPLICAS = 2


def _add_input_ids(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach a deterministic, variable-length token field for packing."""
    idx = int(payload["text"].rsplit(":", 1)[1])
    payload["input_ids"] = list(range(1 + (idx * 5 + 3) % 13))
    return payload


def _sample_texts(item: object) -> list[tuple[int, str]]:
    """``(lane_id, text)`` for every original sample inside a packed record."""
    records = item.records if isinstance(item, SampleBatch) else [item]
    out: list[tuple[int, str]] = []
    for rec in records:
        assert isinstance(rec, SampleRecord)
        for sample in rec.payload["packed_samples"]:
            out.append((rec.meta.lane_id, sample["text"]))
    return out


def _per_lane(pairs: list[tuple[int, str]]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = defaultdict(list)
    for lane, text in pairs:
        out[lane].append(text)
    return out


def _drain(pipeline: Pipeline) -> list[tuple[int, str]]:
    pairs: list[tuple[int, str]] = []
    for item in pipeline:
        pairs.extend(_sample_texts(item))
    return pairs


def _make_pipeline(ds_a: Any, ds_b: Any, runner: str) -> Pipeline:
    work = StaticMixtureWorkSource(
        [ds_a, ds_b],
        {ds_a.name: 0.5, ds_b.name: 0.5},
        chunk_size=4,
        seed=42,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipeline = Pipeline(work)
    pipeline.map_transform(_add_input_ids)
    pipeline.pack_sequences(
        max_length=16, num_bins=2, algorithm="first_fit", tokens_field="input_ids"
    )
    pipeline.options(
        deterministic=True,
        runner=runner,
        max_workers=1,
        default_stage_prefetch=2,
        mtp_mode=False,
        flush_every_k_chunks=2,
        canonical_replicas=CANONICAL_REPLICAS,
    )
    return pipeline


def _inflight_lows(pipe: Pipeline) -> dict[int, int]:
    """Lowest inflight chunk id per lane (only lanes that have any inflight).

    Snapshots each lane's chunk keys with ``list()`` so a concurrent pump
    mutation (threads runner) cannot raise mid-iteration.
    """
    engine = pipe._engine
    assert engine is not None
    lows: dict[int, int] = {}
    for lane, chunks in list(engine.inflight_chunks_per_lane.items()):
        try:
            keys = list(chunks)
        except RuntimeError:
            continue  # mutated mid-iteration; try again on the next record
        if keys:
            lows[lane] = min(keys)
    return lows


@pytest.mark.parametrize("runner", ["inline", "threads"])
def test_pack_multilane_checkpoint_restore_with_misaligned_epochs(runner: str) -> None:
    ds_a = mk_dataset("packA", {0: 80})
    ds_b = mk_dataset("packB", {0: 80})

    baseline = _per_lane(_drain(_make_pipeline(ds_a, ds_b, runner)))
    assert len(baseline) == CANONICAL_REPLICAS
    assert all(texts for texts in baseline.values())

    pipe = _make_pipeline(ds_a, ds_b, runner)
    it = iter(pipe)
    prefix: list[tuple[int, str]] = []
    ckpt: dict[str, Any] | None = None

    try:
        for item in it:
            prefix.extend(_sample_texts(item))
            lows = _inflight_lows(pipe)
            # Checkpoint once the lanes' inflight suffixes are misaligned: one
            # lane has evicted an epoch the other has not, so replay is no
            # longer lockstep.
            if len(lows) == CANONICAL_REPLICAS and len(set(lows.values())) > 1:
                ckpt = pipe.checkpoint()
                break
    finally:
        it.close()

    assert ckpt is not None, "never reached misaligned inflight — cannot probe the bug"

    pipe2 = _make_pipeline(ds_a, ds_b, runner)
    pipe2.restore(ckpt)
    suffix = _drain(pipe2)

    combined = _per_lane(prefix + suffix)
    for lane in baseline:
        assert combined[lane] == baseline[lane], (
            f"[{runner}] lane {lane}: checkpoint/restore diverged from baseline.\n"
            f"  baseline ({len(baseline[lane])}): {baseline[lane][:12]}\n"
            f"  combined ({len(combined[lane])}): {combined[lane][:12]}"
        )
