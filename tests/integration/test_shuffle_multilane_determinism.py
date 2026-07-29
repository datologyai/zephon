# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Integration test: shuffle determinism with multi-lane workers.

When a single rank serves multiple lanes, the engine interleaves elements
from all lanes into a single stream.  The CountingAccumulator must route
elements to per-lane buffers so each shuffle microbatch is lane-pure.

If batches mix lanes, ``batch_seed`` (and therefore shuffle permutation)
depends on the cross-lane interleaving order, breaking determinism on
elastic continuation (where the lane-to-rank mapping changes).

Test strategy
-------------
All three runs use ``canonical_replicas=2`` so the work-source chunk
assignment is identical (lane 0 → even chunks, lane 1 → odd chunks).

- **Run A** (interleaved): ``world_size=1, dp_degree=1`` — one rank owns
  both lanes [0, 1]; the engine round-robins them into a single stream.

- **Run B** (isolated lane 0): ``world_size=2, dp_degree=2, dp_group_id=0``
  — rank 0 owns only lane [0].  No interleaving.

- **Run C** (isolated lane 1): ``world_size=2, dp_degree=2, dp_group_id=1``
  — rank 1 owns only lane [1].  No interleaving.

If the accumulator produces lane-pure batches, Run A's per-lane output
matches Runs B/C.  If it mixes lanes, the shuffle permutations diverge
because ``batch_seed`` sees different record sets.
"""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from typing import Any

import pytest

from zephon import Pipeline as PublicPipeline
from zephon.io import Dataset, InMemoryShard
from zephon.types import SampleRecord
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration

BUFFER_SIZE = 5
SEED = 42
NUM_RECORDS = 20
CHUNK_SIZE = 4


def _make_dataset(name: str, n: int) -> Dataset:
    rows = [{"text": f"{name}-{i}"} for i in range(n)]
    return Dataset.from_dict(name, {0: InMemoryShard(rows)})


def _build_pipe(
    ds: Dataset,
    *,
    canonical_replicas: int = 2,
    world_size: int = 1,
    global_rank: int = 0,
    dp_degree: int = 1,
    dp_group_id: int = 0,
) -> PublicPipeline:
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=CHUNK_SIZE,
        seed=7,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    opts: dict[str, Any] = dict(
        deterministic=True,
        max_workers=1,
        default_stage_prefetch=0,
        mtp_mode=False,
        canonical_replicas=canonical_replicas,
        world_size=world_size,
        global_rank=global_rank,
        dp_degree=dp_degree,
        dp_group_id=dp_group_id,
    )
    if world_size > 1:
        opts["aggregate_dir"] = os.path.join(
            tempfile.gettempdir(), "zephon_shuffle_lane_test_agg"
        )
    return (
        PublicPipeline(work).shuffle(buffer_size=BUFFER_SIZE, seed=SEED).options(**opts)
    )


def _consume_all(pipe: PublicPipeline) -> list[SampleRecord]:
    out: list[SampleRecord] = []
    for rec in pipe:
        assert isinstance(rec, SampleRecord)
        out.append(rec)
    return out


def test_multilane_shuffle_matches_isolated_single_lane_runs() -> None:
    """Per-lane shuffle output is identical whether lanes run together or alone.

    Run A: both lanes on one rank (interleaved by the engine).
    Run B: lane 0 alone on its own rank (world_size=2, dp_group_id=0).
    Run C: lane 1 alone on its own rank (world_size=2, dp_group_id=1).

    All three use canonical_replicas=2 so chunk partitioning is the same.
    If the accumulator is lane-pure, Run A lane 0 == Run B and
    Run A lane 1 == Run C.
    """
    ds = _make_dataset("D", NUM_RECORDS)

    # --- Run A: interleaved (both lanes on 1 rank) ---
    run_a = _consume_all(
        _build_pipe(ds, canonical_replicas=2, world_size=1, dp_degree=1)
    )
    per_lane: dict[int, list[str]] = defaultdict(list)
    for rec in run_a:
        per_lane[rec.meta.lane_id].append(rec.payload["text"])

    assert len(per_lane) == 2, f"Expected 2 lanes, got {sorted(per_lane)}"

    # --- Run B: lane 0 in isolation ---
    run_b = _consume_all(
        _build_pipe(
            ds,
            canonical_replicas=2,
            world_size=2,
            global_rank=0,
            dp_degree=2,
            dp_group_id=0,
        )
    )
    lane0_isolated = [r.payload["text"] for r in run_b]
    assert all(r.meta.lane_id == 0 for r in run_b), "Run B should only have lane 0"

    # --- Run C: lane 1 in isolation ---
    run_c = _consume_all(
        _build_pipe(
            ds,
            canonical_replicas=2,
            world_size=2,
            global_rank=1,
            dp_degree=2,
            dp_group_id=1,
        )
    )
    lane1_isolated = [r.payload["text"] for r in run_c]
    assert all(r.meta.lane_id == 1 for r in run_c), "Run C should only have lane 1"

    # --- Compare ---
    assert per_lane[0] == lane0_isolated, (
        f"Lane 0 shuffle order differs when interleaved vs isolated.\n"
        f"  interleaved: {per_lane[0][:8]}...\n"
        f"  isolated:    {lane0_isolated[:8]}..."
    )
    assert per_lane[1] == lane1_isolated, (
        f"Lane 1 shuffle order differs when interleaved vs isolated.\n"
        f"  interleaved: {per_lane[1][:8]}...\n"
        f"  isolated:    {lane1_isolated[:8]}..."
    )
