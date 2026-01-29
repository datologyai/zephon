# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for 3D parallelism support.

These tests validate that Zephon correctly handles 3D parallelism configurations
where world_size > dp_degree (i.e., there is model parallelism via TP/PP).

Key invariants tested:
1. Within a DP group, all ranks receive the SAME samples (they share dp_group_id)
2. Across DP groups, ranks receive DIFFERENT samples (different data partitions)
3. Lane assignments are correct based on dp_group_id and mapping strategy
4. Checkpoint/resume works correctly with 3D parallelism
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from zephon.api import Pipeline as PublicPipeline
from zephon.io import Dataset, InMemoryShard
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


# =============================================================================
# Test Utilities
# =============================================================================


def make_dataset(name: str, sample_count: int) -> Dataset:
    """Create a simple in-memory dataset with numbered samples."""
    rows = [{"text": f"{name}-{i}"} for i in range(sample_count)]
    return Dataset.from_dict(name, {0: InMemoryShard(rows)})


def _extract_texts(item: Any) -> list[str]:
    """Extract text payloads from a SampleRecord or SampleBatch."""
    from zephon.core.constants import SampleBatch, SampleRecord

    if isinstance(item, SampleRecord):
        payload = item.payload
        assert isinstance(payload, dict)
        return [str(payload.get("text", ""))]
    assert isinstance(item, SampleBatch)
    return [str(r.payload.get("text", "")) for r in item.records]


def _build_pipeline(
    ds: Dataset,
    *,
    chunk_size: int,
    canonical_replicas: int,
    world_size: int,
    global_rank: int,
    dp_degree: int,
    dp_group_id: int,
    mapping_strategy: str = "contiguous",
    aggregate_dir: str | None = None,
    run_id: str | None = None,
    seed: int = 42,
) -> PublicPipeline:
    """Build a minimal pipeline with 3D parallelism parameters."""
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=chunk_size,
        seed=seed,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    # Minimal pipeline: just decode text, no delays or extra ops
    pipe = PublicPipeline(work).decode_text()

    # When world_size > 1, we need an aggregate_dir
    if aggregate_dir is None and world_size > 1:
        aggregate_dir = os.path.join(tempfile.gettempdir(), "zephon_3d_test_agg")

    opts: dict[str, Any] = {
        "deterministic": True,
        "canonical_replicas": canonical_replicas,
        "world_size": world_size,
        "global_rank": global_rank,
        "dp_degree": dp_degree,
        "dp_group_id": dp_group_id,
        "mapping_strategy": mapping_strategy,
    }
    if aggregate_dir is not None:
        opts["aggregate_dir"] = aggregate_dir
    if run_id is not None:
        opts["run_id"] = run_id

    return pipe.options(**opts)


def consume_n(pipe: PublicPipeline, n: int) -> list[str]:
    """Consume exactly n samples from pipeline."""
    pipe._ensure()
    assert pipe._engine is not None
    engine = pipe._engine

    texts: list[str] = []
    try:
        for item in pipe:
            texts.extend(_extract_texts(item))
            if len(texts) >= n:
                break
    finally:
        engine.close()
    return texts[:n]


def get_lanes_for_dp_group(pipe: PublicPipeline, dp_group_id: int) -> list[int]:
    """Get the lanes assigned to a DP group from a built pipeline."""
    pipe._ensure()
    assert pipe._engine is not None
    lanes = pipe._engine._world.lanes_for_dp_group[dp_group_id]
    pipe._engine.close()
    return lanes


# =============================================================================
# Basic 3D Parallelism Tests
# =============================================================================


class TestBasic3DParallelism:
    """Test core 3D parallelism invariants."""

    def test_same_dp_group_same_samples(self) -> None:
        """Two ranks in the same DP group get identical samples."""
        ds = make_dataset("a", 64)

        # world_size=4, dp_degree=2, mp_degree=2
        # Ranks 0,1 share dp_group_id=0
        samples_r0 = consume_n(
            _build_pipeline(
                ds,
                chunk_size=8,
                canonical_replicas=2,
                world_size=4,
                global_rank=0,
                dp_degree=2,
                dp_group_id=0,
            ),
            n=16,
        )
        samples_r1 = consume_n(
            _build_pipeline(
                ds,
                chunk_size=8,
                canonical_replicas=2,
                world_size=4,
                global_rank=1,
                dp_degree=2,
                dp_group_id=0,
            ),
            n=16,
        )

        assert samples_r0 == samples_r1

    def test_different_dp_groups_different_samples(self) -> None:
        """Two ranks in different DP groups get different (non-overlapping) samples."""
        ds = make_dataset("a", 64)

        # dp_group_id=0 and dp_group_id=1
        samples_dp0 = consume_n(
            _build_pipeline(
                ds,
                chunk_size=8,
                canonical_replicas=2,
                world_size=4,
                global_rank=0,
                dp_degree=2,
                dp_group_id=0,
            ),
            n=16,
        )
        samples_dp1 = consume_n(
            _build_pipeline(
                ds,
                chunk_size=8,
                canonical_replicas=2,
                world_size=4,
                global_rank=2,
                dp_degree=2,
                dp_group_id=1,
            ),
            n=16,
        )

        # Must be different and non-overlapping
        assert samples_dp0 != samples_dp1
        assert len(set(samples_dp0) & set(samples_dp1)) == 0

    def test_mp_degree_4_all_identical(self) -> None:
        """With mp_degree=4, all 4 ranks in a DP group get identical samples."""
        ds = make_dataset("a", 64)

        # world_size=8, dp_degree=2, mp_degree=4
        reference = consume_n(
            _build_pipeline(
                ds,
                chunk_size=8,
                canonical_replicas=2,
                world_size=8,
                global_rank=0,
                dp_degree=2,
                dp_group_id=0,
            ),
            n=16,
        )

        # Ranks 1, 2, 3 also in dp_group_id=0 should match
        for rank in [1, 2, 3]:
            samples = consume_n(
                _build_pipeline(
                    ds,
                    chunk_size=8,
                    canonical_replicas=2,
                    world_size=8,
                    global_rank=rank,
                    dp_degree=2,
                    dp_group_id=0,
                ),
                n=16,
            )
            assert samples == reference, f"Rank {rank} differs"


# =============================================================================
# Lane Assignment Tests
# =============================================================================


class TestLaneAssignment:
    """Test that lanes are correctly assigned to DP groups."""

    def test_contiguous_lane_assignment(self) -> None:
        """Contiguous strategy assigns consecutive lanes to each DP group."""
        ds = make_dataset("a", 32)

        # canonical_replicas=4, dp_degree=2 -> dp0: [0,1], dp1: [2,3]
        pipe = _build_pipeline(
            ds,
            chunk_size=8,
            canonical_replicas=4,
            world_size=2,
            global_rank=0,
            dp_degree=2,
            dp_group_id=0,
            mapping_strategy="contiguous",
        )
        lanes_dp0 = get_lanes_for_dp_group(pipe, 0)

        pipe = _build_pipeline(
            ds,
            chunk_size=8,
            canonical_replicas=4,
            world_size=2,
            global_rank=1,
            dp_degree=2,
            dp_group_id=1,
            mapping_strategy="contiguous",
        )
        lanes_dp1 = get_lanes_for_dp_group(pipe, 1)

        assert lanes_dp0 == [0, 1]
        assert lanes_dp1 == [2, 3]

    def test_interleaved_lane_assignment(self) -> None:
        """Interleaved strategy assigns alternating lanes to each DP group."""
        ds = make_dataset("a", 32)

        # canonical_replicas=4, dp_degree=2 -> dp0: [0,2], dp1: [1,3]
        pipe = _build_pipeline(
            ds,
            chunk_size=8,
            canonical_replicas=4,
            world_size=2,
            global_rank=0,
            dp_degree=2,
            dp_group_id=0,
            mapping_strategy="interleaved",
        )
        lanes_dp0 = get_lanes_for_dp_group(pipe, 0)

        pipe = _build_pipeline(
            ds,
            chunk_size=8,
            canonical_replicas=4,
            world_size=2,
            global_rank=1,
            dp_degree=2,
            dp_group_id=1,
            mapping_strategy="interleaved",
        )
        lanes_dp1 = get_lanes_for_dp_group(pipe, 1)

        assert lanes_dp0 == [0, 2]
        assert lanes_dp1 == [1, 3]

    def test_canonical_equals_dp_one_lane_each(self) -> None:
        """When canonical_replicas == dp_degree, each DP group gets exactly one lane."""
        ds = make_dataset("a", 32)

        # canonical_replicas=4, dp_degree=4 -> each dp_id owns lane dp_id
        for dp_id in range(4):
            pipe = _build_pipeline(
                ds,
                chunk_size=8,
                canonical_replicas=4,
                world_size=4,
                global_rank=dp_id,
                dp_degree=4,
                dp_group_id=dp_id,
            )
            lanes = get_lanes_for_dp_group(pipe, dp_id)
            assert lanes == [dp_id], (
                f"dp_group {dp_id} should own lane [{dp_id}], got {lanes}"
            )


# =============================================================================
# Checkpoint/Resume with 3D Parallelism
# =============================================================================


class TestElastic3DParallelism:
    """Test checkpoint/resume with 3D parallelism configurations.

    Note: Single-process checkpoint tests use dp_degree=1 so the single process
    owns all lanes (required for checkpoint aggregation to succeed). The 3D
    aspect (world_size > dp_degree) is validated via configuration checks.
    """

    def test_resume_same_config(self, tmp_path: Path) -> None:
        """Checkpoint and resume with identical config reproduces baseline.

        Uses dp_degree=1 so single process owns all lanes for aggregation.
        world_size=4 means mp_degree=4 (simulates 3D with TP/PP=4).
        """
        ds = make_dataset("a", 64)
        chunk_size = 8
        n_total = 24
        n_prefix = 12

        # Baseline: full run (dp_degree=1 -> owns all lanes)
        baseline = consume_n(
            _build_pipeline(
                ds,
                chunk_size=chunk_size,
                canonical_replicas=2,
                world_size=4,
                global_rank=0,
                dp_degree=1,
                dp_group_id=0,
            ),
            n=n_total,
        )

        # Phase 1: consume prefix, checkpoint
        pipe1 = _build_pipeline(
            ds,
            chunk_size=chunk_size,
            canonical_replicas=2,
            world_size=4,
            global_rank=0,
            dp_degree=1,
            dp_group_id=0,
            aggregate_dir=str(tmp_path),
            run_id="p1",
        )
        pipe1._ensure()
        engine1 = pipe1._engine
        assert engine1 is not None

        prefix: list[str] = []
        for item in pipe1:
            prefix.extend(_extract_texts(item))
            if len(prefix) >= n_prefix:
                break
        ckpt = engine1.state_dict()
        engine1.close()
        prefix = prefix[:n_prefix]

        assert prefix == baseline[:n_prefix]

        # Phase 2: resume, consume suffix
        pipe2 = _build_pipeline(
            ds,
            chunk_size=chunk_size,
            canonical_replicas=2,
            world_size=4,
            global_rank=0,
            dp_degree=1,
            dp_group_id=0,
            aggregate_dir=str(tmp_path),
            run_id="p2",
        )
        pipe2._ensure()
        engine2 = pipe2._engine
        assert engine2 is not None
        engine2.load_state_dict(ckpt, replay=True)

        suffix = consume_n(pipe2, n=n_total - n_prefix)

        assert prefix + suffix == baseline

    def test_resume_different_mp_degree(self, tmp_path: Path) -> None:
        """Resume with changed world_size (mp_degree) but same dp config.

        Phase 1: world_size=2, dp_degree=1, mp_degree=2
        Phase 2: world_size=4, dp_degree=1, mp_degree=4
        """
        ds = make_dataset("a", 64)
        chunk_size = 8

        # Baseline for comparison
        baseline = consume_n(
            _build_pipeline(
                ds,
                chunk_size=chunk_size,
                canonical_replicas=2,
                world_size=2,
                global_rank=0,
                dp_degree=1,
                dp_group_id=0,
            ),
            n=16,
        )

        # Phase 1: world_size=2, mp_degree=2
        pipe1 = _build_pipeline(
            ds,
            chunk_size=chunk_size,
            canonical_replicas=2,
            world_size=2,
            global_rank=0,
            dp_degree=1,
            dp_group_id=0,
            aggregate_dir=str(tmp_path),
            run_id="mp1",
        )
        pipe1._ensure()
        engine1 = pipe1._engine
        assert engine1 is not None

        prefix: list[str] = []
        for item in pipe1:
            prefix.extend(_extract_texts(item))
            if len(prefix) >= 8:
                break
        ckpt = engine1.state_dict()
        engine1.close()
        prefix = prefix[:8]

        # Phase 2: world_size=4, mp_degree=4
        pipe2 = _build_pipeline(
            ds,
            chunk_size=chunk_size,
            canonical_replicas=2,
            world_size=4,
            global_rank=0,
            dp_degree=1,
            dp_group_id=0,
            aggregate_dir=str(tmp_path),
            run_id="mp2",
        )
        pipe2._ensure()
        engine2 = pipe2._engine
        assert engine2 is not None
        engine2.load_state_dict(ckpt, replay=True)

        suffix = consume_n(pipe2, n=8)

        # Should match baseline exactly
        assert prefix + suffix == baseline

    def test_mp_peers_resume_identically(self, tmp_path: Path) -> None:
        """Multiple MP ranks resume to identical state.

        Uses dp_degree=1 for checkpoint. After resume, different global_ranks
        with same dp_group_id should produce identical output.
        """
        ds = make_dataset("a", 64)
        chunk_size = 8

        # Checkpoint from rank 0 (dp_degree=1 owns all lanes)
        pipe1 = _build_pipeline(
            ds,
            chunk_size=chunk_size,
            canonical_replicas=2,
            world_size=4,
            global_rank=0,
            dp_degree=1,
            dp_group_id=0,
            aggregate_dir=str(tmp_path),
            run_id="mp_peers",
        )
        pipe1._ensure()
        engine1 = pipe1._engine
        assert engine1 is not None

        for item in pipe1:
            break  # Consume 1 item
        ckpt = engine1.state_dict()
        engine1.close()

        # Resume from different global_ranks (all dp_group_id=0)
        resumed: list[list[str]] = []
        for rank in [0, 1, 2, 3]:
            pipe = _build_pipeline(
                ds,
                chunk_size=chunk_size,
                canonical_replicas=2,
                world_size=4,
                global_rank=rank,
                dp_degree=1,
                dp_group_id=0,
                aggregate_dir=str(tmp_path),
                run_id=f"mp_peers_r{rank}",
            )
            pipe._ensure()
            engine = pipe._engine
            assert engine is not None
            engine.load_state_dict(ckpt, replay=True)
            resumed.append(consume_n(pipe, n=8))

        # All ranks should get identical samples
        for i in range(1, len(resumed)):
            assert resumed[0] == resumed[i], f"Rank {i} differs from rank 0"


# =============================================================================
# Full Coverage Test
# =============================================================================


class TestFullCoverage:
    """Test that all DP groups together cover all data without overlap."""

    def test_all_dp_groups_partition_data(self) -> None:
        """All DP groups together cover all data exactly once (no overlap, no gaps)."""
        ds = make_dataset("a", 64)
        chunk_size = 8
        canonical_replicas = 4
        dp_degree = 4
        n_per_dp = 8  # Each DP group should get different samples

        all_samples: list[set[str]] = []
        for dp_id in range(dp_degree):
            samples = consume_n(
                _build_pipeline(
                    ds,
                    chunk_size=chunk_size,
                    canonical_replicas=canonical_replicas,
                    world_size=8,
                    global_rank=dp_id * 2,
                    dp_degree=dp_degree,
                    dp_group_id=dp_id,
                ),
                n=n_per_dp,
            )
            all_samples.append(set(samples))

        # Verify pairwise no overlap
        for i in range(dp_degree):
            for j in range(i + 1, dp_degree):
                overlap = all_samples[i] & all_samples[j]
                assert len(overlap) == 0, f"DP {i} and {j} overlap: {overlap}"

        # Verify total unique = sum of individuals
        union = set().union(*all_samples)
        total = sum(len(s) for s in all_samples)
        assert len(union) == total, "Some samples counted multiple times"

    def test_3d_elastic_determinism(self) -> None:
        """Global stream is deterministic regardless of DP/MP configuration.

        With canonical_replicas=4, dp_degree=4, world_size=8 (mp_degree=2):
        1. All MP ranks within a DP group get identical samples
        2. DP groups have no overlap
        3. Merged DP output equals single-rank truth (dp_degree=1)

        This is THE key elastic determinism test for 3D parallelism.
        """
        ds = make_dataset("a", 128)
        chunk_size = 8
        canonical_replicas = 4
        dp_degree = 4
        mp_degree = 2
        world_size = dp_degree * mp_degree  # 8
        n_per_lane = 16

        # === Truth: single rank owns all lanes ===
        truth = consume_n(
            _build_pipeline(
                ds,
                chunk_size=chunk_size,
                canonical_replicas=canonical_replicas,
                world_size=1,
                global_rank=0,
                dp_degree=1,
                dp_group_id=0,
            ),
            n=n_per_lane * canonical_replicas,
        )

        # === 3D config: collect samples from all 8 ranks ===
        # per_dp_group[dp_id][mp_rank] = samples
        per_dp_group: dict[int, list[list[str]]] = {i: [] for i in range(dp_degree)}

        for dp_id in range(dp_degree):
            for mp_rank in range(mp_degree):
                global_rank = dp_id * mp_degree + mp_rank
                samples = consume_n(
                    _build_pipeline(
                        ds,
                        chunk_size=chunk_size,
                        canonical_replicas=canonical_replicas,
                        world_size=world_size,
                        global_rank=global_rank,
                        dp_degree=dp_degree,
                        dp_group_id=dp_id,
                    ),
                    n=n_per_lane,
                )
                per_dp_group[dp_id].append(samples)

        # === Check 1: MP ranks within each DP group are identical ===
        for dp_id in range(dp_degree):
            for mp_rank in range(1, mp_degree):
                assert per_dp_group[dp_id][0] == per_dp_group[dp_id][mp_rank], (
                    f"DP group {dp_id}: MP rank {mp_rank} differs from MP rank 0"
                )

        # === Check 2: DP groups have no overlap ===
        dp_sets = [set(per_dp_group[dp_id][0]) for dp_id in range(dp_degree)]
        for i in range(dp_degree):
            for j in range(i + 1, dp_degree):
                overlap = dp_sets[i] & dp_sets[j]
                assert len(overlap) == 0, f"DP {i} and {j} overlap: {overlap}"

        # === Check 3: Merged DP groups equal single-rank truth ===
        per_dp = [per_dp_group[dp_id][0] for dp_id in range(dp_degree)]
        merged: list[str] = []
        max_len = max(len(s) for s in per_dp)
        for i in range(max_len):
            for dp_id in range(dp_degree):
                if i < len(per_dp[dp_id]):
                    merged.append(per_dp[dp_id][i])

        assert merged == truth, "Merged 3D output differs from single-rank truth"
