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

import multiprocessing as mp
import os
import tempfile
import time
import traceback
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

from zephon import Pipeline as PublicPipeline
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
    from zephon.types import SampleBatch, SampleRecord

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
    aggregate_timeout_s: float | None = None,
    run_id: str | None = None,
    seed: int = 42,
    mtp_mode: bool = False,
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
        "mtp_mode": mtp_mode,
    }
    if aggregate_dir is not None:
        opts["aggregate_dir"] = aggregate_dir
    if aggregate_timeout_s is not None:
        opts["aggregate_timeout_s"] = aggregate_timeout_s
    if run_id is not None:
        opts["run_id"] = run_id

    return pipe.options(**opts)


def consume_n(pipe: PublicPipeline, n: int) -> list[str]:
    """Consume exactly n samples from pipeline."""
    texts: list[str] = []
    it = iter(pipe)
    try:
        for item in it:
            texts.extend(_extract_texts(item))
            if len(texts) >= n:
                break
    finally:
        it.close()
    return texts[:n]


def get_lanes_for_dp_group(pipe: PublicPipeline, dp_group_id: int) -> list[int]:
    """Get the lanes assigned to a DP group from a built pipeline.

    Needs inline mode to inspect internal world topology — no public API for this.
    """
    pipe = pipe.options(mtp_mode=False)
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
    """Checkpoint and resume across 3D-parallelism configurations.

    Each test spawns the declared ``world_size`` ranks as separate
    processes so the multi-rank aggregation path is actually exercised.
    The ``dp_degree=1`` cases verify that all peers in the single DP
    group resume to identical state.
    """

    @pytest.mark.parametrize("mtp_mode", [False, True], ids=["inline", "mtp"])
    def test_resume_same_config(self, tmp_path: Path, mtp_mode: bool) -> None:
        """Checkpoint and resume with identical config reproduces baseline.

        ``dp_degree=1`` with ``world_size=4`` means every rank shares
        ``dp_group_id=0`` and owns the same canonical lanes. Each of the
        4 spawned ranks runs the full prefix → checkpoint → restore → suffix
        cycle; every rank's ``prefix + suffix`` must equal the baseline.
        """
        ds_name = "a"
        ds_sample_count = 64
        chunk_size = 8
        n_total = 24
        n_prefix = 12
        world_size = 4

        # Baseline materializes the deterministic per-rank sample order;
        # ``mtp_mode`` is unconditionally False because the multi-rank
        # aggregation path under test runs in the spawned ranks below.
        ds = make_dataset(ds_name, ds_sample_count)
        baseline = consume_n(
            _build_pipeline(
                ds,
                chunk_size=chunk_size,
                canonical_replicas=2,
                world_size=world_size,
                global_rank=0,
                dp_degree=1,
                dp_group_id=0,
            ),
            n=n_total,
        )

        # mtp_mode spawns a nested subprocess per rank, so bump the aggregation
        # timeout to absorb the extra startup latency.
        aggregate_timeout_s = 30.0 if mtp_mode else 15.0
        results, errors = _run_consume_checkpoint_resume(
            world_size=world_size,
            dp_degree=1,
            canonical_replicas=2,
            chunk_size=chunk_size,
            n_prefix=n_prefix,
            n_suffix=n_total - n_prefix,
            run_id_p1=f"resume_same_config_p1_{int(mtp_mode)}",
            run_id_p2=f"resume_same_config_p2_{int(mtp_mode)}",
            agg_dir=str(tmp_path),
            ds_name=ds_name,
            ds_sample_count=ds_sample_count,
            mtp_mode=mtp_mode,
            aggregate_timeout_s=aggregate_timeout_s,
        )

        assert not errors, (
            "Rank(s) raised during prefix/checkpoint/resume/suffix: "
            + "\n".join(f"rank {r}: {msg}\n{tb}" for r, msg, tb in errors)
        )
        assert len(results) == world_size

        for rank, (prefix, suffix) in results.items():
            assert prefix == baseline[:n_prefix], (
                f"rank {rank} prefix {prefix} != baseline prefix {baseline[:n_prefix]}"
            )
            assert prefix + suffix == baseline, (
                f"rank {rank} prefix+suffix {prefix + suffix} != baseline {baseline}"
            )

    @pytest.mark.parametrize("mtp_mode", [False, True], ids=["inline", "mtp"])
    def test_resume_different_mp_degree(self, tmp_path: Path, mtp_mode: bool) -> None:
        """Resume with changed world_size (mp_degree) but same dp config.

        Phase 1: world_size=2, dp_degree=1 (mp_degree=2) — checkpoint
        Phase 2: world_size=4, dp_degree=1 (mp_degree=4) — restore and consume

        Phase 1 and Phase 2 use different rank counts ⇒ two separate spawn
        waves; rank 0's Phase-1 merged ckpt is broadcast to every rank in
        Phase 2 via mp.Queue.
        """
        ds_name = "a"
        ds_sample_count = 64
        chunk_size = 8
        n_prefix = 8
        n_suffix = 8

        # Baseline materializes the deterministic per-rank sample order;
        # ``mtp_mode`` is unconditionally False because the multi-rank
        # aggregation path under test runs in the spawned ranks below.
        ds = make_dataset(ds_name, ds_sample_count)
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
            n=n_prefix + n_suffix,
        )

        aggregate_timeout_s = 30.0 if mtp_mode else 15.0

        # ---- Wave 1: world_size=2, consume prefix, checkpoint. ----
        wave1_ckpts, wave1_errors = _run_ranks_and_collect(
            world_size=2,
            dp_degree=1,
            canonical_replicas=2,
            chunk_size=chunk_size,
            n_consume=n_prefix,
            run_id=f"mp_change_p1_{int(mtp_mode)}",
            agg_dir=str(tmp_path),
            ds_name=ds_name,
            ds_sample_count=ds_sample_count,
            mtp_mode=mtp_mode,
            aggregate_timeout_s=aggregate_timeout_s,
        )
        assert not wave1_errors, "Wave 1 rank(s) raised: " + "\n".join(
            f"rank {r}: {msg}\n{tb}" for r, msg, tb in wave1_errors
        )
        assert len(wave1_ckpts) == 2
        ckpts_by_rank = dict(wave1_ckpts)
        # All ranks see the same merged ckpt; pick rank 0.
        ckpt = ckpts_by_rank[0]

        # ---- Wave 2: world_size=4 restore. No aggregation in Phase 2, so
        # an in-process loop over the ranks suffices.
        wave2_world_size = 4
        suffixes: list[list[str]] = []
        for rank in range(wave2_world_size):
            pipe = _build_pipeline(
                ds,
                chunk_size=chunk_size,
                canonical_replicas=2,
                world_size=wave2_world_size,
                global_rank=rank,
                dp_degree=1,
                dp_group_id=0,
            )
            pipe.restore(ckpt)
            suffixes.append(consume_n(pipe, n=n_suffix))

        # All ranks share dp_group_id=0 ⇒ identical suffixes; each matches baseline.
        expected_suffix = baseline[n_prefix:]
        for rank, suffix in enumerate(suffixes):
            assert suffix == expected_suffix, (
                f"rank {rank} suffix {suffix} != expected {expected_suffix}"
            )

    @pytest.mark.parametrize("mtp_mode", [False, True], ids=["inline", "mtp"])
    def test_mp_peers_resume_identically(self, tmp_path: Path, mtp_mode: bool) -> None:
        """Peers in the same DP group resume to identical state.

        Phase 1 spawns every declared rank as a separate process to
        exercise the multi-rank aggregation path (parametrized over
        ``mtp_mode``). Phase 2 restores the merged checkpoint on each
        rank in turn and verifies the consumed samples are identical
        across ranks; it runs inline because no aggregation is involved.
        """
        ds_name = "a"
        ds_sample_count = 64
        chunk_size = 8
        world_size = 4

        # Phase 1: spawn all 4 ranks; collect one rank's merged ckpt.
        aggregate_timeout_s = 30.0 if mtp_mode else 15.0
        checkpoints, errors = _run_ranks_and_collect(
            world_size=world_size,
            dp_degree=1,
            canonical_replicas=2,
            chunk_size=chunk_size,
            n_consume=1,
            run_id=f"mp_peers_{int(mtp_mode)}",
            agg_dir=str(tmp_path),
            ds_name=ds_name,
            ds_sample_count=ds_sample_count,
            mtp_mode=mtp_mode,
            aggregate_timeout_s=aggregate_timeout_s,
        )
        assert not errors, "Rank(s) raised during Phase 1: " + "\n".join(
            f"rank {r}: {msg}\n{tb}" for r, msg, tb in errors
        )
        assert len(checkpoints) == world_size
        ckpts_by_rank = dict(checkpoints)
        ckpt = ckpts_by_rank[0]

        # Phase 2: sequential in-process restore on each rank's view.
        ds = make_dataset(ds_name, ds_sample_count)
        resumed: list[list[str]] = []
        for rank in range(world_size):
            pipe = _build_pipeline(
                ds,
                chunk_size=chunk_size,
                canonical_replicas=2,
                world_size=world_size,
                global_rank=rank,
                dp_degree=1,
                dp_group_id=0,
            )
            pipe.restore(ckpt)
            resumed.append(consume_n(pipe, n=8))

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


# =============================================================================
# Multi-process checkpoint aggregation under model parallelism
# =============================================================================
#
# These tests pin the multi-rank aggregation contract under 3D parallelism
# (DP × TP/PP): the leader must merge per-rank state files into a single
# consistent checkpoint that every rank observes identically.
#
# The interesting regime is ``dp_degree >= 2 AND world_size > dp_degree``
# (at ``workers_per_rank=1``, the default for raw zephon without a
# DataLoader). Ranks sharing a ``dp_group_id`` own the *same* canonical
# lanes, so each writes a state file with lane-keyed entries that overlap
# its peers'. The merge must collapse those peer files into one
# representative per group before the lane-uniqueness invariant fires; the
# unit-level repro lives in
# ``tests/zephon/core/test_engine_extras.py::TestMergeStateDictsSharedDPGroup``.
#
# At ``dp_degree == 1`` the regime is different — every rank in the single
# DP group owns every canonical lane, so the leader's own state file
# already satisfies the lane-coverage half of the wait. The leader still
# has to wait for every follower to commit a state file before deleting
# ``round.current``; that branch is exercised by
# ``TestFollowerObservesMergedCheckpoint`` below.
# =============================================================================


def _aggregate_worker(
    rank: int,
    *,
    world_size: int,
    dp_degree: int,
    dp_group_id: int,
    canonical_replicas: int,
    chunk_size: int,
    n_consume: int,
    run_id: str,
    agg_dir: str,
    ds_name: str,
    ds_sample_count: int,
    ckpt_q: "mp.Queue",
    err_q: "mp.Queue",
    start_barrier: "mp.Barrier",
    ckpt_barrier: "mp.Barrier",
    mtp_mode: bool = False,
    aggregate_timeout_s: float = 15.0,
) -> None:
    """Subprocess body: build pipe, consume some samples, checkpoint at barrier."""
    try:
        ds = make_dataset(ds_name, ds_sample_count)
        pipe = _build_pipeline(
            ds,
            chunk_size=chunk_size,
            canonical_replicas=canonical_replicas,
            world_size=world_size,
            global_rank=rank,
            dp_degree=dp_degree,
            dp_group_id=dp_group_id,
            aggregate_dir=agg_dir,
            aggregate_timeout_s=aggregate_timeout_s,
            run_id=run_id,
            mtp_mode=mtp_mode,
        )

        # Synchronize start so all ranks build their engines concurrently.
        start_barrier.wait(timeout=60.0)

        it = iter(pipe)
        try:
            consumed = 0
            for item in it:
                consumed += len(_extract_texts(item))
                if consumed >= n_consume:
                    break
            ckpt_barrier.wait(timeout=60.0)
            ckpt = pipe.checkpoint()
            ckpt_q.put((rank, ckpt))
        finally:
            # Re-wait on ckpt_barrier as a done-barrier: without it the
            # leader can exit and trigger Engine.__del__ -> _clean_merged
            # (deleting merged_<rid>.ckpt) before followers read it.
            try:
                ckpt_barrier.wait(timeout=60.0)
            except Exception:
                pass
            try:
                it.close()
            except Exception:
                pass
    except BaseException as e:  # noqa: BLE001 — must surface any failure
        err_q.put((rank, repr(e), traceback.format_exc()))


def _run_ranks_and_collect(
    *,
    world_size: int,
    dp_degree: int,
    canonical_replicas: int,
    chunk_size: int,
    n_consume: int,
    run_id: str,
    agg_dir: str,
    ds_name: str = "a",
    ds_sample_count: int = 64,
    timeout_s: float = 120.0,
    mtp_mode: bool = False,
    aggregate_timeout_s: float = 15.0,
) -> tuple[list[tuple[int, Any]], list[tuple[int, str, str]]]:
    """Spawn ``world_size`` rank workers, sync them, collect checkpoints/errors.

    Returns ``(checkpoints, errors)`` where ``checkpoints`` is a list of
    ``(rank, ckpt)`` and ``errors`` is a list of ``(rank, repr, traceback)``.
    """
    ctx = mp.get_context("spawn")
    ckpt_q: mp.Queue = ctx.Queue()
    err_q: mp.Queue = ctx.Queue()
    start_barrier = ctx.Barrier(world_size)
    ckpt_barrier = ctx.Barrier(world_size)

    procs: list[mp.Process] = []
    for rank in range(world_size):
        # Contiguous rank → dp_group mapping (matches StaticMixture default).
        ranks_per_dp = world_size // dp_degree
        dp_group_id = rank // ranks_per_dp
        p = ctx.Process(
            target=_aggregate_worker,
            args=(rank,),
            kwargs=dict(
                world_size=world_size,
                dp_degree=dp_degree,
                dp_group_id=dp_group_id,
                canonical_replicas=canonical_replicas,
                chunk_size=chunk_size,
                n_consume=n_consume,
                run_id=run_id,
                agg_dir=agg_dir,
                ds_name=ds_name,
                ds_sample_count=ds_sample_count,
                ckpt_q=ckpt_q,
                err_q=err_q,
                start_barrier=start_barrier,
                ckpt_barrier=ckpt_barrier,
                mtp_mode=mtp_mode,
                aggregate_timeout_s=aggregate_timeout_s,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    checkpoints: list[tuple[int, Any]] = []
    errors: list[tuple[int, str, str]] = []
    deadline = time.monotonic() + timeout_s
    try:
        # Drain both queues until every rank has reported (either ckpt or error)
        # or any process has crashed without sending anything.
        while (
            len(checkpoints) + len(errors) < world_size and time.monotonic() < deadline
        ):
            got = False
            try:
                checkpoints.append(ckpt_q.get(timeout=1.0))
                got = True
            except Empty:
                pass
            try:
                errors.append(err_q.get_nowait())
                got = True
            except Empty:
                pass
            if not got:
                dead = [
                    (p.pid, p.exitcode)
                    for p in procs
                    if not p.is_alive() and p.exitcode not in (None, 0)
                ]
                if dead and (len(checkpoints) + len(errors)) < world_size:
                    # Give a dying process a chance to drain its error queue.
                    try:
                        errors.append(err_q.get(timeout=5.0))
                        continue
                    except Empty:
                        errors.append((-1, f"process(es) died: {dead}", ""))
                        break
            if not any(p.is_alive() for p in procs):
                # All processes exited — drain queues and stop.
                try:
                    while True:
                        checkpoints.append(ckpt_q.get_nowait())
                except Empty:
                    pass
                try:
                    while True:
                        errors.append(err_q.get_nowait())
                except Empty:
                    pass
                break
        if len(checkpoints) + len(errors) < world_size:
            still_alive = [
                (rank, p.pid) for rank, p in enumerate(procs) if p.is_alive()
            ]
            reason = (
                f"collector deadline of {timeout_s}s elapsed"
                if still_alive
                else "all ranks exited without reporting"
            )
            errors.append(
                (
                    -1,
                    f"{reason}; got {len(checkpoints)} ckpts, "
                    f"{len(errors)} errors; still alive (rank, pid): {still_alive}",
                    "",
                )
            )
    finally:
        for p in procs:
            p.join(timeout=timeout_s)
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)

    return checkpoints, errors


class TestCheckpointAggregation3D:
    """End-to-end multi-process checkpoint aggregation under model parallelism.

    These tests live in their own class so they can be filtered with
    ``-k Aggregation3D`` when iterating on the fix.
    """

    @pytest.mark.parametrize(
        ("world_size", "dp_degree", "canonical_replicas"),
        [
            # dp >= 2 with world_size > dp_degree: leader has to wait for
            # other DP groups, so its same-DP-group peers' files get picked up
            # before the merge → duplicate-inflight reliably fires.
            pytest.param(4, 2, 2, id="dp2-tp2"),  # 2 peers per DP group
            pytest.param(6, 2, 2, id="dp2-tp3"),  # 3 peers per DP group
            pytest.param(8, 4, 4, id="dp4-tp2"),  # 2 peers, 4 DP groups
        ],
    )
    def test_checkpoint_aggregates_with_model_parallelism(
        self,
        tmp_path: Path,
        world_size: int,
        dp_degree: int,
        canonical_replicas: int,
    ) -> None:
        """All ranks call checkpoint() concurrently → merged state is consistent.

        With ``world_size > dp_degree``, multiple ranks share a ``dp_group_id``
        and own the same canonical lanes. The leader must produce a single
        consistent merged checkpoint, and every rank (leader and follower)
        must observe it.
        """
        checkpoints, errors = _run_ranks_and_collect(
            world_size=world_size,
            dp_degree=dp_degree,
            canonical_replicas=canonical_replicas,
            chunk_size=4,
            n_consume=8,
            run_id=f"agg3d_w{world_size}_dp{dp_degree}",
            agg_dir=str(tmp_path),
        )

        # Every rank must have produced a checkpoint, none should have errored.
        assert not errors, "Rank(s) raised during checkpoint aggregation: " + "\n".join(
            f"rank {r}: {msg}\n{tb}" for r, msg, tb in errors
        )
        assert len(checkpoints) == world_size, (
            f"Expected {world_size} checkpoints, got {len(checkpoints)}"
        )

        # Every rank should observe the *same* merged checkpoint.
        ckpts_by_rank = dict(checkpoints)
        ranks_sorted = sorted(ckpts_by_rank)
        ref = ckpts_by_rank[ranks_sorted[0]]
        for r in ranks_sorted[1:]:
            assert ckpts_by_rank[r] == ref, (
                f"Rank {r} merged checkpoint diverges from rank {ranks_sorted[0]}"
            )

        # Merged checkpoint must cover every canonical lane exactly once.
        inflight_lanes = {int(k) for k in ref.get("inflight", {}).keys()}
        progress_lanes = {int(k) for k in ref.get("progress", {}).keys()}
        expected_lanes = set(range(canonical_replicas))
        assert progress_lanes == expected_lanes, (
            f"Merged progress covers lanes {progress_lanes}, expected {expected_lanes}"
        )
        # inflight may legitimately be a subset (e.g., all chunks already drained)
        assert inflight_lanes.issubset(expected_lanes), (
            f"Merged inflight has unexpected lanes: {inflight_lanes - expected_lanes}"
        )

        # Cross-rank value consistency: every rank's merged ckpt must agree
        # on the per-lane progress chunk_id/offset and lane_next_cid. (The
        # dict-equality assertion above already implies this, but a focused
        # check pinpoints lane-keyed value drift if it ever resurfaces.)
        ref_progress = {int(k): v for k, v in ref.get("progress", {}).items()}
        ref_lane_next = {
            int(k): int(v) for k, v in ref.get("lane_next_cid", {}).items()
        }
        for r in ranks_sorted[1:]:
            other = ckpts_by_rank[r]
            other_progress = {int(k): v for k, v in other.get("progress", {}).items()}
            other_lane_next = {
                int(k): int(v) for k, v in other.get("lane_next_cid", {}).items()
            }
            assert other_progress == ref_progress, (
                f"Rank {r} progress values diverge from rank {ranks_sorted[0]}"
            )
            assert other_lane_next == ref_lane_next, (
                f"Rank {r} lane_next_cid values diverge from rank {ranks_sorted[0]}"
            )


def _consume_then_checkpoint_worker(
    rank: int,
    *,
    world_size: int,
    dp_degree: int,
    dp_group_id: int,
    canonical_replicas: int,
    chunk_size: int,
    n_prefix: int,
    n_suffix: int,
    run_id_p1: str,
    run_id_p2: str,
    agg_dir: str,
    ds_name: str,
    ds_sample_count: int,
    out_q: "mp.Queue",
    err_q: "mp.Queue",
    start_barrier: "mp.Barrier",
    ckpt_barrier: "mp.Barrier",
    resume_barrier: "mp.Barrier",
    mtp_mode: bool = False,
    aggregate_timeout_s: float = 15.0,
) -> None:
    """Subprocess body for the resume-after-aggregate test."""
    try:
        ds = make_dataset(ds_name, ds_sample_count)
        pipe1 = _build_pipeline(
            ds,
            chunk_size=chunk_size,
            canonical_replicas=canonical_replicas,
            world_size=world_size,
            global_rank=rank,
            dp_degree=dp_degree,
            dp_group_id=dp_group_id,
            aggregate_dir=agg_dir,
            aggregate_timeout_s=aggregate_timeout_s,
            run_id=run_id_p1,
            mtp_mode=mtp_mode,
        )

        start_barrier.wait(timeout=60.0)
        prefix: list[str] = []
        it1 = iter(pipe1)
        try:
            for item in it1:
                prefix.extend(_extract_texts(item))
                if len(prefix) >= n_prefix:
                    break
            ckpt_barrier.wait(timeout=60.0)
            ckpt = pipe1.checkpoint()
        finally:
            try:
                it1.close()
            except Exception:
                pass
        prefix = prefix[:n_prefix]

        # Phase 2: resume from merged checkpoint, consume suffix.
        pipe2 = _build_pipeline(
            ds,
            chunk_size=chunk_size,
            canonical_replicas=canonical_replicas,
            world_size=world_size,
            global_rank=rank,
            dp_degree=dp_degree,
            dp_group_id=dp_group_id,
            aggregate_dir=agg_dir,
            aggregate_timeout_s=aggregate_timeout_s,
            run_id=run_id_p2,
            mtp_mode=mtp_mode,
        )
        pipe2.restore(ckpt)
        resume_barrier.wait(timeout=60.0)
        suffix = consume_n(pipe2, n=n_suffix)

        out_q.put((rank, prefix, suffix))
    except BaseException as e:  # noqa: BLE001
        err_q.put((rank, repr(e), traceback.format_exc()))


def _run_consume_checkpoint_resume(
    *,
    world_size: int,
    dp_degree: int,
    canonical_replicas: int,
    chunk_size: int,
    n_prefix: int,
    n_suffix: int,
    run_id_p1: str,
    run_id_p2: str,
    agg_dir: str,
    ds_name: str = "a",
    ds_sample_count: int = 64,
    mtp_mode: bool = False,
    aggregate_timeout_s: float = 15.0,
    collect_timeout_s: float = 180.0,
) -> tuple[
    dict[int, tuple[list[str], list[str]]],
    list[tuple[int, str, str]],
]:
    """Spawn ``world_size`` ranks of ``_consume_then_checkpoint_worker``.

    Returns ``(results, errors)`` where ``results`` maps rank →
    ``(prefix, suffix)``. Workers are joined inside a ``finally`` block.
    """
    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue()
    err_q: mp.Queue = ctx.Queue()
    start_barrier = ctx.Barrier(world_size)
    ckpt_barrier = ctx.Barrier(world_size)
    resume_barrier = ctx.Barrier(world_size)
    ranks_per_dp = world_size // dp_degree

    procs: list[mp.Process] = []
    for rank in range(world_size):
        dp_group_id = rank // ranks_per_dp
        p = ctx.Process(
            target=_consume_then_checkpoint_worker,
            args=(rank,),
            kwargs=dict(
                world_size=world_size,
                dp_degree=dp_degree,
                dp_group_id=dp_group_id,
                canonical_replicas=canonical_replicas,
                chunk_size=chunk_size,
                n_prefix=n_prefix,
                n_suffix=n_suffix,
                run_id_p1=run_id_p1,
                run_id_p2=run_id_p2,
                agg_dir=agg_dir,
                ds_name=ds_name,
                ds_sample_count=ds_sample_count,
                out_q=out_q,
                err_q=err_q,
                start_barrier=start_barrier,
                ckpt_barrier=ckpt_barrier,
                resume_barrier=resume_barrier,
                mtp_mode=mtp_mode,
                aggregate_timeout_s=aggregate_timeout_s,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    results: dict[int, tuple[list[str], list[str]]] = {}
    errors: list[tuple[int, str, str]] = []
    deadline = time.monotonic() + collect_timeout_s
    try:
        received = 0
        while received < world_size and time.monotonic() < deadline:
            try:
                rank, prefix, suffix = out_q.get(timeout=1.0)
                results[rank] = (prefix, suffix)
                received += 1
                continue
            except Empty:
                pass
            try:
                errors.append(err_q.get_nowait())
                received += 1
                continue
            except Empty:
                pass
            if all(not p.is_alive() for p in procs):
                break
    finally:
        for p in procs:
            p.join(timeout=30.0)
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)

    return results, errors


class TestResumeAfterAggregation3D:
    """Resume-after-merged-checkpoint under model parallelism.

    Even if ``_merge_state_dicts`` were patched to ignore duplicates, the
    *resume path* could still corrupt state if same-DP-group peers don't
    observe identical aggregated state on reload. These tests pin the full
    round-trip.
    """

    @pytest.mark.parametrize(
        ("world_size", "dp_degree", "canonical_replicas"),
        [
            # Both configs require dp_degree >= 2 — see precondition note above
            # TestCheckpointAggregation3D.
            pytest.param(4, 2, 2, id="dp2-tp2"),
            pytest.param(8, 4, 4, id="dp4-tp2"),
        ],
    )
    def test_prefix_plus_suffix_equals_baseline_under_model_parallelism(
        self,
        tmp_path: Path,
        world_size: int,
        dp_degree: int,
        canonical_replicas: int,
    ) -> None:
        """For each DP group, prefix+suffix from a checkpointed run equals the
        baseline samples consumed in one shot."""
        ds_name = "agg3d_resume"
        ds_sample_count = 64
        chunk_size = 4
        n_prefix = 8
        n_suffix = 8

        # ---- Baseline: single-process per DP group, no checkpoint ----
        ds = make_dataset(ds_name, ds_sample_count)
        ranks_per_dp = world_size // dp_degree
        baseline_per_dp: dict[int, list[str]] = {}
        for dp_id in range(dp_degree):
            # Pick the lowest-numbered rank in that DP group as the baseline runner.
            rep_rank = dp_id * ranks_per_dp
            baseline_per_dp[dp_id] = consume_n(
                _build_pipeline(
                    ds,
                    chunk_size=chunk_size,
                    canonical_replicas=canonical_replicas,
                    world_size=world_size,
                    global_rank=rep_rank,
                    dp_degree=dp_degree,
                    dp_group_id=dp_id,
                ),
                n=n_prefix + n_suffix,
            )

        # ---- Multi-process: prefix → checkpoint → resume → suffix ----
        results, errors = _run_consume_checkpoint_resume(
            world_size=world_size,
            dp_degree=dp_degree,
            canonical_replicas=canonical_replicas,
            chunk_size=chunk_size,
            n_prefix=n_prefix,
            n_suffix=n_suffix,
            run_id_p1=f"resume3d_p1_w{world_size}_dp{dp_degree}",
            run_id_p2=f"resume3d_p2_w{world_size}_dp{dp_degree}",
            agg_dir=str(tmp_path),
            ds_name=ds_name,
            ds_sample_count=ds_sample_count,
        )

        assert not errors, (
            "Rank(s) raised during prefix/checkpoint/resume/suffix: "
            + "\n".join(f"rank {r}: {msg}\n{tb}" for r, msg, tb in errors)
        )
        assert len(results) == world_size

        # Every rank in a DP group must agree with the baseline for that group.
        for rank, (prefix, suffix) in results.items():
            dp_id = rank // ranks_per_dp
            expected = baseline_per_dp[dp_id]
            assert prefix + suffix == expected, (
                f"rank {rank} (dp_group {dp_id}): prefix+suffix {prefix + suffix} "
                f"!= baseline {expected}"
            )


# =============================================================================
# dp=1, mp>1 follower must observe the leader's merged checkpoint
# =============================================================================
#
# At ``dp_degree == 1`` the leader's own state file already covers every
# canonical lane on the very first coverage poll. Without the
# every-rank-reports-in coverage requirement, the leader could race
# through merge + ``round.current`` cleanup before a slow follower had
# even read ``round.current`` — leaving the follower unable to learn
# ``round_id`` and timing out, even though the merged ckpt is on disk.
#
# The leader must wait until every declared rank has committed a state
# file (so every follower has already learned ``round_id`` and is now
# blocked on the merged ckpt) before cleaning up ``round.current``.
# =============================================================================


class TestFollowerObservesMergedCheckpoint:
    """Followers in a ``dp_degree=1, mp_degree>1`` round still observe the
    leader's merged checkpoint when their ``checkpoint()`` call lands after
    the leader has finished merging and removed ``round.current``."""

    def test_dp1_tp2_follower_returns_same_merged_ckpt(self, tmp_path: Path) -> None:
        """``world_size=2, dp_degree=1``: both ranks return the same merged
        checkpoint even when the follower's poll arrives after the leader's
        cleanup."""
        checkpoints, errors = _run_ranks_and_collect(
            world_size=2,
            dp_degree=1,
            canonical_replicas=2,
            chunk_size=4,
            n_consume=4,
            run_id="follower_recovery_dp1_tp2",
            agg_dir=str(tmp_path),
        )

        assert not errors, "Rank(s) raised during checkpoint: " + "\n".join(
            f"rank {r}: {msg}\n{tb}" for r, msg, tb in errors
        )
        assert len(checkpoints) == 2
        ckpts_by_rank = dict(checkpoints)
        assert ckpts_by_rank[0] == ckpts_by_rank[1], (
            "Follower must observe the same merged checkpoint as the leader"
        )
