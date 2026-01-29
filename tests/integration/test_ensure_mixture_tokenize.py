# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ensure_mixture with tokenized samples.

Tests the core use case: when tokenization expands samples disproportionately
between components (e.g., "long" samples split into 5x more records than
"short"), ensure_mixture reorders output to maintain target ratios.
"""

from __future__ import annotations

import pytest

from zephon.api import Pipeline
from zephon.core.constants import SampleRecord
from zephon.io import Dataset, InMemoryShard
from zephon.work.base import WorkSource
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def _make_dataset(name: str, words_per_sample: int, sample_count: int) -> Dataset:
    """Create an in-memory dataset with samples containing a fixed number of words."""
    rows = [
        {"text": " ".join(f"{name}{i}w{j}" for j in range(words_per_sample))}
        for i in range(sample_count)
    ]
    shard = InMemoryShard(rows)
    return Dataset.from_dict(name, {0: shard})


def _collect_records(pipe: Pipeline) -> list[SampleRecord]:
    """Collect all SampleRecord items from a pipeline."""
    return [item for item in pipe if isinstance(item, SampleRecord)]


def _get_component(rec: SampleRecord) -> int:
    """Extract the primary component ID from a single-component sample."""
    counts = rec.meta.component_sample_counts
    assert len(counts) == 1, f"Expected single-component, got {counts}"
    return next(iter(counts.keys()))


def _count_by_component(records: list[SampleRecord]) -> dict[int, int]:
    """Count records by component.

    Uses component_sample_counts dict from metadata. Each record is expected to
    be a single-component sample.
    """
    counts: dict[int, int] = {}
    for rec in records:
        cid = _get_component(rec)
        counts[cid] = counts.get(cid, 0) + 1
    return counts


class _DynamicMixtureWorkSource(WorkSource):
    """Custom WorkSource providing different mixtures per chunk for testing.

    Chunk 0: 100% A (count_a samples)
    Chunk 1: 50% B / 50% C (count_b + count_c samples)

    When chunk 1 arrives, A becomes obsolete (not in new mixture target).
    This tests:
    1. Dynamic mixture querying via get_chunk_mixture callback
    2. SWRR adapting to new mixture targets
    3. Process runner proxy works (_RemoteServiceProxy)
    """

    def __init__(
        self,
        ds_a: Dataset,
        ds_b: Dataset,
        ds_c: Dataset,
        count_a: int,
        count_b: int,
        count_c: int,
    ) -> None:
        super().__init__()
        self._datasets = [ds_a, ds_b, ds_c]
        self._datasets_by_id = {0: ds_a, 1: ds_b, 2: ds_c}
        self._chunk_index = 0
        self._count_a = count_a
        self._count_b = count_b
        self._count_c = count_c

    def next_chunk(self):
        """Return next chunk with different mixture per chunk."""
        from zephon.work.base import WorkChunk

        if self._chunk_index >= 2:
            return None

        idx = self._chunk_index
        self._chunk_index += 1

        if idx == 0:
            # Chunk 0: 100% A (all A samples)
            return WorkChunk(
                components={"A": [(0, 0, i) for i in range(self._count_a)]},
                seed=42,
            )
        else:
            # Chunk 1: 50% B, 50% C (all B and C samples)
            return WorkChunk(
                components={
                    "B": [(1, 0, i) for i in range(self._count_b)],
                    "C": [(2, 0, i) for i in range(self._count_c)],
                },
                seed=42,
            )

    def supports_indexing(self) -> bool:
        return False

    def __len__(self) -> int:
        return self._count_a + self._count_b + self._count_c

    def sample_id_at(self, index: int):
        raise NotImplementedError()

    @property
    def datasets_by_id(self):
        return self._datasets_by_id

    def chunk_size_hint(self):
        return None


class TestEnsureMixtureDynamicMixture:
    """Test ensure_mixture with dynamic per-chunk mixtures.

    Uses a custom WorkSource where each chunk has a different mixture,
    testing that get_chunk_mixture callback works correctly (including
    through the process runner's _RemoteServiceProxy mechanism).
    """

    @pytest.mark.parametrize("runner", ["threads", "process"])
    def test_dynamic_mixture_and_obsolete_draining(self, runner: str) -> None:
        """Verify dynamic mixture changes and smooth obsolete component draining.

        Scenario:
        - 3 datasets: A (id=0), B (id=1), C (id=2)
        - Chunk 0: 100% A (4 samples)
        - Chunk 1: 50% B, 50% C (5 + 5 = 10 samples)

        When samples from both chunks arrive together (batched), the SWRR updates
        to chunk 1's mixture {B: 0.5, C: 0.5}. At this point:
        - A becomes obsolete (not in current target)
        - B and C are preferred (in target)
        - A samples are drained at obsolete_drain_rate=0.5 (every 2 B/C emissions)

        With 10 B/C samples and 4 A samples:
        - Need 8 B/C emissions to drain all 4 A's (2 B/C per A drain)
        - Remaining 2 B/C emissions continue after A is exhausted

        Expected sequence with obsolete_drain_rate=0.5:
        - B, C, A (drain) - repeat 4 times to drain all A's
        - B, C - continue with remaining B/C after A exhausted

        This test verifies:
        1. Dynamic mixture querying works (get_chunk_mixture callback)
        2. Obsolete components are drained smoothly at the configured rate
        3. Normal emission continues after obsolete components exhausted
        4. Process runner proxy works (same results as threads)
        """
        # Create 3 datasets - enough B/C to drain all A's smoothly
        count_a, count_b, count_c = 4, 5, 5
        ds_a = Dataset.from_dict(
            "A", {0: InMemoryShard([{"text": f"A{i}"} for i in range(count_a)])}
        )
        ds_b = Dataset.from_dict(
            "B", {0: InMemoryShard([{"text": f"B{i}"} for i in range(count_b)])}
        )
        ds_c = Dataset.from_dict(
            "C", {0: InMemoryShard([{"text": f"C{i}"} for i in range(count_c)])}
        )

        work = _DynamicMixtureWorkSource(ds_a, ds_b, ds_c, count_a, count_b, count_c)

        pipe = (
            Pipeline(work)
            .decode_text()
            .ensure_mixture(
                max_buffer_size=20,  # Large enough to avoid forced draining
                weight_by="samples",
                obsolete_drain_rate=0.5,  # Drain 1 obsolete per 2 emissions
            )
            .options(deterministic=True, runner=runner)
        )

        records = _collect_records(pipe)

        # All 14 samples should be emitted (no drops)
        assert len(records) == 14, f"Expected 14 records, got {len(records)}"

        # Count by component - all samples preserved
        counts = _count_by_component(records)
        assert counts == {0: 4, 1: 5, 2: 5}, (
            f"Expected {{0: 4, 1: 5, 2: 5}}, got {counts}"
        )

        # Extract component sequence
        comp_seq = [_get_component(rec) for rec in records]

        # With obsolete_drain_rate=0.5, we drain 1 obsolete every 2 emissions.
        # SWRR target is {B: 0.5, C: 0.5}, so B/C interleave, with A drained every 2.
        # After 8 B/C emissions, all 4 A's are drained. Then 2 more B/C continue.
        expected_seq = [
            1,
            2,
            0,  # B, C, drain A
            1,
            2,
            0,  # B, C, drain A
            1,
            2,
            0,  # B, C, drain A
            1,
            2,
            0,  # B, C, drain A (all A's now drained)
            1,
            2,  # B, C continue (no more A to drain)
        ]
        assert comp_seq == expected_seq, (
            f"Expected sequence {expected_seq}, got {comp_seq}. "
            f"This verifies smooth B/C interleaving with periodic A draining."
        )


class TestEnsureMixtureTokenizationExpansion:
    """Test ensure_mixture with tokenization-induced sample expansion."""

    def test_without_ensure_mixture_gets_skewed(self) -> None:
        """Verify that without ensure_mixture, tokenization skews the mixture.

        Component "long" (10 words) produces ~5x more tokenized samples than
        component "short" (2 words) when using max_length=2 with split_long_samples.
        Starting from a 50:50 sample-level mixture, the post-tokenization
        mixture should be heavily skewed toward "long" (~83% vs ~17%).
        """
        # 10 words -> 5 segments with seq_len=2
        ds_long = _make_dataset("long", words_per_sample=10, sample_count=20)
        # 2 words -> 1 segment with seq_len=2
        ds_short = _make_dataset("short", words_per_sample=2, sample_count=20)

        work = StaticMixtureWorkSource(
            [ds_long, ds_short],
            {"long": 0.5, "short": 0.5},  # 50:50 at sample level
            chunk_size=10,
            seed=42,
        )

        pipe = (
            Pipeline(work)
            .decode_text()
            .tokenize(
                tokenizer_id="__fallback__",
                split_long_samples=True,
                max_length=2,
                preserve_upstream_payload=True,
            )
            .options(deterministic=True, max_workers=2)
        )

        records = _collect_records(pipe)
        counts = _count_by_component(records)

        # With 20 long samples (5 segments each) and 20 short samples (1 segment each)
        # Expected: ~100 long-derived records vs ~20 short-derived records
        total = sum(counts.values())
        assert total > 0, "Should have records"

        # component_id 0 = "long" (first in list), component_id 1 = "short"
        long_count = counts.get(0, 0)
        short_count = counts.get(1, 0)

        # Verify significant skew: long should be ~5x more than short
        # Ratio of long should be around 83% (100 / 120)
        long_ratio = long_count / total
        assert long_ratio > 0.7, (
            f"Expected skewed mixture (>70% long), got {long_ratio:.1%}. "
            f"Counts: long={long_count}, short={short_count}"
        )

    @pytest.mark.parametrize("runner", ["threads", "process"])
    def test_with_ensure_mixture_maintains_balance_until_exhausted(
        self, runner: str
    ) -> None:
        """Verify that ensure_mixture maintains 50:50 ratio until minority exhausts.

        Despite tokenization causing 5x more samples from "long" component,
        ensure_mixture should reorder output to maintain the target 50:50 mixture
        for as long as the minority component has samples available.

        With 20 "short" samples (1 token-sample each) and 20 "long" samples
        (5 token-samples each), we get 20 short + 100 long = 120 total.
        ensure_mixture can maintain 50:50 for the first 40 samples (20 from each),
        then must emit the remaining 80 "long" samples.
        """
        ds_long = _make_dataset("long", words_per_sample=10, sample_count=20)
        ds_short = _make_dataset("short", words_per_sample=2, sample_count=20)

        work = StaticMixtureWorkSource(
            [ds_long, ds_short],
            {"long": 0.5, "short": 0.5},
            chunk_size=10,
            seed=42,
        )

        pipe = (
            Pipeline(work)
            .decode_text()
            .tokenize(
                tokenizer_id="__fallback__",
                split_long_samples=True,
                max_length=2,
                preserve_upstream_payload=True,
            )
            .ensure_mixture(
                max_buffer_size=50,  # Allow buffering to enable reordering
                weight_by="samples",
            )
            .options(deterministic=True, runner=runner, max_workers=1)
        )

        records = _collect_records(pipe)
        counts = _count_by_component(records)

        # 20 long samples (5 token-samples each) + 20 short samples (1 each) = 120
        total = sum(counts.values())
        assert total == 120, f"Expected 120 total records, got {total}"

        # The "short" component has only 20 samples total after tokenization
        # ensure_mixture should alternate until short is exhausted
        short_count = counts.get(1, 0)
        assert short_count == 20, f"Expected 20 short samples, got {short_count}"

        # Check that the first N samples (where N = 2 * short_count) are balanced
        # This is where ensure_mixture can maintain 50:50
        balanced_region = records[: 2 * short_count]
        balanced_counts = _count_by_component(balanced_region)

        # In the balanced region, both components should have roughly equal counts
        long_in_balanced = balanced_counts.get(0, 0)
        short_in_balanced = balanced_counts.get(1, 0)

        # Allow some tolerance due to buffer constraints and SWRR behavior
        assert short_in_balanced >= 15, (
            f"Expected at least 15 short samples in first {2 * short_count} records, "
            f"got {short_in_balanced}. ensure_mixture may not be working."
        )
        assert long_in_balanced >= 15, (
            f"Expected at least 15 long samples in first {2 * short_count} records, "
            f"got {long_in_balanced}. ensure_mixture may not be working."
        )

        # Verify component IDs are preserved through spawn_child
        # (regression test: children must inherit component from parent)
        component_ids = {_get_component(r) for r in records}
        assert component_ids == {0, 1}, (
            f"Expected components {{0, 1}}, got {component_ids}"
        )

        # Children with lineage should inherit component from parent
        children_with_lineage = [r for r in records if r.meta.lineage]
        assert len(children_with_lineage) > 0, "Expected children from long samples"
        for child in children_with_lineage:
            assert _get_component(child) == 0, (
                "Child should inherit component=0 from long parent"
            )

    def test_comparison_with_and_without_ensure_mixture(self) -> None:
        """Direct comparison showing ensure_mixture corrects tokenization skew.

        Setup: 20 "long" samples (10 words -> 5 token-samples each) and
        20 "short" samples (2 words -> 1 token-sample each).
        Total: 100 long-derived + 20 short-derived = 120 records.

        Without ensure_mixture: In the first 40 records, we expect the mixture
        to be skewed toward "long" because each long sample expands to 5 records
        while short samples don't expand. Expected ratio: ~83% long.

        With ensure_mixture: In the first 40 records (where we have enough of
        both components), the mixture should be close to 50:50 as requested.
        """
        ds_long = _make_dataset("long", words_per_sample=10, sample_count=20)
        ds_short = _make_dataset("short", words_per_sample=2, sample_count=20)

        def make_worksource() -> StaticMixtureWorkSource:
            return StaticMixtureWorkSource(
                [ds_long, ds_short],
                {"long": 0.5, "short": 0.5},
                chunk_size=10,
                seed=42,
            )

        # Pipeline without ensure_mixture
        pipe_without = (
            Pipeline(make_worksource())
            .decode_text()
            .tokenize(
                tokenizer_id="__fallback__",
                split_long_samples=True,
                max_length=2,
                preserve_upstream_payload=True,
            )
            .options(deterministic=True, max_workers=1)
        )
        records_without = _collect_records(pipe_without)

        # Pipeline with ensure_mixture
        pipe_with = (
            Pipeline(make_worksource())
            .decode_text()
            .tokenize(
                tokenizer_id="__fallback__",
                split_long_samples=True,
                max_length=2,
                preserve_upstream_payload=True,
            )
            .ensure_mixture(
                max_buffer_size=50,
                weight_by="samples",
            )
            .options(deterministic=True, max_workers=1)
        )
        records_with = _collect_records(pipe_with)

        # Both should have the same total record count (ensure_mixture only reorders)
        assert len(records_with) == len(records_without), (
            "ensure_mixture should not drop records"
        )

        # Analyze the first 40 samples - this is where we have enough data from
        # both components (20 short samples available, so 40 total at 50:50)
        first_40_with = records_with[:40]
        first_40_without = records_without[:40]

        counts_with = _count_by_component(first_40_with)
        counts_without = _count_by_component(first_40_without)

        long_without = counts_without.get(0, 0)
        short_without = counts_without.get(1, 0)
        long_with = counts_with.get(0, 0)
        short_with = counts_with.get(1, 0)

        # WITHOUT ensure_mixture: mixture should be SKEWED toward long
        # Because long samples expand 5x, we expect ~83% long in first 40
        # (each original sample pair produces 5 long + 1 short = 6 records,
        # ratio = 5/6 = 83%)
        ratio_long_without = long_without / 40
        assert ratio_long_without >= 0.70, (
            f"Without ensure_mixture: expected skewed mixture (>=70% long) in first 40, "
            f"got {ratio_long_without:.1%} (long={long_without}, short={short_without})"
        )

        # WITH ensure_mixture: mixture should be BALANCED (close to 50:50)
        # ensure_mixture actively reorders to maintain the target ratio
        ratio_long_with = long_with / 40
        assert 0.45 <= ratio_long_with <= 0.55, (
            f"With ensure_mixture: expected balanced mixture (45-55% long) in first 40, "
            f"got {ratio_long_with:.1%} (long={long_with}, short={short_with})"
        )
