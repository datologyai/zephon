# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ensure_mixture with simple (non-tokenized) samples.

Tests ensure_mixture behavior with decode_text() samples (no tokenization):
- Basic pipeline smoke test
- Reordering imbalanced input
- Determinism across runs
- Explicit weight overrides and front-loading
- Runner parity (threads vs inline)
- Record preservation (no drops)
- Warn mode
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zephon import Pipeline
from zephon.io.dataset import Dataset
from zephon.types import SampleRecord
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def _write_jsonl_dir(
    root: Path, shard_files: int, values: list[int], name: str
) -> None:
    """Write JSONL files for testing."""
    root.mkdir(parents=True, exist_ok=True)
    per = len(values) // shard_files
    extra = len(values) % shard_files
    offset = 0
    for f in range(shard_files):
        take = per + (1 if f < extra else 0)
        chunk = values[offset : offset + take]
        offset += take
        lines = [json.dumps({"text": str(v), "component": name}) for v in chunk]
        (root / f"data_{f}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_datasets(
    tmp: Path, code_count: int = 100, text_count: int = 100, files: int = 2
) -> tuple[Dataset, Dataset]:
    """Create two datasets: code and text."""
    code_dir = tmp / "code"
    text_dir = tmp / "text"

    _write_jsonl_dir(code_dir, files, list(range(code_count)), "code")
    _write_jsonl_dir(text_dir, files, list(range(1000, 1000 + text_count)), "text")

    ds_code = Dataset.from_path("code", str(code_dir), fmt="jsonl")
    ds_text = Dataset.from_path("text", str(text_dir), fmt="jsonl")
    return ds_code, ds_text


def _count_components(records: list[SampleRecord]) -> dict[str, int]:
    """Count records by dataset_id (component)."""
    counts: dict[int, int] = {}
    for rec in records:
        did = rec.meta.sample_id[0]
        counts[did] = counts.get(did, 0) + 1
    # Convert to named components (assuming 0=code, 1=text)
    return {"code": counts.get(0, 0), "text": counts.get(1, 0)}


def _collect_records(pipe: Pipeline) -> list[SampleRecord]:
    """Collect all records from pipeline."""
    records = []
    for item in pipe:
        if isinstance(item, SampleRecord):
            records.append(item)
    return records


class TestEnsureMixtureIntegration:
    """Integration tests for EnsureMixture with real datasets."""

    def test_ensure_mixture_basic_pipeline(self, tmp_path: Path) -> None:
        """Test basic pipeline with ensure_mixture operator."""
        ds_code, ds_text = _prepare_datasets(tmp_path, code_count=50, text_count=50)

        work = StaticMixtureWorkSource(
            [ds_code, ds_text],
            {"code": 0.3, "text": 0.7},
            chunk_size=20,
            seed=42,
        )

        pipe = (
            Pipeline(work)
            .decode_text()
            .ensure_mixture(max_buffer_size=10, weight_by="samples")
            .options(deterministic=True, max_workers=2)
        )

        records = _collect_records(pipe)
        counts = _count_components(records)

        # Verify we got records from both components
        assert counts["code"] > 0, "Expected some code samples"
        assert counts["text"] > 0, "Expected some text samples"

    def test_ensure_mixture_reorders_imbalanced_input(self, tmp_path: Path) -> None:
        """Test that ensure_mixture reorders imbalanced streams."""
        # Create imbalanced input: more code samples available
        ds_code, ds_text = _prepare_datasets(tmp_path, code_count=100, text_count=100)

        work = StaticMixtureWorkSource(
            [ds_code, ds_text],
            # Request 50/50 ratio
            {"code": 0.5, "text": 0.5},
            chunk_size=20,
            seed=42,
        )

        pipe = (
            Pipeline(work)
            .decode_text()
            .ensure_mixture(max_buffer_size=20, weight_by="samples")
            .options(deterministic=True, max_workers=2)
        )

        records = _collect_records(pipe)
        counts = _count_components(records)

        # With 50/50 target and enough samples, output should be roughly balanced
        total = counts["code"] + counts["text"]
        if total > 0:
            ratio_code = counts["code"] / total
            # Allow some tolerance since SWRR is approximate
            assert 0.4 <= ratio_code <= 0.6, f"Expected ~50% code, got {ratio_code:.2%}"

    def test_ensure_mixture_deterministic(self, tmp_path: Path) -> None:
        """Test that ensure_mixture produces deterministic output."""
        ds_code, ds_text = _prepare_datasets(tmp_path, code_count=50, text_count=50)

        results: list[list[tuple[int, int, int]]] = []

        for run in range(3):
            work = StaticMixtureWorkSource(
                [ds_code, ds_text],
                {"code": 0.4, "text": 0.6},
                chunk_size=16,
                seed=42,
            )

            pipe = (
                Pipeline(work)
                .decode_text()
                .ensure_mixture(max_buffer_size=8, weight_by="samples")
                .options(deterministic=True, max_workers=4)
            )

            ids = [rec.meta.sample_id for rec in _collect_records(pipe)]
            results.append(ids)

        # All runs should produce identical output
        for i, ids in enumerate(results[1:], 1):
            assert ids == results[0], f"Run {i} differs from run 0"

    def test_ensure_mixture_with_tokenize(self, tmp_path: Path) -> None:
        """Test ensure_mixture after tokenization with token-level counting."""
        ds_code, ds_text = _prepare_datasets(tmp_path, code_count=50, text_count=50)

        work = StaticMixtureWorkSource(
            [ds_code, ds_text],
            {"code": 0.5, "text": 0.5},
            chunk_size=20,
            seed=42,
        )

        pipe = (
            Pipeline(work)
            .decode_text()
            .tokenize(
                tokenizer_id="__fallback__",
                field="text",
                preserve_upstream_payload=True,
            )
            .ensure_mixture(
                max_buffer_size=100,  # 100 tokens
                weight_by="auto",
            )
            .options(deterministic=True, max_workers=2)
        )

        records = _collect_records(pipe)
        counts = _count_components(records)

        # Verify we processed records from both components
        assert counts["code"] > 0, "Expected some code samples"
        assert counts["text"] > 0, "Expected some text samples"

    def test_ensure_mixture_explicit_weights_override(self, tmp_path: Path) -> None:
        """Test that explicit weights override WorkSource mixture."""
        ds_code, ds_text = _prepare_datasets(tmp_path, code_count=100, text_count=100)

        # WorkSource has 30/70 ratio. Pin exhausted_policy="stop" (pre-flip
        # default): this test calibrates front-loading to a bounded one-pass
        # input, not the repeating stop_after_passes default.
        work = StaticMixtureWorkSource(
            [ds_code, ds_text],
            {"code": 0.3, "text": 0.7},
            chunk_size=20,
            seed=42,
            exhausted_policy="stop",
        )

        # But ensure_mixture overrides to 80/20
        pipe = (
            Pipeline(work)
            .decode_text()
            .ensure_mixture(
                max_buffer_size=20,
                weight_by="samples",
                mixture={"code": 0.8, "text": 0.2},
            )
            .options(deterministic=True, max_workers=2)
        )

        records = _collect_records(pipe)

        # Verify front-loading: SWRR with 80% code target front-loads code
        # Input is ~30% code globally (~42 code samples out of ~140 total)
        # Without reordering, first half would have ~30% code
        # With SWRR targeting 80%, first half should have nearly ALL code samples
        mid = len(records) // 2
        first_half_counts = _count_components(records[:mid])
        second_half_counts = _count_components(records[mid:])

        total_code = first_half_counts["code"] + second_half_counts["code"]
        first_code_ratio = first_half_counts["code"] / mid if mid > 0 else 0
        second_code_ratio = (
            second_half_counts["code"] / (len(records) - mid)
            if len(records) > mid
            else 0
        )

        # First half should have nearly double the input ratio (~30% -> ~55-60%)
        # This proves SWRR is front-loading code (without reordering it'd be ~30%)
        assert first_code_ratio >= 0.55, (
            f"Expected first half to have at least 55% code (input is ~30%, "
            f"SWRR targets 80%), got {first_code_ratio:.2%}"
        )

        # Second half should have very little code (most consumed in first half)
        # Without reordering it'd be ~30%, with front-loading it should be <15%
        assert second_code_ratio <= 0.15, (
            f"Expected second half to have at most 15% code (code depleted), "
            f"got {second_code_ratio:.2%}"
        )

        # Verify most code ended up in first half (front-loading)
        assert first_half_counts["code"] > 0.8 * total_code, (
            f"Expected >80% of code samples in first half, "
            f"got {first_half_counts['code']}/{total_code}"
        )

    @pytest.mark.parametrize("runner_kind", ["threads", "inline"])
    def test_ensure_mixture_runner_parity(
        self, tmp_path: Path, runner_kind: str
    ) -> None:
        """Test that different runners produce same output with ensure_mixture."""
        ds_code, ds_text = _prepare_datasets(tmp_path, code_count=40, text_count=40)

        def run_pipeline(runner: str) -> list[tuple[int, int, int]]:
            work = StaticMixtureWorkSource(
                [ds_code, ds_text],
                {"code": 0.5, "text": 0.5},
                chunk_size=16,
                seed=42,
            )

            pipe = (
                Pipeline(work)
                .decode_text()
                .ensure_mixture(max_buffer_size=8, weight_by="samples")
                .options(deterministic=True, runner=runner, max_workers=4)
            )

            return [rec.meta.sample_id for rec in _collect_records(pipe)]

        ids_threads = run_pipeline("threads")
        ids_other = run_pipeline(runner_kind)

        assert ids_threads == ids_other, (
            f"Output differs between threads and {runner_kind}"
        )

    def test_ensure_mixture_preserves_all_records(self, tmp_path: Path) -> None:
        """Test that ensure_mixture doesn't drop any records (reorder mode)."""
        ds_code, ds_text = _prepare_datasets(tmp_path, code_count=30, text_count=30)

        work = StaticMixtureWorkSource(
            [ds_code, ds_text],
            {"code": 0.5, "text": 0.5},
            chunk_size=20,
            seed=42,
        )

        # Without ensure_mixture
        pipe_without = (
            Pipeline(work).decode_text().options(deterministic=True, max_workers=2)
        )
        records_without = _collect_records(pipe_without)

        # Reset work source
        work2 = StaticMixtureWorkSource(
            [ds_code, ds_text],
            {"code": 0.5, "text": 0.5},
            chunk_size=20,
            seed=42,
        )

        # With ensure_mixture
        pipe_with = (
            Pipeline(work2)
            .decode_text()
            .ensure_mixture(max_buffer_size=10, weight_by="samples")
            .options(deterministic=True, max_workers=2)
        )
        records_with = _collect_records(pipe_with)

        # Same number of records (no drops)
        assert len(records_with) == len(records_without), (
            f"ensure_mixture changed record count: {len(records_with)} vs {len(records_without)}"
        )

        # Same set of sample_ids (just reordered)
        ids_without = {rec.meta.sample_id for rec in records_without}
        ids_with = {rec.meta.sample_id for rec in records_with}
        assert ids_without == ids_with, "ensure_mixture changed the set of records"

    def test_ensure_mixture_warn_mode(self, tmp_path: Path) -> None:
        """Test that warn mode works without errors."""
        ds_code, ds_text = _prepare_datasets(tmp_path, code_count=50, text_count=50)

        work = StaticMixtureWorkSource(
            [ds_code, ds_text],
            {"code": 0.5, "text": 0.5},
            chunk_size=20,
            seed=42,
        )

        pipe = (
            Pipeline(work)
            .decode_text()
            .ensure_mixture(
                max_buffer_size=10,
                weight_by="samples",
                warn_tolerance=0.05,
            )
            .options(deterministic=True, max_workers=2)
        )

        # Should complete without errors
        records = _collect_records(pipe)
        assert len(records) > 0

    def test_strict_vs_bounded_mixture_quality_tradeoff(self, tmp_path: Path) -> None:
        """Strict mode (max_buffer_size=None) hits the target mixture far more
        closely than bounded mode on an imbalanced stream — at the cost of
        dropping the surplus it cannot place on-target.

        The work source emits a code-heavy stream (~90% code) while the target
        is 50/50. Bounded mode never discards, so reordering cannot change its
        counts: its output keeps the input's skew. Strict mode discards the
        unplaceable code surplus, so its output is balanced but smaller.
        """
        ds_code, ds_text = _prepare_datasets(tmp_path, code_count=300, text_count=300)

        def make_work() -> StaticMixtureWorkSource:
            return StaticMixtureWorkSource(
                [ds_code, ds_text],
                {"code": 0.9, "text": 0.1},  # code-heavy supply
                chunk_size=20,
                seed=42,
            )

        def run(max_buffer_size: int | None) -> dict[str, int]:
            pipe = (
                Pipeline(make_work())
                .decode_text()
                .ensure_mixture(
                    max_buffer_size=max_buffer_size,
                    weight_by="samples",
                    mixture={"code": 0.5, "text": 0.5},  # but we want 50/50
                )
                .options(deterministic=True, max_workers=2)
            )
            return _count_components(_collect_records(pipe))

        # Baseline: same stream, no enforcement — fixes the input skew and size.
        baseline = _count_components(
            _collect_records(
                Pipeline(make_work())
                .decode_text()
                .options(deterministic=True, max_workers=2)
            )
        )
        bounded = run(20)
        strict = run(None)

        def total(c: dict[str, int]) -> int:
            return c["code"] + c["text"]

        def code_ratio(c: dict[str, int]) -> float:
            return c["code"] / total(c) if total(c) else 0.0

        assert total(baseline) > 0
        assert code_ratio(baseline) > 0.7, "baseline supply should be code-heavy"

        # Bounded mode never discards: same counts as the unenforced stream, so
        # it is lossless but cannot undo the skew.
        assert total(bounded) == total(baseline)
        assert code_ratio(bounded) > 0.7

        # Strict mode reaches the 50/50 target...
        assert 0.4 <= code_ratio(strict) <= 0.6
        assert abs(code_ratio(strict) - 0.5) < abs(code_ratio(bounded) - 0.5)

        # ...by dropping the code surplus it could not place on-target.
        assert 0 < total(strict) < total(baseline)
