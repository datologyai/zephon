# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Property-based differential tests for checkpoint and recovery semantics."""

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.fuzzing.checkpoint_oracle import (
    ObservedWindow,
    assert_same_record_multiset,
    assert_source_contributors_closed_once,
    assert_source_ids_delivered_once,
    consume_windows,
)
from zephon import Dataset, InMemoryShard, Pipeline
from zephon.work import StaticMixtureWorkSource

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:\\[zephon\\] Checkpoint taken mid-window:RuntimeWarning"
    ),
]

Runner = Literal["inline", "threads"]
ShuffleAlgorithm = Literal["streaming", "block", "block_warmup"]

_MAX_EXAMPLES = int(os.environ.get("ZEPHON_FUZZ_EXAMPLES", "12"))
_DERANDOMIZE = os.environ.get("ZEPHON_FUZZ_DERANDOMIZE", "1") != "0"


@dataclass(frozen=True)
class CheckpointScenario:
    """Pipeline and runtime choices varied by the checkpoint fuzzer."""

    sample_count: int
    chunk_size: int
    canonical_replicas: int
    max_workers: int
    stage_prefetch: int
    final_prefetch: int
    queue_capacity: int
    batch_size: int | None
    pack_max_length: int | None
    shuffle_algorithm: ShuffleAlgorithm | None
    shuffle_buffer: int
    seed: int
    runners: tuple[Runner, ...]


@st.composite
def checkpoint_scenarios(draw: st.DrawFn) -> CheckpointScenario:
    """Generate bounded scenarios that remain fast enough for integration CI."""
    sample_count = draw(st.integers(min_value=16, max_value=48))
    output_shape = draw(st.sampled_from(["records", "batch", "pack"]))
    with_shuffle = draw(st.booleans())
    return CheckpointScenario(
        sample_count=sample_count,
        chunk_size=draw(st.sampled_from([1, 2, 4, 8])),
        canonical_replicas=draw(st.sampled_from([1, 2, 4])),
        max_workers=draw(st.integers(min_value=1, max_value=4)),
        stage_prefetch=draw(st.sampled_from([0, 1, 2, 4])),
        final_prefetch=draw(st.sampled_from([0, 1, 4, 8])),
        queue_capacity=draw(st.sampled_from([1, 2, 4, 8])),
        batch_size=(
            draw(st.integers(min_value=2, max_value=8))
            if output_shape == "batch"
            else None
        ),
        pack_max_length=(
            draw(st.sampled_from([8, 12, 16])) if output_shape == "pack" else None
        ),
        shuffle_algorithm=(
            draw(st.sampled_from(["streaming", "block", "block_warmup"]))
            if with_shuffle
            else None
        ),
        shuffle_buffer=draw(st.integers(min_value=2, max_value=16)),
        seed=draw(st.integers(min_value=0, max_value=2**32 - 1)),
        runners=tuple(
            draw(
                st.lists(
                    st.sampled_from(["inline", "threads"]),
                    min_size=2,
                    max_size=4,
                )
            )
        ),
    )


def _make_pipeline(scenario: CheckpointScenario, *, runner: Runner) -> Pipeline:
    rows = [{"text": f"sample-{index}"} for index in range(scenario.sample_count)]
    dataset = Dataset.from_dict("checkpoint-fuzz", {0: InMemoryShard(rows)})
    work = StaticMixtureWorkSource(
        [dataset],
        {dataset.name: 1.0},
        chunk_size=scenario.chunk_size,
        seed=scenario.seed,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipeline = Pipeline(work).decode_text(parallelism=scenario.max_workers)
    if scenario.pack_max_length is not None:
        pipeline = pipeline.map_transform(_add_token_ids)
    if scenario.shuffle_algorithm is not None:
        pipeline = pipeline.shuffle(
            buffer_size=scenario.shuffle_buffer,
            seed=scenario.seed,
            algorithm=scenario.shuffle_algorithm,
        )
    if scenario.batch_size is not None:
        pipeline = pipeline.batch(scenario.batch_size, drop_last=False)
    if scenario.pack_max_length is not None:
        pipeline = pipeline.pack_sequences(
            max_length=scenario.pack_max_length,
            num_bins=2,
            algorithm="first_fit",
            tokens_field="input_ids",
        )
    return pipeline.options(
        deterministic=True,
        runner=runner,
        worker_allocation="per_stage_fixed",
        max_workers=scenario.max_workers,
        canonical_replicas=scenario.canonical_replicas,
        world_size=1,
        global_rank=0,
        dp_degree=1,
        dp_group_id=0,
        default_stage_prefetch=scenario.stage_prefetch,
        prefetch_batches=scenario.final_prefetch,
        op_queue_capacity=scenario.queue_capacity,
        allow_latency_flush_in_deterministic=False,
        mtp_mode=False,
    )


def _checkpoint_cuts(window_count: int, numerators: list[int]) -> list[int]:
    assert window_count > 1
    return sorted(
        {
            max(1, min(window_count - 1, window_count * numerator // 16))
            for numerator in numerators
        }
    )


def _expected_sample_ids(scenario: CheckpointScenario) -> list[tuple[int, int, int]]:
    """Return source IDs in complete chunks produced by StaticMixtureWorkSource."""
    delivered_count = scenario.sample_count // scenario.chunk_size * scenario.chunk_size
    return [(0, 0, index) for index in range(delivered_count)]


def _add_token_ids(payload: dict) -> dict:
    """Attach deterministic variable-length tokens for generated pack cases."""
    index = int(payload["text"].rsplit("-", 1)[1])
    return {**payload, "input_ids": list(range(1, 2 + index % 5))}


def _assert_source_coverage(
    windows: list[ObservedWindow],
    scenario: CheckpointScenario,
) -> None:
    expected = _expected_sample_ids(scenario)
    if scenario.pack_max_length is None:
        assert_source_ids_delivered_once(windows, expected)
    else:
        assert_source_contributors_closed_once(windows, expected)


@given(
    scenario=checkpoint_scenarios(),
    cut_numerators=st.lists(
        st.integers(min_value=1, max_value=15),
        min_size=1,
        max_size=3,
        unique=True,
    ),
)
@settings(
    max_examples=_MAX_EXAMPLES,
    deadline=None,
    derandomize=_DERANDOMIZE,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_checkpoint_recovery_preserves_documented_identity(
    scenario: CheckpointScenario, cut_numerators: list[int]
) -> None:
    """Generated checkpoint cycles reconstruct the exact identity stream."""
    baseline, _ = consume_windows(_make_pipeline(scenario, runner="inline"))
    assert len(baseline) > 1

    _assert_source_coverage(baseline, scenario)

    observed: list[ObservedWindow] = []
    checkpoint: dict | None = None
    cuts = _checkpoint_cuts(len(baseline), cut_numerators)
    for cycle, target in enumerate(cuts):
        pipeline = _make_pipeline(
            scenario, runner=scenario.runners[cycle % len(scenario.runners)]
        )
        if checkpoint is not None:
            pipeline.restore(deepcopy(checkpoint))
        segment, checkpoint = consume_windows(
            pipeline,
            limit=target - len(observed),
            checkpoint=True,
        )
        assert checkpoint is not None
        observed.extend(segment)
        assert observed == baseline[:target]

    pipeline = _make_pipeline(
        scenario, runner=scenario.runners[len(cuts) % len(scenario.runners)]
    )
    assert checkpoint is not None
    pipeline.restore(deepcopy(checkpoint))
    suffix, _ = consume_windows(pipeline)
    observed.extend(suffix)

    assert observed == baseline
    _assert_source_coverage(observed, scenario)


def test_multilane_shuffle_batch_replays_exactly() -> None:
    """The minimized cross-lane shuffle counterexample replays exactly."""
    scenario = CheckpointScenario(
        sample_count=16,
        chunk_size=1,
        canonical_replicas=2,
        max_workers=1,
        stage_prefetch=0,
        final_prefetch=0,
        queue_capacity=1,
        batch_size=2,
        pack_max_length=None,
        shuffle_algorithm="streaming",
        shuffle_buffer=2,
        seed=0,
        runners=("inline", "inline"),
    )
    baseline, _ = consume_windows(_make_pipeline(scenario, runner="inline"))

    prefix_pipeline = _make_pipeline(scenario, runner="inline")
    prefix, checkpoint = consume_windows(
        prefix_pipeline,
        limit=3,
        checkpoint=True,
    )
    assert checkpoint is not None

    resumed_pipeline = _make_pipeline(scenario, runner="inline")
    resumed_pipeline.restore(deepcopy(checkpoint))
    suffix, _ = consume_windows(resumed_pipeline)
    observed = prefix + suffix

    expected_ids = _expected_sample_ids(scenario)
    assert_same_record_multiset(observed, baseline)
    assert_source_ids_delivered_once(observed, expected_ids)
    assert observed == baseline


def test_multilane_shuffle_pack_replays_exactly() -> None:
    """Shuffle then pack finds the same replay cursor after restoration."""
    scenario = CheckpointScenario(
        sample_count=32,
        chunk_size=1,
        canonical_replicas=2,
        max_workers=1,
        stage_prefetch=0,
        final_prefetch=0,
        queue_capacity=1,
        batch_size=None,
        pack_max_length=8,
        shuffle_algorithm="streaming",
        shuffle_buffer=2,
        seed=0,
        runners=("inline", "inline"),
    )
    baseline, _ = consume_windows(_make_pipeline(scenario, runner="inline"))

    prefix_pipeline = _make_pipeline(scenario, runner="inline")
    prefix, checkpoint = consume_windows(
        prefix_pipeline,
        limit=3,
        checkpoint=True,
    )
    assert checkpoint is not None

    resumed_pipeline = _make_pipeline(scenario, runner="inline")
    resumed_pipeline.restore(deepcopy(checkpoint))
    suffix, _ = consume_windows(resumed_pipeline)
    observed = prefix + suffix

    assert_same_record_multiset(observed, baseline)
    _assert_source_coverage(observed, scenario)
    assert observed == baseline


def test_repeated_checkpoint_restores_tail_round_robin_pointer() -> None:
    """A restored, aligned checkpoint keeps the next physical lane exact."""
    scenario = CheckpointScenario(
        sample_count=44,
        chunk_size=2,
        canonical_replicas=4,
        max_workers=1,
        stage_prefetch=0,
        final_prefetch=0,
        queue_capacity=1,
        batch_size=None,
        pack_max_length=None,
        shuffle_algorithm=None,
        shuffle_buffer=2,
        seed=0,
        runners=("inline", "inline"),
    )
    baseline, _ = consume_windows(_make_pipeline(scenario, runner="inline"))

    first_pipeline = _make_pipeline(scenario, runner="inline")
    first, first_checkpoint = consume_windows(
        first_pipeline,
        limit=2,
        checkpoint=True,
    )
    assert first_checkpoint is not None
    assert next(iter(first_checkpoint["rr_next_idx"].values())) == 2

    second_pipeline = _make_pipeline(scenario, runner="inline")
    second_pipeline.restore(deepcopy(first_checkpoint))
    second, second_checkpoint = consume_windows(
        second_pipeline,
        limit=6,
        checkpoint=True,
    )
    assert second_checkpoint is not None
    assert second_checkpoint["lane_emitted"] == dict.fromkeys(range(4), 2)
    assert next(iter(second_checkpoint["rr_next_idx"].values())) == 0

    final_pipeline = _make_pipeline(scenario, runner="inline")
    final_pipeline.restore(deepcopy(second_checkpoint))
    final, _ = consume_windows(final_pipeline)

    assert first + second + final == baseline
