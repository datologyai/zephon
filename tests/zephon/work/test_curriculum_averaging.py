# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Check per-step plans against an independent logical-sample reference."""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from zephon.work import (
    MixturePhase,
    PhaseConstantSchedule,
    PhaseLinearSchedule,
    StepMixturePlan,
)
from zephon.work.curriculum import _RoundTable


def _compile(plan: StepMixturePlan) -> _RoundTable:
    return plan._compile(samples_per_step=32, chunk_size=128, canonical_replicas=8)


# Readable examples; test_matches_independent_sample_oracle covers the rest.
@pytest.mark.parametrize(
    "schedule_type,points,expected",
    [
        # A switch at step 8 lands a quarter into round 0 (32 steps per round).
        (PhaseConstantSchedule, [(0, 0.3), (8, 0.6)], [0.525, 0.6]),
        (PhaseLinearSchedule, [(0, 0.2), (40, 0.6)], [0.355, 0.58875, 0.6]),
    ],
    ids=["switch-inside-round", "ramp"],
)
def test_round_weights_average_the_steps_they_cover(
    schedule_type: type[PhaseConstantSchedule | PhaseLinearSchedule],
    points: list[tuple[int, float]],
    expected: list[float],
) -> None:
    plan = StepMixturePlan(
        schedules={
            "a": schedule_type(points),
            "b": schedule_type([(step, 1 - weight) for step, weight in points]),
        },
        total_steps=64,
        boundary="warn",
    )
    result = _compile(plan)
    assert [result.weights_at(r)["a"] for r in range(len(expected))] == pytest.approx(
        expected
    )


@pytest.mark.parametrize("version", [1, 2])
def test_plan_serialization_preserves_compiled_weights(version: int) -> None:
    if version == 1:
        plan = StepMixturePlan(
            [
                MixturePhase(0, {"a": 0.3, "b": 0.7}),
                MixturePhase(8, {"a": 0.6, "b": 0.4}),
            ],
            total_steps=64,
            boundary="warn",
        )
    else:
        plan = StepMixturePlan(
            schedules={
                "a": PhaseLinearSchedule([(0, 0.2), (40, 0.6)]),
                "b": PhaseLinearSchedule([(0, 0.8), (40, 0.4)]),
            },
            total_steps=64,
            boundary="warn",
        )
    payload = plan.to_dict()
    assert payload["version"] == version
    restored = StepMixturePlan.from_dict(payload)
    assert restored == plan
    restored_table, table = _compile(restored), _compile(plan)
    assert restored_table._starts.tolist() == table._starts.tolist()
    assert restored_table._rows.tolist() == table._rows.tolist()


def test_large_constant_ranges_stay_sparse() -> None:
    plan = StepMixturePlan(
        [
            MixturePhase(0, {"a": 0.3, "b": 0.7}),
            MixturePhase(100_000_001, {"a": 0.6, "b": 0.4}),
        ],
        total_steps=200_000_000,
        boundary="warn",
    )
    table = _compile(plan)
    assert table._starts.tolist() == [0, 3_125_000, 3_125_001]


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (
            {
                "phases": [MixturePhase(0, {"a": 1})],
                "schedules": {"a": PhaseConstantSchedule([(0, 1)])},
            },
            "exactly one",
        ),
        ({"schedules": {"a": PhaseLinearSchedule([(0, 1), (9, 2)])}}, "outside"),
        ({"schedules": {"a": PhaseLinearSchedule([(1, 1)])}}, "start at step 0"),
        (
            {
                "schedules": {
                    "a": PhaseLinearSchedule([(0, 1), (2, 0), (3, 1)]),
                    "b": PhaseConstantSchedule([(0, 0)]),
                }
            },
            "positive finite total",
        ),
    ],
    ids=["both-inputs", "past-total-steps", "late-start", "zero-total"],
)
def test_invalid_schedule_plans_are_rejected(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        StepMixturePlan(**kwargs, total_steps=8)


def test_normalized_peaks_cover_both_sides_of_constant_jump() -> None:
    plan = StepMixturePlan(
        schedules={
            "a": PhaseLinearSchedule([(0, 1), (10, 2)]),
            "b": PhaseConstantSchedule([(0, 1), (5, 100)]),
        },
        total_steps=20,
        boundary="warn",
    )
    assert plan.peak_proportions == pytest.approx({"a": 1.4 / 2.4, "b": 100 / 101.5})


@pytest.mark.parametrize("seed", range(200))
def test_matches_independent_sample_oracle(seed: int) -> None:
    rng = random.Random(seed)
    schedules = {}
    for name in ("a", "b", "c"):
        positions = [0] + sorted(rng.sample(range(1, 15), rng.randint(1, 5)))
        points = [(position, rng.randint(1, 20) / 10) for position in positions]
        schedule_type = rng.choice([PhaseConstantSchedule, PhaseLinearSchedule])
        schedules[name] = schedule_type(points)
    samples_per_step = rng.randint(1, 13)
    chunk_size = rng.randint(1, 9)
    canon = rng.randint(1, 4)
    plan = StepMixturePlan(schedules=schedules, total_steps=20, boundary="warn")
    result = plan._compile(
        samples_per_step=samples_per_step,
        chunk_size=chunk_size,
        canonical_replicas=canon,
    )
    length = chunk_size * canon
    for round_index in range(math.ceil(20 * samples_per_step / length) + 2):
        expected = {name: [] for name in schedules}
        for source_sample in range(round_index * length, (round_index + 1) * length):
            step = source_sample // samples_per_step
            # Enumerate source-sample slots, independently of compiler splitting.
            values = {
                name: schedule.value(step) for name, schedule in schedules.items()
            }
            denominator = sum(values.values())
            for name in schedules:
                expected[name].append(values[name] / denominator)
        actual = result.weights_at(round_index)
        assert actual == pytest.approx(
            {name: math.fsum(values) / length for name, values in expected.items()},
            abs=1e-12,
        )
        assert all(
            actual[name] <= plan.peak_proportions[name] + 1e-12 for name in schedules
        )
