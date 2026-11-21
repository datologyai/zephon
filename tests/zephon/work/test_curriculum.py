# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import pytest

from zephon.work import (
    ConstantSchedule,
    MixtureSchedule,
    PhaseConstantSchedule,
    PhaseLinearSchedule,
)


def _crossfade(total_rounds: int) -> MixtureSchedule:
    return MixtureSchedule(
        {
            "alpha": PhaseLinearSchedule([(0, 1.0), (total_rounds, 0.0)]),
            "beta": PhaseLinearSchedule([(0, 0.0), (total_rounds, 1.0)]),
        }
    )


def test_phase_linear_schedule_interpolates_and_holds() -> None:
    schedule = PhaseLinearSchedule([(10, 1.0), (110, 0.0)])
    assert schedule.value(0) == 1.0
    assert math.isclose(schedule.value(60), 0.5)
    assert schedule.value(110) == 0.0
    assert schedule.value(500) == 0.0
    assert schedule.max_weight() == 1.0


@pytest.mark.parametrize("cls", [PhaseConstantSchedule, PhaseLinearSchedule])
def test_breakpoint_schedule_round_trip(cls: type) -> None:
    schedule = cls([(0, 0.8), (250, 0.2)])
    payload = schedule.to_dict()
    assert payload == {"type": cls.__name__, "breakpoints": [[0, 0.8], [250, 0.2]]}
    assert cls.from_dict(payload) == schedule


def test_mixture_schedule_normalizes() -> None:
    assert _crossfade(10).weights_at(3) == pytest.approx({"alpha": 0.7, "beta": 0.3})


def test_mixture_schedule_rejects_all_zero_outputs() -> None:
    schedule = MixtureSchedule({"a": 0.0, "b": 0.0})
    with pytest.raises(ValueError, match="positive finite total"):
        schedule.weights_at(0)


class _ValueOnly:
    def value(self, position: int) -> float:
        return 1.0


@pytest.mark.parametrize(
    "components,message",
    [
        ({1: 1.0}, "names must be strings"),
        ({"a": True, "b": False}, "must be a number"),
        ({"alpha": _ValueOnly(), "beta": 1.0}, "must be a number, ConstantSchedule"),
    ],
    ids=["non-string-name", "bool", "custom-schedule"],
)
def test_mixture_schedule_rejects_invalid_components(
    components: dict[Any, Any], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        MixtureSchedule(components)


def test_mixture_schedule_max_weight_is_normalized_peak() -> None:
    schedule = MixtureSchedule(
        {
            "alpha": PhaseConstantSchedule([(0, 1.0), (5, 3.0)]),
            "beta": PhaseLinearSchedule([(0, 1.0), (5, 3.0)]),
            "gamma": 0.0,
        }
    )
    # Beta peaks at round 4, the last round before alpha's jump: it has
    # ramped to 2.6 while alpha is still 1.0.
    assert schedule.max_weight("alpha") == pytest.approx(0.5)
    assert schedule.max_weight("beta") == pytest.approx(2.6 / 3.6)
    assert schedule.max_weight("gamma") == 0.0
    late = MixtureSchedule(
        {"alpha": PhaseConstantSchedule([(0, 0.0), (5, 3.0)]), "beta": 1.0}
    )
    assert late.max_weight("alpha") == pytest.approx(0.75)
    assert late.max_weight("beta") == 1.0
    ramp = MixtureSchedule(
        {"alpha": PhaseLinearSchedule([(0, 0.0), (10, 1.0)]), "beta": 1.0}
    )
    assert ramp.max_weight("alpha") == pytest.approx(0.5)


def test_schedule_to_dict_round_trip() -> None:
    original = MixtureSchedule(
        {
            "alpha": PhaseLinearSchedule([(0, 1.0), (50, 0.0)]),
            "beta": PhaseConstantSchedule([(0, 0.1), (25, 1.0)]),
            "gamma": ConstantSchedule(0.25),
        }
    )
    blob = original.to_dict()
    assert {name: s["type"] for name, s in blob["schedules"].items()} == {
        "alpha": "PhaseLinearSchedule",
        "beta": "PhaseConstantSchedule",
        "gamma": "ConstantSchedule",
    }

    restored = MixtureSchedule.from_dict(blob)
    for r in (0, 10, 25, 49, 50, 500):
        assert restored.weights_at(r) == original.weights_at(r)


def test_phase_constant_schedule_holds_until_each_breakpoint() -> None:
    schedule = PhaseConstantSchedule([(0, 0.8), (250, 0.2), (500, 0.6)])

    assert schedule.value(0) == pytest.approx(0.8)
    assert schedule.value(249) == pytest.approx(0.8)
    assert schedule.value(250) == pytest.approx(0.2)
    assert schedule.value(499) == pytest.approx(0.2)
    assert schedule.value(500) == pytest.approx(0.6)
    assert schedule.value(5_000) == pytest.approx(0.6)
    assert schedule.max_weight() == pytest.approx(0.8)


@pytest.mark.parametrize("cls", [PhaseConstantSchedule, PhaseLinearSchedule])
@pytest.mark.parametrize(
    "breakpoints,error",
    [
        ([], ValueError),
        ([(0, 0.8), (0, 0.2)], ValueError),
        ([(1, 0.8), (0, 0.2)], ValueError),
        ([(-1, 0.8)], ValueError),
        ([(0.5, 0.8)], TypeError),
        ([(0, "0.8")], TypeError),
        ([(0, -0.1)], ValueError),
        ([(0, math.inf)], ValueError),
    ],
)
def test_breakpoint_schedules_reject_invalid_breakpoints(
    cls: type, breakpoints: list[tuple[Any, Any]], error: type[Exception]
) -> None:
    with pytest.raises(error):
        cls(breakpoints)


@pytest.mark.parametrize(
    "payload,error,match",
    [
        ({"type": "ConstantSchedule", "weight": 1.0, "junk": 2}, ValueError, "unknown"),
        ({"type": "PhaseLinearSchedule", "weight": 1.0}, ValueError, "expected a"),
        ({"type": "ConstantSchedule"}, ValueError, "missing"),
        (["ConstantSchedule"], TypeError, "mapping"),
    ],
)
def test_schedule_from_dict_is_strict(
    payload: Any, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        ConstantSchedule.from_dict(payload)


def test_mixture_schedule_from_dict_is_strict() -> None:
    payload = MixtureSchedule({"alpha": 1.0}).to_dict()
    with pytest.raises(ValueError, match="unknown"):
        MixtureSchedule.from_dict({**payload, "extra": True})
    with pytest.raises(ValueError, match="Unknown schedule type"):
        MixtureSchedule.from_dict(
            {**payload, "schedules": {"alpha": {"type": "LinearSchedule"}}}
        )
