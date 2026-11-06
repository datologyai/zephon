# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import copy
import math
import random
from typing import Any, cast

import pytest

from zephon.io import Dataset, InMemoryShard
from zephon.work import (
    ConstantSchedule,
    MixturePhase,
    MixtureSchedule,
    PhaseConstantSchedule,
    PhaseLinearSchedule,
    StaticMixtureWorkSource,
    StepMixturePlan,
    TokenEstimation,
    WorkChunk,
)
from zephon.work.curriculum import _RoundTable
from zephon.work.static_mixture import LegacyFixedStrategy, _AllocationConfig


def _make_dataset(name: str, n: int) -> Dataset:
    rows = [{"text": f"{name}-{idx}"} for idx in range(n)]
    return Dataset.from_dict(name, {0: InMemoryShard(rows)})


def _two_dataset_work(
    samples: int,
    chunk_size: int,
    schedule: MixtureSchedule | None = None,
    **kwargs: Any,
) -> StaticMixtureWorkSource:
    # Endless by default: the crossfades below leave alpha at weight 0, which
    # a terminating source rejects.
    kwargs.setdefault("exhausted_policy", "repeat")
    return StaticMixtureWorkSource(
        [_make_dataset("alpha", samples), _make_dataset("beta", samples)],
        schedule if schedule is not None else {"alpha": 0.5, "beta": 0.5},
        chunk_size=chunk_size,
        seed=0,
        shuffle_shards=False,
        **kwargs,
    )


def _crossfade(total_rounds: int) -> MixtureSchedule:
    return MixtureSchedule(
        {
            "alpha": PhaseLinearSchedule([(0, 1.0), (total_rounds, 0.0)]),
            "beta": PhaseLinearSchedule([(0, 0.0), (total_rounds, 1.0)]),
        }
    )


def _make_curriculum_work() -> StaticMixtureWorkSource:
    # A fast crossfade makes any cross-lane weight leak change sample ids.
    return _two_dataset_work(4000, chunk_size=10, schedule=_crossfade(20))


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


def test_single_lane_schedule_advances_per_chunk_round() -> None:
    work = _two_dataset_work(200, chunk_size=10, schedule=_crossfade(10))
    ws = work.clone_for_lane(0, canonical_replicas=1)

    chunks = [ws.next_chunk() for _ in range(11)]
    assert all(chunk is not None for chunk in chunks)
    assert chunks[0].components.keys() == {"alpha"}
    assert len(chunks[5].components["alpha"]) == 5
    assert len(chunks[5].components["beta"]) == 5
    assert chunks[10].components.keys() == {"beta"}


def test_chunk_round_indexing_across_lanes_agrees() -> None:
    # With canonical_replicas=2 the lanes split global chunks by parity but
    # share each chunk round, so both must install that round's weights.
    schedule = _crossfade(4)
    work = _two_dataset_work(400, chunk_size=10, schedule=schedule)
    lane0 = work.clone_for_lane(0, canonical_replicas=2)
    lane1 = work.clone_for_lane(1, canonical_replicas=2)

    for chunk_round in range(4):
        assert lane0.next_chunk() is not None
        assert lane1.next_chunk() is not None
        expected = schedule.weights_at(chunk_round)
        assert lane0._alloc_config.weights == expected
        assert lane1._alloc_config.weights == expected


def test_schedule_deterministic_under_out_of_lockstep_lanes() -> None:
    ref_lane0 = _make_curriculum_work().clone_for_lane(0, canonical_replicas=2)
    reference = [ref_lane0.next_chunk().components for _ in range(8)]

    work = _make_curriculum_work()
    lane0 = work.clone_for_lane(0, canonical_replicas=2)
    lane1 = work.clone_for_lane(1, canonical_replicas=2)
    # Lane 0's next internal draw is a round-0 discard that skips re-applying
    # the schedule; lane 1 racing ahead must not change the weights it uses.
    got = [lane0.next_chunk().components]
    for _ in range(12):
        assert lane1.next_chunk() is not None
    got.extend(lane0.next_chunk().components for _ in range(7))

    assert got == reference


def test_schedule_deterministic_under_lanes_per_worker() -> None:
    references = []
    for lane_id in (0, 1):
        lane = _make_curriculum_work().clone_for_lane(lane_id, canonical_replicas=2)
        references.append([lane.next_chunk().components for _ in range(6)])

    work = _make_curriculum_work()
    lanes = [work.clone_for_lane(lane_id, canonical_replicas=2) for lane_id in (0, 1)]
    got: list[list[Any]] = [[], []]
    for lane_id in [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0]:
        got[lane_id].append(lanes[lane_id].next_chunk().components)

    assert got == references


@pytest.mark.parametrize(
    "names", [("alpha", "gamma"), ("alpha",), ("alpha", "beta", "gamma")]
)
def test_schedule_rejects_mismatched_components(names: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="must match work source dataset names"):
        _two_dataset_work(
            20, chunk_size=5, schedule=MixtureSchedule(dict.fromkeys(names, 1.0))
        )


def test_schedule_rejects_token_aware_source() -> None:
    with pytest.raises(ValueError, match="sample-denominated"):
        _two_dataset_work(
            20,
            chunk_size=5,
            schedule=_crossfade(4),
            token_estimation=TokenEstimation(primer=0.25),
        )


def test_schedule_rejects_redistribution() -> None:
    with pytest.raises(ValueError, match="redistribute"):
        _two_dataset_work(
            20,
            chunk_size=5,
            schedule=_crossfade(4),
            exhausted_policy={"alpha": "redistribute", "beta": "repeat"},
        )


def test_schedule_rejects_never_active_components() -> None:
    with pytest.raises(ValueError, match="never active"):
        _two_dataset_work(
            20, chunk_size=5, schedule=MixtureSchedule({"alpha": 1.0, "beta": 0.0})
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"exhausted_policy": {"alpha": "stop", "beta": "repeat"}},
        {"exhausted_policy": "repeat", "max_repeats": {"alpha": 2, "beta": None}},
    ],
    ids=["stop-after-passes", "stop", "capped-repeat"],
)
def test_schedule_rejects_endless_terminating_source(
    kwargs: dict[str, Any],
) -> None:
    # alpha alone can end the run, and the schedule leaves it at weight 0.
    with pytest.raises(ValueError, match="would never stop"):
        StaticMixtureWorkSource(
            [_make_dataset("alpha", 100), _make_dataset("beta", 100)],
            MixtureSchedule(
                {"alpha": PhaseConstantSchedule([(0, 1.0), (5, 0.0)]), "beta": 1.0}
            ),
            chunk_size=10,
            seed=0,
            shuffle_shards=False,
            **kwargs,
        )


def test_schedule_accepts_zero_weight_when_another_dataset_stops() -> None:
    work = StaticMixtureWorkSource(
        [_make_dataset("alpha", 100), _make_dataset("beta", 100)],
        _crossfade(4),
        chunk_size=10,
        seed=0,
        shuffle_shards=False,
        exhausted_policy="stop",
    )
    lane = work.clone_for_lane(0, canonical_replicas=1)
    chunks = 0
    while lane.next_chunk() is not None:
        chunks += 1
    assert 0 < chunks < 20


class _FeedbackSchedule:
    # A mixture-level duck type, like a trainer-feedback controller.
    component_names = ("alpha", "beta")

    def max_weight(self, name: str) -> float:
        return 1.0

    def weights_at(self, chunk_round: int) -> dict[str, float]:
        return {"alpha": 0.0, "beta": 1.0}


def test_duck_typed_schedules_are_rejected() -> None:
    with pytest.raises(TypeError, match="mixture must be"):
        _two_dataset_work(
            100, chunk_size=10, schedule=cast(MixtureSchedule, _FeedbackSchedule())
        )


def test_scheduled_source_has_no_length() -> None:
    plain = _two_dataset_work(100, chunk_size=10, exhausted_policy="stop")
    assert len(plain) == plain.total_samples == 200
    work = _two_dataset_work(
        100,
        chunk_size=10,
        schedule=MixtureSchedule({"alpha": 3.0, "beta": 1.0}),
        exhausted_policy="stop",
    )
    with pytest.raises(TypeError, match="scheduled work source"):
        len(work)
    with pytest.raises(RuntimeError, match="scheduled work source"):
        _ = work.total_samples


def test_schedule_survives_checkpoint_resume() -> None:
    work_save = _two_dataset_work(200, chunk_size=10, schedule=_crossfade(10))
    ws_save = work_save.clone_for_lane(0, canonical_replicas=1)
    for _ in range(5):
        assert ws_save.next_chunk() is not None
    state = ws_save.state_dict()
    assert "schedule" not in state

    work_load = _two_dataset_work(200, chunk_size=10, schedule=_crossfade(10))
    ws_load = work_load.clone_for_lane(0, canonical_replicas=1)
    ws_load.load_state_dict(state)

    for _ in range(3):
        assert ws_load.next_chunk().components == ws_save.next_chunk().components


def test_in_place_restore_reapplies_scheduled_round() -> None:
    # Restoring into a lane that already cached a later round must not keep
    # that round's weights for the restored round.
    work = StaticMixtureWorkSource(
        [_make_dataset("alpha", 40), _make_dataset("beta", 40)],
        MixtureSchedule(
            {
                "alpha": PhaseConstantSchedule([(0, 1.0), (2, 0.0)]),
                "beta": PhaseConstantSchedule([(0, 0.0), (2, 1.0)]),
            }
        ),
        chunk_size=4,
        seed=0,
        shuffle_shards=False,
        exhausted_policy="stop",
    )
    lane = work.clone_for_lane(0, canonical_replicas=1)
    for _ in range(2):
        assert lane.next_chunk().components.keys() == {"alpha"}
    state = copy.deepcopy(lane.state_dict())
    expected = lane.next_chunk().components
    assert expected.keys() == {"beta"}

    lane.load_state_dict(state)

    assert lane.next_chunk().components == expected


def test_repeat_guard_uses_schedule_peak_weight() -> None:
    # alpha peaks at 1.0, so one chunk can draw 10 alpha samples; it has 4.
    with pytest.raises(ValueError, match="schedule peak weight=100%") as excinfo:
        StaticMixtureWorkSource(
            [_make_dataset("alpha", 4), _make_dataset("beta", 200)],
            _crossfade(10),
            chunk_size=10,
            seed=0,
            shuffle_shards=False,
            exhausted_policy="repeat",
        )
    # Schedules reject redistribution, so the error must not suggest it.
    assert "redistribute" not in str(excinfo.value)


def _jump_to_half(chunk_size: int, c_samples: int) -> StaticMixtureWorkSource:
    return StaticMixtureWorkSource(
        [
            _make_dataset("a", 4 * chunk_size),
            _make_dataset("b", 4 * chunk_size),
            _make_dataset("c", c_samples),
        ],
        MixtureSchedule(
            {"a": 3.0, "b": 1.0, "c": PhaseConstantSchedule([(0, 1.0), (3, 4.0)])}
        ),
        chunk_size=chunk_size,
        seed=0,
        shuffle_shards=False,
        exhausted_policy="repeat",
    )


def test_repeat_guard_covers_integral_peak_share_after_weight_change() -> None:
    # c jumps from 0.2 to exactly 0.5: the remainder carried from the 0.2
    # rounds lets round 3 draw 513 of c, one more than its 512 share.
    with pytest.raises(ValueError, match="schedule peak weight"):
        _jump_to_half(1024, 512)
    lane = _jump_to_half(1024, 513).clone_for_lane(0, canonical_replicas=1)
    for _ in range(10):
        assert lane.next_chunk() is not None


def test_allocation_tracks_a_moving_schedule_cumulatively() -> None:
    # The accumulator carries remainders across weight changes, so even a
    # component below 1/chunk_size gets its scheduled share over time.
    chunk_size = 16
    schedule = MixtureSchedule(
        {
            "a": PhaseLinearSchedule([(0, 0.01), (200, 0.3), (400, 0.02)]),
            "b": PhaseConstantSchedule([(0, 1.0), (150, 0.5), (300, 1.0)]),
            "c": 0.05,
        }
    )
    work = StaticMixtureWorkSource(
        [_make_dataset(name, 20_000) for name in ("a", "b", "c")],
        schedule,
        chunk_size=chunk_size,
        seed=0,
        exhausted_policy="repeat",
    )
    lane = work.clone_for_lane(0, canonical_replicas=1)
    target = dict.fromkeys("abc", 0.0)
    drawn = dict.fromkeys("abc", 0)
    for chunk_round in range(500):
        for name, weight in schedule.weights_at(chunk_round).items():
            target[name] += weight * chunk_size
        for name, ids in lane.next_chunk().components.items():
            drawn[name] += len(ids)
        assert all(abs(drawn[n] - target[n]) <= 1.0 for n in "abc"), chunk_round


def _pause_c(c_samples: int, resume_round: int | None) -> StaticMixtureWorkSource:
    # Binary fractions keep the rounding exact: round 0 owes c half a sample.
    c_weights = [(0, 0.5), (1, 0.0)]
    if resume_round is not None:
        c_weights.append((resume_round, 0.5))
    return StaticMixtureWorkSource(
        [
            _make_dataset("a", 100),
            _make_dataset("b", 100),
            _make_dataset("c", c_samples),
        ],
        MixtureSchedule(
            {
                "a": PhaseConstantSchedule([(0, 0.25), (1, 0.5)]),
                "b": PhaseConstantSchedule([(0, 0.25), (1, 0.5)]),
                "c": PhaseConstantSchedule(c_weights),
            }
        ),
        chunk_size=3,
        seed=0,
        shuffle_shards=False,
        exhausted_policy="stop",
    )


def _counts(chunk: WorkChunk | None) -> dict[str, int]:
    assert chunk is not None
    return {name: len(ids) for name, ids in chunk.components.items()}


def test_zero_weight_component_takes_no_draws_until_reactivated() -> None:
    lane = _pause_c(100, resume_round=3).clone_for_lane(0, canonical_replicas=1)
    counts = [_counts(lane.next_chunk()) for _ in range(5)]
    assert counts[0] == {"a": 1, "b": 1, "c": 1}
    assert all("c" not in c and sum(c.values()) == 3 for c in counts[1:3])
    assert all(c.get("c", 0) >= 1 for c in counts[3:])


def test_exhausted_zero_weight_component_does_not_stop_the_source() -> None:
    lane = _pause_c(1, resume_round=None).clone_for_lane(0, canonical_replicas=1)
    assert _counts(lane.next_chunk())["c"] == 1
    for _ in range(10):
        assert "c" not in _counts(lane.next_chunk())


def test_replay_across_zero_weight_transitions() -> None:
    lane = _pause_c(100, resume_round=3).clone_for_lane(0, canonical_replicas=1)
    states, chunks = [], []
    for _ in range(6):
        states.append(copy.deepcopy(lane.state_dict()))
        chunks.append(lane.next_chunk())
    for start, state in enumerate(states):
        resumed = _pause_c(100, resume_round=3).clone_for_lane(0, canonical_replicas=1)
        resumed.load_state_dict(state)
        for expected in chunks[start:]:
            assert resumed.next_chunk().components == expected.components


def test_legacy_fixed_strategy_set_weights_raises() -> None:
    cfg = _AllocationConfig(
        component_order=("a", "b"),
        weights={"a": 0.5, "b": 0.5},
        chunk_size=10,
        exhausted_policy={"a": "stop", "b": "stop"},
        reshuffle_on_repeat={"a": False, "b": False},
        max_repeats={"a": None, "b": None},
    )
    with pytest.raises(NotImplementedError, match="fixed quotas"):
        LegacyFixedStrategy(config=cfg).set_weights({"a": 0.9, "b": 0.1})


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


def _scheduled_lane_state(schedule: MixtureSchedule) -> dict[str, Any]:
    work = _two_dataset_work(100, chunk_size=10, schedule=schedule)
    lane = work.clone_for_lane(0, canonical_replicas=1)
    for _ in range(3):
        lane.next_chunk()
    return lane.state_dict()


def _restore_into(schedule: MixtureSchedule | None, state: dict[str, Any]) -> None:
    work = _two_dataset_work(100, chunk_size=10, schedule=schedule)
    work.clone_for_lane(0, canonical_replicas=1).load_state_dict(state)


def test_checkpoint_records_the_attached_schedule() -> None:
    state = _scheduled_lane_state(_crossfade(10))
    assert state["schedule_fingerprint"] == _crossfade(10)._fingerprint()

    _restore_into(_crossfade(10), state)
    with pytest.raises(RuntimeError, match="with the same schedule"):
        _restore_into(None, state)
    with pytest.raises(RuntimeError, match="with the same schedule"):
        _restore_into(_crossfade(20), state)


@pytest.mark.parametrize("pre_v7", [False, True])
def test_unscheduled_checkpoint_rejects_an_attached_schedule(pre_v7: bool) -> None:
    lane = _two_dataset_work(100, chunk_size=10).clone_for_lane(0, canonical_replicas=1)
    lane.next_chunk()
    state = lane.state_dict()
    assert state["schedule_fingerprint"] is None
    if pre_v7:
        del state["schedule_fingerprint"]
        state["version"] = 6
    _restore_into(None, state)
    with pytest.raises(RuntimeError, match="written with no schedule"):
        _restore_into(_crossfade(10), state)


@pytest.mark.parametrize(
    "phases,total_steps,message",
    [
        (
            [MixturePhase(step=1, proportions={"a": 1.0})],
            10,
            "start at step 0",
        ),
        (
            [
                MixturePhase(step=0, proportions={"a": 1.0}),
                MixturePhase(step=0, proportions={"a": 1.0}),
            ],
            10,
            "strictly increasing",
        ),
        (
            [
                MixturePhase(step=0, proportions={"a": 1.0}),
                MixturePhase(step=5, proportions={"b": 1.0}),
            ],
            10,
            "component mismatch",
        ),
        (
            [MixturePhase(step=0, proportions={"a": 0.7, "b": 0.2})],
            10,
            "not 1.0",
        ),
        (
            [MixturePhase(step=0, proportions={"a": 1.0})],
            0,
            "positive integer",
        ),
        (
            [
                MixturePhase(step=0, proportions={"a": 1.0}),
                MixturePhase(step=10, proportions={"a": 1.0}),
            ],
            10,
            "smaller than total_steps",
        ),
        (
            [MixturePhase(step=0, proportions={"a": 1.0, "b": 0.0})],
            10,
            "never active",
        ),
    ],
)
def test_invalid_step_mixture_plan_is_rejected(
    phases: list[MixturePhase], total_steps: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        StepMixturePlan(phases=phases, total_steps=total_steps)


@pytest.mark.parametrize(
    "phase,error,message",
    [
        ({"proportions": {"a": 1.0}}, ValueError, "missing 'step'"),
        ({"step": 0}, ValueError, "missing 'proportions'"),
        ({"step": 1.5, "proportions": {"a": 1.0}}, TypeError, "must be an integer"),
        ({"step": 0, "proportions": {"a": True}}, TypeError, "must be a number"),
        (
            {"step": 0, "proportions": {"a": 1.0}, "name": "phase-a"},
            ValueError,
            "unknown MixturePhase fields",
        ),
    ],
)
def test_mixture_phase_rejects_malformed_payloads(
    phase: dict, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        MixturePhase.from_dict(phase)


def test_step_mixture_plan_repeat_guard_accepts_dataset_at_scheduled_bound() -> None:
    # alpha peaks at 0.8 of a 10-sample chunk; a varying weight can draw one more.
    work = StaticMixtureWorkSource(
        [_make_dataset("alpha", 9), _make_dataset("beta", 9)],
        StepMixturePlan(
            phases=[
                MixturePhase(0, {"alpha": 0.8, "beta": 0.2}),
                MixturePhase(1, {"alpha": 0.2, "beta": 0.8}),
            ],
            total_steps=2,
        ),
        chunk_size=10,
        seed=0,
        shuffle_shards=False,
        exhausted_policy="repeat",
    )

    work._prepare_for_execution(
        canonical_replicas=1,
        output_batch_size=10,
        preserves_cursor_order=True,
        global_batch_size=10,
    )


def test_step_mixture_plan_rejects_reordering_pipeline() -> None:
    work = StaticMixtureWorkSource(
        [_make_dataset("alpha", 100), _make_dataset("beta", 100)],
        _two_phase_plan(8_000, boundary="warn"),
        chunk_size=128,
        seed=0,
        shuffle_shards=False,
    )

    with pytest.raises(ValueError, match="cursor-order-preserving pipeline"):
        work._prepare_for_execution(
            canonical_replicas=8,
            output_batch_size=4,
            preserves_cursor_order=False,
            global_batch_size=32,
        )


def test_step_plan_switches_every_lane_at_the_same_step() -> None:
    plan = StepMixturePlan.from_dict(
        {
            "type": "StepMixturePlan",
            "version": 1,
            "total_steps": 4,
            "phases": [
                {"step": 0, "proportions": {"a": 1.0, "b": 0.0}},
                {"step": 2, "proportions": {"a": 0.0, "b": 1.0}},
            ],
        }
    )
    work = StaticMixtureWorkSource(
        [_make_dataset("a", 128), _make_dataset("b", 128)],
        plan,
        chunk_size=4,
        seed=0,
        shuffle_shards=True,
        shuffle_within_shard=True,
        exhausted_policy="stop",
    )
    work._prepare_for_execution(
        canonical_replicas=8,
        output_batch_size=4,
        preserves_cursor_order=True,
        global_batch_size=32,
    )
    for lane_id in range(8):
        lane = work.clone_for_lane(lane_id, 8)
        assert [lane.next_chunk().components.keys() for _ in range(4)] == [
            {"a"},
            {"a"},
            {"b"},
            {"b"},
        ]


def test_step_plan_zero_weight_dataset_takes_no_records() -> None:
    # c has one record; drawing it again after its weight reaches 0 would
    # stop the whole source.
    plan = StepMixturePlan(
        phases=[
            MixturePhase(0, {"a": 0.25, "b": 0.25, "c": 0.5}),
            MixturePhase(1, {"a": 0.5, "b": 0.5, "c": 0.0}),
        ],
        total_steps=4,
    )
    work = StaticMixtureWorkSource(
        [_make_dataset("a", 100), _make_dataset("b", 100), _make_dataset("c", 1)],
        plan,
        chunk_size=3,
        seed=0,
        shuffle_shards=False,
        exhausted_policy="stop",
    )
    work._prepare_for_execution(
        canonical_replicas=1,
        output_batch_size=1,
        preserves_cursor_order=True,
        global_batch_size=3,
    )
    lane = work.clone_for_lane(0, canonical_replicas=1)
    assert _counts(lane.next_chunk()) == {"a": 1, "b": 1, "c": 1}
    for _ in range(3):
        counts = _counts(lane.next_chunk())
        assert "c" not in counts and sum(counts.values()) == 3


def _two_phase_plan(boundary_step: int, **kwargs: Any) -> StepMixturePlan:
    return StepMixturePlan(
        phases=[
            MixturePhase(0, {"alpha": 0.8, "beta": 0.2}),
            MixturePhase(boundary_step, {"alpha": 0.2, "beta": 0.8}),
        ],
        total_steps=80_000,
        **kwargs,
    )


def _public(layout: dict[str, int]) -> dict[str, int]:
    public = dict(layout)
    public["global_batch_size"] = public.pop("samples_per_step")
    return public


def test_describe_reports_resolution_and_blended_boundaries() -> None:
    layout = {"samples_per_step": 32, "chunk_size": 128, "canonical_replicas": 8}
    assert _two_phase_plan(8_000).describe(**_public(layout)).splitlines() == [
        "One chunk round is chunk_size=128 x canonical_replicas=8 = 1024 records "
        "= 32 optimizer steps of 32 records; the mixture changes once per round.",
        "  step 8000: exact (starts chunk round 250)",
    ]
    (blended,) = _two_phase_plan(1_000)._blended_boundaries(**layout)
    assert (blended.first_step, blended.last_step) == (992, 1023)
    assert blended.exact_steps == (992, 1024)


def test_describe_reports_fractional_steps_per_round() -> None:
    layout = {"samples_per_step": 3, "chunk_size": 2, "canonical_replicas": 2}
    assert _two_phase_plan(7).describe(**_public(layout)).splitlines() == [
        "One chunk round is chunk_size=2 x canonical_replicas=2 = 4 records "
        "= 1.33333 optimizer steps of 3 records; the mixture changes once per round.",
        "  step 7: averaged over steps 6-7; the nearest exact steps are 4 and 8",
    ]


def test_nearest_exact_steps_stay_between_neighbouring_breakpoints() -> None:
    plan = StepMixturePlan(
        phases=[
            MixturePhase(0, {"a": 1.0, "b": 0.0}),
            MixturePhase(3, {"a": 0.0, "b": 1.0}),
            MixturePhase(5, {"a": 1.0, "b": 0.0}),
        ],
        total_steps=6,
    )
    layout = {"samples_per_step": 1, "chunk_size": 4, "canonical_replicas": 1}
    assert [b.exact_steps for b in plan._blended_boundaries(**layout)] == [(4,), (4,)]
    crowded = StepMixturePlan(
        phases=[
            MixturePhase(0, {"a": 1.0, "b": 0.0}),
            MixturePhase(1, {"a": 0.0, "b": 1.0}),
            MixturePhase(2, {"a": 1.0, "b": 0.0}),
        ],
        total_steps=3,
    )
    blended = crowded._blended_boundaries(**layout)
    assert [b.exact_steps for b in blended] == [(), ()]
    assert "no nearby step is exact" in blended[0].describe()


def test_exact_boundary_mode_rejects_blended_breakpoints() -> None:
    layout = {"samples_per_step": 32, "chunk_size": 128, "canonical_replicas": 8}
    _two_phase_plan(8_000, boundary="exact")._compile(**layout)
    with pytest.raises(ValueError, match="nearest exact steps are 992 and 1024"):
        _two_phase_plan(1_000, boundary="exact")._compile(**layout)
    with pytest.raises(ValueError, match="pass boundary='warn'"):
        _two_phase_plan(1_000)._compile(**layout)


def _warning_for(plan: StepMixturePlan, caplog: pytest.LogCaptureFixture) -> str:
    work = StaticMixtureWorkSource(
        [_make_dataset("alpha", 100), _make_dataset("beta", 100)],
        plan,
        chunk_size=4,
        seed=0,
        exhausted_policy="repeat",
    )
    with caplog.at_level("WARNING", logger="zephon.work.static_mixture"):
        work._prepare_for_execution(
            canonical_replicas=2,
            output_batch_size=2,
            preserves_cursor_order=True,
            global_batch_size=4,
        )
    return caplog.text


def test_blended_boundaries_warn(caplog: pytest.LogCaptureFixture) -> None:
    text = _warning_for(_two_phase_plan(3, boundary="warn"), caplog)
    assert "averaged over steps 2-3" in text


def test_blended_boundary_warning_is_truncated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    alternating = StepMixturePlan(
        phases=[
            MixturePhase(
                step, {"alpha": 0.2 + 0.6 * (step % 2), "beta": 0.8 - 0.6 * (step % 2)}
            )
            for step in range(24)
        ],
        total_steps=24,
        boundary="warn",
    )
    text = _warning_for(alternating, caplog)
    assert text.count("averaged over steps") == 10
    assert "... and 2 more" in text


def test_unknown_boundary_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="boundary must be one of"):
        _two_phase_plan(8, boundary="average")


def test_step_plan_boundary_round_trips() -> None:
    plan = _two_phase_plan(1_000, boundary="warn")
    payload = plan.to_dict()
    assert payload["boundary"] == "warn"
    assert StepMixturePlan.from_dict(payload).boundary == "warn"
    assert "boundary" not in _two_phase_plan(1_000).to_dict()
    assert _two_phase_plan(1_000).boundary == "exact"


def _plan_work(**kwargs: Any) -> StaticMixtureWorkSource:
    return StaticMixtureWorkSource(
        [_make_dataset("alpha", 100), _make_dataset("beta", 100)],
        _two_phase_plan(8),
        chunk_size=4,
        seed=0,
        exhausted_policy="repeat",
        **kwargs,
    )


def test_compiled_step_plan_rejects_a_different_layout() -> None:
    work = _plan_work()
    layout = {"output_batch_size": 2, "preserves_cursor_order": True}
    work._prepare_for_execution(canonical_replicas=2, global_batch_size=8, **layout)
    work._prepare_for_execution(canonical_replicas=2, global_batch_size=8, **layout)
    with pytest.raises(RuntimeError, match="already compiled"):
        work._prepare_for_execution(canonical_replicas=4, global_batch_size=8, **layout)
    with pytest.raises(RuntimeError, match="already compiled"):
        work._prepare_for_execution(
            canonical_replicas=2, global_batch_size=16, **layout
        )


def test_equal_fingerprints_compile_to_identical_tables() -> None:
    # Proportional ramps whose normalized sums round differently by order.
    def plan(order: list[tuple[str, float]]) -> StepMixturePlan:
        return StepMixturePlan(
            schedules={
                name: PhaseLinearSchedule([(0, w), (200, w), (207, 3 * w)])
                for name, w in order
            },
            total_steps=208,
            boundary="warn",
        )

    layout = {"samples_per_step": 1, "chunk_size": 5, "canonical_replicas": 1}
    forward = plan([("c", 0.6), ("b", 0.3), ("a", 0.1)])
    reverse = plan([("a", 0.1), ("b", 0.3), ("c", 0.6)])
    assert forward._fingerprint(**layout) == reverse._fingerprint(**layout)
    first, second = forward._compile(**layout), reverse._compile(**layout)
    assert first._starts.tolist() == second._starts.tolist()
    assert first._rows.tolist() == second._rows.tolist()


def test_step_plan_requires_global_batch_size() -> None:
    with pytest.raises(ValueError, match=r"Pipeline.options\(global_batch_size"):
        _plan_work()._prepare_for_execution(
            canonical_replicas=2, output_batch_size=2, preserves_cursor_order=True
        )


# Step plans: compiled round weights against a logical-sample reference.


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


def test_repeat_guard_uses_normalized_not_raw_peaks() -> None:
    plan = StepMixturePlan(
        schedules={
            "a": PhaseLinearSchedule([(0, 0.001), (4, 0.002)]),
            "b": PhaseConstantSchedule([(0, 0.001)]),
        },
        total_steps=8,
        boundary="warn",
    )
    assert plan.peak_proportions["a"] == pytest.approx(2 / 3)
    datasets = [
        Dataset.from_dict(
            name, {0: InMemoryShard([{"text": str(i)} for i in range(size)])}
        )
        for name, size in (("a", 6), ("b", 20))
    ]
    work = StaticMixtureWorkSource(
        datasets, plan, chunk_size=8, exhausted_policy="repeat"
    )
    with pytest.raises(ValueError, match="schedule peak weight"):
        work._prepare_for_execution(
            canonical_replicas=1,
            output_batch_size=1,
            preserves_cursor_order=True,
            global_batch_size=2,
        )


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


def test_chunk_size_changes_order_but_not_per_round_totals() -> None:
    # Rounds are chunk_size * canonical_replicas records, so a horizon that is
    # whole rounds for every chunk size compares like with like.
    plan = StepMixturePlan(
        phases=[
            MixturePhase(0, {"a": 0.75, "b": 0.25}),
            MixturePhase(16, {"a": 0.25, "b": 0.75}),
        ],
        total_steps=64,
        boundary="warn",
    )
    totals, orders = [], []
    for chunk_size in (4, 8, 16):
        work = StaticMixtureWorkSource(
            [
                Dataset.from_dict(name, {0: InMemoryShard(list(range(1_000)))})
                for name in ("a", "b")
            ],
            plan,
            chunk_size=chunk_size,
            seed=0,
            exhausted_policy="repeat",
        )
        work._prepare_for_execution(
            canonical_replicas=2,
            output_batch_size=2,
            preserves_cursor_order=True,
            global_batch_size=8,
        )
        lanes = [work.clone_for_lane(lane, 2) for lane in (0, 1)]
        drawn: list[str] = []
        while len(drawn) < 256:
            for lane in lanes:
                chunk = lane.next_chunk()
                drawn.extend(
                    name for name, ids in chunk.components.items() for _ in ids
                )
        totals.append({name: drawn[:256].count(name) for name in ("a", "b")})
        orders.append(drawn[:256])
    assert totals[0] == totals[1] == totals[2] == {"a": 128, "b": 128}
    assert orders[0] != orders[2]
