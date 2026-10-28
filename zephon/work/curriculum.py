# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Curriculum schedules that change mixture weights over training.

Schedules are evaluated per chunk round (``global_chunk_index //
canonical_replicas``): the chunks of one round, one per canonical lane, share a
mixture however the lanes are spread over workers and ranks. A
:class:`StepMixturePlan` is written in optimizer steps and compiled to chunk
rounds when the engine starts.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal, cast

import numpy as np

__all__ = [
    "ConstantSchedule",
    "MixturePhase",
    "MixtureSchedule",
    "PhaseConstantSchedule",
    "PhaseLinearSchedule",
    "StepMixturePlan",
]


_MIXTURE_WEIGHT_SUM_TOLERANCE = 1e-8


def _validate_finite_nonnegative(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _payload(
    state: Any, type_name: str, *, required: Iterable[str]
) -> Mapping[str, Any]:
    if not isinstance(state, Mapping):
        raise TypeError(f"{type_name} payload must be a mapping, got {state!r}")
    if state.get("type") != type_name:
        raise ValueError(
            f"expected a {type_name} payload, got type={state.get('type')!r}"
        )
    required = set(required)
    unknown = sorted(set(state) - required - {"type"})
    if unknown:
        raise ValueError(f"unknown {type_name} fields: {unknown}")
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"{type_name} payload is missing {missing}")
    return state


def _parse_breakpoints(
    breakpoints: Sequence[tuple[int, float]], *, owner: str
) -> tuple[tuple[int, float], ...]:
    if not breakpoints:
        raise ValueError(f"{owner} requires a breakpoint")
    parsed: list[tuple[int, float]] = []
    for position, raw_weight in breakpoints:
        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError(
                f"{owner} breakpoint positions must be integers, got {position!r}"
            )
        if position < 0:
            raise ValueError(
                f"{owner} breakpoint positions must be non-negative, got {position}"
            )
        if parsed and position <= parsed[-1][0]:
            raise ValueError(
                f"{owner} breakpoint positions must be strictly increasing"
            )
        weight = _validate_finite_nonnegative(raw_weight, name=f"{owner} weight")
        parsed.append((position, weight))
    return tuple(parsed)


@dataclass(frozen=True)
class ConstantSchedule:
    """The same raw weight at every position."""

    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "weight", _validate_finite_nonnegative(self.weight, name="weight")
        )

    def value(self, position: int, /) -> float:
        """Return the raw weight at ``position``."""
        del position
        return self.weight

    def max_weight(self) -> float:
        """Return the largest raw weight."""
        return self.weight

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible payload."""
        return {"type": "ConstantSchedule", "weight": self.weight}

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> ConstantSchedule:
        """Rebuild a schedule from :meth:`to_dict` output."""
        return cls(_payload(state, "ConstantSchedule", required=("weight",))["weight"])


@dataclass(frozen=True)
class PhaseConstantSchedule:
    """Hold each ``(position, weight)`` breakpoint's weight until the next one.

    The first weight also applies before the first breakpoint. Positions are
    chunk rounds in a :class:`MixtureSchedule` and optimizer steps in a
    :class:`StepMixturePlan`.
    """

    breakpoints: Sequence[tuple[int, float]]
    _positions: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        parsed = _parse_breakpoints(self.breakpoints, owner="PhaseConstantSchedule")
        object.__setattr__(self, "breakpoints", parsed)
        object.__setattr__(self, "_positions", tuple(p for p, _ in parsed))

    def value(self, position: int, /) -> float:
        """Return the raw weight at ``position``."""
        index = max(bisect_right(self._positions, position) - 1, 0)
        return self.breakpoints[index][1]

    def max_weight(self) -> float:
        """Return the largest raw weight."""
        return max(weight for _, weight in self.breakpoints)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible payload."""
        return {
            "type": "PhaseConstantSchedule",
            "breakpoints": [list(point) for point in self.breakpoints],
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> PhaseConstantSchedule:
        """Rebuild a schedule from :meth:`to_dict` output."""
        payload = _payload(state, "PhaseConstantSchedule", required=("breakpoints",))
        return cls([tuple(point) for point in payload["breakpoints"]])


@dataclass(frozen=True)
class PhaseLinearSchedule:
    """Interpolate linearly between ``(position, weight)`` breakpoints.

    The first and last weights hold outside the breakpoints. Positions are
    chunk rounds in a :class:`MixtureSchedule` and optimizer steps in a
    :class:`StepMixturePlan`.
    """

    breakpoints: Sequence[tuple[int, float]]
    _positions: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        parsed = _parse_breakpoints(self.breakpoints, owner="PhaseLinearSchedule")
        object.__setattr__(self, "breakpoints", parsed)
        object.__setattr__(self, "_positions", tuple(p for p, _ in parsed))

    def value(self, position: int, /) -> float:
        """Return the raw weight at ``position``."""
        index = bisect_right(self._positions, position) - 1
        if index < 0:
            return self.breakpoints[0][1]
        if index >= len(self.breakpoints) - 1:
            return self.breakpoints[-1][1]
        left, left_weight = self.breakpoints[index]
        right, right_weight = self.breakpoints[index + 1]
        progress = (position - left) / (right - left)
        return (1.0 - progress) * left_weight + progress * right_weight

    def max_weight(self) -> float:
        """Return the largest raw weight."""
        return max(weight for _, weight in self.breakpoints)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible payload."""
        return {
            "type": "PhaseLinearSchedule",
            "breakpoints": [list(point) for point in self.breakpoints],
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> PhaseLinearSchedule:
        """Rebuild a schedule from :meth:`to_dict` output."""
        payload = _payload(state, "PhaseLinearSchedule", required=("breakpoints",))
        return cls([tuple(point) for point in payload["breakpoints"]])


_Schedule = ConstantSchedule | PhaseConstantSchedule | PhaseLinearSchedule
_SCHEDULE_TYPES = (ConstantSchedule, PhaseConstantSchedule, PhaseLinearSchedule)

_SCHEDULE_FACTORIES: dict[str, Callable[[Mapping[str, Any]], _Schedule]] = {
    "ConstantSchedule": ConstantSchedule.from_dict,
    "PhaseConstantSchedule": PhaseConstantSchedule.from_dict,
    "PhaseLinearSchedule": PhaseLinearSchedule.from_dict,
}


def _schedule_from_dict(state: Any) -> _Schedule:
    if not isinstance(state, Mapping):
        raise TypeError(f"schedule payload must be a mapping, got {state!r}")
    factory = _SCHEDULE_FACTORIES.get(state.get("type"))
    if factory is None:
        raise ValueError(
            f"Unknown schedule type {state.get('type')!r}. Known types: "
            f"{sorted(_SCHEDULE_FACTORIES)}"
        )
    return factory(state)


def _breakpoint_positions(schedule: _Schedule) -> tuple[int, ...]:
    """Positions where a schedule's raw weight stops being affine."""
    if isinstance(schedule, ConstantSchedule):
        return (0,)
    return schedule._positions


class MixtureSchedule:
    """Per-dataset weight schedules normalized together into a mixture.

    Each component is a :class:`ConstantSchedule`, :class:`PhaseConstantSchedule`
    or :class:`PhaseLinearSchedule`; plain numbers are wrapped in
    :class:`ConstantSchedule`. Positions are chunk rounds.
    """

    def __init__(self, schedules: Mapping[str, _Schedule | float]) -> None:
        if not schedules:
            raise ValueError("MixtureSchedule requires at least one component")
        parsed: dict[str, _Schedule] = {}
        for name, raw in schedules.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"MixtureSchedule component names must be strings, got {name!r}"
                )
            if not name:
                raise ValueError("MixtureSchedule component names must be non-empty")
            if isinstance(raw, _SCHEDULE_TYPES):
                parsed[name] = raw
            elif isinstance(raw, numbers.Real) and not isinstance(raw, bool):
                parsed[name] = ConstantSchedule(raw)
            else:
                raise TypeError(
                    f"MixtureSchedule component {name!r} must be a number, "
                    "ConstantSchedule, PhaseConstantSchedule or PhaseLinearSchedule, "
                    f"got {type(raw).__name__}"
                )
        self._schedules = parsed
        self._peaks: dict[str, float] | None = None

    @property
    def component_names(self) -> tuple[str, ...]:
        """Component names in insertion order."""
        return tuple(self._schedules)

    def weights_at(self, chunk_round: int) -> dict[str, float]:
        """Return the normalized mixture at ``chunk_round``."""
        return dict(zip(self._schedules, self._normalized_at(chunk_round)))

    def _normalized_at(self, position: int) -> tuple[float, ...]:
        raw = tuple(schedule.value(position) for schedule in self._schedules.values())
        # fsum: plain float sum() rounds differently from Python 3.12 on.
        total = math.fsum(raw)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(
                f"MixtureSchedule weights at position {position} must have a "
                f"positive finite total, got {total}"
            )
        return tuple(weight / total for weight in raw)

    def max_weight(self, name: str) -> float:
        """Return the largest normalized weight ``name`` reaches."""
        if self._peaks is None:
            self._peaks = self._compute_peaks()
        return self._peaks[name]

    def _final_weights(self) -> dict[str, float]:
        """The mixture held after the last breakpoint."""
        last = max(
            max(_breakpoint_positions(schedule))
            for schedule in self._schedules.values()
        )
        return self.weights_at(last)

    def _fingerprint(self) -> str:
        """A stable digest of the configuration, for checkpoint compatibility."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _compute_peaks(self) -> dict[str, float]:
        positions: set[int] = {0}
        for schedule in self._schedules.values():
            breakpoints = _breakpoint_positions(schedule)
            positions.update(breakpoints)
            positions.update(p - 1 for p in breakpoints if p > 0)
        # Between breakpoints every raw weight is affine, so each normalized
        # weight is monotone and peaks at an interval end; ``p - 1`` is the
        # last position before a jump.
        peaks = dict.fromkeys(self._schedules, 0.0)
        for position in sorted(positions):
            for name, weight in zip(self._schedules, self._normalized_at(position)):
                peaks[name] = max(peaks[name], weight)
        return peaks

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible payload."""
        return {
            "type": "MixtureSchedule",
            "version": 1,
            "schedules": {
                name: schedule.to_dict() for name, schedule in self._schedules.items()
            },
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> MixtureSchedule:
        """Rebuild a schedule from :meth:`to_dict` output."""
        payload = _payload(state, "MixtureSchedule", required=("version", "schedules"))
        if payload["version"] != 1:
            raise ValueError(
                f"Unsupported MixtureSchedule schema version {payload['version']!r}"
            )
        schedules = payload["schedules"]
        if not isinstance(schedules, Mapping):
            raise TypeError(
                f"MixtureSchedule schedules must be a mapping, got {schedules!r}"
            )
        return cls({name: _schedule_from_dict(sub) for name, sub in schedules.items()})


@dataclass(frozen=True)
class MixturePhase:
    """A complete mixture and the optimizer step where it becomes active."""

    step: int
    proportions: Mapping[str, float]

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int):
            raise TypeError(f"phase step must be an integer, got {self.step!r}")
        if self.step < 0:
            raise ValueError(f"phase step must be non-negative, got {self.step}")
        if not isinstance(self.proportions, Mapping):
            raise TypeError(
                f"phase proportions must be a mapping, got {self.proportions!r}"
            )
        if not self.proportions:
            raise ValueError("phase proportions must be non-empty")
        parsed: dict[str, float] = {}
        for name, raw_weight in self.proportions.items():
            if not isinstance(name, str):
                raise TypeError(f"phase component names must be strings, got {name!r}")
            if not name:
                raise ValueError("phase component names must be non-empty")
            parsed[name] = _validate_finite_nonnegative(
                raw_weight, name=f"phase weight for component '{name}'"
            )
        object.__setattr__(self, "proportions", parsed)

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "proportions": dict(self.proportions)}

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> MixturePhase:
        if not isinstance(state, Mapping):
            raise TypeError(f"MixturePhase payload must be a mapping, got {state!r}")
        unknown = sorted(set(state) - {"step", "proportions"})
        if unknown:
            raise ValueError(
                f"unknown MixturePhase fields {unknown}; expected 'step' and "
                "'proportions'"
            )
        if "step" not in state:
            raise ValueError("MixturePhase is missing 'step'")
        if "proportions" not in state:
            raise ValueError("MixturePhase is missing 'proportions'")
        return cls(step=state["step"], proportions=state["proportions"])


_StepSchedule = PhaseConstantSchedule | PhaseLinearSchedule
_STEP_SCHEDULE_TYPES = (PhaseConstantSchedule, PhaseLinearSchedule)
_BoundaryMode = Literal["warn", "exact"]
_BOUNDARY_MODES = ("warn", "exact")
_LISTED_BOUNDARIES = 10


def _list_blended(blended: Sequence[_BlendedBoundary]) -> str:
    shown = [b.describe() for b in blended[:_LISTED_BOUNDARIES]]
    if len(blended) > _LISTED_BOUNDARIES:
        shown.append(f"... and {len(blended) - _LISTED_BOUNDARIES} more")
    return "".join(f"\n  {line}" for line in shown)


@dataclass(frozen=True, init=False)
class StepMixturePlan:
    """A predetermined curriculum in optimizer steps.

    Supply exactly one of ``phases`` (complete mixtures, each held until the
    next) or ``schedules`` (one :class:`PhaseConstantSchedule` or
    :class:`PhaseLinearSchedule` per dataset, positioned in optimizer steps and
    normalized together at every step). Pass it as
    ``StaticMixtureWorkSource(mixture=plan)`` and give the pipeline the
    trainer's ``global_batch_size``. ``total_steps`` bounds the plan; it does
    not stop iteration.

    The mixture changes once per chunk round. With ``boundary="exact"`` (the
    default) a breakpoint that falls inside a round stops the pipeline from
    starting, and the error names the nearest exact steps. ``"warn"`` averages
    such a breakpoint over its round and logs a warning instead.
    :meth:`describe` shows where every breakpoint lands.
    """

    phases: tuple[MixturePhase, ...]
    total_steps: int
    schedules: Mapping[str, _StepSchedule] | None
    boundary: _BoundaryMode
    _step_schedule: MixtureSchedule = field(repr=False, compare=False)

    def __init__(
        self,
        phases: Sequence[MixturePhase | Mapping[str, Any]] | None = None,
        *,
        total_steps: int,
        schedules: Mapping[str, _StepSchedule] | None = None,
        boundary: _BoundaryMode = "exact",
    ) -> None:
        if boundary not in _BOUNDARY_MODES:
            raise ValueError(
                f"boundary must be one of {_BOUNDARY_MODES}, got {boundary!r}"
            )
        _validate_positive_int(total_steps, name="total_steps")
        if (phases is None) == (schedules is None):
            raise ValueError("supply exactly one of phases and schedules")
        if schedules is not None:
            parsed_schedules = _validate_step_schedules(schedules, total_steps)
            parsed_phases: tuple[MixturePhase, ...] = ()
        else:
            assert phases is not None
            parsed_phases = _validate_phases(phases, total_steps)
            parsed_schedules = None
        step_schedule = MixtureSchedule(
            parsed_schedules
            if parsed_schedules is not None
            else {
                name: PhaseConstantSchedule(
                    [(phase.step, phase.proportions[name]) for phase in parsed_phases]
                )
                for name in parsed_phases[0].proportions
            }
        )
        never_active = sorted(
            name
            for name in step_schedule.component_names
            if step_schedule.max_weight(name) <= 0.0
        )
        if never_active:
            raise ValueError(
                "StepMixturePlan components must be active at some step; "
                f"never active: {never_active}"
            )
        object.__setattr__(self, "phases", parsed_phases)
        object.__setattr__(self, "total_steps", total_steps)
        object.__setattr__(self, "schedules", parsed_schedules)
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "_step_schedule", step_schedule)

    @property
    def component_names(self) -> tuple[str, ...]:
        """Dataset names in the plan."""
        return self._step_schedule.component_names

    @property
    def peak_proportions(self) -> dict[str, float]:
        """Largest normalized proportion each component reaches at any step."""
        return {
            name: self._step_schedule.max_weight(name) for name in self.component_names
        }

    def validate_for(self, components: Sequence[str]) -> None:
        """Raise unless the plan names exactly ``components``."""
        expected = set(components)
        configured = set(self.component_names)
        if configured != expected:
            raise ValueError(
                "StepMixturePlan components must match work source dataset names: "
                f"missing={sorted(expected - configured)}, "
                f"extra={sorted(configured - expected)}"
            )

    def describe(
        self, *, global_batch_size: int, chunk_size: int, canonical_replicas: int
    ) -> str:
        """Explain how the plan maps onto chunk rounds for this configuration."""
        _validate_positive_int(global_batch_size, name="global_batch_size")
        _validate_positive_int(chunk_size, name="chunk_size")
        _validate_positive_int(canonical_replicas, name="canonical_replicas")
        samples_per_step = global_batch_size
        samples_per_round = chunk_size * canonical_replicas
        steps_per_round = Fraction(samples_per_round, samples_per_step)
        blended = {
            b.step: b
            for b in self._blended_boundaries(
                samples_per_step=samples_per_step,
                chunk_size=chunk_size,
                canonical_replicas=canonical_replicas,
            )
        }
        lines = [
            f"One chunk round is chunk_size={chunk_size} x canonical_replicas="
            f"{canonical_replicas} = {samples_per_round} records = "
            f"{float(steps_per_round):g} optimizer steps of {samples_per_step} "
            "records; the mixture changes once per round."
        ]
        for step in self._breakpoint_steps():
            if step in blended:
                lines.append(f"  {blended[step].describe()}")
            else:
                chunk_round = step * samples_per_step // samples_per_round
                lines.append(f"  step {step}: exact (starts chunk round {chunk_round})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible payload."""
        if self.schedules is not None:
            payload: dict[str, Any] = {
                "type": "StepMixturePlan",
                "version": 2,
                "total_steps": self.total_steps,
                "schedules": {
                    name: schedule.to_dict()
                    for name, schedule in self.schedules.items()
                },
            }
        else:
            payload = {
                "type": "StepMixturePlan",
                "version": 1,
                "total_steps": self.total_steps,
                "phases": [phase.to_dict() for phase in self.phases],
            }
        if self.boundary != "exact":
            payload["boundary"] = self.boundary
        return payload

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> StepMixturePlan:
        """Rebuild a plan from :meth:`to_dict` output."""
        if not isinstance(state, Mapping):
            raise TypeError(f"StepMixturePlan payload must be a mapping, got {state!r}")
        if state.get("type", "StepMixturePlan") != "StepMixturePlan":
            raise ValueError(
                f"Expected StepMixturePlan payload, got type={state.get('type')!r}"
            )
        version = state.get("version", 1)
        if version not in (1, 2) or isinstance(version, bool):
            raise ValueError(f"Unsupported StepMixturePlan schema version {version!r}")
        input_key = "phases" if version == 1 else "schedules"
        known = {"type", "version", "total_steps", "boundary", input_key}
        unknown = sorted(set(state) - known)
        if unknown:
            raise ValueError(f"unknown StepMixturePlan fields: {unknown}")
        if "total_steps" not in state:
            raise ValueError("StepMixturePlan is missing 'total_steps'")
        if version == 2:
            serialized = state.get("schedules")
            if not isinstance(serialized, Mapping):
                raise TypeError(
                    f"StepMixturePlan schedules must be a mapping, got {serialized!r}"
                )
            schedules: dict[str, _StepSchedule] = {}
            for name, payload in serialized.items():
                schedule = _schedule_from_dict(payload)
                if not isinstance(schedule, _STEP_SCHEDULE_TYPES):
                    raise ValueError(
                        "step plans require PhaseConstantSchedule or PhaseLinearSchedule"
                    )
                schedules[name] = schedule
            return cls(
                schedules=schedules,
                total_steps=state["total_steps"],
                boundary=state.get("boundary", "exact"),
            )
        phases = state.get("phases")
        if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
            raise TypeError(
                f"StepMixturePlan phases must be a sequence, got {phases!r}"
            )
        return cls(
            phases=[MixturePhase.from_dict(phase) for phase in phases],
            total_steps=state["total_steps"],
            boundary=state.get("boundary", "exact"),
        )

    def _breakpoint_steps(self) -> tuple[int, ...]:
        if self.schedules is None:
            return tuple(phase.step for phase in self.phases[1:])
        return tuple(
            sorted(
                {
                    position
                    for schedule in self.schedules.values()
                    for position, _ in schedule.breakpoints
                    if position > 0
                }
            )
        )

    def _blended_boundaries(
        self, *, samples_per_step: int, chunk_size: int, canonical_replicas: int
    ) -> tuple[_BlendedBoundary, ...]:
        samples_per_round = chunk_size * canonical_replicas
        quantum = samples_per_round // math.gcd(samples_per_round, samples_per_step)
        steps = self._breakpoint_steps()
        blended = []
        for index, step in enumerate(steps):
            chunk_round, offset = divmod(step * samples_per_step, samples_per_round)
            if not offset:
                continue
            first = Fraction(chunk_round * samples_per_round, samples_per_step)
            last = Fraction((chunk_round + 1) * samples_per_round, samples_per_step)
            # Suggest only steps that keep the breakpoints in order and inside
            # the plan.
            lower = steps[index - 1] if index else 0
            upper = steps[index + 1] if index + 1 < len(steps) else self.total_steps
            before = step // quantum * quantum
            after = before + quantum
            blended.append(
                _BlendedBoundary(
                    step=step,
                    first_step=math.floor(first),
                    last_step=math.ceil(last) - 1,
                    exact_steps=tuple(s for s in (before, after) if lower < s < upper),
                )
            )
        return tuple(blended)

    def _compile(
        self, *, samples_per_step: int, chunk_size: int, canonical_replicas: int
    ) -> _RoundTable:
        """Average the per-step mixtures over each chunk round.

        Step ``k`` covers source samples ``[k * samples_per_step, (k + 1) *
        samples_per_step)``; each round's target is the sample-weighted mean of
        the steps it overlaps.
        """
        if self.boundary == "exact":
            blended = self._blended_boundaries(
                samples_per_step=samples_per_step,
                chunk_size=chunk_size,
                canonical_replicas=canonical_replicas,
            )
            if blended:
                raise ValueError(
                    "StepMixturePlan: these breakpoints fall inside a chunk round. "
                    "Move them to an exact step, or pass boundary='warn' to "
                    "average each over its round:" + _list_blended(blended)
                )
        return _compile_round_averages(
            self._step_schedule,
            samples_per_step=samples_per_step,
            samples_per_round=chunk_size * canonical_replicas,
        )

    def _fingerprint(
        self, *, samples_per_step: int, chunk_size: int, canonical_replicas: int
    ) -> str:
        """Digest of the plan and the layout it compiles against."""
        canonical = json.dumps(
            {
                "plan": self.to_dict(),
                "samples_per_step": samples_per_step,
                "chunk_size": chunk_size,
                "canonical_replicas": canonical_replicas,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class _BlendedBoundary:
    step: int
    first_step: int
    last_step: int
    exact_steps: tuple[int, ...]

    def describe(self) -> str:
        if not self.exact_steps:
            hint = "no nearby step is exact; lower chunk_size to make one"
        elif len(self.exact_steps) == 1:
            hint = f"the nearest exact step is {self.exact_steps[0]}"
        else:
            hint = f"the nearest exact steps are {self.exact_steps[0]} and {self.exact_steps[1]}"
        return (
            f"step {self.step}: averaged over steps {self.first_step}-"
            f"{self.last_step}; {hint}"
        )


class _RoundTable:
    """A compiled step plan: the mixture of every chunk round.

    Only rounds whose mixture differs from the previous one are stored; the
    last row holds after them.
    """

    def __init__(
        self, names: tuple[str, ...], starts: np.ndarray, rows: np.ndarray
    ) -> None:
        self._names = names
        self._starts = starts
        self._rows = rows

    @property
    def component_names(self) -> tuple[str, ...]:
        return self._names

    def weights_at(self, chunk_round: int) -> dict[str, float]:
        row = self._rows[np.searchsorted(self._starts, chunk_round, side="right") - 1]
        return dict(zip(self._names, row.tolist()))

    def max_weight(self, name: str) -> float:
        return float(self._rows[:, self._names.index(name)].max())

    def _final_weights(self) -> dict[str, float]:
        return dict(zip(self._names, self._rows[-1].tolist()))


def _validate_phases(
    phases: Sequence[MixturePhase | Mapping[str, Any]], total_steps: int
) -> tuple[MixturePhase, ...]:
    if not phases:
        raise ValueError("StepMixturePlan requires at least one phase")
    parsed = tuple(
        phase if isinstance(phase, MixturePhase) else MixturePhase.from_dict(phase)
        for phase in phases
    )
    if parsed[0].step != 0:
        raise ValueError("the first mixture phase must start at step 0")
    expected = set(parsed[0].proportions)
    for previous, phase in zip((None, *parsed), parsed):
        if previous is not None and phase.step <= previous.step:
            raise ValueError("mixture phase steps must be strictly increasing")
        if phase.step >= total_steps:
            raise ValueError(
                f"mixture phase step {phase.step} must be smaller than "
                f"total_steps={total_steps}"
            )
        configured = set(phase.proportions)
        if configured != expected:
            raise ValueError(
                f"mixture phase at step {phase.step} has component mismatch: "
                f"missing={sorted(expected - configured)}, "
                f"extra={sorted(configured - expected)}"
            )
        total = sum(phase.proportions.values())
        if not math.isclose(
            total, 1.0, rel_tol=0.0, abs_tol=_MIXTURE_WEIGHT_SUM_TOLERANCE
        ):
            raise ValueError(
                f"mixture phase at step {phase.step} has proportions summing "
                f"to {total}, not 1.0"
            )
    return parsed


def _validate_step_schedules(
    schedules: Mapping[str, _StepSchedule], total_steps: int
) -> dict[str, _StepSchedule]:
    if not isinstance(schedules, Mapping):
        raise TypeError(f"step schedules must be a mapping, got {schedules!r}")
    if not schedules:
        raise ValueError("step schedules must be non-empty")
    for name, schedule in schedules.items():
        if not isinstance(schedule, _STEP_SCHEDULE_TYPES):
            raise TypeError(
                f"step schedule for {name!r} must be a PhaseConstantSchedule or "
                f"PhaseLinearSchedule, got {type(schedule).__name__}"
            )
        if schedule.breakpoints[0][0] != 0:
            raise ValueError(f"step schedule for {name!r} must start at step 0")
        last = schedule.breakpoints[-1][0]
        # A linear ramp may end exactly at total_steps; a phase cannot start there.
        if last > total_steps or (
            isinstance(schedule, PhaseConstantSchedule) and last == total_steps
        ):
            raise ValueError(
                f"step schedule for {name!r} has a breakpoint at step {last}, "
                f"outside total_steps={total_steps}"
            )
    return dict(schedules)


_STEP_BLOCK = 1 << 15


def _step_values(schedule: _StepSchedule, steps: np.ndarray) -> np.ndarray:
    positions = np.asarray(schedule._positions, dtype=np.float64)
    weights = np.asarray([w for _, w in schedule.breakpoints], dtype=np.float64)
    if isinstance(schedule, PhaseLinearSchedule):
        return np.interp(steps, positions, weights)
    index = np.searchsorted(positions, steps, side="right") - 1
    return weights[np.maximum(index, 0)]


def _compile_round_averages(
    step_schedule: MixtureSchedule, *, samples_per_step: int, samples_per_round: int
) -> _RoundTable:
    # Sorted, so the floating-point result depends only on what the
    # fingerprint hashes, not on mapping order.
    names = tuple(sorted(step_schedule.component_names))
    schedules = cast(
        list[_StepSchedule], [step_schedule._schedules[name] for name in names]
    )

    def normalized_at(step: int) -> np.ndarray:
        weights = step_schedule.weights_at(step)
        return np.asarray([weights[name] for name in names])

    width = len(names)
    start_blocks: list[np.ndarray] = []
    row_blocks: list[np.ndarray] = []
    round_index = 0
    used = 0
    # Weighted samples of the round in progress, per component.
    partial = np.zeros(width)

    def emit(block: np.ndarray) -> None:
        nonlocal round_index
        changed = np.ones(len(block), dtype=bool)
        changed[1:] = np.any(block[1:] != block[:-1], axis=1)
        if row_blocks and np.array_equal(row_blocks[-1][-1], block[0]):
            changed[0] = False
        if changed.any():
            start_blocks.append(round_index + np.flatnonzero(changed))
            row_blocks.append(block[changed])
        round_index += len(block)

    def consume_constant(count: int, weights: np.ndarray) -> None:
        nonlocal round_index, used, partial
        if used:
            taken = min(count, samples_per_round - used)
            partial = partial + taken * weights
            used += taken
            count -= taken
            if used == samples_per_round:
                emit((partial / samples_per_round)[None, :])
                used = 0
                partial = np.zeros(width)
        full_rounds, tail = divmod(count, samples_per_round)
        if full_rounds:
            emit(weights[None, :])
            round_index += full_rounds - 1
        if tail:
            used = tail
            partial = tail * weights

    def consume_steps(weights: np.ndarray) -> None:
        # Each row is one step of samples_per_step records. Cumulative weighted
        # samples, read at the round boundaries, give each round's total.
        nonlocal used, partial
        total = len(weights) * samples_per_step
        cumulative = np.vstack(
            [np.zeros(width), np.cumsum(weights * samples_per_step, axis=0)]
        )
        first = samples_per_round - used
        if first > total:
            partial = partial + cumulative[-1]
            used += total
            return
        offsets = np.arange(first, total + 1, samples_per_round)
        step = np.minimum(offsets // samples_per_step, len(weights) - 1)
        filled = (
            cumulative[step]
            + (offsets - step * samples_per_step)[:, None] * (weights[step])
        )
        sums = np.vstack([partial + filled[0], np.diff(filled, axis=0)])
        emit(sums / samples_per_round)
        used = int(total - offsets[-1])
        partial = cumulative[-1] - filled[-1] if used else np.zeros(width)

    anchors = sorted(
        {
            position
            for schedule in schedules
            for position in _breakpoint_positions(schedule)
        }
    )
    for left, right in zip(anchors, anchors[1:]):
        varies = any(
            isinstance(schedule, PhaseLinearSchedule)
            and schedule.value(left) != schedule.value(right - 1)
            for schedule in schedules
        )
        if not varies:
            # Constant ranges are consumed in bulk and emitted exactly.
            consume_constant(
                (right - left) * samples_per_step,
                normalized_at(left),
            )
            continue
        # Normalized ratios of independent ramps are not linear, so every step
        # is normalized on its own.
        for begin in range(left, right, _STEP_BLOCK):
            steps = np.arange(begin, min(right, begin + _STEP_BLOCK), dtype=np.float64)
            raw = np.stack([_step_values(s, steps) for s in schedules], axis=1)
            totals = raw.sum(axis=1)
            if not np.all(np.isfinite(totals)) or np.any(totals <= 0.0):
                raise ValueError(
                    f"StepMixturePlan weights between steps {begin} and "
                    f"{begin + len(steps)} must have a positive finite total"
                )
            consume_steps(raw / totals[:, None])

    final = normalized_at(anchors[-1])
    if used:
        consume_constant(samples_per_round - used, final)
    emit(final[None, :])
    return _RoundTable(names, np.concatenate(start_blocks), np.vstack(row_blocks))
