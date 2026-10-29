# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Curriculum schedules that change mixture weights over training.

Schedules are evaluated per chunk round (``global_chunk_index //
canonical_replicas``): the chunks of one round, one per canonical lane, share a
mixture however the lanes are spread over workers and ranks.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ConstantSchedule",
    "MixtureSchedule",
    "PhaseConstantSchedule",
    "PhaseLinearSchedule",
]


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
    chunk rounds in a :class:`MixtureSchedule`.
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
    chunk rounds in a :class:`MixtureSchedule`.
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
