# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Utilities for validating and normalising mixture specifications."""

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Mapping


@dataclass(frozen=True)
class MixtureSpec:
    """Mixture weight specification with shared validation/normalisation logic."""

    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        # Eagerly validate so bad specs fail on construction, not on first use.
        # This also populates the cached_property once.
        _ = self.normalized

    def validate_for(self, components: Iterable[str]) -> None:
        """Ensure this mixture matches the provided component names exactly."""
        ordered = list(components)
        normalized = self.normalized

        missing = [name for name in ordered if name not in normalized]
        if missing:
            raise ValueError(
                "MixtureSpec missing weights for components: " + ", ".join(missing)
            )

        component_set = set(ordered)
        extra = [name for name in normalized if name not in component_set]
        if extra:
            raise ValueError(
                "MixtureSpec contains unknown components: " + ", ".join(extra)
            )

    def normalized_for(self, components: Iterable[str]) -> dict[str, float]:
        """Return normalised weights ordered according to ``components``."""
        normalized = self.normalized
        ordered = list(components)
        missing = [name for name in ordered if name not in normalized]
        if missing:
            raise ValueError(
                "MixtureSpec missing weights for components: " + ", ".join(missing)
            )
        return {name: normalized[name] for name in ordered}

    @cached_property
    def normalized(self) -> dict[str, float]:
        ordered = dict(self.weights)  # preserves insertion order
        if not ordered:
            raise ValueError("MixtureSpec must contain at least one component")

        total = 0.0
        normalized: dict[str, float] = {}
        for name, raw_value in ordered.items():
            value = float(raw_value)
            if value <= 0.0:
                raise ValueError(
                    f"MixtureSpec weight for component '{name}' must be positive"
                )
            total += value
            normalized[str(name)] = value

        if total <= 0.0:
            raise ValueError("MixtureSpec weights must sum to more than zero")

        return {name: value / total for name, value in normalized.items()}
