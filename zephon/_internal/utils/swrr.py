# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Smooth Weighted Round Robin (SWRR) implementation.

This module provides a unified SWRR implementation used by both WorkChunk
(for iterating samples in mixture order) and EnsureMixture (for reordering
samples to match target ratios).

The algorithm tracks deficit (how much each component is "owed") and always
selects the component with the highest deficit. This naturally balances
output proportions toward target weights over time.

For equal-weight items (weight=1), this is equivalent to the classic SWRR
algorithm. For variable-weight items (e.g., token counts), the same logic
applies with weights accumulated per emission.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Container, Generic, Iterator, Sequence, TypeVar

K = TypeVar("K")  # Component key type (str for WorkChunk, int for EnsureMixture)
T = TypeVar("T")  # Item type


class SmoothWeightedRoundRobin(Generic[K]):
    """Stateful SWRR selector supporting both sample and token weighting.

    Intuition:
        Each component 'k' has a normalized target weight T[k] (summing to 1.0).
        We track how much each component has emitted (E[k]) and the total emitted.
        On each selection step:

        1) Compute deficit for each available component:
           deficit[k] = T[k] * total_emitted - E[k]

           This represents how much component k is "owed" relative to its target.
           Positive deficit means underrepresented; negative means overrepresented.

        2) Pick the component with maximum deficit (ties → earlier in order wins)

        3) Emit one item from the chosen component

        4) Record the emission: E[k] += weight, total += weight
           This increases total_emitted, raising everyone's "fair share", while
           only the emitting component's E[k] increases, reducing its deficit.

    Properties:
        - Over time, emits in proportion to target weights.
        - Deterministic given insertion order and inputs.
        - O(N) work per selection (N = number of active components).
        - Supports variable-weight items (e.g., token counts) via the weight parameter.

    Edge cases:
        - When total_emitted=0 (initial state), we use effective_total=1.0 so that
          deficits equal target weights, selecting the highest-weighted component first.
        - If a component runs out of items, it's removed from the available set.
          The algorithm naturally adapts since deficit is only computed for available
          components.
        - Components with zero or negative weight are filtered out during initialization.

    Args:
        target: Mapping from component to target weight (will be normalized).
        order: Component ordering for deterministic tie-breaking.
    """

    def __init__(self, target: dict[K, float], order: list[K]) -> None:
        # Normalize target weights
        total_weight = sum(w for w in target.values() if w > 0)
        if total_weight <= 0:
            self._target: dict[K, float] = {}
            self._order: list[K] = []
            self._index: dict[K, int] = {}
        else:
            self._target = {k: w / total_weight for k, w in target.items() if w > 0}
            self._order = [k for k in order if k in self._target]
            self._index = {k: i for i, k in enumerate(self._order)}

        self._emitted: dict[K, float] = defaultdict(float)
        self._total: float = 0.0

    def peek(self, *, skip: Container[K] = ()) -> K | None:
        """Return the highest-deficit component, temporarily excluding ``skip``.

        This is used for adaptive buffering: if the ideal component is available
        in the buffer, we can emit immediately. If not, we should buffer until
        it becomes available or max_buffer_size is reached.

        Skipped components keep their targets and emission history. Selection
        renormalizes the other targets against their remaining emitted mass;
        merely ignoring skipped keys would distort unequal surviving weights.

        Returns:
            The component key with highest deficit, or None if no targets.
        """
        if not self._target or not self._order:
            return None

        # Return highest deficit component, tie-break by original order
        # Only consider components in current target (filter out obsolete)
        active = [k for k in self._order if k in self._target and k not in skip]
        if not active:
            return None
        weight = 1.0
        total = self._total
        if skip:
            excluded = [k for k in self._target if k in skip]
            if excluded:
                weight = sum(self._target[k] for k in active)
                total -= sum(self._emitted[k] for k in excluded)
        effective_total = max(1.0, total)
        return max(
            active,
            key=lambda k: (
                self._target[k] / weight * effective_total - self._emitted[k],
                -self._index[k],
            ),
        )

    def select(self, available: set[K]) -> K | None:
        """Select the next component from the available set.

        Args:
            available: Set of components that have items ready to emit.

        Returns:
            The selected component key, or None if no valid selection.
        """
        # Filter to components that are both available and have positive target weight
        active = [k for k in self._order if k in available and k in self._target]
        if not active:
            return None
        if len(active) == 1:
            return active[0]

        # Calculate deficit: target * total - emitted
        # Use max(1.0, total) to handle the initial case when total=0
        effective_total = max(1.0, self._total)
        deficits = {
            k: self._target[k] * effective_total - self._emitted[k] for k in active
        }

        # Select highest deficit, tie-break by original order (smaller index wins)
        return max(active, key=lambda k: (deficits[k], -self._index[k]))

    def record(self, component: K, weight: float = 1.0) -> None:
        """Record that an item was emitted from a component.

        Args:
            component: The component that emitted.
            weight: The weight of the emitted item (1.0 for samples, token_count for tokens).
        """
        self._emitted[component] += weight
        self._total += weight

    def record_multi(self, contributions: dict[K, float]) -> None:
        """Record emission of a multi-component item (e.g., packed sample).

        When a single sample contains content from multiple components (e.g., after
        packing), this method updates the deficit tracking for all components at once.

        Args:
            contributions: Maps component -> weight contributed by this emission.
                          For example, {0: 300, 1: 200} for a packed sample with
                          300 tokens from component 0 and 200 from component 1.
        """
        for component, weight in contributions.items():
            self._emitted[component] += weight
            self._total += weight

    def get_deficits(self) -> dict[K, float]:
        """Get current deficit per component for benefit scoring.

        Deficit = target * total_emitted - emitted. Positive deficit means the
        component is "owed" more emissions. Used by ensure_mixture to score
        multi-component samples by how well they reduce overall deficit.

        Returns:
            Dict mapping component -> deficit (positive = underserved).
        """
        effective_total = max(1.0, self._total)
        return {
            k: self._target[k] * effective_total - self._emitted[k]
            for k in self._target
        }

    def update_target(self, new_target: dict[K, float]) -> None:
        """Update target weights while preserving emission history.

        This enables smooth transitions when mixture weights change.
        The emission counts are preserved, so the deficit calculation
        naturally adapts to the new targets.

        Args:
            new_target: New target weights (will be normalized).
        """
        total_weight = sum(w for w in new_target.values() if w > 0)
        if total_weight <= 0:
            self._target = {}
            return

        self._target = {k: w / total_weight for k, w in new_target.items() if w > 0}

        # Add new components to order (preserving existing order)
        for k in new_target:
            if k not in self._index:
                self._index[k] = len(self._order)
                self._order.append(k)

    @property
    def total_emitted(self) -> float:
        """Total weight emitted across all components."""
        return self._total

    @property
    def target_ratios(self) -> dict[K, float]:
        """Target ratios per component (read-only to avoid exposing mutable state)."""
        return self._target

    def get_actual_ratios(self) -> dict[K, float]:
        """Get current actual ratios (for monitoring/warnings)."""
        if self._total <= 0:
            return {}
        return {k: v / self._total for k, v in self._emitted.items() if v > 0}


def swrr_iterate(
    components: dict[K, Sequence[T]],
    weights: dict[K, float],
    order: list[K],
) -> Iterator[tuple[T, K]]:
    """Iterate through all items in SWRR order, streaming.

    This is a convenience function for full iteration through a fixed set
    of items, used by WorkChunk. Each item has implicit weight=1.

    The iteration proceeds as follows:
        1) Select the component with highest deficit from those that still have items
        2) Emit one item from that component
        3) Update deficit tracking (record weight=1.0 for the emission)
        4) Repeat until all components are exhausted

    Edge cases:
        - Components with empty item lists or zero/negative weights are skipped.
        - When a component runs out of items, it's removed from consideration and
          the remaining components continue in proportion to their weights.
        - Single-component case is fast-pathed to avoid SWRR overhead.

    Args:
        components: Mapping from component key to list of items.
        weights: Target weights per component.
        order: Component ordering for tie-breaking.

    Yields:
        Tuples of (item, component_key) in SWRR order.
    """
    # Filter to components with items and positive weight
    active_order = [k for k in order if components.get(k) and weights.get(k, 0) > 0]
    if not active_order:
        return

    # Single component fast path
    if len(active_order) == 1:
        k = active_order[0]
        for item in components[k]:
            yield (item, k)
        return

    swrr: SmoothWeightedRoundRobin[K] = SmoothWeightedRoundRobin(weights, active_order)
    iters: dict[K, Iterator[T]] = {k: iter(components[k]) for k in active_order}

    while iters:
        available = set(iters.keys())
        chosen = swrr.select(available)

        if chosen is None:
            break

        try:
            item = next(iters[chosen])
            swrr.record(chosen, 1.0)
            yield (item, chosen)
        except StopIteration:
            # Component exhausted, remove from available
            del iters[chosen]
