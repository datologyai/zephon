# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive unit tests for Smooth Weighted Round Robin (SWRR) implementation."""

import math
from collections import Counter

from zephon._internal.utils.swrr import SmoothWeightedRoundRobin, swrr_iterate

# ----------------------------
# SmoothWeightedRoundRobin Tests
# ----------------------------


class TestSWRRInitialization:
    """Tests for SmoothWeightedRoundRobin initialization."""

    def test_basic_initialization(self):
        """Weights are normalized on init."""
        swrr = SmoothWeightedRoundRobin({"A": 3, "B": 1}, ["A", "B"])
        assert swrr._target == {"A": 0.75, "B": 0.25}
        assert swrr._order == ["A", "B"]
        assert swrr._total == 0.0

    def test_order_preserved(self):
        """Order parameter determines tie-breaking order."""
        swrr = SmoothWeightedRoundRobin({"B": 1, "A": 1}, ["A", "B"])
        assert swrr._order == ["A", "B"]

    def test_order_filters_to_targets(self):
        """Order is filtered to only include components with positive weights."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 0}, ["A", "B", "C"])
        assert swrr._order == ["A"]

    def test_zero_weight_filtered(self):
        """Components with zero weight are filtered out."""
        swrr = SmoothWeightedRoundRobin({"A": 3, "B": 0, "C": 1}, ["A", "B", "C"])
        assert "B" not in swrr._target
        assert swrr._order == ["A", "C"]

    def test_negative_weight_filtered(self):
        """Components with negative weight are filtered out."""
        swrr = SmoothWeightedRoundRobin({"A": 3, "B": -1, "C": 1}, ["A", "B", "C"])
        assert "B" not in swrr._target
        assert swrr._target == {"A": 0.75, "C": 0.25}

    def test_empty_targets(self):
        """Empty target dict results in empty state."""
        swrr = SmoothWeightedRoundRobin({}, [])
        assert swrr._target == {}
        assert swrr._order == []

    def test_all_zero_weights(self):
        """All zero weights results in empty state."""
        swrr = SmoothWeightedRoundRobin({"A": 0, "B": 0}, ["A", "B"])
        assert swrr._target == {}
        assert swrr._order == []

    def test_single_component(self):
        """Single component gets weight 1.0."""
        swrr = SmoothWeightedRoundRobin({"A": 5}, ["A"])
        assert swrr._target == {"A": 1.0}


class TestSWRRSelect:
    """Tests for the select() method."""

    def test_select_highest_deficit(self):
        """Selects component with highest deficit."""
        swrr = SmoothWeightedRoundRobin({"A": 3, "B": 1}, ["A", "B"])
        # Initial: A has target 0.75, B has 0.25
        # Deficits with effective_total=1: A=0.75, B=0.25
        chosen = swrr.select({"A", "B"})
        assert chosen == "A"

    def test_select_from_subset(self):
        """Selects only from available components."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 3}, ["A", "B"])
        # B has higher deficit but only A is available
        chosen = swrr.select({"A"})
        assert chosen == "A"

    def test_select_tiebreak_by_order(self):
        """Equal deficits tie-break by original order."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        chosen = swrr.select({"A", "B"})
        assert chosen == "A"  # A comes first in order

    def test_select_tiebreak_order_reversed(self):
        """Tie-break respects order parameter."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["B", "A"])
        chosen = swrr.select({"A", "B"})
        assert chosen == "B"  # B comes first in order

    def test_select_empty_available(self):
        """Returns None when no components available."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        assert swrr.select(set()) is None

    def test_select_unknown_component(self):
        """Unknown components in available set are ignored."""
        swrr = SmoothWeightedRoundRobin({"A": 1}, ["A"])
        chosen = swrr.select({"A", "X", "Y"})
        assert chosen == "A"

    def test_select_single_available(self):
        """Single available component is returned directly."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        assert swrr.select({"A"}) == "A"
        assert swrr.select({"B"}) == "B"

    def test_select_after_recording(self):
        """Selection changes based on recorded emissions."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        # Initially A wins (by order)
        assert swrr.select({"A", "B"}) == "A"

        # After recording A, B has higher deficit
        swrr.record("A", 1.0)
        assert swrr.select({"A", "B"}) == "B"


class TestSWRRPeek:
    """Tests for the peek() method."""

    def test_peek_returns_highest_deficit(self):
        """Peek returns component with highest deficit ignoring availability."""
        swrr = SmoothWeightedRoundRobin({"A": 3, "B": 1}, ["A", "B"])
        assert swrr.peek() == "A"

    def test_peek_empty_state(self):
        """Peek returns None for empty state."""
        swrr = SmoothWeightedRoundRobin({}, [])
        assert swrr.peek() is None

    def test_peek_tiebreak_by_order(self):
        """Peek uses order for tie-breaking."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["B", "A"])
        assert swrr.peek() == "B"

    def test_peek_changes_after_recording(self):
        """Peek result changes based on recorded emissions."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        assert swrr.peek() == "A"
        swrr.record("A", 1.0)
        assert swrr.peek() == "B"


class TestSWRRRecord:
    """Tests for the record() method."""

    def test_record_updates_emitted(self):
        """Recording updates emission tracking."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        swrr.record("A", 1.0)
        assert swrr._emitted["A"] == 1.0
        assert swrr._total == 1.0

    def test_record_accumulates(self):
        """Multiple records accumulate."""
        swrr = SmoothWeightedRoundRobin({"A": 1}, ["A"])
        swrr.record("A", 1.0)
        swrr.record("A", 2.0)
        swrr.record("A", 0.5)
        assert swrr._emitted["A"] == 3.5
        assert swrr._total == 3.5

    def test_record_multiple_components(self):
        """Records track per-component emissions."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        swrr.record("A", 3.0)
        swrr.record("B", 2.0)
        assert swrr._emitted["A"] == 3.0
        assert swrr._emitted["B"] == 2.0
        assert swrr._total == 5.0

    def test_record_default_weight(self):
        """Default weight is 1.0."""
        swrr = SmoothWeightedRoundRobin({"A": 1}, ["A"])
        swrr.record("A")
        assert swrr._emitted["A"] == 1.0


class TestSWRRRecordMulti:
    """Tests for the record_multi() method."""

    def test_record_multi_updates_all(self):
        """Record multi updates all specified components."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1, "C": 1}, ["A", "B", "C"])
        swrr.record_multi({"A": 100, "B": 200})
        assert swrr._emitted["A"] == 100
        assert swrr._emitted["B"] == 200
        assert swrr._total == 300

    def test_record_multi_empty(self):
        """Empty contributions does nothing."""
        swrr = SmoothWeightedRoundRobin({"A": 1}, ["A"])
        swrr.record_multi({})
        assert swrr._total == 0.0

    def test_record_multi_accumulates(self):
        """Multiple record_multi calls accumulate."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        swrr.record_multi({"A": 10, "B": 5})
        swrr.record_multi({"A": 5, "B": 10})
        assert swrr._emitted["A"] == 15
        assert swrr._emitted["B"] == 15
        assert swrr._total == 30


class TestSWRRGetDeficits:
    """Tests for the get_deficits() method."""

    def test_initial_deficits_equal_targets(self):
        """Initial deficits equal target weights (effective_total=1)."""
        swrr = SmoothWeightedRoundRobin({"A": 3, "B": 1}, ["A", "B"])
        deficits = swrr.get_deficits()
        assert math.isclose(deficits["A"], 0.75)
        assert math.isclose(deficits["B"], 0.25)

    def test_deficits_after_recording(self):
        """Deficits update correctly after recording."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        swrr.record("A", 1.0)
        # total=1, A emitted 1, B emitted 0
        # A deficit: 0.5*1 - 1 = -0.5
        # B deficit: 0.5*1 - 0 = 0.5
        deficits = swrr.get_deficits()
        assert math.isclose(deficits["A"], -0.5)
        assert math.isclose(deficits["B"], 0.5)

    def test_deficits_sum_to_zero_after_emission(self):
        """After emissions, deficits sum to approximately zero."""
        swrr = SmoothWeightedRoundRobin({"A": 3, "B": 1}, ["A", "B"])
        swrr.record("A", 3.0)
        swrr.record("B", 1.0)
        deficits = swrr.get_deficits()
        assert math.isclose(sum(deficits.values()), 0.0, abs_tol=1e-10)


class TestSWRRUpdateTarget:
    """Tests for the update_target() method."""

    def test_update_target_changes_weights(self):
        """Update target changes normalized weights."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        swrr.update_target({"A": 3, "B": 1})
        assert math.isclose(swrr._target["A"], 0.75)
        assert math.isclose(swrr._target["B"], 0.25)

    def test_update_target_preserves_emissions(self):
        """Update target preserves emission history."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        swrr.record("A", 5.0)
        swrr.record("B", 3.0)
        swrr.update_target({"A": 3, "B": 1})
        assert swrr._emitted["A"] == 5.0
        assert swrr._emitted["B"] == 3.0
        assert swrr._total == 8.0

    def test_update_target_adds_new_components(self):
        """Update target can add new components."""
        swrr = SmoothWeightedRoundRobin({"A": 1}, ["A"])
        swrr.update_target({"A": 1, "B": 1})
        assert "B" in swrr._target
        assert "B" in swrr._order
        assert swrr._index["B"] == 1  # Added at end

    def test_update_target_removes_zero_weight(self):
        """Update target filters out zero weights."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        swrr.update_target({"A": 1, "B": 0})
        assert "B" not in swrr._target

    def test_update_target_empty(self):
        """Update to empty targets clears weights."""
        swrr = SmoothWeightedRoundRobin({"A": 1}, ["A"])
        swrr.update_target({})
        assert swrr._target == {}


class TestSWRRProperties:
    """Tests for SWRR properties."""

    def test_total_emitted(self):
        """total_emitted property returns correct value."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        assert swrr.total_emitted == 0.0
        swrr.record("A", 5.0)
        swrr.record("B", 3.0)
        assert swrr.total_emitted == 8.0

    def test_get_actual_ratios_empty(self):
        """get_actual_ratios returns empty dict when nothing emitted."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        assert swrr.get_actual_ratios() == {}

    def test_get_actual_ratios(self):
        """get_actual_ratios returns correct proportions."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        swrr.record("A", 3.0)
        swrr.record("B", 1.0)
        ratios = swrr.get_actual_ratios()
        assert math.isclose(ratios["A"], 0.75)
        assert math.isclose(ratios["B"], 0.25)

    def test_get_actual_ratios_sum_to_one(self):
        """Actual ratios sum to 1.0."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 2, "C": 1}, ["A", "B", "C"])
        swrr.record("A", 10.0)
        swrr.record("B", 20.0)
        swrr.record("C", 5.0)
        ratios = swrr.get_actual_ratios()
        assert math.isclose(sum(ratios.values()), 1.0)


class TestSWRRDeficitAlgorithm:
    """Tests verifying the deficit-based algorithm behavior."""

    def test_deficit_algorithm_three_to_one(self):
        """Verify deficit-based ordering for 3:1 weights."""
        swrr = SmoothWeightedRoundRobin({"A": 3, "B": 1}, ["A", "B"])
        selections = []
        available = {"A", "B"}
        items = {"A": 3, "B": 1}

        while available:
            chosen = swrr.select(available)
            if chosen is None:
                break
            selections.append(chosen)
            swrr.record(chosen, 1.0)
            items[chosen] -= 1
            if items[chosen] == 0:
                available.remove(chosen)

        # Deficit-based: A, B, A, A (not score-based A, A, B, A)
        assert selections == ["A", "B", "A", "A"]

    def test_deficit_algorithm_equal_weights_alternate(self):
        """Equal weights should alternate."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        selections = []
        for _ in range(4):
            chosen = swrr.select({"A", "B"})
            selections.append(chosen)
            swrr.record(chosen, 1.0)
        assert selections == ["A", "B", "A", "B"]

    def test_deficit_algorithm_two_one_one(self):
        """Verify ordering for 2:1:1 weights."""
        swrr = SmoothWeightedRoundRobin({"A": 2, "B": 1, "C": 1}, ["A", "B", "C"])
        selections = []
        available = {"A", "B", "C"}
        items = {"A": 2, "B": 1, "C": 1}

        while available:
            chosen = swrr.select(available)
            if chosen is None:
                break
            selections.append(chosen)
            swrr.record(chosen, 1.0)
            items[chosen] -= 1
            if items[chosen] == 0:
                available.remove(chosen)

        # Expected: A, B, C, A
        assert selections == ["A", "B", "C", "A"]

    def test_deficit_tracks_underrepresentation(self):
        """Component that emits less than its share gets higher deficit."""
        swrr = SmoothWeightedRoundRobin({"A": 1, "B": 1}, ["A", "B"])
        # Record disproportionately
        swrr.record("A", 10.0)
        swrr.record("B", 2.0)

        deficits = swrr.get_deficits()
        # B should have positive deficit (underrepresented)
        # A should have negative deficit (overrepresented)
        assert deficits["B"] > 0
        assert deficits["A"] < 0


# ----------------------------
# swrr_iterate Tests
# ----------------------------


class TestSwrrIterate:
    """Tests for the swrr_iterate() helper function."""

    def test_basic_iteration(self):
        """Basic iteration yields all items with component keys."""
        components = {"A": [1, 2, 3], "B": [4, 5]}
        weights = {"A": 3, "B": 2}
        order = ["A", "B"]

        result = list(swrr_iterate(components, weights, order))

        # Should yield all items
        items = [item for item, _ in result]
        assert Counter(items) == Counter([1, 2, 3, 4, 5])

        # Should yield correct component keys
        for item, comp in result:
            assert item in components[comp]

    def test_empty_components(self):
        """Empty components yields nothing."""
        result = list(swrr_iterate({}, {}, []))
        assert result == []

    def test_single_component(self):
        """Single component yields items in order."""
        components = {"A": [1, 2, 3]}
        weights = {"A": 1}
        order = ["A"]

        result = list(swrr_iterate(components, weights, order))
        assert result == [(1, "A"), (2, "A"), (3, "A")]

    def test_empty_component_filtered(self):
        """Components with empty lists are filtered."""
        components = {"A": [1, 2], "B": [], "C": [3]}
        weights = {"A": 1, "B": 1, "C": 1}
        order = ["A", "B", "C"]

        result = list(swrr_iterate(components, weights, order))
        items = [item for item, _ in result]
        assert Counter(items) == Counter([1, 2, 3])

    def test_zero_weight_filtered(self):
        """Components with zero weight are filtered."""
        components = {"A": [1, 2], "B": [3, 4]}
        weights = {"A": 1, "B": 0}
        order = ["A", "B"]

        result = list(swrr_iterate(components, weights, order))
        assert result == [(1, "A"), (2, "A")]

    def test_deficit_based_ordering(self):
        """Verifies deficit-based SWRR ordering."""
        # 3:1 ratio with items
        components = {"A": ["a1", "a2", "a3"], "B": ["b1"]}
        weights = {"A": 3, "B": 1}
        order = ["A", "B"]

        result = list(swrr_iterate(components, weights, order))
        # Deficit-based order: A, B, A, A
        assert result == [
            ("a1", "A"),
            ("b1", "B"),
            ("a2", "A"),
            ("a3", "A"),
        ]

    def test_equal_weights_alternate(self):
        """Equal weights alternate between components."""
        components = {"A": ["a1", "a2"], "B": ["b1", "b2"]}
        weights = {"A": 1, "B": 1}
        order = ["A", "B"]

        result = list(swrr_iterate(components, weights, order))
        assert result == [
            ("a1", "A"),
            ("b1", "B"),
            ("a2", "A"),
            ("b2", "B"),
        ]

    def test_order_affects_tiebreak(self):
        """Order parameter affects tie-breaking."""
        components = {"A": ["a1"], "B": ["b1"]}
        weights = {"A": 1, "B": 1}

        result1 = list(swrr_iterate(components, weights, ["A", "B"]))
        result2 = list(swrr_iterate(components, weights, ["B", "A"]))

        assert result1 == [("a1", "A"), ("b1", "B")]
        assert result2 == [("b1", "B"), ("a1", "A")]

    def test_exhaustion_handling(self):
        """When one component exhausts, continues with remaining."""
        components = {"A": ["a1"], "B": ["b1", "b2", "b3"]}
        weights = {"A": 1, "B": 1}
        order = ["A", "B"]

        result = list(swrr_iterate(components, weights, order))
        items = [item for item, _ in result]
        # All items emitted
        assert Counter(items) == Counter(["a1", "b1", "b2", "b3"])

    def test_all_items_emitted_exactly_once(self):
        """Every item is emitted exactly once."""
        components = {
            "A": list(range(10)),
            "B": list(range(10, 18)),
            "C": list(range(18, 23)),
        }
        weights = {"A": 5, "B": 4, "C": 2}
        order = ["A", "B", "C"]

        result = list(swrr_iterate(components, weights, order))
        items = [item for item, _ in result]

        expected = list(range(23))
        assert Counter(items) == Counter(expected)

    def test_deterministic(self):
        """Same inputs produce same output."""
        components = {"A": [1, 2, 3], "B": [4, 5, 6]}
        weights = {"A": 2, "B": 1}
        order = ["A", "B"]

        result1 = list(swrr_iterate(components, weights, order))
        result2 = list(swrr_iterate(components, weights, order))
        assert result1 == result2


class TestSwrrIterateEdgeCases:
    """Edge case tests for swrr_iterate."""

    def test_missing_weight_uses_zero(self):
        """Missing weight in weights dict defaults to zero (filtered)."""
        components = {"A": [1, 2], "B": [3, 4]}
        weights = {"A": 1}  # B missing
        order = ["A", "B"]

        result = list(swrr_iterate(components, weights, order))
        assert result == [(1, "A"), (2, "A")]

    def test_extra_order_entries_ignored(self):
        """Extra entries in order are ignored."""
        components = {"A": [1, 2]}
        weights = {"A": 1}
        order = ["A", "B", "C"]  # B, C not in components

        result = list(swrr_iterate(components, weights, order))
        assert result == [(1, "A"), (2, "A")]

    def test_negative_weight_filtered(self):
        """Negative weights are filtered out."""
        components = {"A": [1, 2], "B": [3, 4]}
        weights = {"A": 1, "B": -1}
        order = ["A", "B"]

        result = list(swrr_iterate(components, weights, order))
        assert result == [(1, "A"), (2, "A")]

    def test_float_weights(self):
        """Float weights work correctly."""
        components = {"A": ["a1", "a2", "a3"], "B": ["b1"]}
        weights = {"A": 0.75, "B": 0.25}
        order = ["A", "B"]

        result = list(swrr_iterate(components, weights, order))
        items = [item for item, _ in result]
        assert Counter(items) == Counter(["a1", "a2", "a3", "b1"])

    def test_large_weight_disparity(self):
        """Large weight disparity still emits all items."""
        components = {"A": ["a1"], "B": ["b1", "b2"]}
        weights = {"A": 1000, "B": 1}
        order = ["A", "B"]

        result = list(swrr_iterate(components, weights, order))
        items = [item for item, _ in result]
        assert Counter(items) == Counter(["a1", "b1", "b2"])


class TestSwrrIterateWithIntegerKeys:
    """Tests using integer keys (like EnsureMixture uses)."""

    def test_integer_component_keys(self):
        """Works with integer component keys."""
        components = {0: ["a", "b"], 1: ["c", "d", "e"]}
        weights = {0: 2, 1: 3}
        order = [0, 1]

        result = list(swrr_iterate(components, weights, order))
        items = [item for item, _ in result]
        assert Counter(items) == Counter(["a", "b", "c", "d", "e"])

        # Verify component keys are integers
        for _, comp in result:
            assert isinstance(comp, int)

    def test_mixed_type_order(self):
        """Order must match component key types."""
        components = {0: [1, 2], 1: [3, 4]}
        weights = {0: 1, 1: 1}
        order = [0, 1]

        result = list(swrr_iterate(components, weights, order))
        assert len(result) == 4
