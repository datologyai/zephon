from __future__ import annotations

import pytest

from zephon.work.mixture import MixtureSpec


def test_normalized_preserves_insertion_order_and_caches() -> None:
    spec = MixtureSpec({"beta": 2, "alpha": 1})

    first = spec.normalized
    second = spec.normalized

    assert first is second  # cached_property should memoize
    assert list(first.keys()) == ["beta", "alpha"]
    assert first["beta"] == pytest.approx(2 / 3)
    assert first["alpha"] == pytest.approx(1 / 3)


def test_validate_for_missing_component_raises() -> None:
    spec = MixtureSpec({"alpha": 1.0})
    with pytest.raises(
        ValueError, match="MixtureSpec missing weights for components: beta"
    ):
        spec.validate_for(["alpha", "beta"])


def test_validate_for_extra_component_raises() -> None:
    spec = MixtureSpec({"alpha": 1.0, "beta": 2.0})
    with pytest.raises(
        ValueError, match="MixtureSpec contains unknown components: beta"
    ):
        spec.validate_for(["alpha"])


def test_normalized_for_follows_component_order() -> None:
    spec = MixtureSpec({"alpha": 1, "beta": 3})
    ordered = spec.normalized_for(["beta", "alpha"])
    assert list(ordered.keys()) == ["beta", "alpha"]
    assert ordered["beta"] == pytest.approx(0.75)
    assert ordered["alpha"] == pytest.approx(0.25)


def test_normalized_for_missing_component_raises() -> None:
    spec = MixtureSpec({"alpha": 1})
    with pytest.raises(
        ValueError, match="MixtureSpec missing weights for components: beta"
    ):
        spec.normalized_for(["alpha", "beta"])


def test_non_positive_weights_raise() -> None:
    with pytest.raises(
        ValueError, match="MixtureSpec weight for component 'alpha' must be positive"
    ):
        MixtureSpec({"alpha": 0.0})
    with pytest.raises(
        ValueError, match="MixtureSpec weight for component 'beta' must be positive"
    ):
        MixtureSpec({"beta": -1})


def test_empty_weights_raise() -> None:
    with pytest.raises(
        ValueError, match="MixtureSpec must contain at least one component"
    ):
        MixtureSpec({})
