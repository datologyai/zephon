# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import math
from collections import Counter

import pytest

from zephon.work.base import (
    ComponentOrder,
    MixtureReadConfig,
    MixtureReadMode,
    WorkChunk,
)

# ----------------------------
# Helpers
# ----------------------------


def make_ids(ds, shard, n, start=0):
    """Create n SampleIds starting at local id 'start'."""
    return [(ds, shard, i) for i in range(start, start + n)]


def flatten(dict_of_lists):
    return [x for xs in dict_of_lists.values() for x in xs]


def comp_of(sample_id, comps):
    for name, ids in comps.items():
        if sample_id in ids:
            return name
    raise AssertionError("sample_id not found in components")


def comp_seq(samples, comps):
    return [comp_of(s, comps) for s in samples]


def extract_sample_ids(tuples):
    """Extract sample_ids from (sample_id, component_name) tuples."""
    return [t[0] for t in tuples]


def extract_components(tuples):
    """Extract component names from (sample_id, component_name) tuples."""
    return [t[1] for t in tuples]


# ----------------------------
# Core invariants
# ----------------------------


@pytest.mark.parametrize(
    "mode", [MixtureReadMode.WEIGHTED_ROUND_ROBIN, MixtureReadMode.WEIGHTED_RANDOM]
)
@pytest.mark.parametrize("within", [ComponentOrder.AS_IS, ComponentOrder.SHUFFLE])
def test_emits_every_sample_exactly_once_across_modes_and_orders(mode, within):
    comps = {
        "A": make_ids(0, 0, 7),
        "B": make_ids(1, 0, 5),
        "C": make_ids(2, 0, 3),
        "D": [],  # empty should be ignored
    }
    chunk = WorkChunk(components=comps, seed=123)
    cfg = MixtureReadConfig(mode=mode, seed=123, within_component=within)
    out = list(chunk.iter_samples(cfg))
    sample_ids = extract_sample_ids(out)
    # no losses/dupes
    assert Counter(sample_ids) == Counter(flatten(comps))
    # length matches
    assert len(out) == sum(len(v) for v in comps.values())
    # Each tuple should have (sample_id, component_name)
    for sample_id, comp_name in out:
        assert comp_name in comps
        assert sample_id in comps[comp_name]


def test_single_component_all_modes_emit_all_once_and_ordering_behaviour():
    comps = {"A": make_ids(0, 0, 6)}
    chunk = WorkChunk(components=comps, seed=999)

    # WRR (default) with AS_IS -> identical to input
    cfg_wrr_as_is = MixtureReadConfig(
        MixtureReadMode.WEIGHTED_ROUND_ROBIN,
        seed=None,
        within_component=ComponentOrder.AS_IS,
    )
    out = list(chunk.iter_samples(cfg_wrr_as_is))
    sample_ids = extract_sample_ids(out)
    assert sample_ids == comps["A"]
    # All should be component "A"
    assert all(comp == "A" for _, comp in out)

    # WRR with SHUFFLE -> same multiset, deterministic for same seed
    cfg_wrr_shuf = MixtureReadConfig(
        MixtureReadMode.WEIGHTED_ROUND_ROBIN,
        seed=777,
        within_component=ComponentOrder.SHUFFLE,
    )
    out1 = list(chunk.iter_samples(cfg_wrr_shuf))
    out2 = list(WorkChunk(components=comps, seed=chunk.seed).iter_samples(cfg_wrr_shuf))
    assert Counter(extract_sample_ids(out1)) == Counter(comps["A"])
    assert out1 == out2

    # With different seed, different
    cfg_wrr_shuf2 = MixtureReadConfig(
        MixtureReadMode.WEIGHTED_ROUND_ROBIN,
        seed=1234,
        within_component=ComponentOrder.SHUFFLE,
    )
    out3 = list(chunk.iter_samples(cfg_wrr_shuf2))
    assert out3 != out1


@pytest.mark.parametrize(
    "mode", [MixtureReadMode.WEIGHTED_ROUND_ROBIN, MixtureReadMode.WEIGHTED_RANDOM]
)
def test_empty_chunk_emits_nothing(mode):
    chunk = WorkChunk(components={}, seed=42)
    cfg = MixtureReadConfig(mode=mode, seed=17, within_component=ComponentOrder.AS_IS)
    assert list(chunk.iter_samples(cfg)) == []
    assert list(chunk) == []  # __iter__ path


# ----------------------------
# Small hard-coded WRR sequences
# ----------------------------


def test_wrr_two_equal_components_alternate():
    # Equal sizes -> alternate starting with first component
    a = make_ids(0, 0, 2)
    b = make_ids(1, 0, 2)
    comps = {"A": a, "B": b}
    chunk = WorkChunk(components=comps)
    cfg = MixtureReadConfig(
        MixtureReadMode.WEIGHTED_ROUND_ROBIN,
        seed=None,
        within_component=ComponentOrder.AS_IS,
    )
    out = list(chunk.iter_samples(cfg))
    sample_ids = extract_sample_ids(out)
    comp_names = extract_components(out)
    assert sample_ids == [a[0], b[0], a[1], b[1]]
    assert comp_names == ["A", "B", "A", "B"]


def test_wrr_three_vs_one_simple_pattern():
    # Counts: A=3, B=1 -> should emit 3 A's and 1 B in a balanced pattern
    # Deficit-based SWRR produces: A, B, A, A
    # (A has highest initial deficit, then B catches up, then A dominates)
    a = make_ids(0, 0, 3)
    b = make_ids(1, 0, 1)
    comps = {"A": a, "B": b}
    chunk = WorkChunk(components=comps)
    cfg = MixtureReadConfig(
        MixtureReadMode.WEIGHTED_ROUND_ROBIN,
        seed=None,
        within_component=ComponentOrder.AS_IS,
    )
    out = list(chunk.iter_samples(cfg))
    sample_ids = extract_sample_ids(out)
    comp_names = extract_components(out)
    # Deficit-based SWRR order: A, B, A, A
    assert sample_ids == [a[0], b[0], a[1], a[2]]
    assert comp_names == ["A", "B", "A", "A"]


def test_wrr_two_one_one_pattern():
    # Counts: A=2, B=1, C=1 → expected schedule: A, B, C, A
    a = make_ids(0, 0, 2)
    b = make_ids(1, 0, 1)
    c = make_ids(2, 0, 1)
    comps = {"A": a, "B": b, "C": c}
    chunk = WorkChunk(components=comps)
    cfg = MixtureReadConfig(
        MixtureReadMode.WEIGHTED_ROUND_ROBIN,
        seed=None,
        within_component=ComponentOrder.AS_IS,
    )
    out = list(chunk.iter_samples(cfg))
    sample_ids = extract_sample_ids(out)
    comp_names = extract_components(out)
    assert sample_ids == [a[0], b[0], c[0], a[1]]
    assert comp_names == ["A", "B", "C", "A"]


# ----------------------------
# Weighted-random properties
# ----------------------------


def test_weighted_random_is_seed_deterministic_and_consumes_all():
    comps = {"A": make_ids(0, 0, 5), "B": make_ids(1, 0, 5)}
    cfg = MixtureReadConfig(
        MixtureReadMode.WEIGHTED_RANDOM,
        seed=12345,
        within_component=ComponentOrder.AS_IS,
    )

    out1 = list(WorkChunk(components=comps).iter_samples(cfg))
    out2 = list(WorkChunk(components=comps).iter_samples(cfg))
    assert out1 == out2
    # Every sample exactly once
    assert Counter(extract_sample_ids(out1)) == Counter(flatten(comps))


def test_weighted_random_different_seeds_change_order_most_of_the_time():
    comps = {"A": make_ids(0, 0, 8), "B": make_ids(1, 0, 8)}
    cfg1 = MixtureReadConfig(
        MixtureReadMode.WEIGHTED_RANDOM, seed=1, within_component=ComponentOrder.AS_IS
    )
    cfg2 = MixtureReadConfig(
        MixtureReadMode.WEIGHTED_RANDOM, seed=2, within_component=ComponentOrder.AS_IS
    )
    out1 = list(WorkChunk(components=comps).iter_samples(cfg1))
    out2 = list(WorkChunk(components=comps).iter_samples(cfg2))
    assert out1 != out2
    assert Counter(extract_sample_ids(out1)) == Counter(flatten(comps))
    assert Counter(extract_sample_ids(out2)) == Counter(flatten(comps))


# ----------------------------
# Within-component shuffle properties
# ----------------------------


def test_within_component_shuffle_uses_chunk_seed_when_config_seed_none():
    comps = {"A": make_ids(0, 0, 8), "B": make_ids(1, 0, 8)}
    c1 = WorkChunk(components=comps, seed=99)
    c2 = WorkChunk(components=comps, seed=99)

    cfg = MixtureReadConfig(
        mode=MixtureReadMode.WEIGHTED_ROUND_ROBIN,
        seed=None,  # falls back to chunk.seed
        within_component=ComponentOrder.SHUFFLE,
    )
    o1 = list(c1.iter_samples(cfg))
    o2 = list(c2.iter_samples(cfg))
    assert o1 == o2


def test_within_component_shuffle_changes_with_different_config_seed():
    comps = {"A": make_ids(0, 0, 8), "B": make_ids(1, 0, 8)}
    chunk = WorkChunk(components=comps, seed=99)

    cfg1 = MixtureReadConfig(
        MixtureReadMode.WEIGHTED_ROUND_ROBIN,
        seed=123,
        within_component=ComponentOrder.SHUFFLE,
    )
    cfg2 = MixtureReadConfig(
        MixtureReadMode.WEIGHTED_ROUND_ROBIN,
        seed=456,
        within_component=ComponentOrder.SHUFFLE,
    )
    o1 = list(chunk.iter_samples(cfg1))
    o2 = list(chunk.iter_samples(cfg2))
    assert o1 != o2


# ----------------------------
# materialize_order / precompute / sample_at
# ----------------------------


def test_materialize_order_matches_iter_and_caches():
    comps = {"A": make_ids(0, 0, 5), "B": make_ids(1, 0, 3)}
    chunk = WorkChunk(components=comps, seed=321)

    cfg = MixtureReadConfig(
        MixtureReadMode.WEIGHTED_ROUND_ROBIN,
        seed=None,
        within_component=ComponentOrder.AS_IS,
    )
    m1 = chunk.materialize_order(cfg)
    m2 = chunk.materialize_order(cfg)
    assert m1 == m2  # cache hit path gives same values
    assert list(chunk.iter_samples(cfg)) == m1


def test_iter_with_precompute_true_equals_materialize_order():
    comps = {"A": make_ids(0, 0, 4), "B": make_ids(1, 0, 2)}
    chunk = WorkChunk(components=comps, seed=7)
    cfg_pre = MixtureReadConfig(
        MixtureReadMode.WEIGHTED_ROUND_ROBIN,
        seed=None,
        precompute=True,
        within_component=ComponentOrder.AS_IS,
    )
    it_list = list(chunk.iter_samples(cfg_pre))

    cfg_mat = MixtureReadConfig(
        MixtureReadMode.WEIGHTED_ROUND_ROBIN,
        seed=chunk.seed,
        precompute=False,
        within_component=ComponentOrder.AS_IS,
    )
    mat_list = chunk.materialize_order(cfg_mat)

    assert it_list == mat_list
    assert Counter(extract_sample_ids(it_list)) == Counter(flatten(comps))


def test_sample_at_matches_materialized_sequence():
    comps = {"A": make_ids(0, 0, 3), "B": make_ids(1, 0, 2)}
    chunk = WorkChunk(components=comps, seed=42)
    cfg = MixtureReadConfig(
        MixtureReadMode.WEIGHTED_ROUND_ROBIN,
        seed=None,
        within_component=ComponentOrder.AS_IS,
    )
    mat = chunk.materialize_order(cfg)
    for i in range(len(mat)):
        assert chunk.sample_at(i, cfg) == mat[i]


# ----------------------------
# Mixture property sanity (derived from counts)
# ----------------------------


def test_mixture_from_counts_skips_empty_and_normalizes():
    comps = {"A": make_ids(0, 0, 3), "B": make_ids(1, 0, 1), "C": []}
    chunk = WorkChunk(components=comps)
    mix = chunk.mixture
    assert set(mix) == {"A", "B"}
    assert math.isclose(sum(mix.values()), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(mix["A"], 3 / 4)
    assert math.isclose(mix["B"], 1 / 4)


# ----------------------------
# Robustness with empty components
# ----------------------------


def test_empty_components_are_ignored_in_iteration():
    comps = {"A": make_ids(0, 0, 3), "B": [], "C": make_ids(2, 0, 2)}
    chunk = WorkChunk(components=comps, seed=11)
    cfg = MixtureReadConfig(
        MixtureReadMode.WEIGHTED_ROUND_ROBIN,
        seed=None,
        within_component=ComponentOrder.AS_IS,
    )
    out = list(chunk.iter_samples(cfg))
    sample_ids = extract_sample_ids(out)
    assert Counter(sample_ids) == Counter(comps["A"] + comps["C"])
    # Only A and C appear in component sequence
    comp_names = extract_components(out)
    assert set(comp_names) <= {"A", "C"}


# ----------------------------
# WorkChunk.state_dict / from_state
# ----------------------------


def test_workchunk_state_dict_roundtrip_preserves_components_and_seed():
    comps = {"A": make_ids(0, 0, 3), "B": make_ids(1, 5, 2)}
    chunk = WorkChunk(components=comps, seed=42)

    state = chunk.state_dict()
    restored = WorkChunk.from_state(state)

    assert restored.seed == 42
    assert restored.components["A"] == comps["A"]
    assert restored.components["B"] == comps["B"]
    assert restored._component_order == chunk._component_order


def test_workchunk_state_dict_preserves_none_seed():
    chunk = WorkChunk(components={"A": make_ids(0, 0, 1)}, seed=None)
    state = chunk.state_dict()
    assert state["seed"] is None
    restored = WorkChunk.from_state(state)
    assert restored.seed is None


def test_workchunk_state_dict_uses_typed_schema_version():
    """state_dict() should write the schema's current version."""
    from zephon.core.checkpoint import WORK_CHUNK_VERSION

    chunk = WorkChunk(components={"A": make_ids(0, 0, 1)}, seed=1)
    state = chunk.state_dict()
    assert state["version"] == WORK_CHUNK_VERSION


def test_workchunk_from_state_rejects_bad_sample_id_length():
    bad_state = {
        "version": 1,
        "seed": 1,
        "components": [("A", [[0, 0]])],  # 2-tuple, not 3
        "component_order": ["A"],
        "total_samples": 1,
    }
    with pytest.raises(ValueError, match="Bad SampleId"):
        WorkChunk.from_state(bad_state)


def test_workchunk_from_state_honors_serialized_component_order():
    """Component order in the state dict drives iteration order on rebuild."""
    comps_in_order = {"A": make_ids(0, 0, 1), "B": make_ids(1, 0, 1)}
    chunk = WorkChunk(components=comps_in_order, seed=1)
    state = chunk.state_dict()

    # Flip the order in the serialized payload.
    state = dict(state)
    state["components"] = list(reversed(state["components"]))
    state["component_order"] = list(reversed(state["component_order"]))

    restored = WorkChunk.from_state(state)
    assert tuple(restored.components.keys()) == ("B", "A")


def test_workchunk_from_state_rejects_total_samples_mismatch():
    bad_state = {
        "version": 1,
        "components": [("A", [[0, 0, 1], [0, 0, 2]])],
        "component_order": ["A"],
        "total_samples": 99,  # disk says 99, components say 2
    }
    with pytest.raises(ValueError, match="total_samples mismatch"):
        WorkChunk.from_state(bad_state)


def test_workchunk_from_state_accepts_missing_total_samples():
    """Older checkpoints predating total_samples load without raising."""
    state = {
        "version": 1,
        "components": [("A", [[0, 0, 1]])],
        "component_order": ["A"],
    }
    restored = WorkChunk.from_state(state)
    assert len(restored) == 1


def test_workchunk_from_state_rejects_empty_payload():
    """An empty dict is corruption, not 'old format' — from_state must raise."""
    with pytest.raises(ValueError, match="components: missing"):
        WorkChunk.from_state({})


def test_workchunk_from_state_rejects_missing_components_key():
    """``components`` and ``component_order`` are always-written; absence is corruption."""
    state = {"version": 1, "seed": 7}
    with pytest.raises(ValueError) as exc_info:
        WorkChunk.from_state(state)
    msg = str(exc_info.value)
    assert "components: missing" in msg
    assert "component_order: missing" in msg
