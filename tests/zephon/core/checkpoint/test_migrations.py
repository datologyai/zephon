# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the checkpoint migration registry."""

import pytest

from zephon.core.checkpoint._migrations import (
    _MIGRATIONS,
    CURRENT_VERSIONS,
    migrate,
    register_migration,
)


def test_migrate_noop_at_current_version():
    state = {"version": 1, "world": {}}
    result = migrate("engine", state)
    assert result == state
    assert result is not state  # shallow copy protects caller's top-level keys


def test_migrate_defaults_missing_version_to_1():
    state = {"world": {}}
    result = migrate("engine", state)
    assert result == state


def test_migrate_does_not_mutate_input_top_level():
    """migrate() must not mutate the caller's top-level dict (§1.6)."""
    state = {"version": 1, "world": {}}
    snapshot = dict(state)
    migrate("engine", state)
    assert state == snapshot


def test_migrate_future_version_raises():
    state = {"version": 999}
    with pytest.raises(RuntimeError, match="newer than the supported version"):
        migrate("engine", state)


def test_migrate_unknown_component_raises():
    with pytest.raises(ValueError, match="Unknown checkpoint component"):
        migrate("nonexistent", {"version": 1})


def test_migrate_all_components_at_v1():
    # Each component needs whatever minimal v1-required fields its schema
    # demands so the migration framework's pre-migration validation passes.
    # Components without active migrations (engine, cursor) skip validation
    # entirely since the migration loop never runs.
    minimal_v1: dict[str, dict[str, object]] = {
        "engine": {"version": 1},
        "work_chunk": {
            "version": 1,
            "components": [("a", [[0, 0, 0]])],
            "component_order": ["a"],
        },
        "cursor": {"version": 1},
        "static_mixture": {
            "version": 1,
            "lane_id": 0,
            "canonical_replicas": 1,
            "chunk_size_hint": None,
            "seed": 0,
            "chunk_size": 1,
            "knobs": {"shuffle_shards": False, "shuffle_within_shard": False},
            "global_chunk_index": 0,
            "weights": {},
            "component_order": [],
            "dataset_ids": {},
            "cursor_positions": {},
            "cursor_epochs": {},
        },
    }
    for component, state in minimal_v1.items():
        result = migrate(component, state)
        assert result["version"] == CURRENT_VERSIONS[component]


def test_register_and_run_migration(monkeypatch):
    """Register a v1->v2 migration and verify it runs.

    Migration functions take the validated v_N *instance* and return a
    v_{N+1} dict.
    """
    from zephon.core.checkpoint import CursorStateV1

    monkeypatch.setitem(CURRENT_VERSIONS, "cursor", 2)
    monkeypatch.setitem(_MIGRATIONS, "cursor", {})

    def _cursor_v1_to_v2(v1: CursorStateV1) -> dict:
        d = v1.to_dict()
        d["new_field"] = "added_by_migration"
        return d

    register_migration("cursor", from_version=1, fn=_cursor_v1_to_v2)

    state = {"version": 1, "position": 10}
    result = migrate("cursor", state)
    assert result["version"] == 2
    assert result["new_field"] == "added_by_migration"
    assert result["position"] == 10


def test_migrate_validates_intermediate_version_before_running_migration(
    monkeypatch,
):
    """A garbage v_N input is rejected before the v_N -> v_{N+1} migration runs."""
    import pytest

    from zephon.core.checkpoint import EngineStateV1

    monkeypatch.setitem(CURRENT_VERSIONS, "engine", 2)
    monkeypatch.setitem(_MIGRATIONS, "engine", {})

    sentinel = {"ran": False}

    def _engine_v1_to_v2(v1: EngineStateV1) -> dict:
        sentinel["ran"] = True
        return {"version": 2}

    register_migration("engine", from_version=1, fn=_engine_v1_to_v2)

    # Malformed v1: "world" is not a dict — caught by EngineStateV1.__post_init__.
    broken = {
        "version": 1,
        "world": "not-a-dict",
        "progress": {},
        "lane_next_cid": {},
        "lane_ws_state": {},
        "last_round_id": None,
        "checkpoint_reload_count": 0,
    }
    with pytest.raises(ValueError, match="world must be a dict"):
        migrate("engine", broken)
    assert sentinel["ran"] is False


def test_register_duplicate_migration_raises(monkeypatch):
    monkeypatch.setitem(_MIGRATIONS, "cursor", {})
    register_migration("cursor", from_version=99, fn=lambda s: s)
    with pytest.raises(ValueError, match="already registered"):
        register_migration("cursor", from_version=99, fn=lambda s: s)


def test_migrate_missing_intermediate_schema_raises_clearly(monkeypatch):
    """A registered migration without a paired v_N schema raises with guidance."""
    import pytest

    from zephon.core.checkpoint._migrations import _SCHEMAS

    monkeypatch.setitem(CURRENT_VERSIONS, "cursor", 3)
    monkeypatch.setitem(_MIGRATIONS, "cursor", {})
    monkeypatch.setitem(_SCHEMAS, "cursor", {})

    register_migration("cursor", from_version=1, fn=lambda v: {"version": 2})
    register_migration("cursor", from_version=2, fn=lambda v: {"version": 3})

    with pytest.raises(RuntimeError, match="schema.*v1.*_SCHEMAS"):
        migrate("cursor", {"version": 1})


def test_migrate_raises_on_incomplete_chain(monkeypatch):
    """current=3 but only v1→v2 registered → migrate raises at v2."""
    import pytest

    from zephon.core.checkpoint import CursorStateV1
    from zephon.core.checkpoint._migrations import _SCHEMAS

    monkeypatch.setitem(CURRENT_VERSIONS, "cursor", 3)
    monkeypatch.setitem(_MIGRATIONS, "cursor", {})
    monkeypatch.setitem(_SCHEMAS, "cursor", {1: CursorStateV1})

    register_migration("cursor", from_version=1, fn=lambda v: {"version": 2})

    with pytest.raises(ValueError, match="No migration registered.*v2"):
        migrate("cursor", {"version": 1})


def test_current_version_reflects_registry(monkeypatch):
    from zephon.core.checkpoint import current_version

    assert current_version("engine") == CURRENT_VERSIONS["engine"]
    assert current_version("cursor") == CURRENT_VERSIONS["cursor"]

    monkeypatch.setitem(CURRENT_VERSIONS, "cursor", 42)
    assert current_version("cursor") == 42


# ---------------------------------------------------------------------------
# StaticMixture v1 → v2 migration — direct function tests
# ---------------------------------------------------------------------------


def _v1_static_mixture_raw(**overrides):
    """Smallest dict that satisfies StaticMixtureStateV1's required fields."""
    base = {
        "version": 1,
        "lane_id": 0,
        "canonical_replicas": 1,
        "chunk_size_hint": None,
        "seed": 0,
        "chunk_size": 1,
        "knobs": {"shuffle_shards": True, "shuffle_within_shard": False},
        "global_chunk_index": 0,
        "weights": {"a": 1.0},
        "component_order": ["a"],
        "dataset_ids": {"a": 0},
        "cursor_positions": {"a": 0},
        "cursor_epochs": {"a": 0},
        "exhausted_policy": "stop",
    }
    base.update(overrides)
    return base


def test_v1_to_v2_fans_block_size_out_to_every_cursor():
    """The single v1 knobs.shuffle_block_size becomes one entry per dataset."""
    from zephon.core.checkpoint._migrations import _static_mixture_v1_to_v2
    from zephon.core.checkpoint._schemas import StaticMixtureStateV1

    v1 = StaticMixtureStateV1.from_dict(
        _v1_static_mixture_raw(
            knobs={
                "shuffle_shards": True,
                "shuffle_within_shard": False,
                "shuffle_block_size": 64,
            },
            weights={"a": 0.5, "b": 0.5},
            component_order=["a", "b"],
            dataset_ids={"a": 0, "b": 1},
            cursor_positions={"a": 0, "b": 0},
            cursor_epochs={"a": 0, "b": 0},
        )
    )
    d = _static_mixture_v1_to_v2(v1)
    assert d["cursor_block_sizes"] == {"a": 64, "b": 64}
    assert d["shuffle_block_size_spec"] == 64
    assert "shuffle_block_size" not in d["knobs"]
    assert d["knobs"] == {"shuffle_shards": True, "shuffle_within_shard": False}


def test_v1_to_v2_propagates_none_block_size():
    """v1 without knobs.shuffle_block_size → None for every cursor."""
    from zephon.core.checkpoint._migrations import _static_mixture_v1_to_v2
    from zephon.core.checkpoint._schemas import StaticMixtureStateV1

    v1 = StaticMixtureStateV1.from_dict(
        _v1_static_mixture_raw(
            dataset_ids={"a": 0, "b": 1},
            cursor_positions={"a": 0, "b": 0},
            cursor_epochs={"a": 0, "b": 0},
            weights={"a": 0.5, "b": 0.5},
            component_order=["a", "b"],
        )
    )
    d = _static_mixture_v1_to_v2(v1)
    assert d["cursor_block_sizes"] == {"a": None, "b": None}
    assert d["shuffle_block_size_spec"] is None


def test_v1_to_v2_preserves_explicit_allocation_mode():
    """If v1 already carries allocation_mode, the migration does not touch it."""
    from zephon.core.checkpoint._migrations import _static_mixture_v1_to_v2
    from zephon.core.checkpoint._schemas import StaticMixtureStateV1

    v1 = StaticMixtureStateV1.from_dict(
        _v1_static_mixture_raw(
            allocation_mode="accumulator",
            accumulators={"a": 0.25},
        )
    )
    d = _static_mixture_v1_to_v2(v1)
    assert d["allocation_mode"] == "accumulator"
    assert d["accumulators"] == {"a": 0.25}


def test_v1_to_v2_infers_accumulator_mode_from_accumulators_presence():
    """Pre-strategy v1 with accumulators set → accumulator mode."""
    from zephon.core.checkpoint._migrations import _static_mixture_v1_to_v2
    from zephon.core.checkpoint._schemas import StaticMixtureStateV1

    v1 = StaticMixtureStateV1.from_dict(_v1_static_mixture_raw(accumulators={"a": 0.1}))
    assert v1.allocation_mode is None  # confirm the pre-strategy shape
    d = _static_mixture_v1_to_v2(v1)
    assert d["allocation_mode"] == "accumulator"


def test_v1_to_v2_infers_legacy_fixed_from_absent_accumulators():
    """Pre-strategy v1 without accumulators → legacy_fixed."""
    from zephon.core.checkpoint._migrations import _static_mixture_v1_to_v2
    from zephon.core.checkpoint._schemas import StaticMixtureStateV1

    v1 = StaticMixtureStateV1.from_dict(_v1_static_mixture_raw())
    assert v1.allocation_mode is None and v1.accumulators is None
    d = _static_mixture_v1_to_v2(v1)
    assert d["allocation_mode"] == "legacy_fixed"


def test_v1_to_v2_does_not_mutate_v1_knobs():
    """Building the new knobs dict via comprehension must not alias v1.knobs."""
    from zephon.core.checkpoint._migrations import _static_mixture_v1_to_v2
    from zephon.core.checkpoint._schemas import StaticMixtureStateV1

    knobs = {
        "shuffle_shards": True,
        "shuffle_within_shard": False,
        "shuffle_block_size": 16,
    }
    knobs_snapshot = dict(knobs)
    v1 = StaticMixtureStateV1.from_dict(_v1_static_mixture_raw(knobs=knobs))
    d = _static_mixture_v1_to_v2(v1)
    # v1's knobs (and the caller's original dict) must be untouched.
    assert v1.knobs == knobs_snapshot
    assert knobs == knobs_snapshot
    # The new dict is a fresh object, not the same instance.
    assert d["knobs"] is not v1.knobs


def test_v1_to_v2_migrate_lands_on_valid_v2(monkeypatch):
    """Explicit single-hop check: pinning current to 2 makes migrate() stop at
    v2, so we validate exactly the v1->v2 step — version stamp included —
    without depending on how many versions exist downstream.
    """
    from zephon.core.checkpoint._schemas import StaticMixtureStateV2

    monkeypatch.setitem(CURRENT_VERSIONS, "static_mixture", 2)
    sm = StaticMixtureStateV2.from_dict(
        migrate("static_mixture", _v1_static_mixture_raw())
    )
    assert sm.version == 2
    assert sm.exhausted_policy == "stop"  # scalar policy preserved at v2
    assert "shuffle_block_size" not in sm.knobs  # promoted out of knobs in v2


def test_v1_end_to_end_via_migrate_validates_current():
    """The migrate() entry point produces a dict the current schema accepts."""
    from zephon.core.checkpoint._schemas import StaticMixtureStateV5

    raw = _v1_static_mixture_raw(
        knobs={
            "shuffle_shards": False,
            "shuffle_within_shard": True,
            "shuffle_block_size": 8,
        },
        weights={"a": 0.5, "b": 0.5},
        component_order=["a", "b"],
        dataset_ids={"a": 0, "b": 1},
        cursor_positions={"a": 0, "b": 0},
        cursor_epochs={"a": 0, "b": 0},
    )
    migrated = migrate("static_mixture", raw)
    sm = StaticMixtureStateV5.from_dict(migrated)
    assert sm.version == 5
    assert sm.cursor_block_sizes == {"a": 8, "b": 8}
    assert sm.shuffle_block_size_spec == 8
    assert sm.allocation_mode == "legacy_fixed"
    assert "shuffle_block_size" not in sm.knobs
    # v2 -> v3 broadcasts the scalar exhaustion knobs across component_order.
    assert sm.exhausted_policy == {"a": "stop", "b": "stop"}
    assert sm.reshuffle_on_repeat == {"a": False, "b": False}
    assert sm.max_repeats == {"a": None, "b": None}
    # stop_after_passes is an additive v3 field absent in older checkpoints.
    assert sm.stop_after_passes is None
    # v3 -> v4 fills the routing tag with the pre-fix default.
    assert sm.lane_assignment == "modulo"
    assert sm.token_deficits is None
    assert sm.token_ratios is None


def _v2_static_mixture_raw(**overrides):
    """Smallest dict that satisfies StaticMixtureStateV2 (one component)."""
    base = {
        "version": 2,
        "lane_id": 0,
        "canonical_replicas": 1,
        "chunk_size_hint": None,
        "seed": 0,
        "chunk_size": 1,
        "knobs": {"shuffle_shards": False, "shuffle_within_shard": False},
        "global_chunk_index": 0,
        "weights": {"a": 1.0},
        "component_order": ["a"],
        "dataset_ids": {"a": 0},
        "cursor_positions": {"a": 0},
        "cursor_epochs": {"a": 0},
        "cursor_states": {},
        "cursor_block_sizes": {"a": None},
        "shuffle_block_size_spec": None,
        "exhausted_policy": "stop",
        "reshuffle_on_repeat": False,
        "max_repeats": None,
        "allocation_mode": "legacy_fixed",
        "accumulators": None,
    }
    base.update(overrides)
    return base


def test_v2_to_v3_broadcasts_scalar_policies_across_components():
    """v2's scalar exhaustion knobs fan out into per-dataset dicts keyed by
    component_order, leaving every other field untouched."""
    from zephon.core.checkpoint._migrations import _static_mixture_v2_to_v3
    from zephon.core.checkpoint._schemas import StaticMixtureStateV2

    v2 = StaticMixtureStateV2.from_dict(
        _v2_static_mixture_raw(
            weights={"a": 0.5, "b": 0.5},
            component_order=["a", "b"],
            dataset_ids={"a": 0, "b": 1},
            cursor_positions={"a": 0, "b": 0},
            cursor_epochs={"a": 0, "b": 0},
            cursor_block_sizes={"a": None, "b": None},
            exhausted_policy="repeat",
            reshuffle_on_repeat=True,
            max_repeats=5,
        )
    )
    d = _static_mixture_v2_to_v3(v2)
    assert d["exhausted_policy"] == {"a": "repeat", "b": "repeat"}
    assert d["reshuffle_on_repeat"] == {"a": True, "b": True}
    assert d["max_repeats"] == {"a": 5, "b": 5}
    assert d["stop_after_passes"] is None  # v2 had no global floor
    # Non-policy fields are carried through unchanged.
    assert d["cursor_block_sizes"] == {"a": None, "b": None}
    assert d["allocation_mode"] == "legacy_fixed"


def test_v2_to_v3_migrate_lands_on_valid_v3(monkeypatch):
    """Explicit single-hop check: pin current to 3 so migrate() stops at v3
    (stays a v2->v3-only test even once later versions are added)."""
    from zephon.core.checkpoint._schemas import StaticMixtureStateV3

    monkeypatch.setitem(CURRENT_VERSIONS, "static_mixture", 3)
    sm = StaticMixtureStateV3.from_dict(
        migrate("static_mixture", _v2_static_mixture_raw())
    )
    assert sm.version == 3
    assert sm.exhausted_policy == {"a": "stop"}  # scalar broadcast to a dict
    assert sm.stop_after_passes is None  # additive v3 field, absent in v2 checkpoints


def test_v2_end_to_end_via_migrate_validates_current():
    """migrate() chains a v2 payload all the way to the current schema."""
    from zephon.core.checkpoint._schemas import StaticMixtureStateV5

    migrated = migrate("static_mixture", _v2_static_mixture_raw())
    sm = StaticMixtureStateV5.from_dict(migrated)
    assert sm.version == 5
    assert sm.exhausted_policy == {"a": "stop"}  # v2 -> v3 scalar broadcast
    assert sm.stop_after_passes is None  # additive v3 field, absent in v2 checkpoints
    assert sm.lane_assignment == "modulo"  # v3 -> v4 fill
    assert sm.token_deficits is None
    assert sm.token_ratios is None


def test_v2_to_v3_does_not_mutate_input():
    """The migration must not alias or mutate the source v2 instance/dict."""
    from zephon.core.checkpoint._migrations import _static_mixture_v2_to_v3
    from zephon.core.checkpoint._schemas import StaticMixtureStateV2

    v2 = StaticMixtureStateV2.from_dict(_v2_static_mixture_raw())
    d = _static_mixture_v2_to_v3(v2)
    d["exhausted_policy"]["a"] = "repeat"
    # v2 is frozen; its scalar policy is unaffected by mutating the v3 dict.
    assert v2.exhausted_policy == "stop"


# ---------------------------------------------------------------------------
# StaticMixture v3 -> v4 migration (lane_assignment)
# ---------------------------------------------------------------------------


def _v3_static_mixture_raw(**overrides):
    """Smallest dict that satisfies StaticMixtureStateV3 (one component)."""
    base = {
        "version": 3,
        "lane_id": 0,
        "canonical_replicas": 1,
        "chunk_size_hint": None,
        "seed": 0,
        "chunk_size": 1,
        "knobs": {"shuffle_shards": False, "shuffle_within_shard": False},
        "global_chunk_index": 0,
        "weights": {"a": 1.0},
        "component_order": ["a"],
        "dataset_ids": {"a": 0},
        "cursor_positions": {"a": 0},
        "cursor_epochs": {"a": 0},
        "cursor_states": {},
        "cursor_block_sizes": {"a": None},
        "shuffle_block_size_spec": None,
        "exhausted_policy": {"a": "stop"},
        "reshuffle_on_repeat": {"a": True},
        "max_repeats": {"a": None},
        "stop_after_passes": None,
        "allocation_mode": "legacy_fixed",
        "accumulators": None,
    }
    base.update(overrides)
    return base


def test_v3_to_v4_defaults_lane_assignment_to_modulo():
    """A v3 checkpoint predates the fix, so it must replay through modulo."""
    from zephon.core.checkpoint._migrations import _static_mixture_v3_to_v4
    from zephon.core.checkpoint._schemas import StaticMixtureStateV3

    v3 = StaticMixtureStateV3.from_dict(_v3_static_mixture_raw())
    d = _static_mixture_v3_to_v4(v3)
    assert d["lane_assignment"] == "modulo"


def test_v3_to_v4_preserves_all_other_fields():
    from zephon.core.checkpoint._migrations import _static_mixture_v3_to_v4
    from zephon.core.checkpoint._schemas import StaticMixtureStateV3

    v3 = StaticMixtureStateV3.from_dict(
        _v3_static_mixture_raw(allocation_mode="accumulator", accumulators={"a": 0.3})
    )
    d = _static_mixture_v3_to_v4(v3)
    assert d["allocation_mode"] == "accumulator"
    assert d["accumulators"] == {"a": 0.3}
    assert d["exhausted_policy"] == {"a": "stop"}
    assert d["stop_after_passes"] is None


def test_v3_to_v4_does_not_mutate_input():
    from zephon.core.checkpoint._migrations import _static_mixture_v3_to_v4
    from zephon.core.checkpoint._schemas import StaticMixtureStateV3

    v3 = StaticMixtureStateV3.from_dict(_v3_static_mixture_raw())
    d = _static_mixture_v3_to_v4(v3)
    # The added field lives only on the migrated dict, not the frozen v3.
    assert d["lane_assignment"] == "modulo"
    assert not hasattr(v3, "lane_assignment")


def test_v3_to_v4_migrate_lands_on_valid_v4(monkeypatch):
    """Single-hop check: pin current to 4 so this stays a v3->v4-only test
    even once later versions are added."""
    from zephon.core.checkpoint._schemas import StaticMixtureStateV4

    monkeypatch.setitem(CURRENT_VERSIONS, "static_mixture", 4)
    sm = StaticMixtureStateV4.from_dict(
        migrate("static_mixture", _v3_static_mixture_raw())
    )
    assert sm.version == 4
    assert sm.lane_assignment == "modulo"


def test_v3_end_to_end_validates_current():
    """migrate() upgrades a v3 dict all the way to the current schema."""
    from zephon.core.checkpoint._schemas import StaticMixtureStateV5

    migrated = migrate("static_mixture", _v3_static_mixture_raw())
    sm = StaticMixtureStateV5.from_dict(migrated)
    assert sm.version == 5
    assert sm.lane_assignment == "modulo"  # v3 -> v4 fill
    assert sm.token_deficits is None
    assert sm.token_ratios is None


# ---------------------------------------------------------------------------
# StaticMixture v4 -> v5 migration (token-aware fields)
# ---------------------------------------------------------------------------


def _v4_static_mixture_raw(**overrides):
    """Smallest dict that satisfies StaticMixtureStateV4 (v3 + lane_assignment)."""
    base = _v3_static_mixture_raw(version=4, lane_assignment="modulo")
    base.update(overrides)
    return base


def test_v4_to_v5_fills_inert_token_defaults():
    from zephon.core.checkpoint._migrations import _static_mixture_v4_to_v5
    from zephon.core.checkpoint._schemas import StaticMixtureStateV4

    v4 = StaticMixtureStateV4.from_dict(_v4_static_mixture_raw())
    d = _static_mixture_v4_to_v5(v4)
    assert d["token_deficits"] is None
    assert d["token_ratios"] is None
    assert d["lane_assignment"] == "modulo"
    assert d["exhausted_policy"] == {"a": "stop"}


def test_v4_to_v5_migrate_lands_on_valid_v5(monkeypatch):
    """Single-hop check: pin current to 5 so this stays a v4->v5-only test."""
    from zephon.core.checkpoint._schemas import StaticMixtureStateV5

    monkeypatch.setitem(CURRENT_VERSIONS, "static_mixture", 5)
    sm = StaticMixtureStateV5.from_dict(
        migrate("static_mixture", _v4_static_mixture_raw())
    )
    assert sm.version == 5
    assert sm.token_deficits is None
    assert sm.token_ratios is None


def test_v4_to_v5_does_not_mutate_input():
    from zephon.core.checkpoint._migrations import _static_mixture_v4_to_v5
    from zephon.core.checkpoint._schemas import StaticMixtureStateV4

    v4 = StaticMixtureStateV4.from_dict(_v4_static_mixture_raw())
    d = _static_mixture_v4_to_v5(v4)
    assert d["token_deficits"] is None
    assert not hasattr(v4, "token_deficits")


def test_work_chunk_v1_migrates_to_v2_with_none_target():
    raw = {
        "version": 1,
        "seed": 3,
        "components": [("a", [[0, 0, 0], [0, 0, 1]])],
        "component_order": ["a"],
        "total_samples": 2,
    }
    migrated = migrate("work_chunk", raw)
    assert migrated["version"] == 2
    assert migrated["target_mixture"] is None
    assert migrated["components"] == [("a", [[0, 0, 0], [0, 0, 1]])]


def test_work_chunk_v2_payload_loads_directly():
    from zephon.core.checkpoint import WorkChunkStateV2

    raw = {
        "version": 2,
        "components": [("a", [[0, 0, 0]])],
        "component_order": ["a"],
        "target_mixture": {"a": 1.0},
    }
    chunk = WorkChunkStateV2.load(raw)
    assert chunk.target_mixture == {"a": 1.0}


def test_work_chunk_newer_than_supported_rejected():
    with pytest.raises(RuntimeError, match="newer"):
        migrate("work_chunk", {"version": 3, "components": [], "component_order": []})
