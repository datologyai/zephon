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
    for component in ("engine", "work_chunk", "static_mixture", "cursor"):
        state = {"version": 1}
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
