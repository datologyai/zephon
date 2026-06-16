# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for checkpoint dataclass schemas and CheckpointMixin.from_dict()."""

import dataclasses

from zephon.core.checkpoint import (
    CursorStateV1,
    EngineStateV1,
    StaticMixtureStateV1,
    StaticMixtureStateV2,
    WorkChunkStateV1,
)
from zephon.core.checkpoint._migrations import (
    _MIGRATIONS,
    CURRENT_VERSIONS,
    register_migration,
)

# -- CursorStateV1 -----------------------------------------------------------


def test_cursor_defaults():
    c = CursorStateV1()
    assert c.position == 0
    assert c.epoch == 0
    assert c.block_rng_snapshot is None


def test_cursor_from_dict_full():
    raw = {
        "position": 42,
        "epoch": 3,
        "block_rng_snapshot": {"rng_state": [1], "block_count": 5},
    }
    c = CursorStateV1.from_dict(raw)
    assert c.position == 42
    assert c.epoch == 3
    assert c.block_rng_snapshot == {"rng_state": [1], "block_count": 5}


def test_cursor_from_dict_missing_fields_uses_defaults():
    c = CursorStateV1.from_dict({})
    assert c.position == 0
    assert c.epoch == 0
    assert c.block_rng_snapshot is None


def test_cursor_from_dict_coerces_strings():
    c = CursorStateV1.from_dict({"position": "10", "epoch": "2"})
    assert c.position == 10
    assert c.epoch == 2


def test_cursor_from_dict_ignores_unknown_fields():
    c = CursorStateV1.from_dict({"position": 5, "unknown_future_field": True})
    assert c.position == 5


# -- WorkChunkStateV1 --------------------------------------------------------


def test_work_chunk_roundtrip():
    raw = {
        "version": 1,
        "seed": 42,
        "components": [("a", [[0, 0, 1], [0, 0, 2]])],
        "component_order": ["a"],
        "total_samples": 2,
    }
    wc = WorkChunkStateV1.from_dict(raw)
    assert wc.seed == 42
    assert wc.total_samples == 2
    d = dataclasses.asdict(wc)
    assert d["version"] == 1
    assert d["seed"] == 42


def test_work_chunk_requires_components():
    """components / component_order are always-written; missing raises."""
    import pytest

    with pytest.raises(TypeError, match="missing"):
        WorkChunkStateV1()
    with pytest.raises(ValueError) as exc_info:
        WorkChunkStateV1.from_dict({"version": 1})
    msg = str(exc_info.value)
    assert "components: missing" in msg
    assert "component_order: missing" in msg


def test_work_chunk_defaults_for_optional_fields():
    wc = WorkChunkStateV1(components=[], component_order=[])
    assert wc.version == 1
    assert wc.seed is None
    assert wc.total_samples is None


def test_work_chunk_none_seed_preserved():
    wc = WorkChunkStateV1.from_dict(
        {"components": [], "component_order": [], "seed": None}
    )
    assert wc.seed is None


# -- StaticMixtureStateV1 ----------------------------------------------------


def _minimal_static_mixture_raw(**overrides):
    """Smallest dict that satisfies StaticMixtureStateV1's required fields."""
    base = {
        "version": 1,
        "lane_id": 0,
        "canonical_replicas": 1,
        "chunk_size_hint": None,
        "seed": 0,
        "chunk_size": 0,
        "knobs": {},
        "global_chunk_index": 0,
        "weights": {},
        "component_order": [],
        "dataset_ids": {},
        "cursor_positions": {},
        "cursor_epochs": {},
        "exhausted_policy": "stop",
    }
    base.update(overrides)
    return base


def test_static_mixture_requires_always_written_fields():
    import pytest

    with pytest.raises(ValueError) as exc_info:
        StaticMixtureStateV1.from_dict({"version": 1})
    msg = str(exc_info.value)
    # Every always-written field should be in the error message.
    for name in (
        "lane_id",
        "canonical_replicas",
        "weights",
        "knobs",
        "dataset_ids",
        "cursor_positions",
        "cursor_epochs",
    ):
        assert f"{name}: missing" in msg, f"expected '{name}: missing' in error"


def test_static_mixture_defaults_for_optional_fields():
    sm = StaticMixtureStateV1.from_dict(_minimal_static_mixture_raw())
    assert sm.version == 1
    assert sm.accumulators is None
    assert sm.cursor_states == {}
    assert sm.allocation_mode is None
    assert sm.reshuffle_on_repeat is False
    assert sm.max_repeats is None


def test_static_mixture_from_dict_accumulator_mode():
    raw = _minimal_static_mixture_raw(
        accumulators={"a": 0.7, "b": 0.3},
        weights={"a": 0.7, "b": 0.3},
    )
    sm = StaticMixtureStateV1.from_dict(raw)
    assert sm.accumulators == {"a": 0.7, "b": 0.3}


def test_static_mixture_from_dict_legacy_mode():
    raw = _minimal_static_mixture_raw(weights={"a": 1.0})
    sm = StaticMixtureStateV1.from_dict(raw)
    assert sm.accumulators is None


def test_static_mixture_allocation_mode_roundtrip():
    sm = StaticMixtureStateV1.from_dict(
        _minimal_static_mixture_raw(allocation_mode="accumulator")
    )
    assert sm.allocation_mode == "accumulator"

    sm_legacy = StaticMixtureStateV1.from_dict(
        _minimal_static_mixture_raw(allocation_mode="legacy_fixed")
    )
    assert sm_legacy.allocation_mode == "legacy_fixed"


def test_static_mixture_allocation_mode_missing_defaults_to_none():
    """Pre-strategy v1 checkpoints have no allocation_mode tag."""
    sm = StaticMixtureStateV1.from_dict(_minimal_static_mixture_raw())
    assert sm.allocation_mode is None


def test_static_mixture_exhausted_policy_missing_defaults_to_stop():
    """Pre-#186 v1 checkpoints have no exhausted_policy tag."""
    raw = _minimal_static_mixture_raw()
    del raw["exhausted_policy"]
    sm = StaticMixtureStateV1.from_dict(raw)
    assert sm.exhausted_policy == "stop"


# -- StaticMixtureStateV2 ----------------------------------------------------


def _minimal_static_mixture_v2_raw(**overrides):
    """Smallest dict that satisfies StaticMixtureStateV2's required fields.

    v2 has no field-level defaults (besides ``version``), so the fixture must
    spell out every field. Tests override one slot at a time to exercise the
    invariants in ``__post_init__`` in isolation.
    """
    base = {
        "version": 2,
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
        "cursor_states": {},
        "cursor_block_sizes": {},
        "shuffle_block_size_spec": None,
        "exhausted_policy": "stop",
        "reshuffle_on_repeat": False,
        "max_repeats": None,
        "allocation_mode": "legacy_fixed",
        "accumulators": None,
    }
    base.update(overrides)
    return base


def test_static_mixture_v2_requires_every_field_except_version():
    """v2 has no field-level defaults — promotion-to-required is the contract."""
    import pytest

    with pytest.raises(ValueError) as exc_info:
        StaticMixtureStateV2.from_dict({"version": 2})
    msg = str(exc_info.value)
    for name in (
        "lane_id",
        "canonical_replicas",
        "chunk_size_hint",
        "seed",
        "chunk_size",
        "knobs",
        "global_chunk_index",
        "weights",
        "component_order",
        "dataset_ids",
        "cursor_positions",
        "cursor_epochs",
        "cursor_states",
        "cursor_block_sizes",
        "shuffle_block_size_spec",
        "exhausted_policy",
        "reshuffle_on_repeat",
        "max_repeats",
        "allocation_mode",
        "accumulators",
    ):
        assert f"{name}: missing" in msg, f"expected '{name}: missing' in error"


def test_static_mixture_v2_minimal_roundtrip():
    sm = StaticMixtureStateV2.from_dict(_minimal_static_mixture_v2_raw())
    assert sm.version == 2
    assert sm.allocation_mode == "legacy_fixed"
    assert sm.accumulators is None
    assert sm.cursor_block_sizes == {}
    assert sm.shuffle_block_size_spec is None


def test_static_mixture_v2_rejects_invalid_allocation_mode():
    import pytest

    with pytest.raises(ValueError, match="allocation_mode='banana'"):
        StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(allocation_mode="banana")
        )


def test_static_mixture_v2_rejects_empty_allocation_mode():
    """Empty-string allocation_mode used to silently load as legacy_fixed."""
    import pytest

    with pytest.raises(ValueError, match="allocation_mode=''"):
        StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(allocation_mode="")
        )


def test_static_mixture_v2_accepts_both_valid_allocation_modes():
    for mode in ("accumulator", "legacy_fixed"):
        sm = StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(
                allocation_mode=mode,
                accumulators=({"a": 0.5} if mode == "accumulator" else None),
                weights={"a": 1.0} if mode == "accumulator" else {},
                cursor_block_sizes={"a": None} if mode == "accumulator" else {},
                dataset_ids={"a": 0} if mode == "accumulator" else {},
                cursor_positions={"a": 0} if mode == "accumulator" else {},
                cursor_epochs={"a": 0} if mode == "accumulator" else {},
                cursor_states=(
                    {"a": {"position": 0, "epoch": 0}} if mode == "accumulator" else {}
                ),
                component_order=["a"] if mode == "accumulator" else [],
            )
        )
        assert sm.allocation_mode == mode


def test_static_mixture_v2_rejects_stale_shuffle_block_size_in_knobs():
    """The shuffle_block_size key moved out of knobs in v2."""
    import pytest

    with pytest.raises(ValueError, match="knobs must not carry 'shuffle_block_size'"):
        StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(
                knobs={
                    "shuffle_shards": False,
                    "shuffle_within_shard": False,
                    "shuffle_block_size": 64,
                }
            )
        )


def test_static_mixture_v2_rejects_missing_knob_keys():
    import pytest

    with pytest.raises(ValueError, match="knobs missing required keys"):
        StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(knobs={"shuffle_shards": True})
        )


def test_static_mixture_v2_rejects_non_dict_knobs():
    import pytest

    with pytest.raises(ValueError, match="knobs must be a dict"):
        StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(knobs=["shuffle_shards"])  # type: ignore[arg-type]
        )


def test_static_mixture_v2_rejects_invalid_spec_string():
    import pytest

    with pytest.raises(ValueError, match="not a valid sentinel"):
        StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(shuffle_block_size_spec="banana")
        )


def test_static_mixture_v2_accepts_valid_spec_sentinels():
    for spec in ("auto", "global"):
        sm = StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(shuffle_block_size_spec=spec)
        )
        assert sm.shuffle_block_size_spec == spec


def test_static_mixture_v2_accepts_int_and_none_spec():
    for spec in (None, 128):
        sm = StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(shuffle_block_size_spec=spec)
        )
        assert sm.shuffle_block_size_spec == spec


def test_static_mixture_v2_rejects_bool_spec():
    """bool is an int subclass — explicitly rejected so True doesn't smuggle in as 1."""
    import pytest

    with pytest.raises(ValueError, match="shuffle_block_size_spec must be int"):
        StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(shuffle_block_size_spec=True)
        )


def test_static_mixture_v2_rejects_float_spec():
    import pytest

    with pytest.raises(ValueError, match="shuffle_block_size_spec must be int"):
        StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(shuffle_block_size_spec=1.5)
        )


def test_static_mixture_v2_rejects_list_spec():
    import pytest

    with pytest.raises(ValueError, match="shuffle_block_size_spec must be int"):
        StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(shuffle_block_size_spec=[1, 2])  # type: ignore[arg-type]
        )


def test_static_mixture_v2_rejects_cursor_block_sizes_missing_entry():
    import pytest

    with pytest.raises(ValueError, match="cursor_block_sizes missing entries"):
        StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(
                dataset_ids={"a": 0, "b": 1},
                cursor_block_sizes={"a": 64},  # "b" missing
            )
        )


def test_static_mixture_v2_rejects_cursor_block_sizes_extra_entry():
    import pytest

    with pytest.raises(ValueError, match="cursor_block_sizes has unknown entries"):
        StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(
                dataset_ids={"a": 0},
                cursor_block_sizes={"a": 64, "ghost": 32},
            )
        )


def test_static_mixture_v2_rejects_bool_cursor_block_size():
    """bool slipping in as block size masks an int — reject it."""
    import pytest

    with pytest.raises(ValueError, match="must be int or None, got bool"):
        StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(
                dataset_ids={"a": 0},
                cursor_block_sizes={"a": True},
            )
        )


def test_static_mixture_v2_rejects_string_cursor_block_size():
    """cursor_block_sizes values are resolved ints; sentinels never land here."""
    import pytest

    with pytest.raises(ValueError, match="must be int or None, got str"):
        StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(
                dataset_ids={"a": 0},
                cursor_block_sizes={"a": "auto"},
            )
        )


def test_static_mixture_v2_accumulates_multiple_invariant_errors():
    """Multiple invariant violations surface in a single ValueError."""
    import pytest

    with pytest.raises(ValueError) as exc_info:
        StaticMixtureStateV2.from_dict(
            _minimal_static_mixture_v2_raw(
                allocation_mode="bogus",
                knobs={"shuffle_shards": True, "shuffle_block_size": 1},
                shuffle_block_size_spec=2.5,
            )
        )
    msg = str(exc_info.value)
    assert "allocation_mode='bogus'" in msg
    assert "shuffle_block_size_spec must be int" in msg
    assert "knobs missing required keys" in msg
    assert "knobs must not carry 'shuffle_block_size'" in msg


# -- EngineStateV1 ------------------------------------------------------------


def _minimal_engine_raw(**overrides):
    """Smallest dict that satisfies EngineStateV1's required fields."""
    base = {
        "version": 1,
        "world": {"canonical_replicas": 1},
        "progress": {},
        "lane_next_cid": {},
        "lane_ws_state": {},
        "last_round_id": None,
        "checkpoint_reload_count": 0,
    }
    base.update(overrides)
    return base


def test_engine_requires_always_written_fields():
    import pytest

    with pytest.raises(TypeError, match="missing"):
        EngineStateV1()  # type: ignore[call-arg]
    with pytest.raises(ValueError) as exc_info:
        EngineStateV1.from_dict({"version": 1})
    msg = str(exc_info.value)
    for name in (
        "world",
        "progress",
        "lane_next_cid",
        "lane_ws_state",
        "last_round_id",
        "checkpoint_reload_count",
    ):
        assert f"{name}: missing" in msg, f"expected '{name}: missing' in error"


def test_engine_defaults_for_optional_fields():
    e = EngineStateV1.from_dict(_minimal_engine_raw())
    assert e.version == 1
    assert e.inflight == {}
    assert e.replay_cursors == {}
    assert e.checkpoint_reload_count == 0


def test_engine_from_dict_minimal():
    """Oldest v1 checkpoint — only required fields."""
    raw = {
        "version": 1,
        "world": {"canonical_replicas": 4},
        "progress": {"0": {"chunk_id": 10, "offset": 5}},
        "lane_next_cid": {"0": 11},
        "lane_ws_state": {"0": {}},
        "last_round_id": "abc",
        "checkpoint_reload_count": "2",  # string — should be coerced
    }
    e = EngineStateV1.from_dict(raw)
    assert e.checkpoint_reload_count == 2
    assert e.inflight == {}  # default
    assert e.replay_cursors == {}  # default


def test_engine_from_dict_full():
    raw = {
        "version": 1,
        "world": {"canonical_replicas": 2},
        "progress": {},
        "lane_next_cid": {},
        "lane_ws_state": {},
        "last_round_id": None,
        "checkpoint_reload_count": 0,
        "inflight": {"0": {"5": {"version": 1}}},
        "rr_next_idx": {"key": 3},
        "replay_cursors": {"0": [1, 2, [], [0, 0, 0]]},
        "epoch_boundaries": {"0": [10, 20]},
    }
    e = EngineStateV1.from_dict(raw)
    assert e.inflight == {"0": {"5": {"version": 1}}}
    assert e.rr_next_idx == {"key": 3}
    assert e.epoch_boundaries == {"0": [10, 20]}


def test_engine_asdict_roundtrip():
    e = EngineStateV1(
        version=1,
        world={"canonical_replicas": 1},
        progress={"0": {"chunk_id": 0, "offset": 0}},
        lane_next_cid={"0": 1},
        lane_ws_state={"0": {}},
        last_round_id="test",
        checkpoint_reload_count=0,
    )
    d = dataclasses.asdict(e)
    e2 = EngineStateV1.from_dict(d)
    assert e == e2


def test_engine_lane_emitted_defaults_to_empty():
    """Old v1 checkpoints lack lane_emitted — it must default to {} (unknown)."""
    e = EngineStateV1.from_dict(_minimal_engine_raw())
    assert e.lane_emitted == {}


def test_engine_lane_emitted_roundtrip():
    raw = _minimal_engine_raw(lane_emitted={"0": 5, "1": 4})
    e = EngineStateV1.from_dict(raw)
    assert e.lane_emitted == {"0": 5, "1": 4}
    d = e.to_dict()
    assert d["lane_emitted"] == {"0": 5, "1": 4}
    assert EngineStateV1.from_dict(d) == e


def test_engine_lane_emitted_survives_load_migration_chain():
    """EngineStateV1.load() (migrate + structure) preserves the counters."""
    raw = _minimal_engine_raw(lane_emitted={"0": 7})
    e = EngineStateV1.load(raw)
    assert e.lane_emitted == {"0": 7}


# -- to_dict() helper --------------------------------------------------------


def test_to_dict_shallow_preserves_nested_references():
    """to_dict() must NOT deep-copy nested containers (write-path perf)."""
    nested = [[0, 0, 1], [0, 0, 2]]
    wc = WorkChunkStateV1(
        version=1,
        seed=42,
        components=[("a", nested)],
        component_order=["a"],
        total_samples=2,
    )
    d = wc.to_dict()
    # The inner list should be the *same* object, not a deep copy.
    assert d["components"][0][1] is nested


def test_to_dict_strip_none_drops_none_fields():
    c = CursorStateV1(position=5, epoch=1, block_rng_snapshot=None)
    full = c.to_dict()
    stripped = c.to_dict(strip_none=True)
    assert "block_rng_snapshot" in full
    assert full["block_rng_snapshot"] is None
    assert "block_rng_snapshot" not in stripped
    # Non-None fields stay
    assert stripped["position"] == 5
    assert stripped["epoch"] == 1


def test_to_dict_strip_none_keeps_empty_containers():
    """Empty dict/list are not None — they must survive strip_none."""
    e = EngineStateV1.from_dict(_minimal_engine_raw())
    stripped = e.to_dict(strip_none=True)
    assert stripped["progress"] == {}
    # last_round_id defaults to None and should be dropped
    assert "last_round_id" not in stripped


# -- Schema.load() canonical entry point ------------------------------------


def test_load_runs_migration_chain_before_structuring(monkeypatch):
    """load() = migrate() + from_dict(); migrations must run first."""
    from dataclasses import dataclass
    from typing import ClassVar

    from zephon.core.checkpoint._migrations import _SCHEMAS
    from zephon.core.checkpoint._schemas import CheckpointMixin

    @dataclass
    class _CursorStateV2(CheckpointMixin):
        _COMPONENT: ClassVar[str] = "cursor"
        version: int = 2
        offset_v2: int = 0
        epoch: int = 0

    monkeypatch.setitem(CURRENT_VERSIONS, "cursor", 2)
    monkeypatch.setitem(_MIGRATIONS, "cursor", {})
    monkeypatch.setitem(_SCHEMAS, "cursor", {1: CursorStateV1, 2: _CursorStateV2})

    def _v1_to_v2(v1: CursorStateV1) -> dict:
        return {"version": 2, "offset_v2": v1.position, "epoch": v1.epoch}

    register_migration("cursor", from_version=1, fn=_v1_to_v2)

    raw = {"version": 1, "position": 99, "epoch": 4}
    migrated = _CursorStateV2.load(raw)
    assert migrated.offset_v2 == 99
    assert migrated.epoch == 4


def test_load_does_not_mutate_input_top_level():
    raw = {"version": 1, "position": 7}
    raw_copy = dict(raw)
    CursorStateV1.load(raw)
    assert raw == raw_copy


def test_load_shares_nested_values_with_caller():
    raw = {
        "version": 1,
        "position": 5,
        "block_rng_snapshot": {"rng_state": [1, 2, 3], "block_count": 4},
    }
    inner = raw["block_rng_snapshot"]
    c = CursorStateV1.load(raw)
    assert c.block_rng_snapshot is inner


def test_auto_coerce_bool_before_int():
    """bool fields must coerce to bool, not int (bool is a subclass of int)."""
    sm = StaticMixtureStateV1.from_dict(
        _minimal_static_mixture_raw(reshuffle_on_repeat=1)
    )
    assert sm.reshuffle_on_repeat is True
    assert isinstance(sm.reshuffle_on_repeat, bool)


def test_from_dict_reports_all_structural_errors_at_once():
    """A dict with multiple bad fields surfaces every error in one ValueError."""
    import pytest

    raw = {
        "version": 1,
        "world": {"canonical_replicas": 1},
        # progress, lane_next_cid, lane_ws_state missing
        "last_round_id": None,
        "checkpoint_reload_count": None,  # not Optional → error
    }
    with pytest.raises(ValueError) as exc_info:
        EngineStateV1.from_dict(raw)
    msg = str(exc_info.value)
    # All structural problems should appear in the single message.
    assert "progress: missing" in msg
    assert "lane_next_cid: missing" in msg
    assert "lane_ws_state: missing" in msg
    assert "checkpoint_reload_count: must not be None" in msg


def test_post_init_reports_all_invariant_errors_at_once():
    """A constructed instance with multiple invariant violations lists them all."""
    import pytest

    with pytest.raises(ValueError) as exc_info:
        EngineStateV1(
            version=1,
            world={},  # missing canonical_replicas
            progress={"0": {"chunk_id": 1}},  # missing offset
            lane_next_cid={},
            lane_ws_state={},
            last_round_id=None,
            checkpoint_reload_count=0,
        )
    msg = str(exc_info.value)
    assert "world missing 'canonical_replicas'" in msg
    assert "missing required field 'offset'" in msg


# -- CheckpointMixin extension points --------------------------------------


def test_coerce_override_runs_in_place_of_auto():
    """An explicit ``_COERCE`` entry replaces the annotation-derived coercer."""
    from dataclasses import dataclass
    from typing import ClassVar

    from zephon.core.checkpoint._schemas import CheckpointMixin

    @dataclass(frozen=True)
    class _S(CheckpointMixin):
        _COMPONENT: ClassVar[str] = "cursor"
        _COERCE: ClassVar[dict] = {"port": lambda v: int(v) + 1000}
        port: int = 0

    s = _S.from_dict({"port": "23"})
    assert s.port == 1023


def test_optional_pep604_unwraps_and_coerces_int():
    """``int | None``: ``None`` passes through, ints coerce from str."""
    from dataclasses import dataclass
    from typing import ClassVar

    from zephon.core.checkpoint._schemas import CheckpointMixin

    @dataclass(frozen=True)
    class _S(CheckpointMixin):
        _COMPONENT: ClassVar[str] = "cursor"
        n: int | None = None

    assert _S.from_dict({"n": None}).n is None
    assert _S.from_dict({"n": "7"}).n == 7
    assert _S.from_dict({}).n is None


def test_subclass_without_component_fails_at_definition():
    """Forgetting ``_COMPONENT`` surfaces at class-creation, not at load()."""
    import pytest

    from zephon.core.checkpoint._schemas import CheckpointMixin

    with pytest.raises(TypeError, match="must set _COMPONENT"):

        class _BadSchema(CheckpointMixin):
            pass


def test_load_protects_top_level_from_migration_mutation(monkeypatch):
    """A migration that mutates the top-level dict it receives must not affect *raw*."""
    from dataclasses import dataclass
    from typing import ClassVar

    from zephon.core.checkpoint._migrations import _SCHEMAS
    from zephon.core.checkpoint._schemas import CheckpointMixin

    @dataclass(frozen=True)
    class _CursorStateV2(CheckpointMixin):
        _COMPONENT: ClassVar[str] = "cursor"
        version: int = 2
        position: int = 0
        epoch: int = 0
        injected: str = ""

    monkeypatch.setitem(CURRENT_VERSIONS, "cursor", 2)
    monkeypatch.setitem(_MIGRATIONS, "cursor", {})
    monkeypatch.setitem(_SCHEMAS, "cursor", {1: CursorStateV1, 2: _CursorStateV2})

    def _v1_to_v2(v1: CursorStateV1) -> dict:
        d = v1.to_dict()
        d["version"] = 2
        d["injected"] = "added"
        return d

    register_migration("cursor", from_version=1, fn=_v1_to_v2)

    raw = {"version": 1, "position": 7}
    snapshot = dict(raw)
    _CursorStateV2.load(raw)
    assert raw == snapshot  # migration must not mutate caller's dict
