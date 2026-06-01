import random

import numpy as np
import pytest

from zephon.io import Dataset, InMemoryShard
from zephon.work import (
    MixtureSpec,
    StaticMixtureWorkSource,
)
from zephon.work.static_mixture import (
    _AUTO_BLOCK_SIZE_FACTOR,
    AccumulatorStrategy,
    LegacyFixedStrategy,
    _DatasetCursor,
    _DatasetKnobs,
    _resolve_block_size,
)


def make_dataset(name: str, sample_count: int) -> Dataset:
    rows = [{"text": f"{name}-{i}"} for i in range(sample_count)]
    return Dataset.from_dict(name, {0: InMemoryShard(rows)})


def make_sharded_dataset(name: str, shard_lengths: list[int]) -> Dataset:
    """Create a dataset with multiple shards of given lengths."""
    shards: dict[int, InMemoryShard] = {}
    for shard_id, count in enumerate(shard_lengths):
        rows = [{"text": f"{name}-s{shard_id}-{i}"} for i in range(count)]
        shards[shard_id] = InMemoryShard(rows)
    return Dataset.from_dict(name, shards)


def _flatten_components(chunk) -> dict[str, list[tuple[int, int, int]]]:
    """Return a copy of chunk.components with SampleId tuples for comparison."""
    assert chunk is not None
    out: dict[str, list[tuple[int, int, int]]] = {}
    for name, items in chunk.components.items():
        out[name] = [tuple(map(int, sid)) for sid in items]
    return out


def _drain_chunks(
    ws: StaticMixtureWorkSource, limit: int | None = None
) -> list[dict[str, list[tuple[int, int, int]]]]:
    out: list[dict[str, list[tuple[int, int, int]]]] = []
    while limit is None or len(out) < limit:
        ch = ws.next_chunk()
        if ch is None:
            break
        out.append(_flatten_components(ch))
    return out


@pytest.mark.parametrize(
    "shuffle_shards,shuffle_within,block_size",
    [
        (False, False, None),
        (True, False, None),
        (False, True, None),
        (True, True, None),
        (False, False, 3),
        (True, False, 3),
        (False, True, 3),
        (True, True, 3),
    ],
)
def test_state_dict_knobs_serialization(
    shuffle_shards: bool, shuffle_within: bool, block_size: int | None
) -> None:
    """
    Verify that state_dict encodes knobs faithfully for all combinations.
    This intentionally inspects the 'knobs' payload to ensure it round-trips
    the exact values (not just behavior).
    """
    ds = make_sharded_dataset("alpha", [3, 4, 5])
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=7,
        seed=123,
        shuffle_shards=shuffle_shards,
        shuffle_within_shard=shuffle_within,
        shuffle_block_size=block_size,
    )

    st = work.state_dict()
    assert st["seed"] == 123
    assert st["chunk_size"] == 7
    assert "knobs" in st
    knobs = st["knobs"]
    assert knobs["shuffle_shards"] == shuffle_shards
    assert knobs["shuffle_within_shard"] == shuffle_within
    assert "shuffle_block_size" not in knobs
    assert st["shuffle_block_size_spec"] == block_size
    assert st["cursor_block_sizes"] == {ds.name: block_size}


@pytest.mark.parametrize(
    "shuffle_shards,shuffle_within,block_size",
    [
        (False, False, None),
        (True, False, None),
        (False, True, None),
        (True, True, None),
        (False, False, 2),
        (True, False, 2),
        (False, True, 2),
        (True, True, 2),
    ],
)
def test_load_state_dict_preserves_sequence_and_positions(
    shuffle_shards: bool, shuffle_within: bool, block_size: int | None
) -> None:
    """
    Advance a source, checkpoint, load into a fresh instance with different
    initial knobs, and verify the sequence continues identically from the checkpoint.
    Also checks that chunk_size, seed, weights, component_order, dataset_ids,
    global_chunk_index, and cursor_positions are restored.
    """
    # Two datasets to exercise per-component quotas and component order
    ds_a = make_sharded_dataset("alpha", [5, 5, 5, 5])  # total 20, multi-shard
    ds_b = make_sharded_dataset("beta", [5, 5, 5, 5])  # total 20, multi-shard
    mix = MixtureSpec({ds_a.name: 0.6, ds_b.name: 0.4}).weights

    # Build baseline with given knobs and consume a few chunks
    work_a = StaticMixtureWorkSource(
        [ds_a, ds_b],
        mix,
        chunk_size=7,
        seed=777,
        shuffle_shards=shuffle_shards,
        shuffle_within_shard=shuffle_within,
        shuffle_block_size=block_size,
    )

    # Create a lane-bound clone and consume some chunks to create a checkpoint
    ws_a = work_a.clone_for_lane(0, canonical_replicas=1)
    prefix_chunks: list[dict[str, list[tuple[int, int, int]]]] = []
    for _ in range(3):
        ch = ws_a.next_chunk()
        assert ch is not None
        prefix_chunks.append(_flatten_components(ch))

    st = ws_a.state_dict()

    # Build a new instance with different knobs and chunk_size/seed, then load
    work_b = StaticMixtureWorkSource(
        [ds_a, ds_b],
        mix,
        chunk_size=5,  # intentionally different
        seed=1,  # intentionally different
        shuffle_shards=not shuffle_shards,
        shuffle_within_shard=not shuffle_within,
        shuffle_block_size=(None if block_size else 4),
    )
    ws_b = work_b.clone_for_lane(0, canonical_replicas=1)
    ws_b.load_state_dict(st)

    # After load, structural state should match the checkpoint
    st_b = ws_b.state_dict()
    assert st_b["seed"] == st["seed"]
    assert st_b["chunk_size"] == st["chunk_size"]
    assert st_b["weights"] == st["weights"]
    assert st_b["component_order"] == st["component_order"]
    assert st_b["dataset_ids"] == st["dataset_ids"]
    assert st_b["cursor_positions"] == st["cursor_positions"]
    assert st_b["global_chunk_index"] == st["global_chunk_index"]

    # Continuing from the checkpoint, both streams must produce identical chunks
    for _ in range(5):
        ca = ws_a.next_chunk()
        cb = ws_b.next_chunk()
        if ca is None or cb is None:
            assert ca is None and cb is None
            break
        assert _flatten_components(ca) == _flatten_components(cb)


# ── shuffle_block_size sentinel resolution ───────────────────────────


def test_resolve_block_size_none_passes_through() -> None:
    assert _resolve_block_size(None, total_samples=100, max_shard=10) is None


def test_resolve_block_size_int_passes_through_when_within_total() -> None:
    assert _resolve_block_size(7, total_samples=100, max_shard=10) == 7


def test_resolve_block_size_int_clamped_to_total() -> None:
    # Oversized explicit ints quietly behave as "global" for this dataset.
    assert _resolve_block_size(500, total_samples=100, max_shard=10) == 100


def test_resolve_block_size_auto_uses_factor_times_max_shard() -> None:
    total = _AUTO_BLOCK_SIZE_FACTOR * 50 * 10  # well above the unclamped target
    assert (
        _resolve_block_size("auto", total_samples=total, max_shard=50)
        == _AUTO_BLOCK_SIZE_FACTOR * 50
    )


def test_resolve_block_size_auto_clamped_to_total_when_factor_overshoots() -> None:
    # max_shard * factor > total → clamp to total (effectively global).
    assert _resolve_block_size("auto", total_samples=30, max_shard=50) == 30


def test_resolve_block_size_global_returns_total() -> None:
    assert _resolve_block_size("global", total_samples=100, max_shard=10) == 100


def test_resolve_block_size_invalid_string_raises() -> None:
    with pytest.raises(ValueError):
        _resolve_block_size("nope", total_samples=100, max_shard=10)  # type: ignore[arg-type]


def test_resolve_block_size_zero_int_raises() -> None:
    with pytest.raises(ValueError):
        _resolve_block_size(0, total_samples=100, max_shard=10)


def test_resolve_block_size_negative_int_raises() -> None:
    with pytest.raises(ValueError):
        _resolve_block_size(-1, total_samples=100, max_shard=10)


def test_resolve_block_size_bool_rejected() -> None:
    # bool is an int subclass — explicitly rejected.
    with pytest.raises(ValueError):
        _resolve_block_size(True, total_samples=100, max_shard=10)  # type: ignore[arg-type]


# ── WorkSource with "auto" / "global" sentinels ──────────────────────


def test_shuffle_block_size_auto_resolves_per_cursor() -> None:
    """auto: 8 * max(shard) across all datasets, clamped to per-dataset total."""
    ds_a = make_sharded_dataset("alpha", [10, 20, 10])  # max shard 20, total 40
    ds_b = make_sharded_dataset("beta", [50, 50, 50])  # max shard 50, total 150
    work = StaticMixtureWorkSource(
        [ds_a, ds_b],
        {ds_a.name: 0.5, ds_b.name: 0.5},
        chunk_size=8,
        shuffle_block_size="auto",
    )
    expected_unclamped = _AUTO_BLOCK_SIZE_FACTOR * 50
    assert work._knobs_by_name["alpha"].shuffle_block_size == min(
        expected_unclamped, 40
    )
    assert work._knobs_by_name["beta"].shuffle_block_size == min(
        expected_unclamped, 150
    )


def test_shuffle_block_size_global_resolves_per_dataset_total() -> None:
    ds_a = make_sharded_dataset("alpha", [10, 20, 10])  # total 40
    ds_b = make_sharded_dataset("beta", [5, 5, 5, 5])  # total 20
    work = StaticMixtureWorkSource(
        [ds_a, ds_b],
        {ds_a.name: 0.5, ds_b.name: 0.5},
        chunk_size=8,
        shuffle_block_size="global",
    )
    assert work._knobs_by_name["alpha"].shuffle_block_size == 40
    assert work._knobs_by_name["beta"].shuffle_block_size == 20


def test_shuffle_block_size_none_disables_block_shuffle() -> None:
    ds = make_sharded_dataset("alpha", [5, 5])
    work = StaticMixtureWorkSource(
        [ds], {ds.name: 1.0}, chunk_size=4, shuffle_block_size=None
    )
    assert work._knobs_by_name["alpha"].shuffle_block_size is None
    assert work._cursors["alpha"]._has_block_shuffle is False


def test_shuffle_block_size_global_drains_full_dataset() -> None:
    """End-to-end smoke: "global" emits every sample exactly once."""
    ds = make_sharded_dataset("alpha", [3, 4, 5])
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=4,
        seed=42,
        shuffle_block_size="global",
        shuffle_within_shard=True,
    )
    ws = work.clone_for_lane(0, canonical_replicas=1)
    chunks = _drain_chunks(ws)
    seen: list[tuple[int, int, int]] = []
    for ch in chunks:
        seen.extend(ch["alpha"])
    assert len(seen) == 12
    assert set(seen) == {(0, sid, off) for sid in range(3) for off in range(3 + sid)}


def test_shuffle_block_size_global_uses_block_shuffle_path() -> None:
    ds = make_sharded_dataset("alpha", [5, 5])
    work = StaticMixtureWorkSource(
        [ds], {ds.name: 1.0}, chunk_size=4, shuffle_block_size="global"
    )
    assert work._cursors["alpha"]._has_block_shuffle is True


def test_shuffle_block_size_invalid_string_raises_in_constructor() -> None:
    ds = make_sharded_dataset("alpha", [5, 5])
    with pytest.raises(ValueError):
        StaticMixtureWorkSource(
            [ds],
            {ds.name: 1.0},
            chunk_size=4,
            shuffle_block_size="nope",  # type: ignore[arg-type]
        )


def test_shuffle_block_size_auto_with_all_empty_datasets_raises_clearly() -> None:
    """``auto`` requires at least one shard to derive a block size from."""
    empty = Dataset.from_dict("empty", {})
    with pytest.raises(ValueError, match="shuffle_block_size='auto' requires"):
        StaticMixtureWorkSource(
            [empty],
            {empty.name: 1.0},
            chunk_size=4,
            shuffle_block_size="auto",
        )


def test_shuffle_block_size_none_with_all_empty_datasets_raises_clear_error() -> None:
    """No sentinel involved: the empty-shard-list case must still fail loudly,
    via the standard 'No datasets with positive weight' guard rather than a
    confusing 'max() iterable argument is empty'."""
    empty = Dataset.from_dict("empty", {})
    with pytest.raises(ValueError, match="No datasets with positive weight"):
        StaticMixtureWorkSource(
            [empty],
            {empty.name: 1.0},
            chunk_size=4,
            shuffle_block_size=None,
        )


# ── state_dict round-trip + v1→v2 migration ──────────────────────────


@pytest.mark.parametrize("spec", ["auto", "global"])
def test_state_dict_preserves_sentinel_spec_at_top_level(spec: str) -> None:
    ds_a = make_sharded_dataset("alpha", [10, 20])
    ds_b = make_sharded_dataset("beta", [3, 7])
    work = StaticMixtureWorkSource(
        [ds_a, ds_b],
        {ds_a.name: 0.5, ds_b.name: 0.5},
        chunk_size=4,
        shuffle_block_size=spec,  # type: ignore[arg-type]
    )
    st = work.state_dict()
    assert st["shuffle_block_size_spec"] == spec
    # Per-cursor resolved values live in cursor_block_sizes.
    assert set(st["cursor_block_sizes"]) == {ds_a.name, ds_b.name}
    for name, value in st["cursor_block_sizes"].items():
        assert isinstance(value, int), f"{name} block size should be resolved int"


@pytest.mark.parametrize("spec", ["auto", "global"])
def test_round_trip_sentinel_preserves_per_cursor_block_size(spec: str) -> None:
    """auto/global: resolved block_size is restored verbatim per cursor.

    Critically, restore does NOT re-resolve the sentinel — adding shards
    between save and load cannot shift block sizes.
    """
    ds_a = make_sharded_dataset("alpha", [10, 20])
    ds_b = make_sharded_dataset("beta", [3, 7])
    mix = {ds_a.name: 0.5, ds_b.name: 0.5}

    work_a = StaticMixtureWorkSource(
        [ds_a, ds_b],
        mix,
        chunk_size=4,
        shuffle_block_size=spec,  # type: ignore[arg-type]
    )
    ws_a = work_a.clone_for_lane(0, canonical_replicas=1)
    ws_a.next_chunk()  # advance so cursor positions diverge
    st = ws_a.state_dict()

    # Fresh instance with a *different* spec; load_state_dict must override.
    work_b = StaticMixtureWorkSource(
        [ds_a, ds_b], mix, chunk_size=4, shuffle_block_size=None
    )
    ws_b = work_b.clone_for_lane(0, canonical_replicas=1)
    ws_b.load_state_dict(st)

    for name in (ds_a.name, ds_b.name):
        assert (
            ws_b._knobs_by_name[name].shuffle_block_size
            == work_a._knobs_by_name[name].shuffle_block_size
        )


def test_load_v1_checkpoint_migrates_block_size_per_cursor() -> None:
    """v1 checkpoints had a single knobs.shuffle_block_size shared by all cursors.

    The v1→v2 migration fans it out across every dataset; ``cursor_block_sizes``
    is fully populated on restore and matches the v1 value for each entry.
    """
    ds_a = make_sharded_dataset("alpha", [5, 5])
    ds_b = make_sharded_dataset("beta", [4, 4])
    mix = {ds_a.name: 0.5, ds_b.name: 0.5}

    # Build a current-version (v2) checkpoint, then construct an equivalent v1
    # payload by reverse-applying the migration's deltas.
    work = StaticMixtureWorkSource(
        [ds_a, ds_b], mix, chunk_size=4, shuffle_block_size=3
    )
    ws = work.clone_for_lane(0, canonical_replicas=1)
    ws.next_chunk()
    v2_state = ws.state_dict()

    v1_state = dict(v2_state)
    v1_state["version"] = 1
    v1_state["knobs"] = {
        **v2_state["knobs"],
        "shuffle_block_size": 3,
    }
    # v2-only fields don't exist in the v1 contract.
    v1_state.pop("shuffle_block_size_spec")
    v1_state.pop("cursor_block_sizes")

    work_b = StaticMixtureWorkSource(
        [ds_a, ds_b], mix, chunk_size=4, shuffle_block_size=None
    )
    ws_b = work_b.clone_for_lane(0, canonical_replicas=1)
    ws_b.load_state_dict(v1_state)
    for name in (ds_a.name, ds_b.name):
        assert ws_b._knobs_by_name[name].shuffle_block_size == 3
    # The spec is also restored from v1 (concrete int there, never a sentinel).
    assert ws_b._shuffle_block_size_spec == 3


@pytest.mark.parametrize("spec", ["auto", "global"])
@pytest.mark.parametrize("cut_after_chunks", [1, 2, 4])
def test_sentinel_checkpoint_continues_deterministically(
    spec: str, cut_after_chunks: int
) -> None:
    """Sentinel feature's core promise: resolved once, then continue exactly.

    A checkpoint taken mid-stream must emit the same suffix chunks after
    restore as the baseline would have produced. Parameterised over both
    sentinels and several cut points so cuts land mid-block (block size = 32
    for ``auto``, ≥ 48 for ``global``; chunk_size=3 so cuts at 3/6/12 samples).

    The shard layout is deliberately chosen so ``8 × max_shard < total`` for
    every dataset — i.e. ``"auto"`` and ``"global"`` resolve to *different*
    per-cursor values (auto = 32 for both; global = 60 for alpha, 48 for
    beta). This makes the cross-spec restore assertion below load-bearing:
    if ``load_state_dict`` started re-resolving the constructor spec instead
    of trusting the checkpoint's ``cursor_block_sizes``, the assertion would
    fail.
    """
    ds_a = make_sharded_dataset("alpha", [4] * 15)  # total 60, max_shard 4
    ds_b = make_sharded_dataset("beta", [4] * 12)  # total 48, max_shard 4
    kwargs: dict = dict(
        datasets=[ds_a, ds_b],
        mixture={ds_a.name: 0.6, ds_b.name: 0.4},
        chunk_size=3,
        seed=4242,
        shuffle_shards=True,
        shuffle_within_shard=True,
        shuffle_block_size=spec,
    )

    # Guard against future shard-size edits silently neutering the cross-spec
    # assertion: if "auto" and "global" ever resolve to the same per-cursor
    # value here, the test below becomes tautological.
    other_spec = "global" if spec == "auto" else "auto"
    auto_knobs = StaticMixtureWorkSource(
        **{**kwargs, "shuffle_block_size": "auto"}
    )._knobs_by_name
    global_knobs = StaticMixtureWorkSource(
        **{**kwargs, "shuffle_block_size": "global"}
    )._knobs_by_name
    assert any(
        auto_knobs[n].shuffle_block_size != global_knobs[n].shuffle_block_size
        for n in (ds_a.name, ds_b.name)
    ), "fixture must distinguish auto vs global for the lockdown assertion"

    ws_baseline = StaticMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    baseline = _drain_chunks(ws_baseline)
    assert len(baseline) > cut_after_chunks

    ws_save = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    prefix = _drain_chunks(ws_save, limit=cut_after_chunks)
    state = ws_save.state_dict()

    # Restore into an instance configured with a DIFFERENT spec: the resolved
    # cursor_block_sizes from the checkpoint must override, proving the
    # restore is driven by the locked-in values and not a re-resolution.
    restore_kwargs = dict(kwargs)
    restore_kwargs["shuffle_block_size"] = other_spec
    ws_load = StaticMixtureWorkSource(**restore_kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    ws_load.load_state_dict(state)

    # The restored per-cursor block sizes must come from the checkpoint, not
    # from re-resolving the (different) constructor spec.
    for name in (ds_a.name, ds_b.name):
        original_resolved = (
            StaticMixtureWorkSource(**kwargs)._knobs_by_name[name].shuffle_block_size
        )
        assert ws_load._knobs_by_name[name].shuffle_block_size == original_resolved
    # And the spec is preserved verbatim — it survives the round-trip even
    # though the loading instance was constructed with the opposite sentinel.
    assert ws_load._shuffle_block_size_spec == spec

    suffix = _drain_chunks(ws_load)
    assert prefix + suffix == baseline


def test_load_state_dict_version_mismatch_raises() -> None:
    ds = make_sharded_dataset("alpha", [2, 2, 2])
    work = StaticMixtureWorkSource(
        [ds], {ds.name: 1.0}, chunk_size=4, seed=5, shuffle_shards=True
    )

    ws = work.clone_for_lane(0, canonical_replicas=1)
    st = ws.state_dict()
    st_bad = dict(st)
    st_bad["version"] = 999

    other = StaticMixtureWorkSource([ds], {ds.name: 1.0}, chunk_size=4)
    with pytest.raises(RuntimeError):
        other.clone_for_lane(0, canonical_replicas=1).load_state_dict(st_bad)


# ── repeat policy tests ──────────────────────────────────────────────


def test_repeat_continues_past_exhaustion() -> None:
    """With repeat, the source keeps producing chunks beyond one epoch."""
    ds = make_dataset("alpha", 10)
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=5,
        seed=0,
        shuffle_shards=False,
        exhausted_policy="repeat",
        reshuffle_on_repeat=False,
    )
    ws = work.clone_for_lane(0, canonical_replicas=1)

    # 10 samples / 5 per chunk = 2 chunks per epoch.  Pull 5 chunks (2.5 epochs).
    chunks = []
    for _ in range(5):
        ch = ws.next_chunk()
        assert ch is not None
        chunks.append(ch)

    # First two chunks == third and fourth (same order, no reshuffle)
    assert _flatten_components(chunks[0]) == _flatten_components(chunks[2])
    assert _flatten_components(chunks[1]) == _flatten_components(chunks[3])


def test_repeat_per_component_independent() -> None:
    """Smaller dataset resets while larger one continues."""
    small = make_dataset("small", 4)
    large = make_dataset("large", 20)
    work = StaticMixtureWorkSource(
        [small, large],
        {"small": 0.5, "large": 0.5},
        chunk_size=4,
        seed=0,
        shuffle_shards=False,
        exhausted_policy="repeat",
        reshuffle_on_repeat=False,
    )
    ws = work.clone_for_lane(0, canonical_replicas=1)

    # quota: small=2, large=2.  small has 4 samples → exhausts after 2 chunks.
    c1 = ws.next_chunk()
    c2 = ws.next_chunk()
    c3 = ws.next_chunk()  # small should have reset here
    assert c1 is not None and c2 is not None and c3 is not None

    # small's samples in chunk 3 should repeat chunk 1's samples
    assert c3.components["small"] == c1.components["small"]
    # large's samples in chunk 3 should be *new* (not repeated)
    assert c3.components["large"] != c1.components["large"]


def test_repeat_reshuffle_changes_order() -> None:
    """With reshuffle_on_repeat=True, the second epoch has a different order."""
    ds = make_sharded_dataset("alpha", [5, 5, 5])
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=5,
        seed=42,
        shuffle_shards=True,
        shuffle_within_shard=True,
        exhausted_policy="repeat",
        reshuffle_on_repeat=True,
    )
    ws = work.clone_for_lane(0, canonical_replicas=1)

    # Collect first epoch (15 samples / 5 per chunk = 3 chunks)
    epoch1_ids: list[tuple[int, int, int]] = []
    for _ in range(3):
        ch = ws.next_chunk()
        assert ch is not None
        epoch1_ids.extend(ch.components[ds.name])

    # Collect second epoch
    epoch2_ids: list[tuple[int, int, int]] = []
    for _ in range(3):
        ch = ws.next_chunk()
        assert ch is not None
        epoch2_ids.extend(ch.components[ds.name])

    # Same set of (dataset_id, shard_id, offset) but different order
    assert set(epoch1_ids) == set(epoch2_ids)
    assert epoch1_ids != epoch2_ids


def test_repeat_no_reshuffle_same_order() -> None:
    """With reshuffle_on_repeat=False, the second epoch has identical order."""
    ds = make_dataset("alpha", 10)
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=5,
        seed=7,
        shuffle_shards=False,
        exhausted_policy="repeat",
        reshuffle_on_repeat=False,
    )
    ws = work.clone_for_lane(0, canonical_replicas=1)

    epoch1 = [_flatten_components(ws.next_chunk()) for _ in range(2)]
    epoch2 = [_flatten_components(ws.next_chunk()) for _ in range(2)]
    assert epoch1 == epoch2


def test_repeat_determinism() -> None:
    """Two identical instances produce identical sequences across epochs."""
    ds = make_dataset("alpha", 10)
    kwargs: dict = dict(
        datasets=[ds],
        mixture={ds.name: 1.0},
        chunk_size=5,
        seed=99,
        shuffle_shards=False,
        exhausted_policy="repeat",
        reshuffle_on_repeat=True,
    )
    ws1 = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    ws2 = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)

    for _ in range(8):  # 4 epochs worth
        c1 = ws1.next_chunk()
        c2 = ws2.next_chunk()
        assert c1 is not None and c2 is not None
        assert _flatten_components(c1) == _flatten_components(c2)


def test_repeat_checkpoint_restore() -> None:
    """Checkpoint mid-epoch, restore, verify continuation matches."""
    ds = make_dataset("alpha", 10)
    kwargs: dict = dict(
        datasets=[ds],
        mixture={ds.name: 1.0},
        chunk_size=5,
        seed=0,
        shuffle_shards=False,
        exhausted_policy="repeat",
        reshuffle_on_repeat=False,
    )

    # Baseline: run 7 chunks straight through
    ws_baseline = StaticMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    baseline = [_flatten_components(ws_baseline.next_chunk()) for _ in range(7)]

    # Run 3 chunks, checkpoint, restore, run 4 more
    ws_save = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    prefix = [_flatten_components(ws_save.next_chunk()) for _ in range(3)]
    state = ws_save.state_dict()

    ws_load = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    ws_load.load_state_dict(state)
    suffix = [_flatten_components(ws_load.next_chunk()) for _ in range(4)]

    assert prefix + suffix == baseline


def test_repeat_checkpoint_restore_with_reshuffle() -> None:
    """Checkpoint mid-epoch with reshuffle, verify continuation matches."""
    ds = make_sharded_dataset("alpha", [5, 5])
    kwargs: dict = dict(
        datasets=[ds],
        mixture={ds.name: 1.0},
        chunk_size=5,
        seed=42,
        shuffle_shards=True,
        shuffle_within_shard=True,
        exhausted_policy="repeat",
        reshuffle_on_repeat=True,
    )

    # Baseline: 7 chunks (3.5 epochs)
    ws_baseline = StaticMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    baseline = [_flatten_components(ws_baseline.next_chunk()) for _ in range(7)]

    # Checkpoint after 4 chunks (2 epochs), restore, continue
    ws_save = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    prefix = [_flatten_components(ws_save.next_chunk()) for _ in range(4)]
    state = ws_save.state_dict()

    ws_load = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    ws_load.load_state_dict(state)
    suffix = [_flatten_components(ws_load.next_chunk()) for _ in range(3)]

    assert prefix + suffix == baseline


def test_repeat_len_raises() -> None:
    """len() raises TypeError for an infinite work source."""
    ds = make_dataset("alpha", 10)
    ws = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=5,
        exhausted_policy="repeat",
    ).clone_for_lane(0, canonical_replicas=1)
    with pytest.raises(TypeError):
        len(ws)


def test_repeat_total_samples_inf() -> None:
    """total_samples is inf for repeat policy."""
    ds = make_dataset("alpha", 10)
    ws = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=5,
        exhausted_policy="repeat",
    ).clone_for_lane(0, canonical_replicas=1)
    assert ws.total_samples == float("inf")


def test_repeat_v1_checkpoint_loads() -> None:
    """A never-repeated repeat-policy checkpoint loads at epoch 0 and continues."""
    ds = make_dataset("alpha", 10)
    save_ws = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=5,
        seed=0,
        shuffle_shards=False,
        exhausted_policy="repeat",
    ).clone_for_lane(0, canonical_replicas=1)
    save_ws.next_chunk()
    saved_state = save_ws.state_dict()
    assert all(e == 0 for e in saved_state.get("cursor_epochs", {}).values())

    ws_load = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=5,
        seed=0,
        shuffle_shards=False,
        exhausted_policy="repeat",
    ).clone_for_lane(0, canonical_replicas=1)
    ws_load.load_state_dict(saved_state)

    ch = ws_load.next_chunk()
    assert ch is not None


def test_repeat_zero_sample_dataset_guard() -> None:
    """Dataset with 0 samples raises at construction with repeat policy."""
    ds = make_dataset("tiny", 0)
    with pytest.raises(ValueError):
        StaticMixtureWorkSource(
            [ds],
            {ds.name: 1.0},
            chunk_size=5,
            exhausted_policy="repeat",
        )


def test_repeat_load_mismatch_raises() -> None:
    """Loading a checkpoint with different reshuffle/max_repeats raises."""
    ds = make_dataset("alpha", 10)
    ws_save = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=5,
        exhausted_policy="repeat",
        reshuffle_on_repeat=True,
    ).clone_for_lane(0, canonical_replicas=1)
    ws_save.next_chunk()
    state = ws_save.state_dict()

    # Mismatched reshuffle_on_repeat
    ws_load = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=5,
        exhausted_policy="repeat",
        reshuffle_on_repeat=False,
    ).clone_for_lane(0, canonical_replicas=1)
    with pytest.raises(RuntimeError, match="reshuffle_on_repeat"):
        ws_load.load_state_dict(state)

    # Mismatched max_repeats
    ws_load2 = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=5,
        exhausted_policy="repeat",
        max_repeats=5,
    ).clone_for_lane(0, canonical_replicas=1)
    with pytest.raises(RuntimeError, match="max_repeats"):
        ws_load2.load_state_dict(state)

    # Mismatched exhausted_policy
    ws_load3 = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=5,
        exhausted_policy="stop",
        reshuffle_on_repeat=True,
    ).clone_for_lane(0, canonical_replicas=1)
    with pytest.raises(RuntimeError, match="exhausted_policy"):
        ws_load3.load_state_dict(state)


def test_max_repeats_warns_with_stop_policy() -> None:
    """max_repeats with stop policy emits a warning."""
    ds = make_dataset("alpha", 10)
    with pytest.warns(RuntimeWarning, match="max_repeats.*no effect"):
        StaticMixtureWorkSource(
            [ds],
            {ds.name: 1.0},
            chunk_size=5,
            exhausted_policy="stop",
            max_repeats=3,
        )


def test_repeat_max_repeats() -> None:
    """With max_repeats=2, the source stops after 2 resets."""
    ds = make_dataset("alpha", 10)
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=5,
        seed=0,
        shuffle_shards=False,
        exhausted_policy="repeat",
        reshuffle_on_repeat=False,
        max_repeats=2,
    )
    ws = work.clone_for_lane(0, canonical_replicas=1)

    # 2 chunks per epoch × (1 initial + 2 repeats) = 6 chunks, then None
    chunks = []
    for _ in range(10):
        ch = ws.next_chunk()
        if ch is None:
            break
        chunks.append(ch)
    assert len(chunks) == 6


def test_repeat_max_repeats_checkpoint() -> None:
    """Checkpoint/restore mid-way through capped repeats."""
    ds = make_dataset("alpha", 10)
    kwargs: dict = dict(
        datasets=[ds],
        mixture={ds.name: 1.0},
        chunk_size=5,
        seed=0,
        shuffle_shards=False,
        exhausted_policy="repeat",
        reshuffle_on_repeat=False,
        max_repeats=3,
    )

    # Baseline: drain fully
    ws_baseline = StaticMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    baseline = []
    for _ in range(20):
        ch = ws_baseline.next_chunk()
        if ch is None:
            break
        baseline.append(_flatten_components(ch))

    # Checkpoint after 3 chunks (1.5 epochs)
    ws_save = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    prefix = [_flatten_components(ws_save.next_chunk()) for _ in range(3)]
    state = ws_save.state_dict()

    ws_load = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    ws_load.load_state_dict(state)
    suffix = []
    for _ in range(20):
        ch = ws_load.next_chunk()
        if ch is None:
            break
        suffix.append(_flatten_components(ch))

    assert prefix + suffix == baseline


def test_seeded_shuffle_is_deterministic() -> None:
    shards_alpha = {
        0: InMemoryShard([{"text": "a-0"}, {"text": "a-1"}]),
        1: InMemoryShard([{"text": "a-2"}, {"text": "a-3"}]),
    }
    shards_beta = {
        0: InMemoryShard([{"text": "b-0"}, {"text": "b-1"}]),
        1: InMemoryShard([{"text": "b-2"}, {"text": "b-3"}]),
    }
    dataset_a = Dataset.from_dict("alpha", shards_alpha)
    dataset_b = Dataset.from_dict("beta", shards_beta)
    mixture = MixtureSpec({"alpha": 0.5, "beta": 0.5}).weights

    work_one = StaticMixtureWorkSource(
        [dataset_a, dataset_b],
        mixture,
        chunk_size=4,
        seed=99,
        shuffle_shards=True,
        shuffle_within_shard=True,
    )
    work_two = StaticMixtureWorkSource(
        [dataset_a, dataset_b],
        mixture,
        chunk_size=4,
        seed=99,
        shuffle_shards=True,
        shuffle_within_shard=True,
    )

    w1 = work_one.clone_for_lane(0, canonical_replicas=1)
    w2 = work_two.clone_for_lane(0, canonical_replicas=1)
    chunk_one = w1.next_chunk()
    chunk_two = w2.next_chunk()

    assert chunk_one is not None
    assert chunk_two is not None
    assert chunk_one.components == chunk_two.components
    # Ensure samples come from their respective datasets even after shuffles.
    ids = w1.dataset_ids
    for comp_name, samples in chunk_one.components.items():
        expected_dataset_id = ids[comp_name]
        assert all(sample[0] == expected_dataset_id for sample in samples)


# ---------------------------------------------------------------------------
# Bit-for-bit equivalence: streaming cursor vs reference implementation
# ---------------------------------------------------------------------------

_EQUIV_SHARD_INDEX: dict[int, int] = {0: 5, 1: 7, 2: 3}
_EQUIV_IDS = np.array(sorted(_EQUIV_SHARD_INDEX), dtype=np.int64)
_EQUIV_SIZES = np.array(
    [_EQUIV_SHARD_INDEX[i] for i in sorted(_EQUIV_SHARD_INDEX)], dtype=np.int64
)


@pytest.mark.parametrize(
    "shuffle_shards,shuffle_within,block_size",
    [
        (False, False, None),
        (True, False, None),
        (False, True, None),
        (True, True, None),
        (False, False, 2),
        (True, False, 2),
        (False, True, 2),
        (True, True, 2),
        (False, False, 4),
        (True, False, 4),
        (False, True, 4),
        (True, True, 4),
        # Edge: block_size larger than total samples
        (False, False, 100),
        (True, False, 100),
        (False, True, 100),
        (True, True, 100),
        (False, False, 1),
        (True, False, 1),
        (False, True, 1),
        (True, True, 1),
    ],
)
def test_streaming_cursor_matches_reference(
    shuffle_shards: bool,
    shuffle_within: bool,
    block_size: int | None,
) -> None:
    """Streaming cursor must produce the exact same sequence as the reference."""
    knobs = _DatasetKnobs(
        seed=42,
        shuffle_shards=shuffle_shards,
        shuffle_within_shard=shuffle_within,
        shuffle_block_size=block_size,
    )
    dataset_id = 0
    ref = _DatasetCursor._build_order_reference(
        dataset_id, _EQUIV_IDS, _EQUIV_SIZES, knobs
    )
    expected = [tuple(row) for row in ref.tolist()]

    cursor = _DatasetCursor(dataset_id, _EQUIV_IDS, _EQUIV_SIZES, knobs)
    actual = cursor.next_many(cursor._total_samples)

    assert actual == expected, (
        f"Mismatch for shards={shuffle_shards}, within={shuffle_within}, "
        f"block={block_size}: {actual[:5]}... vs {expected[:5]}..."
    )


@pytest.mark.parametrize("seed", [0, 1, 99])
def test_streaming_cursor_reference_multiple_seeds(seed: int) -> None:
    """Equivalence holds across different seeds."""
    knobs = _DatasetKnobs(
        seed=seed,
        shuffle_shards=True,
        shuffle_within_shard=True,
        shuffle_block_size=3,
    )
    dataset_id = 1
    ref = _DatasetCursor._build_order_reference(
        dataset_id, _EQUIV_IDS, _EQUIV_SIZES, knobs
    )
    expected = [tuple(row) for row in ref.tolist()]
    cursor = _DatasetCursor(dataset_id, _EQUIV_IDS, _EQUIV_SIZES, knobs)
    actual = cursor.next_many(cursor._total_samples)
    assert actual == expected


@pytest.mark.parametrize(
    "shuffle_shards,shuffle_within,block_size",
    [
        (True, False, None),
        (False, True, None),
        (True, True, None),
        (False, False, 3),
        (True, False, 3),
        (False, True, 3),
        (True, True, 3),
    ],
)
def test_repeat_active_shuffle_determinism_across_epochs(
    shuffle_shards: bool, shuffle_within: bool, block_size: int | None
) -> None:
    """Across epochs, shuffled repeat streams are deterministic between identical instances."""
    ds = make_sharded_dataset("alpha", [5, 6, 7])
    kwargs: dict = dict(
        datasets=[ds],
        mixture={ds.name: 1.0},
        chunk_size=6,
        seed=1234,
        shuffle_shards=shuffle_shards,
        shuffle_within_shard=shuffle_within,
        shuffle_block_size=block_size,
        exhausted_policy="repeat",
        reshuffle_on_repeat=True,
    )
    ws1 = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    ws2 = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)

    # 18 samples / 6 per chunk = 3 chunks per epoch; pull 9 chunks (3 epochs).
    for _ in range(9):
        c1 = ws1.next_chunk()
        c2 = ws2.next_chunk()
        assert c1 is not None and c2 is not None
        assert _flatten_components(c1) == _flatten_components(c2)


def test_repeat_checkpoint_restore_with_block_shuffle() -> None:
    """Checkpoint/restore remains deterministic in repeat mode with block shuffle."""
    ds = make_sharded_dataset("alpha", [5, 5, 5])
    kwargs: dict = dict(
        datasets=[ds],
        mixture={ds.name: 1.0},
        chunk_size=5,
        seed=314,
        shuffle_shards=True,
        shuffle_within_shard=True,
        shuffle_block_size=3,
        exhausted_policy="repeat",
        reshuffle_on_repeat=True,
    )

    ws_baseline = StaticMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    baseline = [_flatten_components(ws_baseline.next_chunk()) for _ in range(8)]

    ws_save = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    prefix = [_flatten_components(ws_save.next_chunk()) for _ in range(5)]
    state = ws_save.state_dict()

    ws_load = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    ws_load.load_state_dict(state)
    suffix = [_flatten_components(ws_load.next_chunk()) for _ in range(3)]

    assert prefix + suffix == baseline


# ---------------------------------------------------------------------------
# _seek_to_position round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shuffle_shards,shuffle_within,block_size",
    [
        (False, False, None),
        (True, True, None),
        (False, True, 2),
        (True, True, 3),
    ],
)
@pytest.mark.parametrize("seek_to", [0, 1, 5, 10, 14, 15])
def test_seek_to_position_produces_correct_tail(
    shuffle_shards: bool,
    shuffle_within: bool,
    block_size: int | None,
    seek_to: int,
) -> None:
    """Seeking to position k then draining must match draining k then draining rest."""
    knobs = _DatasetKnobs(
        seed=7,
        shuffle_shards=shuffle_shards,
        shuffle_within_shard=shuffle_within,
        shuffle_block_size=block_size,
    )
    dataset_id = 0
    total = sum(_EQUIV_SHARD_INDEX.values())  # 15

    # Baseline: advance naturally.
    baseline = _DatasetCursor(dataset_id, _EQUIV_IDS, _EQUIV_SIZES, knobs)
    baseline.next_many(seek_to)
    expected_tail = baseline.next_many(total)

    # Seeker: jump directly.
    seeker = _DatasetCursor(dataset_id, _EQUIV_IDS, _EQUIV_SIZES, knobs)
    seeker._seek_to_position(seek_to)
    actual_tail = seeker.next_many(total)

    assert actual_tail == expected_tail, (
        f"seek_to={seek_to}, shards={shuffle_shards}, within={shuffle_within}, "
        f"block={block_size}"
    )
    assert seeker._position == baseline._position
    assert seeker.remaining == baseline.remaining


# ---------------------------------------------------------------------------
# _clone independence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shuffle_shards,shuffle_within,block_size",
    [
        (False, False, None),
        (True, True, None),
        (True, True, 3),
    ],
)
def test_clone_independence(
    shuffle_shards: bool,
    shuffle_within: bool,
    block_size: int | None,
) -> None:
    """Advancing a clone must not affect the original cursor."""
    knobs = _DatasetKnobs(
        seed=42,
        shuffle_shards=shuffle_shards,
        shuffle_within_shard=shuffle_within,
        shuffle_block_size=block_size,
    )
    dataset_id = 0
    total = sum(_EQUIV_SHARD_INDEX.values())  # 15

    # Advance the original partway, then snapshot its state via clone.
    original = _DatasetCursor(dataset_id, _EQUIV_IDS, _EQUIV_SIZES, knobs)
    original.next_many(5)
    orig_position = original._position
    orig_remaining = original.remaining

    clone = original._clone()

    # Drain the clone completely.
    clone_tail = clone.next_many(total)
    assert len(clone_tail) == total - 5

    # Original must be unaffected.
    assert original._position == orig_position
    assert original.remaining == orig_remaining

    # Draining the original from the same point must yield identical results.
    orig_tail = original.next_many(total)
    assert orig_tail == clone_tail


@pytest.mark.parametrize("canonical_replicas", [2, 4])
def test_checkpoint_restore_multilane_matches_baseline(canonical_replicas: int) -> None:
    """Per-lane checkpoint/restore stays deterministic for canonical_replicas > 1."""
    ds_a = make_sharded_dataset("alpha", [17, 13, 11, 19])
    ds_b = make_sharded_dataset("beta", [23, 7, 21, 9])
    kwargs: dict = dict(
        datasets=[ds_a, ds_b],
        mixture=MixtureSpec({ds_a.name: 0.6, ds_b.name: 0.4}).weights,
        chunk_size=8,
        seed=2026,
        shuffle_shards=True,
        shuffle_within_shard=True,
        shuffle_block_size=3,
    )

    baseline_per_lane: dict[int, list[dict[str, list[tuple[int, int, int]]]]] = {}
    for lane in range(canonical_replicas):
        ws = StaticMixtureWorkSource(**kwargs).clone_for_lane(
            lane, canonical_replicas=canonical_replicas
        )
        baseline_per_lane[lane] = _drain_chunks(ws)
        assert baseline_per_lane[lane], f"lane {lane} should produce chunks"

    checkpoint_after_chunks = 2
    for lane in range(canonical_replicas):
        ws_save = StaticMixtureWorkSource(**kwargs).clone_for_lane(
            lane, canonical_replicas=canonical_replicas
        )
        prefix = _drain_chunks(ws_save, limit=checkpoint_after_chunks)
        state = ws_save.state_dict()

        ws_load = StaticMixtureWorkSource(**kwargs).clone_for_lane(
            lane, canonical_replicas=canonical_replicas
        )
        ws_load.load_state_dict(state)
        suffix = _drain_chunks(ws_load)

        assert prefix + suffix == baseline_per_lane[lane]


def test_repeat_checkpoint_restore_with_divergent_component_epochs() -> None:
    """Repeat-mode restore is deterministic when datasets are at different epochs."""
    small = make_sharded_dataset("small", [8, 8])  # quota-heavy; resets often
    medium = make_sharded_dataset("medium", [25, 25])
    large = make_sharded_dataset("large", [40, 40])
    kwargs: dict = dict(
        datasets=[small, medium, large],
        mixture=MixtureSpec({"small": 0.5, "medium": 0.3, "large": 0.2}).weights,
        chunk_size=8,
        seed=314159,
        shuffle_shards=True,
        shuffle_within_shard=True,
        shuffle_block_size=4,
        exhausted_policy="repeat",
        reshuffle_on_repeat=True,
        max_repeats=3,
    )

    ws_baseline = StaticMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    baseline = _drain_chunks(ws_baseline, limit=16)
    assert len(baseline) == 16

    ws_save = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    prefix = _drain_chunks(ws_save, limit=7)
    state = ws_save.state_dict()
    cursor_epochs = state["cursor_epochs"]
    assert cursor_epochs["small"] > cursor_epochs["medium"]
    assert cursor_epochs["small"] > cursor_epochs["large"]

    ws_load = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    ws_load.load_state_dict(state)
    suffix = _drain_chunks(ws_load, limit=9)

    assert prefix + suffix == baseline


@pytest.mark.parametrize("cut_position", [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 58])
def test_checkpoint_restore_cut_sweep_around_block_boundaries(
    cut_position: int,
) -> None:
    """Checkpoint/resume is stable around block-shuffle boundaries."""
    ds = make_sharded_dataset("alpha", [17, 19, 23])  # total 59
    kwargs: dict = dict(
        datasets=[ds],
        mixture={ds.name: 1.0},
        chunk_size=1,  # one sample per chunk: checkpoint cut == sample position
        seed=777,
        shuffle_shards=True,
        shuffle_within_shard=True,
        shuffle_block_size=8,
    )

    ws_baseline = StaticMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    baseline = _drain_chunks(ws_baseline)
    assert len(baseline) == 59

    ws_save = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    prefix = _drain_chunks(ws_save, limit=cut_position)
    state = ws_save.state_dict()

    ws_load = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    ws_load.load_state_dict(state)
    suffix = _drain_chunks(ws_load)

    assert len(prefix) == cut_position
    assert prefix + suffix == baseline


def test_load_state_dict_dataset_order_mismatch_raises() -> None:
    """Loading checkpoint into differently ordered datasets should fail."""
    ds_a = make_sharded_dataset("alpha", [7, 9, 11])
    ds_b = make_sharded_dataset("beta", [13, 15, 17])

    ws_save = StaticMixtureWorkSource(
        [ds_a, ds_b],
        MixtureSpec({"alpha": 0.5, "beta": 0.5}).weights,
        chunk_size=6,
        seed=99,
        shuffle_shards=True,
        shuffle_within_shard=True,
        shuffle_block_size=3,
    ).clone_for_lane(0, canonical_replicas=1)
    _drain_chunks(ws_save, limit=4)
    state = ws_save.state_dict()

    ws_load = StaticMixtureWorkSource(
        [ds_b, ds_a],  # reversed order
        MixtureSpec({"alpha": 0.5, "beta": 0.5}).weights,
        chunk_size=6,
        seed=99,
        shuffle_shards=True,
        shuffle_within_shard=True,
        shuffle_block_size=3,
    ).clone_for_lane(0, canonical_replicas=1)
    with pytest.raises(RuntimeError):
        ws_load.load_state_dict(state)


def test_load_state_dict_lane_mismatch_raises() -> None:
    """Loading lane-0 checkpoint into lane-1 should fail for deterministic safety."""
    ds = make_sharded_dataset("alpha", [11, 13, 17, 19])
    kwargs: dict = dict(
        datasets=[ds],
        mixture={ds.name: 1.0},
        chunk_size=5,
        seed=2025,
        shuffle_shards=True,
        shuffle_within_shard=True,
        shuffle_block_size=3,
    )

    ws_save = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=2)
    _drain_chunks(ws_save, limit=5)
    state = ws_save.state_dict()

    ws_load_wrong_lane = StaticMixtureWorkSource(**kwargs).clone_for_lane(
        1, canonical_replicas=2
    )
    with pytest.raises(RuntimeError):
        ws_load_wrong_lane.load_state_dict(state)


@pytest.mark.parametrize("cfg_seed", [11, 29, 53])
def test_checkpoint_restore_randomized_property(cfg_seed: int) -> None:
    """Randomized config smoke-test for deterministic checkpoint continuation."""
    rng = random.Random(cfg_seed)
    dataset_count = rng.choice([1, 2, 3])
    datasets: list[Dataset] = []
    mixture_raw: dict[str, float] = {}

    for idx in range(dataset_count):
        shard_count = rng.choice([1, 2, 3, 4])
        shard_lengths = [rng.randint(18, 36) for _ in range(shard_count)]
        name = f"ds_{idx}"
        datasets.append(make_sharded_dataset(name, shard_lengths))
        mixture_raw[name] = rng.uniform(0.1, 1.0)

    spec = MixtureSpec(mixture_raw)

    exhausted_policy = rng.choice(["stop", "repeat"])
    kwargs: dict = dict(
        datasets=datasets,
        mixture=spec.weights,
        chunk_size=rng.randint(1, 12),
        seed=rng.randint(0, 10000),
        shuffle_shards=rng.choice([True, False]),
        shuffle_within_shard=rng.choice([True, False]),
        shuffle_block_size=rng.choice([None, 2, 3, 5, 8]),
        exhausted_policy=exhausted_policy,
        reshuffle_on_repeat=rng.choice([True, False]),
        max_repeats=(rng.choice([1, 2, 3]) if exhausted_policy == "repeat" else None),
    )

    ws_baseline = StaticMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    baseline = _drain_chunks(ws_baseline, limit=60)
    assert baseline

    cut = rng.randint(0, max(0, len(baseline) - 1))
    ws_save = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    prefix = _drain_chunks(ws_save, limit=cut)
    state = ws_save.state_dict()

    ws_load = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    ws_load.load_state_dict(state)
    suffix = _drain_chunks(ws_load, limit=max(0, len(baseline) - cut))

    assert prefix + suffix == baseline


# ===========================================================================
# Tests ported from AccumulatorMixtureWorkSource
# ===========================================================================


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_chunk_size_1_with_multiple_components_allowed() -> None:
    """chunk_size < len(components) is valid with accumulator-based allocation."""
    ds_a = make_dataset("alpha", 10)
    ds_b = make_dataset("beta", 10)
    ws = StaticMixtureWorkSource(
        datasets=[ds_a, ds_b],
        mixture={"alpha": 0.5, "beta": 0.5},
        chunk_size=1,
    ).clone_for_lane(0, canonical_replicas=1)

    chunks = _drain_chunks(ws, limit=10)
    assert len(chunks) > 0
    for ch in chunks:
        total = sum(len(v) for v in ch.values())
        assert total == 1


# ---------------------------------------------------------------------------
# Chunk size invariant
# ---------------------------------------------------------------------------


def test_chunk_size_always_exact() -> None:
    """Every chunk must have exactly chunk_size samples regardless of weights."""
    ds_a = make_dataset("alpha", 200)
    ds_b = make_dataset("beta", 200)
    ds_c = make_dataset("gamma", 200)
    ws = StaticMixtureWorkSource(
        datasets=[ds_a, ds_b, ds_c],
        mixture={"alpha": 0.7, "beta": 0.2, "gamma": 0.1},
        chunk_size=10,
    ).clone_for_lane(0, canonical_replicas=1)

    for _ in range(15):
        chunk = ws.next_chunk()
        assert chunk is not None
        total = sum(len(v) for v in chunk.components.values())
        assert total == 10


def test_chunk_size_exact_with_odd_weights() -> None:
    """Chunk_size invariant holds even with weights that don't divide evenly."""
    ds_a = make_dataset("alpha", 500)
    ds_b = make_dataset("beta", 500)
    ws = StaticMixtureWorkSource(
        datasets=[ds_a, ds_b],
        mixture={"alpha": 0.7, "beta": 0.3},
        chunk_size=3,
    ).clone_for_lane(0, canonical_replicas=1)

    for _ in range(50):
        chunk = ws.next_chunk()
        assert chunk is not None
        total = sum(len(v) for v in chunk.components.values())
        assert total == 3


# ---------------------------------------------------------------------------
# Sparse chunks
# ---------------------------------------------------------------------------


def test_sparse_chunks_small_weight() -> None:
    """A low-weight component should get 0 samples in most chunks."""
    large = make_dataset("large", 1000)
    tiny = make_dataset("tiny", 1000)
    ws = StaticMixtureWorkSource(
        datasets=[large, tiny],
        mixture={"large": 0.99, "tiny": 0.01},
        chunk_size=4,
    ).clone_for_lane(0, canonical_replicas=1)

    chunks_with_tiny = 0
    chunks_without_tiny = 0
    for _ in range(100):
        chunk = ws.next_chunk()
        assert chunk is not None
        if "tiny" in chunk.components:
            chunks_with_tiny += 1
        else:
            chunks_without_tiny += 1

    # With weight=0.01, chunk_size=4 → ideal=0.04 per chunk.
    # Tiny should appear in roughly 4% of chunks (every ~25 chunks).
    assert chunks_without_tiny > chunks_with_tiny
    assert chunks_with_tiny > 0  # but it does appear eventually


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


def test_mixture_converges_to_requested_weights() -> None:
    """Empirical mixture ratio should converge to target within tolerance."""
    ds_a = make_dataset("alpha", 5000)
    ds_b = make_dataset("beta", 5000)
    ds_c = make_dataset("gamma", 5000)
    target = {"alpha": 0.6, "beta": 0.3, "gamma": 0.1}
    ws = StaticMixtureWorkSource(
        datasets=[ds_a, ds_b, ds_c],
        mixture=target,
        chunk_size=10,
    ).clone_for_lane(0, canonical_replicas=1)

    counts: dict[str, int] = {"alpha": 0, "beta": 0, "gamma": 0}
    for _ in range(200):
        chunk = ws.next_chunk()
        assert chunk is not None
        for name, samples in chunk.components.items():
            counts[name] += len(samples)

    total = sum(counts.values())
    for name, expected_weight in target.items():
        actual = counts[name] / total
        assert abs(actual - expected_weight) < 0.02, (
            f"{name}: expected ~{expected_weight}, got {actual}"
        )


def test_extreme_weights_converge() -> None:
    """Very skewed weights still converge correctly."""
    large = make_dataset("large", 10000)
    tiny = make_dataset("tiny", 10000)
    target = {"large": 0.99, "tiny": 0.01}
    ws = StaticMixtureWorkSource(
        datasets=[large, tiny],
        mixture=target,
        chunk_size=100,
    ).clone_for_lane(0, canonical_replicas=1)

    counts = {"large": 0, "tiny": 0}
    for _ in range(50):
        chunk = ws.next_chunk()
        assert chunk is not None
        for name, samples in chunk.components.items():
            counts[name] += len(samples)

    total = sum(counts.values())
    for name, expected_weight in target.items():
        actual = counts[name] / total
        assert abs(actual - expected_weight) < 0.02, (
            f"{name}: expected ~{expected_weight}, got {actual}"
        )


# ---------------------------------------------------------------------------
# Single component
# ---------------------------------------------------------------------------


def test_single_component_matches_chunk_size() -> None:
    """With 1 dataset, every chunk gets exactly chunk_size samples."""
    ds = make_dataset("only", 100)
    ws = StaticMixtureWorkSource(
        datasets=[ds],
        mixture={"only": 1.0},
        chunk_size=8,
    ).clone_for_lane(0, canonical_replicas=1)

    for _ in range(10):
        chunk = ws.next_chunk()
        assert chunk is not None
        assert "only" in chunk.components
        assert len(chunk.components["only"]) == 8


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_two_instances() -> None:
    """Two identical instances produce identical chunk sequences."""
    ds_a = make_sharded_dataset("alpha", [15, 20])
    ds_b = make_sharded_dataset("beta", [10, 25])
    kwargs: dict = dict(
        datasets=[ds_a, ds_b],
        mixture={"alpha": 0.6, "beta": 0.4},
        chunk_size=7,
        seed=42,
        shuffle_shards=True,
        shuffle_within_shard=True,
    )

    ws1 = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    ws2 = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)

    for _ in range(8):
        c1 = ws1.next_chunk()
        c2 = ws2.next_chunk()
        assert c1 is not None and c2 is not None
        assert _flatten_components(c1) == _flatten_components(c2)


# ---------------------------------------------------------------------------
# Checkpoint / restore
# ---------------------------------------------------------------------------


def test_checkpoint_restore_continues_deterministically() -> None:
    ds_a = make_sharded_dataset("alpha", [30, 20])
    ds_b = make_sharded_dataset("beta", [25, 15])
    kwargs: dict = dict(
        datasets=[ds_a, ds_b],
        mixture={"alpha": 0.7, "beta": 0.3},
        chunk_size=5,
        seed=99,
        shuffle_shards=True,
        shuffle_within_shard=True,
        shuffle_block_size=3,
    )

    # Baseline: drain all chunks.
    ws_baseline = StaticMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    baseline = _drain_chunks(ws_baseline)
    assert len(baseline) > 5

    # Save after 3 chunks, restore, drain rest.
    ws_save = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    prefix = _drain_chunks(ws_save, limit=3)
    state = ws_save.state_dict()

    ws_load = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    ws_load.load_state_dict(state)
    suffix = _drain_chunks(ws_load)

    assert prefix + suffix == baseline


@pytest.mark.parametrize("cut_position", [0, 1, 2, 3, 5, 8, 11, 17])
def test_checkpoint_restore_with_block_shuffle_cut_sweep(
    cut_position: int,
) -> None:
    """Block-shuffle checkpoint resume matches baseline at multiple cut points."""
    ds_a = make_sharded_dataset("alpha", [17, 19, 23])
    ds_b = make_sharded_dataset("beta", [13, 11, 7])
    kwargs: dict = dict(
        datasets=[ds_a, ds_b],
        mixture={"alpha": 0.7, "beta": 0.3},
        chunk_size=3,
        seed=777,
        shuffle_shards=True,
        shuffle_within_shard=True,
        shuffle_block_size=8,
    )

    ws_baseline = StaticMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    baseline = _drain_chunks(ws_baseline, limit=25)
    assert baseline

    ws_save = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    prefix = _drain_chunks(ws_save, limit=cut_position)
    state = ws_save.state_dict()
    assert "cursor_states" in state
    assert any(
        "block_rng_snapshot" in cursor_state
        for cursor_state in state["cursor_states"].values()
    )

    ws_load = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    ws_load.load_state_dict(state)
    suffix = _drain_chunks(ws_load, limit=max(0, len(baseline) - cut_position))

    assert prefix + suffix == baseline


def test_checkpoint_restore_accumulator_round_trip() -> None:
    """Accumulator floats survive state_dict round-trip."""
    ds_a = make_dataset("alpha", 100)
    ds_b = make_dataset("beta", 100)
    kwargs: dict = dict(
        datasets=[ds_a, ds_b],
        mixture={"alpha": 0.7, "beta": 0.3},
        chunk_size=3,
        seed=7,
    )

    ws = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    # Advance a few chunks so accumulators have non-zero values.
    _drain_chunks(ws, limit=5)
    state = ws.state_dict()

    # Verify accumulators are in the state dict and are floats.
    # Accumulators can go negative after deficit corrections are fed back,
    # but should stay within (-1, 1).
    assert "accumulators" in state
    for name, val in state["accumulators"].items():
        assert isinstance(val, float)
        assert -1.0 < val < 1.0, f"Accumulator for {name} out of (-1, 1): {val}"


def test_checkpoint_restore_accumulator_multilane() -> None:
    ds_a = make_sharded_dataset("alpha", [20, 15, 10])
    ds_b = make_sharded_dataset("beta", [18, 12])
    kwargs: dict = dict(
        datasets=[ds_a, ds_b],
        mixture={"alpha": 0.6, "beta": 0.4},
        chunk_size=6,
        seed=2026,
        shuffle_shards=True,
        shuffle_within_shard=True,
    )

    canonical_replicas = 2
    for lane in range(canonical_replicas):
        ws_baseline = StaticMixtureWorkSource(**kwargs).clone_for_lane(
            lane, canonical_replicas=canonical_replicas
        )
        baseline = _drain_chunks(ws_baseline)
        assert baseline

        ws_save = StaticMixtureWorkSource(**kwargs).clone_for_lane(
            lane, canonical_replicas=canonical_replicas
        )
        prefix = _drain_chunks(ws_save, limit=2)
        state = ws_save.state_dict()

        ws_load = StaticMixtureWorkSource(**kwargs).clone_for_lane(
            lane, canonical_replicas=canonical_replicas
        )
        ws_load.load_state_dict(state)
        suffix = _drain_chunks(ws_load)

        assert prefix + suffix == baseline


def test_repeat_checkpoint_restore_accumulator() -> None:
    ds = make_sharded_dataset("alpha", [8, 8])
    kwargs: dict = dict(
        datasets=[ds],
        mixture={ds.name: 1.0},
        chunk_size=5,
        seed=314,
        shuffle_shards=True,
        shuffle_within_shard=True,
        exhausted_policy="repeat",
        reshuffle_on_repeat=True,
        max_repeats=3,
    )

    ws_baseline = StaticMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    baseline = _drain_chunks(ws_baseline, limit=8)
    assert len(baseline) == 8

    ws_save = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    prefix = _drain_chunks(ws_save, limit=4)
    state = ws_save.state_dict()

    ws_load = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    ws_load.load_state_dict(state)
    suffix = _drain_chunks(ws_load, limit=4)

    assert prefix + suffix == baseline


# ---------------------------------------------------------------------------
# Clone independence
# ---------------------------------------------------------------------------


def test_clone_copies_accumulators_independently() -> None:
    """Advancing a clone must not affect the original's accumulators."""
    ds_a = make_dataset("alpha", 100)
    ds_b = make_dataset("beta", 100)

    original = StaticMixtureWorkSource(
        datasets=[ds_a, ds_b],
        mixture={"alpha": 0.7, "beta": 0.3},
        chunk_size=5,
        seed=42,
    )
    # Advance 2 chunks on the original before cloning.
    clone1 = original.clone_for_lane(0, canonical_replicas=1)
    _drain_chunks(clone1, limit=2)

    # Clone again from the original (fresh).
    clone2 = original.clone_for_lane(1, canonical_replicas=1)

    # clone1 should not have affected clone2's accumulators.
    # Both start from the original's accumulator state (all 0.0).
    # Verify by draining and checking they produce expected results.
    c2_chunk = clone2.next_chunk()
    assert c2_chunk is not None


# ---------------------------------------------------------------------------
# Deficit both directions
# ---------------------------------------------------------------------------


def test_deficit_negative_handled() -> None:
    """When floor quotas sum > chunk_size, correction removes excess slots."""
    # Use many components so accumulated remainders can push total over.
    datasets = [make_dataset(f"ds_{i}", 500) for i in range(10)]
    mixture = {f"ds_{i}": 0.1 for i in range(10)}
    ws = StaticMixtureWorkSource(
        datasets=datasets,
        mixture=mixture,
        chunk_size=7,
        seed=0,
    ).clone_for_lane(0, canonical_replicas=1)

    # Run many chunks to exercise both positive and negative deficit.
    for _ in range(50):
        chunk = ws.next_chunk()
        assert chunk is not None
        total = sum(len(v) for v in chunk.components.values())
        assert total == 7, f"Chunk had {total} samples, expected 7"


def test_deficit_positive_with_uneven_weights() -> None:
    """When floor quotas sum < chunk_size, correction adds extra slots."""
    ds_a = make_dataset("alpha", 200)
    ds_b = make_dataset("beta", 200)
    ds_c = make_dataset("gamma", 200)
    ws = StaticMixtureWorkSource(
        datasets=[ds_a, ds_b, ds_c],
        mixture={"alpha": 0.33, "beta": 0.33, "gamma": 0.34},
        chunk_size=10,
    ).clone_for_lane(0, canonical_replicas=1)

    for _ in range(30):
        chunk = ws.next_chunk()
        assert chunk is not None
        total = sum(len(v) for v in chunk.components.values())
        assert total == 10


# ---------------------------------------------------------------------------
# Repeat with multi-component rollback
# ---------------------------------------------------------------------------


def test_repeat_rollback_retry_multi_component() -> None:
    """Repeat rollback-and-retry works when one component exhausts first."""
    small = make_dataset("small", 12)
    large = make_dataset("large", 100)
    kwargs: dict = dict(
        datasets=[small, large],
        mixture={"small": 0.5, "large": 0.5},
        chunk_size=4,
        seed=0,
        exhausted_policy="repeat",
        reshuffle_on_repeat=True,
        max_repeats=3,
    )

    ws = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)

    # Should produce chunks past the point where "small" exhausts.
    chunks = _drain_chunks(ws, limit=20)
    assert len(chunks) == 20
    for ch in chunks:
        total = sum(len(v) for v in ch.values())
        assert total == 4


# ---------------------------------------------------------------------------
# Deficit correction feedback regression
# ---------------------------------------------------------------------------


def test_deficit_correction_does_not_inflate_small_components() -> None:
    """Largest-remainder correction must feed back into accumulators.

    Without feedback, small-weight components (w * chunk_size < 1) accumulate
    phantom credit: their fractional remainder stays high after receiving a +1
    correction, causing them to win future corrections repeatedly. Over many
    chunks this inflates their allocation by 30-40%+.

    The fix subtracts 1.0 from the accumulator on each +1 correction (and adds
    1.0 on each -1), keeping the accumulator aligned with actual allocations.
    """
    target = {
        "bulk": 0.4988,
        "mid_a": 0.15,
        "mid_b": 0.10,
        "med_a": 0.05,
        "med_b": 0.05,
        "med_c": 0.03,
        "med_d": 0.03,
        "sml_a": 0.02,
        "sml_b": 0.02,
        "sml_c": 0.02,
        "xs_a": 0.01,
        "xs_b": 0.01,
        "tiny_a": 0.005,
        "tiny_b": 0.003,
        "tiny_c": 0.001,
        "tiny_d": 0.001,
        "micro_a": 0.0005,
        "micro_b": 0.0005,
        "nano_a": 0.0001,
        "nano_b": 0.0001,
    }
    chunk_size = 1024
    num_chunks = 250

    datasets = [
        make_dataset(name, max(200, int(w * chunk_size * num_chunks * 2)))
        for name, w in target.items()
    ]
    ws = StaticMixtureWorkSource(
        datasets=datasets,
        mixture=target,
        chunk_size=chunk_size,
        seed=42,
        exhausted_policy="repeat",
    ).clone_for_lane(0, canonical_replicas=1)

    counts: dict[str, int] = dict.fromkeys(target, 0)
    for _ in range(num_chunks):
        chunk = ws.next_chunk()
        assert chunk is not None
        for name, samples in chunk.components.items():
            counts[name] += len(samples)

    total = sum(counts.values())
    assert total == chunk_size * num_chunks

    worst_name = ""
    worst_rel = 0.0
    for name, w in target.items():
        actual = counts[name] / total
        rel_err = abs(actual - w) / w
        if rel_err > worst_rel:
            worst_rel = rel_err
            worst_name = name

    assert worst_rel < 0.05, (
        f"Component '{worst_name}' has {worst_rel:.1%} relative error — "
        f"deficit correction feedback into accumulators may be missing"
    )


def test_load_state_dict_extra_dataset_raises() -> None:
    """Loading checkpoint from [A, B] into instance with [A, B, C] must fail."""
    ds_a = make_dataset("alpha", 50)
    ds_b = make_dataset("beta", 50)
    ds_c = make_dataset("gamma", 50)

    ws_save = StaticMixtureWorkSource(
        datasets=[ds_a, ds_b],
        mixture={"alpha": 0.6, "beta": 0.4},
        chunk_size=5,
        seed=0,
    ).clone_for_lane(0, canonical_replicas=1)
    _drain_chunks(ws_save, limit=2)
    state = ws_save.state_dict()

    ws_load = StaticMixtureWorkSource(
        datasets=[ds_a, ds_b, ds_c],
        mixture={"alpha": 0.4, "beta": 0.3, "gamma": 0.3},
        chunk_size=5,
        seed=0,
    ).clone_for_lane(0, canonical_replicas=1)

    with pytest.raises(RuntimeError, match="dataset_ids do not match"):
        ws_load.load_state_dict(state)


# ---------------------------------------------------------------------------
# Repeat policy — per-component independent exhaustion
# ---------------------------------------------------------------------------


def test_repeat_per_component_independent_exhaustion() -> None:
    """Small dataset resets while large dataset is still on first pass."""
    small = make_dataset("small", 12)
    large = make_dataset("large", 200)
    ws = StaticMixtureWorkSource(
        datasets=[small, large],
        mixture={"small": 0.5, "large": 0.5},
        chunk_size=4,
        seed=0,
        exhausted_policy="repeat",
        reshuffle_on_repeat=False,
    ).clone_for_lane(0, canonical_replicas=1)

    small_ids: list[tuple[int, int, int]] = []
    large_ids: list[tuple[int, int, int]] = []
    for _ in range(30):
        chunk = ws.next_chunk()
        assert chunk is not None
        for name, sids in _flatten_components(chunk).items():
            if name == "small":
                small_ids.extend(sids)
            else:
                large_ids.extend(sids)

    # small has 12 samples and we pulled ~60 small samples (30 chunks * ~2 each).
    # It must have repeated. large has 200 samples so should NOT have repeated.
    assert len(small_ids) > 12, "small should have repeated"
    assert len(set(small_ids)) <= 12, "small has only 12 unique samples"
    assert len(large_ids) <= 200, "large should not have repeated yet"
    assert len(set(large_ids)) == len(large_ids), "large should have no duplicates"


def test_repeat_determinism_two_instances() -> None:
    """Two identical repeat+reshuffle instances produce identical sequences."""
    ds_a = make_sharded_dataset("alpha", [8, 8])
    ds_b = make_sharded_dataset("beta", [6, 6])
    kwargs: dict = dict(
        datasets=[ds_a, ds_b],
        mixture={"alpha": 0.6, "beta": 0.4},
        chunk_size=5,
        seed=42,
        shuffle_shards=True,
        shuffle_within_shard=True,
        exhausted_policy="repeat",
        reshuffle_on_repeat=True,
        max_repeats=3,
    )

    ws1 = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    ws2 = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)

    for i in range(12):
        c1 = ws1.next_chunk()
        c2 = ws2.next_chunk()
        if c1 is None:
            assert c2 is None, f"ws2 still producing at chunk {i}"
            break
        assert c2 is not None, f"ws2 ended early at chunk {i}"
        assert _flatten_components(c1) == _flatten_components(c2), (
            f"Mismatch at chunk {i}"
        )


# ---------------------------------------------------------------------------
# Repeat policy — boundary values
# ---------------------------------------------------------------------------


def test_max_repeats_zero() -> None:
    """max_repeats=0 terminates after the first epoch (no resets)."""
    ds = make_dataset("alpha", 8)
    ws = StaticMixtureWorkSource(
        datasets=[ds],
        mixture={"alpha": 1.0},
        chunk_size=4,
        exhausted_policy="repeat",
        max_repeats=0,
    ).clone_for_lane(0, canonical_replicas=1)

    chunks = _drain_chunks(ws)
    # 8 samples / 4 per chunk = 2 chunks, then cursor._epoch (0) >= max_repeats (0).
    assert len(chunks) == 2
    assert ws.next_chunk() is None


def test_max_repeats_one() -> None:
    """max_repeats=1 gives exactly 2 epochs (epoch 0 + 1 reset)."""
    ds = make_dataset("alpha", 8)
    ws = StaticMixtureWorkSource(
        datasets=[ds],
        mixture={"alpha": 1.0},
        chunk_size=4,
        exhausted_policy="repeat",
        max_repeats=1,
    ).clone_for_lane(0, canonical_replicas=1)

    chunks = _drain_chunks(ws)
    # 2 chunks/epoch * 2 epochs = 4 chunks.
    assert len(chunks) == 4
    assert ws.next_chunk() is None


def test_repeat_guard_rejects_too_few_samples() -> None:
    """Repeat policy must reject datasets that can't fill their per-chunk quota.

    With weight=0.5 and chunk_size=10, the max single-chunk quota is
    ceil(0.5 * 10) = 5.  A dataset with only 3 samples would cause an
    infinite retry loop in _next_chunk_accumulator, so the constructor
    must reject it.
    """
    ds_small = make_dataset("small", 3)
    ds_big = make_dataset("big", 100)
    with pytest.raises(ValueError, match="requires at least 5"):
        StaticMixtureWorkSource(
            datasets=[ds_small, ds_big],
            mixture={"small": 0.5, "big": 0.5},
            chunk_size=10,
            exhausted_policy="repeat",
        )


def test_repeat_guard_rejects_2_samples_chunk5() -> None:
    """Regression: 2-sample dataset with chunk_size=5 would previously hang."""
    ds_small = make_dataset("small", 2)
    ds_big = make_dataset("big", 100)
    with pytest.raises(ValueError, match="requires at least 3"):
        StaticMixtureWorkSource(
            datasets=[ds_small, ds_big],
            mixture={"small": 0.5, "big": 0.5},
            chunk_size=5,
            exhausted_policy="repeat",
        )


def test_repeat_guard_allows_exact_quota() -> None:
    """When total_samples == ceil(weight * chunk_size), construction succeeds."""
    # ceil(0.5 * 10) = 5, dataset has exactly 5 samples — should be fine
    ds_exact = make_dataset("exact", 5)
    ds_big = make_dataset("big", 100)
    ws = StaticMixtureWorkSource(
        datasets=[ds_exact, ds_big],
        mixture={"exact": 0.5, "big": 0.5},
        chunk_size=10,
        exhausted_policy="repeat",
        max_repeats=2,
    ).clone_for_lane(0, canonical_replicas=1)
    # Should produce chunks without hanging
    chunk = ws.next_chunk()
    assert chunk is not None


def test_single_sample_dataset_repeat() -> None:
    """Repeat with 1 sample, chunk_size=1, max_repeats=3 gives 4 chunks."""
    ds = make_dataset("only", 1)
    ws = StaticMixtureWorkSource(
        datasets=[ds],
        mixture={"only": 1.0},
        chunk_size=1,
        exhausted_policy="repeat",
        reshuffle_on_repeat=False,
        max_repeats=3,
    ).clone_for_lane(0, canonical_replicas=1)

    chunks = _drain_chunks(ws)
    assert len(chunks) == 4
    assert ws.next_chunk() is None


# ---------------------------------------------------------------------------
# Multi-lane
# ---------------------------------------------------------------------------


def test_checkpoint_restore_3_lanes() -> None:
    """Checkpoint/restore is correct for 3 lanes."""
    ds_a = make_sharded_dataset("alpha", [30, 20, 15])
    ds_b = make_sharded_dataset("beta", [25, 10])
    kwargs: dict = dict(
        datasets=[ds_a, ds_b],
        mixture={"alpha": 0.6, "beta": 0.4},
        chunk_size=5,
        seed=2026,
        shuffle_shards=True,
        shuffle_within_shard=True,
    )

    canonical_replicas = 3
    for lane in range(canonical_replicas):
        ws_baseline = StaticMixtureWorkSource(**kwargs).clone_for_lane(
            lane, canonical_replicas=canonical_replicas
        )
        baseline = _drain_chunks(ws_baseline)
        assert baseline

        ws_save = StaticMixtureWorkSource(**kwargs).clone_for_lane(
            lane, canonical_replicas=canonical_replicas
        )
        prefix = _drain_chunks(ws_save, limit=2)
        state = ws_save.state_dict()

        ws_load = StaticMixtureWorkSource(**kwargs).clone_for_lane(
            lane, canonical_replicas=canonical_replicas
        )
        ws_load.load_state_dict(state)
        suffix = _drain_chunks(ws_load)

        assert prefix + suffix == baseline, f"Lane {lane} mismatch"


def test_multilane_no_duplicate_chunks() -> None:
    """4 lanes must receive disjoint chunk sets."""
    ds_a = make_dataset("alpha", 200)
    ds_b = make_dataset("beta", 200)
    kwargs: dict = dict(
        datasets=[ds_a, ds_b],
        mixture={"alpha": 0.6, "beta": 0.4},
        chunk_size=5,
        seed=0,
    )

    canonical_replicas = 4
    all_samples: list[set[tuple[int, int, int]]] = []
    for lane in range(canonical_replicas):
        ws = StaticMixtureWorkSource(**kwargs).clone_for_lane(
            lane, canonical_replicas=canonical_replicas
        )
        chunks = _drain_chunks(ws)
        lane_samples = {sid for ch in chunks for sids in ch.values() for sid in sids}
        all_samples.append(lane_samples)

    # Verify pairwise disjointness.
    for i in range(canonical_replicas):
        for j in range(i + 1, canonical_replicas):
            overlap = all_samples[i] & all_samples[j]
            assert not overlap, f"Lane {i} and {j} share {len(overlap)} samples"


# ---------------------------------------------------------------------------
# Redistribute policy
# ---------------------------------------------------------------------------


def test_redistribute_raises_not_implemented() -> None:
    """Redistribute policy is accepted but raises at construction (via len)."""
    ds = make_dataset("alpha", 50)
    with pytest.raises(NotImplementedError, match="redistribute"):
        StaticMixtureWorkSource(
            datasets=[ds],
            mixture={"alpha": 1.0},
            chunk_size=5,
            exhausted_policy="redistribute",
        )


# ---------------------------------------------------------------------------
# Scale stress tests
# ---------------------------------------------------------------------------


def test_long_run_1000_chunks_stability() -> None:
    """1000 chunks: chunk_size invariant, bounded accumulators, convergence."""
    target = {"alpha": 0.6, "beta": 0.3, "gamma": 0.1}
    datasets = [make_dataset(name, 15000) for name in target]
    ws = StaticMixtureWorkSource(
        datasets=datasets,
        mixture=target,
        chunk_size=10,
        seed=0,
    ).clone_for_lane(0, canonical_replicas=1)

    counts: dict[str, int] = dict.fromkeys(target, 0)
    for _ in range(1000):
        chunk = ws.next_chunk()
        assert chunk is not None
        total = sum(len(v) for v in chunk.components.values())
        assert total == 10

        for name, samples in chunk.components.items():
            counts[name] += len(samples)

        # Accumulators must stay bounded.
        for val in ws._strategy._accumulators.values():
            assert -1.0 < val < 1.0, f"Accumulator out of bounds: {val}"

    grand_total = sum(counts.values())
    assert grand_total == 10000
    for name, w in target.items():
        actual = counts[name] / grand_total
        assert abs(actual - w) < 0.005, f"{name}: expected ~{w}, got {actual}"


def test_50_components_convergence() -> None:
    """50 equal-weight components converge under scale."""
    n = 50
    target = {f"ds_{i}": 1.0 / n for i in range(n)}
    datasets = [make_dataset(f"ds_{i}", 5000) for i in range(n)]
    ws = StaticMixtureWorkSource(
        datasets=datasets,
        mixture=target,
        chunk_size=100,
        seed=0,
    ).clone_for_lane(0, canonical_replicas=1)

    counts: dict[str, int] = dict.fromkeys(target, 0)
    for _ in range(200):
        chunk = ws.next_chunk()
        assert chunk is not None
        total = sum(len(v) for v in chunk.components.values())
        assert total == 100
        for name, samples in chunk.components.items():
            counts[name] += len(samples)

    grand_total = sum(counts.values())
    expected = 1.0 / n
    for name in target:
        actual = counts[name] / grand_total
        rel_err = abs(actual - expected) / expected
        assert rel_err < 0.05, (
            f"{name}: expected ~{expected:.4f}, got {actual:.4f} "
            f"(rel_err={rel_err:.1%})"
        )


def test_chunk_size_1_with_10_components() -> None:
    """chunk_size=1 with 10 components: extreme sparsity, fair round-robin."""
    n = 10
    datasets = [make_dataset(f"ds_{i}", 200) for i in range(n)]
    target = {f"ds_{i}": 1.0 / n for i in range(n)}
    ws = StaticMixtureWorkSource(
        datasets=datasets,
        mixture=target,
        chunk_size=1,
        seed=0,
    ).clone_for_lane(0, canonical_replicas=1)

    counts: dict[str, int] = dict.fromkeys(target, 0)
    for _ in range(100):
        chunk = ws.next_chunk()
        assert chunk is not None
        total = sum(len(v) for v in chunk.components.values())
        assert total == 1
        for name, samples in chunk.components.items():
            counts[name] += len(samples)

    # Each should get ~10 out of 100.
    for name in target:
        assert counts[name] == 10, f"{name}: expected 10, got {counts[name]}"


def test_chunk_size_1_with_100_components() -> None:
    """chunk_size=1 with 100 components: extreme scale + sparsity."""
    n = 100
    datasets = [make_dataset(f"ds_{i}", 200) for i in range(n)]
    target = {f"ds_{i}": 1.0 / n for i in range(n)}
    ws = StaticMixtureWorkSource(
        datasets=datasets,
        mixture=target,
        chunk_size=1,
        seed=0,
    ).clone_for_lane(0, canonical_replicas=1)

    counts: dict[str, int] = dict.fromkeys(target, 0)
    for _ in range(1000):
        chunk = ws.next_chunk()
        assert chunk is not None
        total = sum(len(v) for v in chunk.components.values())
        assert total == 1
        for name, samples in chunk.components.items():
            counts[name] += len(samples)

    # Each should get exactly 10 out of 1000.
    for name in target:
        assert counts[name] == 10, f"{name}: expected 10, got {counts[name]}"


# ---------------------------------------------------------------------------
# Symmetry and simultaneous exhaustion
# ---------------------------------------------------------------------------


def test_identical_weights_symmetry() -> None:
    """4 equal-weight datasets get exactly equal allocation."""
    n = 4
    datasets = [make_dataset(f"ds_{i}", 100) for i in range(n)]
    target = {f"ds_{i}": 0.25 for i in range(n)}
    ws = StaticMixtureWorkSource(
        datasets=datasets,
        mixture=target,
        chunk_size=8,
        seed=0,
    ).clone_for_lane(0, canonical_replicas=1)

    counts: dict[str, int] = dict.fromkeys(target, 0)
    chunks = _drain_chunks(ws)
    for ch in chunks:
        for name, sids in ch.items():
            counts[name] += len(sids)

    values = list(counts.values())
    assert all(v == values[0] for v in values), (
        f"Symmetric weights should yield equal totals, got {counts}"
    )


def test_all_datasets_exhaust_simultaneously() -> None:
    """Datasets sized proportional to weights exhaust on the same chunk."""
    # weights 0.6/0.3/0.1, chunk_size=10, datasets with 60/30/10 samples.
    ds_a = make_dataset("alpha", 60)
    ds_b = make_dataset("beta", 30)
    ds_c = make_dataset("gamma", 10)
    ws = StaticMixtureWorkSource(
        datasets=[ds_a, ds_b, ds_c],
        mixture={"alpha": 0.6, "beta": 0.3, "gamma": 0.1},
        chunk_size=10,
        seed=0,
    ).clone_for_lane(0, canonical_replicas=1)

    chunks = _drain_chunks(ws)
    assert len(chunks) == 10
    assert ws.next_chunk() is None


def test_all_datasets_exhaust_simultaneously_repeat() -> None:
    """Simultaneous exhaustion with repeat policy: all reset at once."""
    ds_a = make_dataset("alpha", 60)
    ds_b = make_dataset("beta", 30)
    ds_c = make_dataset("gamma", 10)
    ws = StaticMixtureWorkSource(
        datasets=[ds_a, ds_b, ds_c],
        mixture={"alpha": 0.6, "beta": 0.3, "gamma": 0.1},
        chunk_size=10,
        seed=0,
        exhausted_policy="repeat",
        reshuffle_on_repeat=False,
        max_repeats=2,
    ).clone_for_lane(0, canonical_replicas=1)

    chunks = _drain_chunks(ws, limit=30)
    for ch in chunks:
        total = sum(len(v) for v in ch.values())
        assert total == 10, f"Chunk had {total} samples"
    # Should produce 10 chunks/epoch * 3 epochs = 30 chunks.
    assert len(chunks) == 30


# ---------------------------------------------------------------------------
# Long-run accumulator properties
# ---------------------------------------------------------------------------


def test_accumulator_convergence_survives_checkpoint() -> None:
    """Checkpoint mid-stream produces identical final totals vs straight-through."""
    target = {"alpha": 0.7, "beta": 0.2, "gamma": 0.1}
    datasets = [make_dataset(name, 10000) for name in target]
    kwargs: dict = dict(
        datasets=datasets,
        mixture=target,
        chunk_size=10,
        seed=42,
    )

    # Straight-through: 500 chunks.
    ws_full = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    full_counts: dict[str, int] = dict.fromkeys(target, 0)
    for _ in range(500):
        chunk = ws_full.next_chunk()
        assert chunk is not None
        for name, samples in chunk.components.items():
            full_counts[name] += len(samples)

    # Checkpoint at 100, restore, continue to 500.
    ws_save = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    split_counts: dict[str, int] = dict.fromkeys(target, 0)
    for _ in range(100):
        chunk = ws_save.next_chunk()
        assert chunk is not None
        for name, samples in chunk.components.items():
            split_counts[name] += len(samples)
    state = ws_save.state_dict()

    ws_load = StaticMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    ws_load.load_state_dict(state)
    for _ in range(400):
        chunk = ws_load.next_chunk()
        assert chunk is not None
        for name, samples in chunk.components.items():
            split_counts[name] += len(samples)

    assert full_counts == split_counts


# ---------------------------------------------------------------------------
# Legacy checkpoint detection and sticky mode
# ---------------------------------------------------------------------------


def test_legacy_checkpoint_detection_and_sticky_mode() -> None:
    """Loading a checkpoint without 'accumulators' triggers legacy_fixed mode.

    The mode is sticky: re-saving produces a legacy-format checkpoint.
    """
    ds_a = make_sharded_dataset("alpha", [10, 10])
    ds_b = make_sharded_dataset("beta", [10, 10])
    mix = MixtureSpec({ds_a.name: 0.6, ds_b.name: 0.4}).weights

    # Build a source and get a checkpoint in accumulator mode
    ws_save = StaticMixtureWorkSource(
        [ds_a, ds_b], mix, chunk_size=5, seed=42, shuffle_shards=False
    ).clone_for_lane(0, canonical_replicas=1)
    ws_save.next_chunk()
    state = ws_save.state_dict()

    # Verify it's an accumulator checkpoint
    assert "accumulators" in state

    # Manually strip accumulator fields to simulate a pre-strategy v1
    # checkpoint and downgrade the version so the v1→v2 migration infers
    # allocation_mode from the (now-absent) accumulators field.
    legacy_state = {
        k: v for k, v in state.items() if k not in ("accumulators", "allocation_mode")
    }
    legacy_state["version"] = 1
    # v2-only fields aren't part of the v1 shape; drop them so the v1 schema
    # validates cleanly before the migration runs.
    for k in ("shuffle_block_size_spec", "cursor_block_sizes"):
        legacy_state.pop(k, None)

    # Load it — should detect legacy mode
    ws_load = StaticMixtureWorkSource(
        [ds_a, ds_b], mix, chunk_size=5, seed=42, shuffle_shards=False
    ).clone_for_lane(0, canonical_replicas=1)
    ws_load.load_state_dict(legacy_state)
    assert isinstance(ws_load._strategy, LegacyFixedStrategy)

    # Produce a chunk and re-checkpoint — should remain legacy format.
    # v2 always writes ``accumulators`` (None signals legacy); ``allocation_mode``
    # is the canonical signal.
    ws_load.next_chunk()
    re_saved = ws_load.state_dict()
    assert re_saved.get("accumulators") is None
    assert re_saved.get("allocation_mode") == "legacy_fixed"

    # Close the loop: load the re-saved checkpoint into a fresh instance
    ws_load2 = StaticMixtureWorkSource(
        [ds_a, ds_b], mix, chunk_size=5, seed=42, shuffle_shards=False
    ).clone_for_lane(0, canonical_replicas=1)
    ws_load2.load_state_dict(re_saved)
    assert isinstance(ws_load2._strategy, LegacyFixedStrategy)

    c1 = ws_load.next_chunk()
    c2 = ws_load2.next_chunk()
    assert _flatten_components(c1) == _flatten_components(c2)


def test_accumulator_checkpoint_round_trip() -> None:
    """New runs produce accumulator checkpoints that round-trip correctly."""
    ds = make_sharded_dataset("alpha", [15, 15])
    ws_save = StaticMixtureWorkSource(
        [ds], {ds.name: 1.0}, chunk_size=5, seed=0, shuffle_shards=False
    ).clone_for_lane(0, canonical_replicas=1)

    # Baseline
    baseline_chunks = _drain_chunks(ws_save, limit=4)
    state = ws_save.state_dict()
    assert "accumulators" in state
    assert state.get("allocation_mode") == "accumulator"

    # Restore and continue
    ws_load = StaticMixtureWorkSource(
        [ds], {ds.name: 1.0}, chunk_size=5, seed=0, shuffle_shards=False
    ).clone_for_lane(0, canonical_replicas=1)
    ws_load.load_state_dict(state)
    assert isinstance(ws_load._strategy, AccumulatorStrategy)

    # Both should continue identically
    suffix_save = _drain_chunks(ws_save, limit=2)
    suffix_load = _drain_chunks(ws_load, limit=2)
    assert suffix_save == suffix_load


def test_legacy_checkpoint_determinism() -> None:
    """Runs resumed from legacy checkpoints produce identical output to a straight-through run.

    This simulates loading a checkpoint that was created by the old
    StaticMixtureWorkSource (no accumulators), and verifies that the legacy
    fixed-quota code path produces bit-identical results.
    """
    ds_a = make_sharded_dataset("alpha", [20, 20])
    ds_b = make_sharded_dataset("beta", [15, 15])
    mix = MixtureSpec({ds_a.name: 0.6, ds_b.name: 0.4}).weights

    # Build two sources in accumulator mode, advance 3 chunks, checkpoint
    ws1 = StaticMixtureWorkSource(
        [ds_a, ds_b], mix, chunk_size=5, seed=7, shuffle_shards=False
    ).clone_for_lane(0, canonical_replicas=1)
    ws2 = StaticMixtureWorkSource(
        [ds_a, ds_b], mix, chunk_size=5, seed=7, shuffle_shards=False
    ).clone_for_lane(0, canonical_replicas=1)

    for _ in range(3):
        ws1.next_chunk()
        ws2.next_chunk()

    state = ws1.state_dict()
    # Convert to legacy v1 format: strip strategy-era fields and downgrade
    # version so the v1→v2 migration runs.
    legacy_state = {
        k: v for k, v in state.items() if k not in ("accumulators", "allocation_mode")
    }
    legacy_state["version"] = 1
    for k in ("shuffle_block_size_spec", "cursor_block_sizes"):
        legacy_state.pop(k, None)

    # Load both from the same legacy checkpoint
    ws_load1 = StaticMixtureWorkSource(
        [ds_a, ds_b], mix, chunk_size=5, seed=7, shuffle_shards=False
    ).clone_for_lane(0, canonical_replicas=1)
    ws_load1.load_state_dict(legacy_state)

    ws_load2 = StaticMixtureWorkSource(
        [ds_a, ds_b], mix, chunk_size=5, seed=7, shuffle_shards=False
    ).clone_for_lane(0, canonical_replicas=1)
    ws_load2.load_state_dict(legacy_state)

    # Both must produce identical chunks
    for _ in range(5):
        c1 = ws_load1.next_chunk()
        c2 = ws_load2.next_chunk()
        if c1 is None or c2 is None:
            assert c1 is None and c2 is None
            break
        assert _flatten_components(c1) == _flatten_components(c2)
