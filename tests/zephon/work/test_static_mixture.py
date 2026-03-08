import pytest

from zephon.io import Dataset, InMemoryShard
from zephon.work import MixtureSpec, StaticMixtureWorkSource


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
    # These asserts will surface any mismatch in how knobs are serialized.
    assert knobs["shuffle_shards"] == shuffle_shards
    assert knobs["shuffle_within_shard"] == shuffle_within
    assert knobs["shuffle_block_size"] == block_size


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


def test_static_mixture_emits_fixed_quota_chunks() -> None:
    dataset_a = make_dataset("alpha", 9)
    dataset_b = make_dataset("beta", 6)
    mixture = MixtureSpec({"alpha": 0.6, "beta": 0.4}).weights
    work = StaticMixtureWorkSource(
        [dataset_a, dataset_b],
        mixture,
        chunk_size=5,
        shuffle_shards=False,
    )

    ws = work.clone_for_lane(0, canonical_replicas=1)
    assert len(ws) == 15

    chunk = ws.next_chunk()
    assert chunk is not None
    assert sorted(chunk.components.keys()) == ["alpha", "beta"]
    assert len(chunk.components["alpha"]) == 3
    assert len(chunk.components["beta"]) == 2

    assert len(ws) == 10

    ws.next_chunk()
    ws.next_chunk()

    assert len(ws) == 0
    assert ws.next_chunk() is None


def test_chunk_size_smaller_than_components_raises() -> None:
    dataset_a = make_dataset("alpha", 2)
    dataset_b = make_dataset("beta", 2)
    dataset_c = make_dataset("gamma", 2)
    mixture = MixtureSpec({"alpha": 0.5, "beta": 0.3, "gamma": 0.2}).weights
    with pytest.raises(ValueError):
        StaticMixtureWorkSource(
            [dataset_a, dataset_b, dataset_c],
            mixture,
            chunk_size=2,
            shuffle_shards=False,
        )


def test_warning_for_small_components() -> None:
    dataset_a = make_dataset("alpha", 6)
    dataset_b = make_dataset("beta", 6)
    dataset_c = make_dataset("gamma", 6)
    mixture = {"alpha": 0.9, "beta": 0.09, "gamma": 0.01}
    with pytest.warns(RuntimeWarning):
        StaticMixtureWorkSource(
            [dataset_a, dataset_b, dataset_c],
            MixtureSpec(mixture).weights,
            chunk_size=4,
            shuffle_shards=False,
        )


def test_chunk_samples_match_components_and_counts() -> None:
    dataset_a = make_dataset("alpha", 5)
    dataset_b = make_dataset("beta", 5)
    mixture = MixtureSpec({"alpha": 0.6, "beta": 0.4}).weights
    work = StaticMixtureWorkSource(
        [dataset_a, dataset_b],
        mixture,
        chunk_size=4,
        seed=13,
        shuffle_shards=False,
    )

    ws = work.clone_for_lane(0, canonical_replicas=1)
    dataset_ids = ws.dataset_ids
    assert dataset_ids["alpha"] != dataset_ids["beta"]
    assert len(ws) == 8

    first_chunk = ws.next_chunk()
    assert first_chunk is not None
    assert sorted(first_chunk.components.keys()) == ["alpha", "beta"]
    assert len(first_chunk.components["alpha"]) == 2
    assert len(first_chunk.components["beta"]) == 2
    assert all(
        sample_id[0] == dataset_ids["alpha"]
        for sample_id in first_chunk.components["alpha"]
    )
    assert all(
        sample_id[0] == dataset_ids["beta"]
        for sample_id in first_chunk.components["beta"]
    )
    assert [sample_id[2] for sample_id in first_chunk.components["alpha"]] == [0, 1]
    assert [sample_id[2] for sample_id in first_chunk.components["beta"]] == [0, 1]
    assert len(ws) == 4

    second_chunk = ws.next_chunk()
    assert second_chunk is not None
    assert len(second_chunk.components["alpha"]) == 2
    assert len(second_chunk.components["beta"]) == 2
    assert [sample_id[2] for sample_id in second_chunk.components["alpha"]] == [2, 3]
    assert [sample_id[2] for sample_id in second_chunk.components["beta"]] == [2, 3]
    assert len(ws) == 0

    assert ws.next_chunk() is None


def test_chunk_samples_match_components_and_counts_25_75() -> None:
    dataset_a = make_dataset("alpha", 6)
    dataset_b = make_dataset("beta", 5)
    mixture = MixtureSpec({"alpha": 0.75, "beta": 0.25}).weights
    work = StaticMixtureWorkSource(
        [dataset_a, dataset_b],
        mixture,
        chunk_size=4,
        seed=13,
        shuffle_shards=False,
    )

    ws = work.clone_for_lane(0, canonical_replicas=1)
    dataset_ids = ws.dataset_ids
    assert dataset_ids["alpha"] != dataset_ids["beta"]
    assert len(ws) == 8

    first_chunk = ws.next_chunk()
    assert first_chunk is not None
    assert sorted(first_chunk.components.keys()) == ["alpha", "beta"]
    assert len(first_chunk.components["alpha"]) == 3
    assert len(first_chunk.components["beta"]) == 1
    assert all(
        sample_id[0] == dataset_ids["alpha"]
        for sample_id in first_chunk.components["alpha"]
    )
    assert all(
        sample_id[0] == dataset_ids["beta"]
        for sample_id in first_chunk.components["beta"]
    )
    assert [sample_id[2] for sample_id in first_chunk.components["alpha"]] == [0, 1, 2]
    assert [sample_id[2] for sample_id in first_chunk.components["beta"]] == [0]
    assert len(ws) == 4

    second_chunk = ws.next_chunk()
    assert second_chunk is not None
    assert len(second_chunk.components["alpha"]) == 3
    assert len(second_chunk.components["beta"]) == 1
    assert [sample_id[2] for sample_id in second_chunk.components["alpha"]] == [3, 4, 5]
    assert [sample_id[2] for sample_id in second_chunk.components["beta"]] == [1]
    assert len(ws) == 0

    assert ws.next_chunk() is None


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
    """A v1 checkpoint (no cursor_epochs) loads correctly as epoch 0."""
    ds = make_dataset("alpha", 10)
    # Build a "stop" source, consume 1 chunk, get state_dict
    stop_ws = StaticMixtureWorkSource(
        [ds], {ds.name: 1.0}, chunk_size=5, seed=0, shuffle_shards=False
    ).clone_for_lane(0, canonical_replicas=1)
    stop_ws.next_chunk()
    saved_state = stop_ws.state_dict()
    # All epochs should be 0 (never repeated)
    assert all(e == 0 for e in saved_state.get("cursor_epochs", {}).values())

    # Load into a repeat source with matching reshuffle/max_repeats defaults
    ws_load = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=5,
        seed=0,
        shuffle_shards=False,
        exhausted_policy="repeat",
    ).clone_for_lane(0, canonical_replicas=1)
    ws_load.load_state_dict(saved_state)

    # Should continue producing chunks (repeat policy)
    ch = ws_load.next_chunk()
    assert ch is not None


def test_repeat_small_dataset_guard() -> None:
    """Dataset with fewer samples than quota raises at construction."""
    ds = make_dataset("tiny", 2)
    with pytest.raises(ValueError, match="repeat policy requires at least"):
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
