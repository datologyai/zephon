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

    # Consume some chunks to create a non-trivial checkpoint
    prefix_chunks: list[dict[str, list[tuple[int, int, int]]]] = []
    for _ in range(3):
        ch = work_a.next_chunk_for(0)
        assert ch is not None
        prefix_chunks.append(_flatten_components(ch))

    st = work_a.state_dict()

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
    work_b.load_state_dict(st)

    # After load, structural state should match the checkpoint
    st_b = work_b.state_dict()
    assert st_b["seed"] == st["seed"]
    assert st_b["chunk_size"] == st["chunk_size"]
    assert st_b["weights"] == st["weights"]
    assert st_b["component_order"] == st["component_order"]
    assert st_b["dataset_ids"] == st["dataset_ids"]
    assert st_b["cursor_positions"] == st["cursor_positions"]
    assert st_b["global_chunk_index"] == st["global_chunk_index"]

    # Continuing from the checkpoint, both streams must produce identical chunks
    for _ in range(5):
        ca = work_a.next_chunk_for(0)
        cb = work_b.next_chunk_for(0)
        if ca is None or cb is None:
            assert ca is None and cb is None
            break
        assert _flatten_components(ca) == _flatten_components(cb)


def test_load_state_dict_version_mismatch_raises() -> None:
    ds = make_sharded_dataset("alpha", [2, 2, 2])
    work = StaticMixtureWorkSource(
        [ds], {ds.name: 1.0}, chunk_size=4, seed=5, shuffle_shards=True
    )

    st = work.state_dict()
    st_bad = dict(st)
    st_bad["version"] = 999

    other = StaticMixtureWorkSource([ds], {ds.name: 1.0}, chunk_size=4)
    with pytest.raises(RuntimeError):
        other.load_state_dict(st_bad)


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

    assert len(work) == 15

    chunk = work.next_chunk_for(0)
    assert chunk is not None
    assert sorted(chunk.components.keys()) == ["alpha", "beta"]
    assert len(chunk.components["alpha"]) == 3
    assert len(chunk.components["beta"]) == 2

    assert len(work) == 10

    work.next_chunk_for(0)
    work.next_chunk_for(0)

    assert len(work) == 0
    assert work.next_chunk_for(0) is None


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

    dataset_ids = work.dataset_ids
    assert dataset_ids["alpha"] != dataset_ids["beta"]
    assert len(work) == 8

    first_chunk = work.next_chunk_for(0)
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
    assert len(work) == 4

    second_chunk = work.next_chunk_for(0)
    assert second_chunk is not None
    assert len(second_chunk.components["alpha"]) == 2
    assert len(second_chunk.components["beta"]) == 2
    assert [sample_id[2] for sample_id in second_chunk.components["alpha"]] == [2, 3]
    assert [sample_id[2] for sample_id in second_chunk.components["beta"]] == [2, 3]
    assert len(work) == 0

    assert work.next_chunk_for(0) is None


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

    dataset_ids = work.dataset_ids
    assert dataset_ids["alpha"] != dataset_ids["beta"]
    assert len(work) == 8

    first_chunk = work.next_chunk_for(0)
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
    assert len(work) == 4

    second_chunk = work.next_chunk_for(0)
    assert second_chunk is not None
    assert len(second_chunk.components["alpha"]) == 3
    assert len(second_chunk.components["beta"]) == 1
    assert [sample_id[2] for sample_id in second_chunk.components["alpha"]] == [3, 4, 5]
    assert [sample_id[2] for sample_id in second_chunk.components["beta"]] == [1]
    assert len(work) == 0

    assert work.next_chunk_for(0) is None


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

    chunk_one = work_one.next_chunk_for(0)
    chunk_two = work_two.next_chunk_for(0)

    assert chunk_one is not None
    assert chunk_two is not None
    assert chunk_one.components == chunk_two.components
    # Ensure samples come from their respective datasets even after shuffles.
    ids = work_one.dataset_ids
    for comp_name, samples in chunk_one.components.items():
        expected_dataset_id = ids[comp_name]
        assert all(sample[0] == expected_dataset_id for sample in samples)
