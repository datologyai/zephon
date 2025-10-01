import pytest

from zephon.io import Dataset, InMemoryShard
from zephon.work import MixtureSpec, StaticMixtureWorkSource


def make_dataset(name: str, sample_count: int) -> Dataset:
    rows = [{"text": f"{name}-{i}"} for i in range(sample_count)]
    return Dataset.from_dict(name, {0: InMemoryShard(rows)})


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

    chunk = work.next_chunk()
    assert chunk is not None
    assert sorted(chunk.components.keys()) == ["alpha", "beta"]
    assert len(chunk.components["alpha"]) == 3
    assert len(chunk.components["beta"]) == 2

    assert len(work) == 10

    work.next_chunk()
    work.next_chunk()

    assert len(work) == 0
    assert work.next_chunk() is None


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

    first_chunk = work.next_chunk()
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

    second_chunk = work.next_chunk()
    assert second_chunk is not None
    assert len(second_chunk.components["alpha"]) == 2
    assert len(second_chunk.components["beta"]) == 2
    assert [sample_id[2] for sample_id in second_chunk.components["alpha"]] == [2, 3]
    assert [sample_id[2] for sample_id in second_chunk.components["beta"]] == [2, 3]
    assert len(work) == 0

    assert work.next_chunk() is None


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

    first_chunk = work.next_chunk()
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

    second_chunk = work.next_chunk()
    assert second_chunk is not None
    assert len(second_chunk.components["alpha"]) == 3
    assert len(second_chunk.components["beta"]) == 1
    assert [sample_id[2] for sample_id in second_chunk.components["alpha"]] == [3, 4, 5]
    assert [sample_id[2] for sample_id in second_chunk.components["beta"]] == [1]
    assert len(work) == 0

    assert work.next_chunk() is None


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

    chunk_one = work_one.next_chunk()
    chunk_two = work_two.next_chunk()

    assert chunk_one is not None
    assert chunk_two is not None
    assert chunk_one.components == chunk_two.components
    # Ensure samples come from their respective datasets even after shuffles.
    ids = work_one.dataset_ids
    for comp_name, samples in chunk_one.components.items():
        expected_dataset_id = ids[comp_name]
        assert all(sample[0] == expected_dataset_id for sample in samples)
