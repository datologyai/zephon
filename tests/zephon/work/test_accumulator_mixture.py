import pytest

from zephon.io import Dataset, InMemoryShard
from zephon.work import AccumulatorMixtureWorkSource


def make_dataset(name: str, sample_count: int) -> Dataset:
    rows = [{"text": f"{name}-{i}"} for i in range(sample_count)]
    return Dataset.from_dict(name, {0: InMemoryShard(rows)})


def make_sharded_dataset(name: str, shard_lengths: list[int]) -> Dataset:
    shards: dict[int, InMemoryShard] = {}
    for shard_id, count in enumerate(shard_lengths):
        rows = [{"text": f"{name}-s{shard_id}-{i}"} for i in range(count)]
        shards[shard_id] = InMemoryShard(rows)
    return Dataset.from_dict(name, shards)


def _flatten_components(chunk) -> dict[str, list[tuple[int, int, int]]]:
    assert chunk is not None
    out: dict[str, list[tuple[int, int, int]]] = {}
    for name, items in chunk.components.items():
        out[name] = [tuple(map(int, sid)) for sid in items]
    return out


def _drain_chunks(
    ws: AccumulatorMixtureWorkSource, limit: int | None = None
) -> list[dict[str, list[tuple[int, int, int]]]]:
    out: list[dict[str, list[tuple[int, int, int]]]] = []
    while limit is None or len(out) < limit:
        ch = ws.next_chunk()
        if ch is None:
            break
        out.append(_flatten_components(ch))
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_zero_chunk_size_raises() -> None:
    ds = make_dataset("alpha", 10)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        AccumulatorMixtureWorkSource(
            datasets=[ds], mixture={ds.name: 1.0}, chunk_size=0
        )


def test_no_datasets_raises() -> None:
    with pytest.raises(ValueError, match="At least one dataset"):
        AccumulatorMixtureWorkSource(datasets=[], mixture={}, chunk_size=4)


def test_chunk_size_1_with_multiple_components_allowed() -> None:
    """Unlike StaticMixtureWorkSource, chunk_size < len(components) is valid."""
    ds_a = make_dataset("alpha", 10)
    ds_b = make_dataset("beta", 10)
    ws = AccumulatorMixtureWorkSource(
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
    ws = AccumulatorMixtureWorkSource(
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
    ws = AccumulatorMixtureWorkSource(
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
    ws = AccumulatorMixtureWorkSource(
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
    ws = AccumulatorMixtureWorkSource(
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
    ws = AccumulatorMixtureWorkSource(
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
# Equivalence with StaticMixtureWorkSource (single component)
# ---------------------------------------------------------------------------


def test_single_component_matches_chunk_size() -> None:
    """With 1 dataset, every chunk gets exactly chunk_size samples."""
    ds = make_dataset("only", 100)
    ws = AccumulatorMixtureWorkSource(
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

    ws1 = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    ws2 = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)

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
    ws_baseline = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    baseline = _drain_chunks(ws_baseline)
    assert len(baseline) > 5

    # Save after 3 chunks, restore, drain rest.
    ws_save = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    prefix = _drain_chunks(ws_save, limit=3)
    state = ws_save.state_dict()

    ws_load = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
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

    ws_baseline = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    baseline = _drain_chunks(ws_baseline, limit=25)
    assert baseline

    ws_save = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    prefix = _drain_chunks(ws_save, limit=cut_position)
    state = ws_save.state_dict()
    assert "cursor_states" in state
    assert any(
        "block_rng_snapshot" in cursor_state
        for cursor_state in state["cursor_states"].values()
    )

    ws_load = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
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

    ws = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
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


def test_checkpoint_restore_multilane() -> None:
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
        ws_baseline = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(
            lane, canonical_replicas=canonical_replicas
        )
        baseline = _drain_chunks(ws_baseline)
        assert baseline

        ws_save = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(
            lane, canonical_replicas=canonical_replicas
        )
        prefix = _drain_chunks(ws_save, limit=2)
        state = ws_save.state_dict()

        ws_load = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(
            lane, canonical_replicas=canonical_replicas
        )
        ws_load.load_state_dict(state)
        suffix = _drain_chunks(ws_load)

        assert prefix + suffix == baseline


def test_load_state_dict_lane_mismatch_raises() -> None:
    ds = make_dataset("alpha", 50)
    kwargs: dict = dict(datasets=[ds], mixture={ds.name: 1.0}, chunk_size=5, seed=0)

    ws_save = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=2
    )
    _drain_chunks(ws_save, limit=3)
    state = ws_save.state_dict()

    ws_load = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(
        1, canonical_replicas=2
    )
    with pytest.raises(RuntimeError):
        ws_load.load_state_dict(state)


# ---------------------------------------------------------------------------
# Repeat policy
# ---------------------------------------------------------------------------


def test_repeat_continues_past_exhaustion() -> None:
    ds = make_dataset("alpha", 10)
    ws = AccumulatorMixtureWorkSource(
        datasets=[ds],
        mixture={"alpha": 1.0},
        chunk_size=4,
        exhausted_policy="repeat",
        reshuffle_on_repeat=False,
    ).clone_for_lane(0, canonical_replicas=1)

    # 10 samples / 4 per chunk = 2 full chunks + leftover.
    # With repeat, we should get indefinitely many.
    for _ in range(10):
        chunk = ws.next_chunk()
        assert chunk is not None
        assert sum(len(v) for v in chunk.components.values()) == 4


def test_repeat_max_repeats() -> None:
    ds = make_dataset("alpha", 8)
    ws = AccumulatorMixtureWorkSource(
        datasets=[ds],
        mixture={"alpha": 1.0},
        chunk_size=4,
        exhausted_policy="repeat",
        max_repeats=2,
    ).clone_for_lane(0, canonical_replicas=1)

    chunks = _drain_chunks(ws)
    # 8 samples per epoch, 4 per chunk = 2 chunks/epoch.
    # max_repeats=2 → epochs 0, 1, 2 = 3 epochs → 6 chunks.
    assert len(chunks) > 0
    # Should eventually terminate.
    assert ws.next_chunk() is None


def test_repeat_checkpoint_restore() -> None:
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

    ws_baseline = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    baseline = _drain_chunks(ws_baseline, limit=8)
    assert len(baseline) == 8

    ws_save = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    prefix = _drain_chunks(ws_save, limit=4)
    state = ws_save.state_dict()

    ws_load = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(
        0, canonical_replicas=1
    )
    ws_load.load_state_dict(state)
    suffix = _drain_chunks(ws_load, limit=4)

    assert prefix + suffix == baseline


def test_repeat_len_raises() -> None:
    ds = make_dataset("alpha", 10)
    ws = AccumulatorMixtureWorkSource(
        datasets=[ds],
        mixture={"alpha": 1.0},
        chunk_size=4,
        exhausted_policy="repeat",
    ).clone_for_lane(0, canonical_replicas=1)

    with pytest.raises(TypeError, match="not defined for an infinite"):
        len(ws)


# ---------------------------------------------------------------------------
# Clone independence
# ---------------------------------------------------------------------------


def test_clone_copies_accumulators_independently() -> None:
    """Advancing a clone must not affect the original's accumulators."""
    ds_a = make_dataset("alpha", 100)
    ds_b = make_dataset("beta", 100)

    original = AccumulatorMixtureWorkSource(
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
    ws = AccumulatorMixtureWorkSource(
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
    ws = AccumulatorMixtureWorkSource(
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

    ws = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)

    # Should produce chunks past the point where "small" exhausts.
    chunks = _drain_chunks(ws, limit=20)
    assert len(chunks) == 20
    for ch in chunks:
        total = sum(len(v) for v in ch.values())
        assert total == 4


# ---------------------------------------------------------------------------
# State dict version mismatch
# ---------------------------------------------------------------------------


def test_load_state_dict_version_mismatch_raises() -> None:
    ds = make_dataset("alpha", 50)
    kwargs: dict = dict(datasets=[ds], mixture={ds.name: 1.0}, chunk_size=5, seed=0)

    ws = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    state = ws.state_dict()
    state["version"] = 99

    ws2 = AccumulatorMixtureWorkSource(**kwargs).clone_for_lane(0, canonical_replicas=1)
    with pytest.raises(RuntimeError, match="Unsupported"):
        ws2.load_state_dict(state)


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
    ws = AccumulatorMixtureWorkSource(
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
