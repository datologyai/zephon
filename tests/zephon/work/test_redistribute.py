# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Small deterministic redistribution checks; token ratios are pinned, never measured."""

import math
from collections import Counter
from typing import Any

import pytest

from zephon.io import Dataset, InMemoryShard
from zephon.work import StaticMixtureWorkSource, WorkChunk
from zephon.work.static_mixture import LegacyFixedStrategy, TokenAwareStrategy
from zephon.work.token_estimation import TokenEstimation


def _source(
    mode: str,
    sizes: dict[str, int],
    weights: dict[str, float] | None = None,
    **kwargs: Any,
) -> StaticMixtureWorkSource:
    datasets = [
        Dataset.from_dict(
            name, {0: InMemoryShard([{"text": f"{name}-{i:03}"} for i in range(count)])}
        )
        for name, count in sizes.items()
    ]
    options: dict[str, Any] = {
        "chunk_size": 4,
        "exhausted_policy": "redistribute",
        "shuffle_shards": False,
        **kwargs,
    }
    if mode == "tokens":
        options.setdefault("token_estimation", TokenEstimation(primer=1.0))
    source = StaticMixtureWorkSource(
        datasets, weights or dict.fromkeys(sizes, 1.0), **options
    )
    if mode == "tokens":
        source.prime()
    return source


def _chunks(source: Any) -> list[WorkChunk]:
    # A hard cap turns any accidental nontermination into a small, fast failure.
    chunks = []
    for _ in range(100):
        chunk = source.next_chunk()
        if chunk is None:
            return chunks
        chunks.append(chunk)
    pytest.fail("finite redistribution did not terminate")


@pytest.mark.parametrize("mode", ["samples", "tokens"])
@pytest.mark.parametrize("chunk_size", [1, 4])
def test_redistribute_keeps_every_sample_and_publishes_survivor_targets(
    mode: str, chunk_size: int
) -> None:
    sizes = {"a": 3, "b": 9}
    template = _source(mode, sizes, chunk_size=chunk_size)
    source = template.clone_for_lane(0, 1)
    assert len(source) == sum(sizes.values())
    assert source.continues_after_exhaustion() == frozenset(sizes)
    chunks = _chunks(source)
    ids = [sid for c in chunks for samples in c.components.values() for sid in samples]
    assert len(ids) == len(set(ids)) == sum(sizes.values())
    assert Counter(sid[0] for sid in ids) == {0: 3, 1: 9}
    exhausted = set()
    for chunk in chunks:
        active = {name: 1.0 for name in sizes if name not in exhausted}
        assert chunk.target_mixture == pytest.approx(
            {n: w / len(active) for n, w in active.items()}
        )
        assert 0 < len(chunk) <= chunk_size
        assert exhausted <= set(chunk.source_exhausted)
        assert not exhausted.intersection(chunk.components)
        exhausted.update(chunk.source_exhausted)
    assert exhausted == set(sizes)
    assert source.component_ids() == {"a": 0, "b": 1}
    assert len(source) == 0
    done = source.state_dict()
    assert source.next_chunk() is None
    assert source.state_dict() == done
    # Resume at each boundary, including the exhausting chunk and terminal state.
    for cut in range(len(chunks) + 1):
        original = template.clone_for_lane(0, 1)
        for _ in range(cut):
            original.next_chunk()
        restored = template.clone_for_lane(0, 1)
        restored.load_state_dict(original.state_dict())
        assert [c.state_dict() for c in _chunks(restored)] == [
            c.state_dict() for c in chunks[cut:]
        ]


@pytest.mark.parametrize("mode", ["samples", "tokens"])
def test_survivor_allocation_preserves_sample_or_token_ratios(mode: str) -> None:
    options = (
        {"token_estimation": TokenEstimation(primer={"a": 1.0, "b": 2.0, "c": 4.0})}
        if mode == "tokens"
        else {}
    )
    source = _source(
        mode,
        {"a": 2, "b": 48, "c": 96},
        {"a": 0.5, "b": 0.3, "c": 0.2},
        chunk_size=10,
        **options,
    ).clone_for_lane(0, 1)
    boundary = source.next_chunk()
    assert boundary is not None and "a" in boundary.source_exhausted
    total = Counter()
    largest_cost = 1.0
    for _ in range(3):
        chunk = source.next_chunk()
        assert chunk is not None and len(chunk) == 10
        assert chunk.target_mixture == pytest.approx({"b": 0.6, "c": 0.4})
        for name, samples in chunk.components.items():
            for sid in samples:
                strategy = source._strategy
                cost = (
                    strategy._ensure_cost_table().cost(name, sid)
                    if isinstance(strategy, TokenAwareStrategy)
                    else 1.0
                )
                total[name] += cost
                largest_cost = max(largest_cost, cost)
    assert abs(total["b"] - 0.6 * total.total()) <= 2 * largest_cost


@pytest.mark.parametrize("mode", ["samples", "tokens"])
@pytest.mark.parametrize("assignment", ["modulo", "permute"])
def test_redistribute_lane_partition_and_resume(mode: str, assignment: str) -> None:
    template = _source(
        mode,
        {"a": 3, "b": 13},
        lane_assignment=assignment,
        shuffle_block_size=3,
        shuffle_within_shard=True,
    )
    all_ids = []
    for lane in range(2):
        source = template.clone_for_lane(lane, 2)
        chunks = _chunks(source)
        all_ids.extend(
            sid for c in chunks for samples in c.components.values() for sid in samples
        )
        prefix = template.clone_for_lane(lane, 2)
        assert prefix.next_chunk() is not None
        restored = template.clone_for_lane(lane, 2)
        restored.load_state_dict(prefix.state_dict())
        assert [c.state_dict() for c in _chunks(restored)] == [
            c.state_dict() for c in chunks[1:]
        ]
    assert len(all_ids) == len(set(all_ids)) == 16


@pytest.mark.parametrize("mode", ["samples", "tokens"])
@pytest.mark.parametrize("policy", ["stop", "repeat"])
def test_redistribute_retains_other_components_terminal_policy(
    mode: str, policy: str
) -> None:
    source = _source(
        mode,
        {"a": 2, "b": 8},
        exhausted_policy={"a": "redistribute", "b": policy},
        **({"max_repeats": {"b": 0}} if policy == "repeat" else {}),
    ).clone_for_lane(0, 1)
    assert source.continues_after_exhaustion() == frozenset({"a"})
    assert math.isfinite(source.total_samples)
    chunks = _chunks(source)
    counts = Counter(
        {n: sum(len(c.components.get(n, [])) for c in chunks) for n in ("a", "b")}
    )
    assert counts == {"a": 2, "b": 6}
    assert chunks[-1].target_mixture == {"b": 1.0}
    done = source.state_dict()
    assert source.next_chunk() is None
    assert source.state_dict() == done


@pytest.mark.parametrize("mode", ["samples", "tokens"])
def test_repeat_survivor_keeps_source_unbounded(mode: str) -> None:
    source = _source(
        mode, {"a": 2, "b": 8}, exhausted_policy={"a": "redistribute", "b": "repeat"}
    ).clone_for_lane(0, 1)
    assert math.isinf(source.total_samples)
    with pytest.raises(TypeError, match="infinite"):
        len(source)
    chunks = [source.next_chunk() for _ in range(5)]
    assert all(c is not None for c in chunks)
    assert all(
        c.target_mixture == {"b": 1.0} and c.source_exhausted == ("a",)
        for c in chunks[1:]
    )
    assert sum(len(c.components.get("a", [])) for c in chunks) == 2


@pytest.mark.parametrize("mode", ["samples", "tokens"])
def test_repeat_liveness_uses_survivor_demand(mode: str) -> None:
    with pytest.raises(ValueError, match="requires at least 4"):
        _source(
            mode,
            {"a": 2, "b": 3},
            {"a": 0.75, "b": 0.25},
            exhausted_policy={"a": "redistribute", "b": "repeat"},
        )


@pytest.mark.parametrize("mode", ["samples", "tokens"])
def test_empty_redistributing_component_is_announced(mode: str) -> None:
    source = _source(mode, {"a": 0, "b": 3}).clone_for_lane(0, 1)
    chunks = _chunks(source)
    assert sum(map(len, chunks)) == 3
    assert chunks[0].target_mixture == {"b": 1.0}
    assert chunks[0].source_exhausted == ("a", "b")


def test_legacy_quota_redistribution_restores_from_cursors() -> None:
    template = _source("samples", {"a": 3, "b": 9})
    template._strategy = LegacyFixedStrategy(template._alloc_config)
    source = template.clone_for_lane(0, 1)
    chunks = _chunks(source)
    assert [len(c) for c in chunks] == [4, 3, 4, 1]
    source = template.clone_for_lane(0, 1)
    source.next_chunk()
    source.next_chunk()
    restored = template.clone_for_lane(0, 1)
    restored.load_state_dict(source.state_dict())
    assert [c.state_dict() for c in _chunks(restored)] == [
        c.state_dict() for c in chunks[2:]
    ]


@pytest.mark.parametrize("mode", ["samples", "tokens"])
def test_mixed_stop_rolls_back_an_unfinished_chunk(mode: str) -> None:
    source = _source(
        mode, {"a": 5, "b": 1}, exhausted_policy={"a": "redistribute", "b": "stop"}
    ).clone_for_lane(0, 1)
    before = source.state_dict()
    assert source.next_chunk() is None
    assert source.state_dict() == before
    restored = _source(mode, {"a": 5, "b": 1}).clone_for_lane(0, 1)
    # Omitted policy adopts the checkpoint's policy; an explicit mismatch is rejected.
    with pytest.raises(RuntimeError, match="contradict"):
        restored.load_state_dict(before)


@pytest.mark.parametrize("mode", ["samples", "tokens"])
def test_redistribute_simultaneous_exhaustion(mode: str) -> None:
    source = _source(mode, {"a": 2, "b": 2}).clone_for_lane(0, 1)
    chunks = _chunks(source)
    assert len(chunks) == 1
    assert chunks[0].source_exhausted == ("a", "b")
    assert chunks[0].target_mixture == {"a": 0.5, "b": 0.5}
    assert len(source) == 0


def test_redistribute_discards_retired_fractional_debt() -> None:
    """Removing components must not leave a nonzero total rounding balance."""
    template = _source(
        "samples",
        {"a": 4, "b": 4, "c": 4, "d": 0, "e": 0, "f": 0, "g": 0, "h": 0},
        chunk_size=1,
    )
    source = template.clone_for_lane(0, 1)
    state = source.state_dict()
    # The allocator permits balances above one after a correction. This valid
    # zero-sum carry becomes unbalanced when the five depleted sources leave.
    state["accumulators"] = {
        "a": 1.5,
        "b": 1.5,
        "c": 1.5,
        "d": -0.9,
        "e": -0.9,
        "f": -0.9,
        "g": -0.9,
        "h": -0.9,
    }
    source.load_state_dict(state)
    chunks = _chunks(source)
    assert len(chunks) == 12
    assert all(len(chunk) == 1 for chunk in chunks)
    assert Counter(name for chunk in chunks for name in chunk.components) == {
        "a": 4,
        "b": 4,
        "c": 4,
    }


@pytest.mark.parametrize("mode", ["samples", "tokens"])
def test_mixed_empty_stop_component_still_stops_source(mode: str) -> None:
    source = _source(
        mode, {"a": 5, "b": 0}, exhausted_policy={"a": "redistribute", "b": "stop"}
    ).clone_for_lane(0, 1)
    assert len(source) == 0
    assert source.next_chunk() is None


@pytest.mark.parametrize("mode", ["samples", "tokens"])
def test_all_redistributing_components_initially_empty(mode: str) -> None:
    source = _source(mode, {"a": 0, "b": 0}).clone_for_lane(0, 1)
    assert len(source) == 0
    assert source.next_chunk() is None
    checkpoint = source.state_dict()
    source.load_state_dict(checkpoint)
    assert source.next_chunk() is None
    assert source.state_dict() == checkpoint
