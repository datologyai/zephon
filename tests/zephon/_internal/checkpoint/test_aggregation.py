"""Checkpoint participation and round setup."""

from typing import Any
from unittest.mock import Mock

import pytest

from zephon._internal.checkpoint._aggregation import aggregate_checkpoint


@pytest.mark.parametrize("owns_lanes", [False, True])
def test_local_checkpoint_needs_no_coordinator(owns_lanes: bool) -> None:
    local: dict[str, Any] = {
        "world": {"world_size": 1 if owns_lanes else 2, "canonical_replicas": 2},
        "progress": {0: {}, 1: {}} if owns_lanes else {},
        "lane_emitted": {0: 3, 1: 3},
    }
    merge = Mock()
    assert aggregate_checkpoint(local, None, worker_id=0, merge=merge) is local
    merge.assert_not_called()


def test_sole_owner_checks_window_counts() -> None:
    local = {
        "world": {"world_size": 1, "canonical_replicas": 2},
        "progress": {0: {}, 1: {}},
        "lane_emitted": {0: 3, 1: 4},
    }
    with pytest.warns(RuntimeWarning, match="Checkpoint taken mid-window"):
        assert aggregate_checkpoint(local, None, worker_id=0, merge=Mock()) is local


@pytest.mark.parametrize(
    ("world_size", "rank", "worker", "is_leader"),
    [(2, 0, 0, True), (2, 1, 0, False), (1, 0, 1, False)],
)
def test_partial_owner_joins_the_round(
    world_size: int, rank: int, worker: int, is_leader: bool
) -> None:
    local = {
        "world": {
            "world_size": world_size,
            "canonical_replicas": 2,
            "global_rank": rank,
        },
        "progress": {worker: {}},
    }
    coordinator, merge = Mock(), Mock()
    merged = {"merged": True}
    coordinator.aggregate.return_value = merged

    assert (
        aggregate_checkpoint(local, coordinator, worker_id=worker, merge=merge)
        is merged
    )
    round_ = coordinator.aggregate.call_args.args[0]
    assert round_.local is local
    assert round_.is_leader is is_leader
    assert round_.expected_ranks == frozenset(range(world_size))
    assert round_.expected_lanes == frozenset({0, 1})
    assert coordinator.aggregate.call_args.kwargs["merge"] is merge
