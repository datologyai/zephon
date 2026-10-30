# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint participation and aggregation for local engine snapshots."""

import warnings
from collections.abc import Callable
from typing import Any

from zephon._internal.coordination import AggregationRound, CheckpointCoordinator


def requires_aggregation(local: dict[str, Any]) -> bool:
    """Whether this snapshot owns lanes that need other participants."""
    world = local["world"]
    return bool(local["progress"]) and (
        int(world["world_size"]) != 1
        or len(local["progress"]) != int(world["canonical_replicas"])
    )


def aggregate_checkpoint(
    local: dict[str, Any],
    coordinator: CheckpointCoordinator | None,
    *,
    worker_id: int,
    merge: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    """Validate a sole owner's snapshot or aggregate all active owners."""
    world = local["world"]
    canonical_replicas = int(world["canonical_replicas"])
    if not requires_aggregation(local):
        if local["progress"]:
            check_mid_window_counts(local.get("lane_emitted") or {}, canonical_replicas)
        return local

    assert coordinator is not None
    return coordinator.aggregate(
        AggregationRound(
            local=local,
            is_leader=int(world["global_rank"]) == 0 and worker_id == 0,
            expected_ranks=frozenset(range(int(world["world_size"]))),
            expected_lanes=frozenset(range(canonical_replicas)),
        ),
        merge=merge,
    )


def complete_lane_counts(
    lane_emitted: dict[int, int], canonical_replicas: int
) -> list[int] | None:
    """Counts for every canonical lane, or None if any lane is unknown."""
    counts: list[int] = []
    for lane in range(canonical_replicas):
        count = lane_emitted.get(lane)
        if count is None:
            return None
        counts.append(count)
    return counts


def check_mid_window_counts(
    lane_emitted: dict[int, int], canonical_replicas: int
) -> None:
    """Warn when known delivery counts differ across canonical lanes.

    Partial coverage means another shard holds the rest, or the counters are
    unknown because the run was resumed from a pre-counter checkpoint.
    """
    counts = complete_lane_counts(lane_emitted, canonical_replicas)
    if counts is None:
        return
    lo, hi = min(counts), max(counts)
    if lo == hi:
        return
    warnings.warn(
        f"[zephon] Checkpoint taken mid-window: emitted-batch counts per "
        f"canonical lane are unequal (range [{lo}..{hi}] across "
        f"{canonical_replicas} lanes). Resuming this checkpoint into a "
        f"DIFFERENT topology will permute the remainder of the current "
        f"global window (no samples are lost or duplicated). To avoid "
        f"this, checkpoint at window boundaries: the number of global "
        f"batches per pooled step must be a multiple of "
        f"canonical_replicas ({canonical_replicas}).",
        RuntimeWarning,
        stacklevel=3,
    )
