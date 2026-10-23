import re

import pytest

from zephon._internal.world import World, check_step_lane_windows


def test_world_defaults() -> None:
    w = World()
    assert w.canonical_replicas == 1
    assert w.worker_id == 0
    assert w.workers_per_rank == 1
    assert w.global_rank == 0
    assert w.world_size == 1
    assert w.dp_degree == 1
    assert w.dp_group_id == 0
    assert w.lanes_for_dp_group == {0: [0]}


@pytest.mark.parametrize(
    "records,batch,replicas,dp,hint",
    [
        (32, 8, 8, 1, "Use canonical_replicas=4, or an output batch size of 4."),
        (48, 4, 8, 2, "Use canonical_replicas=6 or 12, or an output batch size of 3."),
        (12, 4, 2, 2, "a multiple of 8, or an output batch size of 3."),
        # Every lane fits the step, but four DP groups cannot share six lanes.
        (24, 4, 6, 4, "Use a global_batch_size that is a multiple of 16."),
    ],
)
def test_lane_window_check_names_layouts_that_fit(
    records: int, batch: int, replicas: int, dp: int, hint: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(hint)):
        check_step_lane_windows(records, batch, replicas, dp)


def test_lane_window_check_rejects_partial_batches() -> None:
    with pytest.raises(ValueError, match="global_batch_size=30 must be a multiple"):
        check_step_lane_windows(30, 4, 1, 1)
