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
        (
            32,
            8,
            8,
            1,
            "Use global_batch_size=64, canonical_replicas=4, or an output batch "
            "size of 4.",
        ),
        (
            48,
            4,
            8,
            2,
            "Use global_batch_size=32 or 64, canonical_replicas=6 or 12, or an "
            "output batch size of 3.",
        ),
        (12, 4, 2, 2, "Use global_batch_size=8 or 16, or an output batch size of 3."),
        # No lane count or batch size fits, so only the step size can change.
        (12, 4, 8, 2, "Use global_batch_size=32."),
        (
            192,
            4,
            64,
            16,
            "Use global_batch_size=256, canonical_replicas=48, or an output batch "
            "size of 3.",
        ),
        # Uneven lane ownership: only lane counts that also split the step.
        (40, 1, 5, 2, "Use canonical_replicas=4 or 8."),
        (
            48,
            8,
            6,
            4,
            "Use canonical_replicas=4 with global_batch_size=32 or 64, or "
            "canonical_replicas=8 with global_batch_size=64.",
        ),
        (
            24,
            4,
            6,
            4,
            "Use canonical_replicas=4 with global_batch_size=16 or 32, or "
            "canonical_replicas=8 with global_batch_size=32.",
        ),
    ],
)
def test_lane_window_check_names_layouts_that_fit(
    records: int, batch: int, replicas: int, dp: int, hint: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(hint)):
        check_step_lane_windows(records, batch, replicas, dp)


def test_lane_window_check_rejects_uneven_lane_ownership_even_if_the_step_splits() -> (
    None
):
    with pytest.raises(ValueError, match="not a multiple of dp_degree=4"):
        check_step_lane_windows(48, 8, 6, 4)


def _suggested_layouts(
    message: str, records: int, batch: int, replicas: int
) -> list[tuple[int, int, int]]:
    hint = message.split(" Use ", 1)[1].removesuffix(".")
    layouts = []
    for option in re.split(r", or |, ", hint):
        if match := re.fullmatch(r"an output batch size of (\d+)", option):
            layouts.append((records, int(match[1]), replicas))
        elif match := re.fullmatch(
            r"canonical_replicas=(\d+) with global_batch_size=(.+)", option
        ):
            layouts += [
                (int(size), batch, int(match[1])) for size in match[2].split(" or ")
            ]
        elif match := re.fullmatch(r"canonical_replicas=(.+)", option):
            layouts += [(records, batch, int(r)) for r in match[1].split(" or ")]
        elif match := re.fullmatch(r"global_batch_size=(.+)", option):
            layouts += [(int(size), batch, replicas) for size in match[1].split(" or ")]
        else:
            raise AssertionError(f"unparsed option {option!r} in {message!r}")
    return layouts


def test_every_suggested_layout_passes_the_check() -> None:
    for dp in range(1, 7):
        for replicas in range(dp, 17):
            for batch in range(1, 9):
                for records in range(batch, 97, batch):
                    try:
                        check_step_lane_windows(records, batch, replicas, dp)
                    except ValueError as exc:
                        layouts = _suggested_layouts(str(exc), records, batch, replicas)
                    else:
                        continue
                    assert layouts
                    for layout in layouts:
                        check_step_lane_windows(*layout, dp)


def test_lane_window_check_rejects_partial_batches() -> None:
    with pytest.raises(ValueError, match="global_batch_size=30 must be a multiple"):
        check_step_lane_windows(30, 4, 1, 1)
