from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from zephon._internal.coordination import AggregationRound, FileCoordinator
from zephon.options import RuntimeOptions


def _coordinator(
    tmp_path: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    worker_id: int = 0,
    workers_per_rank: int = 1,
    serializer: str = "json",
    timeout: float = 5.0,
) -> FileCoordinator:
    return FileCoordinator(
        RuntimeOptions(
            aggregate_dir=str(tmp_path),
            run_id="coordination-test",
            aggregate_serializer=serializer,
            aggregate_compressor="none",
            aggregate_timeout_s=timeout,
        ),
        global_rank=rank,
        world_size=world_size,
        plan_id="test-plan",
        worker_info=lambda: (worker_id, workers_per_rank),
    )


@pytest.mark.parametrize("serializer", ["json", "msgpack"])
@pytest.mark.parametrize("world_size,workers_per_rank", [(2, 1), (1, 2)])
def test_aggregate_returns_same_result_across_rounds(
    tmp_path: Path, serializer: str, world_size: int, workers_per_rank: int
) -> None:
    participants = [
        (rank, worker)
        for rank in range(world_size)
        for worker in range(workers_per_rank)
    ]
    coordinators = [
        _coordinator(
            tmp_path,
            rank=rank,
            world_size=world_size,
            worker_id=worker,
            workers_per_rank=workers_per_rank,
            serializer=serializer,
        )
        for rank, worker in participants
    ]

    def merge(shards: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "progress": {k: v for shard in shards for k, v in shard["progress"].items()}
        }

    leader_merge = Mock(side_effect=merge)
    follower_merge = Mock(side_effect=AssertionError("Only the leader merges"))
    previous_round_id = None
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for step in range(2):
                rounds = [
                    AggregationRound(
                        local={
                            "world": {"global_rank": rank},
                            "progress": {lane: step},
                        },
                        is_leader=lane == 0,
                        expected_ranks=frozenset(range(world_size)),
                        expected_lanes=frozenset({0, 1}),
                    )
                    for lane, (rank, _) in enumerate(participants)
                ]
                futures = [
                    pool.submit(
                        coordinator.aggregate,
                        round_,
                        leader_merge if round_.is_leader else follower_merge,
                    )
                    for coordinator, round_ in zip(coordinators, rounds)
                ]
                results = [future.result(timeout=10) for future in futures]
                keys = ["0", "1"] if serializer == "json" else [0, 1]
                assert results == [{"progress": dict.fromkeys(keys, step)}] * 2
                assert leader_merge.call_count == step + 1
                assert len(leader_merge.call_args.args[0]) == 2
                follower_merge.assert_not_called()
                round_id = coordinators[0].last_round_id
                assert round_id is not None and round_id != previous_round_id
                assert coordinators[1].last_round_id == round_id
                previous_round_id = round_id
    finally:
        for coordinator in coordinators:
            coordinator.cleanup()


@pytest.mark.parametrize(
    "ranks,lanes,missing",
    [
        ({0, 1}, {0}, r"Missing ranks=\[1\]"),
        ({0}, {0, 1}, r"missing lanes=\[1\]"),
    ],
)
def test_aggregate_times_out_without_required_coverage(
    tmp_path: Path, ranks: set[int], lanes: set[int], missing: str
) -> None:
    coordinator = _coordinator(tmp_path, timeout=0.01)
    merge = Mock()
    with pytest.raises(RuntimeError, match=missing):
        coordinator.aggregate(
            AggregationRound(
                local={"world": {"global_rank": 0}, "progress": {0: 0}},
                is_leader=True,
                expected_ranks=frozenset(ranks),
                expected_lanes=frozenset(lanes),
            ),
            merge,
        )
    merge.assert_not_called()


def test_prepare_restore_isolates_round_namespace(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    previous_dir = coordinator.agg_dir
    coordinator.backend.put(coordinator.round_file, b"stale-round")
    coordinator.last_round_id = "saved-round"
    coordinator.reload_count = 1

    coordinator.prepare_restore()
    coordinator.prepare_restore()

    assert coordinator.agg_dir != previous_dir
    assert Path(coordinator.agg_dir).is_dir()
    assert not Path(coordinator.round_file).exists()
    assert coordinator.last_round_id == "saved-round"
    assert coordinator.reload_count == 1


def test_cleanup_removes_retained_result_idempotently(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    result = {"progress": {"0": 7}}
    coordinator.aggregate(
        AggregationRound(
            local={"world": {"global_rank": 0}, "progress": {0: 7}},
            is_leader=True,
            expected_ranks=frozenset({0}),
            expected_lanes=frozenset({0}),
        ),
        lambda _: result,
    )
    merged_files = list(Path(coordinator.agg_dir).glob("merged_*.ckpt"))
    assert len(merged_files) == 1
    other_run = tmp_path / "other-run" / "merged.ckpt"
    coordinator.backend.put(str(other_run), b"other checkpoint")

    coordinator.cleanup()
    coordinator.cleanup()

    assert not merged_files[0].exists()
    assert other_run.read_bytes() == b"other checkpoint"
