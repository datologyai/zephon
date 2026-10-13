from concurrent.futures import ThreadPoolExecutor, TimeoutError
from io import BytesIO
from pathlib import Path
from threading import Event
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
    mapping = {0: [0], 1: [1]} if world_size == 2 else {0: [0, 1]}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for step in range(2):
                rounds = [
                    AggregationRound(
                        local={
                            "world": {
                                "global_rank": rank,
                                "dp_group_id": rank,
                                "mapping": mapping,
                            },
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


def test_aggregate_waits_for_lane_coverage_on_every_rank(
    tmp_path: Path,
) -> None:
    # Model-parallel ranks share two lanes, split between workers 0 and 1.
    # Worker 2 is inactive and must not be needed to finish the round.
    coordinator = _coordinator(
        tmp_path, world_size=2, workers_per_rank=3, serializer="msgpack"
    )
    shards = [
        {
            "world": {
                "global_rank": rank,
                "dp_group_id": 0,
                "mapping": {0: [0, 1]},
            },
            "progress": {worker: 7},
        }
        for rank in range(2)
        for worker in range(2)
    ]
    payloads = {
        f"shard-{i}": coordinator._codec.encode(shard) for i, shard in enumerate(shards)
    }

    def open_shard(path: str, mode: str) -> BytesIO:
        assert mode == "rb"
        return BytesIO(payloads[path])

    backend = Mock()
    backend.exists.side_effect = lambda path: path.endswith(".ack")
    backend.open.side_effect = open_shard
    # The first poll sees every rank and lane globally, but rank 1's lane-1
    # worker has not published yet. Its shard appears on the following poll.
    backend.glob.side_effect = [list(payloads)[:3], list(payloads)]
    coordinator.backend = backend
    merge = Mock(return_value={"progress": {"0": 7, "1": 7}})

    result = coordinator.aggregate(
        AggregationRound(
            local=shards[0],
            is_leader=True,
            expected_ranks=frozenset({0, 1}),
            expected_lanes=frozenset({0, 1}),
        ),
        merge,
    )

    merge.assert_called_once()
    contributions = {
        (shard["world"]["global_rank"], int(lane))
        for shard in merge.call_args.args[0]
        for lane in shard["progress"]
    }
    assert contributions == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert result == {"progress": {"0": 7, "1": 7}}


def test_leader_waits_for_readers_before_return_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both MP ranks own both lanes. Only two of three configured workers
    # participate on each rank; every active worker must acknowledge.
    participants = [(rank, worker) for rank in range(2) for worker in range(2)]
    coordinators = [
        _coordinator(
            tmp_path,
            rank=rank,
            world_size=2,
            worker_id=worker,
            workers_per_rank=3,
        )
        for rank, worker in participants
    ]
    rounds = [
        AggregationRound(
            local={
                "world": {
                    "global_rank": rank,
                    "dp_group_id": 0,
                    "mapping": {0: [0, 1]},
                },
                "progress": {worker: 7},
            },
            is_leader=rank == worker == 0,
            expected_ranks=frozenset({0, 1}),
            expected_lanes=frozenset({0, 1}),
        )
        for rank, worker in participants
    ]
    reading = Event()
    release = Event()
    read_result = coordinators[-1]._read_aggregation_state

    def delayed_read(path: str) -> dict | None:
        reading.set()
        assert release.wait(5), "Test did not release the delayed reader"
        return read_result(path)

    monkeypatch.setattr(coordinators[-1], "_read_aggregation_state", delayed_read)
    expected = {"progress": {"0": 7, "1": 7}}

    def aggregate(index: int) -> dict[str, Any]:
        coordinator = coordinators[index]
        result = coordinator.aggregate(rounds[index], lambda _: expected)
        if index == 0:
            coordinator.cleanup()  # Simulate immediate leader shutdown.
        return result

    with ThreadPoolExecutor(max_workers=len(participants)) as pool:
        futures = [pool.submit(aggregate, i) for i in range(len(participants))]
        try:
            assert reading.wait(5)
            for future in futures[1:-1]:
                assert future.result(timeout=5) == expected
            with pytest.raises(TimeoutError):
                futures[0].result(timeout=0.1)
            assert list(Path(coordinators[0].agg_dir).glob("merged_*.ckpt"))
        finally:
            release.set()
        assert [future.result(timeout=5) for future in futures] == [expected] * len(
            participants
        )

    assert not list(Path(coordinators[0].agg_dir).iterdir())


@pytest.mark.parametrize("failure", ["decode", "acknowledge"])
def test_unacknowledged_result_survives_timeout_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    leader = _coordinator(tmp_path, world_size=2, timeout=0.5)
    follower = _coordinator(tmp_path, rank=1, world_size=2)
    if failure == "decode":
        monkeypatch.setattr(
            follower._codec, "decode", Mock(side_effect=ValueError("Invalid result"))
        )
        error_type, error_message = RuntimeError, "Failed to read merged checkpoint"
    else:
        put = follower.backend.put

        def fail_acknowledgement(path: str, data: bytes) -> None:
            if path.endswith(".ack"):
                raise OSError("Acknowledgement write failed")
            put(path, data)

        monkeypatch.setattr(follower.backend, "put", fail_acknowledgement)
        error_type, error_message = OSError, "Acknowledgement write failed"

    def aggregate(coordinator: FileCoordinator, rank: int) -> dict[str, Any]:
        return coordinator.aggregate(
            AggregationRound(
                local={
                    "world": {
                        "global_rank": rank,
                        "dp_group_id": rank,
                        "mapping": {0: [0], 1: [1]},
                    },
                    "progress": {rank: 7},
                },
                is_leader=rank == 0,
                expected_ranks=frozenset({0, 1}),
                expected_lanes=frozenset({0, 1}),
            ),
            lambda _: {"progress": {"0": 7, "1": 7}},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        leader_result = pool.submit(aggregate, leader, 0)
        follower_result = pool.submit(aggregate, follower, 1)
        with pytest.raises(error_type, match=error_message):
            follower_result.result(timeout=5)
        with pytest.raises(RuntimeError, match="acknowledgements.*state_r1_w0"):
            leader_result.result(timeout=5)

    leader.cleanup()
    leader.cleanup()
    merged_files = list(Path(leader.agg_dir).glob("merged_*.ckpt"))
    assert len(merged_files) == 1
    assert leader._read_aggregation_state(str(merged_files[0])) == {
        "progress": {"0": 7, "1": 7}
    }
    assert not list(Path(leader.agg_dir).glob("*.ack"))


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
                local={
                    "world": {
                        "global_rank": 0,
                        "dp_group_id": 0,
                        "mapping": {0: sorted(lanes)},
                    },
                    "progress": {0: 0},
                },
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
            local={
                "world": {
                    "global_rank": 0,
                    "dp_group_id": 0,
                    "mapping": {0: [0]},
                },
                "progress": {0: 7},
            },
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
