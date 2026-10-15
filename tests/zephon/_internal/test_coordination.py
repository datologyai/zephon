import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import replace
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tests.helpers.kvstore import InMemoryKVStore
from zephon._internal import kvstore
from zephon._internal.coordination import (
    AggregationRound,
    FileCoordinator,
    KVCoordinator,
    RoundCoordinator,
    make_coordinator,
)
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
    run_id: str = "coordination-test",
) -> FileCoordinator:
    return FileCoordinator(
        RuntimeOptions(
            aggregate_dir=str(tmp_path),
            run_id=run_id,
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


def test_leader_waits_for_contributors_and_readers_before_cleanup(
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
            # glob normalizes both ./ and //; participant identity must not
            # depend on the raw spelling of the run directory.
            run_id="./team//exp",
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
    three_contributors = Event()
    release_publish = Event()
    put = coordinators[-1].backend.put
    list_shards = coordinators[0]._list_state_files

    def delayed_publish(path: str, data: bytes) -> None:
        if path.endswith(".ckpt"):
            assert release_publish.wait(5), "Test did not release the delayed writer"
        put(path, data)

    def observe_shards(round_id: str) -> list[str]:
        paths = list_shards(round_id)
        if len(paths) == 3:
            three_contributors.set()
        return paths

    monkeypatch.setattr(coordinators[-1].backend, "put", delayed_publish)
    monkeypatch.setattr(coordinators[0], "_list_state_files", observe_shards)
    read_result = coordinators[-1]._read_aggregation_state

    def delayed_read(path: str) -> dict | None:
        reading.set()
        assert release.wait(5), "Test did not release the delayed reader"
        return read_result(path)

    monkeypatch.setattr(coordinators[-1], "_read_aggregation_state", delayed_read)
    expected = {"progress": {"0": 7, "1": 7}}
    merge = Mock(return_value=expected)
    read_shard = Mock(wraps=coordinators[0]._read)
    monkeypatch.setattr(coordinators[0], "_read", read_shard)

    def aggregate(index: int) -> dict[str, Any]:
        coordinator = coordinators[index]
        result = coordinator.aggregate(rounds[index], merge)
        if index == 0:
            coordinator.cleanup()  # Simulate immediate leader shutdown.
        return result

    with ThreadPoolExecutor(max_workers=len(participants)) as pool:
        futures = [pool.submit(aggregate, i) for i in range(len(participants))]
        try:
            assert three_contributors.wait(5)
            with pytest.raises(FutureTimeoutError):
                futures[0].result(timeout=0.1)
            merge.assert_not_called()
            release_publish.set()
            assert reading.wait(5)
            for future in futures[1:-1]:
                assert future.result(timeout=5) == expected
            with pytest.raises(FutureTimeoutError):
                futures[0].result(timeout=0.1)
            assert list(Path(coordinators[0].agg_dir).glob("merged_*.ckpt"))
        finally:
            release_publish.set()
            release.set()
        assert [future.result(timeout=5) for future in futures] == [expected] * len(
            participants
        )

    merge.assert_called_once()
    contributions = {
        (shard["world"]["global_rank"], int(lane))
        for shard in merge.call_args.args[0]
        for lane in shard["progress"]
    }
    assert contributions == {(0, 0), (0, 1), (1, 0), (1, 1)}
    # Each follower's shard is read once; the leader uses its own directly.
    assert read_shard.call_count == 3
    assert not list(Path(coordinators[0].agg_dir).iterdir())


@pytest.mark.parametrize("failure", ["decode", "acknowledge"])
def test_unacknowledged_result_survives_timeout_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    leader = _coordinator(tmp_path, world_size=3)
    follower = _coordinator(tmp_path, rank=1, world_size=3)
    healthy_follower = _coordinator(tmp_path, rank=2, world_size=3)
    follower_failed = Event()
    wait_for_acks = leader._wait_for_acks
    wait_until = leader._wait_until

    def wait_for_failed_reader(*args: Any) -> None:
        assert follower_failed.wait(5)
        wait_for_acks(*args)

    def wait_for_readers(pred: Callable[[], bool], timeout: float) -> bool:
        # Exercise ack timeout without making discovery depend on startup latency.
        return wait_until(pred, 0.01 if follower_failed.is_set() else timeout)

    monkeypatch.setattr(leader, "_wait_for_acks", wait_for_failed_reader)
    monkeypatch.setattr(leader, "_wait_until", wait_for_readers)
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
                        "mapping": {0: [0], 1: [1], 2: [2]},
                    },
                    "progress": {rank: 7},
                },
                is_leader=rank == 0,
                expected_ranks=frozenset({0, 1, 2}),
                expected_lanes=frozenset({0, 1, 2}),
            ),
            lambda _: {"progress": {"0": 7, "1": 7, "2": 7}},
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        leader_result = pool.submit(aggregate, leader, 0)
        follower_result = pool.submit(aggregate, follower, 1)
        healthy_result = pool.submit(aggregate, healthy_follower, 2)
        with pytest.raises(error_type, match=error_message):
            follower_result.result(timeout=5)
        assert healthy_result.result(timeout=5)
        follower_failed.set()
        with pytest.raises(
            RuntimeError, match="acknowledgements: missing=1.*state_r1_w0.*retained at"
        ):
            leader_result.result(timeout=5)

    leader.cleanup()
    leader.cleanup()
    merged_files = list(Path(leader.agg_dir).glob("merged_*.ckpt"))
    assert len(merged_files) == 1
    assert leader._read_aggregation_state(str(merged_files[0])) == {
        "progress": {"0": 7, "1": 7, "2": 7}
    }
    assert not list(Path(leader.agg_dir).glob("*.ack"))
    assert not list(Path(leader.agg_dir).glob("failed_*.txt"))


def test_cleanup_errors_do_not_fail_acknowledged_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    coordinators = [
        _coordinator(tmp_path, rank=rank, world_size=2) for rank in range(2)
    ]
    for coordinator in coordinators:
        delete = coordinator.backend.delete

        def fail_cleanup(path: str, delete: Callable[[str], None] = delete) -> None:
            if not path.endswith("round.current"):
                raise OSError("Cleanup unavailable")
            delete(path)

        monkeypatch.setattr(coordinator.backend, "delete", fail_cleanup)

    expected = {"progress": {"0": 7, "1": 7}}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                coordinator.aggregate,
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
                lambda _: expected,
            )
            for rank, coordinator in enumerate(coordinators)
        ]
        assert [future.result(timeout=5) for future in futures] == [expected] * 2
    assert "Cleanup unavailable" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["merge", "notification", "published_write"])
def test_leader_failure_notification_preserves_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    leader = _coordinator(tmp_path, world_size=2)
    follower = _coordinator(tmp_path, rank=1, world_size=2, timeout=0.05)
    rounds = [
        AggregationRound(
            {
                "world": {
                    "global_rank": rank,
                    "dp_group_id": rank,
                    "mapping": {0: [0], 1: [1]},
                },
                "progress": {rank: 0},
            },
            rank == 0,
            frozenset({0, 1}),
            frozenset({0, 1}),
        )
        for rank in range(2)
    ]
    for coordinator in (leader, follower):
        monkeypatch.setattr(coordinator, "_open_round_id", lambda *_: "failed-round")
    follower.backend.put(
        follower._state_file_path("failed-round"),
        follower._codec.encode(rounds[1].local),
    )
    put = leader.backend.put
    original = ValueError("Inconsistent snapshots")
    expected = {"progress": {"0": 0, "1": 0}}

    def publish(path: str, data: bytes) -> None:
        if failure == "notification" and path.endswith(".txt"):
            raise OSError("Notification unavailable")
        put(path, data)
        if failure == "published_write" and Path(path).name.startswith("merged_"):
            # A storage write can succeed remotely and still report an error.
            raise original

    monkeypatch.setattr(leader.backend, "put", publish)
    merge = Mock(
        return_value=expected,
        side_effect=original if failure != "published_write" else None,
    )
    with pytest.raises(ValueError) as error:
        leader.aggregate(rounds[0], merge)
    assert error.value is original
    leader.cleanup()
    if failure == "notification":
        assert not list(Path(leader.agg_dir).glob("failed_*.txt"))
    elif failure == "published_write":
        # A failure notice must not revoke an already published result.
        assert list(Path(leader.agg_dir).glob("failed_*.txt"))
        assert follower.aggregate(rounds[1], Mock()) == expected
    else:
        with pytest.raises(
            RuntimeError, match="failed on leader: ValueError: Inconsistent snapshots"
        ):
            follower.aggregate(rounds[1], Mock())


def test_waits_check_readiness_without_remaining_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = _coordinator(tmp_path)
    sleep = Mock(side_effect=AssertionError("An expired wait must not sleep"))
    monkeypatch.setattr(
        "zephon._internal.coordination.time",
        SimpleNamespace(monotonic=lambda: 0.0, sleep=sleep),
    )
    ready = Mock(side_effect=[True, False])
    value = Mock(side_effect=[0, None])
    assert coordinator._wait_until(ready, 0)
    assert coordinator._wait_value(value, 0) == 0
    assert not coordinator._wait_until(ready, 0)
    assert coordinator._wait_value(value, 0) is None
    assert ready.call_count == value.call_count == 2
    sleep.assert_not_called()


def test_contribution_and_ack_waits_share_one_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leader = _coordinator(tmp_path, world_size=2, timeout=1.0)
    follower = _coordinator(tmp_path, rank=1, world_size=2)
    monkeypatch.setattr(leader, "_open_round_id", lambda *_: "timed-round")
    world = {"global_rank": 0, "dp_group_id": 0, "mapping": {0: [0], 1: [1]}}
    shard_path = follower._state_file_path("timed-round")
    elapsed = 0.0

    def advance_time(delay: float) -> None:
        nonlocal elapsed
        elapsed += delay
        if elapsed >= 0.6:
            follower.backend.put(
                shard_path,
                follower._codec.encode(
                    {
                        "world": {**world, "global_rank": 1, "dp_group_id": 1},
                        "progress": {1: 0},
                    }
                ),
            )
        if elapsed >= 1.2:
            follower.backend.put(f"{shard_path}.ack", b"")

    monkeypatch.setattr(
        "zephon._internal.coordination.time",
        SimpleNamespace(monotonic=lambda: elapsed, sleep=advance_time),
    )
    with pytest.raises(
        RuntimeError,
        match=r"after 1\.0s \(aggregate_timeout_s budget: 1\.0s\) waiting for checkpoint acknowledgements: missing=1",
    ):
        leader.aggregate(
            AggregationRound(
                {"world": world, "progress": {0: 0}},
                True,
                frozenset({0, 1}),
                frozenset({0, 1}),
            ),
            lambda _: {"progress": {"0": 0, "1": 0}},
        )
    assert elapsed == pytest.approx(1.0)
    leader.cleanup()
    assert Path(leader._merged_file_path("timed-round")).exists()
    assert not list(Path(leader.agg_dir).glob("failed_*.txt"))


def test_timeout_reports_elapsed_time_beyond_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leader = _coordinator(tmp_path, world_size=2, timeout=1.0)
    follower = _coordinator(tmp_path, rank=1, world_size=2)
    monkeypatch.setattr(leader, "_open_round_id", lambda *_: "slow-round")
    world = {"global_rank": 0, "dp_group_id": 0, "mapping": {0: [0], 1: [1]}}
    follower.backend.put(
        follower._state_file_path("slow-round"),
        follower._codec.encode(
            {"world": {**world, "global_rank": 1, "dp_group_id": 1}, "progress": {1: 0}}
        ),
    )
    elapsed = 0.0

    def slow_merge(_: list[dict[str, Any]]) -> dict[str, Any]:
        # Merge work is not interrupted by the budget.
        nonlocal elapsed
        elapsed += 5.0
        return {"progress": {"0": 0, "1": 0}}

    monkeypatch.setattr(
        "zephon._internal.coordination.time",
        SimpleNamespace(monotonic=lambda: elapsed, sleep=lambda _: None),
    )
    with pytest.raises(
        RuntimeError,
        match=r"after 5\.0s \(aggregate_timeout_s budget: 1\.0s\) waiting for checkpoint acknowledgements",
    ):
        leader.aggregate(
            AggregationRound(
                {"world": world, "progress": {0: 0}},
                True,
                frozenset({0, 1}),
                frozenset({0, 1}),
            ),
            slow_merge,
        )


@pytest.mark.parametrize(
    "world,progress,error",
    [
        (
            {"mapping": {0: [1], 1: [0]}},
            {0: 0},
            "rank 0's shard.*differs from the leader's.*groups=\\[0, 1\\]",
        ),
        ({"dp_group_id": 2}, {0: 0}, "Unknown DP group 2"),
        ({"dp_group_id": 1}, {1: 0}, "DP group mismatch.*0 != 1"),
        ({}, {2: 0}, "Unexpected lanes.*\\[2\\]"),
    ],
)
def test_invalid_shard_fails_without_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    world: dict[str, Any],
    progress: dict[int, int],
    error: str,
) -> None:
    coordinator = _coordinator(tmp_path)
    monkeypatch.setattr(coordinator, "_open_round_id", lambda *_: "test-round")
    local = {
        "world": {"global_rank": 0, "dp_group_id": 0, "mapping": {0: [0], 1: [1]}},
        "progress": {0: 0},
    }
    shard = {"world": {**local["world"], **world}, "progress": progress}
    coordinator.backend.put(
        f"{coordinator.agg_dir}/state_r0_w1_p0_test-round.ckpt",
        coordinator._codec.encode(shard),
    )
    merge = Mock()
    with pytest.raises(RuntimeError, match=error):
        coordinator.aggregate(
            AggregationRound(local, True, frozenset({0}), frozenset({0, 1})), merge
        )
    merge.assert_not_called()


def test_timeout_reports_missing_lane_on_model_parallel_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = _coordinator(tmp_path, world_size=2, timeout=0.01)
    monkeypatch.setattr(coordinator, "_open_round_id", lambda *_: "test-round")
    world = {"global_rank": 0, "dp_group_id": 0, "mapping": {0: [0, 1]}}
    coordinator.backend.put(
        f"{coordinator.agg_dir}/state_r1_w0_p0_test-round.ckpt",
        coordinator._codec.encode(
            {"world": {**world, "global_rank": 1}, "progress": {0: 0}}
        ),
    )
    with pytest.raises(
        RuntimeError,
        match=r"Missing ranks=\[\]; missing lanes=\[\]; missing lanes per rank=\{1: \[1\]\}",
    ):
        coordinator.aggregate(
            AggregationRound(
                {"world": world, "progress": {0: 0, 1: 0}},
                True,
                frozenset({0, 1}),
                frozenset({0, 1}),
            ),
            Mock(),
        )


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


class _DictCoordinator(RoundCoordinator):
    """Round protocol over a shared dict, standing in for a key-value store.

    Like a key-value backend, it can't list acks and keeps the result and a
    failure notice in one conditionally written outcome value.
    """

    def __init__(
        self, store: dict[str, bytes], lock: Lock, *, rank: int, worker_id: int
    ) -> None:
        super().__init__(
            RuntimeOptions(run_id="dict", aggregate_compressor="none"),
            global_rank=rank,
            world_size=2,
            plan_id="test-plan",
            worker_info=lambda: (worker_id, 2),
            timeout_s=5.0,
            poll_base_s=0.001,
            poll_backoff=False,
        )
        self._store, self._lock = store, lock
        self._decode = self._codec.decode
        self._participant = f"r{rank}_w{worker_id}_p{os.getpid()}"

    @property
    def _announcement_location(self) -> str:
        return f"{self._scoped_run_id()}/round"

    def _shard_location(self, round_id: str) -> str:
        return f"{self._scoped_run_id()}/{round_id}/state/{self._participant}"

    def _result_location(self, round_id: str) -> str:
        return f"{self._scoped_run_id()}/{round_id}/outcome"

    def _failure_location(self, round_id: str) -> str:
        return self._result_location(round_id)

    def _publish_outcome(self, round_id: str, value: bytes) -> None:
        with self._lock:
            self._store.setdefault(self._result_location(round_id), value)

    def _publish_result(self, round_id: str, data: bytes) -> None:
        self._publish_outcome(round_id, b"R" + data)

    def _publish_failure(self, round_id: str, message: str) -> None:
        self._publish_outcome(round_id, b"F" + message.encode())

    def _wait_for_outcome(self, round_id: str, timeout: float) -> bool:
        outcome = self._wait_value(
            lambda: self._read(self._result_location(round_id)), timeout
        )
        if outcome is not None and outcome[:1] == b"F":
            raise RuntimeError(f"failed on leader: {outcome[1:].decode()}")
        return outcome is not None

    def _read_result(self, round_id: str) -> dict | None:
        outcome = self._read(self._result_location(round_id))
        return self._decode(outcome[1:]) if outcome else None

    def _ack_location(self, shard_location: str) -> str:
        return shard_location.replace("/state/", "/ack/")

    def _ack_name(self, shard_location: str) -> str:
        return shard_location.rsplit("/", 1)[1]

    def _shard_rank(self, shard_location: str) -> int:
        return int(self._ack_name(shard_location).split("_")[0][1:])

    def _acknowledged(self, round_id: str, pending: set[str]) -> set[str]:
        prefix = f"{self._scoped_run_id()}/{round_id}/ack/"
        with self._lock:
            return {name for name in pending if prefix + name in self._store}

    def _list_shards(self, round_id: str) -> list[str]:
        prefix = f"{self._scoped_run_id()}/{round_id}/state/"
        with self._lock:
            return [k for k in self._store if k.startswith(prefix)]

    def _write_announcement(self, round_id: str) -> None:
        self._put(self._announcement_location, round_id.encode())

    def _read_announcement(self) -> tuple[str, float] | None:
        raw = self._read(self._announcement_location)
        return None if raw is None else (raw.decode(), 0.0)

    def _put(self, location: str, data: bytes) -> None:
        with self._lock:
            self._store[location] = data

    def _read(self, location: str) -> bytes | None:
        with self._lock:
            return self._store.get(location)

    def _exists(self, location: str) -> bool:
        with self._lock:
            return location in self._store

    def _delete(self, location: str) -> None:
        with self._lock:
            self._store.pop(location, None)


def test_round_protocol_runs_over_non_file_storage() -> None:
    # Two MP ranks share one DP group; two workers per rank split its lanes.
    store: dict[str, bytes] = {}
    lock = Lock()
    participants = [(rank, worker) for rank in range(2) for worker in range(2)]
    coordinators = [
        _DictCoordinator(store, lock, rank=rank, worker_id=worker)
        for rank, worker in participants
    ]
    merge = Mock(
        side_effect=lambda shards: {
            "progress": {k: v for s in shards for k, v in s["progress"].items()}
        }
    )
    with ThreadPoolExecutor(max_workers=len(participants)) as pool:
        for step in range(2):
            futures = [
                pool.submit(
                    coordinator.aggregate,
                    AggregationRound(
                        local={
                            "world": {
                                "global_rank": rank,
                                "dp_group_id": 0,
                                "mapping": {0: [0, 1]},
                            },
                            "progress": {worker: step},
                        },
                        is_leader=rank == worker == 0,
                        expected_ranks=frozenset({0, 1}),
                        expected_lanes=frozenset({0, 1}),
                    ),
                    merge,
                )
                for coordinator, (rank, worker) in zip(coordinators, participants)
            ]
            expected = {"progress": {0: step, 1: step}}
            assert [f.result(timeout=10) for f in futures] == [expected] * 4
            assert len(merge.call_args.args[0]) == 4
            # Only the retained result of this round remains.
            round_id = coordinators[0].last_round_id
            assert list(store) == [f"dict-0/{round_id}/outcome"]

    coordinators[0].cleanup()
    assert store == {}


def test_failure_notice_never_revokes_a_published_outcome() -> None:
    leader = _DictCoordinator({}, Lock(), rank=0, worker_id=0)
    leader._publish_result("r", leader._codec.encode({"progress": {0: 1}}))
    leader._publish_failure("r", "late error")
    assert leader._wait_for_outcome("r", 0)
    assert leader._read_result("r") == {"progress": {0: 1}}


# ---------------------------------------------------------------------------
# Behavior shared by FileCoordinator and KVCoordinator
# ---------------------------------------------------------------------------


@pytest.fixture(params=["file", "kv"])
def participant(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    """Build coordinators for one test's participants on the chosen backend."""
    store = InMemoryKVStore()

    def make(
        *,
        rank: int = 0,
        worker_id: int = 0,
        workers_per_rank: int = 1,
        world_size: int = 1,
        timeout: float = 5.0,
    ) -> RoundCoordinator:
        if request.param == "file":
            return _coordinator(
                tmp_path,
                rank=rank,
                world_size=world_size,
                worker_id=worker_id,
                workers_per_rank=workers_per_rank,
                timeout=timeout,
            )
        return KVCoordinator(
            RuntimeOptions(
                run_id="coordination-test",
                aggregate_serializer="json",
                aggregate_compressor="none",
                aggregate_timeout_s=timeout,
            ),
            store,
            global_rank=rank,
            world_size=world_size,
            plan_id="test-plan",
            worker_info=lambda: (worker_id, workers_per_rank),
        )

    make.store = store  # type: ignore[attr-defined]
    make.backend = request.param  # type: ignore[attr-defined]
    return make


def _round(
    rank: int, lanes: list[int], mapping: dict[int, list[int]], *, leader: bool
) -> AggregationRound:
    groups = {lane: group for group, members in mapping.items() for lane in members}
    return AggregationRound(
        local={
            "world": {
                "global_rank": rank,
                "dp_group_id": groups[lanes[0]],
                "mapping": mapping,
            },
            "progress": dict.fromkeys(lanes, 7),
        },
        is_leader=leader,
        expected_ranks=frozenset({0, 1}),
        expected_lanes=frozenset(groups),
    )


def _merge_progress(shards: list[dict[str, Any]]) -> dict[str, Any]:
    return {"progress": {k: v for s in shards for k, v in s["progress"].items()}}


def test_rounds_with_uneven_workers_across_model_parallel_ranks(
    participant: Any,
) -> None:
    # Ranks 0 and 1 are MP peers sharing lanes 0 and 1. Rank 0 splits them
    # across two active workers (a third configured worker is idle); rank 1
    # has one worker owning both.
    mapping = {0: [0, 1]}
    members = [
        (participant(rank=0, worker_id=0, workers_per_rank=3, world_size=2), 0, [0]),
        (participant(rank=0, worker_id=1, workers_per_rank=3, world_size=2), 0, [1]),
        (participant(rank=1, world_size=2), 1, [0, 1]),
    ]
    merge = Mock(side_effect=_merge_progress)
    with ThreadPoolExecutor(max_workers=len(members)) as pool:
        for step in range(3):
            futures = [
                pool.submit(
                    coordinator.aggregate,
                    _round(rank, lanes, mapping, leader=index == 0),
                    merge,
                )
                for index, (coordinator, rank, lanes) in enumerate(members)
            ]
            results = [future.result(timeout=10) for future in futures]
            assert results == [{"progress": {"0": 7, "1": 7}}] * len(members)
            assert merge.call_count == step + 1
            assert len(merge.call_args.args[0]) == len(members)
    if participant.backend == "kv":
        # Only the last round's retained outcome remains in the store.
        assert [key.rsplit("/", 1)[1] for key in participant.store.data] == ["outcome"]
    members[0][0].cleanup()
    if participant.backend == "kv":
        assert participant.store.data == {}


def test_leader_failure_reaches_followers(participant: Any) -> None:
    mapping = {0: [0], 1: [1]}
    leader = participant(world_size=2)
    follower = participant(rank=1, world_size=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        led = pool.submit(
            leader.aggregate,
            _round(0, [0], mapping, leader=True),
            Mock(side_effect=ValueError("Inconsistent snapshots")),
        )
        followed = pool.submit(
            follower.aggregate, _round(1, [1], mapping, leader=False), Mock()
        )
        with pytest.raises(ValueError, match="Inconsistent snapshots"):
            led.result(timeout=10)
        with pytest.raises(
            RuntimeError, match="failed on leader: ValueError: Inconsistent snapshots"
        ):
            followed.result(timeout=10)


def test_unacknowledged_result_is_retained(
    participant: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = {0: [0], 1: [1]}
    leader = participant(world_size=2, timeout=1.0)
    follower = participant(rank=1, world_size=2, timeout=1.0)
    monkeypatch.setattr(follower, "_read_result", lambda _round_id: None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        led = pool.submit(
            leader.aggregate, _round(0, [0], mapping, leader=True), _merge_progress
        )
        followed = pool.submit(
            follower.aggregate, _round(1, [1], mapping, leader=False), Mock()
        )
        with pytest.raises(RuntimeError, match="Failed to read merged checkpoint"):
            followed.result(timeout=10)
        with pytest.raises(RuntimeError, match="acknowledgements: missing=1"):
            led.result(timeout=10)
    leader.cleanup()
    assert leader._read_result(leader.last_round_id) == {"progress": {"0": 7, "1": 7}}


def test_completed_stale_round_is_not_joined(participant: Any) -> None:
    # A previous attempt left a fresh announcement and its result behind.
    previous = participant(world_size=2)
    previous._write_announcement("old")
    previous._publish_result("old", previous._codec.encode({"progress": {}}))
    follower = participant(rank=1, world_size=2, timeout=0.2)
    with pytest.raises(RuntimeError, match="waiting for round id"):
        follower.aggregate(_round(1, [1], {0: [0], 1: [1]}, leader=False), Mock())


def test_coverage_timeout_reports_missing_rank(participant: Any) -> None:
    leader = participant(world_size=2, timeout=0.05)
    merge = Mock()
    with pytest.raises(RuntimeError, match=r"Missing ranks=\[1\]"):
        leader.aggregate(_round(0, [0], {0: [0], 1: [1]}, leader=True), merge)
    merge.assert_not_called()


# ---------------------------------------------------------------------------
# Coordinator selection
# ---------------------------------------------------------------------------


def test_make_coordinator_selects_by_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        kvstore._REGISTRY, "memory", lambda _uri, _timeout_s: InMemoryKVStore()
    )
    common: dict[str, Any] = {
        "global_rank": 0,
        "world_size": 2,
        "plan_id": "test-plan",
        "worker_info": lambda: (0, 1),
    }
    assert isinstance(
        make_coordinator(RuntimeOptions(aggregate_dir=str(tmp_path)), **common),
        FileCoordinator,
    )
    assert isinstance(
        make_coordinator(RuntimeOptions(aggregate_store="memory://"), **common),
        KVCoordinator,
    )
    with pytest.raises(ValueError, match="at most one"):
        make_coordinator(
            RuntimeOptions(aggregate_dir=str(tmp_path), aggregate_store="memory://"),
            **common,
        )
    with pytest.raises(RuntimeError, match="aggregate_store"):
        make_coordinator(RuntimeOptions(), **common)


def test_shard_from_wrong_rank_is_rejected(participant: Any) -> None:
    # A rank-1 participant publishes a shard that claims to be rank 0's.
    mapping = {0: [0], 1: [1]}
    leader = participant(world_size=2, timeout=2.0)
    impostor = participant(rank=1, world_size=2, timeout=0.5)
    merge = Mock()
    with ThreadPoolExecutor(max_workers=2) as pool:
        led = pool.submit(leader.aggregate, _round(0, [0], mapping, leader=True), merge)
        pool.submit(impostor.aggregate, _round(0, [1], mapping, leader=False), Mock())
        with pytest.raises(
            RuntimeError, match="global_rank=0 but was published by rank 1"
        ):
            led.result(timeout=10)
    merge.assert_not_called()


def test_cleanup_in_forked_child_keeps_the_result(
    participant: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    leader = participant()
    solo = AggregationRound(
        local={
            "world": {"global_rank": 0, "dp_group_id": 0, "mapping": {0: [0]}},
            "progress": {0: 7},
        },
        is_leader=True,
        expected_ranks=frozenset({0}),
        expected_lanes=frozenset({0}),
    )
    leader.aggregate(solo, _merge_progress)
    round_id = leader.last_round_id
    pid = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: pid + 1)
    leader.cleanup()
    monkeypatch.undo()
    assert leader._read_result(round_id) == {"progress": {"0": 7}}


_SMALL_WRITE_BYTES = 64


class _RecordingStore(InMemoryKVStore):
    """Small writes, so shards and results are chunked; records every key."""

    max_value_bytes = _SMALL_WRITE_BYTES

    def __init__(self) -> None:
        super().__init__()
        self.written: set[str] = set()

    def _set(self, key: str, value: bytes) -> None:
        assert len(value) <= _SMALL_WRITE_BYTES
        self.written.add(key)
        super()._set(key, value)


def test_kv_rounds_move_values_larger_than_one_write() -> None:
    store = _RecordingStore()
    blob = "x" * 500

    def make(rank: int, worker_id: int) -> KVCoordinator:
        return KVCoordinator(
            RuntimeOptions(
                run_id="chunked",
                aggregate_serializer="json",
                aggregate_compressor="none",
            ),
            store,
            global_rank=rank,
            world_size=2,
            plan_id="test-plan",
            worker_info=lambda: (worker_id, 2),
        )

    def merge(shards: list[dict[str, Any]]) -> dict[str, Any]:
        return {**_merge_progress(shards), "blob": blob}

    mapping = {0: [0, 1]}
    members = [
        (make(rank, worker), rank, [worker]) for rank in (0, 1) for worker in (0, 1)
    ]
    with ThreadPoolExecutor(max_workers=len(members)) as pool:
        for _ in range(3):
            futures = [
                pool.submit(
                    coordinator.aggregate,
                    replace(
                        _round(rank, lanes, mapping, leader=index == 0),
                        local={
                            **_round(rank, lanes, mapping, leader=False).local,
                            "blob": blob,
                        },
                    ),
                    merge,
                )
                for index, (coordinator, rank, lanes) in enumerate(members)
            ]
            results = [future.result(timeout=10) for future in futures]
            assert results == [{"progress": {"0": 7, "1": 7}, "blob": blob}] * 4
    assert any("/state/" in key and "/chunk/" in key for key in store.written)
    assert any("/outcome/" in key and "/chunk/" in key for key in store.written)
    members[0][0].cleanup()
    assert store.data == {}


def test_engine_checks_coordination_storage_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.helpers.engine import make_engine

    check_ready = Mock()

    class _Store(InMemoryKVStore):
        def check_ready(self) -> None:
            check_ready()

    monkeypatch.setitem(kvstore._REGISTRY, "memory", lambda _uri, _t: _Store())
    make_engine(aggregate_store="memory://", canonical_replicas=2)
    check_ready.assert_not_called()
    make_engine(
        aggregate_store="memory://", canonical_replicas=2, world_size=2, global_rank=0
    )
    check_ready.assert_called_once()

    check_ready.side_effect = RuntimeError("store is down")
    with pytest.raises(RuntimeError, match="store is down"):
        make_engine(
            aggregate_store="memory://",
            canonical_replicas=2,
            world_size=2,
            global_rank=1,
        )


def test_torch_store_endpoint_is_needed_only_by_coordinating_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch.distributed")
    from tests.helpers.engine import make_engine

    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)
    # A single active worker on a single rank never coordinates.
    make_engine(aggregate_store="torch://", canonical_replicas=2).close()
    with pytest.raises(ValueError, match="MASTER_ADDR/MASTER_PORT are not set"):
        make_engine(
            aggregate_store="torch://",
            canonical_replicas=2,
            world_size=2,
            global_rank=1,
        )


def test_kv_coordinator_tracks_only_chunked_values() -> None:
    store = InMemoryKVStore()
    leader, follower = (
        KVCoordinator(
            RuntimeOptions(run_id="tracking", aggregate_compressor="none"),
            store,
            global_rank=rank,
            world_size=2,
            plan_id="test-plan",
            worker_info=lambda: (0, 1),
        )
        for rank in (0, 1)
    )
    mapping = {0: [0], 1: [1]}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in range(3):
            futures = [
                pool.submit(
                    c.aggregate, _round(r, [r], mapping, leader=r == 0), _merge_progress
                )
                for r, c in enumerate((leader, follower))
            ]
            [future.result(timeout=10) for future in futures]
    # Nothing here exceeds one write, so no process keeps any bookkeeping.
    assert leader._chunks == {} and follower._chunks == {}
