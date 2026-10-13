"""KVCoordinator over real TCPStore processes and the engine's merge path."""

import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

from tests.helpers.engine import make_engine, seed_inflight_state
from tests.helpers.kvstore import InMemoryKVStore
from zephon._internal.checkpoint import AggregationCodec
from zephon._internal.coordination import (
    AggregationRound,
    FileCoordinator,
    KVCoordinator,
    RoundCoordinator,
    make_coordinator,
)
from zephon.options import RuntimeOptions

pytestmark = pytest.mark.integration

# Two MP ranks share lanes 0 and 1; each rank splits them across two workers.
_PARTICIPANTS = [(rank, worker) for rank in range(2) for worker in range(2)]
_ROUNDS = 3


def _participant_proc(rank: int, worker: int, port: int, out: Any) -> None:
    coordinator = make_coordinator(
        RuntimeOptions(
            aggregate_store=f"torch://127.0.0.1:{port}",
            run_id="kv-integration",
            aggregate_serializer="msgpack",
            aggregate_timeout_s=60.0,
        ),
        global_rank=rank,
        world_size=2,
        plan_id="test-plan",
        worker_info=lambda: (worker, 2),
    )
    results = []
    for step in range(_ROUNDS):
        results.append(
            coordinator.aggregate(
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
                lambda shards: {
                    "progress": {k: v for s in shards for k, v in s["progress"].items()}
                },
            )
        )
    coordinator.cleanup()
    out.put(((rank, worker), results))


def test_rounds_over_tcpstore_processes() -> None:
    dist = pytest.importorskip("torch.distributed")
    # The server must outlive every participant's reads of the result.
    server = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = {
        participant: ctx.Process(
            target=_participant_proc, args=(*participant, server.port, out)
        )
        for participant in _PARTICIPANTS
    }
    for proc in procs.values():
        proc.start()
    results: dict[tuple[int, int], list[Any]] = {}
    try:
        deadline = time.monotonic() + 120
        while len(results) < len(procs):
            try:
                participant, values = out.get(timeout=1)
                results[participant] = values
            except Empty:
                crashed = {
                    p: proc.exitcode
                    for p, proc in procs.items()
                    if p not in results and proc.exitcode not in (None, 0)
                }
                assert not crashed, f"Participants crashed: {crashed}"
                assert time.monotonic() < deadline, "Participants timed out"
    finally:
        for proc in procs.values():
            proc.join(timeout=30)
            if proc.is_alive():
                proc.kill()
    expected = [{"progress": {0: step, 1: step}} for step in range(_ROUNDS)]
    assert results == dict.fromkeys(_PARTICIPANTS, expected)
    assert all(proc.exitcode == 0 for proc in procs.values())
    del server


def _run_round(
    coordinators: list[RoundCoordinator], locals_: list[dict[str, Any]], merge: Any
) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=len(coordinators)) as pool:
        futures = [
            pool.submit(
                coordinator.aggregate,
                AggregationRound(
                    local=local,
                    is_leader=rank == 0,
                    expected_ranks=frozenset(range(len(coordinators))),
                    expected_lanes=frozenset(range(4)),
                ),
                merge,
            )
            for rank, (coordinator, local) in enumerate(zip(coordinators, locals_))
        ]
        return [future.result(timeout=30) for future in futures]


@pytest.mark.parametrize("serializer", ["json", "msgpack"])
def test_kv_merged_state_matches_file_coordinator_bytes(
    tmp_path: Path, serializer: str
) -> None:
    world_size = 2
    engines = [
        make_engine(
            canonical_replicas=4,
            world_size=world_size,
            global_rank=rank,
            dp_degree=world_size,
            dp_group_id=rank,
            aggregate_dir=str(tmp_path / "engine"),
        )
        for rank in range(world_size)
    ]
    for engine in engines:
        seed_inflight_state(engine)
    locals_ = [engine._state_dict_local() for engine in engines]
    merge = engines[0]._merge_state_dicts
    opts = RuntimeOptions(
        run_id="parity",
        aggregate_dir=str(tmp_path / "files"),
        aggregate_serializer=serializer,
        aggregate_compressor="none",
        aggregate_timeout_s=30.0,
    )
    common: dict[str, Any] = {
        "world_size": world_size,
        "plan_id": "test-plan",
        "worker_info": lambda: (0, 1),
    }
    store = InMemoryKVStore()
    file_coordinators: list[RoundCoordinator] = [
        FileCoordinator(opts, global_rank=rank, **common) for rank in range(world_size)
    ]
    kv_coordinators: list[RoundCoordinator] = [
        KVCoordinator(opts, store, global_rank=rank, **common)
        for rank in range(world_size)
    ]
    try:
        file_results = _run_round(file_coordinators, locals_, merge)
        kv_results = _run_round(kv_coordinators, locals_, merge)
    finally:
        for engine in engines:
            engine.close()

    codec = AggregationCodec(serializer=serializer, compressor="none")
    file_bytes = {codec.encode(result) for result in file_results}
    assert len(file_bytes) == 1
    assert {codec.encode(result) for result in kv_results} == file_bytes
