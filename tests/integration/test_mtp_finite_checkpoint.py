"""MTP exhaustion and cleanup do not start collective checkpoint rounds."""

import multiprocessing
from pathlib import Path
from typing import Any

import pytest

from zephon import Dataset, InMemoryShard, Pipeline, StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def fail_one_document(payload: Any) -> Any:
    if payload["input_ids"][0] == 4:
        raise RuntimeError("intentional source failure")
    return payload


def drop_one_document(payload: Any) -> Any:
    # Lanes own disjoint chunk pairs. Dropping one lane-0 document makes
    # emitted batches uneven without changing source topology between ranks.
    return None if payload["input_ids"][0] == 5 else payload


def make_pipeline(
    rank: int, world_size: int, directory: str, fail: bool = False
) -> Pipeline:
    documents = [{"input_ids": [i, i + 1]} for i in range(9)]
    dataset = Dataset.from_dict("tiny", {0: InMemoryShard(documents)})
    source = StaticMixtureWorkSource(
        [dataset],
        {"tiny": 1.0},
        chunk_size=2,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipeline = Pipeline(source)
    pipeline.map_transform(fail_one_document if fail else drop_one_document)
    return pipeline.batch(1).options(
        mtp_mode=True,
        runner="inline",
        deterministic=True,
        shm_enabled=False,
        dp_degree=world_size,
        dp_group_id=rank,
        canonical_replicas=world_size,
        world_size=world_size,
        global_rank=rank,
        aggregate_dir=directory,
        run_id="finite-mtp",
        aggregate_timeout_s=5,
        max_workers=1,
    )


def run_rank(
    rank: int, directory: str, init_file: str, fail: bool, output: Any
) -> None:
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    pipeline = make_pipeline(rank, 2, directory, fail)
    try:
        # DCP obtains a destination template before iteration. Its parent-side
        # engine must not shadow the subsequent MTP stream's terminal state.
        pipeline.checkpoint()
        iterator = iter(pipeline)
        steps = 0
        while True:
            status = 2
            try:
                next(iterator)
            except StopIteration:
                pipeline.close()
                status = 1
            except Exception as exc:
                assert fail and "intentional source failure" in str(exc)
                pipeline.close()
                status = 0
            available = torch.tensor(status)
            dist.all_reduce(available, op=dist.ReduceOp.MIN)
            status = int(available)
            if status != 2:
                break
            steps += 1
        if fail:
            assert status == 0
            pipeline.close()
            dist.barrier()
            output.put((rank, "source-error", steps))
            return
        assert status == 1
        assert steps == 3
        # One rank has already closed, while the other may still be live.
        checkpoint = pipeline.checkpoint()
        pipeline.close()
        # Repeated explicit rounds must stay aligned after local final capture.
        checkpoint = pipeline.checkpoint()
        restored = make_pipeline(rank, 2, directory)
        restored.restore(checkpoint)
        remainder = list(restored)
        restored.close()
        assert remainder == []
        dist.barrier()
        output.put((rank, "exhausted", steps))
    finally:
        pipeline.close()
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "fail", [False, True], ids=["uneven-exhaustion", "one-rank-source-error"]
)
def test_distributed_mtp_finite_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    # Free-threaded PyTorch's Gloo startup can deadlock independently of Zephon.
    monkeypatch.setenv("PYTHON_GIL", "1")
    import sys

    if sys.platform == "darwin":
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0")
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    children = [
        context.Process(
            target=run_rank,
            args=(
                rank,
                str(tmp_path / "aggregate"),
                str(tmp_path / "gloo"),
                fail,
                output,
            ),
        )
        for rank in range(2)
    ]
    for child in children:
        child.start()
    try:
        for child in children:
            child.join(timeout=45)
            assert child.exitcode == 0
        results = [output.get(timeout=5) for _ in children]
        assert {result[0] for result in results} == {0, 1}
        assert {result[1] for result in results} == (
            {"source-error"} if fail else {"exhausted"}
        )
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
        output.close()
        output.join_thread()
