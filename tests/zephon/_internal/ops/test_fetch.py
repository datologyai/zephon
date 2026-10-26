# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from zephon._internal.ops import fetch as fetch_module
from zephon._internal.ops.fetch import FetchOp
from zephon._internal.stream import EngineSample
from zephon.io import Dataset, StoreOptions
from zephon.ops.base import OpContext, StageInfo


def _noop(*args, **kwargs) -> None:  # pragma: no cover - trivial helper
    return None


def _setup_op(
    op: FetchOp,
    ctx_data: dict[str, object],
    *,
    collect_stats: bool = False,
    copies_mapped_arrays: bool = False,
) -> FetchOp:
    ctx = {
        "record_node_metrics": _noop,
        "emit_fetch_metrics": _noop,
        **ctx_data,
    }
    op.setup(
        OpContext(
            ctx,
            StageInfo(
                stage_index=0,
                stage_name="stage0",
                op_index=0,
                collect_stats=collect_stats,
                copies_mapped_arrays=copies_mapped_arrays,
            ),
        )
    )
    return op


@pytest.fixture()
def jsonl_dataset(tmp_path: Path) -> Dataset:
    shard0 = tmp_path / "shard0.jsonl"
    shard1 = tmp_path / "shard1.jsonl"
    shard0.write_text(
        "\n".join(json.dumps({"text": f"sample {i}", "value": i}) for i in range(3)),
        encoding="utf-8",
    )
    shard1.write_text(
        "\n".join(json.dumps({"text": f"sample {i}", "value": i}) for i in range(3, 5)),
        encoding="utf-8",
    )
    return Dataset.from_path("demo", str(tmp_path))


def make_engine_sample(
    dataset_id: int,
    shard_id: int,
    sample_idx: int,
    lane_id: int = 0,
    chunk_id: int = 0,
    chunk_offset: int | None = None,
    component_id: int = 0,
) -> EngineSample:
    """Helper to build EngineSample tuples for direct FetchOp invocation in tests."""
    if chunk_offset is None:
        chunk_offset = sample_idx
    return (
        (dataset_id, shard_id, sample_idx),
        lane_id,
        chunk_id,
        chunk_offset,
        component_id,
    )


def test_fetch_op_reads_jsonl(jsonl_dataset: Dataset) -> None:
    op = _setup_op(FetchOp(), {"datasets_by_id": {0: jsonl_dataset}})
    record = op.process_one(make_engine_sample(0, 0, 1))[0]
    assert record.payload["value"] == 1
    batch = op.process_many(
        [
            make_engine_sample(0, 1, 0, lane_id=0, chunk_id=0),
            make_engine_sample(0, 1, 1, lane_id=0, chunk_id=0),
        ]
    )
    assert [elem.payload["value"] for elem in batch] == [3, 4]


def test_fetch_op_reads_with_cache(jsonl_dataset: Dataset, tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    op = _setup_op(
        FetchOp(),
        {
            "datasets_by_id": {0: jsonl_dataset},
            "io_options": {"cache": {"enabled": True, "root": cache_root}},
        },
    )
    _ = op.process_many(
        [
            make_engine_sample(0, 0, 0, lane_id=0, chunk_id=0),
            make_engine_sample(0, 0, 1, lane_id=0, chunk_id=0),
        ]
    )
    assert cache_root.exists()


def test_fetch_op_setup_requires_datasets() -> None:
    op = FetchOp()
    with pytest.raises(RuntimeError):
        op.setup(
            OpContext(
                {"record_node_metrics": _noop, "emit_fetch_metrics": _noop},
                StageInfo(
                    stage_index=0,
                    stage_name="stage0",
                    op_index=0,
                    collect_stats=False,
                ),
            )
        )


def test_fetch_op_preserves_order_across_groups(jsonl_dataset: Dataset) -> None:
    op = _setup_op(FetchOp(), {"datasets_by_id": {0: jsonl_dataset}})
    # Interleave two shard groups; order should be preserved
    batch = [
        make_engine_sample(0, 1, 1),  # value 4
        make_engine_sample(0, 0, 0),  # value 0
        make_engine_sample(0, 1, 0),  # value 3
        make_engine_sample(0, 0, 2),  # value 2
    ]
    out = op.process_many(batch)
    assert [r.payload["value"] for r in out] == [4, 0, 3, 2]


def test_fetch_op_traits_and_accumulator() -> None:
    op = FetchOp(max_batch=32, max_latency_ms=10)
    t = op.traits()

    assert t.indexable is True and t.parallelism == 4

    # Test deterministic mode disables time-based flushing
    acc_det = op.accumulator(deterministic=True, ctx={})
    assert acc_det._max_batch == 32
    assert acc_det._max_latency_ms is None

    # Test non-deterministic mode preserves latency config
    acc_nondet = op.accumulator(deterministic=False, ctx={})
    assert acc_nondet._max_batch == 32
    assert acc_nondet._max_latency_ms == 10


@pytest.mark.parametrize(
    ("copies_mapped_arrays", "vortex_options", "safe_mmap"),
    [
        (False, {}, True),
        (True, {}, False),
        (False, {"safe_mmap": False}, False),
        (True, {"safe_mmap": True}, False),
    ],
)
@pytest.mark.parametrize("memory_map", [False, True])
def test_fetch_op_copies_mapped_arrays_unless_the_runner_copies(
    jsonl_dataset: Dataset,
    monkeypatch: pytest.MonkeyPatch,
    copies_mapped_arrays: bool,
    vortex_options: dict[str, object],
    safe_mmap: bool,
    memory_map: bool,
) -> None:
    options: list[StoreOptions] = []

    def record_store(_datasets: object, **kwargs: StoreOptions) -> object:
        options.append(kwargs["options"])
        return object()

    monkeypatch.setattr(fetch_module, "build_multi_dataset_store", record_store)
    _setup_op(
        FetchOp(),
        {
            "datasets_by_id": {0: jsonl_dataset},
            "io_options": {"vortex": {"memory_map": memory_map, **vortex_options}},
        },
        copies_mapped_arrays=copies_mapped_arrays,
    )

    assert options[0].vortex.safe_mmap is safe_mmap
    # The runner's copy changes only safe_mmap.
    assert options[0].vortex.memory_map is memory_map
