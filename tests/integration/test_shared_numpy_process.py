"""Resolved NumPy inputs stay shared across real process-runner queues."""

from typing import Any

import pytest

from zephon._internal.graph import Node, Stage
from zephon._internal.runners.process import ProcessStageRunner
from zephon.ops.base import BaseOp
from zephon.ops.traits import OpTraits
from zephon.types import SampleMeta, SampleRecord

pytestmark = pytest.mark.integration
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


class _UpdateSharedArray(BaseOp):
    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True)

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        for record in elems:
            array: Any = record.payload
            owner = array
            while isinstance(owner, np.ndarray):
                owner = owner.base
            assert isinstance(owner, torch.Tensor) and owner.is_shared()
            array[0] += 1
        return elems


def test_process_dispatch_and_output_reuse_numpy_storage() -> None:
    source = torch.arange(32).share_memory_()
    array = source.numpy()[3:20:2]
    record = SampleRecord(SampleMeta((0, 0, 0), 0, 0), array)
    stage = Stage(
        "shared_numpy",
        [
            Node("first", _UpdateSharedArray(), parallelism=1),
            Node("second", _UpdateSharedArray(), parallelism=1),
        ],
        "auto",
        "test",
    )
    runner = ProcessStageRunner(
        stage,
        {"record_node_metrics": lambda _: None},
        max_workers=2,
        deterministic=True,
        stage_output_mode="stream_items",
    )
    try:
        [result] = list(runner.run([record]))
        assert isinstance(result, SampleRecord)
        assert source[3].item() == 5
        assert record.payload is array
        result_array: Any = result.payload
        assert result_array.strides == array.strides
        np.testing.assert_array_equal(result_array, array)
    finally:
        runner.close()
