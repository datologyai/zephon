# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Process transport, shared payloads, and downstream operator integration."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from tests.zephon._internal.runners._helpers import (
    _ctx_services,
    _extract_values,
    _mk_record,
    _mk_records,
)
from zephon._internal.graph import Node, Stage
from zephon._internal.ops.delay import DelayById
from zephon._internal.runners.process import ProcessStageRunner
from zephon._internal.utils.shm_coalesce import PayloadMemoryPolicy
from zephon.ops.accumulators import (
    Accumulator,
    CountingAccumulator,
    PassthroughAccumulator,
)
from zephon.ops.base import BaseOp, OpContext
from zephon.ops.traits import OpTraits
from zephon.types import SampleRecord

pytestmark = pytest.mark.integration

torch = pytest.importorskip("torch")


@dataclass
class _TensorizeValueOp(BaseOp):
    def __post_init__(self) -> None:
        super().__init__()

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=True,
            preserves_cursor_order=True,
            parallelism=1,
            batch_shape_sensitive=False,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return PassthroughAccumulator[SampleRecord]()

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        out: list[SampleRecord] = []
        for elem in elems:
            payload = dict(elem.payload)
            payload["tensor"] = torch.tensor([int(payload["value"])], dtype=torch.int64)
            out.append(SampleRecord(meta=elem.meta, payload=payload))
        return out


@dataclass
class _EncodeLargeBytesOp(BaseOp):
    def __post_init__(self) -> None:
        super().__init__()

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=True,
            preserves_cursor_order=True,
            parallelism=1,
            batch_shape_sensitive=False,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return PassthroughAccumulator[SampleRecord]()

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        out: list[SampleRecord] = []
        for elem in elems:
            value = int(elem.payload["value"])
            payload = dict(elem.payload)
            payload["text"] = (f"value-{value}|".encode("utf-8")) * 1024
            out.append(SampleRecord(meta=elem.meta, payload=payload))
        return out


@dataclass
class _DecodeAndAnnotateBatchOp(BaseOp):
    max_batch: int = 3

    def __post_init__(self) -> None:
        super().__init__()

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=True,
            preserves_cursor_order=True,
            parallelism=1,
            batch_shape_sensitive=False,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return CountingAccumulator[SampleRecord](
            max_batch=self.max_batch,
            max_latency_ms=None if deterministic else 1,
        )

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        from zephon._internal.utils.shm_coalesce import _ShmBytes

        batch_size = len(elems)
        out: list[SampleRecord] = []
        for elem in elems:
            payload = dict(elem.payload)
            assert isinstance(payload["text"], _ShmBytes)
            payload["text"] = payload["text"].decode("utf-8")
            payload["batch_size"] = batch_size
            out.append(SampleRecord(meta=elem.meta, payload=payload))
        return out


def test_process_runner_coalesced_tensors_preserve_deterministic_order() -> None:
    tensorize = Node(name="tensorize", op=_TensorizeValueOp())
    delay = Node(name="delay", op=DelayById(max_delay_ms=1.5), inputs=[tensorize])
    stage = Stage(
        name="coalesced_tensors",
        nodes=[tensorize, delay],
        placement="auto",
        break_reason="test",
    )

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=4,
        deterministic=True,
        stage_output_mode="stream_items",
        memory_policy=PayloadMemoryPolicy(
            min_item_bytes=0, min_new_allocation_bytes=0, min_forward_bytes=0
        ),
    )
    try:
        data = list(range(30))
        out = list(runner.run(iter(_mk_records(data))))
        assert _extract_values(out) == data
        for expected, rec in zip(data, out, strict=True):
            assert rec.payload["tensor"].is_shared()
            assert int(rec.payload["tensor"][0]) == expected
    finally:
        runner.close()


@dataclass
class _BatchTensorizeOp(BaseOp):
    """Tensorize with batching so multiple records land in one microbatch."""

    max_batch: int = 8

    def __post_init__(self) -> None:
        super().__init__()

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=True,
            preserves_cursor_order=True,
            parallelism=1,
            batch_shape_sensitive=False,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return CountingAccumulator[SampleRecord](
            max_batch=self.max_batch, max_latency_ms=None
        )

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        out: list[SampleRecord] = []
        for elem in elems:
            payload = dict(elem.payload)
            payload["tensor"] = torch.tensor([int(payload["value"])], dtype=torch.int64)
            out.append(SampleRecord(meta=elem.meta, payload=payload))
        return out


def test_process_runner_coalesced_tensors_are_zero_copy_views() -> None:
    """Restored tensors after IPC should be views into the same SHM storage."""
    tensorize = Node(name="tensorize", op=_BatchTensorizeOp(max_batch=8))
    stage = Stage(
        name="coalesced_zerocopy",
        nodes=[tensorize],
        placement="auto",
        break_reason="test",
    )

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=1,
        deterministic=True,
        stage_output_mode="stream_items",
        memory_policy=PayloadMemoryPolicy(
            coalesce=True,
            min_item_bytes=0,
            min_new_allocation_bytes=0,
            min_forward_bytes=0,
        ),
    )
    try:
        # 8 records with max_batch=8 → one microbatch → one coalesced buffer
        data = list(range(8))
        out = list(runner.run(iter(_mk_records(data))))
        assert _extract_values(out) == data

        # All tensors from the same coalesced microbatch share one storage
        storages = {rec.payload["tensor"].untyped_storage().data_ptr() for rec in out}
        assert len(storages) == 1, (
            f"Expected all tensors to share one SHM storage, got {len(storages)}"
        )
    finally:
        runner.close()


def test_process_runner_coalesced_bytes_preserve_counting_accumulator_batches() -> None:
    encode = Node(name="encode", op=_EncodeLargeBytesOp())
    decode = Node(
        name="decode",
        op=_DecodeAndAnnotateBatchOp(max_batch=3),
        inputs=[encode],
    )
    stage = Stage(
        name="coalesced_bytes",
        nodes=[encode, decode],
        placement="auto",
        break_reason="test",
    )

    runner = ProcessStageRunner(
        stage,
        ctx_services=_ctx_services(),
        max_workers=4,
        deterministic=True,
        stage_output_mode="stream_items",
        memory_policy=PayloadMemoryPolicy(
            min_item_bytes=0, min_new_allocation_bytes=0, min_forward_bytes=0
        ),
    )
    try:
        out = list(runner.run(iter(_mk_records(range(7)))))
        assert _extract_values(out) == list(range(7))
        assert [rec.payload["batch_size"] for rec in out] == [3, 3, 3, 3, 3, 3, 1]
        for rec in out:
            assert rec.payload["text"].startswith(f"value-{rec.payload['value']}|")
    finally:
        runner.close()


class _CopySharedArray(BaseOp):
    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True)

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        from zephon._internal.utils.shm_coalesce import _shared_numpy_storage

        outputs = []
        for record in elems:
            array: Any = record.payload
            assert _shared_numpy_storage(array) is not None
            copied = array.copy()
            copied[0] += 1
            outputs.append(SampleRecord(meta=record.meta, payload=copied))
        return outputs


def test_process_numpy_reduction_forwards_zephon_buffers_between_ops() -> None:
    import pickle
    from multiprocessing.reduction import ForkingPickler

    from zephon._internal.stream import resolve_lazy_payloads
    from zephon._internal.utils.shm_coalesce import prepare_microbatch

    np = pytest.importorskip("numpy")
    record = _mk_record(0)
    record.payload = np.arange(32)[3:20:2]
    prepared = prepare_microbatch(
        [record],
        PayloadMemoryPolicy(
            min_item_bytes=0, min_new_allocation_bytes=0, min_forward_bytes=0
        ),
    )
    records = pickle.loads(ForkingPickler.dumps(prepared))
    resolve_lazy_payloads(records)
    [record] = records
    array = record.payload
    stage = Stage(
        "shared_numpy",
        [
            Node("first", _CopySharedArray(), parallelism=1),
            Node("second", _CopySharedArray(), parallelism=1),
        ],
        "auto",
        "test",
    )
    runner = ProcessStageRunner(
        stage,
        _ctx_services(),
        max_workers=2,
        deterministic=True,
        memory_policy=PayloadMemoryPolicy(
            min_item_bytes=0, min_new_allocation_bytes=0, min_forward_bytes=0
        ),
        stage_output_mode="stream_items",
    )
    try:
        [result] = list(runner.run([record]))
        assert isinstance(result, SampleRecord)
        assert record.payload is array
        expected = array.copy()
        expected[0] += 2
        np.testing.assert_array_equal(result.payload, expected)
    finally:
        runner.close()


class _CreatePayloadMemory(BaseOp):
    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, preserves_cursor_order=True, parallelism=1)

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        import numpy as np

        return [
            SampleRecord(
                record.meta,
                {
                    "small_t": torch.arange(4),
                    "small_n": np.arange(4),
                    "large_t": torch.arange(64),
                    "large_n": np.arange(64),
                    "large_b": b"x" * 512,
                    "large_ba": bytearray(b"x" * 512),
                },
            )
            for record in elems
        ]


class _CropSharedViews(PassthroughAccumulator[SampleRecord]):
    @property
    def reads_payload(self) -> bool:
        return True

    def push_many(
        self, elems: Sequence[SampleRecord]
    ) -> list[tuple[list[SampleRecord], int]]:
        for record in elems:
            for key in ("large_t", "large_n"):
                record.payload[key] = record.payload[key][:4]
        return super().push_many(elems)


class _InspectPayloadMemory(BaseOp):
    def __init__(self, *, crop: bool = False) -> None:
        super().__init__()
        self.crop = crop

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, preserves_cursor_order=True, parallelism=1)

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return _CropSharedViews() if self.crop else PassthroughAccumulator()

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        from zephon._internal.utils.shm_coalesce import _shared_numpy_storage, _ShmBytes

        for record in elems:
            payload = record.payload
            assert payload["large_t"].is_shared()
            assert payload["large_t"].untyped_storage().nbytes() == 512
            assert _shared_numpy_storage(payload["large_n"]).nbytes() == 512
            assert (
                len(payload["large_t"])
                == len(payload["large_n"])
                == (4 if self.crop else 64)
            )
            for key in ("large_b", "large_ba"):
                assert bytes(payload[key]) == b"x" * 512
                assert isinstance(payload[key], _ShmBytes)
        return elems


def _unexpected_shm_preparation(*args, **kwargs):
    raise AssertionError("unexpected SHM preparation")


class _InspectDisabledShm(BaseOp):
    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, preserves_cursor_order=True, parallelism=1)

    def setup(self, ctx: OpContext) -> None:
        from zephon._internal.runners import process

        process.prepare_microbatch = _unexpected_shm_preparation

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        import numpy as np

        for record in elems:
            assert type(record.payload["bytes"]) is bytes
            assert type(record.payload["mutable"]) is bytearray
            assert type(record.payload["array"]) is np.ndarray
            record.payload["mutable"][0] = 7
            record.payload["array"][0] = 7
        return elems


def test_disabled_shm_skips_worker_preparation(monkeypatch) -> None:
    import numpy as np

    from zephon._internal.runners import process

    monkeypatch.setattr(process, "prepare_microbatch", _unexpected_shm_preparation)
    stage = Stage(
        "disabled_shm", [Node("inspect", _InspectDisabledShm())], "auto", "test"
    )
    record = _mk_record(0)
    record.payload = {"bytes": b"abc", "mutable": bytearray(16), "array": np.arange(16)}
    runner = ProcessStageRunner(
        stage,
        _ctx_services(),
        max_workers=1,
        deterministic=True,
        memory_policy=None,
        stage_output_mode="stream_items",
    )
    try:
        [actual] = list(runner.run([record]))
        assert actual.payload["bytes"] == b"abc"
        assert actual.payload["mutable"][0] == actual.payload["array"][0] == 7
        assert record.payload["mutable"][0] == record.payload["array"][0] == 0
    finally:
        runner.close()


def test_worker_output_policy_and_parent_forwarding_after_resolution(
    monkeypatch,
) -> None:
    from zephon._internal.runners import process
    from zephon._internal.utils.shm_coalesce import _shared_numpy_storage

    # Spawned workers prepare outputs; the parent only forwards them.
    monkeypatch.setattr(process, "prepare_microbatch", _unexpected_shm_preparation)
    stage = Stage(
        "memory_policy",
        [
            Node("create", _CreatePayloadMemory(), parallelism=1),
            Node("lazy", _InspectPayloadMemory(), parallelism=1),
            Node("resolved", _InspectPayloadMemory(crop=True), parallelism=1),
        ],
        "auto",
        "test",
    )
    runner = ProcessStageRunner(
        stage,
        _ctx_services(),
        max_workers=3,
        deterministic=True,
        stage_output_mode="stream_items",
        # Exercise both routes without depending on production tuning defaults.
        memory_policy=PayloadMemoryPolicy(
            min_item_bytes=64, min_new_allocation_bytes=512, min_forward_bytes=256
        ),
    )
    try:
        [actual] = list(runner.run([_mk_record(0)]))
        # The next worker's output policy sends the small crops inline.
        for key in ("small_t", "small_n", "large_t", "large_n"):
            assert actual.payload[key].tolist() == list(range(4))
        assert not actual.payload["large_t"].is_shared()
        assert _shared_numpy_storage(actual.payload["large_n"]) is None
    finally:
        runner.close()
