# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Process transport, shared payloads, and downstream operator integration."""

import pickle
from collections.abc import Sequence
from multiprocessing.reduction import ForkingPickler
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
from zephon._internal.runners import process
from zephon._internal.runners.process import ProcessStageRunner
from zephon._internal.stream import resolve_lazy_payloads
from zephon._internal.utils.shm_coalesce import (
    PayloadMemoryPolicy,
    _shared_numpy_storage,
    _ShmBytes,
    prepare_microbatch,
)
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
np = pytest.importorskip("numpy")
_SHM_POLICY = PayloadMemoryPolicy(
    min_item_bytes=0, min_new_allocation_bytes=0, min_forward_bytes=0
)


def _run(
    nodes: list[Node],
    records: list[SampleRecord],
    *,
    workers: int = 1,
    policy: PayloadMemoryPolicy | None = _SHM_POLICY,
) -> list[SampleRecord]:
    runner = ProcessStageRunner(
        Stage("payload_memory", nodes, "auto", "test"),
        _ctx_services(),
        max_workers=workers,
        deterministic=True,
        stage_output_mode="stream_items",
        memory_policy=policy,
    )
    try:
        return list(runner.run(records))
    finally:
        runner.close()


class _OrderedOp(BaseOp):
    def traits(self) -> OpTraits:
        return OpTraits(calibration_replayable=False, preserves_cursor_order=True)


class _TensorizeValueOp(_OrderedOp):
    def accumulator(self, *, deterministic: bool, ctx: dict[str, Any]) -> Accumulator:
        return CountingAccumulator(max_batch=8, max_latency_ms=None)

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        return [
            SampleRecord(
                e.meta, dict(e.payload, tensor=torch.tensor([e.payload["value"]]))
            )
            for e in elems
        ]


class _EncodeLargeBytesOp(_OrderedOp):
    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        return [
            SampleRecord(
                e.meta,
                dict(e.payload, text=f"value-{e.payload['value']}|".encode() * 1024),
            )
            for e in elems
        ]


class _DecodeAndAnnotateBatchOp(_OrderedOp):
    def accumulator(self, *, deterministic: bool, ctx: dict[str, Any]) -> Accumulator:
        return CountingAccumulator(max_batch=3, max_latency_ms=None)

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        for elem in elems:
            assert isinstance(elem.payload["text"], _ShmBytes)
        return [
            SampleRecord(
                e.meta,
                dict(e.payload, text=e.payload["text"].decode(), batch_size=len(elems)),
            )
            for e in elems
        ]


def test_process_runner_coalesced_tensors_preserve_order_and_shared_storage() -> None:
    tensorize = Node("tensorize", _TensorizeValueOp())
    nodes = [
        tensorize,
        Node("delay", DelayById(max_delay_ms=1.5, max_batch=3), inputs=[tensorize]),
    ]
    data = list(range(30))
    out = _run(nodes, _mk_records(data), workers=4)
    assert _extract_values(out) == data
    for expected, record in zip(data, out, strict=True):
        assert record.payload["tensor"].is_shared()
        assert int(record.payload["tensor"][0]) == expected
    # Each producer microbatch keeps one allocation across the downstream workers.
    for start in range(0, len(out), 8):
        storages = {
            r.payload["tensor"].untyped_storage().data_ptr()
            for r in out[start : start + 8]
        }
        assert len(storages) == 1


def test_process_runner_coalesced_bytes_preserve_counting_accumulator_batches() -> None:
    encode = Node("encode", _EncodeLargeBytesOp())
    nodes = [encode, Node("decode", _DecodeAndAnnotateBatchOp(), inputs=[encode])]
    out = _run(nodes, _mk_records(range(7)), workers=4)
    assert _extract_values(out) == list(range(7))
    assert [r.payload["batch_size"] for r in out] == [3, 3, 3, 3, 3, 3, 1]
    for record in out:
        assert record.payload["text"].startswith(f"value-{record.payload['value']}|")


class _CopySharedArray(_OrderedOp):
    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        outputs = []
        for record in elems:
            assert _shared_numpy_storage(record.payload) is not None
            copied = record.payload.copy()
            copied[0] += 1
            outputs.append(SampleRecord(record.meta, copied))
        return outputs


def test_process_numpy_reduction_forwards_zephon_buffers_between_ops() -> None:
    record = _mk_record(0)
    record.payload = np.arange(32)[3:20:2]
    records = pickle.loads(
        ForkingPickler.dumps(prepare_microbatch([record], _SHM_POLICY))
    )
    resolve_lazy_payloads(records)
    [record] = records
    array = record.payload
    nodes = [Node("first", _CopySharedArray()), Node("second", _CopySharedArray())]
    [result] = _run(nodes, records, workers=2)
    assert record.payload is array
    expected = array.copy()
    expected[0] += 2
    np.testing.assert_array_equal(result.payload, expected)


class _CreatePayloadMemory(_OrderedOp):
    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
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
                    "large_mv": memoryview(bytearray(b"x" * 512)),
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


class _InspectPayloadMemory(_OrderedOp):
    def __init__(self, *, crop: bool = False) -> None:
        super().__init__()
        self.crop = crop

    def accumulator(self, *, deterministic: bool, ctx: dict[str, Any]) -> Accumulator:
        return _CropSharedViews() if self.crop else PassthroughAccumulator()

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        for record in elems:
            payload: Any = record.payload
            assert payload["large_t"].is_shared()
            assert payload["large_t"].untyped_storage().nbytes() == 512
            assert _shared_numpy_storage(payload["large_n"]).nbytes() == 512
            assert (
                len(payload["large_t"])
                == len(payload["large_n"])
                == (4 if self.crop else 64)
            )
            for key in ("large_b", "large_ba", "large_mv"):
                raw = payload[key]
                assert isinstance(raw, (_ShmBytes, bytearray))
                assert bytes(raw) == (b"yzz" if self.crop else b"xxx") + b"x" * 509
                raw[0] = ord("z" if self.crop else "y")
                raw[1:3] = [ord("z"), ord("z")]
        return elems


def _unexpected_shm_preparation(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("unexpected SHM preparation")


class _InspectDisabledShm(_OrderedOp):
    def setup(self, ctx: OpContext) -> None:
        process.prepare_microbatch = _unexpected_shm_preparation

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        for record in elems:
            assert type(record.payload["bytes"]) is bytes
            assert type(record.payload["mutable"]) is bytearray
            assert type(record.payload["array"]) is np.ndarray
            record.payload["mutable"][0] = 7
            record.payload["array"][0] = 7
        return elems


def test_disabled_shm_skips_worker_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process, "prepare_microbatch", _unexpected_shm_preparation)
    record = _mk_record(0)
    record.payload = {"bytes": b"abc", "mutable": bytearray(16), "array": np.arange(16)}
    [actual] = _run([Node("inspect", _InspectDisabledShm())], [record], policy=None)
    assert actual.payload["bytes"] == b"abc"
    assert actual.payload["mutable"][0] == actual.payload["array"][0] == 7
    assert record.payload["mutable"][0] == record.payload["array"][0] == 0


def test_worker_output_policy_and_parent_forwarding_after_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Spawned workers prepare outputs; the parent only forwards them.
    monkeypatch.setattr(process, "prepare_microbatch", _unexpected_shm_preparation)
    nodes = [
        Node("create", _CreatePayloadMemory()),
        Node("lazy", _InspectPayloadMemory()),
        Node("resolved", _InspectPayloadMemory(crop=True)),
    ]
    policy = PayloadMemoryPolicy(
        min_item_bytes=64, min_new_allocation_bytes=512, min_forward_bytes=256
    )
    [actual] = _run(nodes, [_mk_record(0)], workers=3, policy=policy)
    # The next worker's output policy sends the small crops inline.
    for key in ("small_t", "small_n", "large_t", "large_n"):
        assert actual.payload[key].tolist() == list(range(4))
    assert not actual.payload["large_t"].is_shared()
    assert _shared_numpy_storage(actual.payload["large_n"]) is None
    payload: Any = actual.payload
    for key in ("large_b", "large_ba", "large_mv"):
        assert bytes(payload[key]) == b"zzz" + b"x" * 509
        assert not memoryview(payload[key]).readonly
