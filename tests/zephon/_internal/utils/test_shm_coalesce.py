"""Tests for the tensor coalescing SHM optimization."""

from __future__ import annotations

import gc
import pickle
import sys
from io import BytesIO
from multiprocessing.reduction import ForkingPickler

import pytest

from zephon._internal.ops.decode_text import DecodeText
from zephon._internal.stream import LazyPayload, resolve_lazy_payloads
from zephon._internal.utils import shm_coalesce
from zephon._internal.utils.shm_coalesce import (
    CoalescedMicrobatch,
    ShmLazyPayload,
    _ShmBytes,
    _ShmBytesLegacy,
    _ShmLeafPayload,
    coalesce_microbatch,
)
from zephon.ops.accumulators import CountingAccumulator
from zephon.types import SampleBatch, SampleMeta, SampleRecord

torch = pytest.importorskip("torch")


def _forking_round_trip(obj: object) -> bytes:
    buf = BytesIO()
    ForkingPickler(buf).dump(obj)
    return buf.getvalue()


def _meta(i: int) -> SampleMeta:
    return SampleMeta(sample_id=(0, 0, i), lane_id=0, chunk_id=0, chunk_offset=i)


def _round_trip(coalesced: CoalescedMicrobatch) -> list:
    """ForkingPickler round-trip — returns records with unresolved LazyPayloads."""
    blob = _forking_round_trip(coalesced)
    restored = pickle.loads(blob)
    assert isinstance(restored, list)
    return restored


def _round_trip_resolved(coalesced: CoalescedMicrobatch) -> list:
    """Full round-trip simulating worker → pump → worker resolve."""
    restored = _round_trip(coalesced)
    resolve_lazy_payloads(restored)
    return restored


# ---------------------------------------------------------------------------
# Basic round-trip
# ---------------------------------------------------------------------------
def test_training_policies_after_coalesced_transport() -> None:
    """EOS loss masking works after SHM transport and lazy payload resolution."""
    tokens = torch.tensor([1, 10, 2, 1, 20, 2])
    batch = SampleBatch(
        records=(SampleRecord(meta=_meta(0), payload={"input_ids": tokens}),),
    )
    coalesced = coalesce_microbatch([batch], shm_min_item_bytes=0)
    assert coalesced is not None
    [restored_batch] = _round_trip(coalesced)
    assert isinstance(restored_batch.records[0].payload, LazyPayload)
    resolve_lazy_payloads([restored_batch])
    assert restored_batch.to_training(
        return_labels=True,
        eos_mask_loss=True,
        eos_token_id=2,
    )["labels"].tolist() == [[10, 2, -100, 20, 2]]


class TestCoalesceMicrobatchRoundTrip:
    """Coalesce → ForkingPickler → pickle.loads → resolve should recover original data."""

    def test_single_dtype(self) -> None:
        records: list[SampleRecord] = [
            SampleRecord(
                meta=_meta(i),
                payload={
                    "input_ids": torch.arange(5, dtype=torch.long) + i * 10,
                    "attention_mask": torch.ones(5, dtype=torch.long),
                    "text": f"sample-{i}",
                },
            )
            for i in range(4)
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        assert isinstance(coalesced, CoalescedMicrobatch)

        # Only 1 dtype → 1 SHM buffer
        assert len(coalesced.buffers) == 1
        assert "torch.int64" in coalesced.buffers
        buf = coalesced.buffers["torch.int64"]
        assert buf.is_shared()

        restored = _round_trip_resolved(coalesced)
        assert len(restored) == 4
        for i, rec in enumerate(restored):
            assert isinstance(rec, SampleRecord)
            assert rec.meta.chunk_offset == i
            assert rec.payload["text"] == f"sample-{i}"
            torch.testing.assert_close(
                rec.payload["input_ids"],
                torch.arange(5, dtype=torch.long) + i * 10,
            )
            torch.testing.assert_close(
                rec.payload["attention_mask"],
                torch.ones(5, dtype=torch.long),
            )

    def test_mixed_dtypes(self) -> None:
        records: list[SampleRecord] = [
            SampleRecord(
                meta=_meta(i),
                payload={
                    "tokens": torch.arange(3, dtype=torch.long),
                    "features": torch.randn(4, dtype=torch.float32),
                },
            )
            for i in range(2)
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        # 2 dtypes → 2 SHM buffers
        assert len(coalesced.buffers) == 2
        assert "torch.int64" in coalesced.buffers
        assert "torch.float32" in coalesced.buffers

        restored = _round_trip_resolved(coalesced)
        assert len(restored) == 2
        for rec in restored:
            assert rec.payload["tokens"].shape == (3,)
            assert rec.payload["features"].shape == (4,)

    def test_multidimensional_tensors(self) -> None:
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        records = [SampleRecord(meta=_meta(0), payload={"matrix": t})]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        torch.testing.assert_close(restored[0].payload["matrix"], t)
        assert restored[0].payload["matrix"].shape == (3, 4)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestCoalesceEdgeCases:
    def test_no_tensors_returns_none(self) -> None:
        records = [SampleRecord(meta=_meta(0), payload={"text": "hello", "count": 42})]
        assert coalesce_microbatch(records) is None
        assert pickle.loads(_forking_round_trip(records)) == records

    def test_no_tensors_payloads_unchanged(self) -> None:
        """When coalesce returns None, payloads must not be mutated."""
        payload = {"text": "hello", "count": 42}
        records = [SampleRecord(meta=_meta(0), payload=payload)]
        assert coalesce_microbatch(records) is None
        assert records[0].payload is payload
        assert records[0].payload["text"] == "hello"

    def test_returns_none_when_torch_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={
                    "raw": b"x" * 8192,
                    "text": "hello",
                },
            )
        ]
        monkeypatch.setattr("zephon._internal.utils.shm_coalesce._torch", None)
        monkeypatch.setattr("zephon._internal.utils.shm_coalesce._torch_loaded", True)
        assert coalesce_microbatch(records) is None

    def test_empty_tensor(self) -> None:
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={
                    "empty": torch.empty(0, dtype=torch.float32),
                    "ok": torch.tensor([1.0]),
                },
            )
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None
        restored = _round_trip_resolved(coalesced)
        assert restored[0].payload["empty"].numel() == 0
        torch.testing.assert_close(restored[0].payload["ok"], torch.tensor([1.0]))

    def test_nested_list_payload(self) -> None:
        """Tensors inside list-typed payloads are also coalesced."""
        records = [
            SampleRecord(
                meta=_meta(0),
                payload=[torch.tensor([1, 2, 3]), "text", torch.tensor([4.0, 5.0])],
            )
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None
        restored = _round_trip_resolved(coalesced)
        torch.testing.assert_close(restored[0].payload[0], torch.tensor([1, 2, 3]))
        assert restored[0].payload[1] == "text"
        torch.testing.assert_close(restored[0].payload[2], torch.tensor([4.0, 5.0]))

    def test_noncontiguous_torch_tensor(self) -> None:
        """Non-contiguous tensors (slices, transposes) coalesce correctly."""
        t_slice = torch.arange(10, dtype=torch.float32)[::2]  # stride != 1
        t_transposed = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
        assert not t_slice.is_contiguous()
        assert not t_transposed.is_contiguous()
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={"sliced": t_slice, "transposed": t_transposed},
            )
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        torch.testing.assert_close(restored[0].payload["sliced"], t_slice)
        torch.testing.assert_close(restored[0].payload["transposed"], t_transposed)
        assert restored[0].payload["sliced"].shape == t_slice.shape
        assert restored[0].payload["transposed"].shape == t_transposed.shape

    def test_sample_batch_records(self) -> None:
        """SampleBatch items are walked into and coalesced correctly."""
        records = [
            SampleRecord(meta=_meta(0), payload={"t": torch.tensor([1.0])}),
            SampleRecord(meta=_meta(1), payload={"t": torch.tensor([2.0])}),
        ]
        batch = SampleBatch(records=tuple(records))
        coalesced = coalesce_microbatch([batch])
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        assert len(restored) == 1
        assert isinstance(restored[0], SampleBatch)
        assert len(restored[0].records) == 2
        torch.testing.assert_close(
            restored[0].records[0].payload["t"], torch.tensor([1.0])
        )
        torch.testing.assert_close(
            restored[0].records[1].payload["t"], torch.tensor([2.0])
        )


# ---------------------------------------------------------------------------
# SHM allocation
# ---------------------------------------------------------------------------
class TestAllocShmBuffer:
    def test_attaches_shared_storage_on_cpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        allocate = torch.UntypedStorage._new_shared
        allocations: list[object] = []

        def capture_storage(size: int, *, device: str) -> object:
            storage = allocate(size, device=device)
            allocations.append(storage)
            return storage

        monkeypatch.setattr(torch.UntypedStorage, "_new_shared", capture_storage)
        with torch.device("meta"):
            buf = shm_coalesce._alloc_shm_buffer(37, torch.int64, "test")
        assert len(allocations) == 1
        assert buf.untyped_storage() is allocations[0]
        assert buf.device.type == "cpu"
        assert buf.dtype == torch.int64
        assert buf.shape == (37,)
        assert buf.is_shared()
        assert buf.untyped_storage().nbytes() == 37 * 8
        buf.fill_(3)
        assert buf.tolist() == [3] * 37

    @pytest.mark.parametrize("message", ["No space left on device (28)", "Success (0)"])
    def test_retries_shm_exhaustion(
        self, monkeypatch: pytest.MonkeyPatch, message: str
    ) -> None:
        allocate = torch.UntypedStorage._new_shared
        attempts = 0
        waits: list[str] = []

        def transient_failure(size: int, *, device: str) -> object:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError(
                    f"unable to allocate shared memory(shm) for file </torch_1_2_3>: {message}"
                )
            return allocate(size, device=device)

        monkeypatch.setattr(torch.UntypedStorage, "_new_shared", transient_failure)
        monkeypatch.setattr(shm_coalesce, "wait_for_shm_space", waits.append)
        buf = shm_coalesce._alloc_shm_buffer(8, torch.int64, "test")
        assert buf.is_shared()
        assert attempts == 2
        assert waits == ["test"]

    def test_propagates_unrelated_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        waits: list[str] = []

        def fail(*args: object, **kwargs: object) -> None:
            raise RuntimeError("unexpected allocation error")

        monkeypatch.setattr(torch.UntypedStorage, "_new_shared", fail)
        monkeypatch.setattr(shm_coalesce, "wait_for_shm_space", waits.append)
        with pytest.raises(RuntimeError, match="unexpected allocation error"):
            shm_coalesce._alloc_shm_buffer(8, torch.int64, "test")
        assert waits == []


# ---------------------------------------------------------------------------
# SHM properties
# ---------------------------------------------------------------------------
class TestCoalesceShmProperties:
    def test_buffers_are_in_shared_memory(self) -> None:
        records = [
            SampleRecord(
                meta=_meta(i),
                payload={"t": torch.randn(10)},
            )
            for i in range(5)
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None
        for buf in coalesced.buffers.values():
            assert buf.is_shared(), "Coalesced buffer should be in shared memory"

    def test_fewer_shm_segments_than_individual_tensors(self) -> None:
        """Coalescing N same-dtype tensors → 1 SHM buffer (not N)."""
        records = [
            SampleRecord(
                meta=_meta(i),
                payload={
                    "a": torch.randn(8),
                    "b": torch.randn(4),
                },
            )
            for i in range(10)
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        # 10 records × 2 tensors = 20 tensors, but only 1 SHM buffer (all float32)
        assert len(coalesced.buffers) == 1
        buf = coalesced.buffers["torch.float32"]
        assert buf.numel() == 10 * (8 + 4)

    def test_restored_tensors_are_views_not_copies(self) -> None:
        """Restored tensors should be views into the coalesced buffer (zero-copy)."""
        records = [
            SampleRecord(
                meta=_meta(i),
                payload={"t": torch.arange(3, dtype=torch.float32) + i},
            )
            for i in range(3)
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        # All restored tensors should share the same storage
        storages = {rec.payload["t"].untyped_storage().data_ptr() for rec in restored}
        assert len(storages) == 1, "All views should share one underlying storage"

    def test_restored_tensors_are_contiguous(self) -> None:
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={"t": torch.arange(12, dtype=torch.float32).reshape(3, 4)},
            )
        ]
        coalesced = coalesce_microbatch(records)
        restored = _round_trip_resolved(coalesced)
        assert restored[0].payload["t"].is_contiguous()


# ---------------------------------------------------------------------------
# Bytes coalescing
# ---------------------------------------------------------------------------
class TestCoalesceBytes:
    def test_large_bytes_round_trip(self) -> None:
        """bytes payloads above threshold go through SHM and restore correctly."""
        data_a = b"A" * 8192
        data_b = b"B" * 16384
        records = [
            SampleRecord(meta=_meta(0), payload={"raw": data_a, "label": "hello"}),
            SampleRecord(meta=_meta(1), payload={"raw": data_b, "label": "world"}),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None
        assert "_bytes_uint8" in coalesced.buffers

        restored = _round_trip_resolved(coalesced)
        assert bytes(restored[0].payload["raw"]) == data_a
        assert restored[0].payload["label"] == "hello"
        assert bytes(restored[1].payload["raw"]) == data_b

    def test_restored_bytes_are_shm_backed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bytes and bytearrays restore as views, without copying out of SHM."""
        data = b"M" * 8192
        records = [
            SampleRecord(meta=_meta(0), payload={"raw": data}),
            SampleRecord(meta=_meta(1), payload={"raw": bytearray(data)}),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        if sys.version_info >= (3, 12):

            def unexpected_narrow(*args, **kwargs):
                raise AssertionError("restoring shared bytes needs no Tensor view")

            monkeypatch.setattr(torch.Tensor, "narrow", unexpected_narrow)
        restored = _round_trip_resolved(coalesced)
        for i, record in enumerate(restored):
            raw = record.payload["raw"]
            assert isinstance(raw, _ShmBytes)
            assert bytes(raw) == data
            assert len(raw) == len(data)
            if sys.version_info >= (3, 12):
                assert not isinstance(raw, bytes)
                assert raw._buffer.is_shared()
                assert raw._np_view.__array_interface__["data"][
                    0
                ] == raw._buffer.data_ptr() + i * len(data)

    def test_small_bytes_stay_inline(self) -> None:
        """bytes below the threshold are left in the pickle stream."""
        small = b"tiny"
        records = [SampleRecord(meta=_meta(0), payload={"data": small})]
        coalesced = coalesce_microbatch(records)
        # No tensors, no large bytes → nothing to coalesce
        assert coalesced is None

    def test_mixed_tensors_and_bytes(self) -> None:
        """Tensors and large bytes coalesce into separate SHM buffers."""
        records = [
            SampleRecord(
                meta=_meta(i),
                payload={
                    "image": b"\xff" * 10000,
                    "tokens": torch.arange(5, dtype=torch.long),
                },
            )
            for i in range(3)
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        # 1 buffer for int64 tensors + 1 buffer for bytes
        assert "torch.int64" in coalesced.buffers
        assert "_bytes_uint8" in coalesced.buffers

        restored = _round_trip_resolved(coalesced)
        for rec in restored:
            assert bytes(rec.payload["image"]) == b"\xff" * 10000
            torch.testing.assert_close(
                rec.payload["tokens"], torch.arange(5, dtype=torch.long)
            )

    def test_memoryview_payload_coalesced(self) -> None:
        """Typed views retain every byte without staging through bytes()."""
        data = bytearray(range(256)) * 32
        view = memoryview(data).cast("I", shape=(32, 64))
        records = [SampleRecord(meta=_meta(0), payload={"mv": view})]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None
        restored = _round_trip_resolved(coalesced)
        assert bytes(restored[0].payload["mv"]) == data
        assert isinstance(restored[0].payload["mv"], _ShmBytes)

    def test_empty_multidim_memoryview_stays_inline(self) -> None:
        empty = memoryview(np.zeros((0, 4), dtype=np.int32))
        record = SampleRecord(meta=_meta(0), payload=empty)
        prepared = coalesce_microbatch([record], shm_min_item_bytes=0)
        assert prepared is not None and not prepared.buffers
        assert _round_trip_resolved(prepared)[0].payload == b""

    def test_strided_memoryviews_preserve_bytes_and_offsets(self) -> None:
        array = np.arange(12, dtype=np.int32)
        view = memoryview(array)[::-2]
        # Field assignment would discard these deliberately nonzero padding bytes.
        padded = np.zeros(
            4, dtype=np.dtype({"names": ["x"], "formats": ["i1"], "itemsize": 8})
        )
        padded.view(np.uint8)[:] = np.arange(padded.nbytes, dtype=np.uint8)
        structured = memoryview(padded)[::2]
        payload = {"a": b"abc", "b": view, "c": structured, "d": b"tail"}
        coalesced = coalesce_microbatch(
            [SampleRecord(meta=_meta(0), payload=payload)], shm_min_item_bytes=0
        )
        assert coalesced is not None
        [record] = _round_trip_resolved(coalesced)
        assert bytes(record.payload["a"]) == b"abc"
        assert bytes(record.payload["b"]) == view.tobytes()
        assert len(record.payload["b"]) == view.nbytes
        assert bytes(record.payload["c"]) == structured.tobytes()
        assert bytes(record.payload["d"]) == b"tail"

    def test_shm_bytes_survives_repickling_zero_copy(self) -> None:
        """Full round-trip: worker→pump→worker2, bytes stay in SHM throughout.

        Simulates the real pipeline path:
          1. Worker coalesces bytes into SHM, puts CoalescedMicrobatch on queue
          2. Pump thread unpickles (hop 1) → gets list[StreamItem] with ShmLazyPayload
          3. Pump thread re-pickles records to next worker (hop 2) — ShmLazyPayload
             forwards SHM refs without resolving
          4. Worker 2 unpickles → gets ShmLazyPayload → resolves → _ShmBytes backed by SHM
        """
        data = b"Z" * 8192
        records = [SampleRecord(meta=_meta(0), payload={"raw": data})]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        # Hop 1: worker → pump thread (CoalescedMicrobatch → ShmLazyPayload)
        restored = _round_trip(coalesced)
        assert isinstance(restored[0].payload, ShmLazyPayload)

        # Hop 2: pump thread → worker 2 (ShmLazyPayload re-pickled via FD passing)
        blob2 = _forking_round_trip(restored)
        restored2 = pickle.loads(blob2)
        # Still lazy after hop 2
        assert isinstance(restored2[0].payload, ShmLazyPayload)

        # Worker 2 resolves
        resolve_lazy_payloads(restored2)
        raw2 = restored2[0].payload["raw"]
        assert isinstance(raw2, _ShmBytes)
        assert bytes(raw2) == data

    def test_shm_bytes_decode(self) -> None:
        """_ShmBytes.decode() works for text ops."""
        text = "hello world"
        data = text.encode("utf-8")
        # Pad to exceed threshold
        padded = data + b"\x00" * (DEFAULT_SHM_MIN_ITEM_BYTES - len(data) + 1)
        records = [SampleRecord(meta=_meta(0), payload={"text": padded})]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None
        restored = _round_trip_resolved(coalesced)
        result = restored[0].payload["text"].decode("utf-8")
        assert result.startswith("hello world")

    def test_legacy_shm_bytes_supports_torch_frombuffer(self) -> None:
        """Python < 3.12 fallback should still satisfy torch.frombuffer()."""
        tensor_view = torch.arange(16, dtype=torch.uint8)
        raw = _ShmBytesLegacy(tensor_view)

        assert isinstance(raw, bytes)
        t = torch.frombuffer(raw, dtype=torch.uint8)
        assert t.shape == (16,)
        assert bytes(t.numpy()) == bytes(tensor_view.numpy())

    def test_restored_bytes_work_with_decode_text_op(self) -> None:
        """DecodeText accepts coalesced bytes payloads without special casing."""
        op = DecodeText(fields=("text",), max_batch=3)
        payload = {"text": ("hello world|".encode("utf-8")) * 512}
        records = [SampleRecord(meta=_meta(0), payload=payload)]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        decoded = op.process_many(restored)
        assert decoded[0].payload["text"].startswith("hello world|")

    def test_restored_bytes_flow_through_counting_accumulator(self) -> None:
        """Round-tripped coalesced items keep count-based batching semantics.

        Accumulator routes by meta only (lazy payloads stay unresolved),
        then we resolve after accumulator output to verify data.
        """
        acc = CountingAccumulator[SampleRecord](max_batch=3, max_latency_ms=None)
        records = [
            SampleRecord(
                meta=_meta(i),
                payload={"text": (f"value-{i}|".encode("utf-8")) * 1024, "value": i},
            )
            for i in range(7)
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip(coalesced)
        # Accumulator routes without resolving — only reads .meta
        ready = acc.push_many(restored)
        ready.extend(acc.flush())

        assert [len(batch) for batch, _ in ready] == [3, 3, 1]
        # Now resolve for data verification
        flattened = [rec for batch, _ in ready for rec in batch]
        resolve_lazy_payloads(flattened)
        assert [int(rec.payload["value"]) for rec in flattened] == list(range(7))
        assert all(isinstance(rec.payload["text"], _ShmBytes) for rec in flattened)

    @pytest.mark.skipif(
        sys.version_info < (3, 12), reason="__buffer__ requires Python 3.12+"
    )
    def test_shm_bytes_buffer_protocol(self) -> None:
        """On 3.12+, torch.frombuffer works zero-copy on _ShmBytes."""
        data = b"\x01\x02\x03\x04" * 2048  # 8 KiB
        records = [SampleRecord(meta=_meta(0), payload={"raw": data})]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None
        restored = _round_trip_resolved(coalesced)
        raw = restored[0].payload["raw"]
        t = torch.frombuffer(raw, dtype=torch.uint8)
        assert t.shape == (len(data),)
        assert bytes(t.numpy()) == data
        # Verify zero-copy: the tensor from frombuffer should share the same
        # underlying storage as the SHM-backed _ShmBytes (no data copy).
        assert t.data_ptr() == raw._tensor_view.data_ptr()


# Import threshold for the decode test
from zephon._internal.utils.shm_coalesce import DEFAULT_SHM_MIN_ITEM_BYTES

np = pytest.importorskip("numpy")


# ---------------------------------------------------------------------------
# Numpy ndarray coalescing
# ---------------------------------------------------------------------------
class TestCoalesceNdarray:
    def test_shared_numpy_reduction_preserves_views_and_reuses_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        coalesced = coalesce_microbatch(
            [SampleRecord(_meta(0), np.arange(24))], shm_min_item_bytes=0
        )
        assert coalesced is not None
        base = _round_trip_resolved(coalesced)[0].payload
        source = shm_coalesce._array_owner(base)
        array = base.reshape(4, 6)[::-1, ::2]
        array.flags.writeable = False
        tiny = base.view(np.uint8)[3:6]
        structured = base.view(np.dtype([("first", "<i4"), ("second", "<i4")]))
        payload = {
            "array": array,
            "tiny": tiny,
            "structured": structured,
            "slice": base[3:9],
        }
        records = [SampleRecord(meta=_meta(0), payload=payload)]
        assert coalesce_microbatch(records, shm_min_item_bytes=0) is None
        assert records[0].payload is payload

        storage_reductions = 0
        reduce_storage = ForkingPickler._extra_reducers[torch.UntypedStorage]

        def count_storage(storage: object) -> object:
            nonlocal storage_reductions
            storage_reductions += 1
            return reduce_storage(storage)

        monkeypatch.setitem(
            ForkingPickler._extra_reducers, torch.UntypedStorage, count_storage
        )
        [restored] = pickle.loads(_forking_round_trip(records))
        assert storage_reductions == 1
        result = restored.payload
        for name, expected in payload.items():
            actual = result[name]
            assert type(actual) is np.ndarray
            assert actual.dtype == expected.dtype
            assert actual.strides == expected.strides
            assert actual.flags.writeable == expected.flags.writeable
            np.testing.assert_array_equal(actual, expected)
            owner = shm_coalesce._array_owner(actual)
            assert (
                owner.untyped_storage().data_ptr()
                == source.untyped_storage().data_ptr()
            )
        # A small view retains the same storage, as a shared Torch slice does.
        assert (
            shm_coalesce._array_owner(result["tiny"]).untyped_storage().nbytes()
            == source.numel() * source.element_size()
        )
        # Restored views retain ownership through another serialization hop.
        result = pickle.loads(_forking_round_trip(result))
        assert shm_coalesce._shared_numpy_storage(result["array"]) is not None
        expected = array.copy()
        del source, base, array, tiny, structured, records, payload, restored, owner
        del coalesced
        gc.collect()
        np.testing.assert_array_equal(result["array"], expected)

    def test_numpy_reducer_private_fallback(self) -> None:
        private = torch.arange(8).numpy()[::2]
        values = {
            "private": private,
            "same": private,
            "empty": np.empty((0, 3)),
            "object": np.array([{}]),
        }
        wire = _forking_round_trip(values)
        assert b"zephon" not in wire
        restored = pickle.loads(wire)
        assert restored["private"] is restored["same"]
        private[:] = 99
        np.testing.assert_array_equal(restored["private"], [0, 2, 4, 6])
        assert restored["empty"].shape == (0, 3)
        assert restored["object"].tolist() == [{}]
        cycle = np.empty(1, dtype=object)
        cycle[0] = cycle
        restored_cycle = pickle.loads(_forking_round_trip(cycle))
        assert restored_cycle[0] is restored_cycle

    def test_numpy_reducer_preserves_protocol5_buffers(self) -> None:
        buffers: list[pickle.PickleBuffer] = []
        stream = BytesIO()
        pickler = pickle.Pickler(stream, protocol=5, buffer_callback=buffers.append)
        pickler.dispatch_table = ForkingPickler._extra_reducers.copy()
        pickler.dump(np.arange(8))
        assert len(buffers) == 1
        assert b"zephon" not in stream.getvalue()
        np.testing.assert_array_equal(
            pickle.loads(stream.getvalue(), buffers=buffers), np.arange(8)
        )

    def test_numpy_pickle_preserves_unrelated_shared_array_copy_semantics(self) -> None:
        shared = torch.arange(8).share_memory_().numpy()
        assert shm_coalesce._shared_numpy_storage(shared) is None
        for wire in (pickle.dumps(shared), _forking_round_trip(shared)):
            assert b"zephon" not in wire
            restored = pickle.loads(wire)
            np.testing.assert_array_equal(restored, shared)
            assert not np.shares_memory(restored, shared)

    def test_numpy_reducer_rejects_out_of_bounds_views(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        coalesced = coalesce_microbatch(
            [SampleRecord(_meta(0), np.arange(8))], shm_min_item_bytes=0
        )
        assert coalesced is not None
        shared = _round_trip_resolved(coalesced)[0].payload
        assert shm_coalesce._shared_numpy_storage(shared) is not None
        # Simulate a claimed span outside the owner without accessing invalid memory.
        start = shared.__array_interface__["data"][0]
        monkeypatch.setattr(
            shm_coalesce, "_byte_bounds", lambda _: (start - 1, start + 64)
        )
        assert shm_coalesce._shared_numpy_storage(shared) is None
        copied = pickle.loads(_forking_round_trip(shared))
        np.testing.assert_array_equal(copied, shared)
        assert not np.shares_memory(copied, shared)

    def test_unsupported_numpy_dtypes_stay_inline(self) -> None:
        values = {
            "text": np.array(["ab", "cd"]),
            "structured": np.array([(1, 2.5)], dtype=[("id", "i4"), ("score", "f8")]),
            "non_native": np.array(
                [1, 2], dtype=">i4" if sys.byteorder == "little" else "<i4"
            ),
        }
        payload = dict(values, tensor=torch.ones(1))
        coalesced = coalesce_microbatch([SampleRecord(meta=_meta(0), payload=payload)])
        assert coalesced is not None
        [restored] = _round_trip_resolved(coalesced)
        for name, expected in values.items():
            actual = restored.payload[name]
            inline = pickle.loads(_forking_round_trip(expected))
            assert actual.dtype == inline.dtype
            np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize(
        "array",
        [
            np.arange(2048, dtype=np.uint32),
            np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::-1],
            np.asfortranarray(np.arange(12, dtype=np.int16).reshape(3, 4)),
            np.array(7, dtype=np.int64),
            np.empty((0, 3), dtype=np.float32),
        ],
    )
    def test_root_array_stays_lazy_across_forwarding(self, array: np.ndarray) -> None:
        original = array.copy()
        meta = _meta(0).child(3)
        meta.tags["source"] = "text"
        records = [
            SampleRecord(meta=meta, payload=array),
            # Ensure an empty array can share a message with a nonempty buffer.
            SampleRecord(meta=_meta(1), payload=np.arange(4, dtype=np.float32)),
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        first_hop = _round_trip(coalesced)
        assert isinstance(first_hop[0].payload, _ShmLeafPayload)
        second_hop = pickle.loads(_forking_round_trip(first_hop))
        assert isinstance(second_hop[0], SampleRecord)
        assert isinstance(second_hop[0].payload, _ShmLeafPayload)
        assert second_hop[0].meta == meta
        resolve_lazy_payloads(second_hop)
        result = second_hop[0].payload
        assert isinstance(result, np.ndarray)
        assert result.dtype == original.dtype
        assert result.shape == original.shape
        np.testing.assert_array_equal(result, original)
        del records, coalesced, first_hop, second_hop
        gc.collect()
        np.testing.assert_array_equal(result, original)

    def test_batched_root_arrays_preserve_records_and_metadata(self) -> None:
        arrays = [np.arange(8, dtype=np.uint32) + i for i in range(2)]
        metas = [_meta(i).child(i + 1) for i in range(2)]
        records = tuple(
            SampleRecord(meta=meta, payload=array) for meta, array in zip(metas, arrays)
        )
        coalesced = coalesce_microbatch(
            [SampleBatch(records=records)], shm_min_item_bytes=0
        )
        assert coalesced is not None
        forwarded = pickle.loads(_forking_round_trip(_round_trip(coalesced)))
        assert isinstance(forwarded[0], SampleBatch)
        assert all(isinstance(r.payload, LazyPayload) for r in forwarded[0].records)
        resolve_lazy_payloads(forwarded)
        for record, meta, array in zip(forwarded[0].records, metas, arrays):
            assert isinstance(record, SampleRecord)
            assert record.meta == meta
            np.testing.assert_array_equal(record.payload, array)

    def test_root_arrays_and_nested_payloads_share_one_message(self) -> None:
        root = np.arange(2048, dtype=np.uint32)
        nested = np.arange(6, dtype=np.float64).reshape(2, 3)
        records = [
            SampleRecord(meta=_meta(0), payload=root),
            SampleRecord(meta=_meta(1), payload={"array": nested, "label": "nested"}),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None
        restored = _round_trip_resolved(coalesced)
        np.testing.assert_array_equal(restored[0].payload, root)
        np.testing.assert_array_equal(restored[1].payload["array"], nested)
        assert restored[1].payload["label"] == "nested"

    def test_root_arrays_are_views_into_the_coalesced_buffer(self) -> None:
        records = [
            SampleRecord(meta=_meta(i), payload=np.arange(8, dtype=np.float32) + i)
            for i in range(2)
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        restored = _round_trip_resolved(coalesced)
        buffer = next(iter(coalesced.buffers.values()))
        buffer[0] = 123
        assert restored[0].payload[0] == 123
        assert restored[1].payload.ctypes.data - restored[0].payload.ctypes.data == 32

    def test_single_dtype_round_trip(self) -> None:
        """numpy arrays coalesce and restore with correct values and dtype."""
        records = [
            SampleRecord(
                meta=_meta(i),
                payload={"features": np.arange(5, dtype=np.float64) + i},
            )
            for i in range(3)
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        assert len(coalesced.buffers) == 1

        restored = _round_trip_resolved(coalesced)
        assert len(restored) == 3
        for i, rec in enumerate(restored):
            expected = np.arange(5, dtype=np.float64) + i
            np.testing.assert_array_equal(rec.payload["features"], expected)
            assert rec.payload["features"].dtype == np.float64

    def test_mixed_dtypes(self) -> None:
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={
                    "ints": np.array([1, 2, 3], dtype=np.int32),
                    "floats": np.array([1.0, 2.0], dtype=np.float32),
                },
            )
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        assert len(coalesced.buffers) == 2

        restored = _round_trip_resolved(coalesced)
        np.testing.assert_array_equal(
            restored[0].payload["ints"], np.array([1, 2, 3], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            restored[0].payload["floats"], np.array([1.0, 2.0], dtype=np.float32)
        )

    def test_multidimensional(self) -> None:
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        records = [SampleRecord(meta=_meta(0), payload={"matrix": arr})]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        np.testing.assert_array_equal(restored[0].payload["matrix"], arr)
        assert restored[0].payload["matrix"].shape == (3, 4)

    def test_restored_arrays_are_views_not_copies(self) -> None:
        """Restored numpy arrays should be views into the same SHM storage."""
        records = [
            SampleRecord(
                meta=_meta(i),
                payload={"arr": np.arange(4, dtype=np.float32) + i},
            )
            for i in range(3)
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        # All restored arrays share one underlying torch storage
        ptrs = {rec.payload["arr"].ctypes.data for rec in restored}
        # They should be at different offsets but in the same allocation;
        # at minimum, verify they're contiguous (ptr differences = 4 floats)
        sorted_ptrs = sorted(ptrs)
        for a, b in zip(sorted_ptrs, sorted_ptrs[1:]):
            assert b - a == 4 * 4  # 4 float32s = 16 bytes

    def test_mixed_torch_and_numpy(self) -> None:
        """Torch tensors and numpy arrays coalesce into separate buffers."""
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={
                    "torch_t": torch.tensor([1.0, 2.0], dtype=torch.float32),
                    "np_arr": np.array([3.0, 4.0], dtype=np.float32),
                    "text": "hello",
                },
            )
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        # Separate buffers: torch.float32 vs np:float32
        assert len(coalesced.buffers) == 2

        restored = _round_trip_resolved(coalesced)
        torch.testing.assert_close(
            restored[0].payload["torch_t"],
            torch.tensor([1.0, 2.0], dtype=torch.float32),
        )
        np.testing.assert_array_equal(
            restored[0].payload["np_arr"],
            np.array([3.0, 4.0], dtype=np.float32),
        )
        assert isinstance(restored[0].payload["torch_t"], torch.Tensor)
        assert isinstance(restored[0].payload["np_arr"], np.ndarray)
        assert restored[0].payload["text"] == "hello"

    def test_object_dtype_passthrough(self) -> None:
        """Object-dtype arrays cannot be memcpy'd and must pass through unchanged."""
        obj_arr = np.array(["hello", {"nested": True}], dtype=object)
        normal_arr = np.array([1.0, 2.0], dtype=np.float32)
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={"objects": obj_arr, "normal": normal_arr},
            )
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        # Only the float32 array should be coalesced, not the object array.
        assert len(coalesced.buffers) == 1

        restored = _round_trip_resolved(coalesced)
        np.testing.assert_array_equal(restored[0].payload["objects"], obj_arr)
        assert restored[0].payload["objects"].dtype == object
        np.testing.assert_array_equal(
            restored[0].payload["normal"], np.array([1.0, 2.0], dtype=np.float32)
        )

    def test_noncontiguous_numpy(self) -> None:
        """Strided arrays write directly to SHM without a contiguous temporary."""
        arr_slice = np.arange(10, dtype=np.float32)[::-2]
        arr_slice.flags.writeable = False
        arr_fortran = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4))
        assert not arr_slice.flags["C_CONTIGUOUS"]
        assert not arr_fortran.flags["C_CONTIGUOUS"]
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={"sliced": arr_slice, "fortran": arr_fortran},
            )
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        for name, expected in (("sliced", arr_slice), ("fortran", arr_fortran)):
            actual = restored[0].payload[name]
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == expected.dtype
            assert actual.flags.c_contiguous
        assert restored[0].payload["fortran"].shape == (3, 4)

    def test_empty_array(self) -> None:
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={
                    "empty": np.empty(0, dtype=np.float32),
                    "ok": np.array([1.0]),
                },
            )
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        restored = _round_trip_resolved(coalesced)
        assert restored[0].payload["empty"].size == 0
        np.testing.assert_array_equal(restored[0].payload["ok"], np.array([1.0]))


# ---------------------------------------------------------------------------
# Lazy payload behavior
# ---------------------------------------------------------------------------
class TestRootLeafPayload:
    @pytest.mark.parametrize(
        "tensor",
        [
            torch.arange(8, dtype=torch.int64),
            torch.arange(12, dtype=torch.float64).reshape(3, 4).t(),
            torch.tensor(7, dtype=torch.int16),
            torch.empty((0, 3), dtype=torch.int64),
        ],
    )
    def test_tensor_forwarding_preserves_metadata_and_storage(
        self, tensor: torch.Tensor
    ) -> None:
        meta = _meta(0).child(2)
        meta.tags["source"] = "tensor"
        records = [
            SampleRecord(meta=meta, payload=tensor),
            # An empty int64 root has no buffer of its own dtype.
            SampleRecord(meta=_meta(1), payload=torch.ones(1, dtype=torch.float32)),
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        first_hop = _round_trip(coalesced)
        assert isinstance(first_hop[0].payload, _ShmLeafPayload)
        forwarded = pickle.loads(_forking_round_trip(first_hop))
        assert isinstance(forwarded[0].payload, _ShmLeafPayload)
        assert forwarded[0].meta == meta
        resolve_lazy_payloads(forwarded)
        result = forwarded[0].payload
        torch.testing.assert_close(result, tensor)
        if tensor.numel():
            buffer = coalesced.buffers[str(tensor.dtype)]
            assert result.is_shared()
            assert result.untyped_storage().data_ptr() == buffer.data_ptr()
            buffer[0] = 123
            assert result.reshape(-1)[0] == 123
        expected = result.clone()
        del records, coalesced, first_hop, forwarded
        gc.collect()
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("kind", ["numpy", "torch"])
    @pytest.mark.parametrize("container", ["dict", "list"])
    def test_one_leaf_container_retains_its_structure(
        self, kind: str, container: str
    ) -> None:
        leaf = np.arange(4) if kind == "numpy" else torch.arange(4)
        payload = {"tokens": leaf} if container == "dict" else [leaf]
        coalesced = coalesce_microbatch(
            [SampleRecord(meta=_meta(0), payload=payload)], shm_min_item_bytes=0
        )
        assert coalesced is not None
        forwarded = _round_trip(coalesced)
        assert isinstance(forwarded[0].payload, ShmLazyPayload)
        resolve_lazy_payloads(forwarded)
        result = forwarded[0].payload
        assert type(result) is type(payload)
        restored_leaf = result["tokens"] if container == "dict" else result[0]
        if kind == "numpy":
            np.testing.assert_array_equal(restored_leaf, leaf)
        else:
            torch.testing.assert_close(restored_leaf, leaf)

    @pytest.mark.parametrize("payload", [b"x" * 8192, [1, 2, 3], [1.5, 2.5]])
    def test_other_extracted_roots_forward_lazily(self, payload: object) -> None:
        coalesced = coalesce_microbatch(
            [SampleRecord(meta=_meta(0), payload=payload)], shm_min_item_bytes=0
        )
        assert coalesced is not None
        forwarded = pickle.loads(_forking_round_trip(_round_trip(coalesced)))
        assert isinstance(forwarded[0].payload, _ShmLeafPayload)
        resolve_lazy_payloads(forwarded)
        assert forwarded[0].payload == payload

    def test_inline_roots_can_share_a_message_with_extracted_leaves(self) -> None:
        shared = torch.arange(4).share_memory_()
        objects = np.array(["hello", {"nested": True}], dtype=object)
        payloads = [None, "text", b"tiny", [], (1, "two"), shared, objects]
        records = [
            SampleRecord(meta=_meta(i), payload=p) for i, p in enumerate(payloads)
        ]
        records.append(SampleRecord(meta=_meta(7), payload=np.arange(4)))
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        forwarded = pickle.loads(_forking_round_trip(_round_trip(coalesced)))
        resolve_lazy_payloads(forwarded)
        assert [r.payload for r in forwarded[:5]] == payloads[:5]
        torch.testing.assert_close(forwarded[5].payload, shared)
        assert forwarded[5].payload.untyped_storage().data_ptr() == shared.data_ptr()
        np.testing.assert_array_equal(forwarded[6].payload, objects)


class TestShmLazyPayload:
    def test_unpickle_produces_lazy_payload(self) -> None:
        """Unpickled records have ShmLazyPayload on .payload, meta is accessible."""
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={"t": torch.tensor([1.0, 2.0])},
            )
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip(coalesced)
        assert len(restored) == 1
        assert isinstance(restored[0].payload, ShmLazyPayload)
        # Meta is accessible without resolution
        assert restored[0].meta.chunk_offset == 0
        assert restored[0].meta.sample_id == (0, 0, 0)

    def test_resolve_produces_correct_data(self) -> None:
        """ShmLazyPayload.resolve_payload() returns the original payload structure."""
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={"t": torch.tensor([1.0, 2.0]), "label": "hello"},
            )
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip(coalesced)
        payload = restored[0].payload
        assert isinstance(payload, ShmLazyPayload)
        resolved = payload.resolve_payload()
        assert isinstance(resolved, dict)
        torch.testing.assert_close(resolved["t"], torch.tensor([1.0, 2.0]))
        assert resolved["label"] == "hello"

    def test_lazy_payload_survives_repickling(self) -> None:
        """ShmLazyPayload pickles without resolving and round-trips correctly."""
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={"t": torch.arange(5, dtype=torch.float32)},
            )
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        # Hop 1: CoalescedMicrobatch → list with ShmLazyPayload
        restored1 = _round_trip(coalesced)
        assert isinstance(restored1[0].payload, ShmLazyPayload)

        # Hop 2: re-pickle the lazy records (simulating pump → worker)
        blob2 = _forking_round_trip(restored1)
        restored2 = pickle.loads(blob2)
        assert isinstance(restored2[0].payload, ShmLazyPayload)

        # Worker resolves
        resolve_lazy_payloads(restored2)
        torch.testing.assert_close(
            restored2[0].payload["t"], torch.arange(5, dtype=torch.float32)
        )

    def test_counting_accumulator_routes_lazy_records(self) -> None:
        """CountingAccumulator routes lazy records by meta without resolving."""
        acc = CountingAccumulator[SampleRecord](max_batch=2, max_latency_ms=None)
        records = [
            SampleRecord(
                meta=_meta(i),
                payload={"t": torch.tensor([float(i)])},
            )
            for i in range(3)
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip(coalesced)
        # All records have lazy payloads
        assert all(isinstance(r.payload, ShmLazyPayload) for r in restored)

        # Accumulator routes without resolving
        ready = acc.push_many(restored)
        ready.extend(acc.flush())
        assert [len(batch) for batch, _ in ready] == [2, 1]

        # Verify payloads still lazy in batches
        for batch, _ in ready:
            assert all(isinstance(r.payload, ShmLazyPayload) for r in batch)

        # Resolve and verify data
        all_recs = [r for batch, _ in ready for r in batch]
        resolve_lazy_payloads(all_recs)
        for i, rec in enumerate(all_recs):
            torch.testing.assert_close(rec.payload["t"], torch.tensor([float(i)]))

    def test_resolve_is_noop_on_non_lazy(self) -> None:
        """resolve_lazy_payloads is a no-op for regular records."""
        records = [
            SampleRecord(meta=_meta(0), payload={"x": 42}),
            SampleRecord(meta=_meta(1), payload={"x": 99}),
        ]
        resolve_lazy_payloads(records)
        assert records[0].payload["x"] == 42
        assert records[1].payload["x"] == 99

    def test_lazy_payload_sample_batch(self) -> None:
        """ShmLazyPayload works for SampleBatch records."""
        records = [
            SampleRecord(meta=_meta(0), payload={"t": torch.tensor([1.0])}),
            SampleRecord(meta=_meta(1), payload={"t": torch.tensor([2.0])}),
        ]
        batch = SampleBatch(records=tuple(records))
        coalesced = coalesce_microbatch([batch])
        assert coalesced is not None

        restored = _round_trip(coalesced)
        assert isinstance(restored[0], SampleBatch)
        for rec in restored[0].records:
            assert isinstance(rec.payload, ShmLazyPayload)

        resolve_lazy_payloads(restored)
        torch.testing.assert_close(
            restored[0].records[0].payload["t"], torch.tensor([1.0])
        )
        torch.testing.assert_close(
            restored[0].records[1].payload["t"], torch.tensor([2.0])
        )


# ---------------------------------------------------------------------------
# Structured type support (dataclasses, pydantic, attrs)
# ---------------------------------------------------------------------------
import dataclasses


@dataclasses.dataclass
class DCSample:
    image: object  # torch.Tensor at runtime
    label: str


@dataclasses.dataclass
class DCNested:
    inner: DCSample
    score: float


@dataclasses.dataclass
class DCWithPostInit:
    values: object  # torch.Tensor at runtime
    tag: str
    description: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.description = f"tag={self.tag}"


# Pydantic models — defined only when pydantic is installed.
try:
    from pydantic import BaseModel, computed_field

    _has_pydantic = True
except ImportError:
    _has_pydantic = False

# Attrs — defined only when attrs is installed.
try:
    import attr

    _has_attrs = True
except ImportError:
    _has_attrs = False

if _has_pydantic:

    class PydMessage(BaseModel):  # type: ignore[misc]
        role: str
        token_ids: list[int]

    class PydConversation(BaseModel):  # type: ignore[misc]
        messages: list[PydMessage]
        images: list[object]
        model_config = {"arbitrary_types_allowed": True}

    class PydSample(BaseModel):  # type: ignore[misc]
        image: object  # torch.Tensor at runtime
        label: str

        model_config = {"arbitrary_types_allowed": True}

    class PydNested(BaseModel):  # type: ignore[misc]
        inner: PydSample
        score: float

        model_config = {"arbitrary_types_allowed": True}

    class PydWithComputed(BaseModel):  # type: ignore[misc]
        values: object  # torch.Tensor at runtime
        tag: str

        model_config = {"arbitrary_types_allowed": True}

        @computed_field  # type: ignore[prop-decorator]
        @property
        def description(self) -> str:
            return f"tag={self.tag}"


if _has_attrs:

    @attr.s(auto_attribs=True)
    class AttrsSample:
        image: object  # torch.Tensor at runtime
        label: str


class TestStructDataclass:
    """Dataclass payloads get SHM coalescing through _StructSlot decomposition."""

    def test_dataclass_as_top_level_payload(self) -> None:
        t = torch.arange(6, dtype=torch.float32)
        records = [
            SampleRecord(meta=_meta(0), payload=DCSample(image=t, label="cat")),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        assert isinstance(restored[0].payload, DCSample)
        torch.testing.assert_close(restored[0].payload.image, t)
        assert restored[0].payload.label == "cat"

    def test_dataclass_inside_dict(self) -> None:
        t = torch.randn(4)
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={"sample": DCSample(image=t, label="dog"), "extra": 42},
            ),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        assert isinstance(restored[0].payload["sample"], DCSample)
        torch.testing.assert_close(restored[0].payload["sample"].image, t)
        assert restored[0].payload["sample"].label == "dog"
        assert restored[0].payload["extra"] == 42

    def test_nested_dataclasses(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0])
        records = [
            SampleRecord(
                meta=_meta(0),
                payload=DCNested(inner=DCSample(image=t, label="nested"), score=0.95),
            ),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        p = restored[0].payload
        assert isinstance(p, DCNested)
        assert isinstance(p.inner, DCSample)
        torch.testing.assert_close(p.inner.image, t)
        assert p.inner.label == "nested"
        assert p.score == 0.95

    def test_dataclass_with_post_init(self) -> None:
        t = torch.tensor([10.0, 20.0])
        records = [
            SampleRecord(meta=_meta(0), payload=DCWithPostInit(values=t, tag="test")),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        p = restored[0].payload
        assert isinstance(p, DCWithPostInit)
        torch.testing.assert_close(p.values, t)
        assert p.tag == "test"
        assert p.description == "tag=test"

    def test_dataclass_no_tensors_returns_none(self) -> None:
        records = [
            SampleRecord(
                meta=_meta(0),
                payload=DCSample(image="not_a_tensor", label="text"),
            ),
        ]
        assert coalesce_microbatch(records) is None

    def test_dataclass_multiple_records(self) -> None:
        records = [
            SampleRecord(
                meta=_meta(i),
                payload=DCSample(
                    image=torch.arange(3, dtype=torch.float32) + i, label=f"s{i}"
                ),
            )
            for i in range(5)
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        assert len(coalesced.buffers) == 1

        restored = _round_trip_resolved(coalesced)
        assert len(restored) == 5
        for i, rec in enumerate(restored):
            assert isinstance(rec.payload, DCSample)
            torch.testing.assert_close(
                rec.payload.image, torch.arange(3, dtype=torch.float32) + i
            )
            assert rec.payload.label == f"s{i}"

    def test_dataclass_lazy_payload_survives_repickling(self) -> None:
        t = torch.arange(4, dtype=torch.float32)
        records = [
            SampleRecord(meta=_meta(0), payload=DCSample(image=t, label="re")),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        # Hop 1
        restored1 = _round_trip(coalesced)
        assert isinstance(restored1[0].payload, LazyPayload)

        # Hop 2
        blob2 = _forking_round_trip(restored1)
        restored2 = pickle.loads(blob2)
        assert isinstance(restored2[0].payload, LazyPayload)

        # Resolve
        resolve_lazy_payloads(restored2)
        assert isinstance(restored2[0].payload, DCSample)
        torch.testing.assert_close(restored2[0].payload.image, t)


@pytest.mark.skipif(not _has_pydantic, reason="pydantic not installed")
class TestStructPydantic:
    """Pydantic model payloads get SHM coalescing through _StructSlot decomposition."""

    def test_pydantic_as_top_level_payload(self) -> None:
        t = torch.arange(6, dtype=torch.float32)
        records = [
            SampleRecord(meta=_meta(0), payload=PydSample(image=t, label="cat")),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        assert isinstance(restored[0].payload, PydSample)
        torch.testing.assert_close(restored[0].payload.image, t)
        assert restored[0].payload.label == "cat"

    def test_pydantic_inside_dict(self) -> None:
        t = torch.randn(4)
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={"sample": PydSample(image=t, label="dog"), "extra": 42},
            ),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        assert isinstance(restored[0].payload["sample"], PydSample)
        torch.testing.assert_close(restored[0].payload["sample"].image, t)
        assert restored[0].payload["extra"] == 42

    def test_nested_pydantic(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0])
        records = [
            SampleRecord(
                meta=_meta(0),
                payload=PydNested(inner=PydSample(image=t, label="nested"), score=0.95),
            ),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        p = restored[0].payload
        assert isinstance(p, PydNested)
        assert isinstance(p.inner, PydSample)
        torch.testing.assert_close(p.inner.image, t)
        assert p.inner.label == "nested"
        assert p.score == 0.95

    def test_pydantic_with_computed_field(self) -> None:
        t = torch.tensor([10.0, 20.0])
        records = [
            SampleRecord(meta=_meta(0), payload=PydWithComputed(values=t, tag="test")),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        p = restored[0].payload
        assert isinstance(p, PydWithComputed)
        torch.testing.assert_close(p.values, t)
        assert p.tag == "test"
        assert p.description == "tag=test"

    def test_pydantic_no_tensors_returns_none(self) -> None:
        records = [
            SampleRecord(
                meta=_meta(0),
                payload=PydSample(image="not_a_tensor", label="text"),
            ),
        ]
        assert coalesce_microbatch(records) is None

    def test_pydantic_multiple_records(self) -> None:
        records = [
            SampleRecord(
                meta=_meta(i),
                payload=PydSample(
                    image=torch.arange(3, dtype=torch.float32) + i, label=f"s{i}"
                ),
            )
            for i in range(5)
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        assert len(restored) == 5
        for i, rec in enumerate(restored):
            assert isinstance(rec.payload, PydSample)
            torch.testing.assert_close(
                rec.payload.image, torch.arange(3, dtype=torch.float32) + i
            )

    def test_pydantic_lazy_payload_survives_repickling(self) -> None:
        t = torch.arange(4, dtype=torch.float32)
        records = [
            SampleRecord(meta=_meta(0), payload=PydSample(image=t, label="re")),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored1 = _round_trip(coalesced)
        assert isinstance(restored1[0].payload, LazyPayload)

        blob2 = _forking_round_trip(restored1)
        restored2 = pickle.loads(blob2)
        resolve_lazy_payloads(restored2)
        assert isinstance(restored2[0].payload, PydSample)
        torch.testing.assert_close(restored2[0].payload.image, t)


@pytest.mark.skipif(not _has_attrs, reason="attrs not installed")
class TestStructAttrs:
    """Attrs class payloads get SHM coalescing through _StructSlot decomposition."""

    def test_attrs_as_top_level_payload(self) -> None:
        t = torch.arange(6, dtype=torch.float32)
        records = [
            SampleRecord(meta=_meta(0), payload=AttrsSample(image=t, label="cat")),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        assert isinstance(restored[0].payload, AttrsSample)
        torch.testing.assert_close(restored[0].payload.image, t)
        assert restored[0].payload.label == "cat"

    def test_attrs_inside_dict(self) -> None:
        t = torch.randn(4)
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={"sample": AttrsSample(image=t, label="dog"), "extra": 42},
            ),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        assert isinstance(restored[0].payload["sample"], AttrsSample)
        torch.testing.assert_close(restored[0].payload["sample"].image, t)
        assert restored[0].payload["extra"] == 42


@pytest.mark.skipif(
    not (_has_pydantic and _has_attrs), reason="pydantic and/or attrs not installed"
)
class TestStructMixed:
    """Mixed struct types and cross-framework nesting."""

    @pytest.mark.parametrize("depth", [1, 8])
    def test_deeply_nested_mixed_payloads_keep_structure_and_shared_storage(
        self, depth: int
    ) -> None:
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        array = np.arange(6, dtype=np.uint32).reshape(2, 3)
        # Descriptor-shaped user fields must remain ordinary user data.
        lookalike = {
            "slots": ["user"],
            "spec": None,
            "dtype_key": "np:uint32",
            "offset": 0,
            "shape": (2, 3),
            "bind": "user",
            "restore": "user",
        }
        payload = {"tensor": tensor, "array": array, "raw": b"x" * 8192, **lookalike}
        for level in range(depth):
            payload = DCSample(
                image=[
                    {
                        "child": PydSample(
                            image=AttrsSample(image=payload, label=f"attrs-{level}"),
                            label=f"pyd-{level}",
                        )
                    }
                ],
                label=f"dc-{level}",
            )
        meta = _meta(0).child(3)
        meta.tags["source"] = "nested"
        coalesced = coalesce_microbatch(
            [SampleBatch(records=(SampleRecord(meta=meta, payload=payload),))],
            shm_min_item_bytes=0,
        )
        assert coalesced is not None
        assert set(coalesced.buffers) == {"torch.float32", "np:uint32", "_bytes_uint8"}
        first_hop = _round_trip(coalesced)
        forwarded = pickle.loads(_forking_round_trip(first_hop))
        record = forwarded[0].records[0]
        assert isinstance(record.payload, LazyPayload)
        assert record.meta == meta
        resolve_lazy_payloads(forwarded)
        result = record.payload
        for level in reversed(range(depth)):
            assert isinstance(result, DCSample)
            assert result.label == f"dc-{level}"
            assert isinstance(result.image, list) and len(result.image) == 1
            assert isinstance(result.image[0], dict)
            pyd = result.image[0]["child"]
            assert isinstance(pyd, PydSample) and pyd.label == f"pyd-{level}"
            assert isinstance(pyd.image, AttrsSample)
            assert pyd.image.label == f"attrs-{level}"
            result = pyd.image.image
        assert isinstance(result, dict)
        assert {key: result[key] for key in lookalike} == lookalike
        torch.testing.assert_close(result["tensor"], tensor)
        np.testing.assert_array_equal(result["array"], array)
        assert bytes(result["raw"]) == b"x" * 8192
        # Check actual shared storage, not merely equal values after pickling.
        coalesced.buffers["torch.float32"][0] = 123
        coalesced.buffers["np:uint32"][0] = 456
        assert result["tensor"][0, 0] == 123
        assert result["array"][0, 0] == 456
        if sys.version_info >= (3, 12):
            coalesced.buffers["_bytes_uint8"][0] = ord("z")
            assert bytes(result["raw"]).startswith(b"z")

    def test_pydantic_inside_dataclass(self) -> None:
        t = torch.tensor([1.0, 2.0])
        inner = PydSample(image=t, label="inner")
        records = [
            SampleRecord(meta=_meta(0), payload=DCNested(inner=inner, score=0.5)),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        p = restored[0].payload
        assert isinstance(p, DCNested)
        # inner was PydSample but DCNested.inner is typed as DCSample —
        # reconstruction uses DCNested(**kwargs) so it just stores whatever
        # the resolved value is.
        assert isinstance(p.inner, PydSample)
        torch.testing.assert_close(p.inner.image, t)
        assert p.score == 0.5

    def test_dict_of_mixed_structs(self) -> None:
        t1 = torch.tensor([1.0])
        t2 = torch.tensor([2.0])
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={
                    "dc": DCSample(image=t1, label="dc"),
                    "pyd": PydSample(image=t2, label="pyd"),
                    "plain": "hello",
                },
            ),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        p = restored[0].payload
        assert isinstance(p["dc"], DCSample)
        assert isinstance(p["pyd"], PydSample)
        torch.testing.assert_close(p["dc"].image, t1)
        torch.testing.assert_close(p["pyd"].image, t2)
        assert p["plain"] == "hello"

    def test_list_of_structs(self) -> None:
        records = [
            SampleRecord(
                meta=_meta(0),
                payload=[
                    DCSample(image=torch.tensor([float(i)]), label=f"s{i}")
                    for i in range(3)
                ],
            ),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        p = restored[0].payload
        assert len(p) == 3
        for i, item in enumerate(p):
            assert isinstance(item, DCSample)
            torch.testing.assert_close(item.image, torch.tensor([float(i)]))
            assert item.label == f"s{i}"

    def test_struct_with_numpy_field(self) -> None:
        arr = np.arange(6, dtype=np.float32)
        records = [
            SampleRecord(meta=_meta(0), payload=DCSample(image=arr, label="numpy")),
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        p = restored[0].payload
        assert isinstance(p, DCSample)
        np.testing.assert_array_equal(p.image, arr)
        assert isinstance(p.image, np.ndarray)
        assert p.label == "numpy"

    def test_struct_with_large_bytes_field(self) -> None:
        data = b"X" * 8192
        records = [
            SampleRecord(meta=_meta(0), payload=DCSample(image=data, label="bytes")),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        p = restored[0].payload
        assert isinstance(p, DCSample)
        assert bytes(p.image) == data
        assert p.label == "bytes"

    def test_struct_with_mixed_tensor_types(self) -> None:
        records = [
            SampleRecord(
                meta=_meta(0),
                payload=DCSample(
                    image=torch.randn(3, 4, dtype=torch.float32), label="multi"
                ),
            ),
            SampleRecord(
                meta=_meta(1),
                payload=DCSample(image=torch.arange(5, dtype=torch.int64), label="int"),
            ),
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        assert len(coalesced.buffers) == 2

        restored = _round_trip_resolved(coalesced)
        assert restored[0].payload.image.shape == (3, 4)
        assert restored[0].payload.image.dtype == torch.float32
        assert restored[1].payload.image.shape == (5,)
        assert restored[1].payload.image.dtype == torch.int64


# ---------------------------------------------------------------------------
# Primitive-list-as-leaf optimisation
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class DCWithTokenIds:
    """Mimics a VLM Message: tensor + large List[int] field."""

    image: object  # torch.Tensor at runtime
    token_ids: list[int]
    label: str


class _NumberList(list[int]):
    pass


class TestPrimitiveListAsLeaf:
    """List[int] inside structs must be treated as atomic leaves, not flattened."""

    def test_list_of_ints_round_trips_through_struct(self) -> None:
        token_ids = list(range(500))
        t = torch.randn(3, 224, 224)
        records = [
            SampleRecord(
                meta=_meta(0),
                payload=DCWithTokenIds(image=t, token_ids=token_ids, label="msg"),
            ),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        p = restored[0].payload
        assert isinstance(p, DCWithTokenIds)
        torch.testing.assert_close(p.image, t)
        assert p.token_ids == token_ids
        assert p.label == "msg"

    def test_list_of_ints_becomes_numeric_list_slot(self) -> None:
        """A numeric list becomes a single _NumericListSlot, coalesced into SHM."""
        from zephon._internal.utils.shm_coalesce import (
            _extract_payload_pytree,
            _NumericListSlot,
        )

        token_ids = list(range(500))
        payload = {"ids": token_ids, "tensor": torch.randn(4)}
        collector: dict[str, list] = {}
        offsets: dict[str, int] = {}
        skel = _extract_payload_pytree(
            payload, collector, offsets, shm_min_item_bytes=0
        )
        # dict with 2 keys → 2 leaves (the list as 1 + the tensor slot as 1)
        assert len(skel.slots) == 2
        int_slots = [s for s in skel.slots if isinstance(s, _NumericListSlot)]
        assert len(int_slots) == 1
        assert int_slots[0].length == 500
        # The list lands in the int64 buffer alongside other int64 tensors.
        assert str(torch.int64) in collector

    def test_list_of_tensors_still_recurses(self) -> None:
        """Lists starting with a tensor must NOT be treated as leaf."""
        tensors = [torch.randn(4) for _ in range(3)]
        records = [
            SampleRecord(meta=_meta(0), payload={"items": tensors}),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        for i, t in enumerate(restored[0].payload["items"]):
            torch.testing.assert_close(t, tensors[i])

    def test_list_of_floats_round_trips(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.99, -1.5]
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={"scores": scores, "t": torch.tensor([1.0])},
            ),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        assert restored[0].payload["scores"] == pytest.approx(scores)

    def test_list_of_strings_is_leaf_not_coalesced(self) -> None:
        """String lists are treated as leaves but NOT promoted to SHM."""
        from zephon._internal.utils.shm_coalesce import (
            _extract_payload_pytree,
            _NumericListSlot,
        )

        payload = {"roles": ["system", "user", "assistant"], "t": torch.ones(2)}
        collector: dict[str, list] = {}
        offsets: dict[str, int] = {}
        skel = _extract_payload_pytree(
            payload, collector, offsets, shm_min_item_bytes=0
        )
        # String list should be inline, not a _NumericListSlot
        assert not any(isinstance(s, _NumericListSlot) for s in skel.slots)

        records = [SampleRecord(meta=_meta(0), payload=payload)]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None
        restored = _round_trip_resolved(coalesced)
        assert restored[0].payload["roles"] == ["system", "user", "assistant"]

    def test_empty_list_still_recurses(self) -> None:
        """Empty lists should not match the primitive heuristic."""
        records = [
            SampleRecord(meta=_meta(0), payload={"empty": [], "t": torch.ones(2)}),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        assert restored[0].payload["empty"] == []

    def test_multiple_int_lists_coalesce_into_one_buffer(self) -> None:
        """Several List[int] fields should all land in the same int64 SHM buffer."""
        ids_a = list(range(100))
        ids_b = list(range(200, 500))
        ids_c = list(range(1000, 1050))
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={
                    "a": ids_a,
                    "b": ids_b,
                    "c": ids_c,
                    "img": torch.randn(2, 2),
                },
            ),
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        # int64 buffer should contain all three lists (+ possibly the float tensor in its own)
        assert str(torch.int64) in coalesced.buffers
        assert coalesced.buffers[str(torch.int64)].numel() == 100 + 300 + 50

        restored = _round_trip_resolved(coalesced)
        p = restored[0].payload
        assert p["a"] == ids_a
        assert p["b"] == ids_b
        assert p["c"] == ids_c

    def test_multiple_records_with_int_lists(self) -> None:
        """Int lists across multiple records coalesce into the same buffer."""
        records = [
            SampleRecord(
                meta=_meta(i),
                payload={
                    "token_ids": list(range(i * 100, (i + 1) * 100)),
                    "t": torch.tensor([float(i)]),
                },
            )
            for i in range(4)
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        # All 4 × 100 ints in one buffer
        assert coalesced.buffers[str(torch.int64)].numel() == 400

        restored = _round_trip_resolved(coalesced)
        for i, rec in enumerate(restored):
            assert rec.payload["token_ids"] == list(range(i * 100, (i + 1) * 100))

    def test_homogeneous_lists_round_trip_exactly(self) -> None:
        """Homogeneous lists preserve values and their Python list types."""
        int_ids = [0, 2**63 - 1, -(2**63)]
        float_scores = [0.1 * i for i in range(30)]
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={
                    "ids": int_ids,
                    "scores": float_scores,
                    "t": torch.ones(1),
                },
            ),
        ]

        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None
        assert str(torch.int64) in coalesced.buffers
        assert str(torch.float64) in coalesced.buffers

        restored = _round_trip_resolved(coalesced)
        assert type(restored[0].payload["ids"]) is list
        assert type(restored[0].payload["scores"]) is list
        assert restored[0].payload["ids"] == int_ids
        assert restored[0].payload["scores"] == float_scores

    def test_unsupported_numeric_lists_round_trip_unchanged(self) -> None:
        values = {
            "scores": [1, 0.5, 0.25, 1],
            "large_mixed": [0.5, 2**60 + 1],
            "uint64": [0, 2**63, 2**64 - 1],
            "bool": [True, False],
            "subclass": _NumberList([1, 2, 3]),
        }
        payload = dict(values, tensor=torch.ones(1))
        coalesced = coalesce_microbatch([SampleRecord(meta=_meta(0), payload=payload)])
        assert coalesced is not None
        [restored] = _round_trip_resolved(coalesced)
        for name, expected in values.items():
            actual = restored.payload[name]
            assert actual == expected
            assert type(actual) is type(expected)
            assert [type(value) for value in actual] == [
                type(value) for value in expected
            ]

    def test_single_element_list(self) -> None:
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={"one": [42], "t": torch.ones(1)},
            ),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        assert restored[0].payload["one"] == [42]

    def test_numeric_list_only_payload_still_coalesces(self) -> None:
        """Numeric lists coalesce even when no Torch tensor is in the payload."""
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={"ids": list(range(100)), "more": [1.0, 2.0]},
            ),
        ]
        coalesced = coalesce_microbatch(records, shm_min_item_bytes=0)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        assert restored[0].payload["ids"] == list(range(100))
        assert restored[0].payload["more"] == pytest.approx([1.0, 2.0])

    @pytest.mark.skipif(not _has_pydantic, reason="pydantic not installed")
    def test_pydantic_with_list_int_field(self) -> None:
        """End-to-end: Pydantic model with List[int] (the VLM Message pattern)."""
        msgs = [
            PydMessage(role="system", token_ids=list(range(200))),
            PydMessage(role="user", token_ids=list(range(300))),
        ]
        images = [torch.randn(3, 224, 224), torch.randn(3, 224, 224)]
        conv = PydConversation(messages=msgs, images=images)

        records = [
            SampleRecord(
                meta=_meta(0),
                payload={"conversation": conv, "length": 500},
            ),
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        p = restored[0].payload
        assert isinstance(p["conversation"], PydConversation)
        assert len(p["conversation"].messages) == 2
        assert p["conversation"].messages[0].role == "system"
        assert p["conversation"].messages[0].token_ids == list(range(200))
        assert p["conversation"].messages[1].token_ids == list(range(300))
        for i in range(2):
            torch.testing.assert_close(p["conversation"].images[i], images[i])
        assert p["length"] == 500


def test_memory_policy_applies_the_same_threshold_to_supported_payloads() -> None:
    policy = shm_coalesce.PayloadMemoryPolicy(
        shm_min_item_bytes=64, min_new_allocation_bytes=0
    )
    small = {
        "torch": torch.arange(4),
        "numpy": np.arange(4),
        "bytes": b"abc",
        "view": memoryview(b"abc"),
        "mutable": bytearray(b"abc"),
        "list": [1, 2, 3],
        "tuple": (torch.arange(4), np.arange(4)),
    }
    large = {"torch": torch.arange(16), "numpy": np.arange(16), "bytes": b"x" * 64}
    prepared = coalesce_microbatch(
        [SampleRecord(_meta(0), {"small": small, "large": large})], policy=policy
    )
    assert prepared is not None
    [record] = _round_trip_resolved(prepared)
    actual = record.payload
    assert not actual["small"]["torch"].is_shared()
    assert shm_coalesce._shared_numpy_storage(actual["small"]["numpy"]) is None
    assert actual["large"]["torch"].is_shared()
    assert shm_coalesce._shared_numpy_storage(actual["large"]["numpy"]) is not None
    torch.testing.assert_close(actual["small"]["torch"], small["torch"])
    np.testing.assert_array_equal(actual["small"]["numpy"], small["numpy"])
    assert actual["small"]["bytes"] == actual["small"]["view"] == b"abc"
    assert type(actual["small"]["mutable"]) is bytearray
    assert actual["small"]["list"] == [1, 2, 3]
    assert isinstance(actual["small"]["tuple"], tuple)
    assert not actual["small"]["tuple"][0].is_shared()
    assert shm_coalesce._shared_numpy_storage(actual["small"]["tuple"][1]) is None
    # Taking the inline route does not globally replace Torch's reducer.
    tensor = pickle.loads(_forking_round_trip(torch.arange(4)))
    assert tensor.is_shared()


def test_memory_policy_accounts_for_other_values_using_the_same_buffer() -> None:
    policy = shm_coalesce.PayloadMemoryPolicy(
        shm_min_item_bytes=64,
        min_new_allocation_bytes=256,
        min_forward_bytes=256,
        compact_min_savings_bytes=512,
        compact_above_ratio=2,
    )
    for numpy in (False, True):
        values = [torch.arange(16) for _ in range(4)]
        if numpy:
            values = [value.numpy() for value in values]

        def transport(payloads, active_policy=policy):
            prepared = coalesce_microbatch(
                [SampleRecord(_meta(i), value) for i, value in enumerate(payloads)],
                policy=active_policy,
                ensure_prepared=True,
            )
            return [record.payload for record in _round_trip_resolved(prepared)]

        def storage(value):
            return (
                shm_coalesce._shared_numpy_storage(value)
                if numpy
                else value.untyped_storage()
                if value.is_shared()
                else None
            )

        assert storage(transport(values[:1])[0]) is None
        shared = transport(values)
        assert {storage(value).data_ptr() for value in shared} == {
            storage(shared[0]).data_ptr()
        }
        assert storage(shared[0]).nbytes() == 512
        # The allocation cap can split a batch into groups below the buffer floor.
        capped = shm_coalesce.PayloadMemoryPolicy(
            shm_min_item_bytes=64, min_new_allocation_bytes=256, max_coalesced_bytes=128
        )
        assert all(storage(value) is None for value in transport(values, capped))

        slab = torch.arange(128).share_memory_()
        owner = shm_coalesce._shared_numpy_view(slab) if numpy else slab
        # Each view alone qualifies for compaction, but together they use the slab.
        siblings = transport([owner[i : i + 32] for i in range(0, 128, 32)])
        assert all(storage(value).data_ptr() == slab.data_ptr() for value in siblings)
        [crop] = transport([owner[:32]])
        assert storage(crop).nbytes() == 256
        # Compacted siblings use the same coalescing path as private values.
        compacted = transport([owner[:16], owner[32:48]])
        assert storage(compacted[0]).nbytes() == 256
        assert storage(compacted[0]).data_ptr() == storage(compacted[1]).data_ptr()
        assert compacted[0].tolist() == list(range(16))
        assert compacted[1].tolist() == list(range(32, 48))
        # Existing siblings below the fresh-item floor still share their handle.
        tiny_siblings = transport([owner[i : i + 4] for i in range(0, 128, 4)])
        assert all(
            storage(value).data_ptr() == slab.data_ptr() for value in tiny_siblings
        )


def test_view_compaction_is_shared_by_numpy_and_torch_and_does_not_repeat() -> None:
    source = coalesce_microbatch(
        [SampleRecord(_meta(0), {"t": torch.arange(8192), "n": np.arange(8192)})]
    )
    values = _round_trip_resolved(source)[0].payload
    payload = {"t": values["t"][::512], "n": values["n"][::-512]}
    payload["n"].flags.writeable = False
    policy = shm_coalesce.PayloadMemoryPolicy(
        shm_min_item_bytes=0,
        min_new_allocation_bytes=0,
        min_forward_bytes=0,
        compact_above_ratio=4,
        compact_min_savings_bytes=1024,
    )
    assert policy.shared_action(512, 2048) == "keep"  # Absolute condition alone.
    assert policy.shared_action(32, 256) == "keep"  # Relative condition alone.
    assert policy.shared_action(32, 4096) == "fresh"  # Both conditions.
    records = [SampleRecord(_meta(0), payload)]
    prepared = coalesce_microbatch(records, policy=policy)
    [record] = _round_trip_resolved(prepared)
    torch.testing.assert_close(record.payload["t"], payload["t"])
    np.testing.assert_array_equal(record.payload["n"], payload["n"])
    assert not record.payload["n"].flags.writeable
    assert record.payload["t"].untyped_storage().nbytes() == 128
    assert shm_coalesce._shared_numpy_storage(record.payload["n"]).nbytes() == 128
    assert coalesce_microbatch([record], policy=policy) is None
    # Compaction does not change an independently retained source view.
    assert values["t"].untyped_storage().nbytes() == 65536
    assert shm_coalesce._shared_numpy_storage(values["n"]).nbytes() == 65536


def test_coalescing_limit_and_disable_apply_to_all_buffer_types() -> None:
    for enabled, expected_buffers in ((True, 6), (False, 9)):
        policy = shm_coalesce.PayloadMemoryPolicy(
            shm_min_item_bytes=0,
            min_new_allocation_bytes=0,
            min_forward_bytes=0,
            coalesce=enabled,
            max_coalesced_bytes=128,
        )
        records = [
            SampleRecord(
                _meta(i), {"t": torch.arange(8), "n": np.arange(8), "b": bytes(64)}
            )
            for i in range(3)
        ]
        prepared = coalesce_microbatch(records, policy=policy)
        assert len(prepared.buffers) == expected_buffers
        assert all(
            buf.untyped_storage().nbytes() <= 128 for buf in prepared.buffers.values()
        )
        restored = _round_trip_resolved(prepared)
        assert len(restored) == 3
        for rec in restored:
            assert (
                rec.payload["t"].tolist() == rec.payload["n"].tolist() == list(range(8))
            )
            assert bytes(rec.payload["b"]) == bytes(64)

    # The cap bounds coalescing, not the size of an individual shared value.
    large = coalesce_microbatch(
        [
            SampleRecord(
                _meta(0), {"t": torch.arange(32), "n": np.arange(32), "b": bytes(256)}
            )
        ],
        policy=shm_coalesce.PayloadMemoryPolicy(
            shm_min_item_bytes=0,
            min_new_allocation_bytes=0,
            min_forward_bytes=0,
            max_coalesced_bytes=128,
        ),
    )
    assert len(large.buffers) == 3
    assert all(buf.untyped_storage().nbytes() == 256 for buf in large.buffers.values())


def test_lazy_payload_retains_only_its_own_allocation() -> None:
    prepared = coalesce_microbatch(
        [
            SampleRecord(_meta(0), torch.arange(1024)),
            SampleRecord(_meta(1), np.arange(1024)),
        ]
    )
    restored = _round_trip(prepared)
    assert len(restored[0].payload._buffers) == 1
    assert len(restored[1].payload._buffers) == 1
    assert not (
        restored[0].payload._buffers.keys() & restored[1].payload._buffers.keys()
    )


def test_transport_policy_preserves_retry_records_and_inline_tensors() -> None:
    policy = shm_coalesce.PayloadMemoryPolicy(shm_min_item_bytes=8192)
    tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).t()
    array = shm_coalesce._shared_numpy_view(torch.arange(512).share_memory_())
    readonly = np.arange(6)
    readonly.flags.writeable = False
    meta = SampleMeta((1, 2, 3), 4, 5, 6, {7: 8}, {7: 9}, (2,), {"note": "kept"})
    records = [
        SampleRecord(meta, {"tensor": tensor, "array": array, "readonly": readonly}),
        SampleRecord(meta, "plain"),
    ]
    for protocol in (4, 5):
        transport = shm_coalesce.TransportMicrobatch(records, policy)
        result = pickle.loads(ForkingPickler.dumps(transport, protocol))
        result = pickle.loads(
            ForkingPickler.dumps(
                shm_coalesce.TransportMicrobatch(result, policy), protocol
            )
        )
        resolve_lazy_payloads(result)
        assert result[0].meta == meta
        assert result[0].meta is result[1].meta
        actual = result[0].payload["tensor"]
        assert not actual.is_shared()
        torch.testing.assert_close(actual, tensor)
        actual[0, 0] = -1
        assert tensor[0, 0].item() == 0
        assert records[0].payload["tensor"] is tensor
        assert not tensor.is_shared()
        actual_array = result[0].payload["array"]
        np.testing.assert_array_equal(actual_array, array)
        assert actual_array.flags.writeable
        assert shm_coalesce._shared_numpy_storage(actual_array) is None
        assert records[0].payload["array"] is array
        np.testing.assert_array_equal(result[0].payload["readonly"], readonly)
        assert not result[0].payload["readonly"].flags.writeable


def test_prepared_shared_views_reuse_descriptors_without_reducing_numpy_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.arange(16).share_memory_()
    array = shm_coalesce._shared_numpy_view(tensor)
    payload = {"t": tensor[2:6], "n": array[4:8], "reverse": array[::-1]}
    payload["n"].flags.writeable = False
    calls = 0
    reducer = ForkingPickler._extra_reducers[np.ndarray]

    def reduce_numpy(value):
        nonlocal calls
        calls += 1
        return reducer(value)

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("forwarding views must not allocate SHM")

    monkeypatch.setitem(ForkingPickler._extra_reducers, np.ndarray, reduce_numpy)
    monkeypatch.setattr(shm_coalesce, "_alloc_shm_buffer", unexpected_allocation)
    prepared = coalesce_microbatch(
        [SampleRecord(_meta(0), payload)], shm_min_item_bytes=0, ensure_prepared=True
    )
    [record] = _round_trip_resolved(prepared)
    # The strided view keeps its normal reducer; contiguous views use descriptors.
    assert calls == 1
    torch.testing.assert_close(record.payload["t"], tensor[2:6])
    np.testing.assert_array_equal(record.payload["n"], array[4:8])
    np.testing.assert_array_equal(record.payload["reverse"], array[::-1])
    assert not record.payload["n"].flags.writeable
    assert record.payload["reverse"].strides == array[::-1].strides
    assert (
        record.payload["t"].untyped_storage().data_ptr()
        == tensor.untyped_storage().data_ptr()
    )


def test_view_compaction_does_not_wait_for_its_own_shared_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.arange(8192).share_memory_()
    views = [source[:16], source[32:48], torch.arange(16)]
    policy = shm_coalesce.PayloadMemoryPolicy(
        shm_min_item_bytes=0,
        min_new_allocation_bytes=0,
        min_forward_bytes=0,
        compact_min_savings_bytes=1024,
    )

    def full(*args, **kwargs):
        raise RuntimeError(
            "unable to open shared memory object: No space left on device (28)"
        )

    def unexpected_wait(*args):
        raise AssertionError("optional view compaction must not wait for SHM")

    monkeypatch.setattr(torch.UntypedStorage, "_new_shared", full)
    monkeypatch.setattr(shm_coalesce, "wait_for_shm_space", unexpected_wait)
    prepared = coalesce_microbatch([SampleRecord(_meta(0), views)], policy=policy)
    actual = _round_trip_resolved(prepared)[0].payload
    for result, view in zip(actual, views, strict=True):
        torch.testing.assert_close(result, view)
    assert all(
        value.untyped_storage().data_ptr() == source.data_ptr() for value in actual[:2]
    )
    assert not actual[2].is_shared()


def test_lazy_payload_reapplies_a_changed_transport_policy() -> None:
    records = [SampleRecord(_meta(0), {"t": torch.arange(4), "n": np.arange(4)})]
    prepared = coalesce_microbatch(records, shm_min_item_bytes=0)
    lazy = _round_trip(prepared)
    policy = shm_coalesce.PayloadMemoryPolicy(shm_min_item_bytes=4096)
    restored = pickle.loads(
        _forking_round_trip(shm_coalesce.TransportMicrobatch(lazy, policy))
    )
    resolve_lazy_payloads(restored)
    assert not restored[0].payload["t"].is_shared()
    assert shm_coalesce._shared_numpy_storage(restored[0].payload["n"]) is None


def test_bulk_writes_target_final_shared_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    concatenate, cat = np.concatenate, torch.cat
    destinations: set[int] = set()

    def numpy_write(*args, **kwargs):
        out = kwargs["out"]
        assert out.base.is_shared()
        destinations.add(out.base.data_ptr())
        return concatenate(*args, **kwargs)

    def torch_write(*args, **kwargs):
        out = kwargs["out"]
        assert out.is_shared()
        destinations.add(out.data_ptr())
        return cat(*args, **kwargs)

    monkeypatch.setattr(np, "concatenate", numpy_write)
    monkeypatch.setattr(torch, "cat", torch_write)
    values = [
        {"t": torch.arange(6).reshape(2, 3), "n": np.arange(12)[::-2], "b": b"abc"},
        {"t": torch.arange(4), "n": np.arange(8).reshape(2, 4).T, "b": b"de"},
    ]
    prepared = coalesce_microbatch(
        [SampleRecord(_meta(i), value) for i, value in enumerate(values)],
        shm_min_item_bytes=0,
    )
    assert destinations == {b.data_ptr() for b in prepared.buffers.values()}
    for actual, expected in zip(_round_trip_resolved(prepared), values):
        torch.testing.assert_close(actual.payload["t"], expected["t"])
        np.testing.assert_array_equal(actual.payload["n"], expected["n"])
        assert bytes(actual.payload["b"]) == expected["b"]


def test_single_tensor_transport_preserves_private_input_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocations = []
    allocate = shm_coalesce._alloc_shm_buffer

    def tracked(numel, dtype, label, **kwargs):
        allocations.append(numel)
        return allocate(numel, dtype, label, **kwargs)

    monkeypatch.setattr(shm_coalesce, "_alloc_shm_buffer", tracked)
    policy = shm_coalesce.PayloadMemoryPolicy(
        shm_min_item_bytes=0, min_new_allocation_bytes=0, min_forward_bytes=0
    )
    for tensor in (
        torch.arange(32),
        torch.arange(64)[:32],
        (torch.arange(32) + 1j).conj(),
    ):
        actual = pickle.loads(
            _forking_round_trip(
                shm_coalesce.TransportMicrobatch(
                    [SampleRecord(_meta(0), tensor)], policy
                )
            )
        )
        resolve_lazy_payloads(actual)
        torch.testing.assert_close(actual[0].payload, tensor)
        assert actual[0].payload.is_shared()
        assert actual[0].payload.untyped_storage().nbytes() == 256
        assert not tensor.is_shared()
    assert allocations == [32, 32, 32]


def test_primitive_tuples_stay_whole_beside_shared_tensors() -> None:
    tokens = tuple(range(2048))
    payload = {"tokens": tokens, "tensors": (torch.arange(8), torch.arange(8))}
    prepared = coalesce_microbatch(
        [SampleRecord(_meta(0), payload)], shm_min_item_bytes=0
    )
    assert len(prepared.skeleton[0].payload.slots) == 3
    actual = _round_trip_resolved(prepared)[0].payload
    assert actual["tokens"] == tokens
    assert type(actual["tokens"]) is tuple
    assert all(tensor.is_shared() for tensor in actual["tensors"])


@pytest.mark.parametrize("memmap", [False, True], ids=["subclass", "memmap"])
def test_numpy_subclasses_use_shared_transport(tmp_path, memmap: bool) -> None:
    class Array(np.ndarray):
        pass

    if memmap:
        path = tmp_path / "tokens.npy"
        np.save(path, np.arange(8192))
        source = np.load(path, mmap_mode="r")[::2]
    else:
        source = np.arange(8192).view(Array)[::2]
    policy = shm_coalesce.PayloadMemoryPolicy(
        shm_min_item_bytes=0, min_new_allocation_bytes=0
    )
    encoded = _forking_round_trip(
        shm_coalesce.TransportMicrobatch([SampleRecord(_meta(0), source)], policy)
    )
    assert len(encoded) < source.nbytes // 2
    [record] = pickle.loads(encoded)
    resolve_lazy_payloads([record])
    assert type(record.payload) is np.ndarray
    assert record.payload.flags.writeable == source.flags.writeable
    assert shm_coalesce._shared_numpy_storage(record.payload) is not None
    np.testing.assert_array_equal(record.payload, source)


@pytest.mark.parametrize("shared", [False, True], ids=["inline", "shm"])
def test_parameter_transport_preserves_values_grad_and_retry_storage(
    shared: bool,
) -> None:
    source = torch.nn.Parameter(torch.arange(16, dtype=torch.float32))
    address = source.data_ptr()
    policy = shm_coalesce.PayloadMemoryPolicy(
        shm_min_item_bytes=0, min_new_allocation_bytes=0 if shared else 1024
    )
    encoded = _forking_round_trip(
        shm_coalesce.TransportMicrobatch([SampleRecord(_meta(0), source)], policy)
    )
    [record] = pickle.loads(encoded)
    resolve_lazy_payloads([record])
    actual = record.payload
    torch.testing.assert_close(actual, source)
    assert actual.requires_grad
    assert actual.is_leaf
    assert actual.is_shared() == shared
    assert not source.is_shared()
    assert source.data_ptr() == address


def test_numpy_view_does_not_duplicate_storage_in_plain_tensor_pickle() -> None:
    buffer = torch.arange(65536, dtype=torch.uint8).share_memory_()
    array = shm_coalesce._shared_numpy_view(buffer)
    encoded = pickle.dumps(buffer)
    assert len(encoded) < buffer.numel() + 4096
    assert shm_coalesce._shared_numpy_storage(array) is not None
    torch.testing.assert_close(pickle.loads(encoded), buffer)
