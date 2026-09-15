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
        coalesced = coalesce_microbatch(records)
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
        coalesced = coalesce_microbatch(records)
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
        coalesced = coalesce_microbatch(records)
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
        coalesced = coalesce_microbatch(records)
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

    def test_restored_bytes_are_shm_backed(self) -> None:
        """Restored bytes are _ShmBytes backed by SHM tensor."""
        data = b"M" * 8192
        records = [SampleRecord(meta=_meta(0), payload={"raw": data})]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        raw = restored[0].payload["raw"]
        assert isinstance(raw, _ShmBytes)
        assert bytes(raw) == data
        assert len(raw) == len(data)

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
        coalesced = coalesce_microbatch(records)
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
        """memoryview payloads are also packed into the bytes SHM buffer."""
        data = b"X" * 8192
        records = [SampleRecord(meta=_meta(0), payload={"mv": memoryview(data)})]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None
        restored = _round_trip_resolved(coalesced)
        assert bytes(restored[0].payload["mv"]) == data
        assert isinstance(restored[0].payload["mv"], _ShmBytes)

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
        padded = data + b"\x00" * (DEFAULT_SHM_MIN_SIZE - len(data) + 1)
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
        coalesced = coalesce_microbatch(records)
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
from zephon._internal.utils.shm_coalesce import DEFAULT_SHM_MIN_SIZE

np = pytest.importorskip("numpy")


# ---------------------------------------------------------------------------
# Numpy ndarray coalescing
# ---------------------------------------------------------------------------
class TestCoalesceNdarray:
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
        coalesced = coalesce_microbatch(records)
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
        coalesced = coalesce_microbatch([SampleBatch(records=records)])
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
        coalesced = coalesce_microbatch(records)
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
        coalesced = coalesce_microbatch(records)
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
        coalesced = coalesce_microbatch(records)
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
        coalesced = coalesce_microbatch(records)
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
        coalesced = coalesce_microbatch(records)
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
        coalesced = coalesce_microbatch(records)
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
        coalesced = coalesce_microbatch(records)
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
        """Non-contiguous numpy arrays (slices, F-order) coalesce correctly."""
        arr_slice = np.arange(10, dtype=np.float32)[::2]
        arr_fortran = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4))
        assert not arr_slice.flags["C_CONTIGUOUS"]
        assert not arr_fortran.flags["C_CONTIGUOUS"]
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={"sliced": arr_slice, "fortran": arr_fortran},
            )
        ]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip_resolved(coalesced)
        np.testing.assert_array_equal(restored[0].payload["sliced"], arr_slice)
        np.testing.assert_array_equal(restored[0].payload["fortran"], arr_fortran)
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
        coalesced = coalesce_microbatch(records)
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
        coalesced = coalesce_microbatch(records)
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
        coalesced = coalesce_microbatch([SampleRecord(meta=_meta(0), payload=payload)])
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
        coalesced = coalesce_microbatch([SampleRecord(meta=_meta(0), payload=payload)])
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
        coalesced = coalesce_microbatch(records)
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
        coalesced = coalesce_microbatch(records)
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
            [SampleBatch(records=(SampleRecord(meta=meta, payload=payload),))]
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
        coalesced = coalesce_microbatch(records)
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
        coalesced = coalesce_microbatch(records)
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
        skel = _extract_payload_pytree(payload, collector, offsets, shm_min_size=4096)
        # dict with 2 keys → 2 leaves (the list as 1 + the tensor slot as 1)
        assert len(skel.slots) == 2
        int_slots = [s for s in skel.slots if isinstance(s, _NumericListSlot)]
        assert len(int_slots) == 1
        assert int_slots[0].length == 500
        # The list's tensor lands in the int64 buffer alongside other int64 tensors
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
        skel = _extract_payload_pytree(payload, collector, offsets, shm_min_size=4096)
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
        coalesced = coalesce_microbatch(records)
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
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None
        # All 4 × 100 ints in one buffer
        assert coalesced.buffers[str(torch.int64)].numel() == 400

        restored = _round_trip_resolved(coalesced)
        for i, rec in enumerate(restored):
            assert rec.payload["token_ids"] == list(range(i * 100, (i + 1) * 100))

    def test_mixed_int_and_float_lists(self) -> None:
        """Int and float lists go into separate dtype buffers."""
        int_ids = list(range(50))
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
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None
        assert str(torch.int64) in coalesced.buffers
        assert str(torch.float64) in coalesced.buffers

        restored = _round_trip_resolved(coalesced)
        assert restored[0].payload["ids"] == int_ids
        assert restored[0].payload["scores"] == pytest.approx(float_scores)

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
        """A payload with ONLY numeric lists (no tensors) should still coalesce."""
        records = [
            SampleRecord(
                meta=_meta(0),
                payload={"ids": list(range(100)), "more": [1.0, 2.0]},
            ),
        ]
        coalesced = coalesce_microbatch(records)
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
