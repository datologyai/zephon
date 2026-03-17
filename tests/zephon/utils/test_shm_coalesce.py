"""Tests for the tensor coalescing SHM optimization."""

from __future__ import annotations

import pickle
import sys
from io import BytesIO
from multiprocessing.reduction import ForkingPickler

import pytest

from zephon.core.accumulators import CountingAccumulator
from zephon.core.constants import SampleBatch, SampleMeta, SampleRecord, lane_of
from zephon.ops.decode_text import DecodeText
from zephon.utils.shm_coalesce import (
    CoalescedMicrobatch,
    _ShmBytes,
    _ShmBytesLegacy,
    coalesce_microbatch,
)

torch = pytest.importorskip("torch")


def _forking_round_trip(obj: object) -> bytes:
    buf = BytesIO()
    ForkingPickler(buf).dump(obj)
    return buf.getvalue()


def _meta(i: int) -> SampleMeta:
    return SampleMeta(sample_id=(0, 0, i), lane_id=0, chunk_id=0, chunk_offset=i)


def _round_trip(coalesced: CoalescedMicrobatch) -> list:
    """ForkingPickler round-trip (simulates worker → pump thread)."""
    blob = _forking_round_trip(coalesced)
    restored = pickle.loads(blob)
    # Eager __reduce__: unpickle produces list[StreamItem], not CoalescedMicrobatch.
    assert isinstance(restored, list)
    return restored


# ---------------------------------------------------------------------------
# Basic round-trip
# ---------------------------------------------------------------------------
class TestCoalesceMicrobatchRoundTrip:
    """Coalesce → ForkingPickler → pickle.loads should recover original data."""

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

        restored = _round_trip(coalesced)
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

        restored = _round_trip(coalesced)
        assert len(restored) == 2
        for rec in restored:
            assert rec.payload["tokens"].shape == (3,)
            assert rec.payload["features"].shape == (4,)

    def test_multidimensional_tensors(self) -> None:
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        records = [SampleRecord(meta=_meta(0), payload={"matrix": t})]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip(coalesced)
        torch.testing.assert_close(restored[0].payload["matrix"], t)
        assert restored[0].payload["matrix"].shape == (3, 4)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestCoalesceEdgeCases:
    def test_no_tensors_returns_none(self) -> None:
        records = [SampleRecord(meta=_meta(0), payload={"text": "hello", "count": 42})]
        assert coalesce_microbatch(records) is None

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
        monkeypatch.setattr("zephon.utils.shm_coalesce._torch", None)
        monkeypatch.setattr("zephon.utils.shm_coalesce._torch_loaded", True)
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
        restored = _round_trip(coalesced)
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
        restored = _round_trip(coalesced)
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

        restored = _round_trip(coalesced)
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

        restored = _round_trip(coalesced)
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

        restored = _round_trip(coalesced)
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
        restored = _round_trip(coalesced)
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

        restored = _round_trip(coalesced)
        assert bytes(restored[0].payload["raw"]) == data_a
        assert restored[0].payload["label"] == "hello"
        assert bytes(restored[1].payload["raw"]) == data_b

    def test_restored_bytes_are_shm_backed(self) -> None:
        """Restored bytes are _ShmBytes backed by SHM tensor."""
        data = b"M" * 8192
        records = [SampleRecord(meta=_meta(0), payload={"raw": data})]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        restored = _round_trip(coalesced)
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

        restored = _round_trip(coalesced)
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
        restored = _round_trip(coalesced)
        assert bytes(restored[0].payload["mv"]) == data
        assert isinstance(restored[0].payload["mv"], _ShmBytes)

    def test_shm_bytes_survives_repickling_zero_copy(self) -> None:
        """Full round-trip: worker→pump→worker2, bytes stay in SHM throughout.

        Simulates the real pipeline path:
          1. Worker coalesces bytes into SHM, puts CoalescedMicrobatch on queue
          2. Pump thread unpickles (hop 1) → gets list[StreamItem] with _ShmBytes
          3. Pump thread puts records on next worker's queue (hop 2)
          4. Worker 2 unpickles → gets _ShmBytes still backed by same SHM

        The data_ptr must be identical across all hops — proving bytes never
        leave SHM and only FDs are passed through the pickle stream.
        """
        data = b"Z" * 8192
        records = [SampleRecord(meta=_meta(0), payload={"raw": data})]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None

        # Hop 1: worker → pump thread (CoalescedMicrobatch → list[StreamItem])
        restored = _round_trip(coalesced)
        raw1 = restored[0].payload["raw"]
        assert isinstance(raw1, _ShmBytes)
        assert bytes(raw1) == data

        # Hop 2: pump thread → worker 2 (_ShmBytes re-pickled via FD passing)
        blob2 = _forking_round_trip(restored)
        restored2 = pickle.loads(blob2)
        raw2 = restored2[0].payload["raw"]
        assert isinstance(raw2, _ShmBytes)
        assert bytes(raw2) == data

        # On 3.12+ _ShmBytes312 keeps a tensor view — assert zero-copy.
        # On older Python the legacy path copies, so skip the pointer check.
        if sys.version_info >= (3, 12):
            ptr1 = raw1._tensor_view.data_ptr()
            ptr2 = raw2._tensor_view.data_ptr()
            assert ptr1 == ptr2, (
                f"data_ptr changed across hops ({ptr1:#x} → {ptr2:#x}), "
                "bytes were copied instead of staying in SHM"
            )

    def test_shm_bytes_decode(self) -> None:
        """_ShmBytes.decode() works for text ops."""
        text = "hello world"
        data = text.encode("utf-8")
        # Pad to exceed threshold
        padded = data + b"\x00" * (DEFAULT_SHM_MIN_SIZE - len(data) + 1)
        records = [SampleRecord(meta=_meta(0), payload={"text": padded})]
        coalesced = coalesce_microbatch(records)
        assert coalesced is not None
        restored = _round_trip(coalesced)
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

        restored = _round_trip(coalesced)
        decoded = op.process_many(restored)
        assert decoded[0].payload["text"].startswith("hello world|")

    def test_restored_bytes_flow_through_counting_accumulator(self) -> None:
        """Round-tripped coalesced items keep count-based batching semantics."""
        acc = CountingAccumulator[SampleRecord](
            max_batch=3, max_latency_ms=None, key_fn=lane_of
        )
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
        ready = acc.push_many(restored)
        ready.extend(acc.flush())

        assert [len(batch) for batch, _ in ready] == [3, 3, 1]
        flattened = [rec for batch, _ in ready for rec in batch]
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
        restored = _round_trip(coalesced)
        raw = restored[0].payload["raw"]
        t = torch.frombuffer(raw, dtype=torch.uint8)
        assert t.shape == (len(data),)
        assert bytes(t.numpy()) == data
        # Verify zero-copy: the tensor from frombuffer should share the same
        # underlying storage as the SHM-backed _ShmBytes (no data copy).
        assert t.data_ptr() == raw._tensor_view.data_ptr()


# Import threshold for the decode test
from zephon.utils.shm_coalesce import DEFAULT_SHM_MIN_SIZE

np = pytest.importorskip("numpy")


# ---------------------------------------------------------------------------
# Numpy ndarray coalescing
# ---------------------------------------------------------------------------
class TestCoalesceNdarray:
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

        restored = _round_trip(coalesced)
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

        restored = _round_trip(coalesced)
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

        restored = _round_trip(coalesced)
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

        restored = _round_trip(coalesced)
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

        restored = _round_trip(coalesced)
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

        restored = _round_trip(coalesced)
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

        restored = _round_trip(coalesced)
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
        restored = _round_trip(coalesced)
        assert restored[0].payload["empty"].size == 0
        np.testing.assert_array_equal(restored[0].payload["ok"], np.array([1.0]))
