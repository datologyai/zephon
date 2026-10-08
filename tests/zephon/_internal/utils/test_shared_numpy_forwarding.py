"""Forward resolved NumPy views while retaining their shared storage."""

import gc
import pickle
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import Any

import pytest

from zephon._internal.stream import resolve_lazy_payloads
from zephon._internal.utils import shm_coalesce
from zephon.types import SampleBatch, SampleMeta, SampleRecord, StreamItem

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


def _record(payload: Any) -> SampleRecord:
    return SampleRecord(SampleMeta((0, 0, 0), 0, 0), payload)


def _resolve(message: object) -> list:
    records = pickle.loads(ForkingPickler.dumps(message))
    resolve_lazy_payloads(records)
    return records


@pytest.mark.parametrize(
    "view",
    ["slice", "reverse", "transpose", "reinterpret", "readonly", "broadcast", "scalar"],
)
def test_recoalescing_reuses_shared_numpy_storage(
    monkeypatch: pytest.MonkeyPatch, view: str
) -> None:
    initial = shm_coalesce.coalesce_microbatch([_record(np.arange(24, dtype=np.int64))])
    [record] = _resolve(initial)
    base = record.payload
    choices = {
        "slice": base[2::3],
        "reverse": base[::-2],
        "transpose": base.reshape(4, 6).T,
        "reinterpret": base.view(np.uint8)[3::5],
        "readonly": base.view(),
        "broadcast": np.broadcast_to(base[:1], (4,)),
        "scalar": base[:1].reshape(()),
    }
    array = choices[view]
    if view == "readonly":
        array.flags.writeable = False
    record.payload = array
    expected = array.copy()

    def reject_allocation(*args: object, **kwargs: object) -> None:
        pytest.fail("Already shared arrays must not allocate another shared buffer")

    monkeypatch.setattr(shm_coalesce, "_alloc_shm_buffer", reject_allocation)
    forwarded = shm_coalesce.coalesce_microbatch([record])
    assert forwarded is not None
    [restored] = _resolve(forwarded)
    result = restored.payload
    assert type(result) is np.ndarray
    assert result.dtype == array.dtype
    assert result.strides == array.strides
    assert result.flags.writeable == array.flags.writeable
    np.testing.assert_array_equal(result, expected)
    base[:] = 7
    np.testing.assert_array_equal(result, array)
    assert not np.array_equal(result, expected)
    del initial, forwarded, restored, record, array, choices, base
    gc.collect()
    assert result.size > 0


def test_shared_and_private_numpy_arrays_in_one_message() -> None:
    base = torch.arange(32).share_memory_().numpy()
    records: list[StreamItem] = [
        _record({"shared": base[4:12], "private": np.arange(8)})
    ]
    message = shm_coalesce.coalesce_microbatch(records)
    assert message is not None
    assert len(message.buffers) == 2
    [restored] = _resolve(message)
    base[4] = 100
    assert restored.payload["shared"][0] == 100
    assert restored.payload["private"][0] == 0


@dataclass
class _Payload:
    array: object
    label: str


def test_dispatch_preserves_original_records_and_deduplicates_storage() -> None:
    base = torch.arange(262144).share_memory_().numpy()
    array = base[2:]
    payload = _Payload(array, "label")
    record = _record(payload)
    batch = SampleBatch(records=(record, _record({"array": base[::-1]})))
    message = shm_coalesce.forward_shared_numpy([batch])
    assert message is not None
    assert len(message.buffers) == 1
    assert record.payload is payload
    assert payload.array is array
    wire = ForkingPickler.dumps(message)
    assert len(wire) < 4096
    [restored] = _resolve(message)
    base[2] = 123
    assert restored.records[0].payload.array[0] == 123
    assert restored.records[0].payload.label == "label"
    assert restored.records[1].payload["array"][-3] == 123
    # This branch does not register a global NumPy reducer or change MTP.
    assert len(ForkingPickler.dumps(array)) > array.nbytes


def test_dispatch_does_not_coalesce_private_values() -> None:
    payload = {"array": np.arange(16), "tokens": [1, 2, 3], "bytes": b"a" * 8192}
    record = _record(payload)
    assert shm_coalesce.forward_shared_numpy([record]) is None
    assert record.payload is payload


def test_lazy_numpy_dispatch_remains_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    message = shm_coalesce.coalesce_microbatch([_record(np.arange(16))])
    records = pickle.loads(ForkingPickler.dumps(message))
    lazy = records[0].payload

    def reject_traversal(*args: object, **kwargs: object) -> None:
        pytest.fail("Lazy-only dispatch must not re-extract payloads")

    monkeypatch.setattr(shm_coalesce, "_extract_from_records", reject_traversal)
    assert shm_coalesce.forward_shared_numpy(records) is None
    assert records[0].payload is lazy


@pytest.mark.parametrize("batched", [False, True])
def test_dispatch_mixes_lazy_and_resolved_numpy(batched: bool) -> None:
    message = shm_coalesce.coalesce_microbatch([_record(np.arange(16))])
    [lazy_record] = pickle.loads(ForkingPickler.dumps(message))
    lazy = lazy_record.payload
    owner = torch.arange(16).share_memory_()
    records = [lazy_record, _record(owner.numpy())]
    items = [SampleBatch(records=tuple(records))] if batched else records
    forwarded = shm_coalesce.forward_shared_numpy(items)
    assert forwarded is not None
    restored = _resolve(forwarded)
    if batched:
        restored = restored[0].records
    for rec in restored:
        assert type(rec.payload) is np.ndarray
        np.testing.assert_array_equal(rec.payload, np.arange(16))
    assert lazy_record.payload is lazy
    owner[0] = 123
    assert restored[1].payload[0] == 123


def test_private_torch_backed_numpy_is_still_copied() -> None:
    owner = torch.arange(16)
    array = owner.numpy()
    assert shm_coalesce.forward_shared_numpy([_record(array)]) is None
    [restored] = _resolve(shm_coalesce.coalesce_microbatch([_record(array)]))
    owner[0] = 99
    assert restored.payload[0] == 0
