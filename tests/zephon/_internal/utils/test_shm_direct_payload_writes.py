"""Populate final shared buffers without changing restored payload contracts."""

import pickle
from multiprocessing.reduction import ForkingPickler
from typing import Any

import pytest

from zephon._internal.stream import resolve_lazy_payloads
from zephon._internal.utils import shm_coalesce
from zephon.types import SampleMeta, SampleRecord

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


def _record(payload: Any) -> SampleRecord:
    return SampleRecord(SampleMeta((0, 0, 0), 0, 0), payload)


def _roundtrip(payload: Any) -> Any:
    message = shm_coalesce.coalesce_microbatch([_record(payload)], shm_min_size=0)
    assert message is not None
    records = pickle.loads(ForkingPickler.dumps(message))
    resolve_lazy_payloads(records)
    return records[0].payload


@pytest.mark.parametrize(
    "kind",
    [
        "slice",
        "reverse",
        "fortran",
        "broadcast",
        "readonly",
        "scalar",
        "complex",
        "empty",
    ],
)
def test_numpy_writes_directly_to_final_buffer(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    base = np.arange(24, dtype=np.int64).reshape(4, 6)
    choices = {
        "slice": base[:, ::2],
        "reverse": base[::-1, ::-2],
        "fortran": np.asfortranarray(base),
        "broadcast": np.broadcast_to(base[:1, :1], (3, 4)),
        "readonly": base.view(),
        "scalar": np.array(7, dtype=np.uint32),
        "complex": base.astype(np.complex128) * (1 + 2j),
        "empty": np.empty((0, 3), dtype=np.float32),
    }
    source = choices[kind]
    if kind == "readonly":
        source.flags.writeable = False
    expected = source.copy()
    from_numpy = torch.from_numpy

    def reject_staging(*args: Any, **kwargs: Any) -> None:
        pytest.fail("NumPy payloads must not be made contiguous in a temporary buffer")

    def only_empty_numpy(array: Any) -> Any:
        assert array.size == 0, "Torch is needed only to resolve the dtype"
        return from_numpy(array)

    monkeypatch.setattr(np, "ascontiguousarray", reject_staging)
    monkeypatch.setattr(torch, "from_numpy", only_empty_numpy)
    result = _roundtrip({"array": source, "keep_nonempty": np.arange(1)})["array"]
    assert type(result) is np.ndarray
    assert result.dtype == source.dtype
    assert result.shape == source.shape
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "kind",
    [
        "bytes",
        "typed",
        "multidimensional",
        "strided",
        "reverse_typed",
        "fortran",
        "structured",
    ],
)
def test_memoryviews_preserve_all_bytes(kind: str) -> None:
    source = bytearray(range(256)) * 32
    structured = np.zeros(
        16,
        dtype=np.dtype(
            {
                "names": ["x", "y"],
                "formats": ["i1", "i1"],
                "offsets": [0, 7],
                "itemsize": 8,
            }
        ),
    )
    structured.view(np.uint8)[:] = np.arange(structured.nbytes, dtype=np.uint8)
    views = {
        "bytes": memoryview(source),
        "typed": memoryview(source).cast("I"),
        "multidimensional": memoryview(source).cast("I", shape=(32, 64)),
        "strided": memoryview(source)[::2],
        "reverse_typed": memoryview(source).cast("I")[::-2],
        "fortran": memoryview(
            np.asfortranarray(np.arange(24, dtype=np.float64).reshape(4, 6))
        ),
        "structured": memoryview(structured)[::2],
    }
    view = views[kind]
    collector: dict[str, list[Any]] = {}
    shm_coalesce._extract_record_payload(view, collector, {}, 0)
    [collected] = collector[shm_coalesce._BYTES_DTYPE_KEY]
    if view.c_contiguous:
        assert isinstance(collected, memoryview)
        assert collected.obj is view.obj
        assert collected.nbytes == view.nbytes
    elif kind != "structured":
        assert isinstance(collected, np.ndarray)
        assert not collected.flags.owndata
        assert isinstance(collected.base, memoryview)
        assert collected.base.obj is view.obj
    result = _roundtrip(view)
    assert isinstance(result, shm_coalesce._ShmBytes)
    assert bytes(result) == view.tobytes()
    assert len(result) == view.nbytes


@pytest.mark.parametrize(
    "values", [[0, 2**63 - 1, -(2**63)], [0.0, -0.0, 1.25, float("inf"), float("nan")]]
)
def test_homogeneous_lists_fill_shared_storage_directly(
    monkeypatch: pytest.MonkeyPatch, values: list[Any]
) -> None:
    companion = torch.arange(4, dtype=torch.int64)

    def reject_temporary(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Homogeneous lists must not allocate a temporary tensor")

    monkeypatch.setattr(torch, "tensor", reject_temporary)
    result = _roundtrip({"values": values, "tensor": companion})
    assert type(result["values"]) is list
    assert all(type(value) is type(values[0]) for value in result["values"])
    np.testing.assert_array_equal(result["values"], values)
    np.testing.assert_array_equal(result["tensor"].numpy(), companion.numpy())


@pytest.mark.parametrize("values", [[1, 2.75, 3], [1.25, 2, 3.5], [True, False, True]])
def test_mixed_lists_retain_existing_torch_conversion(values: list[Any]) -> None:
    dtype = torch.int64 if isinstance(values[0], int) else torch.float64
    expected = torch.tensor(values, dtype=dtype).tolist()
    result = _roundtrip(values)
    assert type(result) is list
    assert result == expected
    assert [type(value) for value in result] == [type(value) for value in expected]


def test_integer_overflow_keeps_records_intact() -> None:
    values = [0, 2**63]
    with pytest.raises(Exception) as baseline:
        torch.tensor(values, dtype=torch.int64)
    record = _record(values)
    with pytest.raises(type(baseline.value), match=str(baseline.value)):
        shm_coalesce.coalesce_microbatch([record])
    assert record.payload is values


def test_strided_typed_view_at_unaligned_byte_offset() -> None:
    view = memoryview(np.arange(12, dtype=np.int32))[::2]
    result = _roundtrip({"a": b"abc", "b": view, "c": b"tail"})
    assert bytes(result["a"]) == b"abc"
    assert bytes(result["b"]) == view.tobytes()
    assert bytes(result["c"]) == b"tail"


def test_numeric_lists_keep_torch_fallback_without_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shm_coalesce, "_get_numpy", lambda: None)
    assert _roundtrip([1, 2, 3]) == [1, 2, 3]
