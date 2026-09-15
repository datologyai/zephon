"""Unit tests for optional LitData dependencies and serializer access."""

import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

pytest.importorskip("litdata")

from zephon._internal.io.formats.litdata_support import dependencies
from zephon._internal.io.formats.litdata_support.dependencies import (
    _NUMPY_DTYPES_REVERSE,
    _TORCH_DTYPES_MAPPING,
    NoHeaderNumpySerializer,
    NoHeaderTensorSerializer,
    PILSerializer,
    Serializer,
    _get_serializers,
)


def test_dependency_initialization_does_not_import_readers() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import zephon._internal.io.formats.litdata_support

torch_modules = {name for name in sys.modules if name == 'torch' or name.startswith('torch.')}
from zephon._internal.io.formats.litdata_support import dependencies

assert not dependencies._litdata_deps_ready
assert 'litdata' not in sys.modules
assert {name for name in sys.modules if name == 'torch' or name.startswith('torch.')} == torch_modules
dependencies.ensure_litdata_deps()
assert dependencies._litdata_deps_ready
assert dependencies._SERIALIZERS
assert dependencies._NUMPY_DTYPES_MAPPING
assert dependencies._TORCH_DTYPES_MAPPING
assert 'zephon._internal.io.formats.litdata_support.pytree' not in sys.modules
assert 'zephon._internal.io.formats.litdata_support.arrow' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_litdata_dependency_initialization_is_single_flight(monkeypatch) -> None:
    monkeypatch.setattr(dependencies, "_litdata_deps_ready", False)
    calls = 0

    def load() -> None:
        nonlocal calls
        calls += 1
        time.sleep(0.01)
        dependencies._litdata_deps_ready = True

    monkeypatch.setattr(dependencies, "_load_litdata_deps", load)
    barrier = threading.Barrier(8)

    def initialize() -> None:
        barrier.wait()
        dependencies.ensure_litdata_deps()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _index: initialize(), range(8)))

    assert calls == 1


class _EchoSerializer(Serializer):
    def serialize(self, data):
        return str(data).encode("utf-8"), None

    def deserialize(self, data):
        return data.decode("utf-8")

    def can_serialize(self, data):
        return True


def test_litdata_serializers_roundtrip_basic_types():
    serializers = _get_serializers()
    samples = {
        "str": "hello world",
        "bool": True,
        "int": 123,
        "float": 3.14159,
        "bytes": b"\x00\x01payload",
        "numpy": np.arange(5, dtype=np.float32),
        "pickle": {"alpha": 1, "beta": [1, 2, 3]},
    }

    for key, value in samples.items():
        serializer = serializers[key]
        payload, _ = serializer.serialize(value)
        restored = serializer.deserialize(payload)
        if isinstance(value, np.ndarray):
            assert isinstance(restored, np.ndarray)
            assert np.array_equal(restored, value)
        else:
            assert restored == value


def test_litdata_serializers_can_be_overridden():
    override = {"str": _EchoSerializer()}
    serializers = _get_serializers(override)
    assert isinstance(serializers["str"], _EchoSerializer)
    payload, _ = serializers["str"].serialize(42)
    assert serializers["str"].deserialize(payload) == "42"


def test_no_header_numpy_serializer_roundtrip_and_metadata():
    serializer = NoHeaderNumpySerializer()
    sample = np.arange(6, dtype=np.uint16)

    payload, metadata = serializer.serialize(sample)
    expected_index = _NUMPY_DTYPES_REVERSE[np.dtype(np.uint16)]
    assert metadata == f"no_header_numpy:{expected_index}"

    rehydrator = NoHeaderNumpySerializer()
    rehydrator.setup(metadata)
    restored = rehydrator.deserialize(payload)
    assert isinstance(restored, np.ndarray)
    assert restored.dtype == sample.dtype
    assert np.array_equal(restored, sample)
    assert serializer.can_serialize(sample)
    assert not serializer.can_serialize(sample.reshape(2, 3))


def test_no_header_tensor_serializer_roundtrip_and_metadata():
    torch = pytest.importorskip("torch")

    serializer = NoHeaderTensorSerializer()
    sample = torch.arange(5, dtype=torch.int32)

    payload, metadata = serializer.serialize(sample)
    reverse_mapping = {dtype: idx for idx, dtype in _TORCH_DTYPES_MAPPING.items()}
    expected_index = reverse_mapping[sample.dtype]
    assert metadata == f"no_header_tensor:{expected_index}"

    rehydrator = NoHeaderTensorSerializer()
    rehydrator.setup(metadata)
    restored = rehydrator.deserialize(payload)
    assert torch.equal(restored, sample)
    assert serializer.can_serialize(sample)
    assert not serializer.can_serialize(sample.reshape(1, 5))


def test_pil_serializer_roundtrip():
    pytest.importorskip("PIL")
    from PIL import Image

    serializer = PILSerializer()
    image = Image.new("RGB", (4, 3), color=(12, 34, 56))
    payload, metadata = serializer.serialize(image)
    assert metadata is None

    restored = serializer.deserialize(payload)
    assert isinstance(restored, Image.Image)
    assert restored.mode == image.mode
    assert restored.size == image.size
    assert restored.tobytes() == image.tobytes()
    assert serializer.can_serialize(image)
