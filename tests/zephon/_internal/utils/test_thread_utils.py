# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Process-wide reader pool configuration and legacy environment overrides."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from zephon._internal.utils import thread_utils
from zephon.io import ArrowOptions, VortexOptions


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    for name in ("ZEPHON_PARQUET_ARROW_THREADS", "VORTEX_MAX_THREADS"):
        monkeypatch.delenv(name, raising=False)
    module = ModuleType("native")
    module.set_worker_threads = Mock()
    module.set_cpu_count = Mock()
    module.set_io_thread_count = Mock()
    return module


@pytest.mark.parametrize("count", [None, 0, 1, 3])
def test_vortex_explicit_worker_count(native: ModuleType, count: int | None) -> None:
    thread_utils.configure_vortex_threads(native, VortexOptions(worker_threads=count))
    if count is None:
        native.set_worker_threads.assert_not_called()
    else:
        native.set_worker_threads.assert_called_once_with(count)


def test_vortex_default_and_native_override(native: ModuleType, monkeypatch) -> None:
    thread_utils.configure_vortex_threads(native, VortexOptions())
    native.set_worker_threads.assert_called_once_with(1)
    native.set_worker_threads.reset_mock()
    monkeypatch.setenv("VORTEX_MAX_THREADS", "3")
    thread_utils.configure_vortex_threads(native, VortexOptions())
    native.set_worker_threads.assert_not_called()
    thread_utils.configure_vortex_threads(native, VortexOptions(worker_threads=1))
    native.set_worker_threads.assert_called_once_with(1)


def test_arrow_counts_are_independent(native: ModuleType) -> None:
    thread_utils.configure_arrow_threads(
        native, ArrowOptions(cpu_threads=3, io_threads=None)
    )
    native.set_cpu_count.assert_called_once_with(3)
    native.set_io_thread_count.assert_not_called()
    native.set_cpu_count.reset_mock()
    thread_utils.configure_arrow_threads(
        native, ArrowOptions(cpu_threads=None, io_threads=2)
    )
    native.set_cpu_count.assert_not_called()
    native.set_io_thread_count.assert_called_once_with(2)


def test_arrow_defaults(native: ModuleType) -> None:
    thread_utils.configure_arrow_threads(native, ArrowOptions())
    native.set_cpu_count.assert_called_once_with(1)
    native.set_io_thread_count.assert_called_once_with(1)


@pytest.mark.parametrize("count", ["0", "-1", "4"])
def test_arrow_legacy_fallback_only_for_omitted_fields(
    native, monkeypatch, count
) -> None:
    monkeypatch.setenv("ZEPHON_PARQUET_ARROW_THREADS", count)
    with pytest.warns(DeprecationWarning, match="io_options.arrow"):
        thread_utils.configure_arrow_threads(native, ArrowOptions(cpu_threads=2))
    native.set_cpu_count.assert_called_once_with(2)
    if int(count) > 0:
        native.set_io_thread_count.assert_called_once_with(int(count))
    else:
        native.set_io_thread_count.assert_not_called()


def test_arrow_explicit_options_ignore_legacy_environment(native, monkeypatch) -> None:
    monkeypatch.setenv("ZEPHON_PARQUET_ARROW_THREADS", "invalid")
    thread_utils.configure_arrow_threads(
        native, ArrowOptions(cpu_threads=None, io_threads=2)
    )
    native.set_cpu_count.assert_not_called()
    native.set_io_thread_count.assert_called_once_with(2)


@pytest.mark.parametrize("library", ["vortex", "arrow"])
def test_native_configuration_errors_propagate(native, library) -> None:
    if library == "vortex":
        native.set_worker_threads.side_effect = RuntimeError("native failure")
        configure = lambda: thread_utils.configure_vortex_threads(
            native, VortexOptions()
        )
    else:
        native.set_cpu_count.side_effect = RuntimeError("native failure")
        configure = lambda: thread_utils.configure_arrow_threads(native, ArrowOptions())
    with pytest.raises(RuntimeError, match="native failure"):
        configure()


def test_general_suppression_does_not_reset_reader_pools(native, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "vortex", native)
    monkeypatch.setitem(sys.modules, "pyarrow", native)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=Mock()))
    tf_threads = SimpleNamespace(
        set_intra_op_parallelism_threads=Mock(), set_inter_op_parallelism_threads=Mock()
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorflow",
        SimpleNamespace(config=SimpleNamespace(threading=tf_threads)),
    )
    thread_utils.suppress_library_threads()
    native.set_worker_threads.assert_not_called()
    native.set_cpu_count.assert_not_called()
    native.set_io_thread_count.assert_not_called()
