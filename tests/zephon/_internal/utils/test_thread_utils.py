# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Process-wide reader pool configuration and legacy environment overrides."""

import os
import sys
import warnings
from types import ModuleType, SimpleNamespace
from typing import Iterator
from unittest.mock import Mock

import pytest

from zephon._internal.utils import thread_utils
from zephon.io import ArrowOptions, LanceOptions, VortexOptions


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


_LANCE_VARS = ("TOKIO_WORKER_THREADS", "LANCE_CPU_THREADS")


@pytest.fixture
def lance_file(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """A pylance stand-in recording the environment its runtime starts with."""
    # The CPU variable is left set by design, so restore both by hand.
    saved = {name: os.environ.pop(name, None) for name in _LANCE_VARS}
    monkeypatch.setattr(thread_utils, "_lance_pools_started", False)
    monkeypatch.setattr(thread_utils, "_lance_pools_size", None)
    seen: list[dict[str, str | None]] = []

    def start_session(uri: str) -> None:
        seen.append({name: os.environ.get(name) for name in _LANCE_VARS})

    yield SimpleNamespace(LanceFileSession=start_session, seen=seen)
    for name, value in saved.items():
        os.environ.pop(name, None)
        if value is not None:
            os.environ[name] = value


def test_lance_tokio_size_is_set_only_while_pylance_starts(lance_file) -> None:
    thread_utils.configure_lance_threads(lance_file, LanceOptions())

    assert lance_file.seen == [dict.fromkeys(_LANCE_VARS, "2")]
    assert "TOKIO_WORKER_THREADS" not in os.environ
    # Read only when the CPU pool first runs, which can be long after setup.
    assert os.environ["LANCE_CPU_THREADS"] == "2"


def test_lance_pools_are_sized_once_per_process(lance_file) -> None:
    thread_utils.configure_lance_threads(lance_file, LanceOptions(worker_threads=2))
    with pytest.warns(RuntimeWarning, match="worker_threads=3 has no effect"):
        thread_utils.configure_lance_threads(lance_file, LanceOptions(worker_threads=3))

    assert lance_file.seen == [dict.fromkeys(_LANCE_VARS, "2")]


def test_lance_failed_start_is_not_recorded(lance_file) -> None:
    lance_file.LanceFileSession = Mock(side_effect=[RuntimeError("no runtime"), None])

    with pytest.raises(RuntimeError, match="no runtime"):
        thread_utils.configure_lance_threads(lance_file, LanceOptions())
    thread_utils.configure_lance_threads(lance_file, LanceOptions())

    assert lance_file.LanceFileSession.call_count == 2
    assert "TOKIO_WORKER_THREADS" not in os.environ


def test_lance_resizing_to_the_applied_count_is_silent(lance_file) -> None:
    thread_utils.configure_lance_threads(lance_file, LanceOptions())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        thread_utils.configure_lance_threads(lance_file, LanceOptions(worker_threads=2))
        thread_utils.configure_lance_threads(lance_file, LanceOptions())


def test_lance_explicit_count_is_checked_against_environment_sized_pools(
    lance_file,
) -> None:
    for name in _LANCE_VARS:
        os.environ[name] = "4"
    thread_utils.configure_lance_threads(lance_file, LanceOptions())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        thread_utils.configure_lance_threads(lance_file, LanceOptions(worker_threads=4))
    with pytest.warns(RuntimeWarning, match="worker_threads=3 has no effect"):
        thread_utils.configure_lance_threads(lance_file, LanceOptions(worker_threads=3))


def test_lance_omitted_count_respects_the_environment(lance_file) -> None:
    os.environ["TOKIO_WORKER_THREADS"] = "4"

    thread_utils.configure_lance_threads(lance_file, LanceOptions())

    assert lance_file.seen == [{"TOKIO_WORKER_THREADS": "4", "LANCE_CPU_THREADS": "2"}]
    assert os.environ["TOKIO_WORKER_THREADS"] == "4"


def test_lance_explicit_count_overrides_the_environment(lance_file) -> None:
    os.environ["TOKIO_WORKER_THREADS"] = "4"

    thread_utils.configure_lance_threads(lance_file, LanceOptions(worker_threads=2))

    assert lance_file.seen == [dict.fromkeys(_LANCE_VARS, "2")]
    assert os.environ["TOKIO_WORKER_THREADS"] == "4"
    assert os.environ["LANCE_CPU_THREADS"] == "2"


def test_lance_none_leaves_the_pools_alone(lance_file) -> None:
    thread_utils.configure_lance_threads(lance_file, LanceOptions(worker_threads=None))

    assert lance_file.seen == []


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_lance_rejects_counts_pylance_cannot_start_with(count) -> None:
    with pytest.raises(ValueError, match="lance.worker_threads must be a positive"):
        LanceOptions(worker_threads=count)


def test_store_options_carry_lance_options() -> None:
    from zephon.io import StoreOptions

    options = StoreOptions.from_any({"lance": {"worker_threads": 2}})
    merged = StoreOptions().merge(options)

    assert options.lance.worker_threads == 2
    assert merged.lance.worker_threads == 2
    assert StoreOptions().lance.worker_threads == 2
