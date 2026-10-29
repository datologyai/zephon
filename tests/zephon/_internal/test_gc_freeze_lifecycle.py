# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import multiprocessing
from collections.abc import Iterator
from typing import Any
from unittest import mock

import cloudpickle
import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon import Pipeline
from zephon._internal import mtp
from zephon._internal.utils import gc_freeze


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    events: list[str] = []
    monkeypatch.setattr(gc_freeze.gc, "get_freeze_count", lambda: 0)
    for name in ["collect", "freeze", "unfreeze"]:
        monkeypatch.setattr(gc_freeze.gc, name, lambda name=name: events.append(name))
    return events


def _pipe(**options: Any) -> Pipeline:
    ds = make_inmem_dataset("test", [{"text": "test"}])
    return Pipeline(FakeIndexableWorkSource(ds, chunk_size=1)).options(**options)


@pytest.mark.parametrize("option", [None, False, True])
@pytest.mark.parametrize("daemon_fallback", [False, True])
def test_inline_policy_order_and_close(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
    option: bool | None,
    daemon_fallback: bool,
) -> None:
    pipe = _pipe(gc_freeze=option, mtp_mode=daemon_fallback)
    engine = mock.Mock(close=lambda: calls.append("engine.close"))

    def build(**_kwargs: Any) -> tuple[Any, Any, bool]:
        calls.append("build")
        return engine, iter([1, 2]), True

    monkeypatch.setattr(pipe, "_build_raw_iter", build)
    monkeypatch.setattr(pipe, "_yield_while_notifying", lambda iterator: iterator)
    if daemon_fallback:
        monkeypatch.setattr(
            multiprocessing, "current_process", lambda: mock.Mock(daemon=True)
        )
        with pytest.warns(RuntimeWarning, match="Falling back"):
            assert list(pipe._iter_body()) == [1, 2]
    else:
        assert list(pipe._iter_inline()) == [1, 2]
    expected = ["build", "engine.close"]
    if option is True:
        expected = ["collect", "freeze", *expected, "unfreeze"]
    assert calls == expected


def test_inline_unfreezes_when_engine_construction_fails(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
    pipe = _pipe(gc_freeze=True)
    monkeypatch.setattr(
        pipe, "_build_raw_iter", mock.Mock(side_effect=RuntimeError("build"))
    )
    with pytest.raises(RuntimeError, match="build"):
        next(pipe._iter_inline())
    assert calls == ["collect", "freeze", "unfreeze"]


@pytest.mark.parametrize(
    "failure", [None, "deserialize", "build", "iterate", "close", "signal"]
)
def test_mtp_bootstrap_order_and_failure_cleanup(
    monkeypatch: pytest.MonkeyPatch, calls: list[str], failure: str | None
) -> None:
    monkeypatch.setattr(mtp.signal, "signal", mock.Mock())
    monkeypatch.setattr(mtp, "setup_faulthandler", mock.Mock())
    monkeypatch.setattr(mtp, "_wait_for_shutdown", mock.Mock())
    engine = mock.Mock()

    def close() -> None:
        calls.append("engine.close")
        if failure == "close":
            raise RuntimeError("close")

    engine.close = close

    def iterate() -> Iterator[Any]:
        if failure == "iterate":
            raise RuntimeError("iterate")
        if failure == "signal":
            raise SystemExit(0)
        yield from ()

    def build(**_kwargs: Any) -> tuple[Any, Any, bool]:
        calls.append("build")
        if failure == "build":
            raise RuntimeError("build")
        return engine, iterate(), True

    def deserialize(_data: bytes) -> mock.Mock:
        calls.append("deserialize")
        if failure == "deserialize":
            raise RuntimeError("deserialize")
        return mock.Mock(_build_raw_iter=build)

    monkeypatch.setattr(cloudpickle, "loads", deserialize)
    output = mock.Mock()
    if failure == "signal":
        with pytest.raises(SystemExit):
            mtp._mtp_worker(b"pipeline", None, output, mock.Mock(), 2)
    else:
        mtp._mtp_worker(b"pipeline", None, output, mock.Mock(), 2)
    assert calls[:3] == ["collect", "freeze", "deserialize"]
    assert calls[-1] == "unfreeze"
    if failure not in ("deserialize", "build"):
        assert calls[-2] == "engine.close"
    if failure in ("deserialize", "build", "iterate"):
        assert isinstance(output.put.call_args.args[0], mtp._ErrorSentinel)
