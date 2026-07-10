# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared tokenize-op scaffolding."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from zephon.ops.tokenize_base import _MISSING, TokenizeBase


class _CountingBase(TokenizeBase):
    """Records finalize calls; optionally fails them."""

    def __init__(self, *args: Any, fail_finalize: bool = False, **kwargs: Any) -> None:
        TokenizeBase.__init__(self, *args, **kwargs)
        self.finalize_calls = 0
        self.fail_finalize = fail_finalize

    def _finalize_setup(self) -> None:
        self.finalize_calls += 1
        if self.fail_finalize:
            raise ValueError("finalize boom")


def _base(**kwargs: Any) -> _CountingBase:
    defaults: dict[str, Any] = {
        "use_fast": True,
        "max_batch": 64,
        "max_latency_ms": 3,
    }
    defaults.update(kwargs)
    return _CountingBase(
        defaults.pop("tokenizer", None), defaults.pop("tokenizer_id", None), **defaults
    )


def test_lazy_setup_loads_and_finalizes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[tuple[str | None, bool | None]] = []

    def fake_load(tokenizer_id: str | None, *, use_fast: bool | None) -> Any:
        loads.append((tokenizer_id, use_fast))
        return SimpleNamespace(name_or_path="loaded")

    monkeypatch.setattr("zephon.ops.tokenize_base.load_hf_tokenizer", fake_load)
    op = _base(tokenizer_id="some-model", use_fast=False)
    op._setup_tokenizer()
    op._setup_tokenizer()
    assert loads == [("some-model", False)]
    assert op.finalize_calls == 1
    assert op._tokenizer_instantiated is True


def test_provided_tokenizer_skips_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_load(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("loader must not run when a tokenizer is provided")

    monkeypatch.setattr("zephon.ops.tokenize_base.load_hf_tokenizer", fail_load)
    provided = SimpleNamespace(name_or_path="user-supplied")
    op = _base(tokenizer=provided)
    op._setup_tokenizer()
    assert op.tok is provided


def test_finalize_failure_is_cached_and_reraised() -> None:
    op = _base(tokenizer=SimpleNamespace(), fail_finalize=True)
    with pytest.raises(ValueError, match="finalize boom") as first:
        op._setup_tokenizer()
    with pytest.raises(ValueError, match="finalize boom") as second:
        op._setup_tokenizer()
    # Same cached error, no half-initialised retry.
    assert second.value is first.value
    assert op.finalize_calls == 1
    assert op._tokenizer_instantiated is False


def test_lookup_path_semantics() -> None:
    op = _base()
    payload = {"a": {"b": 1}, "flat": "x"}
    assert op._lookup_path(payload, ("a", "b")) == 1
    assert op._lookup_path(payload, ("flat",)) == "x"
    assert op._lookup_path(payload, ()) is payload
    assert op._lookup_path(payload, ("missing",)) is _MISSING
    assert op._lookup_path(payload, ("a", "missing")) is _MISSING
    assert op._lookup_path(payload, ("flat", "deeper")) is _MISSING
    assert op._lookup_path("not-a-mapping", ("a",)) is _MISSING


def test_configured_tokenizer_id_precedence() -> None:
    assert _base(tokenizer_id="hub/id").configured_tokenizer_id() == "hub/id"
    tok = SimpleNamespace(name_or_path="local-tok")
    assert _base(tokenizer=tok).configured_tokenizer_id() == "local-tok"
    assert _base().configured_tokenizer_id() == "__fallback__"


def test_accumulator_settings() -> None:
    op = _base(max_batch=48, max_latency_ms=15)
    acc_det = op.accumulator(deterministic=True, ctx={})
    assert acc_det._max_batch == 48
    assert acc_det._max_latency_ms is None
    acc_nondet = op.accumulator(deterministic=False, ctx={})
    assert acc_nondet._max_batch == 48
    assert acc_nondet._max_latency_ms == 15
