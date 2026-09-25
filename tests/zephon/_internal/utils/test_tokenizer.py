# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared tokenizer contract, fallback tokenizer, and HF loader."""

from __future__ import annotations

import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from tenacity.wait import wait_none

from zephon._internal.utils.tokenizer import fallback_tokenizer, load_hf_tokenizer
from zephon._internal.utils.tokenizer_cloud import (
    TransientCloudTokenizerError,
    resolve_tokenizer_id_with_retry,
)


def test_fallback_content_ids_avoid_reserved_range() -> None:
    """``abs(hash(word)) % 10000`` can be 0 or 1 → without an offset, content
    tokens collide with bos=1 / eos=2. The offset puts content at >= 10."""
    tok = fallback_tokenizer()
    # Try a handful of words; the offset must hold for all of them.
    out = tok(["a b c d e f g h hello world"])
    assert all(tid >= 10 for tid in out["input_ids"][0])


def test_fallback_pads_batch_to_longest() -> None:
    tok = fallback_tokenizer()
    out = tok(["one two three", "solo"], padding=True)
    lengths = {len(ids) for ids in out["input_ids"]}
    assert lengths == {3}
    # Shorter row is padded with pad_token=0 and its mask zeroed there.
    assert out["input_ids"][1][1:] == [0, 0]
    assert out["attention_mask"][1] == [1, 0, 0]


def test_fallback_truncates_to_max_length() -> None:
    tok = fallback_tokenizer()
    out = tok(["a b c d e"], truncation=True, max_length=2)
    assert len(out["input_ids"][0]) == 2


def test_fallback_rejects_unknown_return_tensors() -> None:
    tok = fallback_tokenizer()
    with pytest.raises(ValueError, match="return_tensors"):
        tok(["hello"], return_tensors="jax")


# ---------------------------------------------------------------------------
# load_hf_tokenizer
# ---------------------------------------------------------------------------


def _install_fake_transformers(
    monkeypatch: pytest.MonkeyPatch, from_pretrained: Any
) -> None:
    fake = types.ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> Any:
            return from_pretrained(name, **kwargs)

    fake.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake)


def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tenacity.nap.time", types.SimpleNamespace(sleep=lambda s: None)
    )


class _Tok:
    name_or_path = "model"
    pad_token = None
    eos_token = 0


def test_load_fallback_ids_return_stub() -> None:
    for tokenizer_id in (None, "__fallback__"):
        tok = load_hf_tokenizer(tokenizer_id)
        assert tok.name_or_path == "__fallback__"


def test_load_serializes_lazy_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_transformers(monkeypatch, lambda *args, **kwargs: _Tok())
    fake = sys.modules["transformers"]
    auto_tokenizer = fake.AutoTokenizer
    monkeypatch.delattr(fake, "AutoTokenizer")
    importing = threading.Lock()
    start = threading.Barrier(2, timeout=5)

    def lazy_export(name: str) -> Any:
        if name != "AutoTokenizer":
            raise AttributeError(name)
        if not importing.acquire(blocking=False):
            raise ImportError("overlapping lazy imports")
        try:
            time.sleep(0.02)  # Simulate import work that releases the GIL.
            return auto_tokenizer
        finally:
            importing.release()

    monkeypatch.setattr(fake, "__getattr__", lazy_export, raising=False)
    monkeypatch.setattr(
        "zephon._internal.utils.tokenizer.suppress_library_threads", lambda: None
    )

    def load() -> Any:
        start.wait()
        return load_hf_tokenizer("model")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(load) for _ in range(2)]
        assert all(isinstance(result.result(timeout=5), _Tok) for result in results)


def test_load_keeps_construction_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    constructing = threading.Barrier(2, timeout=2)

    def from_pretrained(name: str, **kwargs: Any) -> Any:
        try:
            constructing.wait()
        except threading.BrokenBarrierError as exc:
            raise ValueError("tokenizer construction was serialized") from exc
        return _Tok()

    _install_fake_transformers(monkeypatch, from_pretrained)
    monkeypatch.setattr(
        "zephon._internal.utils.tokenizer.suppress_library_threads", lambda: None
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(load_hf_tokenizer, "model") for _ in range(2)]
        assert all(isinstance(result.result(timeout=5), _Tok) for result in results)


def test_load_forwards_use_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def from_pretrained(name: str, **kwargs: Any) -> Any:
        calls["name"] = name
        calls["kwargs"] = kwargs
        return _Tok()

    _install_fake_transformers(monkeypatch, from_pretrained)
    load_hf_tokenizer("hf-model", use_fast=False)
    assert calls["name"] == "hf-model"
    assert calls["kwargs"]["use_fast"] is False


def test_load_fallback_rejects_eos_token() -> None:
    # Silently returning the stub would drop an explicit EOS override.
    with pytest.raises(ValueError, match="eos_token"):
        load_hf_tokenizer(None, eos_token="<|stop|>")


def test_load_omits_use_fast_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def from_pretrained(name: str, **kwargs: Any) -> Any:
        calls["kwargs"] = kwargs
        return _Tok()

    _install_fake_transformers(monkeypatch, from_pretrained)
    load_hf_tokenizer("hf-model", use_fast=None)
    assert "use_fast" not in calls["kwargs"]


def test_load_retries_without_use_fast_on_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_history: list[dict[str, Any]] = []

    def from_pretrained(name: str, **kwargs: Any) -> Any:
        call_history.append(kwargs)
        if "use_fast" in kwargs:
            raise TypeError("got an unexpected keyword argument 'use_fast'")
        return _Tok()

    _install_fake_transformers(monkeypatch, from_pretrained)
    tok = load_hf_tokenizer("flaky-model", use_fast=True)
    assert isinstance(tok, _Tok)
    assert call_history[0].get("use_fast") is True
    assert "use_fast" not in call_history[-1]


def test_load_raises_other_type_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def from_pretrained(name: str, **kwargs: Any) -> Any:
        raise TypeError("Something else completely broken")

    _install_fake_transformers(monkeypatch, from_pretrained)
    with pytest.raises(TypeError, match="Something else completely broken"):
        load_hf_tokenizer("broken-model")


def test_load_retries_transient_failure_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_retry_sleep(monkeypatch)
    attempts = 0

    def from_pretrained(name: str, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("simulated transient hub failure")
        return _Tok()

    _install_fake_transformers(monkeypatch, from_pretrained)
    tok = load_hf_tokenizer("hf-model")
    assert isinstance(tok, _Tok)
    assert attempts == 2


def test_load_reraises_after_exhausted_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_retry_sleep(monkeypatch)
    attempts = 0

    def from_pretrained(name: str, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("permanently down")

    _install_fake_transformers(monkeypatch, from_pretrained)
    with pytest.raises(RuntimeError, match="permanently down"):
        load_hf_tokenizer("hf-model")
    assert attempts == 5


def test_load_resolves_cloud_uri_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    received_ids: list[str] = []

    def from_pretrained(name: str, **kwargs: Any) -> Any:
        received_ids.append(name)
        return _Tok()

    _install_fake_transformers(monkeypatch, from_pretrained)
    resolved_path = str(tmp_path / "fake-resolved-tokenizer")

    def fake_resolve(tokenizer_id: str) -> str:
        assert tokenizer_id == "s3://fake-bucket/fake/prefix/"
        return resolved_path

    monkeypatch.setattr(
        "zephon._internal.utils.tokenizer_cloud.resolve_tokenizer_id", fake_resolve
    )
    load_hf_tokenizer("s3://fake-bucket/fake/prefix/")
    assert received_ids == [resolved_path]


def test_load_passes_hub_id_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_ids: list[str] = []

    def from_pretrained(name: str, **kwargs: Any) -> Any:
        received_ids.append(name)
        return _Tok()

    _install_fake_transformers(monkeypatch, from_pretrained)
    load_hf_tokenizer("meta-llama/Llama-3.2-1B")
    assert received_ids == ["meta-llama/Llama-3.2-1B"]


def test_load_retries_transient_cloud_sync_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    resolved_path = str(tmp_path / "fake-resolved-tokenizer")
    resolve_attempts = 0

    def flaky_resolve(tokenizer_id: str) -> str:
        nonlocal resolve_attempts
        resolve_attempts += 1
        if resolve_attempts == 1:
            raise TransientCloudTokenizerError("simulated transient S3 5xx")
        return resolved_path

    monkeypatch.setattr(
        "zephon._internal.utils.tokenizer_cloud.resolve_tokenizer_id", flaky_resolve
    )
    # Drop the resolve backoff so the test runs fast. The retry envelope is
    # built at import, so mutate the controller rather than patch the wait.
    monkeypatch.setattr(resolve_tokenizer_id_with_retry.retry, "wait", wait_none())

    received_ids: list[str] = []

    def from_pretrained(name: str, **kwargs: Any) -> Any:
        received_ids.append(name)
        return _Tok()

    _install_fake_transformers(monkeypatch, from_pretrained)
    load_hf_tokenizer("s3://fake-bucket/fake/prefix/")
    assert resolve_attempts == 2, "Transient sync error must be retried once"
    assert received_ids == [resolved_path]
