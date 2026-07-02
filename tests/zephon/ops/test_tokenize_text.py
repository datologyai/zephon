# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
import types
from typing import Any

import pytest
from tenacity.wait import wait_none

from zephon.core.constants import SampleMeta, SampleRecord
from zephon.core.op_base import OpContext
from zephon.ops.tokenize_text import (
    SpecialTokensMode,
    TokenizeText,
)
from zephon.utils.tokenizer_cloud import (
    TransientCloudTokenizerError,
    resolve_tokenizer_id_with_retry,
)
from zephon.utils.torch_compat import _should_use_tensor_lock


def _noop(*args, **kwargs) -> None:  # pragma: no cover - trivial helper
    return None


def _setup(
    op: TokenizeText,
    ctx_data: dict[str, object] | None = None,
    *,
    collect_stats: bool = False,
) -> TokenizeText:
    ctx = {"record_node_metrics": _noop}
    if ctx_data:
        ctx.update(ctx_data)
    op.setup(
        OpContext(ctx),
        stage_index=0,
        stage_name="stage0",
        op_index=0,
        collect_stats=collect_stats,
    )
    return op


def _rec(text: Any, *, field: str = "text") -> SampleRecord:
    meta = SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0)
    return SampleRecord(meta=meta, payload={field: text})


def _payload_dict(record: SampleRecord) -> dict[str, Any]:
    payload = record.payload
    assert isinstance(payload, dict)
    return payload


def test_tokenize_fallback_process_one_and_many() -> None:
    op = _setup(TokenizeText(tokenizer=None, tokenizer_id="__fallback__", field="text"))
    r1 = _rec("hello world")
    out1 = op.process_one(r1)[0]
    payload = _payload_dict(out1)
    assert "input_ids" in payload and "attention_mask" in payload
    # many
    r2 = _rec("more words")
    bulk = op.process_many([r1, r2])
    assert len(bulk) == 2
    for rec in bulk:
        payload = _payload_dict(rec)
        assert "input_ids" in payload
        assert "attention_mask" in payload


def test_tokenize_custom_field_and_missing_field() -> None:
    op = _setup(
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            field="title",
            special_tokens="tokenizer_default",
        )
    )
    r = _rec("ignored", field="text")  # text present, but tokenizer uses "title"
    out = op.process_one(r)[0]
    # When field is missing, fallback tokenizer sees empty string
    payload = _payload_dict(out)
    assert payload.get("input_ids", []) == []
    assert payload.get("attention_mask", []) == []


def test_tokenize_nested_field_dot_path() -> None:
    """Dot-notation field reaches into nested mappings (e.g. parquet structs)."""
    op = _setup(
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            field="text.content",
            special_tokens="tokenizer_default",
        )
    )
    meta = SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0)
    rec = SampleRecord(
        meta=meta, payload={"text": {"content": "hello world", "lang": "en"}}
    )
    out = op.process_one(rec)[0]
    payload = _payload_dict(out)
    # Fallback tokenizer maps each whitespace-split token to a positive id, so
    # two-token text yields a length-2 input_ids list.
    assert len(payload.get("input_ids", [])) == 2


def test_tokenize_nested_field_missing_intermediate() -> None:
    """Missing intermediate key resolves to empty string, not an error."""
    op = _setup(
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            field="text.content",
            special_tokens="tokenizer_default",
        )
    )
    # ``text`` is absent entirely
    out = op.process_one(_rec("ignored", field="other"))[0]
    payload = _payload_dict(out)
    assert payload.get("input_ids", []) == []


def test_tokenize_nested_field_intermediate_not_mapping() -> None:
    """Intermediate non-mapping value resolves to empty string."""
    op = _setup(
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            field="text.content",
            special_tokens="tokenizer_default",
        )
    )
    # ``text`` exists but is a scalar, not a mapping
    out = op.process_one(_rec("scalar text"))[0]
    payload = _payload_dict(out)
    assert payload.get("input_ids", []) == []


def test_tokenize_disable_attention_mask() -> None:
    op = TokenizeText(
        tokenizer=None,
        tokenizer_id="__fallback__",
        field="text",
        add_attention_mask=False,
    )
    op = _setup(op)
    out = op.process_one(_rec("hi there"))[0]
    payload = _payload_dict(out)
    assert "input_ids" in payload
    assert "attention_mask" not in payload


def test_tokenize_replaces_payload_by_default() -> None:
    op = _setup(TokenizeText(tokenizer=None, tokenizer_id="__fallback__", field="text"))
    rec = _rec("keep me")
    payload = _payload_dict(rec)
    payload["extra"] = 99
    out = op.process_one(rec)[0]
    assert out is rec  # fast path reuses record
    payload_out = _payload_dict(out)
    assert "extra" not in payload_out
    assert "text" not in payload_out


def test_tokenize_preserves_payload_when_requested() -> None:
    op = _setup(
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            field="text",
            preserve_upstream_payload=True,
        )
    )
    rec = _rec("keep me")
    payload = _payload_dict(rec)
    payload["extra"] = 99
    out = op.process_one(rec)[0]
    payload_out = _payload_dict(out)
    assert payload_out["extra"] == 99
    assert payload_out["text"] == "keep me"


def test_tokenize_custom_tokenizer_and_resolved_id() -> None:
    class ToyTok:
        name_or_path = "toy-tokenizer"

        def __call__(
            self,
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        ):
            if isinstance(texts, str):
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
            return {
                "input_ids": [[1, 2, 3] for _ in texts],
                "attention_mask": [[1, 1, 1] for _ in texts],
            }

    tok = ToyTok()
    op = _setup(
        TokenizeText(tokenizer=tok, field="text", special_tokens="tokenizer_default")
    )
    out = op.process_many([_rec("x"), _rec("y")])
    assert _payload_dict(out[0])["input_ids"] == [1, 2, 3]
    assert op.configured_tokenizer_id() == "toy-tokenizer"


def test_tokenize_passes_extra_hf_options() -> None:
    class CaptureTok:
        name_or_path = "cap-tokenizer"

        def __init__(self) -> None:
            self.pad_token = None
            self.eos_token = 99
            self.calls: list[dict[str, Any]] = []

        def __call__(self, texts, **kwargs):
            self.calls.append(kwargs)
            return {
                "input_ids": [[1, 2] for _ in texts],
                "attention_mask": [[1, 1] for _ in texts],
            }

    tok = CaptureTok()
    op = _setup(
        TokenizeText(
            tokenizer=tok,
            field="text",
            padding=True,
            truncation=True,
            max_length=4,
            return_tensors="pt",
            special_tokens="tokenizer_default",
        )
    )
    out = op.process_many([_rec("x"), _rec("y")])
    assert len(out) == 2
    assert tok.pad_token == tok.eos_token
    kwargs = tok.calls[-1]
    assert kwargs["padding"] is True
    assert kwargs["truncation"] is True
    assert kwargs["max_length"] == 4
    # On free-threaded Python with PyTorch < 2.10, we request numpy from the HF
    # tokenizer and convert to pytorch ourselves under a lock to avoid an allocator
    # race condition. The user still gets pytorch tensors, but the kwargs passed
    # to the tokenizer have return_tensors="np" internally.
    # See: https://github.com/pytorch/pytorch/issues/171992
    expected_return_tensors = "np" if _should_use_tensor_lock() else "pt"
    assert kwargs["return_tensors"] == expected_return_tensors


def test_fallback_respects_padding_and_truncation_options() -> None:
    op = _setup(
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            field="text",
            padding=True,
            truncation=True,
            max_length=2,
            special_tokens="tokenizer_default",
        )
    )
    out = op.process_many([_rec("a b c"), _rec("d")])
    assert len(out) == 2
    for rec in out:
        payload = _payload_dict(rec)
        assert len(payload["input_ids"]) == 2
        assert len(payload["attention_mask"]) == 2


def test_split_long_samples_requires_max_length_and_no_truncation() -> None:
    with pytest.raises(ValueError):
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            field="text",
            split_long_samples=True,
            truncation=True,
            max_length=8,
        )
    with pytest.raises(ValueError):
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            field="text",
            split_long_samples=True,
            max_length=None,
        )


def test_split_long_samples_fanout_lineage_and_contributors() -> None:
    op = _setup(
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            field="text",
            split_long_samples=True,
            max_length=2,
            padding=False,
            special_tokens="tokenizer_default",
        )
    )
    out = op.process_one(_rec("a b c d e"))  # 5 tokens => 3 children
    assert len(out) == 3
    for idx, rec in enumerate(out):
        payload = _payload_dict(rec)
        assert len(payload["input_ids"]) <= 2
        assert rec.meta.lineage == (idx,)
        contributors = rec.meta.contributors
        assert len(contributors) == 1
        assert contributors[0].is_last_child is (idx == 2)


def test_split_long_samples_padding_applied() -> None:
    op = _setup(
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            field="text",
            split_long_samples=True,
            max_length=3,
            padding=True,
            special_tokens="tokenizer_default",
        )
    )
    out = op.process_one(_rec("a b c d"))  # 4 tokens -> 2 chunks
    assert len(out) == 2
    lengths = [len(_payload_dict(rec)["input_ids"]) for rec in out]
    assert lengths == [3, 3]


def test_split_long_samples_respects_return_tensors() -> None:
    torch = pytest.importorskip("torch")
    op = _setup(
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            field="text",
            split_long_samples=True,
            max_length=2,
            padding=False,
            return_tensors="pt",
            special_tokens="tokenizer_default",
        )
    )
    out = op.process_one(_rec("a b c"))
    assert len(out) == 2
    for rec in out:
        payload = _payload_dict(rec)
        assert isinstance(payload["input_ids"], torch.Tensor)
        assert payload["input_ids"].ndim == 1


def test_split_long_samples_numpy_tensors() -> None:
    np = pytest.importorskip("numpy")
    op = _setup(
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            field="text",
            split_long_samples=True,
            max_length=2,
            padding=False,
            return_tensors="np",
            special_tokens="tokenizer_default",
        )
    )
    out = op.process_one(_rec("a b c"))
    assert len(out) == 2
    for rec in out:
        payload = _payload_dict(rec)
        assert isinstance(payload["input_ids"], np.ndarray)


def test_convert_tensor_does_not_copy_torch_segments() -> None:
    torch = pytest.importorskip("torch")

    class TorchTok:
        name_or_path = "torch"
        pad_token = None
        eos_token = 0
        last_ids = None

        def __call__(self, texts, **kwargs):
            self.last_ids = torch.arange(6, dtype=torch.long)
            mask = torch.ones(6, dtype=torch.long)
            return {"input_ids": self.last_ids, "attention_mask": mask}

    tok = TorchTok()
    op = _setup(
        TokenizeText(
            tokenizer=tok,
            field="text",
            split_long_samples=True,
            max_length=3,
            padding=False,
            return_tensors="pt",
            special_tokens="tokenizer_default",
        )
    )
    out = op.process_one(_rec("x"))
    assert len(out) == 2
    # segments are slices; convert_tensor should not copy
    first = out[0].payload["input_ids"]
    second = out[1].payload["input_ids"]
    ids = tok.last_ids
    assert ids is not None
    assert first.storage().data_ptr() == ids.storage().data_ptr()
    assert second.storage().data_ptr() == ids.storage().data_ptr()


def test_split_segments_torch_backend() -> None:
    torch = pytest.importorskip("torch")
    op = _setup(
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            field="text",
            split_long_samples=True,
            max_length=3,
            padding=True,
            return_tensors="pt",
        )
    )
    ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    mask = torch.ones_like(ids)
    segments = op._split_segments_torch(ids, mask, 3, pad_id=0)
    assert len(segments) == 2
    assert segments[0][0].tolist() == [1, 2, 3]
    assert segments[1][0].tolist() == [4, 5, 0]


def test_split_segments_numpy_backend() -> None:
    np = pytest.importorskip("numpy")
    op = _setup(
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            field="text",
            split_long_samples=True,
            max_length=2,
            padding=True,
            return_tensors="np",
        )
    )
    ids = np.array([1, 2, 3], dtype=np.int64)
    mask = np.ones_like(ids)
    segments = op._split_segments_numpy(ids, mask, 2, pad_id=0)
    assert len(segments) == 2
    assert segments[0][0].tolist() == [1, 2]
    assert segments[1][0].tolist() == [3, 0]


def test_split_segments_tf_backend() -> None:
    tf = pytest.importorskip("tensorflow")
    op = _setup(
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            field="text",
            split_long_samples=True,
            max_length=2,
            padding=True,
            return_tensors="tf",
        )
    )
    ids = tf.constant([1, 2, 3], dtype=tf.int32)
    mask = tf.ones_like(ids)
    segments = op._split_segments_tf(ids, mask, 2, pad_id=0)
    assert len(segments) == 2
    assert list(segments[0][0].numpy()) == [1, 2]
    assert list(segments[1][0].numpy()) == [3, 0]


def test_tokenizes_non_mapping_payload_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="zephon.ops.tokenize_text")
    op = _setup(TokenizeText(tokenizer=None, tokenizer_id="__fallback__", field="text"))
    rec = _rec("hello world")
    rec = SampleRecord(meta=rec.meta, payload="plain string")
    out = op.process_one(rec)[0]
    payload = _payload_dict(out)
    assert payload["input_ids"]
    assert any("expected mapping payloads" in r.message for r in caplog.records)
    caplog.clear()
    _ = op.process_one(rec)
    assert not caplog.records  # warning only once


def test_preserve_payload_warns_when_non_mapping(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="zephon.ops.tokenize_text")
    op = _setup(
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            field="text",
            preserve_upstream_payload=True,
        )
    )
    rec = SampleRecord(meta=_rec("hi").meta, payload="raw string")
    out = op.process_one(rec)[0]
    payload = _payload_dict(out)
    assert "text" not in payload
    assert "input_ids" in payload
    assert any("preserve_upstream_payload" in r.message for r in caplog.records)
    caplog.clear()
    _ = op.process_one(rec)
    assert not caplog.records  # warning only once


def test_tokenizer_id_load_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="zephon.ops.tokenize_text")
    fake = types.ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name: str):  # pragma: no cover - simple stub
            raise RuntimeError("fail")

    fake.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake)

    op = TokenizeText(tokenizer=None, tokenizer_id="some-model", field="text")
    _setup(op)
    with pytest.raises(RuntimeError):
        _ = op.process_one(_rec("hi"))


def test_tokenizer_use_fast_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    fake = types.ModuleType("transformers")

    class _Tok:
        name_or_path = "model"
        pad_token = None
        eos_token = 0

        def __call__(self, texts, **kwargs):
            return {
                "input_ids": [[1, 2] for _ in texts],
                "attention_mask": [[1, 1] for _ in texts],
            }

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **kwargs):
            calls["name"] = name
            calls["kwargs"] = kwargs
            return _Tok()

    fake.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake)

    op = _setup(
        TokenizeText(
            tokenizer=None,
            tokenizer_id="hf-model",
            field="text",
            use_fast=False,
            special_tokens="tokenizer_default",
        )
    )
    _ = op.process_one(_rec("hi"))
    assert calls["name"] == "hf-model"
    assert calls["kwargs"]["use_fast"] is False


def test_tokenizer_initializes_on_first_process() -> None:
    op = TokenizeText(tokenizer=None, tokenizer_id=None, field="text")
    out = op.process_one(_rec("text"))
    payload = _payload_dict(out[0])
    assert "input_ids" in payload


def test_tokenizer_traits_and_default_accumulator() -> None:
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            max_batch=48,
            max_latency_ms=15,
        )
    )
    t = op.traits()

    assert t.indexable is True and t.parallelism == 4

    # Test deterministic mode disables time-based flushing
    acc_det = op.accumulator(deterministic=True, ctx={})
    assert acc_det._max_batch == 48
    assert acc_det._max_latency_ms is None

    # Test non-deterministic mode preserves latency config
    acc_nondet = op.accumulator(deterministic=False, ctx={})
    assert acc_nondet._max_batch == 48
    assert acc_nondet._max_latency_ms == 15


# --- Performance & Optimization Tests ---


def test_fast_path_modifies_in_place() -> None:
    """
    Verifies that when split_long_samples=False, the operator modifies
    the existing SampleRecord objects rather than creating new ones.
    This confirms the zero-allocation optimization.
    """
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            split_long_samples=False,
        )
    )

    r1 = _rec("test one")
    r2 = _rec("test two")
    input_list = [r1, r2]

    # Store original IDs to compare later
    id1, id2 = id(r1), id(r2)

    out = op.process_many(input_list)

    # 1. Check that the returned list is the EXACT same list object
    assert out is input_list

    # 2. Check that the items inside are the EXACT same record objects
    assert id(out[0]) == id1
    assert id(out[1]) == id2

    # 3. Check that payloads were actually updated
    assert "input_ids" in out[0].payload
    assert "input_ids" in out[1].payload


def test_slow_path_creates_new_objects() -> None:
    """
    Verifies that split_long_samples=True correctly creates NEW records,
    preserving the intended lineage logic.
    """
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            split_long_samples=True,
            max_length=10,  # Large enough to not actually split, but trigger path
        )
    )

    r1 = _rec("test")
    input_list = [r1]

    out = op.process_many(input_list)

    # Should be a new list with new records
    assert out is not input_list
    assert out[0] is not r1
    # But metadata should point to same sample_id
    assert out[0].meta.sample_id == r1.meta.sample_id


def test_kwargs_caching_behavior() -> None:
    """
    Verifies that tokenizer kwargs are computed once during setup.
    Modifying properties after setup should NOT change behavior.
    """
    op = TokenizeText(tokenizer_id="__fallback__", field="text", padding=False)
    op = _setup(op)

    # Hack: Inspect cached kwargs directly
    assert op._cached_kwargs["padding"] is False

    # Maliciously change property after setup
    op.padding = True

    # Process a sample
    # If caching works, this should still run with padding=False
    # (Fallback tokenizer produces ragged arrays if padding=False, which is fine here)
    out = op.process_one(_rec("short"))

    # Check internal cache remained strict
    assert op._cached_kwargs["padding"] is False


# --- Initialization & Retry Logic Tests ---


def test_setup_retries_without_use_fast_on_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Tests the try/except block in setup() that handles tokenizers
    rejecting the 'use_fast' argument.
    """
    fake_mod = types.ModuleType("transformers")

    class FlakyTokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1]], "attention_mask": [[1]]}

    call_history = []

    class MockAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            call_history.append(kwargs)
            if "use_fast" in kwargs:
                # Simulate an older Transformer version or specific model failure
                raise TypeError("got an unexpected keyword argument 'use_fast'")
            return FlakyTokenizer()

    fake_mod.AutoTokenizer = MockAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_mod)

    # Init with use_fast=True (default)
    op = TokenizeText(
        tokenizer_id="flaky-model",
        field="text",
        use_fast=True,
        special_tokens="tokenizer_default",
    )
    _setup(op)
    _ = op.process_one(_rec("hi"))

    # With tenacity retrying, we expect multiple attempts with use_fast=True
    # followed by a final successful load without use_fast.
    assert len(call_history) >= 2
    assert all(k.get("use_fast") is True for k in call_history[:-1])
    assert "use_fast" not in call_history[-1]
    assert isinstance(op.tok, FlakyTokenizer)


def test_setup_raises_other_type_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure we don't swallow TypeErrors that are unrelated to 'use_fast'.
    """
    fake_mod = types.ModuleType("transformers")

    class MockAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            raise TypeError("Something else completely broken")

    fake_mod.AutoTokenizer = MockAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_mod)

    op = TokenizeText(tokenizer_id="broken-model", field="text")
    with pytest.raises(TypeError, match="Something else completely broken"):
        _ = _setup(op).process_one(_rec("hello"))


# --- Batch Normalization & Edge Cases ---


def test_normalize_batch_logic() -> None:
    """
    Directly tests _normalize_batch logic used for backend compatibility.
    """
    op = TokenizeText(tokenizer_id="__fallback__", field="text")

    # Case 1: Standard List - Matches Batch Size
    batch_list = [[1, 2], [3, 4]]
    norm = op._normalize_batch(batch_list, batch_size=2)
    assert norm == batch_list

    # Case 2: Standard List - Single Item Mismatch
    # Tokenizer output: [1, 2] (one sequence)
    # Batch size: 1
    # Should wrap in list -> [[1, 2]]
    single_seq = [1, 2]
    norm = op._normalize_batch(single_seq, batch_size=1)
    assert len(norm) == 1
    assert norm[0] == single_seq

    # Case 3: Mock Tensor with Shape [B, T] (Matches)
    class MockTensor:
        shape = (2, 5)

    t = MockTensor()
    norm = op._normalize_batch(t, batch_size=2)  # type: ignore
    assert norm is t

    # Case 4: Mock Tensor with Shape [T] (Mismatch / Squeezed)
    # Output: Tensor shape (5,) -> single sequence
    # Batch size: 1
    # Should wrap -> [Tensor]
    class MockSqueezedTensor:
        shape = (5,)

    t2 = MockSqueezedTensor()
    norm = op._normalize_batch(t2, batch_size=1)  # type: ignore
    assert isinstance(norm, list)
    assert norm[0] is t2

    # Case 5: 0-d Tensor (Scalar) edge case
    class MockScalar:
        shape = ()

    t3 = MockScalar()
    norm = op._normalize_batch(t3, batch_size=1)  # type: ignore
    assert isinstance(norm, list)
    assert norm[0] is t3


def test_empty_batch_processing() -> None:
    """Ensure processing an empty list returns empty list immediately."""
    op = _setup(TokenizeText(tokenizer_id="__fallback__", field="text"))
    assert op.process_many([]) == []


def test_empty_string_input() -> None:
    """Ensure empty strings are handled gracefully by fallback."""
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            special_tokens="tokenizer_default",
        )
    )
    out = op.process_one(_rec(""))
    payload = _payload_dict(out[0])
    assert payload["input_ids"] == []
    # Fallback logic: 0 tokens -> empty list.
    # If add_attention_mask is True (default), it creates empty mask
    assert payload["attention_mask"] == []


# --- Splitting Boundary Conditions ---


def test_splitting_exact_boundaries() -> None:
    """
    Test splitting behavior at exact multiples of max_length.
    Fallback tokenizer: 1 word = 1 token.
    """
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            split_long_samples=True,
            max_length=2,
            padding=False,
            special_tokens="tokenizer_default",
        )
    )

    # Case 1: Length < Max (1 token)
    # Expect: 1 Record
    out = op.process_one(_rec("a"))
    assert len(out) == 1
    assert len(out[0].payload["input_ids"]) == 1

    # Case 2: Length == Max (2 tokens)
    # Expect: 1 Record (should not split needlessly if it fits exactly)
    out = op.process_one(_rec("a b"))
    assert len(out) == 1
    assert len(out[0].payload["input_ids"]) == 2

    # Case 3: Length == Max + 1 (3 tokens)
    # Expect: 2 Records (2 tokens, 1 token)
    out = op.process_one(_rec("a b c"))
    assert len(out) == 2
    assert len(out[0].payload["input_ids"]) == 2
    assert len(out[1].payload["input_ids"]) == 1

    # Verify Lineage
    assert out[0].meta.lineage == (0,)
    assert out[1].meta.lineage == (1,)

    # Verify Tombstone/End flags
    # Note: Implementation logic implies 1:N split.
    # Check contributors on the children
    assert not out[0].meta.contributors[0].is_last_child
    assert out[1].meta.contributors[0].is_last_child


def test_splitting_with_padding_fill() -> None:
    """
    Test that the last segment is correctly padded when splitting.
    """
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            split_long_samples=True,
            max_length=3,
            padding=True,
            special_tokens="tokenizer_default",
        )
    )

    # Input: 4 tokens ("a b c d") -> Split into [3, 1]
    # Second segment should pad from 1 to 3
    out = op.process_one(_rec("a b c d"))

    assert len(out) == 2

    # First segment: [a, b, c]
    assert len(out[0].payload["input_ids"]) == 3

    # Second segment: [d, pad, pad]
    ids = out[1].payload["input_ids"]
    mask = out[1].payload["attention_mask"]

    assert len(ids) == 3
    assert len(mask) == 3
    # Check padding values (fallback uses 0 for pad)
    assert ids[1] == 0
    assert mask[1] == 0


# --- Free-threaded Python tensor iteration safety tests ---


def test_normalize_batch_converts_torch_tensor_to_tuple_when_lock_set() -> None:
    """Test that _normalize_batch converts PyTorch tensors to tuples on free-threaded Python."""
    torch = pytest.importorskip("torch")
    import threading

    from zephon.ops import tokenize_text

    # Save original lock value
    original_lock = tokenize_text._TENSOR_ITER_LOCK

    try:
        # Simulate free-threaded Python by setting the lock
        tokenize_text._TENSOR_ITER_LOCK = threading.Lock()

        op = TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            return_tensors="pt",
        )

        # Create a 2D tensor (batch_size=3, seq_len=4)
        tensor = torch.randn(3, 4)

        result = op._normalize_batch(tensor, batch_size=3)

        # Should return a tuple of tensors, not the original tensor
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3
        for i, row in enumerate(result):
            assert isinstance(row, torch.Tensor)
            assert row.shape == (4,)
            # Verify the data is correct
            assert torch.allclose(row, tensor[i])

    finally:
        # Restore original lock value
        tokenize_text._TENSOR_ITER_LOCK = original_lock


def test_normalize_batch_returns_tensor_when_lock_not_set() -> None:
    """Test that _normalize_batch returns tensor as-is on regular Python (with GIL)."""
    torch = pytest.importorskip("torch")

    from zephon.ops import tokenize_text

    # Save original lock value
    original_lock = tokenize_text._TENSOR_ITER_LOCK

    try:
        # Simulate regular Python by clearing the lock
        tokenize_text._TENSOR_ITER_LOCK = None

        op = TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            return_tensors="pt",
        )

        tensor = torch.randn(3, 4)

        result = op._normalize_batch(tensor, batch_size=3)

        # Should return the original tensor as-is
        assert result is tensor

    finally:
        tokenize_text._TENSOR_ITER_LOCK = original_lock


def test_normalize_batch_leaves_numpy_unchanged_even_with_lock() -> None:
    """Test that _normalize_batch does not convert numpy arrays (only PyTorch has the race)."""
    np = pytest.importorskip("numpy")
    import threading

    from zephon.ops import tokenize_text

    # Save original lock value
    original_lock = tokenize_text._TENSOR_ITER_LOCK

    try:
        # Simulate free-threaded Python by setting the lock
        tokenize_text._TENSOR_ITER_LOCK = threading.Lock()

        op = TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            return_tensors="np",
        )

        arr = np.random.randn(3, 4)

        result = op._normalize_batch(arr, batch_size=3)

        # Should return the original numpy array as-is (not converted to tuple)
        assert result is arr

    finally:
        tokenize_text._TENSOR_ITER_LOCK = original_lock


# --- PyTorch version detection tests ---


def test_torch_has_allocator_fix_detects_old_versions() -> None:
    """Test that _torch_has_allocator_fix returns False for PyTorch < 2.10."""
    from unittest.mock import MagicMock, patch

    from zephon.utils import torch_compat

    mock_torch = MagicMock()

    # Test version 2.9.0 - should return False
    mock_torch.__version__ = "2.9.0"
    with patch.dict("sys.modules", {"torch": mock_torch}):
        # Need to reimport the function to use the mocked torch
        result = torch_compat._torch_has_allocator_fix()
        assert result is False, "2.9.0 should not have the fix"

    # Test version 2.5.1 - should return False
    mock_torch.__version__ = "2.5.1"
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = torch_compat._torch_has_allocator_fix()
        assert result is False, "2.5.1 should not have the fix"


# --- Cloud tokenizer URI resolution ---


def test_setup_resolves_cloud_uri_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """``AutoTokenizer.from_pretrained`` receives the resolved local path, not the URI."""
    received_ids: list[str] = []
    fake = types.ModuleType("transformers")

    class _Tok:
        name_or_path = "model"
        pad_token = None
        eos_token = 0

        def __call__(self, texts, **kwargs):  # pragma: no cover - unused
            return {
                "input_ids": [[1, 2] for _ in texts],
                "attention_mask": [[1, 1] for _ in texts],
            }

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **k: Any):
            received_ids.append(name)
            return _Tok()

    fake.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake)

    resolved_path = str(tmp_path / "fake-resolved-tokenizer")

    def fake_resolve(tokenizer_id):
        assert tokenizer_id == "s3://fake-bucket/fake/prefix/"
        return resolved_path

    monkeypatch.setattr(
        "zephon.utils.tokenizer_cloud.resolve_tokenizer_id", fake_resolve
    )

    op = TokenizeText(
        tokenizer=None,
        tokenizer_id="s3://fake-bucket/fake/prefix/",
        field="text",
        special_tokens="tokenizer_default",
    )
    # tokenizer_id stays as the URI; resolution is lazy.
    assert op.tokenizer_id == "s3://fake-bucket/fake/prefix/"
    assert op.tok is None

    _setup(op)
    _ = op.process_one(_rec("hello"))

    assert received_ids == [resolved_path]


def test_setup_passes_hub_id_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HF Hub ids are not cloud URIs; the resolver must pass them through."""
    received_ids: list[str] = []
    fake = types.ModuleType("transformers")

    class _Tok:
        name_or_path = "meta-llama/Llama-3.2-1B"
        pad_token = None
        eos_token = 0

        def __call__(self, texts, **kwargs):  # pragma: no cover - unused
            return {
                "input_ids": [[1, 2] for _ in texts],
                "attention_mask": [[1, 1] for _ in texts],
            }

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **k: Any):
            received_ids.append(name)
            return _Tok()

    fake.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake)

    op = TokenizeText(
        tokenizer=None,
        tokenizer_id="meta-llama/Llama-3.2-1B",
        field="text",
        special_tokens="tokenizer_default",
    )
    _setup(op)
    _ = op.process_one(_rec("hello"))

    assert received_ids == ["meta-llama/Llama-3.2-1B"]


def test_setup_does_not_resolve_when_tokenizer_passed_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing a pre-built tokenizer object short-circuits the resolver."""
    calls: list[str | None] = []

    def fake_resolve(tokenizer_id):
        calls.append(tokenizer_id)
        return tokenizer_id

    monkeypatch.setattr(
        "zephon.utils.tokenizer_cloud.resolve_tokenizer_id", fake_resolve
    )

    class _Tok:
        name_or_path = "user-supplied"
        pad_token = None
        eos_token = 0

        def __call__(self, texts, **kwargs):
            return {
                "input_ids": [[1, 2] for _ in texts],
                "attention_mask": [[1, 1] for _ in texts],
            }

    tok = _Tok()
    op = TokenizeText(
        tokenizer=tok,
        tokenizer_id=None,
        field="text",
        special_tokens="tokenizer_default",
    )
    _setup(op)
    _ = op.process_one(_rec("hello"))

    assert calls == [], "resolve_tokenizer_id must not run when tok is provided"


def test_setup_retries_transient_cloud_sync_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Setup retries a transient resolver failure, then loads the tokenizer."""
    resolved_path = str(tmp_path / "fake-resolved-tokenizer")
    resolve_attempts = 0

    def flaky_resolve(tokenizer_id):
        nonlocal resolve_attempts
        resolve_attempts += 1
        if resolve_attempts == 1:
            raise TransientCloudTokenizerError("simulated transient S3 5xx")
        return resolved_path

    monkeypatch.setattr(
        "zephon.utils.tokenizer_cloud.resolve_tokenizer_id", flaky_resolve
    )

    # Drop the resolve backoff so the test runs fast. The retry envelope is
    # built at import, so mutate the controller rather than patch the wait.
    monkeypatch.setattr(resolve_tokenizer_id_with_retry.retry, "wait", wait_none())

    fake = types.ModuleType("transformers")

    class _Tok:
        name_or_path = "model"
        pad_token = None
        eos_token = 0

        def __call__(self, texts, **kwargs):  # pragma: no cover - unused
            return {
                "input_ids": [[1, 2] for _ in texts],
                "attention_mask": [[1, 1] for _ in texts],
            }

    received_ids: list[str] = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **k: Any):
            received_ids.append(name)
            return _Tok()

    fake.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake)

    op = TokenizeText(
        tokenizer=None,
        tokenizer_id="s3://fake-bucket/fake/prefix/",
        field="text",
        special_tokens="tokenizer_default",
    )
    _setup(op)
    _ = op.process_one(_rec("hello"))

    assert resolve_attempts == 2, "Transient sync error must be retried once"
    assert received_ids == [resolved_path]


def test_torch_has_allocator_fix_detects_new_versions() -> None:
    """Test that _torch_has_allocator_fix returns True for PyTorch >= 2.10."""
    from unittest.mock import MagicMock, patch

    from zephon.utils import torch_compat

    mock_torch = MagicMock()

    # Test version 2.10.0 - should return True
    mock_torch.__version__ = "2.10.0"
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = torch_compat._torch_has_allocator_fix()
        assert result is True, "2.10.0 should have the fix"

    # Test version 2.10.0+cu124 - should return True (strip build metadata)
    mock_torch.__version__ = "2.10.0+cu124"
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = torch_compat._torch_has_allocator_fix()
        assert result is True, "2.10.0+cu124 should have the fix"

    # Test version 3.0.0 - should return True
    mock_torch.__version__ = "3.0.0"
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = torch_compat._torch_has_allocator_fix()
        assert result is True, "3.0.0 should have the fix"

    # Test version 2.11.0.dev - should return True
    mock_torch.__version__ = "2.11.0.dev20250101"
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = torch_compat._torch_has_allocator_fix()
        assert result is True, "2.11.0.dev should have the fix"


def test_torch_has_allocator_fix_handles_missing_torch() -> None:
    """Test that _torch_has_allocator_fix returns False when torch is not installed."""
    from unittest.mock import patch

    from zephon.utils import torch_compat

    # Mock torch import to raise ImportError
    with patch.dict("sys.modules", {"torch": None}):
        result = torch_compat._torch_has_allocator_fix()
        assert result is False, "Should return False when torch is not available"


def test_should_use_tensor_lock_when_gil_enabled() -> None:
    """Test that _should_use_tensor_lock returns False when GIL is enabled."""
    from unittest.mock import patch

    from zephon.utils import torch_compat

    with patch.object(torch_compat, "_gil_disabled", return_value=False):
        result = torch_compat._should_use_tensor_lock()
        assert result is False, "Should return False when GIL is enabled"


def test_should_use_tensor_lock_when_torch_fixed() -> None:
    """Test that _should_use_tensor_lock returns False when torch >= 2.10."""
    from unittest.mock import patch

    from zephon.utils import torch_compat

    with (
        patch.object(torch_compat, "_gil_disabled", return_value=True),
        patch.object(torch_compat, "_torch_has_allocator_fix", return_value=True),
    ):
        result = torch_compat._should_use_tensor_lock()
        assert result is False, "Should return False when torch has the fix"


def test_should_use_tensor_lock_when_needed() -> None:
    """Test that _should_use_tensor_lock returns True when GIL disabled and torch < 2.10."""
    from unittest.mock import patch

    from zephon.utils import torch_compat

    with (
        patch.object(torch_compat, "_gil_disabled", return_value=True),
        patch.object(torch_compat, "_torch_has_allocator_fix", return_value=False),
    ):
        result = torch_compat._should_use_tensor_lock()
        assert result is True, "Should return True when GIL disabled and torch unfixed"


# ---------------------------------------------------------------------------
# special_tokens API: explicit BOS/EOS bracketing (new default "bos_eos")
# ---------------------------------------------------------------------------


# Fallback's resolved ids — keep tests readable.
_BOS = 1
_EOS = 2
_PAD = 0


def test_special_tokens_default_brackets_bos_and_eos() -> None:
    """The default ``special_tokens="bos_eos"`` mode brackets every input
    document explicitly as ``[BOS, ...content, EOS]``."""
    op = _setup(TokenizeText(tokenizer_id="__fallback__", field="text"))
    out = op.process_one(_rec("hello world"))[0]
    ids = _payload_dict(out)["input_ids"]
    mask = _payload_dict(out)["attention_mask"]
    # 2 content tokens + BOS + EOS = 4
    assert len(ids) == 4
    assert ids[0] == _BOS
    assert ids[-1] == _EOS
    # Attention mask is 1 on both bracket positions.
    assert mask[0] == 1
    assert mask[-1] == 1


def test_special_tokens_bos_only() -> None:
    op = _setup(
        TokenizeText(tokenizer_id="__fallback__", field="text", special_tokens="bos")
    )
    out = op.process_one(_rec("a b c"))[0]
    ids = _payload_dict(out)["input_ids"]
    assert ids[0] == _BOS
    assert ids[-1] != _EOS  # no EOS appended
    assert len(ids) == 4  # 3 content + BOS


def test_special_tokens_eos_only() -> None:
    op = _setup(
        TokenizeText(tokenizer_id="__fallback__", field="text", special_tokens="eos")
    )
    out = op.process_one(_rec("a b c"))[0]
    ids = _payload_dict(out)["input_ids"]
    assert ids[0] != _BOS
    assert ids[-1] == _EOS
    assert len(ids) == 4  # 3 content + EOS


def test_special_tokens_none_emits_raw_content() -> None:
    op = _setup(
        TokenizeText(tokenizer_id="__fallback__", field="text", special_tokens="none")
    )
    out = op.process_one(_rec("a b c"))[0]
    ids = _payload_dict(out)["input_ids"]
    assert _BOS not in ids[:1]
    assert _EOS not in ids[-1:]
    assert len(ids) == 3  # exactly content


def test_special_tokens_tokenizer_default_delegates_to_template() -> None:
    """Legacy path: kwargs carry add_special_tokens=True; we do not bracket."""
    captured: dict[str, Any] = {}

    class _Tok:
        name_or_path = "tt"
        pad_token = None
        pad_token_id = None
        bos_token_id = 99
        eos_token_id = 100
        eos_token = 100

        def __call__(self, texts, **kwargs):
            captured.update(kwargs)
            return {
                "input_ids": [[5, 6, 7] for _ in texts],
                "attention_mask": [[1, 1, 1] for _ in texts],
            }

    tok = _Tok()
    op = _setup(
        TokenizeText(tokenizer=tok, field="text", special_tokens="tokenizer_default")
    )
    out = op.process_many([_rec("x")])
    # Tokenizer template owns specials; we do not add 99/100.
    assert _payload_dict(out[0])["input_ids"] == [5, 6, 7]
    assert captured["add_special_tokens"] is True


def test_special_tokens_bracket_mode_calls_hf_with_add_special_false() -> None:
    """Bracket modes pass add_special_tokens=False so the template is bypassed."""
    captured: dict[str, Any] = {}

    class _Tok:
        name_or_path = "tt"
        pad_token = None
        pad_token_id = None
        bos_token_id = 11
        eos_token_id = 22

        def __call__(self, texts, **kwargs):
            captured.update(kwargs)
            return {
                "input_ids": [[5, 6] for _ in texts],
                "attention_mask": [[1, 1] for _ in texts],
            }

    tok = _Tok()
    op = _setup(TokenizeText(tokenizer=tok, field="text"))  # default = bos_eos
    out = op.process_many([_rec("x")])
    assert captured["add_special_tokens"] is False
    ids = _payload_dict(out[0])["input_ids"]
    assert ids[0] == 11 and ids[-1] == 22
    assert ids[1:-1] == [5, 6]


def test_invalid_special_tokens_value_raises_in_init() -> None:
    with pytest.raises(ValueError, match="special_tokens"):
        TokenizeText(
            tokenizer_id="__fallback__", field="text", special_tokens="weird_mode"
        )  # type: ignore[arg-type]


def test_bracket_mode_missing_bos_token_raises_eagerly() -> None:
    """Tokenizer with no bos_token_id + bos-using mode + no override → eager error."""

    class _NoBos:
        name_or_path = "no-bos"
        pad_token = None
        pad_token_id = None
        bos_token_id = None
        eos_token_id = 5

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1] for _ in texts], "attention_mask": [[1]]}

    op = _setup(
        TokenizeText(tokenizer=_NoBos(), field="text", special_tokens="bos_eos")
    )
    with pytest.raises(ValueError, match="requires a BOS token"):
        op.process_one(_rec("x"))


def test_bracket_mode_missing_eos_token_raises_eagerly() -> None:
    class _NoEos:
        name_or_path = "no-eos"
        pad_token = None
        pad_token_id = None
        bos_token_id = 7
        eos_token_id = None

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1] for _ in texts], "attention_mask": [[1]]}

    op = _setup(
        TokenizeText(tokenizer=_NoEos(), field="text", special_tokens="bos_eos")
    )
    with pytest.raises(ValueError, match="requires an EOS token"):
        op.process_one(_rec("x"))


def test_bos_token_id_override_wins_over_tokenizer_attr() -> None:
    """Explicit override beats whatever the tokenizer exposes."""

    class _Tok:
        name_or_path = "tt"
        pad_token = None
        pad_token_id = None
        bos_token_id = 100  # tokenizer's choice
        eos_token_id = 200

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[9] for _ in texts], "attention_mask": [[1]]}

    op = _setup(
        TokenizeText(
            tokenizer=_Tok(),
            field="text",
            special_tokens="bos_eos",
            bos_token_id=42,  # override
            eos_token_id=99,  # override
        )
    )
    out = op.process_one(_rec("x"))[0]
    ids = _payload_dict(out)["input_ids"]
    assert ids[0] == 42
    assert ids[-1] == 99


def test_override_unblocks_tokenizer_without_bos_eos() -> None:
    """Overriding ids lets bracketing work on tokenizers that lack the attributes."""

    class _Bare:
        name_or_path = "bare"
        pad_token = None
        pad_token_id = None
        # deliberately no bos_token_id / eos_token_id

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[3, 4] for _ in texts], "attention_mask": [[1, 1]]}

    op = _setup(
        TokenizeText(
            tokenizer=_Bare(),
            field="text",
            special_tokens="bos_eos",
            bos_token_id=8,
            eos_token_id=9,
        )
    )
    out = op.process_one(_rec("x"))[0]
    assert _payload_dict(out)["input_ids"] == [8, 3, 4, 9]


def test_split_long_samples_bos_eos_lands_only_on_outer_chunks() -> None:
    """Bracket once per *document*, then slice — BOS on first chunk, EOS on last."""
    # Content "a b c d" = 4 fallback tokens; bracket → 6; max_length=3 → 2 chunks.
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            split_long_samples=True,
            max_length=3,
            padding=False,
            # default special_tokens="bos_eos"
        )
    )
    out = op.process_one(_rec("a b c d"))
    assert len(out) == 2
    first_ids = _payload_dict(out[0])["input_ids"]
    last_ids = _payload_dict(out[1])["input_ids"]
    assert first_ids[0] == _BOS
    assert _EOS not in first_ids  # no EOS at slice boundary
    assert _BOS not in last_ids  # no BOS at slice boundary
    assert last_ids[-1] == _EOS
    # First chunk fully consumed by [BOS, content, content], no padding needed.
    assert len(first_ids) == 3
    assert len(last_ids) == 3


def test_split_long_samples_with_tokenizer_default_forces_hf_padding_off() -> None:
    """Regression: under ``tokenizer_default`` + ``split_long_samples`` the HF
    call must still receive ``padding=False``. The operator's per-chunk
    padding owns the output length; forwarding ``self.padding`` to HF here
    would pad the whole sequence *before* it is sliced, inflating every
    chunk to the pre-split padded length and wasting tokens."""
    captured: dict[str, Any] = {}

    class _Tok:
        name_or_path = "tt"
        pad_token = "<pad>"
        pad_token_id = 0
        bos_token_id = 11
        eos_token_id = 22

        def __call__(self, texts, **kwargs):
            captured.update(kwargs)
            # 6 content tokens regardless of HF's padding setting; the test
            # is about what we ASK HF for, not what HF returns.
            return {
                "input_ids": [[5, 6, 7, 8, 9, 10] for _ in texts],
                "attention_mask": [[1, 1, 1, 1, 1, 1] for _ in texts],
            }

    op = _setup(
        TokenizeText(
            tokenizer=_Tok(),
            field="text",
            special_tokens="tokenizer_default",
            split_long_samples=True,
            max_length=3,
            padding="max_length",
        )
    )
    out = op.process_many([_rec("a b c d e f")])
    assert captured["padding"] is False
    # Operator slices into max_length=3 chunks (6 tokens → 2 chunks of 3).
    chunks = [_payload_dict(r)["input_ids"] for r in out]
    assert len(chunks) == 2
    assert all(len(c) == 3 for c in chunks)


def test_split_long_samples_bos_eos_three_chunks() -> None:
    """Longer doc: BOS on chunk 0, EOS on the LAST chunk only, nothing in middle."""
    # 8 content tokens, bracket → 10, max_length=4 → 3 chunks (4, 4, 2).
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            split_long_samples=True,
            max_length=4,
            padding=False,
        )
    )
    out = op.process_one(_rec("a b c d e f g h"))
    assert len(out) == 3
    ids = [_payload_dict(r)["input_ids"] for r in out]
    assert ids[0][0] == _BOS
    assert _EOS not in ids[0]
    assert _BOS not in ids[1] and _EOS not in ids[1]
    assert ids[2][-1] == _EOS
    # is_last_child set correctly on the EOS-bearing chunk only.
    assert out[0].meta.contributors[0].is_last_child is False
    assert out[1].meta.contributors[0].is_last_child is False
    assert out[2].meta.contributors[0].is_last_child is True


def test_bracket_with_padding_pads_to_max_length() -> None:
    """In bracket mode, padding=True pads the bracketed sequence to max_length."""
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            padding="max_length",
            max_length=8,
            truncation=False,
            # default special_tokens="bos_eos"
        )
    )
    out = op.process_one(_rec("hello world"))[0]
    payload = _payload_dict(out)
    ids = payload["input_ids"]
    mask = payload["attention_mask"]
    # Real content: BOS + 2 content + EOS = 4 tokens. Pad to 8 with pad=0.
    assert len(ids) == 8
    assert ids[0] == _BOS
    assert ids[3] == _EOS  # last real content position
    assert ids[4:] == [_PAD] * 4  # padding
    assert mask[:4] == [1, 1, 1, 1]
    assert mask[4:] == [0, 0, 0, 0]


def test_bracket_with_truncation_reserves_room_for_specials() -> None:
    """When truncation=True, HF receives max_length - num_specials so the
    final bracketed output fits exactly in the user's max_length."""
    captured: dict[str, Any] = {}

    class _Tok:
        name_or_path = "tt"
        pad_token = None
        pad_token_id = None
        bos_token_id = 11
        eos_token_id = 22

        def __call__(self, texts, **kwargs):
            captured.update(kwargs)
            # Pretend to truncate to whatever max_length was requested.
            mx = kwargs.get("max_length", len(texts[0].split()))
            ids = list(range(50, 50 + mx))
            return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}

    op = _setup(
        TokenizeText(
            tokenizer=_Tok(),
            field="text",
            truncation=True,
            max_length=6,
            special_tokens="bos_eos",
        )
    )
    out = op.process_one(_rec("a b c d e f g h"))[0]
    # HF should have been called with max_length=4 (6 - 2 specials).
    assert captured["max_length"] == 4
    ids = _payload_dict(out)["input_ids"]
    # Final length is the user's max_length, bracketed.
    assert len(ids) == 6
    assert ids[0] == 11 and ids[-1] == 22


def test_bracket_max_length_too_small_for_specials_raises_in_init() -> None:
    """max_length=1 with bos_eos (2 specials) has zero content room → fail fast.

    Validation moved from ``setup()`` to ``__init__`` so misconfigured ops
    surface at construction rather than the first batch.
    """
    with pytest.raises(ValueError, match="cannot fit"):
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            truncation=True,
            max_length=1,  # cannot fit BOS + EOS
            special_tokens="bos_eos",
        )


def test_bracket_with_numpy_return_tensors() -> None:
    np = pytest.importorskip("numpy")
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            return_tensors="np",
            # default special_tokens="bos_eos"
        )
    )
    out = op.process_one(_rec("a b"))[0]
    ids = _payload_dict(out)["input_ids"]
    mask = _payload_dict(out)["attention_mask"]
    assert isinstance(ids, np.ndarray)
    assert isinstance(mask, np.ndarray)
    assert ids[0] == _BOS
    assert ids[-1] == _EOS
    assert mask[0] == 1 and mask[-1] == 1


def test_bracket_with_torch_return_tensors() -> None:
    torch = pytest.importorskip("torch")
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            return_tensors="pt",
            # default special_tokens="bos_eos"
        )
    )
    out = op.process_one(_rec("a b"))[0]
    ids = _payload_dict(out)["input_ids"]
    assert isinstance(ids, torch.Tensor)
    assert ids[0].item() == _BOS
    assert ids[-1].item() == _EOS


def test_bracket_one_std_no_existing_mask_creates_one() -> None:
    """Direct unit test on _bracket_std: when mask is None and
    add_attention_mask=True, a mask of 1s is created at the bracketed length."""
    op = _setup(TokenizeText(tokenizer_id="__fallback__", field="text"))
    # _resolve_special_token_ids needs to have run; force it.
    op._setup_tokenizer()
    new_ids, new_mask = op._bracket_std([7, 8, 9], None, bos=_BOS, eos=_EOS)
    assert new_ids == [_BOS, 7, 8, 9, _EOS]
    assert new_mask == [1, 1, 1, 1, 1]


def test_bracket_one_std_existing_mask_extended() -> None:
    op = _setup(TokenizeText(tokenizer_id="__fallback__", field="text"))
    op._setup_tokenizer()
    new_ids, new_mask = op._bracket_std([7, 8], [1, 1], bos=_BOS, eos=_EOS)
    assert new_ids == [_BOS, 7, 8, _EOS]
    # Mask gets a 1 at each new bracket position.
    assert new_mask == [1, 1, 1, 1]


def test_bracket_mode_runs_in_python_lists_then_converts() -> None:
    """In bracket mode the operator does *not* ask HF for tensors — it pulls
    lists from HF, brackets/pads in Python (one allocation-free path that
    works regardless of whether the batch is ragged), and converts to the
    user's requested backend at the end. The per-backend bracket helpers
    (``_bracket_numpy`` / ``_bracket_torch`` / ``_bracket_tf``) only fire as
    a safety net for user-provided tokenizers that emit a backend regardless
    of the ``return_tensors`` kwarg we passed (which is ``None`` here)."""
    np = pytest.importorskip("numpy")

    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            return_tensors="np",
            # default special_tokens="bos_eos"
        )
    )
    called_std = {"n": 0}
    called_numpy = {"n": 0}
    real_std = op._bracket_std
    real_numpy = op._bracket_numpy

    def spy_std(*args, **kwargs):
        called_std["n"] += 1
        return real_std(*args, **kwargs)

    def spy_numpy(*args, **kwargs):
        called_numpy["n"] += 1
        return real_numpy(*args, **kwargs)

    op._bracket_std = spy_std  # type: ignore[method-assign]
    op._bracket_numpy = spy_numpy  # type: ignore[method-assign]
    out = op.process_one(_rec("a b c"))[0]
    payload = _payload_dict(out)
    # User's requested backend is honored.
    assert isinstance(payload["input_ids"], np.ndarray)
    # But the bracket itself ran in lists, because HF returned lists.
    assert called_std["n"] == 1
    assert called_numpy["n"] == 0


def test_bracket_mode_keeps_lists_when_return_tensors_is_none() -> None:
    """With ``return_tensors=None`` the tokenizer returns Python lists and
    bracket runs through the list code path. All four per-framework bracket
    helpers (std/numpy/torch/tf) stay reachable across realistic configs."""
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            return_tensors=None,
            # default special_tokens="bos_eos"
        )
    )
    out = op.process_one(_rec("a b c"))[0]
    payload = _payload_dict(out)
    assert isinstance(payload["input_ids"], list)


def test_bracket_pt_with_free_threaded_race_uses_from_numpy_under_lock() -> None:
    """When the free-threaded-Python + old-PyTorch combination is active and
    the user requests ``return_tensors="pt"`` in bracket mode, the operator
    must NOT call ``torch.tensor`` directly on the bracketed lists — that
    path allocates a torch wrapper, which races on free-threaded + old
    PyTorch. The end-of-pipeline ``_convert_tensor`` therefore detours
    through numpy (``np.asarray`` then ``torch.from_numpy`` under
    ``_tensor_lock_ctx``).
    See: https://github.com/pytorch/pytorch/issues/171992
    """
    torch = pytest.importorskip("torch")
    from unittest.mock import patch

    # Simulate the free-threaded + old-torch combination.
    with patch("zephon.ops.tokenize_text._should_use_tensor_lock", return_value=True):
        op = _setup(
            TokenizeText(tokenizer_id="__fallback__", field="text", return_tensors="pt")
        )
        # Bracket mode no longer forwards return_tensors to HF.
        assert "return_tensors" not in op._cached_kwargs
        assert op._convert_np_to_pt is False

        real_from_numpy = torch.from_numpy
        with patch.object(
            torch, "from_numpy", side_effect=real_from_numpy
        ) as mock_from_numpy:
            out = op.process_one(_rec("hello world"))[0]
        payload = _payload_dict(out)
        assert isinstance(payload["input_ids"], torch.Tensor)
        # Per-row conversion lifts each list via from_numpy under the lock.
        assert mock_from_numpy.call_count >= 1
        assert payload["input_ids"][0].item() == _BOS


def test_bracket_mode_does_not_forward_return_tensors_to_hf() -> None:
    """Bracket mode does *not* forward ``return_tensors`` to HF. With HF
    padding forced off so BOS/EOS can attach to real content, forwarding a
    tensor backend would force HF to build a 2-D tensor on a ragged batch
    and raise. Instead the operator pulls lists from HF, brackets/pads in
    Python, then converts to the user's backend at the end."""
    captured: dict[str, Any] = {}

    class _Tok:
        name_or_path = "tt"
        pad_token = None
        pad_token_id = None
        bos_token_id = 11
        eos_token_id = 22

        def __call__(self, texts, **kwargs):
            captured.update(kwargs)
            return {
                "input_ids": [[1, 2] for _ in texts],
                "attention_mask": [[1, 1] for _ in texts],
            }

    op = _setup(TokenizeText(tokenizer=_Tok(), field="text", return_tensors="np"))
    op.process_many([_rec("x")])
    assert "return_tensors" not in captured
    # HF padding must stay off so brackets attach to real content, not to
    # the far side of pad tokens.
    assert captured["padding"] is False


# ---------------------------------------------------------------------------
# Bracket mode + tensor backend + variable-length batches: the path the
# previous version raised on (it forwarded return_tensors to HF, which then
# tried to build a 2-D tensor from a deliberately-ragged batch).
# ---------------------------------------------------------------------------


def test_bracket_mode_ragged_numpy_with_longest_padding() -> None:
    """``padding=True`` (longest-in-batch) + ``return_tensors='np'`` + bracket
    on a ragged batch must produce a 2-D numpy array padded to the longest
    bracketed row, *not* raise. This is the path the previous version
    documented as a 'known limitation'."""
    np = pytest.importorskip("numpy")

    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            return_tensors="np",
            padding=True,
        )
    )
    out = op.process_many([_rec("a b"), _rec("c d e f")])
    # Longest content = 4 words → 4 + 2 specials = 6.
    rows = [_payload_dict(r)["input_ids"] for r in out]
    assert all(isinstance(r, np.ndarray) for r in rows)
    assert [r.shape[0] for r in rows] == [6, 6]
    # The shorter row is pad-filled (with attention_mask=0).
    masks = [_payload_dict(r)["attention_mask"] for r in out]
    assert masks[0].tolist() == [1, 1, 1, 1, 0, 0]
    assert masks[1].tolist() == [1, 1, 1, 1, 1, 1]


def test_bracket_mode_ragged_torch_with_max_length_padding() -> None:
    """``padding='max_length'`` + ``return_tensors='pt'`` + bracket on a
    ragged batch produces torch tensors padded to ``max_length``."""
    torch = pytest.importorskip("torch")
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            return_tensors="pt",
            padding="max_length",
            max_length=8,
        )
    )
    out = op.process_many([_rec("a b"), _rec("c d e f")])
    rows = [_payload_dict(r)["input_ids"] for r in out]
    assert all(isinstance(r, torch.Tensor) for r in rows)
    assert [r.shape[0] for r in rows] == [8, 8]
    # BOS at front, EOS adjacent to real content (not after pad), pads tail.
    row0 = rows[0].tolist()
    assert row0[0] == _BOS
    assert row0[3] == _EOS  # 1 BOS + 2 content + EOS = position 3
    assert row0[4:] == [_PAD, _PAD, _PAD, _PAD]


def test_bracket_mode_split_long_samples_with_torch_return_tensors_ragged() -> None:
    """``split_long_samples`` + bracket + ``return_tensors='pt'`` over a
    ragged batch: each input fans out into per-chunk torch tensors. Used to
    raise because we asked HF for ``pt`` on a ragged batch."""
    torch = pytest.importorskip("torch")
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            split_long_samples=True,
            max_length=4,
            return_tensors="pt",
        )
    )
    out = op.process_many([_rec("a b"), _rec("c d e f g h i")])
    # Sample 1: 2 content + BOS + EOS = 4 tokens → 1 chunk (4).
    # Sample 2: 7 content + BOS + EOS = 9 tokens → 3 chunks (4, 4, 1).
    assert len(out) == 4
    for rec in out:
        ids = _payload_dict(rec)["input_ids"]
        assert isinstance(ids, torch.Tensor)


# ---------------------------------------------------------------------------
# C3: padding=True is "pad to longest-in-batch", not "pad to max_length".
# ---------------------------------------------------------------------------


def test_padding_true_pads_to_longest_not_max_length() -> None:
    """``padding=True`` with ``max_length`` set still pads to the longest
    bracketed row in the batch, not to ``max_length``. (HF semantics:
    ``padding=True`` is "longest" — ``max_length`` only caps when truncation
    is also enabled.)"""
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__", field="text", padding=True, max_length=8
        )
    )
    out = op.process_many([_rec("a b"), _rec("c d e f")])
    lens = [len(_payload_dict(r)["input_ids"]) for r in out]
    # Longest bracketed = 4 content + 2 specials = 6, NOT max_length=8.
    assert lens == [6, 6]


def test_padding_max_length_still_pads_to_max_length() -> None:
    """``padding='max_length'`` keeps pad-to-max_length semantics."""
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            padding="max_length",
            max_length=8,
        )
    )
    out = op.process_many([_rec("a b"), _rec("c d e f")])
    lens = [len(_payload_dict(r)["input_ids"]) for r in out]
    assert lens == [8, 8]


def test_padding_do_not_pad_string_treated_as_no_padding() -> None:
    """HF accepts ``padding="do_not_pad"`` as an explicit no-padding string,
    equivalent to ``padding=False``. Bracket-owned modes must respect that
    and leave the batch ragged — the previous implementation read any truthy
    padding value as "padding active" and silently padded to longest."""
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            special_tokens="none",
            padding="do_not_pad",
        )
    )
    out = op.process_many([_rec("a"), _rec("b c d")])
    lens = [len(_payload_dict(r)["input_ids"]) for r in out]
    assert lens == [1, 3]  # ragged; no padding


def test_padding_do_not_pad_bracket_mode_keeps_rows_ragged() -> None:
    """Same as above but in a bracket mode: the bracketing still runs, but
    no padding is applied so rows of different bracketed lengths stay ragged."""
    op = _setup(
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            special_tokens="bos_eos",
            padding="do_not_pad",
        )
    )
    out = op.process_many([_rec("a"), _rec("b c d")])
    lens = [len(_payload_dict(r)["input_ids"]) for r in out]
    # Bracketed: 1+2 = 3 and 3+2 = 5; no padding flattens them.
    assert lens == [3, 5]


# ---------------------------------------------------------------------------
# Validation moves from per-batch runtime to construction time.
# ---------------------------------------------------------------------------


def test_padding_max_length_without_max_length_raises_in_init() -> None:
    """Combining ``padding='max_length'`` with no ``max_length`` was a
    runtime error in ``_pad_after_bracket``; it now raises at construction
    so the misconfiguration surfaces before the first batch."""
    with pytest.raises(ValueError, match="padding='max_length' requires max_length"):
        TokenizeText(tokenizer_id="__fallback__", field="text", padding="max_length")


# ---------------------------------------------------------------------------
# Override semantics — bos/eos overrides are bracket-only. Passing one under
# a mode that wouldn't apply it raises at __init__ rather than silently
# dropping the value, so a typo can't lead to training under the wrong BOS.
# ---------------------------------------------------------------------------


def test_override_under_tokenizer_default_raises() -> None:
    with pytest.raises(ValueError, match="has no effect under"):
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            special_tokens="tokenizer_default",
            bos_token_id=42,
        )


def test_override_under_none_raises() -> None:
    with pytest.raises(ValueError, match="has no effect under"):
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            special_tokens="none",
            eos_token_id=99,
        )


def test_override_under_bracket_mode_is_accepted() -> None:
    """Bracket modes are the only place overrides apply, so they must
    construct without raising."""
    op = TokenizeText(
        tokenizer_id="__fallback__",
        field="text",
        special_tokens="bos_eos",
        bos_token_id=42,
        eos_token_id=99,
    )
    assert op._bos_id_override == 42
    assert op._eos_id_override == 99


# ---------------------------------------------------------------------------
# Fallback tokenizer content ids must not collide with the reserved range
# (pad=0, bos=1, eos=2). PYTHONHASHSEED makes this hash-seed-dependent in
# the worst case; the offset pushes content past the reserved range.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bracket mode + padding on a tokenizer without an explicit pad_token must
# fall back to eos for the pad id (the established HF pattern for
# Llama/GPT-2/etc.). The cache had to be populated *after*
# ``_ensure_padding_token`` runs or the operator would silently pad with id
# 0 even though HF's lazy ``pad_token_id`` would have resolved to eos.
# ---------------------------------------------------------------------------


def test_bracket_padding_falls_back_to_eos_when_tokenizer_lacks_pad() -> None:
    """``_ensure_padding_token`` mutates ``tok.pad_token = eos_token``; the
    cached ``_pad_token_id`` must reflect that mutation. Regression for the
    case where the cache was populated *before* the mutation, leading to
    pad-with-0 instead of pad-with-eos."""

    class _NoPad:
        # Mirrors the HF surface for a Llama/GPT-2-style tokenizer: bos/eos
        # are *strings* and the corresponding ``*_token_id`` attrs hold the
        # ints. ``pad_token`` starts as ``None`` and ``pad_token_id``
        # resolves lazily once ``pad_token`` is set, the way real HF
        # tokenizers do via ``convert_tokens_to_ids``.
        name_or_path = "no-pad"
        bos_token = "<s>"
        bos_token_id = 11
        eos_token = "</s>"
        eos_token_id = 22

        def __init__(self) -> None:
            self._pad_token: str | None = None

        @property
        def pad_token(self) -> str | None:
            return self._pad_token

        @pad_token.setter
        def pad_token(self, value: str | None) -> None:
            self._pad_token = value

        @property
        def pad_token_id(self) -> int | None:
            # Real HF: ``convert_tokens_to_ids(pad_token)``. Here we only
            # know one mapping — when pad_token has been pointed at eos,
            # surface the eos id.
            if self._pad_token is None:
                return None
            if self._pad_token == self.eos_token:
                return self.eos_token_id
            return None

        def __call__(self, texts, **kwargs):
            return {
                "input_ids": [[5, 6] for _ in texts],
                "attention_mask": [[1, 1] for _ in texts],
            }

    op = _setup(
        TokenizeText(
            tokenizer=_NoPad(),
            field="text",
            special_tokens="bos_eos",
            padding="max_length",
            max_length=8,
        )
    )
    out = op.process_many([_rec("x")])
    ids = _payload_dict(out[0])["input_ids"]
    # Bracketed content: [BOS=11, 5, 6, EOS=22] = 4 tokens. Pad slots
    # (positions 4..7) must be eos=22, not 0.
    assert ids[:4] == [11, 5, 6, 22]
    assert ids[4:] == [22, 22, 22, 22], (
        f"pad slots should fall back to eos=22, got {ids[4:]}"
    )


# ---------------------------------------------------------------------------
# Sticky setup failure: once specials resolution raises, every subsequent
# call must re-raise rather than silently degrade to half-bracketed output.
# ---------------------------------------------------------------------------


def test_setup_failure_is_sticky_across_calls() -> None:
    """A first call that raises during specials resolution must keep raising
    on every subsequent call. The previous behavior marked the op
    ``_tokenizer_instantiated`` before the resolve, so a caller swallowing
    the first exception could continue using a half-initialized op."""

    class _NoBos:
        name_or_path = "no-bos"
        pad_token = None
        pad_token_id = None
        bos_token_id = None
        eos_token_id = 5

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1] for _ in texts], "attention_mask": [[1]]}

    op = _setup(
        TokenizeText(tokenizer=_NoBos(), field="text", special_tokens="bos_eos")
    )
    with pytest.raises(ValueError, match="requires a BOS token"):
        op.process_one(_rec("x"))
    # Second call must also raise (same exception type and message).
    with pytest.raises(ValueError, match="requires a BOS token"):
        op.process_one(_rec("y"))
    # And the op must not be marked as successfully instantiated.
    assert op._tokenizer_instantiated is False
    assert op._setup_error is not None


# ---------------------------------------------------------------------------
# truncation=True with no max_length is rejected in bracket modes — the
# implicit HF fallback to tokenizer.model_max_length cannot account for
# BOS/EOS, so the bracketed output would silently exceed the model limit.
# ---------------------------------------------------------------------------


def test_bracket_truncation_without_max_length_raises_in_init() -> None:
    with pytest.raises(ValueError, match="truncation=True requires"):
        TokenizeText(
            tokenizer_id="__fallback__",
            field="text",
            truncation=True,
            max_length=None,
            special_tokens="bos_eos",
        )


def test_tokenizer_default_truncation_without_max_length_allowed() -> None:
    """``tokenizer_default`` delegates truncation to HF (which uses the
    tokenizer's ``model_max_length`` fallback). No bracketing happens here,
    so no reservation is needed."""
    # Should construct + setup without raising.
    op = TokenizeText(
        tokenizer_id="__fallback__",
        field="text",
        truncation=True,
        max_length=None,
        special_tokens="tokenizer_default",
    )
    _setup(op)
    assert op._cached_kwargs["truncation"] is True


# ---------------------------------------------------------------------------
# ``SpecialTokensMode`` is the public Literal alias for the five modes.
# ---------------------------------------------------------------------------


def test_special_tokens_mode_values() -> None:
    """The Literal's parameter list pins the supported modes. Reordering or
    renaming a mode here is a behavior change and must break this test."""
    import typing as _typing

    assert set(_typing.get_args(SpecialTokensMode)) == {
        "bos_eos",
        "bos",
        "eos",
        "none",
        "tokenizer_default",
    }
