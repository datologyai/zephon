# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
import types
from typing import Any

import pytest

from zephon.core.constants import SampleMeta, SampleRecord
from zephon.core.op_base import OpContext
from zephon.ops.tokenize_text import TokenizeText
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
    op = _setup(TokenizeText(tokenizer=None, tokenizer_id="__fallback__"))
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
        TokenizeText(tokenizer=None, tokenizer_id="__fallback__", field="title")
    )
    r = _rec("ignored", field="text")  # text present, but tokenizer uses "title"
    out = op.process_one(r)[0]
    # When field is missing, fallback tokenizer sees empty string
    payload = _payload_dict(out)
    assert payload.get("input_ids", []) == []
    assert payload.get("attention_mask", []) == []


def test_tokenize_disable_attention_mask() -> None:
    op = TokenizeText(
        tokenizer=None, tokenizer_id="__fallback__", add_attention_mask=False
    )
    op = _setup(op)
    out = op.process_one(_rec("hi there"))[0]
    payload = _payload_dict(out)
    assert "input_ids" in payload
    assert "attention_mask" not in payload


def test_tokenize_replaces_payload_by_default() -> None:
    op = _setup(TokenizeText(tokenizer=None, tokenizer_id="__fallback__"))
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
            tokenizer=None, tokenizer_id="__fallback__", preserve_upstream_payload=True
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
    op = _setup(TokenizeText(tokenizer=tok))
    out = op.process_many([_rec("x"), _rec("y")])
    assert _payload_dict(out[0])["input_ids"] == [1, 2, 3]
    assert op.resolved_tokenizer_id() == "toy-tokenizer"


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
            padding=True,
            truncation=True,
            max_length=4,
            return_tensors="pt",
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
            padding=True,
            truncation=True,
            max_length=2,
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
            split_long_samples=True,
            truncation=True,
            max_length=8,
        )
    with pytest.raises(ValueError):
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            split_long_samples=True,
            max_length=None,
        )


def test_split_long_samples_fanout_lineage_and_contributors() -> None:
    op = _setup(
        TokenizeText(
            tokenizer=None,
            tokenizer_id="__fallback__",
            split_long_samples=True,
            max_length=2,
            padding=False,
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
            split_long_samples=True,
            max_length=3,
            padding=True,
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
            split_long_samples=True,
            max_length=2,
            padding=False,
            return_tensors="pt",
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
            split_long_samples=True,
            max_length=2,
            padding=False,
            return_tensors="np",
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
            split_long_samples=True,
            max_length=3,
            padding=False,
            return_tensors="pt",
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

    op = TokenizeText(tokenizer=None, tokenizer_id="some-model")
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

    op = _setup(TokenizeText(tokenizer=None, tokenizer_id="hf-model", use_fast=False))
    _ = op.process_one(_rec("hi"))
    assert calls["name"] == "hf-model"
    assert calls["kwargs"]["use_fast"] is False


def test_tokenizer_initializes_on_first_process() -> None:
    op = TokenizeText(tokenizer=None, tokenizer_id=None)
    out = op.process_one(_rec("text"))
    payload = _payload_dict(out[0])
    assert "input_ids" in payload


def test_tokenizer_traits_and_default_accumulator() -> None:
    op = _setup(
        TokenizeText(tokenizer_id="__fallback__", max_batch=48, max_latency_ms=15)
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
    op = _setup(TokenizeText(tokenizer_id="__fallback__", split_long_samples=False))

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
    op = TokenizeText(tokenizer_id="__fallback__", padding=False)
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
    op = TokenizeText(tokenizer_id="flaky-model", use_fast=True)
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

    op = TokenizeText(tokenizer_id="broken-model")
    with pytest.raises(TypeError, match="Something else completely broken"):
        _ = _setup(op).process_one(_rec("hello"))


# --- Batch Normalization & Edge Cases ---


def test_normalize_batch_logic() -> None:
    """
    Directly tests _normalize_batch logic used for backend compatibility.
    """
    op = TokenizeText(tokenizer_id="__fallback__")

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
    op = _setup(TokenizeText(tokenizer_id="__fallback__"))
    assert op.process_many([]) == []


def test_empty_string_input() -> None:
    """Ensure empty strings are handled gracefully by fallback."""
    op = _setup(TokenizeText(tokenizer_id="__fallback__"))
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
            split_long_samples=True,
            max_length=2,
            padding=False,
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
            split_long_samples=True,
            max_length=3,
            padding=True,
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

        op = TokenizeText(tokenizer_id="__fallback__", return_tensors="pt")

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

        op = TokenizeText(tokenizer_id="__fallback__", return_tensors="pt")

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

        op = TokenizeText(tokenizer_id="__fallback__", return_tensors="np")

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
