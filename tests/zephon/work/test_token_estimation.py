# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for zephon.work.token_estimation (cost model + estimation primitives)."""

import json
from pathlib import Path

import numpy as np
import pytest

from zephon.io import Dataset, InMemoryShard
from zephon.observability.size_estimator import content_bytes
from zephon.ops.tokenize_text import TokenizeText
from zephon.utils.tokenizer import fallback_tokenizer
from zephon.work.token_estimation import (
    DEFAULT_FALLBACK_TOKENS_PER_BYTE,
    PerShardByteSize,
    PerShardTokenCost,
    TokenEstimation,
    TokenizeProfile,
    TokenRatio,
    _count_raw_tokens,
    _instantiate_tokenizer,
    _pretokenized_field,
    _token_array_length,
    build_byte_source,
    choose_text_field,
    extract_text,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_inmem_dataset(
    name: str, docs_per_shard: int, words_per_doc: int, n_shards: int = 2
) -> Dataset:
    text = " ".join(f"{name}w{i}" for i in range(words_per_doc))
    shards = {
        s: InMemoryShard([{"text": text} for _ in range(docs_per_shard)])
        for s in range(n_shards)
    }
    return Dataset.from_dict(name, shards)


def make_jsonl_dataset(
    tmp_path: Path,
    name: str,
    docs_per_shard: int,
    words_per_doc: int,
    n_shards: int = 2,
) -> Dataset:
    root = tmp_path / name
    root.mkdir()
    text = " ".join(f"{name}w{i}" for i in range(words_per_doc))
    for s in range(n_shards):
        lines = [
            json.dumps({"text": text, "id": f"{name}-{s}-{i}"})
            for i in range(docs_per_shard)
        ]
        (root / f"shard_{s:05d}.jsonl").write_text("\n".join(lines) + "\n")
    return Dataset.from_path(name, str(root))


# ---------------------------------------------------------------------------
# TokenEstimation config validation
# ---------------------------------------------------------------------------


def test_token_estimation_defaults_valid():
    est = TokenEstimation()
    assert est.primer == "measure"
    assert est.calibration_samples == 2048
    assert est.calibration_shards_min == 4
    assert est.calibration_shards_max == 16
    assert est.fallback_tokens_per_byte == DEFAULT_FALLBACK_TOKENS_PER_BYTE


@pytest.mark.parametrize(
    "kwargs",
    [
        {"primer": "magic"},
        {"primer": True},
        {"primer": -0.5},
        {"primer": 0.0},
        {"primer": float("nan")},
        {"primer": {"a": -1.0}},
        {"primer": {"a": True}},
        {"primer": ["a"]},
        {"calibration_samples": 0},
        {"calibration_shards_min": 0},
        {"calibration_shards_min": 8, "calibration_shards_max": 4},
        {"fallback_tokens_per_byte": 0.0},
        {"fallback_tokens_per_byte": float("inf")},
    ],
)
def test_token_estimation_rejects_bad_config(kwargs):
    with pytest.raises(ValueError):
        TokenEstimation(**kwargs)


def test_token_estimation_accepts_partial_pin_mapping():
    est = TokenEstimation(primer={"a": 0.3, "b": 1.5})
    assert est.primer == {"a": 0.3, "b": 1.5}


# ---------------------------------------------------------------------------
# TokenRatio
# ---------------------------------------------------------------------------


def test_token_ratio_state_round_trip():
    ratio = TokenRatio(0.31, "measured")
    assert TokenRatio.from_state(ratio.to_state()) == ratio


@pytest.mark.parametrize("raw", [[0.0, "measured"], [1.0, "magic"], [-1.0, "pinned"]])
def test_token_ratio_from_state_rejects_bad_values(raw):
    with pytest.raises(ValueError):
        TokenRatio.from_state(raw)


@pytest.mark.parametrize(
    "args",
    [(0.0, "measured"), (-1.0, "pinned"), (float("nan"), "measured"), (1.0, "bogus")],
)
def test_token_ratio_construction_validates(args):
    with pytest.raises(ValueError):
        TokenRatio(*args)


# ---------------------------------------------------------------------------
# Delivered-token arithmetic
# ---------------------------------------------------------------------------


def test_delivered_tokens_bracket_modes_add_specials():
    assert TokenizeProfile(special_tokens="bos_eos").delivered_tokens(100) == 102
    assert TokenizeProfile(special_tokens="bos").delivered_tokens(100) == 101
    assert TokenizeProfile(special_tokens="eos").delivered_tokens(100) == 101
    assert TokenizeProfile(special_tokens="none").delivered_tokens(100) == 100


def test_delivered_tokens_truncation_destroys_mass():
    cfg = TokenizeProfile(special_tokens="bos_eos", truncation=True, max_length=50)
    # Specials reserve two slots, so content truncates to 48.
    assert cfg.delivered_tokens(100) == 50
    assert cfg.delivered_tokens(10) == 12


def test_delivered_tokens_split_preserves_mass():
    cfg = TokenizeProfile(
        special_tokens="bos_eos", split_long_samples=True, max_length=50
    )
    assert cfg.delivered_tokens(1000) == 1002


def test_delivered_tokens_tokenizer_default_counts_template():
    cfg = TokenizeProfile(special_tokens="tokenizer_default")
    assert cfg.delivered_tokens(100) == 100
    truncating = TokenizeProfile(
        special_tokens="tokenizer_default", truncation=True, max_length=64
    )
    assert truncating.delivered_tokens(100) == 64


@pytest.mark.parametrize(
    "mode,prepend,append",
    [("bos_eos", 1, 1), ("bos", 1, 0), ("eos", 0, 1), ("none", 0, 0)],
)
def test_num_specials_by_mode(mode, prepend, append):
    assert TokenizeProfile(special_tokens=mode).num_specials == prepend + append


def test_tokenize_profile_from_op_captures_count_fields():
    tok = fallback_tokenizer()
    op = TokenizeText(
        tokenizer=tok,
        field="text",
        max_length=128,
        truncation=True,
        special_tokens="bos",
    )
    profile = TokenizeProfile.from_op(op)
    assert profile.tokenizer is tok
    assert profile.field == "text"
    assert profile.max_length == 128
    assert profile.truncation is True
    assert profile.split_long_samples is False
    assert profile.special_tokens == "bos"


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def test_extract_text_configured_field_wins():
    payload = {"text": "body words", "title": "a much longer title than the body"}
    assert extract_text(payload, ("text",)) == "body words"


def test_extract_text_nested_field_path():
    payload = {"doc": {"inner": {"text": "nested body"}}}
    assert extract_text(payload, ("doc", "inner", "text")) == "nested body"


def test_extract_text_configured_field_decodes_bytes():
    payload = {"text": "raw bytes body".encode("utf-8")}
    assert extract_text(payload, ("text",)) == "raw bytes body"


def test_extract_text_plain_string_payload():
    assert extract_text("just a string document") == "just a string document"


def test_extract_text_utf8_bytes_payload():
    assert extract_text("bytes document".encode("utf-8")) == "bytes document"


def test_extract_text_binary_bytes_unmeasurable():
    assert extract_text(b"\xff\xfe\x00\x01binary") is None


def test_extract_text_missing_configured_field_unmeasurable():
    # Extraction does not silently cross over to a different field.
    assert extract_text({"content": "some body"}, ("text",)) is None


def test_extract_text_non_text_payload_unmeasurable():
    assert extract_text(12345) is None
    assert extract_text([1, 2, 3]) is None
    assert extract_text({"a": 1, "b": 2}) is None


def test_choose_text_field_prefers_common_keys():
    payload = {"id": "abc-123", "url": "https://x", "text": "short"}
    assert choose_text_field(payload) == "text"


def test_choose_text_field_single_candidate():
    payload = {"n_tokens": 7, "body_field": "the only string"}
    assert choose_text_field(payload) == "body_field"


def test_choose_text_field_multiple_candidates_picks_longest():
    payload = {"id": "x" * 8, "document_body": "y" * 100}
    assert choose_text_field(payload) == "document_body"


def test_choose_text_field_decodable_bytes_count_as_text():
    payload = {"blob": "decodable text".encode("utf-8")}
    assert choose_text_field(payload) == "blob"


def test_choose_text_field_no_text_returns_none():
    assert choose_text_field({"a": 1}) is None
    assert choose_text_field("not a mapping") is None


# ---------------------------------------------------------------------------
# Pretokenized data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64, np.uint16])
def test_token_array_length_counts_elements_regardless_of_dtype(dtype):
    assert _token_array_length(np.arange(20, dtype=dtype)) == 20


def test_token_array_length_rejects_non_token_arrays():
    assert _token_array_length(np.arange(5, dtype=np.float32)) is None
    assert _token_array_length(np.array([True, False])) is None
    assert _token_array_length([1, 2, 3]) == 3
    assert _token_array_length([1.0, 2.0]) is None
    assert _token_array_length([True, False]) is None
    assert _token_array_length("a string") is None
    assert _token_array_length(b"\x00\x01") is None
    assert _token_array_length([]) is None


def test_token_array_length_duck_typed_integer_tensor():
    # torch is optional, so the tensor branch is detected by duck typing.
    class _IntTensor:
        def is_floating_point(self):
            return False

        def is_complex(self):
            return False

        def numel(self):
            return 42

    class _FloatTensor:
        def is_floating_point(self):
            return True

        def numel(self):
            return 42

    assert _token_array_length(_IntTensor()) == 42
    assert _token_array_length(_FloatTensor()) is None


def test_pretokenized_field_resolution_order():
    # () means "the payload itself is the token array".
    assert _pretokenized_field(np.arange(8, dtype=np.int32), None) == ()
    assert _pretokenized_field({"input_ids": np.arange(12, dtype=np.int64)}, None) == (
        "input_ids",
    )
    nested = {"enc": {"ids": np.arange(7, dtype=np.int32)}}
    assert _pretokenized_field(nested, ("enc", "ids")) == ("enc", "ids")
    assert _pretokenized_field({"text": "words go here"}, None) is None
    assert _pretokenized_field({"text": "words"}, ("text",)) is None


# ---------------------------------------------------------------------------
# Byte sources
# ---------------------------------------------------------------------------


def test_inmem_byte_source_per_shard_average():
    short, long = {"text": "aa"}, {"text": "a" * 20}
    ds = Dataset.from_dict(
        "mixed", {0: InMemoryShard([short] * 10), 1: InMemoryShard([long] * 10)}
    )
    source = build_byte_source(ds)
    assert source is not None
    assert source.shard_avg_bytes[0] == pytest.approx(content_bytes(short))
    assert source.shard_avg_bytes[1] == pytest.approx(content_bytes(long))
    assert source.shard_avg_bytes[1] > source.shard_avg_bytes[0]
    assert source.mean_bytes == pytest.approx(
        (content_bytes(short) + content_bytes(long)) / 2
    )


def test_catalog_byte_source_from_jsonl(tmp_path):
    ds = make_jsonl_dataset(tmp_path, "web", docs_per_shard=20, words_per_doc=10)
    source = build_byte_source(ds)
    assert source is not None
    sizes = [source.shard_avg_bytes[int(sid)] for sid in ds.ids()]
    assert all(s > 0 for s in sizes)
    files = sorted(Path(ds.path).glob("*.jsonl"))
    assert sizes == pytest.approx([f.stat().st_size / 20 for f in files])


def test_byte_source_handles_empty_dataset_gracefully():
    empty = Dataset.from_dict("empty", {0: InMemoryShard([])})
    assert build_byte_source(empty) is None


def test_byte_source_direct_construction_floors_mean():
    source = PerShardByteSize({5: 100.0}, mean_bytes=0.0)
    assert source.shard_avg_bytes == {5: 100.0}
    assert source.mean_bytes == 1.0


# ---------------------------------------------------------------------------
# PerShardTokenCost
# ---------------------------------------------------------------------------


def test_per_shard_token_cost_floors_and_lookup():
    ds = make_inmem_dataset("a", docs_per_shard=5, words_per_doc=3)
    costs = PerShardTokenCost({"a": ds}, {"a": TokenRatio(1e-9, "pinned")})
    # Tiny ratio floors at 1 token per sample.
    assert costs.cost("a", (0, 0, 0)) == 1.0
    assert costs.mean_cost("a") == 1.0


def test_per_shard_token_cost_varies_by_shard_and_falls_back_to_mean():
    short, long = {"text": "aa"}, {"text": "a" * 40}
    ds = Dataset.from_dict(
        "mixed", {0: InMemoryShard([short] * 8), 1: InMemoryShard([long] * 8)}
    )
    costs = PerShardTokenCost({"mixed": ds}, {"mixed": TokenRatio(1.0, "measured")})
    c0 = costs.cost("mixed", (0, 0, 0))
    c1 = costs.cost("mixed", (0, 1, 0))
    assert c1 > c0
    # An unseen shard id falls back to the dataset mean.
    assert costs.cost("mixed", (0, 99, 0)) == pytest.approx(costs.mean_cost("mixed"))


def test_per_shard_token_cost_warns_without_byte_metadata():
    backendless = Dataset(name="ghost", backend={"kind": "inmem", "shards": {}})
    with pytest.warns(RuntimeWarning, match="no byte metadata"):
        costs = PerShardTokenCost(
            {"ghost": backendless}, {"ghost": TokenRatio(0.5, "pinned")}
        )
    assert costs.cost("ghost", (0, 0, 0)) >= 1.0
    assert costs.mean_cost("ghost") >= 1.0


# ---------------------------------------------------------------------------
# Tokenizer plumbing (calibration)
# ---------------------------------------------------------------------------


def test_instantiate_tokenizer_returns_passed_instance():
    tok = fallback_tokenizer()
    assert _instantiate_tokenizer(TokenizeProfile(tokenizer=tok)) is tok


@pytest.mark.parametrize("tokenizer_id", [None, "__fallback__"])
def test_instantiate_tokenizer_falls_back_without_model(tokenizer_id):
    tok = _instantiate_tokenizer(TokenizeProfile(tokenizer_id=tokenizer_id))
    assert tok.bos_token_id == 1  # the in-process fallback stub


def test_count_raw_tokens_counts_whitespace_words():
    tok = fallback_tokenizer()
    profile = TokenizeProfile(special_tokens="none")
    assert _count_raw_tokens(tok, "one two three four", profile) == 4


def test_count_raw_tokens_passes_add_special_tokens_by_mode():
    class _SpecialsAware:
        def __call__(self, text, add_special_tokens=False):
            n = len(text.split())
            return {"input_ids": list(range(n + (2 if add_special_tokens else 0)))}

    tok = _SpecialsAware()
    # Bracket modes call HF with add_special_tokens=False (raw content only).
    assert (
        _count_raw_tokens(tok, "a b c", TokenizeProfile(special_tokens="bos_eos")) == 3
    )
    # tokenizer_default lets the template add specials, so they land in the count.
    assert (
        _count_raw_tokens(
            tok, "a b c", TokenizeProfile(special_tokens="tokenizer_default")
        )
        == 5
    )


def test_count_raw_tokens_reads_input_ids_attribute():
    class _AttrResult:
        def __init__(self, ids):
            self.input_ids = ids

    class _AttrTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return _AttrResult(list(range(len(text.split()))))

    profile = TokenizeProfile(special_tokens="none")
    assert _count_raw_tokens(_AttrTokenizer(), "a b c d", profile) == 4
