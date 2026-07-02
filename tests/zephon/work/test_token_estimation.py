# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for zephon.work.token_estimation (cost model + estimation + priming)."""

import json
import pickle
from pathlib import Path

import cloudpickle
import numpy as np
import pytest

from zephon.io import Dataset, InMemoryShard
from zephon.io.stores.multi import build_multi_dataset_store
from zephon.observability.size_estimator import content_bytes
from zephon.ops.tokenize_text import TokenizeText
from zephon.utils.tokenizer import fallback_tokenizer
from zephon.work import token_estimation
from zephon.work.token_estimation import (
    DEFAULT_FALLBACK_TOKENS_PER_BYTE,
    PerShardByteSize,
    PerShardTokenCost,
    TokenEstimation,
    TokenizeProfile,
    TokenRatio,
    _calibration_shard_count,
    _CalibrationScan,
    _count_raw_tokens,
    _dataset_content_key,
    _DatasetMeasurement,
    _fallback,
    _fetch_payloads,
    _hansen_hurwitz_ratio,
    _instantiate_tokenizer,
    _measure_dataset,
    _plan_delivery,
    _pps_select,
    _pretokenized_field,
    _prime_cache_key,
    _read_prime_cache,
    _shard_bytes_cv,
    _token_array_length,
    _write_prime_cache,
    build_byte_source,
    choose_text_field,
    extract_text,
    prime_token_ratios,
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


def test_token_estimation_std_pickles_with_lambda_measure():
    est = TokenEstimation(measure=lambda p: len(p["text"]))
    restored = pickle.loads(pickle.dumps(est))
    assert restored.measure is not None
    assert restored.measure({"text": "abc"}) == 3
    # cloudpickle transport (census inputs) must keep working too.
    doubled = cloudpickle.loads(cloudpickle.dumps(est))
    assert doubled.measure is not None
    assert doubled.measure({"text": "abcd"}) == 4


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


# ---------------------------------------------------------------------------
# Priming: sampling / estimator primitives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cv,expected",
    [(0.0, 4), (0.03, 4), (0.06, 4), (0.09, 9), (0.3, 16)],
)
def test_calibration_shard_count_scales_with_spread(cv, expected):
    assert _calibration_shard_count(cv, 4, 16) == expected


def test_shard_bytes_cv_uniform_is_zero():
    assert _shard_bytes_cv(np.array([10, 10, 10]), np.array([100, 100, 100])) == 0.0


def test_shard_bytes_cv_positive_when_heterogeneous():
    assert _shard_bytes_cv(np.array([10, 10]), np.array([100, 900])) > 0


def test_shard_bytes_cv_skips_empty_shards():
    assert _shard_bytes_cv(np.array([0, 0]), np.array([100, 200])) == 0.0


def test_hansen_hurwitz_ratio_is_byte_weighted():
    # (1 draw)*10/100 + (3 draws)*30/100 = 1.0, over 4 draws -> 0.25.
    assert _hansen_hurwitz_ratio([(10, 100, 1), (30, 100, 3)]) == pytest.approx(0.25)


def test_hansen_hurwitz_ratio_single_record():
    assert _hansen_hurwitz_ratio([(42, 84, 7)]) == pytest.approx(0.5)


def test_pps_select_is_independent_of_candidate_dict_order():
    # scan.sizes is merged in nondeterministic census-completion order; the seeded
    # draw must select the same records regardless of that ordering.
    sizes = {(s, o): s * 10 + o + 1 for s in range(4) for o in range(50)}
    forward = dict(sorted(sizes.items()))
    reverse = dict(sorted(sizes.items(), reverse=True))
    assert _pps_select(0, forward, 200, seed=7) == _pps_select(0, reverse, 200, seed=7)


def test_pps_select_weights_by_size():
    sizes = {(0, 0): 1000, (0, 1): 10, (0, 2): 10}
    ids, counts = _pps_select(0, sizes, 1000, seed=1)
    by_offset = {sid[2]: c for sid, c in zip(ids, counts)}
    assert by_offset[0] > by_offset[1] + by_offset[2]


def test_pps_select_draws_with_replacement_conserve_count():
    sizes = {(0, 0): 5, (0, 1): 5}
    ids, counts = _pps_select(0, sizes, 100, seed=2)
    assert sum(counts) == 100
    assert len(ids) <= 2


def test_pps_select_empty_pool():
    assert _pps_select(0, {}, 10, seed=0) == ([], [])
    assert _pps_select(0, {(0, 0): 0}, 10, seed=0) == ([], [])


def test_pps_select_returns_records_in_fetch_order():
    sizes = {(2, 0): 3, (0, 5): 3, (1, 1): 3}
    ids, _ = _pps_select(0, sizes, 300, seed=3)
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Priming: delivery planning + fallback
# ---------------------------------------------------------------------------


def test_plan_delivery_prefers_measure_callable():
    est = TokenEstimation(measure=lambda p: 1)
    assert _plan_delivery(est, None, [{"text": "hi"}]).mode == "measure callable"


def test_plan_delivery_detects_pretokenized():
    plan = _plan_delivery(TokenEstimation(), None, [{"input_ids": np.arange(5)}])
    assert plan.mode == "pretokenized"
    assert plan.path == ("input_ids",)


def test_plan_delivery_falls_through_to_text():
    plan = _plan_delivery(TokenEstimation(), None, [{"body": "hello world"}])
    assert plan.mode == "text"
    assert plan.path == ("body",)


def test_plan_delivery_uses_configured_field_path():
    plan = _plan_delivery(TokenEstimation(), ("doc", "text"), [{"doc": {"text": "x"}}])
    assert plan.mode == "text"
    assert plan.path == ("doc", "text")


def test_plan_delivery_prefix_outvotes_anomalous_first_payload():
    payloads = [{"url": "https://example.com/a", "id": "0"}] + [
        {"url": "https://example.com/b", "text": "real document body"}
    ] * 7
    plan = _plan_delivery(TokenEstimation(), None, payloads)
    assert plan.mode == "text"
    assert plan.path == ("text",)


def test_plan_delivery_prefix_detects_pretokenized_past_bad_record():
    payloads = [{"input_ids": None}, {"input_ids": np.arange(4)}]
    plan = _plan_delivery(TokenEstimation(), None, payloads)
    assert plan.mode == "pretokenized"
    assert plan.path == ("input_ids",)


def test_plan_delivery_prefix_outvotes_anomalous_pretokenized_record():
    payloads = [{"text": "x", "input_ids": np.arange(3)}] + [
        {"text": "real document body"}
    ] * 7
    plan = _plan_delivery(TokenEstimation(), None, payloads)
    assert plan.mode == "text"
    assert plan.path == ("text",)


def test_plan_delivery_majority_field_beats_minority_priority_key():
    payloads = [{"text": "t"}] + [{"content": "long body"}] * 7
    plan = _plan_delivery(TokenEstimation(), None, payloads)
    assert plan.mode == "text"
    assert plan.path == ("content",)


def test_plan_delivery_configured_field_mode_follows_majority():
    payloads = [{"f": np.arange(4)}] + [{"f": "words in the field"}] * 7
    plan = _plan_delivery(TokenEstimation(), ("f",), payloads)
    assert plan.mode == "text"
    assert plan.path == ("f",)


def test_fallback_without_scan_uses_raw_constant():
    m = _fallback(TokenEstimation(fallback_tokens_per_byte=0.3), None, "boom")
    assert m.ratio.tokens_per_byte == pytest.approx(0.3)
    assert m.ratio.source == "fallback"
    assert m.reason == "boom"
    assert m.retryable is False


def test_fallback_with_scan_rebases_onto_raw_bytes():
    scan = _CalibrationScan(sizes={}, payload_total=200.0, raw_total=100.0)
    m = _fallback(
        TokenEstimation(fallback_tokens_per_byte=0.3), scan, "io", retryable=True
    )
    assert m.ratio.tokens_per_byte == pytest.approx(0.6)
    assert m.retryable is True


# ---------------------------------------------------------------------------
# Priming: single-flight cache
# ---------------------------------------------------------------------------


def test_dataset_content_key_stable_and_data_sensitive():
    a = Dataset.from_dict("a", {0: InMemoryShard([{"text": "x"}] * 3)})
    a_again = Dataset.from_dict("a", {0: InMemoryShard([{"text": "x"}] * 3)})
    bigger = Dataset.from_dict("a", {0: InMemoryShard([{"text": "x"}] * 4)})
    assert _dataset_content_key(a) == _dataset_content_key(a_again)
    assert _dataset_content_key(a) != _dataset_content_key(bigger)


def test_prime_cache_key_stable_and_input_sensitive():
    by_name = {"a": Dataset.from_dict("a", {0: InMemoryShard([{"text": "x"}] * 3)})}
    est = TokenEstimation()
    key = _prime_cache_key(["a"], by_name, est, None, seed=0)
    assert key == _prime_cache_key(["a"], by_name, est, None, seed=0)
    assert key != _prime_cache_key(["a"], by_name, est, None, seed=1)
    assert key != _prime_cache_key(
        ["a"], by_name, TokenEstimation(calibration_samples=999), None, seed=0
    )
    assert key != _prime_cache_key(
        ["a"], by_name, est, TokenizeProfile(max_length=128), seed=0
    )


def test_prime_cache_round_trip(tmp_path):
    path = tmp_path / "ratios.json"
    measured = {
        "a": _DatasetMeasurement(TokenRatio(0.3, "measured")),
        "b": _DatasetMeasurement(TokenRatio(0.25, "fallback"), reason="no text"),
    }
    _write_prime_cache(path, "KEY", measured)
    got = _read_prime_cache(path, "KEY")
    assert got is not None
    assert got["a"].ratio == TokenRatio(0.3, "measured")
    assert got["b"].ratio == TokenRatio(0.25, "fallback")
    assert got["b"].reason == "no text"


def test_prime_cache_miss_on_absent_file(tmp_path):
    assert _read_prime_cache(tmp_path / "nope.json", "KEY") is None


def test_prime_cache_miss_on_key_mismatch(tmp_path):
    path = tmp_path / "r.json"
    _write_prime_cache(
        path, "KEY", {"a": _DatasetMeasurement(TokenRatio(0.3, "measured"))}
    )
    assert _read_prime_cache(path, "OTHER") is None


def test_prime_cache_miss_on_corruption(tmp_path):
    path = tmp_path / "r.json"
    path.write_text("{ not json")
    assert _read_prime_cache(path, "KEY") is None


# ---------------------------------------------------------------------------
# Priming: measuring one dataset (no process pool)
# ---------------------------------------------------------------------------


def _uniform_dataset(name: str, row: dict, per_shard: int = 40, n_shards: int = 2):
    return Dataset.from_dict(
        name, {s: InMemoryShard([row] * per_shard) for s in range(n_shards)}
    )


def test_fetch_payloads_keys_payloads_by_sample_id():
    ds = Dataset.from_dict(
        "d",
        {
            0: InMemoryShard([{"i": i} for i in range(10)]),
            1: InMemoryShard([{"i": 100 + i} for i in range(10)]),
        },
    )
    store = build_multi_dataset_store({0: ds})
    # Deliberately unsorted: pairing must not depend on caller ordering.
    sample_ids = [(0, 1, 7), (0, 0, 5), (0, 0, 2), (0, 1, 1)]
    payloads = _fetch_payloads(store, 0, sample_ids)
    assert {sid: p["i"] for sid, p in payloads.items()} == {
        (0, 0, 2): 2,
        (0, 0, 5): 5,
        (0, 1, 1): 101,
        (0, 1, 7): 107,
    }


def test_measure_dataset_measure_callable_matches_byte_weighted_ratio():
    row = {"text": "a b c d"}
    ds = _uniform_dataset("d", row)
    store = build_multi_dataset_store({0: ds})
    est = TokenEstimation(
        measure=lambda p: len(p["text"].split()), calibration_samples=200
    )
    m = _measure_dataset(ds, 0, store, est, None, None, seed=1)
    assert m.ratio.source == "measured"
    assert m.ratio.tokens_per_byte == pytest.approx(4 / content_bytes(row))


def test_measure_dataset_text_path_tokenizes():
    row = {"text": "one two three four five"}
    ds = _uniform_dataset("d", row)
    store = build_multi_dataset_store({0: ds})
    profile = TokenizeProfile(special_tokens="none")
    m = _measure_dataset(
        ds,
        0,
        store,
        TokenEstimation(calibration_samples=100),
        profile,
        fallback_tokenizer(),
        seed=1,
    )
    assert m.ratio.source == "measured"
    assert m.ratio.tokens_per_byte == pytest.approx(5 / content_bytes(row))


def test_measure_dataset_pretokenized_counts_exactly():
    row = {"input_ids": np.arange(8, dtype=np.int64)}
    ds = _uniform_dataset("d", row, per_shard=30)
    store = build_multi_dataset_store({0: ds})
    m = _measure_dataset(
        ds, 0, store, TokenEstimation(calibration_samples=100), None, None, seed=1
    )
    assert m.ratio.source == "measured"
    assert m.ratio.tokens_per_byte == pytest.approx(8 / content_bytes(row))


def test_measure_dataset_falls_back_when_unmeasurable():
    ds = _uniform_dataset("d", {"score": 1.0, "n": 3}, per_shard=20)
    store = build_multi_dataset_store({0: ds})
    est = TokenEstimation(calibration_samples=50, fallback_tokens_per_byte=0.3)
    m = _measure_dataset(
        ds, 0, store, est, TokenizeProfile(), fallback_tokenizer(), seed=1
    )
    assert m.ratio.source == "fallback"
    assert m.reason is not None
    assert m.retryable is False


def test_measure_dataset_measure_callable_zero_counts_names_plan():
    ds = _uniform_dataset("d", {"text": "a b c d"})
    store = build_multi_dataset_store({0: ds})
    est = TokenEstimation(measure=lambda p: 0, calibration_samples=50)
    m = _measure_dataset(ds, 0, store, est, None, None, seed=1)
    assert m.ratio.source == "fallback"
    assert m.reason is not None
    assert "measure callable" in m.reason
    # No false "text" diagnosis when the callable owns counting.
    assert "field" not in m.reason


def test_measure_dataset_falls_back_when_plan_coverage_is_low():
    # The configured field covers only a sliver of the PPS draw mass.
    rows = [{"text": "five words of real text"}] + [{"image": b"\xff" * 200}] * 3
    ds = Dataset.from_dict("d", {s: InMemoryShard(rows * 10) for s in range(2)})
    store = build_multi_dataset_store({0: ds})
    est = TokenEstimation(calibration_samples=100, fallback_tokens_per_byte=0.3)
    m = _measure_dataset(
        ds,
        0,
        store,
        est,
        TokenizeProfile(field="text", special_tokens="none"),
        fallback_tokenizer(),
        seed=1,
    )
    assert m.ratio.source == "fallback"
    assert m.reason is not None
    assert "measured only" in m.reason
    assert "(text, field: text)" in m.reason
    assert m.retryable is False


class _PoisonTokenizer:
    """Raises on documents containing 'poison'; whitespace-counts otherwise."""

    def __init__(self):
        self.raised = 0

    def __call__(self, text, **kwargs):
        if "poison" in text:
            self.raised += 1
            raise RuntimeError("tokenizer exploded")
        return {"input_ids": text.split()}


def test_measure_dataset_skips_docs_that_crash_the_tokenizer():
    good = {"text": "one two three four five"}
    bad = {"text": "poison"}
    ds = Dataset.from_dict("d", {s: InMemoryShard([good, bad] * 20) for s in range(2)})
    store = build_multi_dataset_store({0: ds})
    tok = _PoisonTokenizer()
    m = _measure_dataset(
        ds,
        0,
        store,
        TokenEstimation(calibration_samples=100),
        TokenizeProfile(special_tokens="none"),
        tok,
        seed=1,
    )
    assert tok.raised > 0
    assert m.ratio.source == "measured"
    assert m.ratio.tokens_per_byte == pytest.approx(5 / content_bytes(good))


def test_measure_dataset_falls_back_when_tokenizer_always_crashes():
    ds = _uniform_dataset("d", {"text": "poison"}, per_shard=20)
    store = build_multi_dataset_store({0: ds})
    est = TokenEstimation(calibration_samples=50, fallback_tokens_per_byte=0.3)
    m = _measure_dataset(
        ds,
        0,
        store,
        est,
        TokenizeProfile(special_tokens="none"),
        _PoisonTokenizer(),
        seed=1,
    )
    assert m.ratio.source == "fallback"
    assert m.reason is not None
    assert "tokenizer failed" in m.reason
    assert "tokenizer exploded" in m.reason
    assert m.retryable is False


# ---------------------------------------------------------------------------
# prime_token_ratios orchestration
# ---------------------------------------------------------------------------


def test_prime_token_ratios_float_primer_pins_all_without_io():
    ratios = prime_token_ratios(
        datasets=[],
        dataset_ids={"a": 0, "b": 1},
        estimation=TokenEstimation(primer=0.3),
        tokenize_profile=None,
    )
    assert set(ratios) == {"a", "b"}
    assert all(
        r.source == "pinned" and r.tokens_per_byte == pytest.approx(0.3)
        for r in ratios.values()
    )


def test_prime_token_ratios_rejects_unknown_pins():
    with pytest.raises(ValueError, match="unknown datasets"):
        prime_token_ratios(
            datasets=[],
            dataset_ids={"a": 0},
            estimation=TokenEstimation(primer={"ghost": 0.5}),
            tokenize_profile=None,
        )


def test_prime_token_ratios_missing_descriptor_falls_back():
    with pytest.warns(RuntimeWarning, match="fell back"):
        ratios = prime_token_ratios(
            datasets=[],
            dataset_ids={"ghost": 0},
            estimation=TokenEstimation(primer="measure"),
            tokenize_profile=None,
        )
    assert ratios["ghost"].source == "fallback"


def test_prime_token_ratios_caches_census_result(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEPHON_CATALOG_DIR", str(tmp_path))
    ds = Dataset.from_dict("b", {0: InMemoryShard([{"text": "x"}] * 4)})
    calls: list[int] = []

    def fake_census(*args, **kwargs):
        calls.append(1)
        return {"b": _DatasetMeasurement(TokenRatio(0.4, "measured"))}

    monkeypatch.setattr(token_estimation, "_run_census", fake_census)
    kwargs = dict(
        datasets=[ds],
        dataset_ids={"b": 0},
        estimation=TokenEstimation(primer="measure"),
        tokenize_profile=None,
    )
    first = prime_token_ratios(**kwargs)
    second = prime_token_ratios(**kwargs)
    assert first["b"].tokens_per_byte == pytest.approx(0.4)
    assert second["b"] == first["b"]
    assert len(calls) == 1


def test_prime_token_ratios_does_not_cache_transient_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEPHON_CATALOG_DIR", str(tmp_path))
    ds = Dataset.from_dict("b", {0: InMemoryShard([{"text": "x"}] * 4)})
    calls: list[int] = []

    def fake_census(*args, **kwargs):
        calls.append(1)
        return {
            "b": _DatasetMeasurement(
                TokenRatio(0.25, "fallback"),
                reason="calibration fetch failed: boom",
                retryable=True,
            )
        }

    monkeypatch.setattr(token_estimation, "_run_census", fake_census)
    kwargs = dict(
        datasets=[ds],
        dataset_ids={"b": 0},
        estimation=TokenEstimation(primer="measure"),
        tokenize_profile=None,
    )
    with pytest.warns(RuntimeWarning):
        prime_token_ratios(**kwargs)
    with pytest.warns(RuntimeWarning):
        prime_token_ratios(**kwargs)
    assert len(calls) == 2


def test_prime_token_ratios_end_to_end_measure_callable(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEPHON_CATALOG_DIR", str(tmp_path / "cat"))
    root = tmp_path / "web"
    root.mkdir()
    for s in range(2):
        lines = [json.dumps({"text": "a b c d", "id": f"{s}-{i}"}) for i in range(30)]
        (root / f"shard_{s:05d}.jsonl").write_text("\n".join(lines) + "\n")
    ds = Dataset.from_path("web", str(root))
    est = TokenEstimation(
        measure=lambda p: len(p["text"].split()),
        calibration_samples=50,
        calibration_shards_min=1,
        calibration_shards_max=2,
    )
    ratios = prime_token_ratios(
        datasets=[ds],
        dataset_ids={"web": 0},
        estimation=est,
        tokenize_profile=None,
        seed=1,
    )
    assert ratios["web"].source == "measured"
    assert ratios["web"].tokens_per_byte > 0


class _ClosureTokenizer:
    """Whitespace tokenizer whose lambda attribute defeats standard pickle."""

    def __init__(self):
        self.split = lambda text: text.split()

    def __call__(self, text, **kwargs):
        return {"input_ids": self.split(text)}


def test_prime_token_ratios_live_tokenizer_survives_spawn(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEPHON_CATALOG_DIR", str(tmp_path / "cat"))
    root = tmp_path / "web"
    root.mkdir()
    for s in range(2):
        lines = [json.dumps({"text": "a b c d", "id": f"{s}-{i}"}) for i in range(30)]
        (root / f"shard_{s:05d}.jsonl").write_text("\n".join(lines) + "\n")
    ds = Dataset.from_path("web", str(root))
    tok = _ClosureTokenizer()
    # The test is only meaningful while std pickle rejects this tokenizer.
    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(tok)
    ratios = prime_token_ratios(
        datasets=[ds],
        dataset_ids={"web": 0},
        estimation=TokenEstimation(
            calibration_samples=50,
            calibration_shards_min=1,
            calibration_shards_max=2,
        ),
        tokenize_profile=TokenizeProfile(tokenizer=tok, special_tokens="none"),
        seed=1,
    )
    assert ratios["web"].source == "measured"
    assert ratios["web"].tokens_per_byte > 0


class _CountingTokenizer:
    """Live tokenizer whose counting behavior is instance state."""

    def __init__(self, factor: int):
        self.factor = factor

    def __call__(self, text, **kwargs):
        return {"input_ids": text.split() * self.factor}


class _UnpicklableTokenizer:
    """Simulates a live tokenizer that cannot leave the driver process."""

    def __getstate__(self):
        raise TypeError("cannot pickle live tokenizer")

    def __call__(self, text, **kwargs):
        return {"input_ids": text.split()}


def test_prime_cache_key_distinguishes_live_tokenizer_state():
    ds = Dataset.from_dict("b", {0: InMemoryShard([{"text": "x"}] * 4)})
    est = TokenEstimation()

    def key(profile):
        return _prime_cache_key(["b"], {"b": ds}, est, profile, seed=0)

    k1 = key(TokenizeProfile(tokenizer=_CountingTokenizer(1)))
    k3 = key(TokenizeProfile(tokenizer=_CountingTokenizer(3)))
    assert k1 != k3
    assert k1 == key(TokenizeProfile(tokenizer=_CountingTokenizer(1)))
    # An unpicklable live tokenizer must not alias the no-tokenizer key.
    assert key(TokenizeProfile(tokenizer=_UnpicklableTokenizer())) != key(
        TokenizeProfile()
    )


def test_prime_token_ratios_unpicklable_tokenizer_degrades_to_fallback(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ZEPHON_CATALOG_DIR", str(tmp_path / "cat"))
    root = tmp_path / "web"
    root.mkdir()
    lines = [json.dumps({"text": "a b c d", "id": str(i)}) for i in range(30)]
    (root / "shard_00000.jsonl").write_text("\n".join(lines) + "\n")
    ds = Dataset.from_path("web", str(root))
    # Defeats even the profile's cloudpickle hook: worker spawn fails in the
    # parent and the pool-level guard degrades every dataset to a fallback.
    with pytest.warns(RuntimeWarning, match="census pool failed"):
        ratios = prime_token_ratios(
            datasets=[ds],
            dataset_ids={"web": 0},
            estimation=TokenEstimation(
                calibration_samples=10,
                calibration_shards_min=1,
                calibration_shards_max=1,
            ),
            tokenize_profile=TokenizeProfile(
                tokenizer=_UnpicklableTokenizer(), special_tokens="none"
            ),
            seed=1,
        )
    assert ratios["web"].source == "fallback"
    # A crash is retryable: nothing cached, a later prime retries.
    assert not list((tmp_path / "cat").rglob("token_ratios/*.json"))


def test_tokenize_profile_std_pickles_with_closure_tokenizer():
    profile = TokenizeProfile(tokenizer=_ClosureTokenizer(), special_tokens="none")
    restored = pickle.loads(pickle.dumps(profile))
    assert restored.tokenizer is not None
    assert restored.tokenizer("a b c")["input_ids"] == ["a", "b", "c"]


class _ExplodesOnUnpickle:
    """Simulates a tokenizer that cannot be rebuilt inside pool workers."""

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        raise RuntimeError("cannot rebuild in worker")

    def __call__(self, text, **kwargs):
        return {"input_ids": text.split()}


def test_prime_token_ratios_broken_pool_degrades_to_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEPHON_CATALOG_DIR", str(tmp_path / "cat"))
    root = tmp_path / "web"
    root.mkdir()
    lines = [json.dumps({"text": "a b c d", "id": str(i)}) for i in range(30)]
    (root / "shard_00000.jsonl").write_text("\n".join(lines) + "\n")
    ds = Dataset.from_path("web", str(root))
    with pytest.warns(RuntimeWarning, match="measurement crashed"):
        ratios = prime_token_ratios(
            datasets=[ds],
            dataset_ids={"web": 0},
            estimation=TokenEstimation(
                calibration_samples=10,
                calibration_shards_min=1,
                calibration_shards_max=1,
            ),
            tokenize_profile=TokenizeProfile(
                tokenizer=_ExplodesOnUnpickle(), special_tokens="none"
            ),
            seed=1,
        )
    assert ratios["web"].source == "fallback"
    # A crash is a transient fallback: nothing cached, a later prime retries.
    assert not list((tmp_path / "cat").rglob("token_ratios/*.json"))
