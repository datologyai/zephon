# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for zephon.work.token_estimation (cost model + estimation + priming)."""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cloudpickle
import numpy as np
import pytest

from zephon._internal.io.stores.multi import build_multi_dataset_store
from zephon._internal.observability.size_estimator import content_bytes
from zephon._internal.ops.map_transform import MapTransform
from zephon._internal.ops.tokenize_chat import ChatTokenCountingSpec
from zephon._internal.token_counting import (
    CountPlan,
    DeliveredTokenCounter,
    FatalCountError,
    TextTokenCountingSpec,
    _TextCounter,
)
from zephon._internal.utils.tokenizer import fallback_tokenizer
from zephon.io import Dataset, InMemoryShard
from zephon.work import token_estimation
from zephon.work.token_estimation import (
    DEFAULT_FALLBACK_TOKENS_PER_BYTE,
    PerShardTokenCost,
    TokenEstimation,
    TokenRatio,
    _build_byte_source,
    _calibration_shard_count,
    _CalibrationScan,
    _dataset_content_key,
    _DatasetMeasurement,
    _fallback,
    _fetch_payloads,
    _hansen_hurwitz_ratio,
    _measure_dataset,
    _MeasurePlan,
    _PerShardByteSize,
    _pps_select,
    _PreTokenizeReplay,
    _prime_cache_key,
    _read_prime_cache,
    _shard_bytes_cv,
    _UnreplayableOp,
    _write_prime_cache,
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
# Byte sources
# ---------------------------------------------------------------------------


def test_inmem_byte_source_per_shard_average():
    short, long = {"text": "aa"}, {"text": "a" * 20}
    ds = Dataset.from_dict(
        "mixed", {0: InMemoryShard([short] * 10), 1: InMemoryShard([long] * 10)}
    )
    source = _build_byte_source(ds)
    assert source is not None
    assert source.shard_avg_bytes[0] == pytest.approx(content_bytes(short))
    assert source.shard_avg_bytes[1] == pytest.approx(content_bytes(long))
    assert source.shard_avg_bytes[1] > source.shard_avg_bytes[0]
    assert source.mean_bytes == pytest.approx(
        (content_bytes(short) + content_bytes(long)) / 2
    )


def test_catalog_byte_source_from_jsonl(tmp_path):
    ds = make_jsonl_dataset(tmp_path, "web", docs_per_shard=20, words_per_doc=10)
    source = _build_byte_source(ds)
    assert source is not None
    sizes = [source.shard_avg_bytes[int(sid)] for sid in ds.ids()]
    assert all(s > 0 for s in sizes)
    files = sorted(Path(ds.path).glob("*.jsonl"))
    assert sizes == pytest.approx([f.stat().st_size / 20 for f in files])


def test_byte_source_handles_empty_dataset_gracefully():
    empty = Dataset.from_dict("empty", {0: InMemoryShard([])})
    assert _build_byte_source(empty) is None


def test_byte_source_direct_construction_floors_mean():
    source = _PerShardByteSize({5: 100.0}, mean_bytes=0.0)
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


def test_measure_worker_initializes_litdata_before_measurement(monkeypatch) -> None:
    from zephon._internal.io.formats import litdata_support

    events: list[str] = []
    measurement = object()
    monkeypatch.setattr(
        litdata_support,
        "dependencies",
        SimpleNamespace(ensure_litdata_deps=lambda: events.append("initialize")),
        raising=False,
    )
    monkeypatch.setattr(
        token_estimation,
        "_measure_dataset",
        lambda *_args, **_kwargs: events.append("measure") or measurement,
    )
    monkeypatch.setattr(
        token_estimation,
        "_prime_worker",
        {
            "datasets": {0: Dataset("lit", {"kind": "litdata"})},
            "store": None,
            "estimation": None,
            "counter": None,
            "seed": 0,
            "pre_tokenize_replay": None,
        },
    )

    assert token_estimation._measure_in_worker(0) is measurement
    assert events == ["initialize", "measure"]


def test_measure_plan_aborts_dataset_on_error():
    def boom(_payload):
        raise RuntimeError("measure boom")

    ds = _uniform_dataset("d", {"text": "a b c d"}, per_shard=20)
    store = build_multi_dataset_store({0: ds})
    est = TokenEstimation(
        measure=boom, calibration_samples=50, fallback_tokens_per_byte=0.3
    )
    m = _measure_dataset(ds, 0, store, est, None, seed=1)
    # A user measure defines the unit, so one failure invalidates the dataset:
    # abort straight to fallback rather than average a partial ratio.
    assert m.ratio.source == "fallback"
    assert m.reason is not None
    assert "measure callable" in m.reason
    assert "measure boom" in m.reason


def test_measure_plan_treats_nonpositive_counts_as_unmeasurable():
    assert _MeasurePlan(lambda p: 0).count({"x": 1}) is None
    assert _MeasurePlan(lambda p: -3).count({"x": 1}) is None


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
        ["a"], by_name, est, TextTokenCountingSpec(max_length=128), seed=0
    )
    assert key != _prime_cache_key(
        ["a"],
        by_name,
        est,
        None,
        seed=0,
        pre_tokenize_replay=_PreTokenizeReplay([MapTransform(_lift_n_words)]),
    )


def test_prime_cache_key_tracks_chat_config():
    dataset = Dataset.from_dict("d", {0: InMemoryShard([{"text": "x"}])})
    estimation = TokenEstimation()

    def key(spec):
        return _prime_cache_key(["d"], {"d": dataset}, estimation, spec, seed=0)

    a = key(ChatTokenCountingSpec(tokenizer_id="t", max_length=8))
    b = key(ChatTokenCountingSpec(tokenizer_id="t", max_length=16))
    c = key(ChatTokenCountingSpec(tokenizer_id="t", max_length=8))
    d = key(TextTokenCountingSpec(tokenizer_id="t", max_length=8))
    assert a != b
    assert a == c
    assert a != d


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
    m = _measure_dataset(ds, 0, store, est, None, seed=1)
    assert m.ratio.source == "measured"
    assert m.ratio.tokens_per_byte == pytest.approx(4 / content_bytes(row))


def test_measure_dataset_text_path_tokenizes():
    row = {"text": "one two three four five"}
    ds = _uniform_dataset("d", row)
    store = build_multi_dataset_store({0: ds})
    spec = TextTokenCountingSpec(special_tokens="none")
    m = _measure_dataset(
        ds,
        0,
        store,
        TokenEstimation(calibration_samples=100),
        _TextCounter(fallback_tokenizer(), spec),
        seed=1,
    )
    assert m.ratio.source == "measured"
    assert m.ratio.tokens_per_byte == pytest.approx(5 / content_bytes(row))


def test_measure_dataset_pretokenized_counts_exactly():
    row = {"input_ids": np.arange(8, dtype=np.int64)}
    ds = _uniform_dataset("d", row, per_shard=30)
    store = build_multi_dataset_store({0: ds})
    # Pretokenized counting never touches the tokenizer slot.
    counter = _TextCounter(None, TextTokenCountingSpec())
    m = _measure_dataset(
        ds, 0, store, TokenEstimation(calibration_samples=100), counter, seed=1
    )
    assert m.ratio.source == "measured"
    assert m.ratio.tokens_per_byte == pytest.approx(8 / content_bytes(row))


def test_measure_dataset_falls_back_when_unmeasurable():
    ds = _uniform_dataset("d", {"score": 1.0, "n": 3}, per_shard=20)
    store = build_multi_dataset_store({0: ds})
    est = TokenEstimation(calibration_samples=50, fallback_tokens_per_byte=0.3)
    m = _measure_dataset(
        ds,
        0,
        store,
        est,
        _TextCounter(fallback_tokenizer(), TextTokenCountingSpec()),
        seed=1,
    )
    assert m.ratio.source == "fallback"
    assert m.reason is not None
    assert m.retryable is False


def test_measure_dataset_measure_callable_zero_counts_names_plan():
    ds = _uniform_dataset("d", {"text": "a b c d"})
    store = build_multi_dataset_store({0: ds})
    est = TokenEstimation(measure=lambda p: 0, calibration_samples=50)
    m = _measure_dataset(ds, 0, store, est, None, seed=1)
    assert m.ratio.source == "fallback"
    assert m.reason is not None
    assert "measure callable" in m.reason
    # No false "text" diagnosis when the callable owns counting.
    assert "field" not in m.reason


class _FieldPlan(CountPlan):
    @property
    def description(self) -> str:
        return "stub field"

    def count(self, payload):
        return payload["tokens"]


class _FieldCounter(DeliveredTokenCounter):
    def plan(self, sample_payloads):
        return _FieldPlan()


def test_measure_dataset_zero_counts_enter_ratio_and_coverage():
    # Zero-yield samples (e.g. chat drops) consume scheduled bytes: they must
    # depress the ratio, not trip the coverage gate.
    rows = [{"pad": "x" * 64, "tokens": 8 * (i % 2)} for i in range(40)]
    ds = Dataset.from_dict(
        "d", {0: InMemoryShard(rows[:20]), 1: InMemoryShard(rows[20:])}
    )
    store = build_multi_dataset_store({0: ds})
    m = _measure_dataset(
        ds, 0, store, TokenEstimation(calibration_samples=200), _FieldCounter(), seed=1
    )
    assert m.ratio.source == "measured"
    assert 0 < m.ratio.tokens_per_byte < 8 / content_bytes(rows[0])


def test_measure_dataset_all_zero_counts_fall_back():
    ds = _uniform_dataset("d", {"pad": "x" * 64, "tokens": 0}, per_shard=20)
    store = build_multi_dataset_store({0: ds})
    m = _measure_dataset(
        ds, 0, store, TokenEstimation(calibration_samples=50), _FieldCounter(), seed=1
    )
    assert m.ratio.source == "fallback"
    assert m.reason is not None
    assert "not positive" in m.reason


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
        _TextCounter(
            fallback_tokenizer(),
            TextTokenCountingSpec(
                field="text", missing_field="empty", special_tokens="none"
            ),
        ),
        seed=1,
    )
    assert m.ratio.source == "fallback"
    assert m.reason is not None
    assert "measured only" in m.reason
    assert "(text, field: text)" in m.reason
    assert m.retryable is False


def test_measure_dataset_missing_required_text_field_is_fatal():
    row = {"input_ids": np.arange(8, dtype=np.int64)}
    ds = _uniform_dataset("d", row, per_shard=20)
    store = build_multi_dataset_store({0: ds})
    counter = _TextCounter(fallback_tokenizer(), TextTokenCountingSpec(field="text"))

    with pytest.raises(FatalCountError, match="no resolvable 'text' field"):
        _measure_dataset(
            ds,
            0,
            store,
            TokenEstimation(calibration_samples=50),
            counter,
            seed=1,
        )


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
        _TextCounter(tok, TextTokenCountingSpec(special_tokens="none")),
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
        _TextCounter(_PoisonTokenizer(), TextTokenCountingSpec(special_tokens="none")),
        seed=1,
    )
    assert m.ratio.source == "fallback"
    assert m.reason is not None
    assert "counting failed" in m.reason
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
        counting_spec=None,
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
            counting_spec=None,
        )


def test_prime_token_ratios_missing_descriptor_falls_back():
    with pytest.warns(RuntimeWarning, match="fell back"):
        ratios = prime_token_ratios(
            datasets=[],
            dataset_ids={"ghost": 0},
            estimation=TokenEstimation(primer="measure"),
            counting_spec=None,
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
        counting_spec=None,
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
        counting_spec=None,
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
        counting_spec=None,
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
        counting_spec=TextTokenCountingSpec(tokenizer=tok, special_tokens="none"),
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

    def key(spec):
        return _prime_cache_key(["b"], {"b": ds}, est, spec, seed=0)

    k1 = key(TextTokenCountingSpec(tokenizer=_CountingTokenizer(1)))
    k3 = key(TextTokenCountingSpec(tokenizer=_CountingTokenizer(3)))
    assert k1 != k3
    assert k1 == key(TextTokenCountingSpec(tokenizer=_CountingTokenizer(1)))
    # An unpicklable live tokenizer must not alias the no-tokenizer key.
    assert key(TextTokenCountingSpec(tokenizer=_UnpicklableTokenizer())) != key(
        TextTokenCountingSpec()
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
            counting_spec=TextTokenCountingSpec(
                tokenizer=_UnpicklableTokenizer(), special_tokens="none"
            ),
            seed=1,
        )
    assert ratios["web"].source == "fallback"
    # A crash is retryable: nothing cached, a later prime retries.
    assert not list((tmp_path / "cat").rglob("token_ratios/*.json"))


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
            counting_spec=TextTokenCountingSpec(
                tokenizer=_ExplodesOnUnpickle(), special_tokens="none"
            ),
            seed=1,
        )
    assert ratios["web"].source == "fallback"
    # A crash is a transient fallback: nothing cached, a later prime retries.
    assert not list((tmp_path / "cat").rglob("token_ratios/*.json"))


# ---------------------------------------------------------------------------
# Pre-tokenize replay
# ---------------------------------------------------------------------------


def _lift_n_words(payload: dict[str, int]) -> dict[str, str]:
    return {"text": " ".join(f"w{i}" for i in range(payload["n"]))}


def _lift_2n_words(payload: dict[str, int]) -> dict[str, str]:
    return {"text": " ".join(f"w{i}" for i in range(2 * payload["n"]))}


def test_pre_tokenize_replay_applies_map_ops() -> None:
    replay = _PreTokenizeReplay([MapTransform(_lift_n_words)])
    assert replay.apply({"n": 3}) == [{"text": "w0 w1 w2"}]


def test_pre_tokenize_replay_preserves_drop_semantics() -> None:
    dropped = _PreTokenizeReplay([MapTransform(lambda p: None, drop_none=True)])
    assert dropped.apply({"n": 1}) == []
    kept = _PreTokenizeReplay([MapTransform(lambda p: None, drop_none=False)])
    assert kept.apply({"n": 1}) == [{"n": 1}]


def test_pre_tokenize_replay_survives_cloudpickle() -> None:
    replay = cloudpickle.loads(
        cloudpickle.dumps(_PreTokenizeReplay([MapTransform(_lift_n_words)]))
    )
    assert replay.apply({"n": 2}) == [{"text": "w0 w1"}]


def _int_rows_dataset(n_rows: int = 40) -> Dataset:
    rows = [{"n": 5} for _ in range(n_rows)]
    return Dataset.from_dict(
        "d",
        {0: InMemoryShard(rows[: n_rows // 2]), 1: InMemoryShard(rows[n_rows // 2 :])},
    )


def _fallback_text_counter() -> DeliveredTokenCounter:
    return TextTokenCountingSpec(
        tokenizer=fallback_tokenizer(), special_tokens="none"
    ).build_counter()


def test_measure_dataset_counts_through_pre_tokenize_replay() -> None:
    ds = _int_rows_dataset()
    store = build_multi_dataset_store({0: ds})
    counter = _fallback_text_counter()
    est = TokenEstimation(calibration_samples=64)

    raw = _measure_dataset(ds, 0, store, est, counter, seed=3)
    assert raw.ratio.source == "fallback"  # int-only rows have nothing to count

    lift = _PreTokenizeReplay([MapTransform(_lift_n_words)])
    measured = _measure_dataset(
        ds, 0, store, est, counter, seed=3, pre_tokenize_replay=lift
    )
    assert measured.ratio.source == "measured"

    double = _PreTokenizeReplay([MapTransform(_lift_2n_words)])
    doubled = _measure_dataset(
        ds, 0, store, est, counter, seed=3, pre_tokenize_replay=double
    )
    assert doubled.ratio.tokens_per_byte == pytest.approx(
        2 * measured.ratio.tokens_per_byte
    )


def _pop_n_words(payload: dict[str, int]) -> dict[str, str]:
    n_words = payload.pop("n")
    return {"text": " ".join(f"w{i}" for i in range(n_words))}


def test_measure_dataset_replays_planning_payloads_once() -> None:
    ds = _int_rows_dataset(n_rows=8)
    store = build_multi_dataset_store({0: ds})
    measured = _measure_dataset(
        ds,
        0,
        store,
        TokenEstimation(calibration_samples=64),
        _fallback_text_counter(),
        seed=3,
        pre_tokenize_replay=_PreTokenizeReplay([MapTransform(_pop_n_words)]),
    )
    assert measured.ratio.source == "measured"


def _drop_odd(payload: dict[str, int]) -> dict[str, str] | None:
    if payload["i"] % 2:
        return None
    return {"text": "a b c d"}


def test_measure_dataset_replay_drops_are_zero_yield() -> None:
    rows = [{"i": i} for i in range(40)]
    ds = Dataset.from_dict(
        "d", {0: InMemoryShard(rows[:20]), 1: InMemoryShard(rows[20:])}
    )
    store = build_multi_dataset_store({0: ds})
    counter = _fallback_text_counter()
    est = TokenEstimation(calibration_samples=256)

    keep_all = _measure_dataset(
        ds,
        0,
        store,
        est,
        counter,
        seed=5,
        pre_tokenize_replay=_PreTokenizeReplay(
            [MapTransform(lambda p: {"text": "a b c d"})]
        ),
    )
    drop_half = _measure_dataset(
        ds,
        0,
        store,
        est,
        counter,
        seed=5,
        pre_tokenize_replay=_PreTokenizeReplay([MapTransform(_drop_odd)]),
    )
    assert keep_all.ratio.source == "measured"
    assert drop_half.ratio.source == "measured"
    assert 0 < drop_half.ratio.tokens_per_byte < 0.7 * keep_all.ratio.tokens_per_byte


def _boom(payload: dict[str, int]) -> None:
    raise RuntimeError("boom")


def test_measure_dataset_replay_errors_fall_back() -> None:
    ds = _int_rows_dataset()
    store = build_multi_dataset_store({0: ds})
    m = _measure_dataset(
        ds,
        0,
        store,
        TokenEstimation(calibration_samples=32),
        _fallback_text_counter(),
        seed=7,
        pre_tokenize_replay=_PreTokenizeReplay([MapTransform(_boom)]),
    )
    assert m.ratio.source == "fallback"
    assert m.reason is not None and "boom" in m.reason


def test_prime_token_ratios_unreplayable_op_raises_when_measuring() -> None:
    ds = make_inmem_dataset("d", 4, 3)
    with pytest.raises(ValueError, match="ShuffleBuffer"):
        prime_token_ratios(
            datasets=[ds],
            dataset_ids={"d": 0},
            estimation=TokenEstimation(),
            counting_spec=None,
            pre_tokenize_replay=_UnreplayableOp("ShuffleBuffer"),
        )


def test_prime_token_ratios_unreplayable_op_ignored_when_pinned() -> None:
    ds = make_inmem_dataset("d", 4, 3)
    ratios = prime_token_ratios(
        datasets=[ds],
        dataset_ids={"d": 0},
        estimation=TokenEstimation(primer={"d": 0.5}),
        counting_spec=None,
        pre_tokenize_replay=_UnreplayableOp("ShuffleBuffer"),
    )
    assert ratios["d"].source == "pinned"


def test_prime_token_ratios_measure_callable_skips_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZEPHON_CATALOG_DIR", str(tmp_path))
    ds = Dataset.from_dict("b", {0: InMemoryShard([{"text": "x"}] * 4)})
    seen: dict[str, Any] = {}

    def fake_census(
        measured_datasets: Any,
        store_options: Any,
        estimation: Any,
        counting_spec: Any,
        seed: int,
        mp_context: Any,
        pre_tokenize_replay: Any = None,
    ) -> dict[str, _DatasetMeasurement]:
        seen["pre_tokenize_replay"] = pre_tokenize_replay
        return {"b": _DatasetMeasurement(TokenRatio(0.4, "measured"))}

    monkeypatch.setattr(token_estimation, "_run_census", fake_census)
    ratios = prime_token_ratios(
        datasets=[ds],
        dataset_ids={"b": 0},
        estimation=TokenEstimation(measure=len),
        counting_spec=None,
        pre_tokenize_replay=_UnreplayableOp("ShuffleBuffer"),
    )
    assert ratios["b"].source == "measured"
    assert seen["pre_tokenize_replay"] is None
