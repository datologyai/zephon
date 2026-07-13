# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end token-aware mixture tests with tokenize and ensure_mixture."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.integration.test_elastic_continuation import consume_until
from zephon.api import Pipeline
from zephon.core.constants import SampleRecord
from zephon.io import Dataset
from zephon.ops.tokenize_chat import TokenizeChat
from zephon.work.static_mixture import StaticMixtureWorkSource
from zephon.work.token_estimation import TextTokenCountingSpec, TokenEstimation

pytestmark = pytest.mark.integration

#: Whitespace-token counts per document; fallback maps one word to one token.
SHORT_WORDS = 12
LONG_WORDS = 48

CHAT_SHORT_WORDS = 4
CHAT_LONG_WORDS = 40
_CHAT_VOCAB_WORDS = "what is the answer yes no ? .".split()
_CHAT_TEMPLATE = (
    "{% for message in messages %}{% if message['role'] == 'assistant' %}"
    "<|assistant|>{% generation %}{{ message['content'] }}<|end|>"
    "{% endgeneration %}{% else %}<|{{ message['role'] }}|>"
    "{{ message['content'] }}<|end|>{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


@pytest.fixture(scope="module")
def chat_tokenizer() -> Any:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {"[UNK]": 0, **{word: i + 1 for i, word in enumerate(_CHAT_VOCAB_WORDS)}}
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        eos_token="<|end|>",
        pad_token="<|pad|>",
    )
    fast.add_special_tokens(
        {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
    )
    return fast


def make_jsonl_dataset(
    root: Path, name: str, words_per_doc: int, docs_per_shard: int, n_shards: int
) -> Dataset:
    path = root / name
    if not path.exists():
        path.mkdir(parents=True)
        for s in range(n_shards):
            lines = [
                json.dumps(
                    {
                        "text": " ".join(f"{name}w{j}" for j in range(words_per_doc)),
                        "id": f"{name}-{s}-{i}",
                    }
                )
                for i in range(docs_per_shard)
            ]
            (path / f"shard_{s:05d}.jsonl").write_text("\n".join(lines) + "\n")
    return Dataset.from_path(name, str(path))


def make_work_source(
    tmp_path: Path, *, mixture_unit: str, seed: int = 11, **kwargs
) -> StaticMixtureWorkSource:
    short = make_jsonl_dataset(
        tmp_path / mixture_unit, "short", SHORT_WORDS, docs_per_shard=120, n_shards=4
    )
    long = make_jsonl_dataset(
        tmp_path / mixture_unit, "long", LONG_WORDS, docs_per_shard=120, n_shards=4
    )
    extra = {}
    if mixture_unit == "tokens":
        extra["token_estimation"] = TokenEstimation(calibration_samples=24)
    return StaticMixtureWorkSource(
        [short, long],
        {"short": 0.5, "long": 0.5},
        chunk_size=64,
        seed=seed,
        **extra,
        **kwargs,
    )


def make_pipe(ws: StaticMixtureWorkSource, **options) -> Pipeline:
    return (
        Pipeline(ws)
        .tokenize(tokenizer_id="__fallback__", field="text", special_tokens="none")
        .ensure_mixture(max_buffer_size=256)
        .options(flush_every_k_chunks=4, **options)
    )


def token_share_by_dataset(records: list[SampleRecord]) -> dict[int, float]:
    """Token mass per dataset_id (sample_id[0]), normalized."""
    tokens: dict[int, int] = {}
    for rec in records:
        ds_id = int(rec.meta.sample_id[0])
        tokens[ds_id] = tokens.get(ds_id, 0) + len(rec.payload["input_ids"])
    total = sum(tokens.values())
    assert total > 0
    return {ds: t / total for ds, t in tokens.items()}


def test_token_mode_delivers_token_target_sample_mode_shows_skew(tmp_path) -> None:
    """Token mode hits the token target; sample mode shows length skew."""
    token_records = [
        item
        for item in make_pipe(make_work_source(tmp_path, mixture_unit="tokens"))
        if isinstance(item, SampleRecord)
    ]
    token_share = token_share_by_dataset(token_records)
    assert token_share[0] == pytest.approx(0.5, abs=0.03)  # short
    assert token_share[1] == pytest.approx(0.5, abs=0.03)  # long

    sample_records = [
        item
        for item in make_pipe(make_work_source(tmp_path, mixture_unit="samples"))
        if isinstance(item, SampleRecord)
    ]
    sample_share = token_share_by_dataset(sample_records)
    expected_skew = SHORT_WORDS / (SHORT_WORDS + LONG_WORDS)  # 0.2 for 12/48
    assert sample_share[0] == pytest.approx(expected_skew, abs=0.05)


def _chat_messages(answer_words: int) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "what is the answer ?"},
        {"role": "assistant", "content": " ".join(["yes"] * answer_words)},
    ]


def _multi_turn_chat_messages(answer_words: int) -> list[dict[str, str]]:
    first_answer_words = answer_words // 2
    return [
        {"role": "user", "content": "what is the answer ?"},
        {"role": "assistant", "content": " ".join(["yes"] * first_answer_words)},
        {"role": "user", "content": "what is the answer ?"},
        {
            "role": "assistant",
            "content": " ".join(["no"] * (answer_words - first_answer_words)),
        },
    ]


def _prompt_response_messages(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = payload.pop("prompt")
    response = payload.pop("response")
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def make_chat_jsonl_dataset(
    root: Path,
    name: str,
    answer_words: int,
    docs_per_shard: int,
    n_shards: int,
    messages_builder: Callable[[int], list[dict[str, str]]],
) -> Dataset:
    path = root / name
    if not path.exists():
        path.mkdir(parents=True)
        messages = messages_builder(answer_words)
        for shard_id in range(n_shards):
            lines = [
                json.dumps(
                    {
                        "messages": messages,
                        "id": f"{name}-{shard_id}-{row_id}",
                    }
                )
                for row_id in range(docs_per_shard)
            ]
            (path / f"shard_{shard_id:05d}.jsonl").write_text("\n".join(lines) + "\n")
    return Dataset.from_path(name, str(path))


def make_prompt_response_dataset(
    root: Path,
    name: str,
    answer_words: int,
    *,
    docs_per_shard: int = 4,
    n_shards: int = 2,
) -> Dataset:
    path = root / name
    path.mkdir(parents=True)
    for shard_id in range(n_shards):
        lines = [
            json.dumps(
                {
                    "prompt": "what is the answer ?",
                    "response": " ".join(["yes"] * answer_words),
                    "id": f"{name}-{shard_id}-{row_id}",
                }
            )
            for row_id in range(docs_per_shard)
        ]
        (path / f"shard_{shard_id:05d}.jsonl").write_text("\n".join(lines) + "\n")
    return Dataset.from_path(name, str(path))


def make_chat_work_source(
    tmp_path: Path,
    *,
    token_aware: bool,
    messages_builder: Callable[[int], list[dict[str, str]]] = _chat_messages,
    seed: int = 11,
) -> StaticMixtureWorkSource:
    root = tmp_path / ("tokens" if token_aware else "samples")
    short = make_chat_jsonl_dataset(
        root,
        "chat_short",
        CHAT_SHORT_WORDS,
        docs_per_shard=120,
        n_shards=4,
        messages_builder=messages_builder,
    )
    long = make_chat_jsonl_dataset(
        root,
        "chat_long",
        CHAT_LONG_WORDS,
        docs_per_shard=120,
        n_shards=4,
        messages_builder=messages_builder,
    )
    extra = (
        {"token_estimation": TokenEstimation(calibration_samples=24)}
        if token_aware
        else {}
    )
    return StaticMixtureWorkSource(
        [short, long],
        {"chat_short": 0.5, "chat_long": 0.5},
        chunk_size=64,
        seed=seed,
        **extra,
    )


def make_chat_pipe(ws: StaticMixtureWorkSource, tokenizer: Any) -> Pipeline:
    return (
        Pipeline(ws)
        .tokenize_chat(
            tokenizer,
            chat_template=_CHAT_TEMPLATE,
            span_source="generation_tags",
        )
        .ensure_mixture(max_buffer_size=256)
        .options(flush_every_k_chunks=4)
    )


def _assert_chat_token_mode_delivers_target_while_sample_mode_skews(
    tmp_path: Path,
    chat_tokenizer: Any,
    messages_builder: Callable[[int], list[dict[str, str]]],
) -> None:
    token_records = [
        item
        for item in make_chat_pipe(
            make_chat_work_source(
                tmp_path,
                token_aware=True,
                messages_builder=messages_builder,
            ),
            chat_tokenizer,
        )
        if isinstance(item, SampleRecord)
    ]
    token_share = token_share_by_dataset(token_records)
    assert token_share[0] == pytest.approx(0.5, abs=0.03)
    assert token_share[1] == pytest.approx(0.5, abs=0.03)

    sample_records = [
        item
        for item in make_chat_pipe(
            make_chat_work_source(
                tmp_path,
                token_aware=False,
                messages_builder=messages_builder,
            ),
            chat_tokenizer,
        )
        if isinstance(item, SampleRecord)
    ]
    sample_share = token_share_by_dataset(sample_records)

    reference = TokenizeChat(
        chat_tokenizer,
        chat_template=_CHAT_TEMPLATE,
        span_source="generation_tags",
    )
    short_tokens = reference.count_delivered_tokens(
        {"messages": messages_builder(CHAT_SHORT_WORDS)}
    )
    long_tokens = reference.count_delivered_tokens(
        {"messages": messages_builder(CHAT_LONG_WORDS)}
    )
    expected_short_share = short_tokens / (short_tokens + long_tokens)
    assert sample_share[0] == pytest.approx(expected_short_share, abs=0.03)
    assert sample_share[1] == pytest.approx(1.0 - expected_short_share, abs=0.03)
    assert abs(sample_share[0] - 0.5) > 0.15
    assert abs(token_share[0] - 0.5) < abs(sample_share[0] - 0.5)


def test_chat_token_mode_delivers_target_while_sample_mode_skews(
    tmp_path: Path, chat_tokenizer: Any
) -> None:
    _assert_chat_token_mode_delivers_target_while_sample_mode_skews(
        tmp_path, chat_tokenizer, _chat_messages
    )


def test_multi_turn_chat_token_mode_delivers_target_while_sample_mode_skews(
    tmp_path: Path, chat_tokenizer: Any
) -> None:
    _assert_chat_token_mode_delivers_target_while_sample_mode_skews(
        tmp_path, chat_tokenizer, _multi_turn_chat_messages
    )


def test_chat_priming_replays_pre_tokenize_map(
    tmp_path: Path, chat_tokenizer: Any
) -> None:
    root = tmp_path / "prompt-response"
    short = make_prompt_response_dataset(root, "short", CHAT_SHORT_WORDS)
    long = make_prompt_response_dataset(root, "long", CHAT_LONG_WORDS)
    ws = StaticMixtureWorkSource(
        [short, long],
        {"short": 0.5, "long": 0.5},
        chunk_size=8,
        exhausted_policy="stop",
        token_estimation=TokenEstimation(
            calibration_samples=24,
            calibration_shards_min=2,
            calibration_shards_max=2,
        ),
    )
    pipe = (
        Pipeline(ws)
        .map_transform(_prompt_response_messages)
        .tokenize_chat(
            chat_tokenizer,
            chat_template=_CHAT_TEMPLATE,
            span_source="generation_tags",
        )
        .options(deterministic=True, max_workers=1, default_stage_prefetch=0)
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        records = [item for item in pipe if isinstance(item, SampleRecord)]

    assert records
    assert all("input_ids" in record.payload for record in records)
    assert not any("priming fell back" in str(item.message) for item in caught)


def test_token_mode_is_lossless_with_bounded_buffer(tmp_path) -> None:
    """Bounded ensure_mixture should not discard token-balanced input."""
    # Single pass makes losslessness observable as unique sample IDs.
    ws = make_work_source(tmp_path, mixture_unit="tokens", exhausted_policy="stop")
    delivered = [item for item in make_pipe(ws) if isinstance(item, SampleRecord)]
    delivered_ids = [tuple(map(int, rec.meta.sample_id)) for rec in delivered]
    assert len(delivered_ids) == len(set(delivered_ids)), "duplicate deliveries"

    # Compare against the deterministic source admission set.
    twin = make_work_source(
        tmp_path / "twin", mixture_unit="tokens", exhausted_policy="stop"
    )
    twin.prime(
        counting_spec=TextTokenCountingSpec(
            tokenizer_id="__fallback__", field="text", special_tokens="none"
        )
    )
    lane = twin.clone_for_lane(0, canonical_replicas=1)
    admitted: set[tuple[int, int, int]] = set()
    while (chunk := lane.next_chunk()) is not None:
        for ids in chunk.components.values():
            admitted.update(tuple(map(int, sid)) for sid in ids)
    assert set(delivered_ids) == admitted


@pytest.mark.parametrize("mtp_mode", [False, True], ids=["inline", "mtp"])
def test_token_mode_checkpoint_resume_matches_baseline(tmp_path, mtp_mode) -> None:
    """Checkpoint/restore preserves the token-balanced stream."""

    def fresh_pipe() -> Pipeline:
        return make_pipe(
            make_work_source(tmp_path, mixture_unit="tokens"), mtp_mode=mtp_mode
        )

    baseline, _ = consume_until(fresh_pipe())
    assert baseline

    cut = len(baseline) // 2 + 3
    p1 = fresh_pipe()
    prefix, ckpt = consume_until(p1, flat_limit=cut)
    assert ckpt is not None
    assert prefix == baseline[:cut]

    p2 = fresh_pipe()
    p2.restore(ckpt)
    suffix, ckpt2 = consume_until(p2, flat_limit=32)
    assert prefix + suffix == baseline[: cut + len(suffix)]
    # Restored engines must remain checkpointable.
    assert ckpt2 is not None

    p3 = fresh_pipe()
    p3.restore(ckpt2)
    tail, _ = consume_until(p3)
    assert prefix + suffix + tail == baseline


def test_token_mode_restored_run_does_not_reprime(tmp_path, monkeypatch) -> None:
    """Restore must take ratios from the checkpoint: priming is forbidden."""
    import zephon.work.token_estimation as te

    pipe = make_pipe(make_work_source(tmp_path, mixture_unit="tokens"))
    _, ckpt = consume_until(pipe, flat_limit=40)
    assert ckpt is not None

    def boom(**kwargs):
        raise AssertionError("prime_token_ratios must not run on restore")

    monkeypatch.setattr(te, "prime_token_ratios", boom)
    monkeypatch.setattr("zephon.work.static_mixture.prime_token_ratios", boom)
    p2 = make_pipe(make_work_source(tmp_path / "r", mixture_unit="tokens"))
    p2.restore(ckpt)
    out, _ = consume_until(p2, flat_limit=20)
    assert out


# Pretokenized data, with no tokenize op in the pipeline.


def make_pretokenized_dataset(
    root: Path, name: str, tokens_per_doc: int, docs_per_shard: int, n_shards: int
) -> Dataset:
    path = root / name
    if not path.exists():
        path.mkdir(parents=True)
        for s in range(n_shards):
            lines = [
                json.dumps(
                    {"input_ids": list(range(tokens_per_doc)), "id": f"{name}-{s}-{i}"}
                )
                for i in range(docs_per_shard)
            ]
            (path / f"shard_{s:05d}.jsonl").write_text("\n".join(lines) + "\n")
    return Dataset.from_path(name, str(path))


def make_pretok_work_source(
    tmp_path: Path, *, token_aware: bool, seed: int = 11, **kwargs
) -> StaticMixtureWorkSource:
    short = make_pretokenized_dataset(
        tmp_path / "pretok", "pshort", SHORT_WORDS, docs_per_shard=120, n_shards=4
    )
    long = make_pretokenized_dataset(
        tmp_path / "pretok", "plong", LONG_WORDS, docs_per_shard=120, n_shards=4
    )
    extra = (
        {"token_estimation": TokenEstimation(calibration_samples=24)}
        if token_aware
        else {}
    )
    return StaticMixtureWorkSource(
        [short, long],
        {"pshort": 0.5, "plong": 0.5},
        chunk_size=64,
        seed=seed,
        **extra,
        **kwargs,
    )


def make_pretok_pipe(ws: StaticMixtureWorkSource, **options) -> Pipeline:
    return (
        Pipeline(ws)
        .ensure_mixture(max_buffer_size=256)
        .options(flush_every_k_chunks=4, **options)
    )


def test_pretokenized_token_mode_delivers_token_target(tmp_path) -> None:
    """Pretokenized token mode hits the token target without a tokenize op."""
    token_records = [
        item
        for item in make_pretok_pipe(
            make_pretok_work_source(tmp_path, token_aware=True)
        )
        if isinstance(item, SampleRecord)
    ]
    token_share = token_share_by_dataset(token_records)
    assert token_share[0] == pytest.approx(0.5, abs=0.03)  # pshort
    assert token_share[1] == pytest.approx(0.5, abs=0.03)  # plong

    sample_records = [
        item
        for item in make_pretok_pipe(
            make_pretok_work_source(tmp_path, token_aware=False)
        )
        if isinstance(item, SampleRecord)
    ]
    sample_share = token_share_by_dataset(sample_records)
    expected_skew = SHORT_WORDS / (SHORT_WORDS + LONG_WORDS)  # 0.2 for 12/48
    assert sample_share[0] == pytest.approx(expected_skew, abs=0.05)


def test_token_mode_default_exhaustion_is_finite(tmp_path) -> None:
    """Default token pipelines terminate after one slowest-dataset pass."""
    ws = make_work_source(tmp_path, mixture_unit="tokens")
    assert ws._alloc_config.stop_after_passes == 1
    # Bound the test so an infinite-stream regression fails as a cap hit.
    bound = 20000
    records, ckpt = consume_until(make_pipe(ws), flat_limit=bound)
    assert len(records) < bound, (
        f"token pipeline with default stop_after_passes=1 did not terminate; "
        f"hit the {bound}-record cap (infinite stream)"
    )
    assert ckpt is None


# Node-local single-flight for priming.


def _prime_one_rank(catalog_dir: Path, root: Path, marker: Path, result: Path) -> None:
    """Prime one spawned rank and record if it performed the census."""
    os.environ["ZEPHON_CATALOG_DIR"] = str(catalog_dir)
    import zephon.work.token_estimation as te
    from zephon.io.catalog import set_catalog_dir

    set_catalog_dir(None)  # Pick up the process env override.

    real_census = te._run_census

    def recording_census(measured_datasets, *args, **kwargs):
        with open(marker, "a") as fobj:
            fobj.write(f"{os.getpid()}\n")
        return real_census(measured_datasets, *args, **kwargs)

    te._run_census = recording_census

    short = Dataset.from_path("short", str(root / "short"))
    long = Dataset.from_path("long", str(root / "long"))
    ratios = te.prime_token_ratios(
        datasets=[short, long],
        dataset_ids={"short": 0, "long": 1},
        estimation=TokenEstimation(calibration_samples=16),
        counting_spec=TextTokenCountingSpec(
            tokenizer_id="__fallback__", field="text", special_tokens="none"
        ),
        seed=11,
    )
    result.write_text(
        json.dumps({n: [r.tokens_per_byte, r.source] for n, r in ratios.items()})
    )


def test_priming_is_single_flight_across_ranks(tmp_path) -> None:
    """Concurrent ranks share one census through the node-local cache."""
    root = tmp_path / "data"
    make_jsonl_dataset(root, "short", SHORT_WORDS, docs_per_shard=80, n_shards=4)
    make_jsonl_dataset(root, "long", LONG_WORDS, docs_per_shard=80, n_shards=4)
    catalog_dir = tmp_path / "shared-catalog"
    marker = tmp_path / "censused-pids.txt"

    ctx = mp.get_context("spawn")
    results = [tmp_path / f"ratios-{i}.json" for i in range(4)]
    procs = [
        ctx.Process(
            target=_prime_one_rank, args=(catalog_dir, root, marker, results[i])
        )
        for i in range(4)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]

    censused = marker.read_text().split() if marker.exists() else []
    assert len(censused) == 1, f"census ran {len(censused)} times, expected 1"

    loaded = [json.loads(r.read_text()) for r in results]
    base = loaded[0]
    assert base["short"][1] == "measured" and base["long"][1] == "measured"
    for other in loaded[1:]:
        assert other.keys() == base.keys()
        for name, (ratio, source) in base.items():
            assert other[name][0] == pytest.approx(ratio)
            assert other[name][1] == source
