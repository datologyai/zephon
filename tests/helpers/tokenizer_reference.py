"""HF's recorded outputs on the committed tokenizer folders, and how to reproduce them.

``backend="rust"`` needs tokenizers 1.x and ``backend="hf"`` needs transformers,
which pins tokenizers 0.x, so no environment runs both. Each comparison is
instead a pure function of the backend; ``generate_tokenizer_reference``
records HF's results in ``reference.json``, the main test job checks HF still
reproduces them, and the ``rust`` job checks the rust backend does.
"""

from __future__ import annotations

import functools
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon import Pipeline
from zephon._internal.ops.tokenize_text import TokenizeText
from zephon._internal.utils.tokenizer import TOKENIZER_LOADERS
from zephon.ops.base import OpContext, StageInfo
from zephon.types import SampleBatch, SampleMeta, SampleRecord

DATA = Path(__file__).resolve().parents[1] / "data" / "tokenizers"
DIRS = {"deepseek": DATA / "deepseek_like", "llama3": DATA / "llama3_like"}
REFERENCE_PATH = DATA / "reference.json"

# Special and ``normalized: true`` added tokens of the tokenizer folders, as in
# DeepSeek V4's and Llama-3.2-1B's tokenizer.json (bos, eos, pad first).
DEEPSEEK_LIKE_SPECIALS = (
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜▁pad▁｜>",
)
DEEPSEEK_LIKE_NORMALIZED = (
    "<｜fim▁hole｜>",
    "<｜fim▁begin｜>",
    "<｜fim▁end｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
)
LLAMA3_LIKE_SPECIALS = (
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|finetune_right_pad_id|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token_0|>",
)
_FAMILY_TOKENS = {
    "deepseek": (DEEPSEEK_LIKE_SPECIALS, DEEPSEEK_LIKE_NORMALIZED),
    "llama3": (LLAMA3_LIKE_SPECIALS, ()),
}

# The modules holding ``rust``-marked tests: the rust CI job runs exactly these
# (the others need the test group to import), and conftest rejects a rust test
# collected anywhere else, which that job would never run.
RUST_TEST_FILES = (
    "tests/zephon/_internal/utils/test_tokenizer.py",
    "tests/zephon/_internal/ops/test_tokenize_text.py",
    "tests/zephon/_internal/test_token_counting.py",
    "tests/zephon/test_pipeline_tokenize_decode.py",
)

# hf is skipped only where transformers is absent (the 3.14t job), as every
# transformers-backed test is; rust runs only in the tokenizers 1.x job.
BACKENDS = [
    pytest.param(
        "hf",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("transformers") is None,
            reason="transformers is not installed",
        ),
    ),
    pytest.param("rust", marks=pytest.mark.rust),
]


def adversarial_texts(family: str) -> dict[str, str]:
    """Strings where HF and the rust engine are likeliest to disagree.

    Special-token text must stay content (``add_special_tokens=False``), and
    ``normalized: true`` added tokens match after normalization.
    """
    specials, normalized = _FAMILY_TOKENS[family]
    bos, eos, pad = specials[:3]
    return {
        "prose": "The quick brown fox jumps over the lazy dog.",
        "contractions": "It's what they'll say: we've done it, I'D agree.",
        "empty": "",
        "space": " ",
        "whitespace": "\n\n\t x\r\n",
        "leading_spaces": "  leading",
        "multibyte": "emoji 😀👍🏽 中文 日本語 ﬁ Ａ café",
        "digits": "1234567",
        "run_5000_chars": "lorem ipsum dolor " * 278,
        "eos_text": f"hello {eos} world",
        "bos_text": f"{bos}x",
        "pad_text": f"a{pad}b",
        **{f"special_{i}": f"x{t}y" for i, t in enumerate(specials[3:])},
        "unknown_special_text": "x<｜place▁holder▁no▁3｜>y<|reserved_special_token_9|>",
        **{f"normalized_{i}": f"q {t} z" for i, t in enumerate(normalized)},
    }


TRUNCATION_TEXTS = ["The quick brown fox jumps over the lazy dog.", "dog"]
OP_TEXTS = [
    "The quick brown fox jumps over the lazy dog, again and again. " * 4,
    "",
    "short 数据 😀",
    "q <｜fim▁hole｜> z",
]
OP_KWARGS: dict[str, dict[str, Any]] = {
    "bos_eos_split": {"split_long_samples": True, "max_length": 8},
    "truncation": {"truncation": True, "max_length": 8},
    "padding": {"padding": "max_length", "max_length": 64},
}
COUNT_TEXTS = ["The quick brown fox", "数据 😀 " * 20, "q <｜fim▁hole｜> z", "x"]


def as_json(value: Any) -> Any:
    """``value`` as it reads back from ``reference.json`` (arrays become lists)."""
    return json.loads(json.dumps(value, default=lambda a: a.tolist()))


@functools.cache
def load(backend: str, family: str) -> Any:
    return TOKENIZER_LOADERS[backend](str(DIRS[family]))


def ids(backend: str, family: str, text: str) -> Any:
    return load(backend, family)([text], add_special_tokens=False)["input_ids"][0]


def special_ids(backend: str, family: str) -> list[int | None]:
    tok = load(backend, family)
    return [tok.bos_token_id, tok.eos_token_id, tok.pad_token_id]


def truncated(backend: str) -> Any:
    return load(backend, "deepseek")(
        TRUNCATION_TEXTS, add_special_tokens=False, truncation=True, max_length=4
    )["input_ids"]


def op_records(backend: str, case: str) -> list[tuple[Any, Any, Any]]:
    """``TokenizeText.process_many`` output on ``OP_TEXTS`` under ``OP_KWARGS[case]``."""
    op = TokenizeText(
        tokenizer_id=str(DIRS["deepseek"]),
        field="text",
        backend=backend,
        **OP_KWARGS[case],
    )
    op.setup(
        OpContext(
            {"record_node_metrics": lambda *a, **k: None},
            StageInfo(stage_index=0, stage_name="stage0", op_index=0),
        )
    )
    records = [
        SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, i), lane_id=0, chunk_id=0),
            payload={"text": text},
        )
        for i, text in enumerate(OP_TEXTS)
    ]
    return [
        (rec.meta.sample_id, rec.meta.lineage, rec.payload)
        for rec in op.process_many(records)
    ]


def counts(backend: str) -> list[int | None]:
    """Token counts of ``COUNT_TEXTS`` from the op's calibration counter."""
    op = TokenizeText(tokenizer_id=str(DIRS["deepseek"]), field="text", backend=backend)
    payloads = [{"text": t} for t in COUNT_TEXTS]
    plan = op.token_counting_spec().build_counter().plan(payloads)
    return [plan.count(p) for p in payloads]


def training_windows(backend: str) -> list[dict[str, Any]]:
    """Packed training windows of a shuffled, tokenized pipeline."""
    rows = [
        {"text": f"Document {i}: the quick brown fox 数据 😀 " * (1 + i % 5)}
        for i in range(40)
    ]
    ws = FakeIndexableWorkSource(make_inmem_dataset("docs", rows), chunk_size=8)
    pipe = (
        Pipeline(ws)
        .decode_text()
        .tokenize(
            tokenizer_id=str(DIRS["deepseek"]),
            field="text",
            backend=backend,
            add_attention_mask=False,
        )
        .shuffle(buffer_size=16, seed=0)
        .pack_flat(max_length=32, algorithm="wrap")
        .batch(microbatch_size=4, drop_last=False)
        .options(deterministic=True, max_workers=1, default_stage_prefetch=0)
    )
    windows = []
    for batch in pipe:
        assert isinstance(batch, SampleBatch)
        out = batch.to_training(return_labels=True)
        windows.append({k: np.asarray(out[k]) for k in ("input_ids", "labels")})
    return windows


def compute(backend: str) -> dict[str, Any]:
    """Every output the tests compare, keyed as in ``reference.json``."""
    return as_json(
        {
            "ids": {
                family: {
                    name: ids(backend, family, text)
                    for name, text in adversarial_texts(family).items()
                }
                for family in DIRS
            },
            "special_ids": {family: special_ids(backend, family) for family in DIRS},
            "truncated": truncated(backend),
            "op_records": {case: op_records(backend, case) for case in OP_KWARGS},
            "counts": counts(backend),
            "training_windows": training_windows(backend),
        }
    )


@functools.cache
def reference() -> dict[str, Any]:
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
