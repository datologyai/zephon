# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end: sample-aware mixture → tokenize_chat → pack_flat.

Every span source must deliver, through the full engine, exactly the
supervision the op produces standalone: packing may reorder and pad but never
gain or lose supervised tokens, and every packed document must start
unsupervised (the packed-shift boundary invariant).
"""

from __future__ import annotations

import os
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from zephon import Pipeline
from zephon._internal.ops.tokenize_chat import TokenizeChat
from zephon.io import Dataset, InMemoryShard
from zephon.ops.config import LossScope
from zephon.types import SampleMeta, SampleRecord
from zephon.work.static_mixture import StaticMixtureWorkSource
from zephon.work.token_estimation import TokenEstimation

pytestmark = pytest.mark.integration

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")


class _ProbeRecordingTokenizer(transformers.PreTrainedTokenizerFast):
    """Record HF mask comparisons across parent and worker processes."""

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("return_assistant_tokens_mask"):
            with open(self.probe_log, "a") as output:
                output.write(f"{os.getpid()} {bool(kwargs.get('truncation'))}\n")
        return super().apply_chat_template(*args, **kwargs)


def test_process_worker_reuses_preflight_setup_checks(tmp_path: Path) -> None:
    from tests.helpers.chat import CYCLE, SPLIT, make_fast_tokenizer

    base = make_fast_tokenizer()
    tokenizer = _ProbeRecordingTokenizer(
        tokenizer_object=base.backend_tokenizer, **base.special_tokens_map
    )
    tokenizer.chat_template = {"default": SPLIT, "tool_use": "<|sep|>" + SPLIT}
    log = tmp_path / "probes.txt"
    tokenizer.probe_log = str(log)
    work = StaticMixtureWorkSource(
        [
            Dataset.from_dict(
                "chat",
                {
                    0: InMemoryShard(
                        [{"messages": CYCLE}, {"messages": CYCLE, "tools": []}]
                    )
                },
            )
        ],
        {"chat": 1.0},
        chunk_size=1,
    )
    pipe = (
        Pipeline(work)
        .tokenize_chat(tokenizer, loss_scope="after_last_user")
        .options(runner="process", max_workers=1, canonical_replicas=1)
    )
    assert pipe.preflight_tokenizers().ok
    records = list(pipe)
    assert len(records) == 2
    for record in records:
        targets = tokenizer.decode(
            record.payload["input_ids"][record.payload["loss_mask"].astype(bool)]
        )
        assert "how" in targets and "answer" in targets and "12" not in targets
    sep = tokenizer.convert_tokens_to_ids("<|sep|>")
    assert records[0].payload["input_ids"][0] != sep
    assert records[1].payload["input_ids"][0] == sep
    probes = [line.split() for line in log.read_text().splitlines()]
    parent = str(os.getpid())
    assert sum(pid == parent and truncated == "True" for pid, truncated in probes) == 4
    workers = [truncated for pid, truncated in probes if pid != parent]
    assert workers == ["False", "False"]


@pytest.mark.parametrize("loss_scope", ["all_tokens", "all_assistant"])
def test_chat_tokenizer_specials_and_supervision_survive_process_resume(
    loss_scope: LossScope,
) -> None:
    from tests.helpers.chat import make_fast_tokenizer

    tokenizer = make_fast_tokenizer()
    tokenizer.add_special_tokens({"bos_token": "<|start|>"})
    tokenizer.backend_tokenizer.post_processor = (
        tokenizers.processors.TemplateProcessing(
            single="<|start|> $A <|end|>",
            special_tokens=[
                ("<|start|>", tokenizer.bos_token_id),
                ("<|end|>", tokenizer.eos_token_id),
            ],
        )
    )
    documents = [
        {"prompt": prompt + " ", "answer": answer}
        for prompt, answer in (
            ("hi", "there"),
            ("good", "morning"),
            ("the", "answer"),
            ("is", "4"),
        )
    ]

    def make_pipeline() -> Pipeline:
        source = StaticMixtureWorkSource(
            [Dataset.from_dict("chat", {0: InMemoryShard(documents)})],
            {"chat": 1.0},
            chunk_size=1,
            shuffle_shards=False,
            shuffle_within_shard=False,
            token_estimation=TokenEstimation(
                calibration_samples=4,
                calibration_shards_min=1,
                calibration_shards_max=1,
            ),
        )
        return (
            Pipeline(source)
            .map_transform(
                lambda p: {
                    "messages": [
                        {"role": "user", "content": p["prompt"]},
                        {"role": "assistant", "content": p["answer"]},
                    ]
                }
            )
            .tokenize_chat(
                tokenizer,
                apply_chat_template=False,
                special_tokens="tokenizer_default",
                loss_scope=loss_scope,
            )
            .pack_flat(
                9,
                algorithm="first_fit",
                num_bins=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            .batch(1)
            .options(
                runner="process",
                deterministic=True,
                max_workers=1,
                canonical_replicas=1,
            )
        )

    pipeline, restored = make_pipeline(), make_pipeline()
    assert pipeline.preflight_tokenizers().ok
    with closing(iter(pipeline)) as iterator:
        first = next(iterator).to_training(dtype=None, return_labels=True)
        bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
        hi, there, good, morning = tokenizer.convert_tokens_to_ids(
            ["hi", "there", "good", "morning"]
        )
        assert first["input_ids"] == [[bos, hi, there, eos, bos, good, morning, eos]]
        assert first["labels"] == [
            [
                hi if loss_scope == "all_tokens" else -100,
                there,
                eos,
                -100,
                good if loss_scope == "all_tokens" else -100,
                morning,
                eos,
                -100,
            ]
        ]
        checkpoint = pipeline.checkpoint()
        remaining = [
            batch.to_training(dtype=None, return_labels=True) for batch in iterator
        ]
        assert len(remaining) == 1
    restored.restore(checkpoint)
    with closing(iter(restored)) as iterator:
        assert [
            batch.to_training(dtype=None, return_labels=True) for batch in iterator
        ] == remaining


SEQ = 32


_WORDS = "hello hi there how are you good morning the answer is what thanks yes no 1 2 4 + ? ! .".split()

TAGGED_TEMPLATE = (
    "{% for message in messages %}{% if message['role'] == 'assistant' %}"
    "<|assistant|>{% generation %}{{ message['content'] }}<|end|>{% endgeneration %}"
    "{% else %}<|{{ message['role'] }}|>{{ message['content'] }}<|end|>{% endif %}"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
UNTAGGED_TEMPLATE = (
    "{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}"
    "<|end|>{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)

#: The same conversations exercise: single-turn, system-start, multi-turn,
#: truncation (renders past SEQ), and the degenerate-drop path (empty).
CONVERSATIONS = [
    [
        {"role": "user", "content": "what is 2 + 2 ?"},
        {"role": "assistant", "content": "the answer is 4 ."},
    ],
    [
        {"role": "system", "content": "you are good ."},
        {"role": "user", "content": "hello ?"},
        {"role": "assistant", "content": "hi there !"},
    ],
    [
        {"role": "user", "content": "how are you ?"},
        {"role": "assistant", "content": "good morning ."},
        {"role": "user", "content": "thanks !"},
        {"role": "assistant", "content": "yes ."},
    ],
    [
        {"role": "user", "content": "how are you ?"},
        {"role": "assistant", "content": "good morning . " * 6},
        {"role": "user", "content": "what is 1 + 1 ?"},
        {"role": "assistant", "content": "the answer is 2 ."},
    ],
    [],
]

#: Only the template is given; span_source stays "auto" so the e2e also
#: exercises the tagged-vs-untagged inference (asserted per mode below).
MODES = {
    "generation_tags": {"chat_template": TAGGED_TEMPLATE},
    "prefix_diff": {"chat_template": UNTAGGED_TEMPLATE},
    "no_template": {"apply_chat_template": False},
}


@pytest.fixture(scope="module")
def fast_tokenizer():
    vocab = {"[UNK]": 0, **{w: i + 1 for i, w in enumerate(_WORDS)}}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        eos_token="<|end|>",
        pad_token="<|pad|>",
    )
    fast.add_special_tokens(
        {"additional_special_tokens": ["<|user|>", "<|assistant|>", "<|system|>"]}
    )
    return fast


def _rows(prefix: int) -> list[dict]:
    # Distinct row identity per dataset; both cycle the same conversations.
    return [
        {"messages": CONVERSATIONS[(prefix + i) % len(CONVERSATIONS)]}
        for i in range(10)
    ]


def _reference_outputs(op: TokenizeChat, rows: list[dict]) -> list[dict]:
    """What the op alone delivers per row, drops applied."""
    records = [
        SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, i), lane_id=0, chunk_id=0), payload=row
        )
        for i, row in enumerate(rows)
    ]
    kept = [r for r in op.process_many(records) if not r.meta.tombstone]
    return [r.payload for r in kept]  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mode", "runner"),
    [
        ("generation_tags", "threads"),
        ("prefix_diff", "process"),
        ("no_template", "threads"),
    ],
)
def test_mixture_tokenize_chat_pack_conserves_supervision(
    fast_tokenizer, mode: str, runner: str
) -> None:
    op_kwargs = MODES[mode]
    rows_a, rows_b = _rows(0), _rows(2)
    work = StaticMixtureWorkSource(
        [
            Dataset.from_dict("chat_a", {0: InMemoryShard(rows_a)}),
            Dataset.from_dict("chat_b", {0: InMemoryShard(rows_b)}),
        ],
        {"chat_a": 0.5, "chat_b": 0.5},
        # Must divide the total row count: the mixture emits complete chunks.
        chunk_size=10,
        seed=7,
    )
    pipe = (
        Pipeline(work)
        .tokenize_chat(fast_tokenizer, max_length=SEQ, **op_kwargs)
        .pack_flat(
            SEQ,
            num_bins=4,
            algorithm="best_fit",
            pad_token_id=fast_tokenizer.pad_token_id,
        )
        .options(deterministic=True, runner=runner, max_workers=2)
    )
    windows = [r for r in pipe if isinstance(r, SampleRecord) and not r.meta.tombstone]
    assert len(windows) > 1

    ref_op = TokenizeChat(fast_tokenizer, max_length=SEQ, **op_kwargs)
    reference = _reference_outputs(ref_op, rows_a + rows_b)
    if mode != "no_template":
        # Auto-detection must have picked the span source the mode expects.
        assert ref_op._fixed_template.span_source == mode
    # The empty conversation must have been dropped, everything else kept.
    assert len(reference) == len(rows_a + rows_b) - 4

    pad_id = fast_tokenizer.pad_token_id
    marker_ids = fast_tokenizer.convert_tokens_to_ids(
        ["<|user|>", "<|assistant|>", "<|system|>"]
    )
    assistant_id = marker_ids[1]
    packed_real = packed_supervised = 0
    for window in windows:
        payload = window.payload
        assert isinstance(payload, dict)
        ids, mask, positions = (
            np.asarray(payload["input_ids"]),
            np.asarray(payload["loss_mask"]),
            np.asarray(payload["positions"]),
        )
        assert len(ids) == len(mask) == len(positions) == SEQ
        assert ids.dtype == np.int64 and mask.dtype == np.uint8
        # Packed-shift boundary invariant: every document (incl. the pad
        # tail) starts unsupervised.
        assert (mask[np.flatnonzero(positions == 0)] == 0).all()
        # Pad never carries supervision (the serializer zero-fills).
        assert (mask[ids == pad_id] == 0).all()
        # Turn-level mask validity, independent of the op's own output:
        # role headers are never supervised, and every supervised run sits
        # directly behind an <|assistant|> header (assistant content + its
        # turn EOS, nothing of the user/system turns).
        if mode != "no_template":
            assert (mask[np.isin(ids, marker_ids)] == 0).all()
            run_starts = np.flatnonzero(np.diff(mask.astype(np.int8)) == 1) + 1
            assert (ids[run_starts - 1] == assistant_id).all()
        packed_real += int((ids != pad_id).sum())
        packed_supervised += int(mask.sum())

    assert packed_real == sum(len(p["input_ids"]) for p in reference)
    assert packed_supervised == sum(int(p["loss_mask"].sum()) for p in reference)
