# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from zephon import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.types import SampleRecord
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("runner", "drop_final"), [("threads", False), ("process", True)]
)
def test_query_prefixes_with_unanswered_rows_and_downstream_drops(
    runner: str, drop_final: bool
) -> None:
    shapes = [
        ["user", "assistant"] * 3,
        ["assistant", "user"],
        ["system", "user", "assistant"],
        ["user", "assistant", "user"],
        ["user", "user", "assistant", "user", "assistant"],
        ["user"],
    ]
    rows = [
        {
            "id": i,
            "messages": [
                {"role": role, "content": str(j)} for j, role in enumerate(roles)
            ],
        }
        for i, roles in enumerate(shapes)
    ]

    def build() -> Pipeline:
        work = StaticMixtureWorkSource(
            [Dataset.from_dict("chat", {0: InMemoryShard(rows)})],
            {"chat": 1.0},
            chunk_size=2,
            seed=3,
        )
        pipe = Pipeline(work).expand_chat()
        pipe.map_transform(lambda p: p)
        if drop_final:
            pipe.map_transform(lambda p: p if len(p["messages"]) < 6 else None)
        return pipe.options(
            runner=runner,
            deterministic=True,
            canonical_replicas=1,
            max_workers=2,
            default_stage_prefetch=0,
            prefetch_batches=0,
        )

    def identity(item: SampleRecord) -> tuple[Any, ...]:
        return (
            item.meta.as_cursor_key(),
            item.payload["id"],
            len(item.payload["messages"]),
        )

    baseline = [identity(item) for item in build()]
    assert len(baseline) == (6 if drop_final else 7)
    assert [item[2] for item in baseline] == (
        [2, 4, 3, 2, 3, 5] if drop_final else [2, 4, 6, 3, 2, 3, 5]
    )


def test_chat_calibration_counts_expanded_tokens_over_source_bytes() -> None:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    chat_tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        eos_token="<eos>",
    )
    template = (
        "{% for message in messages %}{% if message.role == 'assistant' %}"
        "{% generation %}{{ message.content }}{{ eos_token }}{% endgeneration %}"
        "{% else %}{{ message.content }} {% endif %}{% endfor %}"
    )
    from zephon._internal.observability.size_estimator import content_bytes
    from zephon.work.token_estimation import TokenEstimation

    payload = {
        "messages": [
            {"role": "user", "content": "what is the answer"},
            {"role": "assistant", "content": "yes"},
        ]
        * 2
    }
    dataset = Dataset.from_dict("chat", {0: InMemoryShard([payload])})
    work = StaticMixtureWorkSource(
        [dataset],
        {"chat": 1.0},
        chunk_size=1,
        exhausted_policy="stop",
        token_estimation=TokenEstimation(
            calibration_samples=1,
            calibration_shards_min=1,
            calibration_shards_max=1,
        ),
    )
    records = list(
        Pipeline(work)
        .expand_chat()
        .tokenize_chat(chat_tokenizer, chat_template=template)
        .options(runner="threads", max_workers=1, canonical_replicas=1)
    )
    assert len(records) == 2
    tokens = sum(len(record.payload["input_ids"]) for record in records)
    ratio, source = work.state_dict()["token_ratios"]["chat"]
    assert source == "measured"
    assert ratio == pytest.approx(tokens / content_bytes(payload))
    assert tokens > len(records[-1].payload["input_ids"])


def test_expanded_scoped_tokens_pack_with_document_boundaries() -> None:
    import numpy as np

    from tests.helpers.chat import CYCLE, SPLIT, make_fast_tokenizer

    tok = make_fast_tokenizer()
    work = StaticMixtureWorkSource(
        [Dataset.from_dict("chat", {0: InMemoryShard([{"messages": CYCLE}])})],
        {"chat": 1.0},
        chunk_size=1,
        seed=3,
    )
    windows = list(
        Pipeline(work)
        .expand_chat()
        .tokenize_chat(
            tok,
            chat_template=SPLIT,
            loss_scope="after_last_user",
        )
        .pack_flat(128, num_bins=2, pad_token_id=tok.pad_token_id)
        .options(
            runner="threads",
            deterministic=True,
            canonical_replicas=1,
            max_workers=2,
        )
    )
    targets: list[int] = []
    for window in windows:
        ids, mask, positions = (
            window.payload[key] for key in ("input_ids", "loss_mask", "positions")
        )
        assert not mask[np.flatnonzero(positions == 0)].any()
        assert not mask[ids == tok.pad_token_id].any()
        targets.extend(ids[mask.astype(bool)].tolist())
    assert targets.count(tok.convert_tokens_to_ids("hi")) == 1
    assert targets.count(tok.convert_tokens_to_ids("how")) == 1
    assert targets.count(tok.convert_tokens_to_ids("yes")) == 1
    assert tok.convert_tokens_to_ids("12") not in targets


def test_scoped_tokenization_drops_unanswered_rows() -> None:
    from tests.helpers.chat import CYCLE, SPLIT, make_fast_tokenizer

    rows = [
        {"messages": CYCLE},
        {
            "messages": [
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "hello"},
            ]
        },
        {"messages": CYCLE[:2]},
        {"messages": CYCLE},
    ]

    def build() -> Pipeline:
        work = StaticMixtureWorkSource(
            [Dataset.from_dict("chat", {0: InMemoryShard(rows)})],
            {"chat": 1.0},
            chunk_size=2,
            seed=3,
        )
        return (
            Pipeline(work)
            .expand_chat()
            .tokenize_chat(
                make_fast_tokenizer(), chat_template=SPLIT, loss_scope="after_last_user"
            )
            .options(
                runner="process",
                deterministic=True,
                canonical_replicas=1,
                max_workers=2,
                default_stage_prefetch=0,
                prefetch_batches=0,
            )
        )

    def identity(item: SampleRecord) -> tuple[Any, ...]:
        return (
            item.meta.cursor,
            item.payload["input_ids"].tolist(),
            item.payload["loss_mask"].tolist(),
        )

    baseline = [identity(item) for item in build()]
    assert len(baseline) == 5
    assert all(any(item[2]) for item in baseline)


def test_expansion_renders_each_query_as_final() -> None:
    import copy

    from tests.helpers.chat import CONVERSATION, SPLIT, make_fast_tokenizer
    from zephon._internal.ops.expand_chat import ExpandChat
    from zephon._internal.ops.tokenize_chat import TokenizeChat
    from zephon.types import SampleMeta

    fast_tokenizer = make_fast_tokenizer()
    template = (
        "{% set ns=namespace(user=0) %}{% for m in messages %}{% if m.role == 'user' %}{% set ns.user=loop.index0 %}{% endif %}{% endfor %}"
        + SPLIT.replace(
            "{% if m.reasoning_content %}",
            "{% if m.reasoning_content and loop.index0 > ns.user %}",
        )
    )
    messages = copy.deepcopy(CONVERSATION)
    messages[1]["reasoning_content"] = "how"
    messages[3]["reasoning_content"] = "yes"
    record = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"messages": messages},
    )
    prefixes = ExpandChat().process_one(record)
    op = TokenizeChat(
        fast_tokenizer, chat_template=template, loss_scope="after_last_user"
    )
    outputs = op.process_many(prefixes)
    assert len(outputs) == 2
    texts = [
        fast_tokenizer.decode(
            r.payload["input_ids"][r.payload["loss_mask"].astype(bool)]
        )
        for r in outputs
    ]
    assert "how" in texts[0] and "yes" in texts[1] and "how" not in texts[1]
