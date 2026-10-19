# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

import pytest

from zephon import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.types import SampleRecord
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("runner", "drop_final"), [("threads", False), ("process", True)]
)
@pytest.mark.parametrize("copy_mode", ["deep", "shallow"])
def test_query_prefixes_with_unanswered_rows_and_downstream_drops(
    runner: str, drop_final: bool, copy_mode: Literal["deep", "shallow"]
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
        pipe = Pipeline(work).expand_chat(copy_mode=copy_mode)
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
