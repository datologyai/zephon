# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from zephon import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.types import SampleRecord
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("runner", ["threads", "process"])
@pytest.mark.parametrize("drop_final", [False, True])
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
