# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Calibrate custom operators through the public authoring API."""

import pytest

from zephon import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.ops import BaseOp, OpContext, OpTraits
from zephon.ops.children import spawn_child
from zephon.types import SampleRecord
from zephon.work import StaticMixtureWorkSource
from zephon.work.token_estimation import TokenEstimation

pytestmark = pytest.mark.integration


def duplicate_records(records: list[SampleRecord]) -> list[SampleRecord]:
    return [
        SampleRecord(
            meta=spawn_child(record.meta, index, is_last_child=index == 1),
            payload=dict(record.payload),
        )
        for record in records
        for index in range(2)
    ]


class DuplicateRecords(BaseOp):
    def setup(self, ctx: OpContext) -> None:
        super().setup(ctx)
        self._transform = duplicate_records

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=False, preserves_cursor_order=True, calibration_replayable=True
        )

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        return self._transform(elems)


@pytest.mark.parametrize("form", ["instance", "kwargs"])
def test_custom_fanout_changes_calibrated_token_ratio(form: str) -> None:
    dataset = Dataset.from_dict("text", {0: InMemoryShard([{"text": "one two"}] * 2)})

    def run(duplicate: bool) -> tuple[int, float]:
        work = StaticMixtureWorkSource(
            [dataset],
            {"text": 1.0},
            chunk_size=1,
            exhausted_policy="stop",
            token_estimation=TokenEstimation(
                calibration_samples=8,
                calibration_shards_min=1,
                calibration_shards_max=1,
            ),
        )
        pipe = Pipeline(work)
        if duplicate:
            if form == "instance":
                pipe.add_op(DuplicateRecords())
            else:
                pipe.add_op(
                    "duplicate",
                    process_many=duplicate_records,
                    preserves_cursor_order=True,
                    calibration_replayable=True,
                )
        records = list(
            pipe.tokenize(
                tokenizer_id="__fallback__", field="text", special_tokens="none"
            ).options(
                runner="threads",
                deterministic=True,
                canonical_replicas=1,
                max_workers=1,
                auto_validation="off",
            )
        )
        ratio, source = work.state_dict()["token_ratios"]["text"]
        assert source == "measured"
        return sum(len(record.payload["input_ids"]) for record in records), ratio

    baseline_tokens, baseline_ratio = run(False)
    tokens, ratio = run(True)
    assert baseline_tokens > 0
    assert baseline_ratio > 0
    assert tokens == 2 * baseline_tokens
    assert ratio == pytest.approx(2 * baseline_ratio)
