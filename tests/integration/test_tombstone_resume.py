# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import pytest

from zephon import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("runner", ["threads", "process"])
@pytest.mark.parametrize("downstream", ["map", "batch"])
def test_resume_after_mid_pipeline_drops(runner: str, downstream: str) -> None:
    def build() -> Pipeline:
        work = StaticMixtureWorkSource(
            [Dataset.from_dict("d", {0: InMemoryShard([{"id": i} for i in range(8)])})],
            {"d": 1.0},
            chunk_size=2,
            seed=3,
        )
        pipe = Pipeline(work).map_transform(lambda p: None if p["id"] in {2, 5} else p)
        if downstream == "batch":
            pipe.batch(2)
        else:
            pipe.map_transform(lambda p: p)
        return pipe.options(
            runner=runner,
            max_workers=2,
            canonical_replicas=1,
            deterministic=True,
            default_stage_prefetch=0,
            prefetch_batches=0,
        )

    def identity(item):
        records = item.records if hasattr(item, "records") else [item]
        return tuple((r.meta.as_cursor_key(), r.payload["id"]) for r in records)

    baseline = [identity(item) for item in build()]
    assert sum(len(item) for item in baseline) == 6
    for count in range(1, len(baseline) + 1):
        pipe = build()
        iterator = iter(pipe)
        try:
            prefix = [identity(next(iterator)) for _ in range(count)]
            checkpoint = pipe.checkpoint()
        finally:
            iterator.close()
        resumed = build()
        resumed.restore(checkpoint)
        assert prefix + [identity(item) for item in resumed] == baseline, count
