# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Source exhaustion through runners, epoch resets, buffering and replay."""

from collections import Counter, defaultdict
from typing import Any

import pytest

from zephon import Pipeline
from zephon._internal.ops.ensure_mixture import EnsureMixtureAccumulator
from zephon.io import Dataset, InMemoryShard
from zephon.ops.accumulators import ReadyBatch
from zephon.types import SampleRecord
from zephon.work import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def _make_pipe(
    *,
    runner: str = "inline",
    mtp_mode: bool = False,
    lanes: int = 1,
    flush_every: int = 128,
    shuffle: bool = False,
    limit: int | None = None,
) -> Pipeline:
    datasets = [
        Dataset.from_dict(
            name, {0: InMemoryShard([{"text": f"{name}-{i}"} for i in range(count)])}
        )
        for name, count in [("a", 3), ("b", 1000)]
    ]
    source = StaticMixtureWorkSource(
        datasets,
        {"a": 0.05, "b": 0.95},
        chunk_size=10,
        exhausted_policy="stop",
        shuffle_shards=False,
        seed=0,
    )
    pipe = Pipeline(source)
    if shuffle:
        pipe.shuffle(buffer_size=7, seed=4)
    return pipe.ensure_mixture(
        weight_by="samples",
        mixture={"a": 0.5, "b": 0.5},
        max_buffer_size=limit,
    ).options(
        runner=runner,
        mtp_mode=mtp_mode,
        deterministic=True,
        canonical_replicas=lanes,
        max_workers=2,
        default_stage_prefetch=2,
        flush_every_k_chunks=flush_every,
    )


def _value(record: SampleRecord) -> tuple[int, str]:
    assert isinstance(record, SampleRecord)
    assert not record.meta.is_sentinel
    return record.meta.lane_id, record.payload["text"]


def _per_lane(items: list[tuple[int, str]]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = defaultdict(list)
    for lane, text in items:
        out[lane].append(text)
    return out


@pytest.mark.parametrize(
    "runner,mtp_mode",
    [
        ("inline", False),
        ("threads", False),
        ("process", False),
        ("inline", True),
    ],
)
@pytest.mark.parametrize("limit", [None, 64])
def test_source_exhaustion_releases_survivors_through_transports(
    runner: str,
    mtp_mode: bool,
    limit: int | None,
) -> None:
    pipe = _make_pipe(runner=runner, mtp_mode=mtp_mode, limit=limit)
    records = [_value(r) for r in pipe]
    # Without source exhaustion strict mode delivers 3 A and 3 B, discarding
    # the other 54 B. Announcing A's exhaustion releases that live surplus.
    assert len(records) == len(set(records)) == 60
    assert sum(text.startswith("a-") for _, text in records) == 3


class _CaptureExhaustion(EnsureMixtureAccumulator):
    calls: list[tuple[int, int]] = []

    def _on_source_exhausted(
        self, lane_id: int, component_id: int
    ) -> list[ReadyBatch[SampleRecord]]:
        self.calls.append((lane_id, component_id))
        return super()._on_source_exhausted(lane_id, component_id)


def test_notification_is_once_per_lane_across_epoch_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zephon._internal.ops.ensure_mixture as em

    _CaptureExhaustion.calls = []
    monkeypatch.setattr(em, "EnsureMixtureAccumulator", _CaptureExhaustion)
    records = [_value(r) for r in _make_pipe(lanes=2, flush_every=1)]
    assert sorted(_CaptureExhaustion.calls) == [(0, 0), (1, 0)]
    # Both lanes' final chunks follow the global exhaustion point. Later
    # epochs retain exhaustion, including on the lane without the last A.
    assert all(len(texts) >= 10 for texts in _per_lane(records).values())


@pytest.mark.parametrize(
    "runner,mtp_mode",
    [
        ("inline", False),
        ("threads", False),
        ("process", False),
        ("inline", True),
    ],
)
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("cut", [2, 12])
def test_source_exhaustion_checkpoint_replays_with_upstream_buffers(
    runner: str,
    mtp_mode: bool,
    shuffle: bool,
    cut: int,
) -> None:
    options = dict(
        runner=runner, mtp_mode=mtp_mode, shuffle=shuffle, lanes=2, flush_every=2
    )
    baseline = [_value(r) for r in _make_pipe(**options)]
    pipe = _make_pipe(**options)
    it = iter(pipe)
    prefix = []
    checkpoint: dict[str, Any]
    try:
        # Exercise an early cut and one after a marker released starvation.
        prefix = [_value(next(it)) for _ in range(cut)]
        checkpoint = pipe.checkpoint()
    finally:
        it.close()
    restored = _make_pipe(**options)
    restored.restore(checkpoint)
    suffix = [_value(r) for r in restored]
    assert _per_lane(prefix + suffix) == _per_lane(baseline)
    # A checkpoint after the final marker cannot acknowledge its dummy cursor.
    done = restored.checkpoint()
    final = _make_pipe(**options)
    final.restore(done)
    assert list(final) == []


@pytest.mark.parametrize("limit", [None, 64])
def test_large_shuffle_keeps_exhausted_dataset_records(limit: int | None) -> None:
    """Notifications may pass most of A while its records remain in shuffle."""
    datasets = [
        Dataset.from_dict(
            name, {0: InMemoryShard([{"text": f"{name}-{i}"} for i in range(count)])}
        )
        for name, count in (("a", 200), ("b", 5000))
    ]
    source = StaticMixtureWorkSource(
        datasets, {"a": 0.5, "b": 0.5}, chunk_size=10, exhausted_policy="stop", seed=0
    )
    pipe = (
        Pipeline(source)
        .shuffle(buffer_size=256, seed=4)
        .ensure_mixture(weight_by="samples", max_buffer_size=limit)
        .options(
            runner="inline",
            deterministic=True,
            max_workers=1,
            canonical_replicas=1,
            flush_every_k_chunks=1000,
        )
    )
    values = [_value(r) for r in pipe]
    assert len(values) == len(set(values)) == 400
    assert Counter(text.split("-")[0] for _, text in values) == {"a": 200, "b": 200}
