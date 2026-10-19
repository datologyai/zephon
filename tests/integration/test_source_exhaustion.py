# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Source exhaustion through runners, epoch resets, buffering and replay."""

import logging
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from zephon import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.ops.accumulators import Accumulator, ReadyBatch
from zephon.ops.base import BaseOp
from zephon.ops.traits import OpTraits
from zephon.types import SampleRecord
from zephon.work import StaticMixtureWorkSource, WorkChunk, WorkSource

pytestmark = [
    pytest.mark.integration,
    # The pass-through observer intentionally omits process_many validation samples.
    pytest.mark.filterwarnings(
        "ignore:(?s).*OP_INSTANCE_RUNTIME_CHECKS_SKIPPED in '_ObserveExhaustion':UserWarning"
    ),
]

_CHUNK_SIZE = 16
_LANE_SIZE = 8 * _CHUNK_SIZE


class _ContinuingSource(StaticMixtureWorkSource):
    """Opt into continuation handling within the finite fixture's final chunks."""

    def continues_after_exhaustion(self) -> frozenset[str]:
        return frozenset(self.component_ids())


class _ScriptedSource(WorkSource):
    """Four A/B chunks followed by four B-only chunks per lane, with resumable state."""

    def __init__(self, *, source_targets: bool) -> None:
        super().__init__()
        self.source_targets = source_targets
        self.position = 0
        self.datasets = {
            cid: Dataset.from_dict(
                name,
                {
                    0: InMemoryShard(
                        [{"text": f"{name}-{i}"} for i in range(2 * _LANE_SIZE)]
                    )
                },
            )
            for cid, name in enumerate(("a", "b"))
        }

    @property
    def datasets_by_id(self) -> Mapping[int, Dataset]:
        return self.datasets

    def component_ids(self) -> Mapping[str, int]:
        return {"a": 0, "b": 1}

    def continues_after_exhaustion(self) -> frozenset[str]:
        return frozenset({"a"})

    def chunk_size_hint(self) -> int:
        return _CHUNK_SIZE

    def next_chunk(self) -> WorkChunk | None:
        if self.position == 8:
            return None
        assert self._lane is not None
        index = self.position
        self.position += 1
        start = self._lane * _LANE_SIZE + index * _CHUNK_SIZE
        count_b = _CHUNK_SIZE // 2 if index < 4 else _CHUNK_SIZE
        components = {"b": [(1, 0, start + i) for i in range(count_b)]}
        if index < 4:
            components["a"] = [(0, 0, start + i) for i in range(_CHUNK_SIZE // 2)]
        exhausted = index >= 3
        target = (
            {"b": 1.0} if self.source_targets and exhausted else {"a": 0.5, "b": 0.5}
        )
        return WorkChunk(
            components=components,
            target_mixture=target,
            source_exhausted=("a",) if exhausted else (),
        )

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "position": self.position}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.position = state["position"]


class _ObserveAccumulator(Accumulator[SampleRecord]):
    def __init__(self, events: list[tuple[str, int, int]]) -> None:
        self.events = events

    def push_many(
        self, elems: Sequence[SampleRecord]
    ) -> list[ReadyBatch[SampleRecord]]:
        self.events.extend(("data", r.meta.lane_id, r.meta.chunk_id) for r in elems)
        return [(list(elems), 0)] if elems else []

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[SampleRecord]]:
        if reset:
            self.events.append(("flush", lane_id if lane_id is not None else -1, -1))
        return []

    def on_source_exhausted(
        self, lane_id: int, component_id: int
    ) -> list[ReadyBatch[SampleRecord]]:
        self.events.append(("exhausted", lane_id, component_id))
        return []


class _ObserveExhaustion(BaseOp):
    def __init__(self, events: list[tuple[str, int, int]]) -> None:
        super().__init__()
        self.events = events

    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True)

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return _ObserveAccumulator(self.events)

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        return elems


def _continuing_pipe(
    *,
    runner: str = "inline",
    mtp_mode: bool = False,
    source_targets: bool = False,
    shuffle: bool = False,
    events: list[tuple[str, int, int]] | None = None,
) -> Pipeline:
    # Match fetch batching to source chunks so the exhaustion boundary still
    # follows fetched data from the preceding chunks.
    pipe = Pipeline(_ScriptedSource(source_targets=source_targets)).fetch(
        max_batch=_CHUNK_SIZE
    )
    if shuffle:
        pipe.shuffle(buffer_size=7, seed=4)
    pipe.ensure_mixture(
        weight_by="samples",
        max_buffer_size=None,
        mixture=None if source_targets else {"a": 0.5, "b": 0.5},
    )
    if events is not None:
        pipe.add_op(_ObserveExhaustion(events))
    return pipe.options(
        runner=runner,
        mtp_mode=mtp_mode,
        deterministic=True,
        canonical_replicas=2,
        max_workers=2,
        default_stage_prefetch=0,
        flush_every_k_chunks=2,
    )


def _make_pipe(
    *,
    limit: int | None = None,
) -> Pipeline:
    datasets = [
        Dataset.from_dict(
            name, {0: InMemoryShard([{"text": f"{name}-{i}"} for i in range(count)])}
        )
        for name, count in [("a", 3), ("b", 100)]
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
    return pipe.ensure_mixture(
        weight_by="samples",
        mixture={"a": 0.5, "b": 0.5},
        max_buffer_size=limit,
    ).options(
        runner="inline",
        deterministic=True,
        canonical_replicas=1,
        max_workers=1,
        default_stage_prefetch=2,
        flush_every_k_chunks=128,
    )


def _value(record: SampleRecord) -> tuple[int, str]:
    assert isinstance(record, SampleRecord)
    assert not record.meta.is_sentinel
    assert isinstance(record.payload, dict)
    text = record.payload["text"]
    assert isinstance(text, str)
    return record.meta.lane_id, text


def _per_lane(items: list[tuple[int, str]]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = defaultdict(list)
    for lane, text in items:
        out[lane].append(text)
    return out


@pytest.mark.parametrize("source_targets", [False, True])
def test_notification_is_once_per_lane_across_epoch_resets(
    source_targets: bool,
) -> None:
    events: list[tuple[str, int, int]] = []
    records = [
        _value(r)
        for r in _continuing_pipe(events=events, source_targets=source_targets)
    ]
    assert sorted(
        (lane, cid) for event, lane, cid in events if event == "exhausted"
    ) == [(0, 0), (1, 0)]
    for lane in (0, 1):
        after = events[events.index(("exhausted", lane, 0)) + 1 :]
        assert sum(event == "flush" and lid == lane for event, lid, _ in after) >= 2
        assert {
            cid for event, lid, cid in after if event == "data" and lid == lane
        } >= {4, 5, 6, 7}
    # Every B record survives across the two full post-exhaustion epochs.
    assert Counter(lane for lane, text in records if text.startswith("b-")) == {
        0: 96,
        1: 96,
    }
    # Source-published removal makes the last A chunk obsolete before its EOD.
    expected_a = 25 if source_targets else 32
    assert Counter(lane for lane, text in records if text.startswith("a-")) == {
        0: expected_a,
        1: expected_a,
    }


@pytest.mark.parametrize(
    "runner,mtp_mode,source_targets",
    [
        pytest.param("inline", False, False, id="inline-override"),
        pytest.param("inline", False, True, id="inline-source-target"),
        pytest.param("threads", False, False, id="threads"),
        pytest.param("process", False, False, id="process"),
        pytest.param("inline", True, False, id="mtp"),
    ],
)
def test_source_exhaustion_checkpoint_replays_with_upstream_buffers(
    runner: str,
    mtp_mode: bool,
    source_targets: bool,
) -> None:
    # Cover both target policies inline. Transport cases use an override so
    # releasing B requires the EOD signal to traverse the real runner/ACK path.
    options: dict[str, Any] = dict(
        runner=runner, mtp_mode=mtp_mode, shuffle=True, source_targets=source_targets
    )
    baseline = [_value(r) for r in _continuing_pipe(**options)]
    assert len(baseline) == len(set(baseline))
    assert Counter(lane for lane, text in baseline if text.startswith("b-")) == {
        0: 96,
        1: 96,
    }
    # The source has four complete B-only chunks after exhaustion; cut during
    # that suffix, with more B-only chunks and an epoch boundary still ahead.
    cut = next(
        i + 1
        for i, (_, text) in enumerate(baseline)
        if text.startswith("b-") and int(text[2:]) % _LANE_SIZE >= 4 * _CHUNK_SIZE
    )
    pipe = _continuing_pipe(**options)
    it = iter(pipe)
    try:
        prefix = [_value(next(it)) for _ in range(cut)]
        checkpoint = pipe.checkpoint()
    finally:
        it.close()
    restored = _continuing_pipe(**options)
    restored.restore(checkpoint)
    suffix = [_value(r) for r in restored]
    assert suffix
    assert _per_lane(prefix + suffix) == _per_lane(baseline)
    done = restored.checkpoint()
    final = _continuing_pipe(**options)
    final.restore(done)
    assert list(final) == []


def test_checkpoint_before_notification_replays_later_exhaustion() -> None:
    events: list[tuple[str, int, int]] = []
    baseline = [_value(r) for r in _continuing_pipe(events=[])]
    pipe = _continuing_pipe(events=events)
    it = iter(pipe)
    try:
        prefix = [_value(next(it)) for _ in range(2)]
        assert not any(event == "exhausted" for event, _, _ in events)
        checkpoint = pipe.checkpoint()
    finally:
        it.close()
    restored_events: list[tuple[str, int, int]] = []
    restored = _continuing_pipe(events=restored_events)
    restored.restore(checkpoint)
    assert _per_lane(prefix + [_value(r) for r in restored]) == _per_lane(baseline)
    assert sorted(
        (lane, cid) for event, lane, cid in restored_events if event == "exhausted"
    ) == [(0, 0), (1, 0)]


@pytest.mark.parametrize("limit", [None, 64])
def test_large_shuffle_handles_exhausted_tail_by_mode(limit: int | None) -> None:
    """Notifications may pass most of A while its records remain in shuffle."""
    datasets = [
        Dataset.from_dict(
            name, {0: InMemoryShard([{"text": f"{name}-{i}"} for i in range(count)])}
        )
        for name, count in (("a", 200), ("b", 400))
    ]
    source = _ContinuingSource(
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
    assert len(values) == len(set(values))
    counts = Counter(text.split("-")[0] for _, text in values)
    assert counts["b"] == 200
    if limit is None:
        # Strict mode interleaves what fits, then drops the withheld A tail.
        assert 0 < counts["a"] < 200
    else:
        assert counts["a"] == 200


@pytest.mark.parametrize("limit", [None, 8, 64])
def test_stop_policy_keeps_strict_and_bounded_behavior(limit: int | None) -> None:
    # Stop policy does not opt into target relaxation. The two bounded limits
    # cover preservation under buffer pressure and during the terminal flush.
    records = [_value(r) for r in _make_pipe(limit=limit)]
    assert len(records) == len(set(records))
    if limit is None:
        assert Counter(text.split("-")[0] for _, text in records) == {"a": 3, "b": 3}
    else:
        assert Counter(text.split("-")[0] for _, text in records) == {"a": 3, "b": 57}


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_static_redistribution_with_buffering_and_checkpoint(
    strict: bool, override: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Real survivor targets and early EOD compose through shuffle, correction and replay."""

    def make_pipe() -> Pipeline:
        datasets = [
            Dataset.from_dict(
                name, {0: InMemoryShard([{"text": f"{name}-{i}"} for i in range(size)])}
            )
            for name, size in (("a", 3), ("b", 29))
        ]
        source = StaticMixtureWorkSource(
            datasets,
            {"a": 0.5, "b": 0.5},
            chunk_size=4,
            exhausted_policy="redistribute",
            shuffle_shards=False,
        )
        return (
            Pipeline(source)
            .fetch(max_batch=4)
            .shuffle(buffer_size=7, seed=4)
            .ensure_mixture(
                weight_by="samples",
                max_buffer_size=None if strict else 16,
                mixture={"a": 0.5, "b": 0.5} if override else None,
            )
            .options(
                runner="inline",
                canonical_replicas=1,
                max_workers=1,
                default_stage_prefetch=0,
                flush_every_k_chunks=2,
            )
        )

    with caplog.at_level(logging.DEBUG, logger="zephon._internal.ops.ensure_mixture"):
        baseline = [r.payload["text"] for r in make_pipe()]
    assert len(baseline) == len(set(baseline))
    counts = Counter(text.split("-")[0] for text in baseline)
    discarded_obsolete = sum(
        record.args[2]
        for record in caplog.records
        if record.name == "zephon._internal.ops.ensure_mixture"
        and record.msg.startswith("EnsureMixture strict mode discarded")
    )
    if strict:
        # Two A records miss their drain slots. The final B EOD must keep B
        # eligible without admitting the earlier retired A tail.
        assert counts == {"a": 1, "b": 29}
        assert discarded_obsolete == 2
    else:
        assert counts == {"a": 3, "b": 29}
        assert discarded_obsolete == 0
        assert set(baseline) == {
            f"{name}-{i}" for name, size in (("a", 3), ("b", 29)) for i in range(size)
        }
    pipe = make_pipe()
    iterator = iter(pipe)
    try:
        prefix = [next(iterator).payload["text"] for _ in range(3)]
        checkpoint = pipe.checkpoint()
    finally:
        iterator.close()
    restored = make_pipe()
    restored.restore(checkpoint)
    assert prefix + [r.payload["text"] for r in restored] == baseline
    finished = make_pipe()
    finished.restore(restored.checkpoint())
    assert list(finished) == []
