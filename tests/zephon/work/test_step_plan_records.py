# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Step plans run only pipelines that keep one record per source record."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import pytest

from zephon import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.ops import BaseOp, OpTraits, PassthroughAccumulator
from zephon.types import SampleBatch
from zephon.work import (
    MixturePhase,
    MixtureSchedule,
    PhaseConstantSchedule,
    StaticMixtureWorkSource,
    StepMixturePlan,
)

# All-a for steps 0-3, all-b from step 4; two records per step and four per
# chunk round, so the switch is exact at record 8.
_PLAN = StepMixturePlan(
    phases=[
        MixturePhase(0, {"a": 1.0, "b": 0.0}),
        MixturePhase(4, {"a": 0.0, "b": 1.0}),
    ],
    total_steps=8,
)
_OPTIONS = {"canonical_replicas": 1, "global_batch_size": 2, "mtp_mode": False}
_EXPECTED = ["a"] * 8 + ["b"] * 8


def _work(mixture: Any = _PLAN) -> StaticMixtureWorkSource:
    datasets = [
        Dataset.from_dict(
            name, {0: InMemoryShard([{"name": name, "i": i} for i in range(64)])}
        )
        for name in ("a", "b")
    ]
    return StaticMixtureWorkSource(
        datasets,
        mixture,
        chunk_size=4,
        seed=0,
        shuffle_shards=False,
        exhausted_policy="repeat",
    )


def _names(pipe: Pipeline, batches: int = 8) -> list[str]:
    return [
        record.payload["name"]
        for batch in itertools.islice(pipe, batches)
        for record in batch.records
    ]


def _drop_odd_a(payload: dict[str, Any]) -> dict[str, Any] | None:
    return None if payload["name"] == "a" and payload["i"] % 2 else payload


def _passthrough(elems: list[Any]) -> list[Any]:
    return elems


@pytest.mark.parametrize(
    "ops",
    [
        lambda p: p.map_transform(lambda r: {**r, "seen": True}).batch(2),
        lambda p: p.map_transform(_drop_odd_a, drop_none=False).batch(2),
        lambda p: p.batch(2).map_batch(lambda b: SampleBatch(records=b.records)),
        lambda p: p.add_op(
            "custom",
            process_many=_passthrough,
            preserves_cursor_order=True,
            one_to_one_records=True,
        ).batch(2),
    ],
    ids=["map", "map-keeping-none", "map-batch", "declared-custom-op"],
)
def test_one_to_one_pipelines_keep_exact_transition_timing(
    ops: Callable[[Pipeline], Pipeline],
) -> None:
    assert _names(ops(Pipeline(_work())).options(**_OPTIONS)) == _EXPECTED


def test_filtering_map_raises_instead_of_shifting_steps() -> None:
    pipe = Pipeline(_work()).map_transform(_drop_odd_a).batch(2).options(**_OPTIONS)
    with pytest.raises(ValueError, match="map_transform dropped a record at lane 0"):
        _names(pipe)


def _shrink_in_place(batch: SampleBatch) -> None:
    object.__setattr__(batch, "records", batch.records[:1])


@pytest.mark.parametrize(
    "transform,drop_none,message",
    [
        (lambda b: None, True, "map_batch dropped a batch"),
        (
            lambda b: SampleBatch(records=b.records[:1]),
            True,
            "batch of 2 records into 1 different",
        ),
        (
            lambda b: SampleBatch(records=b.records + b.records[:1]),
            True,
            "batch of 2 records into 3 different",
        ),
        (
            lambda b: SampleBatch(records=(b.records[0], b.records[0])),
            True,
            "batch of 2 records into 2 different",
        ),
        (lambda b: list(b.records), True, "into a non-batch"),
        (_shrink_in_place, False, "batch of 2 records into 1 different"),
    ],
    ids=["drop", "shrink", "expand", "duplicate", "non-batch", "mutate-in-place"],
)
def test_map_batch_must_keep_every_record(
    transform: Callable[[SampleBatch], Any], drop_none: bool, message: str
) -> None:
    pipe = (
        Pipeline(_work())
        .batch(2)
        .map_batch(transform, drop_none=drop_none)
        .options(**_OPTIONS)
    )
    with pytest.raises(ValueError, match=message):
        _names(pipe)


def test_custom_op_must_declare_one_to_one_records() -> None:
    undeclared = (
        Pipeline(_work())
        .add_op("custom", process_many=_passthrough, preserves_cursor_order=True)
        .batch(2)
        .options(**_OPTIONS)
    )
    with pytest.raises(ValueError, match=r"may drop, add, split, or merge.*'custom'"):
        _names(undeclared)


class _UndeclaredOp(BaseOp):
    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True)

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> PassthroughAccumulator[Any]:
        return PassthroughAccumulator()

    def process_many(self, elems: list[Any]) -> list[Any]:
        return elems


def test_custom_op_instance_without_the_trait_is_rejected() -> None:
    pipe = Pipeline(_work()).add_op(_UndeclaredOp()).batch(2).options(**_OPTIONS)
    with pytest.raises(ValueError, match="_UndeclaredOp"):
        _names(pipe)


def test_chunk_round_schedules_keep_filtering() -> None:
    schedule = MixtureSchedule(
        {
            "a": PhaseConstantSchedule([(0, 1.0), (2, 0.0)]),
            "b": PhaseConstantSchedule([(0, 0.0), (2, 1.0)]),
        }
    )
    pipe = (
        Pipeline(_work(schedule))
        .map_transform(_drop_odd_a)
        .batch(2)
        .options(canonical_replicas=1, mtp_mode=False)
    )
    # Half of the two all-a rounds is filtered out, so b arrives earlier.
    assert _names(pipe, 4) == ["a"] * 4 + ["b"] * 4


def test_reused_source_still_checks_records() -> None:
    work = _work()
    assert _names(Pipeline(work).batch(2).options(**_OPTIONS)) == _EXPECTED
    pipe = Pipeline(work).map_transform(_drop_odd_a).batch(2).options(**_OPTIONS)
    with pytest.raises(ValueError, match="map_transform dropped a record"):
        _names(pipe)


def test_mid_chunk_restore_passes_replay_tombstones() -> None:
    def build() -> Pipeline:
        return (
            Pipeline(_work())
            .map_transform(lambda p: {**p, "seen": True})
            .batch(2)
            .options(**_OPTIONS)
        )

    first = build()
    iterator = iter(first)
    # Three batches end mid-way through the second chunk.
    prefix = [
        r.payload["name"] for b in itertools.islice(iterator, 3) for r in b.records
    ]
    state = first.checkpoint()
    iterator.close()

    restored = build()
    restored.restore(state)
    assert prefix + _names(restored, 5) == _EXPECTED
