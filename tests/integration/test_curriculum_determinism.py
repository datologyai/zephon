# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Curriculum schedules produce the same stream under any execution config.

Every test compares a candidate run against an inline, unprefetched,
single-worker reference built from the same datasets, seed, schedule,
``chunk_size`` and ``canonical_replicas``.
"""

from __future__ import annotations

import functools
import hashlib
import random
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from zephon import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.types import SampleBatch, SampleRecord
from zephon.work import (
    MixtureSchedule,
    PhaseConstantSchedule,
    PhaseLinearSchedule,
    StaticMixtureWorkSource,
    TokenEstimation,
)

# Several tests cut checkpoints mid-window on purpose.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore:.*Checkpoint taken mid-window.*:RuntimeWarning"),
]

CANONICAL_REPLICAS = 4
CHUNK_SIZE = 8
ROUND_SIZE = CANONICAL_REPLICAS * CHUNK_SIZE
NAMES = ("alpha", "beta", "gamma")
ROUNDS_TO_CONSUME = 13
RECORDS_TO_CONSUME = ROUNDS_TO_CONSUME * ROUND_SIZE

APIS = ["chunk_rounds"]

INLINE: dict[str, Any] = {}
THREADS_RACY: dict[str, Any] = {
    "runner": "threads",
    "max_workers": 8,
    "default_stage_prefetch": 16,
    "prefetch_batches": 64,
    "op_queue_capacity": 1,
}
THREADS_SMALL: dict[str, Any] = {
    "runner": "threads",
    "max_workers": 3,
    "default_stage_prefetch": 1,
    "prefetch_batches": 1,
}
PROCESS: dict[str, Any] = {
    "runner": "process",
    "max_workers": 3,
    "default_stage_prefetch": 4,
    "prefetch_batches": 8,
}
MTP: dict[str, Any] = {
    "mtp_mode": True,
    "max_workers": 8,
    "default_stage_prefetch": 8,
    "prefetch_batches": 32,
}

RecordKey = tuple[int, int, int, tuple[int, ...], tuple[tuple[int, int], ...], str]


def _make_dataset(name: str, sample_count: int = 4_096, shards: int = 4) -> Dataset:
    per_shard = sample_count // shards
    return Dataset.from_dict(
        name,
        {
            shard: InMemoryShard(
                [
                    {
                        "text": f"{name}-{shard * per_shard + i}",
                        "n": shard * per_shard + i,
                    }
                    for i in range(per_shard)
                ]
            )
            for shard in range(shards)
        },
    )


def _default_datasets() -> list[Dataset]:
    return [_make_dataset(name) for name in NAMES]


def _chunk_round_schedule() -> MixtureSchedule:
    # Changes at rounds 2, 5, 9 and 12, with a ramp over rounds 3..7.
    return MixtureSchedule(
        {
            "alpha": PhaseLinearSchedule([(0, 0.7), (3, 0.7), (7, 0.1), (12, 0.4)]),
            "beta": PhaseConstantSchedule([(0, 0.2), (2, 0.5), (9, 0.1)]),
            "gamma": PhaseConstantSchedule([(0, 0.1), (5, 0.6), (12, 0.2)]),
        }
    )


def _jitter(payload: Any) -> Any:
    # Heavy-tailed, unseeded sleeps make workers and lanes race.
    time.sleep(random.random() ** 3 * 0.002)
    return payload


def _drop_every_fifth(payload: Any) -> Any:
    if payload["n"] % 5 == 0:
        return None
    return _jitter(payload)


def _with_tokens(payload: Any) -> Any:
    return {"tokens": [payload["n"]] * (payload["n"] % 13 + 1)}


def _racy_map(pipe: Pipeline) -> Pipeline:
    return pipe.map_transform(_jitter, parallelism=8)


def _build(
    api: str,
    *,
    options: dict[str, Any] | None = None,
    ops: Callable[[Pipeline], Pipeline] = _racy_map,
    batch: int | None = 1,
    datasets: Callable[[], list[Dataset]] = _default_datasets,
    work_kwargs: dict[str, Any] | None = None,
) -> Pipeline:
    work_kwargs = {
        "seed": 7,
        "shuffle_shards": True,
        "shuffle_within_shard": True,
        **(work_kwargs or {}),
    }
    assert api == "chunk_rounds"
    work_source = StaticMixtureWorkSource(
        datasets(), _chunk_round_schedule(), chunk_size=CHUNK_SIZE, **work_kwargs
    )
    pipe = ops(Pipeline(work_source))
    if batch is not None:
        pipe = pipe.batch(batch)
    return pipe.options(
        **{
            "deterministic": True,
            "canonical_replicas": CANONICAL_REPLICAS,
            "default_stage_prefetch": 0,
            "prefetch_batches": 0,
            "max_workers": 1,
            "mtp_mode": False,
            **(options or {}),
        }
    )


def _rank_options(dp_degree: int, rank: int, aggregate_dir: Path) -> dict[str, Any]:
    return {
        "world_size": dp_degree,
        "global_rank": rank,
        "dp_degree": dp_degree,
        "dp_group_id": rank,
        "aggregate_dir": str(aggregate_dir),
        "mtp_auto_checkpoint": False,
    }


def _record_key(record: SampleRecord) -> RecordKey:
    payload = record.payload
    assert isinstance(payload, dict)
    if "text" in payload:
        content = str(payload["text"])
    else:
        content = hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()
    return (
        int(record.meta.lane_id),
        int(record.meta.chunk_id),
        int(record.meta.chunk_offset),
        tuple(int(part) for part in record.meta.sample_id),
        tuple(sorted(record.meta.component_sample_counts.items())),
        content,
    )


def _consume(
    pipe: Pipeline,
    limit: int | None,
    *,
    checkpoint: bool = False,
    consumer_sleep_s: float = 0.0,
    timeout_s: float = 120.0,
) -> tuple[list[RecordKey], dict[str, Any] | None]:
    """Pull ``limit`` pipeline items (``None`` drains), flattening batches."""
    deadline = time.monotonic() + timeout_s
    pause = random.Random(0)
    records: list[RecordKey] = []
    state = None
    iterator = iter(pipe)
    try:
        pulled = 0
        while limit is None or pulled < limit:
            assert time.monotonic() < deadline, "pipeline consumption timed out"
            try:
                item = next(iterator)
            except StopIteration:
                break
            pulled += 1
            batch = item.records if isinstance(item, SampleBatch) else [item]
            for record in batch:
                assert isinstance(record, SampleRecord)
                records.append(_record_key(record))
            if consumer_sleep_s:
                time.sleep(pause.random() * consumer_sleep_s)
        if checkpoint:
            state = pipe.checkpoint()
    finally:
        iterator.close()
    return records, state


def _consume_checkpoint(
    pipe: Pipeline, limit: int
) -> tuple[list[RecordKey], dict[str, Any]]:
    records, state = _consume(pipe, limit, checkpoint=True)
    assert state is not None
    return records, state


def _consume_ranks(
    builders: list[Callable[[], Pipeline]],
    limit: int | None,
    *,
    checkpoint: bool = False,
) -> list[tuple[list[RecordKey], dict[str, Any] | None]]:
    # Ranks run concurrently so their collective checkpoint can aggregate.
    results: list[Any] = [None] * len(builders)
    errors: list[BaseException] = []

    def run(rank: int) -> None:
        try:
            results[rank] = _consume(builders[rank](), limit, checkpoint=checkpoint)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(rank,), daemon=True)
        for rank in range(len(builders))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=150)
    if errors:
        raise errors[0]
    assert not any(thread.is_alive() for thread in threads), "rank thread hung"
    return results


def _by_lane(records: list[RecordKey]) -> dict[int, list[RecordKey]]:
    lanes: dict[int, list[RecordKey]] = defaultdict(list)
    for record in records:
        lanes[record[0]].append(record)
    return dict(lanes)


def _assert_lane_prefixes(
    got: dict[int, list[RecordKey]], reference: list[RecordKey]
) -> None:
    expected = _by_lane(reference)
    assert set(got) == set(expected)
    for lane, records in got.items():
        assert len(records) <= len(expected[lane])
        assert records == expected[lane][: len(records)], f"lane {lane} diverged"


def _elastic_chain(
    build: Callable[[dict[str, Any]], Pipeline],
    segments: list[tuple[int, int]],
    tmp_path: Path,
) -> list[list[list[RecordKey]]]:
    """Run (dp_degree, items per rank) segments, restoring each from the last."""
    state: dict[str, Any] | None = None
    streams: list[list[list[RecordKey]]] = []
    for index, (dp_degree, budget) in enumerate(segments):

        def builder(
            rank: int, dp_degree: int = dp_degree, index: int = index
        ) -> Callable[[], Pipeline]:
            def run() -> Pipeline:
                options = dict(THREADS_RACY if rank % 2 == 0 else INLINE)
                if dp_degree > 1:
                    options.update(
                        _rank_options(dp_degree, rank, tmp_path / f"segment-{index}")
                    )
                pipe = build(options)
                if state is not None:
                    pipe.restore(state)
                return pipe

            return run

        outputs = _consume_ranks(
            [builder(rank) for rank in range(dp_degree)], budget, checkpoint=True
        )
        streams.append([records for records, _ in outputs])
        state = outputs[0][1]
    return streams


@functools.lru_cache(maxsize=None)
def _reference(api: str, limit: int = RECORDS_TO_CONSUME) -> list[RecordKey]:
    return _consume(_build(api), limit)[0]


# -- execution config -------------------------------------------------------


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize(
    "options,consumer_sleep_s",
    [
        pytest.param(THREADS_RACY, 0.0, id="threads-w8-deep-prefetch-q1"),
        pytest.param(THREADS_SMALL, 0.003, id="threads-w3-slow-consumer"),
        pytest.param(MTP, 0.0, id="mtp-deep-prefetch"),
        pytest.param({**MTP, "runner": "process"}, 0.002, id="mtp-process-slow"),
    ],
)
def test_stream_is_independent_of_execution_config(
    api: str, options: dict[str, Any], consumer_sleep_s: float
) -> None:
    candidate, _ = _consume(
        _build(
            api,
            options=options,
            ops=lambda p: _racy_map(p).map_transform(_jitter, parallelism=3),
        ),
        RECORDS_TO_CONSUME,
        consumer_sleep_s=consumer_sleep_s,
    )
    assert candidate == _reference(api)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize(
    "dp_degree,mapping_strategy",
    [(2, "interleaved"), (4, "contiguous")],
    ids=["dp2-interleaved", "dp4-contiguous"],
)
def test_lane_streams_are_independent_of_dp_degree(
    api: str, dp_degree: int, mapping_strategy: str, tmp_path: Path
) -> None:
    def builder(rank: int) -> Callable[[], Pipeline]:
        return lambda: _build(
            api,
            options={
                **THREADS_RACY,
                **_rank_options(dp_degree, rank, tmp_path),
                "mapping_strategy": mapping_strategy,
            },
        )

    outputs = _consume_ranks(
        [builder(rank) for rank in range(dp_degree)], RECORDS_TO_CONSUME // dp_degree
    )
    merged: dict[int, list[RecordKey]] = {}
    for records, _ in outputs:
        for lane, lane_records in _by_lane(records).items():
            assert lane not in merged
            merged[lane] = lane_records
    assert merged == _by_lane(_reference(api))


# -- checkpoint / resume ----------------------------------------------------


def _cut_points(api: str) -> list[int]:
    transitions = [5 * ROUND_SIZE, 9 * ROUND_SIZE]
    # At, just before, and mid-chunk and mid-round after each transition.
    offsets = (0, -1, CHUNK_SIZE + 1)
    return sorted({t + d for t in transitions for d in offsets})


@pytest.mark.parametrize("api", APIS)
def test_resume_at_transitions_with_changed_execution(api: str) -> None:
    reference = _reference(api)
    for index, cut in enumerate(_cut_points(api)):
        prefix, state = _consume_checkpoint(
            _build(api, options=THREADS_RACY if index % 2 else INLINE), cut
        )
        restored = _build(api, options=INLINE if index % 2 else THREADS_RACY)
        restored.restore(state)
        suffix, _ = _consume(restored, RECORDS_TO_CONSUME - cut)
        assert prefix + suffix == reference, f"cut={cut}"


@pytest.mark.parametrize("api", APIS)
def test_resume_twice_across_runners(api: str) -> None:
    reference = _reference(api)
    first_cut, second_cut = 3 * ROUND_SIZE + 5, 9 * ROUND_SIZE + CHUNK_SIZE + 2
    prefix, state = _consume_checkpoint(_build(api, options=THREADS_RACY), first_cut)
    middle_pipe = _build(api, options=MTP)
    middle_pipe.restore(state)
    middle, state = _consume_checkpoint(middle_pipe, second_cut - first_cut)
    final_pipe = _build(api, options=PROCESS)
    final_pipe.restore(state)
    suffix, _ = _consume(final_pipe, RECORDS_TO_CONSUME - second_cut)
    assert prefix + middle + suffix == reference


@pytest.mark.parametrize("api", APIS)
def test_elastic_resume_chain_keeps_lane_streams(api: str, tmp_path: Path) -> None:
    # (dp_degree, items per rank); budgets leave lanes at unequal frontiers.
    segments = [(1, 61), (2, 37), (4, 23), (1, 101)]
    collected: dict[int, list[RecordKey]] = defaultdict(list)
    for rank_streams in _elastic_chain(
        lambda options: _build(api, options=options), segments, tmp_path
    ):
        for records in rank_streams:
            for lane, lane_records in _by_lane(records).items():
                collected[lane].extend(lane_records)
    # Every lane crossed several schedule transitions.
    assert min(len(records) for records in collected.values()) > 6 * CHUNK_SIZE
    _assert_lane_prefixes(dict(collected), _reference(api))


# -- downstream operators ---------------------------------------------------


def _shuffle(pipe: Pipeline) -> Pipeline:
    return _racy_map(pipe).shuffle(24, seed=3)


def _ensure_mixture(pipe: Pipeline) -> Pipeline:
    return _racy_map(pipe).ensure_mixture(weight_by="samples")


def _pack_sequences(pipe: Pipeline) -> Pipeline:
    return pipe.map_transform(_with_tokens).pack_sequences(
        24, num_bins=3, tokens_field="tokens"
    )


def _pack_wrap(pipe: Pipeline) -> Pipeline:
    return pipe.map_transform(_with_tokens).pack_sequences(
        20, algorithm="wrap", tokens_field="tokens"
    )


def _pack_flat(pipe: Pipeline) -> Pipeline:
    return pipe.map_transform(_with_tokens).pack_flat(
        24, num_bins=2, tokens_field="tokens", pad_token_id=0
    )


def _filter_then_shuffle(pipe: Pipeline) -> Pipeline:
    return pipe.map_transform(_drop_every_fifth, parallelism=8).shuffle(10, seed=1)


def _pack_flush_every(k: int) -> dict[str, Any]:
    return {
        "ops": _pack_sequences,
        "batch": None,
        "options": {"flush_every_k_chunks": k},
    }


@pytest.mark.parametrize(
    "api,variant",
    [
        pytest.param("chunk_rounds", {"ops": _shuffle}, id="shuffle"),
        pytest.param(
            "chunk_rounds", {"ops": _ensure_mixture, "batch": 3}, id="ensure_mixture"
        ),
        pytest.param(
            "chunk_rounds", {"ops": _pack_sequences, "batch": None}, id="pack_sequences"
        ),
        pytest.param("chunk_rounds", _pack_flush_every(1), id="pack-flush-every-1"),
        pytest.param("chunk_rounds", _pack_flush_every(3), id="pack-flush-every-3"),
        pytest.param("chunk_rounds", {"ops": _pack_wrap, "batch": 2}, id="pack_wrap"),
        pytest.param("chunk_rounds", {"ops": _pack_flat, "batch": 3}, id="pack_flat"),
        pytest.param(
            "chunk_rounds",
            {"ops": _filter_then_shuffle, "batch": 5},
            id="filter-shuffle",
        ),
        pytest.param(
            "chunk_rounds",
            {"work_kwargs": {"lane_assignment": "modulo"}},
            id="chunk-modulo",
        ),
    ],
)
def test_variants_are_execution_independent(api: str, variant: dict[str, Any]) -> None:
    extra_options = variant.get("options", {})

    def build(options: dict[str, Any]) -> Pipeline:
        return _build(api, **{**variant, "options": {**options, **extra_options}})

    items = 10 * ROUND_SIZE // (variant.get("batch", 1) or 1)
    reference, _ = _consume(build(INLINE), items)
    candidate, _ = _consume(build(THREADS_RACY), items)
    assert candidate == reference

    cut = items // 2 + 1
    prefix, state = _consume_checkpoint(build(THREADS_RACY), cut)
    restored = build(MTP)
    restored.restore(state)
    suffix, _ = _consume(restored, items - cut)
    assert prefix + suffix == reference


def test_ensure_mixture_follows_the_schedule() -> None:
    records, _ = _consume(_build("chunk_rounds", ops=_ensure_mixture), 16 * ROUND_SIZE)
    by_round: dict[int, Counter[str]] = defaultdict(Counter)
    for _, chunk_id, *_, text in records:
        by_round[chunk_id][text.partition("-")[0]] += 1
    # EnsureMixture holds records back, so only fully delivered rounds count.
    complete = [r for r, counts in by_round.items() if counts.total() == ROUND_SIZE]
    assert set(range(8)) <= set(complete)
    schedule = _chunk_round_schedule()
    for chunk_round in complete:
        counts = by_round[chunk_round]
        for name in NAMES:
            target = schedule.weights_at(chunk_round)[name] * ROUND_SIZE
            assert abs(counts[name] - target) <= CANONICAL_REPLICAS + 1, (
                chunk_round,
                name,
            )


_TOKEN_ESTIMATION = {"token_estimation": TokenEstimation(primer=0.25)}


@pytest.mark.parametrize(
    "api,variant,match",
    [
        pytest.param(
            "chunk_rounds",
            {
                "ops": lambda p: p.ensure_mixture(
                    weight_by="samples",
                    mixture={"alpha": 0.4, "beta": 0.3, "gamma": 0.3},
                )
            },
            "cannot be combined with a curriculum",
            id="ensure-override",
        ),
        *(
            pytest.param(
                api,
                {"work_kwargs": _TOKEN_ESTIMATION},
                "token_estimation is not supported",
                id=f"{api}-token-estimation",
            )
            for api in APIS
        ),
    ],
)
def test_schedules_reject_unsupported_pipelines(
    api: str, variant: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _consume(_build(api, **variant), 1)


# -- exhaustion policies ----------------------------------------------------


def _small_repeat_datasets() -> list[Dataset]:
    return [
        _make_dataset("alpha"),
        _make_dataset("beta", 64, shards=2),
        _make_dataset("gamma", 48, shards=3),
    ]


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("reshuffle", [False, True], ids=["fixed", "reshuffle"])
def test_repeating_datasets_wrap_deterministically(api: str, reshuffle: bool) -> None:
    def build(options: dict[str, Any]) -> Pipeline:
        return _build(
            api,
            options=options,
            datasets=_small_repeat_datasets,
            work_kwargs={
                "exhausted_policy": "repeat",
                "reshuffle_on_repeat": reshuffle,
            },
        )

    reference, _ = _consume(build(INLINE), RECORDS_TO_CONSUME)
    # The small datasets wrap several times inside the consumed range.
    assert (
        max(Counter(text for *_, text in reference if "alpha" not in text).values())
        >= 3
    )
    candidate, _ = _consume(build(THREADS_RACY), RECORDS_TO_CONSUME)
    assert candidate == reference
    for cut in (3 * ROUND_SIZE + 3, 7 * ROUND_SIZE + CHUNK_SIZE + 1):
        prefix, state = _consume_checkpoint(build(THREADS_RACY), cut)
        restored = build(INLINE)
        restored.restore(state)
        suffix, _ = _consume(restored, RECORDS_TO_CONSUME - cut)
        assert prefix + suffix == reference, f"cut={cut}"


def _stop_datasets() -> list[Dataset]:
    return [
        _make_dataset("alpha", 400, 2),
        _make_dataset("beta", 120, 2),
        _make_dataset("gamma", 90, 3),
    ]


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize(
    "policy",
    [
        pytest.param({}, id="stop_after_passes"),
        pytest.param(
            {
                "exhausted_policy": {
                    "alpha": "repeat",
                    "beta": "stop",
                    "gamma": "repeat",
                }
            },
            id="stop-on-beta",
        ),
        pytest.param(
            {
                "exhausted_policy": "repeat",
                "max_repeats": {"alpha": None, "beta": None, "gamma": 2},
            },
            id="max_repeats",
        ),
    ],
)
def test_terminating_policies_end_mid_schedule_deterministically(
    api: str, policy: dict[str, Any], tmp_path: Path
) -> None:
    def build(options: dict[str, Any]) -> Pipeline:
        return _build(api, options=options, datasets=_stop_datasets, work_kwargs=policy)

    reference, _ = _consume(build(INLINE), None)
    # The stream ends after several schedule transitions but before the plan does.
    assert 8 * ROUND_SIZE < len(reference) < 40 * ROUND_SIZE
    candidate, _ = _consume(build(THREADS_RACY), None)
    assert candidate == reference

    outputs = _consume_ranks(
        [
            lambda rank=rank: build(
                {**THREADS_RACY, **_rank_options(2, rank, tmp_path)}
            )
            for rank in range(2)
        ],
        None,
    )
    merged: dict[int, list[RecordKey]] = {}
    for records, _ in outputs:
        merged.update(_by_lane(records))
    assert merged == _by_lane(reference)

    cut = len(reference) * 2 // 3 + 1
    prefix, state = _consume_checkpoint(build(THREADS_RACY), cut)
    restored = build(MTP)
    restored.restore(state)
    suffix, _ = _consume(restored, None)
    assert prefix + suffix == reference


# -- other knobs ------------------------------------------------------------


@pytest.mark.parametrize("api", APIS)
def test_torch_dataloader_workers_keep_the_stream(api: str) -> None:
    torch_data = pytest.importorskip("torch.utils.data")
    limit = 8 * ROUND_SIZE
    loader = torch_data.DataLoader(
        _build(api, options={"default_stage_prefetch": 2}).to_torch_dataset(),
        batch_size=None,
        num_workers=2,
    )
    records: list[RecordKey] = []
    deadline = time.monotonic() + 120
    for item in loader:
        batch = item.records if isinstance(item, SampleBatch) else [item]
        records.extend(_record_key(record) for record in batch)
        if len(records) >= limit or time.monotonic() > deadline:
            break
    del loader
    assert records[:limit] == _reference(api)[:limit]
