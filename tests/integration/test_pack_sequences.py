# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end sequence packing, batching, and training conversion."""

from itertools import groupby
from typing import Any, get_args

import numpy as np
import pytest

from zephon.io import Dataset, InMemoryShard
from zephon.ops import PackingAlgorithm
from zephon.pipeline import Pipeline
from zephon.types import SampleBatch
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration

_PACKING_ALGORITHMS: tuple[PackingAlgorithm, ...] = get_args(PackingAlgorithm)


# Run the semantic matrix inline; checkpoint and runner parity have separate tests.
_MATRIX_SEQS = [[10, 11, 12], [20, 21], [30, 31, 32, 33], [40]]
_MATRIX_MAXLEN = 4


def _matrix_work(*, supervised: bool = False) -> StaticMixtureWorkSource:
    rows = []
    for seq in _MATRIX_SEQS:
        row = {"input_ids": list(seq)}
        if supervised:
            row["loss_mask"] = [int(i in (1, 2)) for i in range(len(seq))]
        rows.append(row)
    ds = Dataset.from_dict("m", {0: InMemoryShard(rows)})
    return StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=2,
        seed=42,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )


@pytest.mark.parametrize("algorithm", _PACKING_ALGORITHMS)
def test_pack_envelope_matrix_end_to_end(
    algorithm: PackingAlgorithm,
) -> None:
    """Envelope output across the algorithm axis: each record is a list of
    per-segment dicts carrying the token field (boundaries preserved)."""
    pipeline = Pipeline(_matrix_work())
    pipeline.pack_sequences(
        max_length=_MATRIX_MAXLEN,
        num_bins=8 if algorithm in ("first_fit", "best_fit") else None,
        algorithm=algorithm,
    )
    pipeline.options(deterministic=True, max_workers=1, runner="inline")

    records = list(pipeline)
    assert records
    for rec in records:
        assert set(rec.payload) == {"packed_samples"}
        segs = rec.payload["packed_samples"]
        assert isinstance(segs, list) and segs
        assert all("input_ids" in seg for seg in segs)


# Backend/layout combinations are covered by conversion unit tests; these cases
# exercise the distinct packing and supervision behaviors through the pipeline.
@pytest.mark.parametrize(
    ("algorithm", "framework", "supervised", "emit_positions", "flatten"),
    [
        pytest.param(
            "first_fit", "torch", True, True, False, id="first-fit-sft-batched"
        ),
        pytest.param("best_fit", "numpy", True, True, True, id="best-fit-sft-flat"),
        pytest.param("wrap", "torch", True, True, True, id="wrap-sft-fragments"),
        pytest.param(
            "best_fit_wrap", None, True, True, False, id="best-fit-wrap-sft-lists"
        ),
        pytest.param(
            "first_fit", "numpy", False, True, False, id="pretraining-padding"
        ),
        pytest.param(
            "best_fit_wrap", "torch", False, False, False, id="missing-positions"
        ),
    ],
)
def test_pack_flat_to_training_end_to_end(
    algorithm: PackingAlgorithm,
    emit_positions: bool,
    supervised: bool,
    framework: str | None,
    flatten: bool,
) -> None:
    """Packing and batching preserve document boundaries and label supervision."""
    module = pytest.importorskip(framework) if framework is not None else None
    dtype = module.int64 if module is not None else None
    pipeline = Pipeline(_matrix_work(supervised=supervised))
    pipeline.pack_flat(
        max_length=_MATRIX_MAXLEN,
        num_bins=8 if algorithm in ("first_fit", "best_fit") else None,
        algorithm=algorithm,
        pad_token_id=-1,
        emit_positions=emit_positions,
    )
    pipeline.batch(2, drop_last=False)
    pipeline.options(deterministic=True, max_workers=1, runner="inline")

    batches = list(pipeline)
    # Ten source tokens make two full wrap bins; the other algorithms also
    # deliver a padded third bin and its incomplete final training batch.
    assert [len(batch.records) for batch in batches] == (
        [2] if algorithm == "wrap" else [2, 1]
    )
    expected_keys = {"input_ids", "positions"} if emit_positions else {"input_ids"}
    if supervised:
        expected_keys.add("loss_mask")
    for batch in batches:
        assert isinstance(batch, SampleBatch)
        for rec in batch.records:
            assert isinstance(rec.payload, dict)
            payload: dict[str, Any] = rec.payload
            assert set(payload) == expected_keys
            assert len(payload["input_ids"]) == _MATRIX_MAXLEN
            assert rec.meta.padding_length == list(payload["input_ids"]).count(-1)
            if emit_positions:
                assert len(payload["positions"]) == _MATRIX_MAXLEN
                assert payload["positions"][0] == 0

        reference = _training_reference(batch, supervised=supervised)
        out = batch.to_training(
            dtype=dtype,
            return_labels=True,
            return_loss_mask=True,
            return_cu_seqlens=emit_positions,
            return_num_valid_tokens=True,
            ignore_index=-777,
            flatten=flatten,
            rename_fields={"input_ids": "tokens"},
            exclude_fields=("ids", "texts"),
        )
        expected_fields = {"tokens", "labels", "loss_mask", "num_valid_tokens"}
        if emit_positions:
            expected_fields.update(("positions", "cu_seqlens", "max_seqlen"))
        assert set(out) == expected_fields
        for key in ("tokens", "labels", "loss_mask", "positions"):
            if key not in out:
                continue
            expected = reference[key]
            if flatten:
                expected = [value for row in expected for value in row]
            np.testing.assert_array_equal(out[key], expected)
        assert out["num_valid_tokens"] == sum(map(sum, reference["loss_mask"]))
        if module is not None:
            assert out["loss_mask"].dtype == module.float32
        if emit_positions:
            boundaries = reference["cu_seqlens"]
            if flatten:
                width = _MATRIX_MAXLEN - 1
                expected_cu = [
                    i * width + start
                    for i, row in enumerate(boundaries)
                    for start in row[:-1]
                ] + [len(boundaries) * width]
                assert type(out["max_seqlen"]) is int
                assert out["max_seqlen"] == max(reference["max_seqlen"])
            else:
                boundary_count = max(map(len, boundaries))
                expected_cu = [
                    row + [row[-1]] * (boundary_count - len(row)) for row in boundaries
                ]
                np.testing.assert_array_equal(
                    out["max_seqlen"], reference["max_seqlen"]
                )
                if module is not None:
                    assert out["max_seqlen"].dtype == module.int32
            np.testing.assert_array_equal(out["cu_seqlens"], expected_cu)
            if module is not None:
                assert out["cu_seqlens"].dtype == module.int32
        else:
            with pytest.raises(ValueError, match="positions"):
                batch.to_training(dtype=dtype, return_cu_seqlens=True, flatten=flatten)


def test_pack_flat_eos_loss_and_continuous_positions() -> None:
    """Model positions and EOS loss policy leave packed attention boundaries intact."""
    torch = pytest.importorskip("torch")
    dataset = Dataset.from_dict(
        "docs",
        {0: InMemoryShard([{"tokens": [1, 11, 2]}, {"tokens": [1, 22, 2]}])},
    )
    work = StaticMixtureWorkSource(
        [dataset],
        {"docs": 1.0},
        chunk_size=2,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipeline = Pipeline(work)
    pipeline.pack_flat(max_length=6, algorithm="wrap", emit_positions=True)
    pipeline.batch(1)
    pipeline.options(deterministic=True, max_workers=1, runner="inline")

    batches = list(pipeline)
    assert len(batches) == 1
    out = batches[0].to_training(
        dtype=torch.long,
        return_labels=True,
        return_loss_mask=True,
        return_cu_seqlens=True,
        return_num_valid_tokens=True,
        eos_mask_loss=True,
        eos_token_id=2,
        position_mode="sequence",
    )
    assert out["input_ids"].tolist() == [[1, 11, 2, 1, 22]]
    assert out["labels"].tolist() == [[11, 2, -100, 22, 2]]
    assert out["loss_mask"].tolist() == [[1.0, 1.0, 0.0, 1.0, 1.0]]
    assert out["positions"].tolist() == [[0, 1, 2, 3, 4]]
    assert out["cu_seqlens"].tolist() == [[0, 3, 5]]
    assert out["max_seqlen"].tolist() == [3]
    assert out["num_valid_tokens"] == 4


def _training_reference(batch: SampleBatch, *, supervised: bool) -> dict[str, Any]:
    """Use source document identities and supervision, not emitted positions/masks."""
    source = {
        token: (doc, i in (1, 2))
        for doc, seq in enumerate(_MATRIX_SEQS)
        for i, token in enumerate(seq)
    }
    source[-1] = (-1, False)
    expected: dict[str, Any] = {
        key: []
        for key in (
            "tokens",
            "labels",
            "loss_mask",
            "positions",
            "cu_seqlens",
            "max_seqlen",
        )
    }
    for rec in batch.records:
        assert isinstance(rec.payload, dict)
        payload: dict[str, Any] = rec.payload
        tokens = list(payload["input_ids"])
        inputs = tokens[:-1]
        loss_mask = [
            float(token != -1 and (not supervised or source[token][1]))
            for token in tokens[1:]
        ]
        labels = [token if keep else -777 for token, keep in zip(tokens[1:], loss_mask)]
        # Group by original document identity; a wrap fragment starts a new row
        # segment, and padding is its own segment. The shifted-away token is absent.
        lengths = [
            len(list(group))
            for _, group in groupby(inputs, key=lambda token: source[token][0])
        ]
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        expected["tokens"].append(inputs)
        expected["labels"].append(labels)
        expected["loss_mask"].append(loss_mask)
        expected["positions"].append([i for length in lengths for i in range(length)])
        expected["cu_seqlens"].append(boundaries)
        expected["max_seqlen"].append(max(lengths))
    return expected


@pytest.mark.parametrize("algorithm", _PACKING_ALGORITHMS)
def test_pack_flat_equals_flatten_envelope_end_to_end(
    algorithm: PackingAlgorithm,
) -> None:
    """flat tokens (minus pad) reconstruct the envelope's concatenated segment
    tokens for the same input and algorithm — flat is a serialization of the
    same packing, not a different one."""

    def env_tokens() -> list[int]:
        p = Pipeline(_matrix_work())
        p.pack_sequences(
            max_length=_MATRIX_MAXLEN,
            num_bins=8 if algorithm in ("first_fit", "best_fit") else None,
            algorithm=algorithm,
        )
        p.options(deterministic=True, max_workers=1, runner="inline")
        return [
            t
            for rec in p
            for seg in rec.payload["packed_samples"]
            for t in seg["input_ids"]
        ]

    def flat_tokens() -> list[int]:
        p = Pipeline(_matrix_work())
        p.pack_flat(
            max_length=_MATRIX_MAXLEN,
            num_bins=8 if algorithm in ("first_fit", "best_fit") else None,
            algorithm=algorithm,
            pad_token_id=-1,
        )
        p.options(deterministic=True, max_workers=1, runner="inline")
        return [t for rec in p for t in rec.payload["input_ids"] if t != -1]

    assert flat_tokens() == env_tokens()


def _capped_work() -> StaticMixtureWorkSource:
    """Dataset large enough to fan many bins across worker threads."""
    seqs = [list(range(i, i + (i % 7) + 1)) for i in range(200)]
    ds = Dataset.from_dict(
        "big", {0: InMemoryShard([{"input_ids": seq} for seq in seqs])}
    )
    return StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=8,
        seed=7,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )


def _norm_payload(payload: Any) -> Any:
    """Normalize a packed payload for equality (numpy arrays -> lists)."""

    def norm(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k: norm(x) for k, x in v.items()}
        if isinstance(v, list):
            return [norm(x) for x in v]
        return v

    return norm(payload)


@pytest.mark.parametrize("algorithm", _PACKING_ALGORITHMS)
@pytest.mark.parametrize("output", ["envelope", "flat"], ids=["envelope", "flat"])
def test_capped_packing_parallelism_invariant(
    algorithm: PackingAlgorithm, output: str
) -> None:
    """parallelism>1 produces identical payloads, order, and lineage as
    parallelism=1: bin assignment stays serial, only materialization fans out."""

    def run(parallelism: int) -> list[tuple[Any, Any]]:
        p = Pipeline(_capped_work())
        if output == "flat":
            p.pack_flat(
                max_length=8,
                num_bins=8 if algorithm in ("first_fit", "best_fit") else None,
                algorithm=algorithm,
                pad_token_id=-1,
                max_sequences_per_bin=2,
                parallelism=parallelism,
            )
        else:
            p.pack_sequences(
                max_length=8,
                num_bins=8 if algorithm in ("first_fit", "best_fit") else None,
                algorithm=algorithm,
                parallelism=parallelism,
                max_sequences_per_bin=2,
            )
        p.options(deterministic=True, runner="threads", default_stage_prefetch=16)
        return [(r.meta.cursor.as_key(), _norm_payload(r.payload)) for r in p]

    serial = run(1)
    assert serial
    assert run(4) == serial


@pytest.mark.parametrize("algorithm", _PACKING_ALGORITHMS)
@pytest.mark.parametrize("runner", ["inline", "threads", "process"])
def test_capped_training_microbatch_checkpoint_resume(
    algorithm: PackingAlgorithm, runner: str
) -> None:
    """Count-limited bins preserve the full training contract through replay."""

    def make_pipeline() -> Pipeline:
        pipe = Pipeline(_capped_work())
        pipe.pack_flat(
            max_length=9,
            num_bins=4 if algorithm in ("first_fit", "best_fit") else None,
            algorithm=algorithm,
            max_sequences_per_bin=2,
            pad_token_id=0,
            candidate_pool_size=8 if algorithm == "best_fit_wrap" else None,
            parallelism=2,
        ).batch(3)
        return pipe.options(
            deterministic=True,
            runner=runner,
            mtp_mode=False,
            canonical_replicas=2,
            # These tiny batches test packing/replay, not shared-memory transport.
            coalesce_tensors=False,
        )

    def convert(batch: SampleBatch) -> tuple[Any, Any]:
        training = batch.to_training(
            dtype=None,
            return_labels=True,
            return_loss_mask=True,
            return_padding_mask=True,
            return_num_valid_tokens=True,
            return_cu_seqlens=True,
            flatten=True,
            rename_fields={"input_ids": "input"},
            exclude_fields=("ids", "texts"),
        )
        assert len(training["input"]) == len(training["padding_mask"]) == 24
        assert (
            sum(
                p == 0 and not pad
                for p, pad in zip(training["positions"], training["padding_mask"])
            )
            <= 6
        )
        assert training["num_valid_tokens"] == sum(training["loss_mask"])
        assert training["cu_seqlens"][-1] == 24
        assert all(
            r.meta.tags["_packing_metadata"]["num_sequences"] <= 2
            for r in batch.records
        )
        return tuple(r.meta.cursor.as_key() for r in batch.records), training

    baseline = [convert(batch) for batch in make_pipeline()]
    assert len(baseline) > 6
    first = make_pipeline()
    iterator = iter(first)
    try:
        prefix = [convert(next(iterator)) for _ in range(6)]
        checkpoint = first.checkpoint()
    finally:
        iterator.close()
    resumed = make_pipeline()
    resumed.restore(checkpoint)
    suffix = [convert(batch) for batch in resumed]
    assert prefix + suffix == baseline
    assert resumed._engine is not None
    assert not any(resumed._engine.inflight_chunks_per_lane.values())
