# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PackSequences operator.

Packing is two orthogonal axes: the algorithm (``first_fit`` / ``best_fit`` /
``wrap`` / ``best_fit_wrap``) and the output serialization (``envelope`` list
vs ``flat`` training record, with optional ``positions``). Tests are grouped
accordingly, with a final matrix section asserting every algorithm × output
combination and the ``flat == flatten(envelope)`` relationship.
"""

from typing import Any, get_args

import pytest

from zephon.core.constants import (
    ContributorRef,
    SampleBatch,
    SampleCursor,
    SampleMeta,
    SampleRecord,
)
from zephon.ops.grouping import DomainGroups
from zephon.ops.pack_sequences import (
    PackingAccumulator,
    PackingAlgorithm,
    PackSequences,
    Segment,
    _DeferredBin,
    _EnvelopeSerializer,
    _FlatSerializer,
)

_PACKING_ALGORITHMS: tuple[PackingAlgorithm, ...] = get_args(PackingAlgorithm)


def _rec(
    i: int, length: int, *, lane: int = 0, chunk: int = 0, **payload: Any
) -> SampleRecord:
    """A record carrying a precomputed ``length`` field (no token field)."""
    meta = SampleMeta(sample_id=(0, 0, i), lane_id=lane, chunk_id=chunk)
    return SampleRecord(meta=meta, payload={"value": i, "length": length, **payload})


def _rec_tokens(
    i: int,
    tokens: list[int] | int,
    *,
    field: str = "input_ids",
    lane: int = 0,
    chunk: int = 0,
    as_numpy: bool = False,
) -> SampleRecord:
    """A record carrying a token field (list, int-length, or numpy array)."""
    meta = SampleMeta(sample_id=(0, 0, i), lane_id=lane, chunk_id=chunk)
    tok: Any = tokens
    if as_numpy:
        np = pytest.importorskip("numpy")
        if not isinstance(tokens, int):
            tok = np.array(tokens, dtype=np.uint32)
    return SampleRecord(meta=meta, payload={"value": i, field: tok})


def _simple_length_fn(rec: SampleRecord) -> int:
    """Length from the ``length`` payload field (for token-less records)."""
    payload = rec.payload
    return payload.get("length", 0) if isinstance(payload, dict) else 0


class _MaterializingAccumulator:
    """Test shim: real accumulator decides bins; operator materializes payloads."""

    def __init__(self, op: PackSequences, acc: PackingAccumulator) -> None:
        self._op = op
        self._acc = acc

    def _materialize(self, batches: list[Any]) -> list[Any]:
        return [(self._op.process_many(records), wait) for records, wait in batches]

    def push_many(self, elems: Any) -> list[Any]:
        return self._materialize(self._acc.push_many(elems))

    def flush(self, *, reset: bool = False, lane_id: int | None = None) -> list[Any]:
        return self._materialize(self._acc.flush(reset=reset, lane_id=lane_id))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._acc, name)


def _pack(
    max_length: int, *, num_bins: int = 8, **kwargs: Any
) -> _MaterializingAccumulator:
    """Build an accumulator via the operator and materialize emitted bins inline."""
    algorithm = kwargs.get("algorithm", "first_fit")
    if algorithm in ("first_fit", "best_fit"):
        kwargs["num_bins"] = num_bins
    op = PackSequences(max_length=max_length, **kwargs)
    acc = op.accumulator(deterministic=False, ctx={})
    assert isinstance(acc, PackingAccumulator)
    return _MaterializingAccumulator(op, acc)


def _records(batches: list[Any]) -> list[SampleRecord]:
    """First record of each ready batch ``(records, wait_ns)``."""
    return [b[0][0] for b in batches]


# ---------------------------------------------------------------------------
# Operator construction / validation
# ---------------------------------------------------------------------------


def test_invalid_max_length_raises() -> None:
    with pytest.raises(ValueError, match="max_length must be positive"):
        PackSequences(max_length=0, num_bins=4)


def test_unknown_algorithm_raises() -> None:
    with pytest.raises(ValueError, match="Unknown algorithm"):
        PackSequences(max_length=4, num_bins=4, algorithm="nope")  # type: ignore[arg-type]


def test_traits() -> None:
    traits = PackSequences(max_length=10, num_bins=4).traits()
    assert traits.indexable is False
    assert traits.requires_serial_state is False
    assert traits.preserves_cursor_order is False


def test_accumulator_config_passthrough() -> None:
    acc = _pack(10, num_bins=5, tokens_field="length", algorithm="first_fit")
    assert acc.max_length == 10
    assert acc.num_bins == 5
    assert acc.algorithm == "first_fit"


@pytest.mark.parametrize("algorithm", ["first_fit", "best_fit"])
def test_whole_record_algorithms_require_num_bins(
    algorithm: PackingAlgorithm,
) -> None:
    with pytest.raises(ValueError, match="num_bins is required"):
        PackSequences(max_length=4, algorithm=algorithm)


@pytest.mark.parametrize("algorithm", ["wrap", "best_fit_wrap"])
def test_slicing_algorithms_reject_num_bins(algorithm: PackingAlgorithm) -> None:
    with pytest.raises(ValueError, match="num_bins does not apply"):
        PackSequences(max_length=4, num_bins=1, algorithm=algorithm)


@pytest.mark.parametrize("algorithm", ["wrap", "best_fit_wrap"])
def test_slicing_algorithms_forbid_explicit_drop_oversized(
    algorithm: PackingAlgorithm,
) -> None:
    with pytest.raises(ValueError, match="drop_oversized=True is not allowed"):
        PackSequences(
            max_length=4,
            algorithm=algorithm,
            drop_oversized=True,
        )


@pytest.mark.parametrize(
    ("algorithm", "expected"),
    [
        ("first_fit", True),
        ("best_fit", True),
        ("wrap", False),
        ("best_fit_wrap", False),
    ],
)
def test_drop_oversized_default_depends_on_algorithm(
    algorithm: PackingAlgorithm, expected: bool
) -> None:
    kwargs = {"num_bins": 1} if algorithm in ("first_fit", "best_fit") else {}
    op = PackSequences(max_length=4, algorithm=algorithm, **kwargs)
    assert op.drop_oversized is expected


def test_string_length_fn_rejected() -> None:
    """length_fn is callable-or-None; a field name goes through tokens_field."""
    with pytest.raises(ValueError, match="length_fn must be a callable or None"):
        PackSequences(max_length=4, num_bins=4, length_fn="input_ids")  # type: ignore[arg-type]


def test_operator_passthrough_non_deferred() -> None:
    """process_many leaves non-deferred payloads (e.g. tombstones) untouched."""
    op = PackSequences(max_length=10, num_bins=10, tokens_field="length")
    packed = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"packed_samples": [{"value": 1}]},
    )
    result = op.process_many([packed])
    assert result == [packed]


# ---------------------------------------------------------------------------
# Deferred materialization: accumulator emits a plan; the operator builds payload
# ---------------------------------------------------------------------------


def test_accumulator_defers_payload_but_meta_is_complete() -> None:
    """push_many emits a deferred payload plan with final lineage/meta."""
    op = PackSequences(
        max_length=10, num_bins=10, length_fn=_simple_length_fn, drop_oversized=False
    )
    acc = op.accumulator(deterministic=False, ctx={})
    rec = _rec(0, 10)  # exactly fills a bin -> self-emits
    ready = acc.push_many([rec])
    assert len(ready) == 1
    record = ready[0][0][0]

    assert isinstance(record.payload, _DeferredBin)
    assert [seg.record for seg in record.payload.segments] == [rec]

    meta = record.meta.tags["_packing_metadata"]
    assert meta["total_length"] == 10
    assert meta["packing_efficiency"] == 1.0
    assert {r.cursor for r in record.meta.contributors} == {rec.meta.cursor}


def test_process_many_materializes_deferred_bins() -> None:
    """process_many turns deferred plans into real payloads, preserving meta."""
    op = PackSequences(
        max_length=4,
        algorithm="wrap",
        output="flat",
        tokens_field="input_ids",
        drop_oversized=False,
    )
    acc = op.accumulator(deterministic=False, ctx={})
    deferred = [
        b[0][0] for b in acc.push_many([_rec_tokens(0, [1, 2, 3, 4, 5, 6, 7, 8])])
    ]
    assert deferred and all(isinstance(r.payload, _DeferredBin) for r in deferred)

    materialized = op.process_many(deferred)
    assert [r.payload["input_ids"] for r in materialized] == [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    assert [r.meta for r in materialized] == [r.meta for r in deferred]


def test_process_many_is_stateless_across_instances() -> None:
    """Deferred bins materialize identically on any operator instance (parallel
    workers are deep-copied instances), so output is parallelism-invariant."""
    import copy

    op = PackSequences(
        max_length=4,
        algorithm="wrap",
        output="flat",
        tokens_field="input_ids",
        drop_oversized=False,
    )
    acc = op.accumulator(deterministic=False, ctx={})
    deferred = [b[0][0] for b in acc.push_many([_rec_tokens(0, list(range(8)))])]

    worker_a, worker_b = copy.deepcopy(op), copy.deepcopy(op)
    out_a = worker_a.process_many([deferred[0]])
    out_b = worker_b.process_many([deferred[1]])
    assert out_a[0].payload["input_ids"] == [0, 1, 2, 3]
    assert out_b[0].payload["input_ids"] == [4, 5, 6, 7]


# ---------------------------------------------------------------------------
# Envelope: first_fit / best_fit packing behaviour
# ---------------------------------------------------------------------------


def test_first_fit_basic() -> None:
    """[3,4,2] fills bin0 to 9 (kept); 5 opens bin1. Flush emits both."""
    acc = _pack(10, num_bins=10, length_fn=_simple_length_fn, drop_oversized=False)
    ready = acc.push_many([_rec(0, 3), _rec(1, 4), _rec(2, 2), _rec(3, 5)])
    assert ready == []

    tail = acc.flush()
    assert len(tail) == 2
    sample_counts = sorted(len(r.payload["packed_samples"]) for r in _records(tail))
    assert sample_counts == [1, 3]


def test_full_bin_emits_immediately() -> None:
    acc = _pack(10, num_bins=10, length_fn=_simple_length_fn, drop_oversized=False)
    ready = acc.push_many([_rec(0, 10)])
    assert len(ready) == 1
    meta = _records(ready)[0].meta.tags["_packing_metadata"]
    assert meta["total_length"] == 10
    assert meta["packing_efficiency"] == 1.0


def test_envelope_first_fit_keeps_whole_payloads() -> None:
    """first/best envelope is the list of whole record payloads."""
    acc = _pack(8, num_bins=4, drop_oversized=False)
    ready = acc.push_many([_rec_tokens(0, [10, 11, 12]), _rec_tokens(1, [20, 21])])
    tail = acc.flush()
    payload = _records(tail)[0].payload
    assert payload["packed_samples"] == [
        {"value": 0, "input_ids": [10, 11, 12]},
        {"value": 1, "input_ids": [20, 21]},
    ]


def test_oversized_drop_keeps_rest() -> None:
    acc = _pack(10, num_bins=10, length_fn=_simple_length_fn, drop_oversized=True)
    ready = acc.push_many([_rec(0, 5), _rec(1, 15), _rec(2, 5)])
    # [5, 5] fills a bin; the oversized 15 is dropped (its tombstone rides along).
    bins = [r for r in _records(ready) if not r.meta.tombstone]
    assert len(bins) == 1
    assert len(bins[0].payload["packed_samples"]) == 2


def test_oversized_drop_is_logged(caplog) -> None:
    import logging

    acc = _pack(10, num_bins=10, length_fn=_simple_length_fn, drop_oversized=True)
    acc.push_many([_rec(0, 15), _rec(1, 5), _rec(2, 12)])
    with caplog.at_level(logging.WARNING):
        acc.flush()
    assert any(
        "dropped 2 document(s)" in r.message and "27 token(s)" in r.message
        for r in caplog.records
    ), [r.message for r in caplog.records]


def test_oversized_raises_when_not_dropping() -> None:
    acc = _pack(10, num_bins=10, length_fn=_simple_length_fn, drop_oversized=False)
    with pytest.raises(ValueError, match="exceeds max_length"):
        acc.push_many([_rec(0, 15)])


def test_lane_isolation() -> None:
    acc = _pack(10, num_bins=10, length_fn=_simple_length_fn, drop_oversized=False)
    ready = acc.push_many(
        [_rec(0, 5, lane=0), _rec(1, 5, lane=1), _rec(2, 5, lane=0), _rec(3, 5, lane=1)]
    )
    assert len(ready) == 2
    for rec in _records(ready):
        assert len(rec.payload["packed_samples"]) == 2


def test_has_pending_data() -> None:
    acc = _pack(10, num_bins=10, length_fn=_simple_length_fn, drop_oversized=False)
    assert not acc.has_pending_data()
    acc.push_many([_rec(0, 3)])
    assert acc.has_pending_data()
    acc.flush()
    assert not acc.has_pending_data()


def test_flush_reset_is_lane_scoped() -> None:
    """A per-lane flush sentinel must only flush its own lane's bins.

    Bins are per-lane and history-dependent; flushing every lane at one
    lane's epoch boundary corrupts the others' packing and breaks replay.
    """
    acc = _pack(10, num_bins=10, length_fn=_simple_length_fn, drop_oversized=False)
    # Partial open bins in two lanes (length 3 << max_length=10).
    acc.push_many([_rec(0, 3, lane=0), _rec(1, 3, lane=1)])

    flushed = acc.flush(reset=True, lane_id=0)

    assert {r.meta.lane_id for r in _records(flushed)} == {0}
    assert not acc.has_pending_data(lane_id=0)
    assert acc.has_pending_data(lane_id=1)


def test_contributors_set_for_1to1_inputs() -> None:
    acc = _pack(10, num_bins=10, length_fn=_simple_length_fn, drop_oversized=False)
    rec1, rec2 = _rec(0, 5), _rec(1, 5)
    ready = acc.push_many([rec1, rec2])
    packed = _records(ready)[0]
    assert len(packed.meta.contributors) == 2
    cursors = {ref.cursor for ref in packed.meta.contributors}
    assert cursors == {rec1.meta.cursor, rec2.meta.cursor}
    assert all(ref.is_last_child for ref in packed.meta.contributors)


def test_best_fit_packs() -> None:
    acc = _pack(
        10,
        num_bins=10,
        length_fn=_simple_length_fn,
        algorithm="best_fit",
        drop_oversized=False,
    )
    ready = acc.push_many([_rec(0, 3), _rec(1, 4), _rec(2, 2), _rec(3, 5)])
    assert ready == []
    assert len(acc.flush()) == 2


def test_shuffle_length_sorts_descending() -> None:
    acc = _pack(
        10,
        num_bins=10,
        length_fn=_simple_length_fn,
        drop_oversized=False,
        shuffle_strategy="length",
    )
    acc.push_many([_rec(0, 2), _rec(1, 3), _rec(2, 4), _rec(3, 5)])
    tail = acc.flush()
    totals = sorted(
        r.meta.tags["_packing_metadata"]["total_length"] for r in _records(tail)
    )
    # Descending sort packs [5,4] and [3,2].
    assert totals == [5, 9]


def test_num_bins_limit_flushes_completed() -> None:
    acc = _pack(10, num_bins=2, length_fn=_simple_length_fn, drop_oversized=False)
    ready: list[Any] = []
    ready += acc.push_many([_rec(0, 3)])
    ready += acc.push_many([_rec(1, 4)])
    ready += acc.push_many([_rec(2, 2)])  # bin0 -> remaining 1
    ready += acc.push_many([_rec(3, 5)])  # bin1, at num_bins limit
    ready += acc.push_many([_rec(4, 6)])  # forces a flush of the oldest bin
    assert len(ready) == 1
    assert len(_records(ready)[0].payload["packed_samples"]) == 3
    assert len(acc.flush()) == 2


# ---------------------------------------------------------------------------
# Length resolution (tokens_field vs callable length_fn)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["input_ids", "tokens", "token_ids", "ids"])
def test_auto_detect_token_field(field: str) -> None:
    acc = _pack(10, num_bins=10, drop_oversized=False)
    ready = acc.push_many(
        [
            _rec_tokens(0, [1, 2, 3, 4], field=field),
            _rec_tokens(1, [1, 2, 3, 4, 5, 6], field=field),
        ]
    )
    assert len(ready) == 1
    assert _records(ready)[0].meta.tags["_packing_metadata"]["total_length"] == 10


def test_auto_detect_priority_input_ids_over_tokens() -> None:
    acc = _pack(10, num_bins=10, drop_oversized=False)
    rec = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"input_ids": [1, 2, 3, 4, 5], "tokens": [1, 2, 3]},
    )
    rec2 = _rec_tokens(1, [1, 2, 3, 4, 5])
    ready = acc.push_many([rec, rec2])
    assert len(ready) == 1
    assert _records(ready)[0].meta.tags["_packing_metadata"]["total_length"] == 10


def test_auto_detect_no_field_raises() -> None:
    acc = _pack(10, num_bins=10, drop_oversized=False)
    rec = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"text": "hello", "label": 1},
    )
    with pytest.raises(ValueError, match="auto-detect"):
        acc.push_many([rec])


def test_int_length_value() -> None:
    """A token field stored as an int is taken as the precomputed length."""
    acc = _pack(100, num_bins=10, drop_oversized=False)
    ready = acc.push_many([_rec_tokens(0, 40), _rec_tokens(1, 60)])
    assert len(ready) == 1
    assert _records(ready)[0].meta.tags["_packing_metadata"]["total_length"] == 100


def test_explicit_tokens_field() -> None:
    acc = _pack(10, num_bins=10, tokens_field="my_length", drop_oversized=False)
    recs = [
        SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, i), lane_id=0, chunk_id=0),
            payload={"my_length": list(range(n))},
        )
        for i, n in enumerate((4, 6))
    ]
    ready = acc.push_many(recs)
    assert len(ready) == 1
    assert _records(ready)[0].meta.tags["_packing_metadata"]["total_length"] == 10


def test_explicit_tokens_field_missing_raises() -> None:
    acc = _pack(10, num_bins=10, tokens_field="missing", drop_oversized=False)
    rec = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"input_ids": [1, 2, 3]},
    )
    with pytest.raises(ValueError, match="missing"):
        acc.push_many([rec])


def test_callable_length_fn_envelope() -> None:
    acc = _pack(
        10,
        num_bins=10,
        drop_oversized=False,
        length_fn=lambda r: r.payload.get("size", 0),
    )
    recs = [
        SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, i), lane_id=0, chunk_id=0),
            payload={"size": n},
        )
        for i, n in enumerate((3, 7))
    ]
    ready = acc.push_many(recs)
    assert len(ready) == 1
    assert _records(ready)[0].meta.tags["_packing_metadata"]["total_length"] == 10


def test_custom_pack_payloads_merge() -> None:
    """Envelope keeps the datnanovlm pattern: a custom callable merges the list."""
    acc = _pack(
        10,
        num_bins=10,
        drop_oversized=False,
        length_fn=lambda r: len(r.payload["input_ids"]),
        pack_payloads=lambda payloads: {"conv": [p["input_ids"] for p in payloads]},
    )
    acc.push_many([_rec_tokens(0, [1, 2, 3]), _rec_tokens(1, [4, 5])])
    payload = _records(acc.flush())[0].payload
    assert payload == {"packed_samples": {"conv": [[1, 2, 3], [4, 5]]}}


# ---------------------------------------------------------------------------
# wrap algorithm (envelope output = list of per-slice payloads)
# ---------------------------------------------------------------------------


def _wrap(
    max_length: int, *, tokens_field: str = "input_ids", **kw: Any
) -> _MaterializingAccumulator:
    return _pack(
        max_length,
        algorithm="wrap",
        tokens_field=tokens_field,
        drop_oversized=False,
        **kw,
    )


def test_wrap_envelope_is_list_of_slices() -> None:
    """wrap envelope preserves document boundaries as a list of per-slice dicts."""
    acc = _wrap(4, tokens_field="tokens")
    ready = acc.push_many(
        [
            _rec_tokens(i, t, field="tokens")
            for i, t in enumerate(([1, 2, 3], [4, 5, 6], [7, 8, 9]))
        ]
    )
    assert len(ready) == 2
    bin0, bin1 = _records(ready)
    # bin0 = rec0[1,2,3] + rec1[4]; bin1 = rec1[5,6] + rec2[7,8]; 9 stays buffered.
    assert bin0.payload["packed_samples"] == [{"tokens": [1, 2, 3]}, {"tokens": [4]}]
    assert bin1.payload["packed_samples"] == [{"tokens": [5, 6]}, {"tokens": [7, 8]}]
    assert bin0.meta.tags["_packing_metadata"]["total_length"] == 4
    assert bin0.meta.tags["_packing_metadata"]["packing_efficiency"] == 1.0
    assert bin0.meta.component_token_counts == {0: 4}
    assert bin1.meta.component_token_counts == {0: 4}

    assert acc.has_pending_data()
    tail = acc.flush()
    assert len(tail) == 1 and _records(tail)[0].meta.tombstone
    assert not acc.has_pending_data()


def test_wrap_oversized_splits_into_bins() -> None:
    acc = _wrap(4)
    ready = acc.push_many([_rec_tokens(0, list(range(10, 20)))])
    assert len(ready) == 2
    bin0, bin1 = _records(ready)
    assert bin0.payload["packed_samples"] == [{"input_ids": [10, 11, 12, 13]}]
    assert bin1.payload["packed_samples"] == [{"input_ids": [14, 15, 16, 17]}]
    assert bin0.meta.component_token_counts == {0: 4}
    tail = acc.flush()
    assert len(tail) == 1 and _records(tail)[0].meta.tombstone


def test_wrap_lane_isolation() -> None:
    acc = _wrap(4)
    ready = acc.push_many(
        [
            _rec_tokens(0, [1, 2, 3], lane=0),
            _rec_tokens(1, [10, 20, 30], lane=1),
            _rec_tokens(2, [4, 5, 6], lane=0),
            _rec_tokens(3, [40, 50, 60], lane=1),
        ]
    )
    assert len(ready) == 2
    by_lane = {
        r.meta.lane_id: [s["input_ids"] for s in r.payload["packed_samples"]]
        for r in _records(ready)
    }
    assert by_lane[0] == [[1, 2, 3], [4]]
    assert by_lane[1] == [[10, 20, 30], [40]]
    assert all(r.meta.tombstone for r in _records(acc.flush()))


def test_wrap_marks_last_child_only_on_final_bin() -> None:
    acc = _wrap(4)
    rec0 = _rec_tokens(0, [1, 2, 3, 4, 5, 6])  # spans bin0 (4) and bin1 (2)
    rec1 = _rec_tokens(1, [7, 8, 9, 10])
    ready = acc.push_many([rec0, rec1])
    assert len(ready) == 2
    bin0, bin1 = _records(ready)

    def closing(rec: SampleRecord) -> set[Any]:
        return {ref.cursor for ref in rec.meta.contributors if ref.is_last_child}

    assert rec0.meta.cursor not in closing(bin0)
    assert rec0.meta.cursor in closing(bin1)  # rec0's last token lands in bin1
    assert rec1.meta.cursor not in closing(bin1)  # rec1 not finished in bin1
    assert _records(acc.flush())[0].meta.tombstone


def test_wrap_primary_cursors_unique_across_bins() -> None:
    acc = _wrap(4)
    ready = acc.push_many([_rec_tokens(0, list(range(12)))])
    assert len(ready) == 3
    cursors = [r.meta.cursor for r in _records(ready)]
    assert len(set(cursors)) == 3


def test_wrap_flush_reset_clears_state() -> None:
    acc = _wrap(4)
    acc.push_many([_rec_tokens(0, [1, 2, 3])])
    assert acc.has_pending_data()
    acc.flush(reset=True)
    assert not acc.has_pending_data()
    ready = acc.push_many([_rec_tokens(1, [10, 20, 30, 40])])
    assert len(ready) == 1
    assert _records(ready)[0].payload["packed_samples"] == [
        {"input_ids": [10, 20, 30, 40]}
    ]


def test_wrap_numpy_arrays_preserve_dtype() -> None:
    np = pytest.importorskip("numpy")
    acc = _wrap(4)
    rec0 = _rec_tokens(0, [1, 2, 3], as_numpy=True)
    rec1 = SampleRecord(
        meta=rec0.meta.with_lineage((1,)),
        payload={"input_ids": np.array([4, 5, 6], dtype=np.uint32)},
    )
    ready = acc.push_many([rec0, rec1])
    assert len(ready) == 1
    slices = _records(ready)[0].payload["packed_samples"]
    assert slices[0]["input_ids"].tolist() == [1, 2, 3]
    assert slices[0]["input_ids"].dtype == np.uint32


def test_wrap_aligned_secondary_fields() -> None:
    acc = _wrap(4)
    rec0 = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
    )
    rec1 = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0),
        payload={"input_ids": [4, 5, 6, 7], "attention_mask": [1, 1, 1, 1]},
    )
    ready = acc.push_many([rec0, rec1])
    assert len(ready) == 1
    slices = _records(ready)[0].payload["packed_samples"]
    # bin0 = rec0[0:3] + rec1[0:1]; aligned mask is sliced in lockstep.
    assert slices == [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4], "attention_mask": [1]},
    ]


def test_wrap_misaligned_secondary_field_raises() -> None:
    acc = _wrap(4)
    rec = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"input_ids": [1, 2, 3, 4], "labels": [0, 0]},
    )
    with pytest.raises(ValueError, match="length 2"):
        acc.push_many([rec])


def test_wrap_inconsistent_auto_field_per_lane_raises() -> None:
    acc = _wrap(4, tokens_field="auto")
    acc.push_many(
        [
            SampleRecord(
                meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
                payload={"tokens": [1, 2, 3, 4]},
            )
        ]
    )
    with pytest.raises(ValueError, match="inconsistent auto-detected length field"):
        acc.push_many(
            [
                SampleRecord(
                    meta=SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0),
                    payload={"input_ids": [5, 6, 7, 8]},
                )
            ]
        )


def test_wrap_tail_tombstone_only_closing_contributors() -> None:
    """Tail-drop tombstones mirror tombstones_for_record: only is_last_child refs."""
    closing = SampleCursor(
        chunk_id=0, chunk_offset=0, sample_id=(0, 0, 0), lineage=(0,)
    )
    nonclosing = SampleCursor(
        chunk_id=0, chunk_offset=0, sample_id=(0, 0, 0), lineage=(1,)
    )
    meta = SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0).with_contributors(
        (
            ContributorRef(cursor=closing, is_last_child=True),
            ContributorRef(cursor=nonclosing, is_last_child=False),
        )
    )
    acc = _wrap(4)
    acc.push_many([SampleRecord(meta=meta, payload={"input_ids": [1, 2]})])
    tail = acc.flush()
    assert len(tail) == 1
    tomb = _records(tail)[0]
    assert tomb.meta.tombstone
    assert [r.cursor for r in tomb.meta.contributors] == [closing]


def test_wrap_component_token_remainder_matches_targets() -> None:
    """Floor splits + last-slice remainder preserve per-component token totals."""
    meta = SampleMeta(
        sample_id=(0, 0, 0),
        lane_id=0,
        chunk_id=0,
        component_sample_counts={0: 1, 1: 1},
        component_token_counts={0: 3, 1: 5},
    )
    acc = _wrap(4)
    ready = acc.push_many([SampleRecord(meta=meta, payload={"input_ids": [1] * 8})])
    assert len(ready) == 2
    cts0, cts1 = (r.meta.component_token_counts for r in _records(ready))
    assert cts0 is not None and cts1 is not None
    assert {k: cts0[k] + cts1[k] for k in (0, 1)} == {0: 3, 1: 5}


# ---------------------------------------------------------------------------
# best_fit_wrap algorithm (buffered largest-fit, one split closes each bin)
# ---------------------------------------------------------------------------


def _bfw(
    max_length: int, *, candidate_pool_size: int, **kw: Any
) -> _MaterializingAccumulator:
    return _pack(
        max_length,
        algorithm="best_fit_wrap",
        candidate_pool_size=candidate_pool_size,
        **kw,
    )


def test_bfw_largest_fit_builds_exact_bins() -> None:
    acc = _bfw(10, candidate_pool_size=3)
    ready = acc.push_many(
        [
            _rec_tokens(0, [1, 2, 3]),
            _rec_tokens(1, [11, 12, 13, 14, 15, 16, 17]),
            _rec_tokens(2, [21, 22, 23, 24, 25]),
        ]
    )
    assert len(ready) == 1
    bin0 = _records(ready)[0]
    assert bin0.payload["packed_samples"] == [
        {"value": 1, "input_ids": [11, 12, 13, 14, 15, 16, 17]},
        {"value": 0, "input_ids": [1, 2, 3]},
    ]
    assert bin0.meta.tags["_packing_metadata"]["packing_efficiency"] == 1.0
    assert acc.has_pending_data()


def test_bfw_exact_length_envelope_preserves_whole_payload() -> None:
    acc = _bfw(4, candidate_pool_size=8)
    ready = acc.push_many([_rec_tokens(0, [1, 2, 3, 4])])
    assert _records(ready)[0].payload["packed_samples"] == [
        {"value": 0, "input_ids": [1, 2, 3, 4]}
    ]


def test_bfw_split_closes_bin_and_remainder_places_later() -> None:
    acc = _bfw(8, candidate_pool_size=2)
    rec_a = _rec_tokens(0, [1, 2, 3, 4, 5])
    rec_b = _rec_tokens(1, [11, 12, 13, 14, 15])
    ready = acc.push_many([rec_a, rec_b])
    assert len(ready) == 1
    bin0 = _records(ready)[0]
    # a places whole (oldest of the length-5 tie); b's head fills the gap.
    assert bin0.payload["packed_samples"] == [
        {"value": 0, "input_ids": [1, 2, 3, 4, 5]},
        {"input_ids": [11, 12, 13]},
    ]

    def closing(rec: SampleRecord) -> set[Any]:
        return {ref.cursor for ref in rec.meta.contributors if ref.is_last_child}

    assert rec_a.meta.cursor in closing(bin0)
    assert rec_b.meta.cursor not in closing(bin0)

    ready = acc.push_many([_rec_tokens(2, [21, 22, 23, 24, 25, 26])])
    assert len(ready) == 1
    bin1 = _records(ready)[0]
    assert bin1.payload["packed_samples"] == [
        {"value": 2, "input_ids": [21, 22, 23, 24, 25, 26]},
        {"input_ids": [14, 15]},
    ]
    assert rec_b.meta.cursor in closing(bin1)
    assert not acc.has_pending_data()


def test_bfw_tail_fill_when_cheaper() -> None:
    """A tail cut may emit a suffix before its BOS-anchored prefix."""
    meta = SampleMeta(
        sample_id=(0, 0, 1),
        lane_id=0,
        chunk_id=0,
        component_sample_counts={0: 1},
        component_token_counts={0: 6},
    )
    rec_b = SampleRecord(meta=meta, payload={"input_ids": [11, 12, 13, 14, 15, 16]})
    acc = _bfw(8, candidate_pool_size=1)
    ready = acc.push_many([_rec_tokens(0, [1, 2, 3, 4, 5, 6]), rec_b])
    assert len(ready) == 1
    bin0 = _records(ready)[0]
    assert [s["input_ids"] for s in bin0.payload["packed_samples"]] == [
        [1, 2, 3, 4, 5, 6],
        [15, 16],
    ]

    tail = acc.flush()
    assert len(tail) == 1
    tail_rec = _records(tail)[0]
    assert not tail_rec.meta.tombstone
    assert [s["input_ids"] for s in tail_rec.payload["packed_samples"]] == [
        [11, 12, 13, 14]
    ]
    assert bin0.meta.component_token_counts == {0: 8}  # a's 6 + floor share 2 of b
    assert tail_rec.meta.component_token_counts == {0: 4}  # b's exact remainder


def test_bfw_oversized_emits_direct_full_bins() -> None:
    acc = _bfw(4, candidate_pool_size=4)
    ready = acc.push_many([_rec_tokens(0, list(range(10, 20)))])
    assert len(ready) == 2
    bin0, bin1 = _records(ready)
    assert bin0.payload["packed_samples"] == [{"input_ids": [10, 11, 12, 13]}]
    assert bin1.payload["packed_samples"] == [{"input_ids": [14, 15, 16, 17]}]
    tail = acc.flush()
    assert [s["input_ids"] for s in _records(tail)[0].payload["packed_samples"]] == [
        [18, 19]
    ]


def test_bfw_flush_drains_full_bins_before_tail() -> None:
    acc = _bfw(4, candidate_pool_size=100)
    acc.push_many(
        [_rec_tokens(i, [10 * i + 1, 10 * i + 2, 10 * i + 3]) for i in range(3)]
    )
    ready = acc.flush()
    assert len(ready) == 3
    contents = [
        [s["input_ids"] for s in r.payload["packed_samples"]] for r in _records(ready)
    ]
    # Bin 1's gap of 1 tail-fills from rec1 (severs 1 token vs 2 via its head);
    # bin 2's gap of 1 then head-fills the remaining [11, 12] head fragment.
    assert contents == [
        [[1, 2, 3], [13]],
        [[21, 22, 23], [11]],
        [[12]],
    ]


def test_bfw_flat_flush_pads_tail_instead_of_dropping() -> None:
    import numpy as np

    acc = _flat(
        8,
        algorithm="best_fit_wrap",
        pad_token_id=99,
        candidate_pool_size=4,
    )
    acc.push_many([_rec_tokens(0, [1, 2, 3]), _rec_tokens(1, [11, 12])])
    assert acc.has_pending_data()
    tail = acc.flush()
    assert len(tail) == 1
    rec = _records(tail)[0]
    assert not rec.meta.tombstone
    assert rec.payload["input_ids"] == [1, 2, 3, 11, 12, 99, 99, 99]
    np.testing.assert_array_equal(
        rec.payload["positions"], np.array([0, 1, 2, 0, 1, 0, 1, 2], dtype=np.int32)
    )
    assert rec.meta.padding_length == 3
    assert not acc.has_pending_data()


def test_bfw_age_guard_rescues_shadowed_item() -> None:
    """The age guard rescues a length-4 item shadowed by exact (5, 2) pairs."""

    def push_all(acc: _MaterializingAccumulator) -> list[Any]:
        rows: list[Any] = []
        rows += acc.push_many([_rec_tokens(0, [91, 92, 93, 94])])
        for i in range(1, 4):
            rows += acc.push_many([_rec_tokens(2 * i, [1, 2, 3, 4, 5])])
            rows += acc.push_many([_rec_tokens(2 * i + 1, [11, 12])])
        return rows

    guarded = _bfw(7, candidate_pool_size=3, max_candidate_age=4)
    rows = _records(push_all(guarded))
    assert len(rows) == 3
    assert rows[2].payload["packed_samples"][0] == {
        "value": 0,
        "input_ids": [91, 92, 93, 94],
    }

    # The default age of 8 * candidate_pool_size is not reached here.
    unguarded = _bfw(7, candidate_pool_size=3)
    rows = _records(push_all(unguarded))
    assert all(
        seg.get("value") != 0 for r in rows for seg in r.payload["packed_samples"]
    )
    assert unguarded.has_pending_data()


def test_bfw_rejects_callable_length_fn() -> None:
    with pytest.raises(ValueError, match="length_fn is only supported"):
        PackSequences(
            max_length=4,
            algorithm="best_fit_wrap",
            length_fn=lambda r: 4,
        )


def test_bfw_flat_requires_pad_token_id() -> None:
    with pytest.raises(ValueError, match="pad_token_id is required"):
        PackSequences(
            max_length=4,
            output="flat",
            algorithm="best_fit_wrap",
        )


def test_bfw_candidate_pool_defaults_and_validation() -> None:
    op = PackSequences(max_length=4, algorithm="best_fit_wrap")
    assert op.candidate_pool_size == 1024
    assert op.max_candidate_age == 8192

    with pytest.raises(ValueError, match="candidate_pool_size must be positive"):
        PackSequences(
            max_length=4,
            algorithm="best_fit_wrap",
            candidate_pool_size=0,
        )
    with pytest.raises(ValueError, match="max_candidate_age must be positive"):
        PackSequences(
            max_length=4,
            algorithm="best_fit_wrap",
            max_candidate_age=0,
        )


def test_candidate_pool_params_rejected_for_other_algorithms() -> None:
    op = PackSequences(max_length=4, num_bins=1)
    assert op.max_candidate_age is None

    with pytest.raises(ValueError, match="only applies to"):
        PackSequences(max_length=4, num_bins=1, candidate_pool_size=8)
    with pytest.raises(ValueError, match="only applies to"):
        PackSequences(
            max_length=4,
            algorithm="wrap",
            max_candidate_age=8,
        )


# ---------------------------------------------------------------------------
# flat output (pack_flat): positions, padding, validation
# ---------------------------------------------------------------------------


def _flat(
    max_length: int, *, algorithm: str = "first_fit", **kw: Any
) -> _MaterializingAccumulator:
    kw.setdefault("pad_token_id", 0)
    return _pack(
        max_length,
        num_bins=8,
        output="flat",
        algorithm=algorithm,
        drop_oversized=False,
        **kw,
    )


@pytest.mark.parametrize("algorithm", ["first_fit", "best_fit"])
def test_flat_first_best_pads_and_resets_positions(algorithm: str) -> None:
    import numpy as np

    acc = _flat(8, algorithm=algorithm, pad_token_id=999)
    acc.push_many([_rec_tokens(0, [10, 11, 12]), _rec_tokens(1, [20, 21])])
    record = _records(acc.flush())[0]
    assert "packed_samples" not in record.payload
    # 3 + 2 real, padded to 8; positions reset per doc, pad tail is its own doc.
    assert record.payload["input_ids"] == [10, 11, 12, 20, 21, 999, 999, 999]
    np.testing.assert_array_equal(
        record.payload["positions"], np.array([0, 1, 2, 0, 1, 0, 1, 2], dtype=np.int32)
    )
    # Efficiency uses real (non-pad) tokens.
    meta = record.meta.tags["_packing_metadata"]
    assert meta["total_length"] == 5
    assert meta["packing_efficiency"] == 5 / 8
    assert meta["padding_length"] == 3  # 8 - 5
    assert record.meta.padding_length == 3  # property reads the tag


def test_flat_wrap_no_padding() -> None:
    import numpy as np

    acc = _flat(7, algorithm="wrap")
    ready = acc.push_many([_rec_tokens(0, [1, 2, 3, 4]), _rec_tokens(1, [5, 6, 7])])
    record = _records(ready)[0]
    assert "packed_samples" not in record.payload
    assert record.payload["input_ids"] == [1, 2, 3, 4, 5, 6, 7]
    np.testing.assert_array_equal(
        record.payload["positions"], np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int32)
    )
    assert record.meta.padding_length == 0  # wrap fills exactly


def test_flat_wrap_split_record_positions_reset_per_bin() -> None:
    """A record split across bins restarts positions in the next bin."""
    import numpy as np

    acc = _flat(4, algorithm="wrap")
    ready = acc.push_many([_rec_tokens(0, list(range(8)))])
    assert len(ready) == 2
    for rec in _records(ready):
        np.testing.assert_array_equal(
            rec.payload["positions"], np.array([0, 1, 2, 3], dtype=np.int32)
        )


@pytest.mark.parametrize("algorithm", ["first_fit", "best_fit"])
def test_flat_requires_pad_token_id(algorithm: str) -> None:
    with pytest.raises(ValueError, match="pad_token_id is required"):
        PackSequences(
            max_length=8,
            num_bins=8,
            output="flat",
            algorithm=algorithm,
            drop_oversized=False,
        )


def test_flat_wrap_no_pad_token_id_needed() -> None:
    # wrap fills bins exactly, so no padding and no pad_token_id requirement.
    PackSequences(max_length=4, output="flat", algorithm="wrap", drop_oversized=False)


def test_flat_rejects_callable_length_fn() -> None:
    """In flat the packing length is len(tokens_field), so a custom length_fn
    (which could only disagree) is rejected at construction."""
    with pytest.raises(ValueError, match="length_fn is only supported"):
        PackSequences(
            max_length=8,
            num_bins=8,
            output="flat",
            algorithm="first_fit",
            pad_token_id=0,
            tokens_field="input_ids",
            length_fn=lambda r: len(r.payload["input_ids"]),
        )


def test_wrap_rejects_callable_length_fn() -> None:
    """wrap slices tokens_field, so its length governs packing — no length_fn."""
    with pytest.raises(ValueError, match="length_fn is only supported"):
        PackSequences(
            max_length=4,
            algorithm="wrap",
            drop_oversized=False,
            length_fn=lambda r: len(r.payload["input_ids"]),
        )


def test_flat_emit_positions_false_omits_key() -> None:
    acc = _flat(8, algorithm="first_fit", pad_token_id=0, emit_positions=False)
    acc.push_many([_rec_tokens(0, [10, 11, 12]), _rec_tokens(1, [20, 21])])
    payload = _records(acc.flush())[0].payload
    assert "positions" not in payload
    assert payload["input_ids"] == [10, 11, 12, 20, 21, 0, 0, 0]


def test_flat_positions_field_collision_raises() -> None:
    acc = _flat(8, algorithm="first_fit", pad_token_id=0, emit_positions=True)
    rec = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"input_ids": [1, 2, 3], "positions": [9, 9, 9]},
    )
    with pytest.raises(ValueError, match="already has a 'positions' field"):
        acc.push_many([rec])
        acc.flush()


def test_flat_feeds_to_training() -> None:
    """A flat record drops straight into to_training: positions is auto-surfaced,
    and the pad tail (meta.padding_length=2) is masked from the labels."""
    import numpy as np

    acc = _flat(6, algorithm="first_fit", pad_token_id=999)
    acc.push_many([_rec_tokens(0, [100, 101]), _rec_tokens(1, [110, 111])])
    record = _records(acc.flush())[0]
    assert record.meta.padding_length == 2
    out = SampleBatch(records=[record]).to_training(return_labels=True)
    np.testing.assert_array_equal(
        out["input_ids"], np.array([[100, 101, 110, 111, 999]])
    )
    np.testing.assert_array_equal(
        out["labels"], np.array([[101, 110, 111, -100, -100]])
    )
    np.testing.assert_array_equal(out["positions"], np.array([[0, 1, 0, 1, 0]]))


def test_flat_loss_mask_feeds_to_training() -> None:
    """Packed SFT windows must not supervise cross-document labels."""
    import numpy as np

    acc = _flat(12, algorithm="best_fit", pad_token_id=0)
    doc1 = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={
            "input_ids": [11, 12, 13, 14, 15, 16],
            "loss_mask": [0, 0, 0, 1, 1, 1],
        },
    )
    doc2 = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0),
        payload={"input_ids": [21, 22, 23, 24], "loss_mask": [0, 0, 1, 1]},
    )
    acc.push_many([doc1, doc2])
    record = _records(acc.flush())[0]
    assert record.payload["loss_mask"] == [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]
    assert record.meta.padding_length == 2

    out = SampleBatch(records=[record]).to_training(return_labels=True)
    assert "loss_mask" not in out
    np.testing.assert_array_equal(
        out["labels"],
        np.array([[-100, -100, 14, 15, 16, -100, -100, 23, 24, -100, -100]]),
    )
    np.testing.assert_array_equal(
        out["positions"], np.array([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0]])
    )


# ---------------------------------------------------------------------------
# Lineage: every dropped record closes its contributor offsets via a tombstone
# ---------------------------------------------------------------------------


def test_oversized_drop_emits_tombstone() -> None:
    """first/best oversized drop closes the record's offsets (elastic resume)."""
    acc = _pack(4, num_bins=4, drop_oversized=True)
    rec = _rec_tokens(0, [1, 2, 3, 4, 5, 6])
    ready = acc.push_many([rec])
    tombs = [r for r in _records(ready) if r.meta.tombstone]
    assert len(tombs) == 1
    assert tombs[0].payload is None
    assert {r.cursor for r in tombs[0].meta.contributors} == {rec.meta.cursor}


def test_wrap_empty_record_emits_tombstone() -> None:
    """A zero-length record carries no tokens but still closes its offsets."""
    acc = _wrap(4)
    rec = _rec_tokens(0, [])
    ready = acc.push_many([rec])
    assert len(ready) == 1
    assert _records(ready)[0].meta.tombstone
    assert not acc.has_pending_data()


# ---------------------------------------------------------------------------
# The matrix: every algorithm × output combination, and flat == flatten(envelope)
# ---------------------------------------------------------------------------

_MATRIX_RECORDS = [
    _rec_tokens(0, [10, 11, 12]),
    _rec_tokens(1, [20, 21]),
    _rec_tokens(2, [30, 31, 32, 33]),
]


@pytest.mark.parametrize("algorithm", _PACKING_ALGORITHMS)
def test_matrix_envelope_is_list(algorithm: PackingAlgorithm) -> None:
    acc = _pack(4, num_bins=4, algorithm=algorithm, drop_oversized=False)
    recs = [
        r
        for r in _records(acc.push_many(_MATRIX_RECORDS) + acc.flush())
        if r.payload is not None
    ]
    assert recs
    for rec in recs:
        assert set(rec.payload) == {"packed_samples"}
        assert isinstance(rec.payload["packed_samples"], list)
        assert rec.meta.padding_length is None  # envelope never pads


@pytest.mark.parametrize("algorithm", _PACKING_ALGORITHMS)
@pytest.mark.parametrize("emit_positions", [True, False])
def test_matrix_flat_uniform_shape(
    algorithm: PackingAlgorithm, emit_positions: bool
) -> None:
    """Every algorithm emits the identical flat shape; positions is opt-out."""
    acc = _flat(4, algorithm=algorithm, pad_token_id=0, emit_positions=emit_positions)
    recs = [
        r
        for r in _records(acc.push_many(_MATRIX_RECORDS) + acc.flush())
        if r.payload is not None
    ]
    assert recs
    expected_keys = {"input_ids", "positions"} if emit_positions else {"input_ids"}
    for rec in recs:
        assert set(rec.payload) == expected_keys
        assert len(rec.payload["input_ids"]) == 4  # always padded/sliced to max_length
        if emit_positions:
            assert len(rec.payload["positions"]) == 4


@pytest.mark.parametrize("algorithm", _PACKING_ALGORITHMS)
def test_matrix_flat_equals_flatten_envelope(algorithm: PackingAlgorithm) -> None:
    """flat tokens (minus pad) == concat of the envelope's per-segment token slices."""
    env = _pack(4, num_bins=4, algorithm=algorithm, drop_oversized=False)
    env_recs = [
        r
        for r in _records(env.push_many(_MATRIX_RECORDS) + env.flush())
        if r.payload is not None
    ]
    env_tokens = []
    for rec in env_recs:
        for seg in rec.payload["packed_samples"]:
            env_tokens.extend(seg["input_ids"])

    flat = _flat(4, algorithm=algorithm, pad_token_id=-1, emit_positions=True)
    flat_recs = [
        r
        for r in _records(flat.push_many(_MATRIX_RECORDS) + flat.flush())
        if r.payload is not None
    ]
    flat_tokens = []
    for rec in flat_recs:
        flat_tokens.extend(t for t in rec.payload["input_ids"] if t != -1)

    assert flat_tokens == env_tokens


def test_matrix_flat_first_fit_multicomponent_apportionment_matches_envelope() -> None:
    """Token apportionment for a multi-component record is identical envelope vs flat."""
    meta = SampleMeta(
        sample_id=(0, 0, 0),
        lane_id=0,
        chunk_id=0,
        component_sample_counts={0: 1, 1: 1},
        component_token_counts={0: 2, 1: 3},
    )
    rec = SampleRecord(meta=meta, payload={"input_ids": [1, 2, 3, 4, 5]})

    env = _pack(8, num_bins=4, drop_oversized=False)
    env_rec = _records(env.push_many([rec]) + env.flush())[0]

    flat = _flat(8, algorithm="first_fit", pad_token_id=0)
    flat_rec = _records(flat.push_many([rec]) + flat.flush())[0]

    assert env_rec.meta.component_token_counts == {0: 2, 1: 3}
    assert flat_rec.meta.component_token_counts == env_rec.meta.component_token_counts


# ---------------------------------------------------------------------------
# Serializers in isolation
# ---------------------------------------------------------------------------


def test_envelope_serializer_whole_vs_slice_segments() -> None:
    rec = _rec_tokens(0, [1, 2, 3, 4])
    whole = Segment(record=rec, start=0, end=4, seq_len=4, is_last=True, is_slice=False)
    sliced = Segment(
        record=rec,
        start=1,
        end=3,
        seq_len=4,
        field="input_ids",
        is_last=False,
        is_slice=True,
    )

    ser = _EnvelopeSerializer(lambda payloads: payloads)
    assert ser.build_payload([whole]) == {
        "packed_samples": [{"value": 0, "input_ids": [1, 2, 3, 4]}]
    }
    assert ser.build_payload([sliced]) == {"packed_samples": [{"input_ids": [2, 3]}]}


def test_flat_serializer_pad_value_per_field() -> None:
    """Token field pads with pad_token_id; aligned fields pad with 0."""
    rec = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"input_ids": [1, 2], "attention_mask": [1, 1]},
    )
    seg = Segment(
        record=rec, start=0, end=2, seq_len=2, field=None, is_last=True, is_slice=False
    )
    ser = _FlatSerializer(
        max_length=4, tokens_field="input_ids", pad_token_id=7, emit_positions=True
    )
    payload = ser.build_payload([seg])
    assert payload["input_ids"] == [1, 2, 7, 7]
    assert payload["attention_mask"] == [1, 1, 0, 0]


# ---------------------------------------------------------------------------
# Component token apportionment + remaining edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", _PACKING_ALGORITHMS)
def test_multicomponent_token_apportionment_conserves_length(
    algorithm: PackingAlgorithm,
) -> None:
    """A multi-component record with NO explicit component_token_counts gets its
    length apportioned by sample share, summing to the record length exactly
    (largest-remainder) — not independent per-component rounding which would
    under-count. Identical across algorithms (one unified lineage path)."""
    meta = SampleMeta(
        sample_id=(0, 0, 0),
        lane_id=0,
        chunk_id=0,
        component_sample_counts={0: 1, 1: 1, 2: 1},  # 3 components, no token counts
    )
    rec = SampleRecord(meta=meta, payload={"input_ids": list(range(10))})
    acc = _pack(10, num_bins=4, algorithm=algorithm, drop_oversized=False)
    out = _records(acc.push_many([rec]) + acc.flush())[0]
    cts = out.meta.component_token_counts
    assert cts is not None
    assert sum(cts.values()) == 10  # conserved exactly (sums to the record length)
    assert cts == {0: 3, 1: 3, 2: 4}  # remainder lands on the last cid


def test_flat_inconsistent_keys_across_records_raises() -> None:
    """flat concatenation needs homogeneous sliceable keys across a bin."""
    acc = _flat(8, algorithm="first_fit", pad_token_id=0)
    r0 = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"input_ids": [1, 2], "attention_mask": [1, 1]},
    )
    r1 = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0),
        payload={"input_ids": [3, 4]},  # missing attention_mask
    )
    acc.push_many([r0, r1])
    with pytest.raises(ValueError, match="inconsistent payload keys"):
        acc.flush()


def test_first_fit_zero_length_record_kept_and_closed() -> None:
    """A 0-length (by token count) record is KEPT by first/best — it may carry
    non-token payload (e.g. multimodal) — contributes 0 tokens, and still closes
    its contributor offset. (wrap, which only streams tokens, drops empties.)"""
    acc = _pack(8, num_bins=4, drop_oversized=False)
    rec = _rec_tokens(0, [])
    out = _records(acc.push_many([rec]) + acc.flush())
    assert len(out) == 1
    packed = out[0]
    assert not packed.meta.tombstone
    assert packed.payload["packed_samples"] == [{"value": 0, "input_ids": []}]
    assert {r.cursor for r in packed.meta.contributors} == {rec.meta.cursor}
    assert packed.meta.component_token_counts in (None, {})  # no tokens charged


# ---------------------------------------------------------------------------
# Cross-lane isolation, field validation, and bin-flush edge cases
# ---------------------------------------------------------------------------


def test_wrap_no_cross_lane_token_corruption() -> None:
    """component_token_counts must not leak across lanes that share a cursor key.

    Chunk ids restart per lane, so two lanes can hold split records with the same
    cursor key; each lane's per-component token total must conserve independently."""

    def wrec(lane: int, i: int, toks: list[int]) -> SampleRecord:
        meta = SampleMeta(
            sample_id=(0, 0, i),
            lane_id=lane,
            chunk_id=0,
            component_sample_counts={0: 1},
            component_token_counts={0: len(toks)},
        )
        return SampleRecord(meta=meta, payload={"input_ids": list(toks)})

    acc = _wrap(4)
    # Lanes 0 and 1 each: a len-6 record (splits across 2 bins, SAME cursor key
    # across lanes) + a len-2 record that closes the second bin.
    recs = [
        wrec(0, 0, [1, 2, 3, 4, 5, 6]),
        wrec(1, 0, [11, 12, 13, 14, 15, 16]),
        wrec(0, 2, [7, 8]),
        wrec(1, 2, [17, 18]),
    ]
    by_lane: dict[int, int] = {0: 0, 1: 0}
    for rec in _records(acc.push_many(recs)):
        cts = rec.meta.component_token_counts
        if cts:
            by_lane[rec.meta.lane_id] += cts.get(0, 0)
    # Each lane emitted 8 real tokens (6 + 2); a shared key would skew the totals.
    assert by_lane == {0: 8, 1: 8}


@pytest.mark.parametrize("algorithm", _PACKING_ALGORITHMS)
def test_flat_rejects_int_token_field(algorithm: PackingAlgorithm) -> None:
    """An int token field (precomputed length) is envelope-only; flat/wrap reject
    it with a clear message instead of a downstream 'int not subscriptable'."""
    acc = _flat(4, algorithm=algorithm, pad_token_id=0)
    rec = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"input_ids": 4},
    )
    with pytest.raises(ValueError, match="sliceable sequence"):
        acc.push_many([rec])


def test_flat_negative_pad_into_unsigned_dtype_raises_clearly() -> None:
    """A pad id outside the token dtype (e.g. -1 into uint32) fails with an
    actionable message, not a raw numpy OverflowError."""
    np = pytest.importorskip("numpy")
    assert np is not None
    acc = _flat(8, algorithm="first_fit", pad_token_id=-1)
    acc.push_many([_rec_tokens(0, [1, 2, 3], as_numpy=True)])  # uint32 tokens
    with pytest.raises(ValueError, match="not representable"):
        acc.flush()


def test_flat_scalar_positions_collision_raises() -> None:
    """A pre-existing scalar `positions` (not a sliceable field) is still caught."""
    acc = _flat(8, algorithm="first_fit", pad_token_id=0, emit_positions=True)
    rec = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"input_ids": [1, 2, 3], "positions": 99},
    )
    acc.push_many([rec])
    with pytest.raises(ValueError, match="already has a 'positions' field"):
        acc.flush()


def test_envelope_rejects_pad_token_id() -> None:
    with pytest.raises(ValueError, match="pad_token_id only applies"):
        PackSequences(max_length=4, num_bins=4, output="envelope", pad_token_id=0)


def test_flat_rejects_pack_payloads() -> None:
    with pytest.raises(ValueError, match="pack_payloads only applies"):
        PackSequences(
            max_length=4,
            output="flat",
            algorithm="wrap",
            drop_oversized=False,
            pack_payloads="numpy_array",
        )


def test_serializers_reject_empty_bin() -> None:
    with pytest.raises(ValueError, match="empty bin"):
        _FlatSerializer(4, "input_ids", 0, True).build_payload([])
    with pytest.raises(ValueError, match="empty bin"):
        _EnvelopeSerializer(lambda p: p).build_payload([])


def test_no_premature_flush_when_new_record_self_emits() -> None:
    """A record that fills a fresh bin exactly (self-emits) must not evict an
    existing partial bin to make room it never uses."""
    acc = _pack(4, num_bins=2, drop_oversized=False)
    acc.push_many([_rec_tokens(0, [1, 2, 3])])  # bin0, remaining 1
    acc.push_many([_rec_tokens(1, [4, 5, 6])])  # bin1, remaining 1 (at num_bins)
    ready = acc.push_many([_rec_tokens(2, [7, 8, 9, 10])])  # exact max → self-emits
    assert len(ready) == 1  # only the self-emitted bin; no partial bin evicted
    assert acc.has_pending_data()  # bin0 and bin1 still buffered


# ---------------------------------------------------------------------------
# Homogeneous packing (packed samples limited to one mixing domain)
# ---------------------------------------------------------------------------


def _crec(
    i: int, length: int, component: int, *, lane: int = 0, chunk: int = 0
) -> SampleRecord:
    """Length-field record in mixing domain ``component``."""
    meta = SampleMeta(
        sample_id=(0, 0, i),
        lane_id=lane,
        chunk_id=chunk,
        component_sample_counts={component: 1},
    )
    return SampleRecord(meta=meta, payload={"value": i, "length": length})


def _crec_tokens(
    i: int, tokens: list[int], component: int, *, lane: int = 0, chunk: int = 0
) -> SampleRecord:
    """Token-field record in mixing domain ``component``."""
    meta = SampleMeta(
        sample_id=(0, 0, i),
        lane_id=lane,
        chunk_id=chunk,
        component_sample_counts={component: 1},
    )
    return SampleRecord(meta=meta, payload={"input_ids": list(tokens)})


def _bin_components(rec: SampleRecord) -> set[int]:
    """Component ids (mixing domains) in a packed sample."""
    return set(rec.meta.component_sample_counts)


def test_unknown_homogeneity_raises() -> None:
    with pytest.raises(ValueError, match="Unknown homogeneity"):
        PackSequences(max_length=4, num_bins=4, homogeneity="nope")  # type: ignore[arg-type]


@pytest.mark.parametrize("algorithm", ["first_fit", "best_fit"])
def test_homogeneous_full_never_mixes_domains(algorithm: str) -> None:
    acc = _pack(
        10,
        num_bins=8,
        length_fn=_simple_length_fn,
        drop_oversized=False,
        algorithm=algorithm,
        homogeneity="full",
    )
    # Interleaved domains, each record len 4 (two fit in one max_length=10 bin).
    recs = [
        _crec(0, 4, 0),
        _crec(1, 4, 1),
        _crec(2, 4, 0),
        _crec(3, 4, 1),
    ]
    out = _records(acc.push_many(recs)) + _records(acc.flush())
    assert all(len(_bin_components(r)) == 1 for r in out)
    # {0: 2}/{1: 2}: each domain's two records packed together, not split apart.
    per_domain = sorted(
        (r.meta.component_sample_counts for r in out), key=lambda c: sorted(c)
    )
    assert per_domain == [{0: 2}, {1: 2}]


def test_mixed_default_still_mixes_domains() -> None:
    """Back-compat: the default homogeneity='none' still mixes domains in a bin."""
    acc = _pack(
        10,
        num_bins=8,
        length_fn=_simple_length_fn,
        drop_oversized=False,
        algorithm="first_fit",
        homogeneity="none",
    )
    recs = [_crec(0, 4, 0), _crec(1, 4, 1), _crec(2, 4, 0), _crec(3, 4, 1)]
    out = _records(acc.push_many(recs)) + _records(acc.flush())
    assert any(len(_bin_components(r)) == 2 for r in out)


def test_homogeneous_full_num_bins_is_per_domain() -> None:
    acc = _pack(
        10,
        num_bins=1,
        length_fn=_simple_length_fn,
        drop_oversized=False,
        algorithm="first_fit",
        homogeneity="full",
    )
    # Two partial records, different domains. With a single shared bin one would
    # evict the other; per-domain bins keep both buffered.
    ready = acc.push_many([_crec(0, 4, 0), _crec(1, 4, 1)])
    assert ready == []  # neither bin full, no eviction across domains
    out = _records(acc.flush())
    per_domain = sorted(
        (r.meta.component_sample_counts for r in out), key=lambda c: sorted(c)
    )
    assert per_domain == [{0: 1}, {1: 1}]


@pytest.mark.parametrize("counts", [{0: 1, 1: 1}, {}], ids=["multi", "empty"])
def test_homogeneous_full_rejects_non_single_component(counts: dict[int, int]) -> None:
    acc = _pack(
        10,
        num_bins=8,
        length_fn=_simple_length_fn,
        drop_oversized=False,
        homogeneity="full",
    )
    rec = SampleRecord(
        meta=SampleMeta(
            sample_id=(0, 0, 0),
            lane_id=0,
            chunk_id=0,
            component_sample_counts=counts,
        ),
        payload={"value": 0, "length": 4},
    )
    with pytest.raises(ValueError, match="single-component records"):
        acc.push_many([rec])


def test_homogeneous_full_wrap_streams_per_domain() -> None:
    acc = _wrap(4, homogeneity="full")
    recs = [
        _crec_tokens(0, [1, 2, 3, 4], 0),
        _crec_tokens(1, [5, 6, 7, 8], 1),
        _crec_tokens(2, [9, 10, 11, 12], 0),
    ]
    out = _records(acc.push_many(recs)) + _records(acc.flush())
    assert all(len(_bin_components(r)) == 1 for r in out)
    domains = sorted(next(iter(r.meta.component_sample_counts)) for r in out)
    assert domains == [0, 0, 1]  # two domain-0 bins, one domain-1 bin


def test_homogeneous_full_best_fit_wrap_buffers_per_domain() -> None:
    acc = _bfw(6, candidate_pool_size=2, homogeneity="full")
    recs = [
        _crec_tokens(0, [1, 2, 3], 0),
        _crec_tokens(1, [4, 5, 6], 1),
        _crec_tokens(2, [7, 8, 9], 0),
        _crec_tokens(3, [10, 11, 12], 1),
    ]
    out = _records(acc.push_many(recs))
    assert [rec.meta.component_sample_counts for rec in out] == [{0: 2}, {1: 2}]
    assert acc.has_pending_data() is False


def test_homogeneous_full_best_fit_wrap_flushes_by_lane_and_domain() -> None:
    acc = _bfw(6, candidate_pool_size=8, homogeneity="full")
    ready = acc.push_many(
        [
            _crec_tokens(0, [1, 2], 0, lane=0),
            _crec_tokens(1, [3, 4], 1, lane=0),
            _crec_tokens(2, [5, 6], 0, lane=1),
        ]
    )
    assert ready == []

    out0 = _records(acc.flush(lane_id=0))
    assert sorted(_bin_key(rec) for rec in out0) == [(0, 0, 1), (0, 1, 1)]
    assert acc.has_pending_data(0) is False
    assert acc.has_pending_data(1) is True

    out1 = _records(acc.flush(lane_id=1))
    assert [_bin_key(rec) for rec in out1] == [(1, 0, 1)]
    assert acc.has_pending_data() is False


def test_homogeneous_full_wrap_drops_partial_tail_per_domain(caplog: Any) -> None:
    import logging

    acc = _wrap(4, homogeneity="full")
    recs = [
        _crec_tokens(0, [1, 2], 0),  # dom0
        _crec_tokens(1, [7, 8, 9], 1),  # dom1: never fills a bin -> tail dropped
        _crec_tokens(2, [3, 4], 0),  # dom0: completes a {0:2} bin with rec0
        _crec_tokens(3, [5, 6], 0),  # dom0: leftover -> tail dropped
    ]
    emitted = _records(acc.push_many(recs))
    # Exactly one full bin, single domain 0, from the two complete dom0 records.
    assert len(emitted) == 1
    assert _bin_components(emitted[0]) == {0}

    with caplog.at_level(logging.WARNING):
        flushed = _records(acc.flush())
    # Both leftover tails (dom0 [5,6], dom1 [7,8,9]) drop as tombstones; no real bin.
    assert flushed and all(r.meta.tombstone for r in flushed)
    assert any(
        "2 packing group(s) in 1 lane(s)" in r.message for r in caplog.records
    ), [r.message for r in caplog.records]


def test_homogeneous_full_wrap_auto_field_is_consistent_lane_wide() -> None:
    # tokens_field defaults to "auto"; two domains in one lane must resolve to the
    # same auto-detected field, even though they buffer separately.
    acc = _pack(4, algorithm="wrap", drop_oversized=False, homogeneity="full")

    def rec(i: int, field: str, component: int) -> SampleRecord:
        meta = SampleMeta(
            sample_id=(0, 0, i),
            lane_id=0,
            chunk_id=0,
            component_sample_counts={component: 1},
        )
        return SampleRecord(meta=meta, payload={field: [1, 2, 3, 4]})

    acc.push_many([rec(0, "input_ids", 0)])  # lane's field -> input_ids
    with pytest.raises(ValueError, match="inconsistent auto-detected length field"):
        acc.push_many([rec(1, "tokens", 1)])  # different field, same lane -> rejected


def _bin_key(rec: SampleRecord) -> tuple[int, int, int]:
    """(lane_id, sole component id, sample count) of a single-domain packed sample."""
    counts = rec.meta.component_sample_counts
    assert len(counts) == 1
    cid, n = next(iter(counts.items()))
    return (rec.meta.lane_id, cid, n)


def test_homogeneous_full_partitions_by_lane_and_domain() -> None:
    """Keying uses BOTH (lane, domain), tested where records could really merge."""
    acc = _pack(
        10,
        num_bins=8,
        length_fn=_simple_length_fn,
        drop_oversized=False,
        homogeneity="full",
    )
    recs = [
        _crec(0, 4, 0, lane=0),  # (lane0, dom0): opens a bin with remaining 6
        _crec(1, 4, 0, lane=1),  # (lane1, dom0): same domain, WOULD fit lane0's bin
        #                          by capacity, but different lane -> own bin
        _crec(2, 4, 1, lane=0),  # (lane0, dom1): same lane, WOULD fit lane0's bin
        #                          by capacity, but different domain -> own bin
        _crec(3, 4, 0, lane=0),  # (lane0, dom0): same lane AND domain -> joins rec0
    ]
    out = _records(acc.push_many(recs)) + _records(acc.flush())
    assert all(len(_bin_components(r)) == 1 for r in out)
    # A lane/domain key collision would merge rec1 or rec2 into rec0's group;
    # only exact (lane, domain) matches share one.
    assert sorted(_bin_key(r) for r in out) == [(0, 0, 2), (0, 1, 1), (1, 0, 1)]


def test_homogeneous_full_flush_is_lane_scoped_across_domains() -> None:
    acc = _pack(
        10,
        num_bins=8,
        length_fn=_simple_length_fn,
        drop_oversized=False,
        homogeneity="full",
    )
    # Two domains buffered in lane 0, one in lane 1 (all partial -> nothing emits).
    ready = acc.push_many(
        [_crec(0, 4, 0, lane=0), _crec(1, 4, 1, lane=0), _crec(2, 4, 0, lane=1)]
    )
    assert ready == []

    out0 = _records(acc.flush(lane_id=0))
    # Both lane-0 domains drained; lane 1 untouched.
    assert sorted(_bin_key(r) for r in out0) == [(0, 0, 1), (0, 1, 1)]
    assert acc.has_pending_data(0) is False
    assert acc.has_pending_data(1) is True

    out1 = _records(acc.flush(lane_id=1))
    assert sorted(_bin_key(r) for r in out1) == [(1, 0, 1)]
    assert acc.has_pending_data() is False


def test_homogeneous_full_flush_order_is_deterministic() -> None:
    """Flush order is fixed by (lane, domain), independent of arrival order."""
    acc = _pack(
        10,
        num_bins=8,
        length_fn=_simple_length_fn,
        drop_oversized=False,
        homogeneity="full",
    )
    # Arrive out of order (2, 0, 1); each partial bin stays buffered until flush.
    ready = acc.push_many([_crec(0, 4, 2), _crec(1, 4, 0), _crec(2, 4, 1)])
    assert ready == []
    out = _records(acc.flush())
    # Emitted sorted by domain (all lane 0), not by arrival order.
    assert [next(iter(r.meta.component_sample_counts)) for r in out] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Grouped homogeneous packing (packed samples limited to one group of domains)
# ---------------------------------------------------------------------------

# Component-id -> component-name registry standing in for the engine's service.
_ID_TO_NAME = {0: "python", 1: "java", 2: "c4", 3: "wiki"}
_GROUPS = {"code": ["python", "java"], "web": ["c4"]}  # wiki is ungrouped


def _pack_group(
    max_length: int,
    *,
    num_bins: int | None = 8,
    id_to_name: dict[int, str],
    groups: dict[str, list[str]],
    **kwargs: Any,
) -> _MaterializingAccumulator:
    """Build a grouped-homogeneous accumulator with a component-id registry."""
    # num_bins is required for first/best and rejected by wrapping algorithms.
    if num_bins is not None:
        kwargs["num_bins"] = num_bins
    op = PackSequences(
        max_length=max_length,
        homogeneity="group",
        groups=DomainGroups(groups),
        **kwargs,
    )
    name_to_id = {name: cid for cid, name in id_to_name.items()}
    ctx = {"get_component_id": lambda name: name_to_id[name]}
    acc = op.accumulator(deterministic=False, ctx=ctx)
    assert isinstance(acc, PackingAccumulator)
    return _MaterializingAccumulator(op, acc)


def _packing_domain(rec: SampleRecord, id_to_name: dict[int, str]) -> set[str]:
    """The set of packing-domain groups a bin drew from (should be size 1)."""

    def group_of(name: str) -> str:
        for g, members in _GROUPS.items():
            if name in members:
                return g
        return name  # ungrouped singleton

    return {group_of(id_to_name[cid]) for cid in rec.meta.component_sample_counts}


def test_group_requires_groups() -> None:
    with pytest.raises(ValueError, match="requires groups"):
        PackSequences(max_length=4, num_bins=4, homogeneity="group")


def test_groups_rejected_without_group_mode() -> None:
    with pytest.raises(ValueError, match="only valid with homogeneity='group'"):
        PackSequences(
            max_length=4, num_bins=4, groups=DomainGroups({"code": ["python"]})
        )


def test_group_requires_component_id_service() -> None:
    op = PackSequences(
        max_length=4,
        num_bins=4,
        homogeneity="group",
        groups=DomainGroups({"code": ["python"]}),
    )
    with pytest.raises(ValueError, match="get_component_id"):
        op.accumulator(deterministic=False, ctx={})


def test_group_unknown_member_rejected_at_setup() -> None:
    # Member names resolve to ids at accumulator setup, so a misspelled member is
    # rejected eagerly (get_component_id raises) rather than silently self-grouping.
    op = PackSequences(
        max_length=4,
        num_bins=4,
        homogeneity="group",
        groups=DomainGroups({"code": ["python", "ghost"]}),
    )

    def get_component_id(name: str) -> int:
        ids = {"python": 0}
        if name not in ids:
            raise ValueError(f"Unknown mixture component {name!r}")
        return ids[name]

    with pytest.raises(ValueError, match="Unknown mixture component 'ghost'"):
        op.accumulator(deterministic=False, ctx={"get_component_id": get_component_id})


@pytest.mark.parametrize("algorithm", ["first_fit", "best_fit"])
def test_group_never_crosses_group_boundary(algorithm: str) -> None:
    acc = _pack_group(
        8,
        num_bins=8,
        length_fn=_simple_length_fn,
        drop_oversized=False,
        algorithm=algorithm,
        id_to_name=_ID_TO_NAME,
        groups=_GROUPS,
    )
    # Interleave code (python/java), web (c4), and ungrouped wiki. len 4 into
    # max_length 8 -> 2 per bin, so within a domain records pair up; a lane-only
    # key would instead pair across domains.
    recs = [_crec(i, 4, cid) for i, cid in enumerate([0, 2, 1, 3, 2, 0, 1])]
    out = _records(acc.push_many(recs)) + _records(acc.flush())
    # Never across a group boundary...
    assert all(len(_packing_domain(r, _ID_TO_NAME)) == 1 for r in out)
    # ...but within-group mixing (python+java) does happen for this algorithm.
    assert any(_bin_components(r) == {0, 1} for r in out)
    # Ungrouped-vs-grouped singletons: web (c4) and wiki keep their own bins.
    assert any(_bin_components(r) == {2} for r in out)
    assert any(_bin_components(r) == {3} for r in out)


def test_group_wrap_streams_per_group() -> None:
    acc = _pack_group(
        4,
        num_bins=None,  # wrap rejects num_bins
        algorithm="wrap",
        drop_oversized=False,
        id_to_name=_ID_TO_NAME,
        groups=_GROUPS,
    )
    # Interleave web (c4) between the two code records: a single shared buffer
    # would splice c4 into the code bin (crossing groups); per-group buffers keep
    # python+java together and c4 in its own bin.
    recs = [
        _crec_tokens(0, [1, 2], 0),  # python (code)
        _crec_tokens(1, [5, 6, 7, 8], 2),  # c4 (web) -> fills a web bin
        _crec_tokens(2, [3, 4], 1),  # java (code) -> completes the code bin
    ]
    out = _records(acc.push_many(recs)) + _records(acc.flush())
    assert all(len(_packing_domain(r, _ID_TO_NAME)) == 1 for r in out)
    # The code bin mixed python+java within the group.
    assert any(_bin_components(r) == {0, 1} for r in out)


def test_group_name_colliding_with_component_name_does_not_merge() -> None:
    id_to_name = {0: "c4", 1: "web"}  # cid 0 -> grouped; cid 1 -> ungrouped component
    acc = _pack_group(
        10,
        num_bins=8,
        length_fn=_simple_length_fn,
        drop_oversized=False,
        id_to_name=id_to_name,
        groups={"web": ["c4"]},  # group name collides with the cid-1 component name
    )
    out = _records(acc.push_many([_crec(0, 4, 0), _crec(1, 4, 1)])) + _records(
        acc.flush()
    )
    # Two separate single-component bins; a string-key collision would merge them
    # into one {0: 1, 1: 1} bin.
    assert len(out) == 2
    assert all(len(_bin_components(r)) == 1 for r in out)
    assert sorted(_bin_components(r).pop() for r in out) == [0, 1]


def _group_bin_key(rec: SampleRecord) -> tuple[int, str]:
    """(lane_id, sole packing-domain group) of a single-domain group bin."""
    doms = _packing_domain(rec, _ID_TO_NAME)
    assert len(doms) == 1
    return (rec.meta.lane_id, next(iter(doms)))


def test_group_partitions_by_lane_and_group() -> None:
    acc = _pack_group(
        10,
        num_bins=8,
        length_fn=_simple_length_fn,
        drop_oversized=False,
        id_to_name=_ID_TO_NAME,
        groups=_GROUPS,
    )
    recs = [
        _crec(0, 4, 0, lane=0),  # python (code), lane 0
        _crec(1, 4, 1, lane=0),  # java   (code), lane 0 -> joins the lane-0 code bin
        _crec(2, 4, 0, lane=1),  # python (code), lane 1 -> separate (different lane)
        _crec(3, 4, 2, lane=0),  # c4     (web),  lane 0 -> separate (different group)
    ]
    out = _records(acc.push_many(recs)) + _records(acc.flush())
    assert all(len(_packing_domain(r, _ID_TO_NAME)) == 1 for r in out)
    # Three bins: lane-0 code (python+java mixed), lane-1 code, lane-0 web.
    assert sorted(_group_bin_key(r) for r in out) == [
        (0, "code"),
        (0, "web"),
        (1, "code"),
    ]
    # The lane-0 code bin mixed both components; lane-1 code did NOT merge into it
    # (a lane collision would instead yield one code bin with {0: 2, 1: 1}).
    lane0_code = next(r for r in out if _group_bin_key(r) == (0, "code"))
    assert _bin_components(lane0_code) == {0, 1}


def test_group_num_bins_is_per_group() -> None:
    acc = _pack_group(
        10,
        num_bins=1,
        length_fn=_simple_length_fn,
        drop_oversized=False,
        algorithm="first_fit",
        id_to_name=_ID_TO_NAME,
        groups=_GROUPS,
    )
    # python (code) and c4 (web): different groups, both partial -> no eviction.
    ready = acc.push_many([_crec(0, 4, 0), _crec(2, 4, 2)])
    assert ready == []
    out = _records(acc.flush())
    assert sorted(_group_bin_key(r) for r in out) == [(0, "code"), (0, "web")]


def test_group_flush_order_is_deterministic() -> None:
    """Flush order is fixed by the sorted keys, independent of arrival order."""
    acc = _pack_group(
        10,
        num_bins=8,
        length_fn=_simple_length_fn,
        drop_oversized=False,
        id_to_name=_ID_TO_NAME,
        groups=_GROUPS,
    )
    # Arrive as web, wiki, code; each partial bin buffers until flush.
    ready = acc.push_many([_crec(0, 4, 2), _crec(1, 4, 3), _crec(2, 4, 0)])
    assert ready == []
    out = _records(acc.flush())
    # Sorted by tagged key: ("component", 3=wiki) < ("group", "code") < ("group", "web").
    assert [next(iter(_packing_domain(r, _ID_TO_NAME))) for r in out] == [
        "wiki",
        "code",
        "web",
    ]
