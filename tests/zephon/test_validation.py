"""Tests for the custom-op validation harness (``Pipeline.validate``).

Each broken-fixture test deliberately violates one contract and asserts
that the corresponding error code is reported.  The happy-path tests
confirm that legitimate ops pass validation and that the auto-run hook
gates iteration before any data flows.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon import Pipeline as PublicPipeline
from zephon.ops import Accumulator, BaseOp, CountingAccumulator, OpTraits, ReadyBatch
from zephon.types import SampleBatch, SampleMeta, SampleRecord
from zephon.validation import (
    ValidationError,
    ValidationReport,
    preflight_tokenizers,
)


def _empty_pipeline() -> PublicPipeline:
    ds = make_inmem_dataset("tiny", [{"text": "x"}])
    ws = FakeIndexableWorkSource(ds, chunk_size=1)
    return PublicPipeline(ws)


def _codes(report: ValidationReport) -> list[str]:
    return [i.code for i in report.issues]


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_validate_returns_ok_for_empty_pipeline() -> None:
    """No user ops → no issues."""
    report = _empty_pipeline().validate()
    assert report.ok
    assert report.issues == []


def test_validate_passes_for_stateless_passthrough_op() -> None:
    """A textbook stateless op + default accumulator validates cleanly."""
    pipe = _empty_pipeline().add_op(
        "stateless",
        process_many=lambda elems: list(elems),
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    assert report.ok, report.format()
    assert _codes(report) == []


def test_validate_passes_for_counting_accumulator() -> None:
    """A real-world pattern: per-lane windowing via CountingAccumulator."""
    pipe = _empty_pipeline().add_op(
        "windowed",
        process_many=lambda elems: list(elems),
        accumulator=lambda: CountingAccumulator(max_batch=4),
        preserves_cursor_order=False,
    )
    report = pipe.validate()
    assert report.ok, report.format()


# ---------------------------------------------------------------------------
# OP_NONDETERMINISTIC
# ---------------------------------------------------------------------------


def test_validate_catches_cross_call_state_leakage() -> None:
    """State that only surfaces between *different* inputs → OP_CROSS_CALL_STATE.

    Same-input determinism (covered by ``OP_NONDETERMINISTIC``) doesn't see
    state that depends on what input came before.  The A-B-A probe does:
    if calling with B between two calls to A changes A's output, the op is
    carrying state across invocations.
    """
    last_seen: dict[str, Any] = {}

    def cross_call_leaker(elems: list[Any]) -> list[Any]:
        out = list(elems)
        first_id = elems[0].meta.sample_id if elems else None
        if "id" in last_seen and last_seen["id"] != first_id:
            # Output depends on whether the previous batch had a different ID.
            out = list(reversed(out))
        last_seen["id"] = first_id
        return out

    pipe = _empty_pipeline().add_op(
        "leaker", process_many=cross_call_leaker, preserves_cursor_order=True
    )
    report = pipe.validate()
    assert "OP_CROSS_CALL_STATE" in _codes(report)


def test_validate_catches_nondeterministic_process_many() -> None:
    """A process_many with hidden state → OP_NONDETERMINISTIC."""
    counter = {"n": 0}

    def stateful(elems: list[Any]) -> list[Any]:
        # Hidden state leaks into output ordering.
        counter["n"] += 1
        return list(elems) if counter["n"] % 2 == 0 else list(reversed(elems))

    pipe = _empty_pipeline().add_op(
        "oops", process_many=stateful, preserves_cursor_order=True
    )
    report = pipe.validate()
    assert not report.ok
    assert "OP_NONDETERMINISTIC" in _codes(report)


def test_validate_catches_payload_level_nondeterminism() -> None:
    """Same sample IDs but different payloads must still be flagged.

    Sample-ID-only comparison would have passed this op — the payload
    counter mutates between calls without changing IDs.  This test
    locks in the structural comparison fix.
    """
    counter = {"n": 0}

    def payload_drifter(elems: list[Any]) -> list[Any]:
        counter["n"] += 1
        for elem in elems:
            payload = dict(elem.payload) if isinstance(elem.payload, dict) else {}
            payload["call_index"] = counter["n"]
            elem.payload = payload
        return list(elems)

    pipe = _empty_pipeline().add_op(
        "payload_drifter",
        process_many=payload_drifter,
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    assert "OP_NONDETERMINISTIC" in _codes(report)


def test_validate_catches_divergent_exception_behavior() -> None:
    """An op that succeeds once and crashes once is a determinism violation."""
    counter = {"n": 0}

    def flaky(elems: list[Any]) -> list[Any]:
        counter["n"] += 1
        if counter["n"] == 2:
            raise RuntimeError("intermittent failure")
        return list(elems)

    pipe = _empty_pipeline().add_op(
        "flaky", process_many=flaky, preserves_cursor_order=True
    )
    report = pipe.validate()
    assert "OP_NONDETERMINISTIC" in _codes(report)


def test_consistent_raise_degrades_to_warning_regardless_of_exception_type() -> None:
    """A consistent raise on synthetic input degrades to a non-blocking warning
    for *any* exception type, not just KeyError/AttributeError/TypeError.

    The validator only ever feeds one synthetic input shape and never sees the
    op succeed, so it can't distinguish "needs a specific payload" from "just
    broken" — the exception type is a leaky proxy.  A ValueError (which would
    arise from e.g. ``int(payload["text"])`` or ``json.loads``) must therefore
    degrade to OP_REJECTS_GENERIC_PAYLOAD (warning, non-blocking), exactly like
    a KeyError would — never to a blocking error.
    """

    def always_value_error(elems: list[Any]) -> list[Any]:
        raise ValueError("needs a payload the synthetic record lacks")

    pipe = _empty_pipeline().add_op(
        "raises_value_error",
        process_many=always_value_error,
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    codes = _codes(report)
    assert "OP_REJECTS_GENERIC_PAYLOAD" in codes
    # The collapsed code path: no blocking error for a consistent raise.
    assert report.ok


def test_validate_reports_loud_warning_on_shape_sensitive_op() -> None:
    """Shape-sensitive ops produce a *loud* warning that won't be mistaken for routine.

    A consistent raise on the synthetic input means the validator could not run
    any op-level check, so the warning text must say so explicitly — users must
    not read a green-looking report as ``ok``.
    """

    def needs_special_field(elems: list[Any]) -> list[Any]:
        return [e.payload["nonexistent_field"] for e in elems]

    pipe = _empty_pipeline().add_op(
        "needs_field",
        process_many=needs_special_field,
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    codes = _codes(report)
    assert "OP_REJECTS_GENERIC_PAYLOAD" in codes
    # Warning-only — validation should not block iteration.
    assert report.ok

    # The warning text must be loud and explicit about what was skipped.
    message = next(
        i.message for i in report.issues if i.code == "OP_REJECTS_GENERIC_PAYLOAD"
    )
    assert "NOT VALIDATED" in message, (
        "Warning must announce non-validation in unmistakable terms; got: " + message
    )
    # And it must enumerate what was skipped so users know the scope.
    for skipped_check in ("determinism", "cross-call", "sample identity"):
        assert skipped_check in message.lower(), (
            f"Warning must mention skipped check {skipped_check!r}; got: " + message
        )


# ---------------------------------------------------------------------------
# ACC_FLUSH_LOSES_SAMPLES
# ---------------------------------------------------------------------------


class _DroppingAccumulator(Accumulator[Any]):
    """Buggy accumulator that drops every other sample on flush."""

    def __init__(self) -> None:
        self._buf: list[Any] = []

    def push_many(self, elems: Any) -> list[ReadyBatch[Any]]:
        self._buf.extend(elems)
        return []

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[Any]]:
        kept = self._buf[::2]  # drop every other sample
        self._buf = []
        return [(kept, 0)] if kept else []

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        return bool(self._buf)


def test_validate_catches_flush_dropping_samples() -> None:
    pipe = _empty_pipeline().add_op(
        "dropper",
        process_many=lambda elems: list(elems),
        accumulator=_DroppingAccumulator,
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    assert "ACC_FLUSH_LOSES_SAMPLES" in _codes(report)


# ---------------------------------------------------------------------------
# ACC_RESET_NOT_FRESH
# ---------------------------------------------------------------------------


class _ResetIgnoringAccumulator(Accumulator[Any]):
    """flush(reset=True) silently fails to clear the buffer."""

    def __init__(self) -> None:
        self._buf: list[Any] = []

    def push_many(self, elems: Any) -> list[ReadyBatch[Any]]:
        self._buf.extend(elems)
        return []

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[Any]]:
        out = list(self._buf)
        if not reset:
            self._buf = []
        # When reset=True we deliberately keep _buf — bug.
        return [(out, 0)] if out else []

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        return bool(self._buf)


def test_validate_catches_reset_that_does_not_clear_state() -> None:
    pipe = _empty_pipeline().add_op(
        "leaky_reset",
        process_many=lambda elems: list(elems),
        accumulator=_ResetIgnoringAccumulator,
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    assert "ACC_RESET_NOT_FRESH" in _codes(report)


class _RoundCounterAccumulator(Accumulator[Any]):
    """Buggy accumulator whose reset preserves sample IDs but leaks a counter
    into output payloads — only catchable by structural (payload-aware)
    comparison, not by ID equality alone.
    """

    def __init__(self) -> None:
        self._buf: list[Any] = []
        self._round = 0

    def push_many(self, elems: Any) -> list[ReadyBatch[Any]]:
        self._buf.extend(elems)
        return []

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[Any]]:
        out: list[Any] = []
        for elem in self._buf:
            payload = dict(elem.payload) if isinstance(elem.payload, dict) else {}
            payload["round"] = self._round
            elem.payload = payload
            out.append(elem)
        self._buf = []
        if reset:
            # Bug: should set self._round = 0; instead we increment.
            self._round += 1
        return [(out, 0)] if out else []

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        return bool(self._buf)


def test_validate_catches_reset_that_leaks_state_into_payloads() -> None:
    """Reset must restore the accumulator's payload-affecting state, not just IDs.

    This fixture preserves sample IDs across reset (the old ID-only check
    would have passed) but stamps a monotonically-increasing ``round``
    counter into every output payload.  Locks in the structural comparison
    fix for ACC_RESET_NOT_FRESH.
    """
    pipe = _empty_pipeline().add_op(
        "round_counter",
        process_many=lambda elems: list(elems),
        accumulator=_RoundCounterAccumulator,
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    assert "ACC_RESET_NOT_FRESH" in _codes(report)


# ---------------------------------------------------------------------------
# ACC_HAS_PENDING_LIES
# ---------------------------------------------------------------------------


class _LyingPendingAccumulator(Accumulator[Any]):
    """has_pending_data() is always True regardless of state."""

    def __init__(self) -> None:
        self._buf: list[Any] = []

    def push_many(self, elems: Any) -> list[ReadyBatch[Any]]:
        self._buf.extend(elems)
        return []

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[Any]]:
        out = list(self._buf)
        self._buf = []
        return [(out, 0)] if out else []

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        return True  # always lies


def test_validate_catches_has_pending_data_liar() -> None:
    pipe = _empty_pipeline().add_op(
        "liar",
        process_many=lambda elems: list(elems),
        accumulator=_LyingPendingAccumulator,
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    assert "ACC_HAS_PENDING_LIES" in _codes(report)


class _AlwaysFalseAccumulator(Accumulator[Any]):
    """has_pending_data() is always False, even when items are buffered.

    A buffering accumulator that lies about emptiness — the failure mode the
    original two-corner check (fresh + post-flush) silently missed.
    """

    def __init__(self) -> None:
        self._buf: list[Any] = []

    def push_many(self, elems: Any) -> list[ReadyBatch[Any]]:
        self._buf.extend(elems)
        return []  # always buffer; never emit

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[Any]]:
        out = list(self._buf)
        self._buf = []
        return [(out, 0)] if out else []

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        return False  # always lies


def test_validate_catches_always_false_has_pending_data() -> None:
    """Accumulator buffers samples but reports no pending data → ACC_HAS_PENDING_LIES."""
    pipe = _empty_pipeline().add_op(
        "always_false",
        process_many=lambda elems: list(elems),
        accumulator=_AlwaysFalseAccumulator,
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    assert "ACC_HAS_PENDING_LIES" in _codes(report)


# ---------------------------------------------------------------------------
# Sample identity (op-level)
# ---------------------------------------------------------------------------


def test_validate_catches_fabricated_sample_ids() -> None:
    """An op that invents new sample_ids → OP_FABRICATES_SAMPLE_ID."""

    def fabricator(elems: list[SampleRecord]) -> list[SampleRecord]:
        out: list[SampleRecord] = []
        for i, e in enumerate(elems):
            new_meta = SampleMeta(
                sample_id=("NEW", i),  # fabricated — not in input
                lane_id=e.meta.lane_id,
                chunk_id=e.meta.chunk_id,
                chunk_offset=e.meta.chunk_offset,
            )
            out.append(SampleRecord(meta=new_meta, payload=e.payload))
        return out

    pipe = _empty_pipeline().add_op(
        "fab", process_many=fabricator, preserves_cursor_order=True
    )
    report = pipe.validate()
    assert "OP_FABRICATES_SAMPLE_ID" in _codes(report)


def test_validate_catches_fanout_without_child() -> None:
    """Multiple outputs with the same sample_id AND identical lineage → bug."""

    def bad_fanout(elems: list[SampleRecord]) -> list[SampleRecord]:
        # Duplicate every input without using SampleMeta.child() — the
        # duplicates share both sample_id AND lineage, which corrupts the
        # cursor ordering downstream consumers rely on.
        out: list[SampleRecord] = []
        for e in elems:
            for _ in range(3):
                out.append(SampleRecord(meta=e.meta, payload=e.payload))
        return out

    pipe = _empty_pipeline().add_op(
        "bad_fanout", process_many=bad_fanout, preserves_cursor_order=True
    )
    report = pipe.validate()
    assert "OP_FANOUT_DUPLICATE_LINEAGE" in _codes(report)


def test_validate_passes_for_correct_fanout_via_child() -> None:
    """Fan-out that uses SampleMeta.child() must validate cleanly."""

    def good_fanout(elems: list[SampleRecord]) -> list[SampleRecord]:
        out: list[SampleRecord] = []
        for e in elems:
            for i in range(2):
                child_meta = e.meta.child(i)
                out.append(SampleRecord(meta=child_meta, payload=e.payload))
        return out

    pipe = _empty_pipeline().add_op(
        "good_fanout", process_many=good_fanout, preserves_cursor_order=True
    )
    report = pipe.validate()
    codes = _codes(report)
    assert "OP_FABRICATES_SAMPLE_ID" not in codes
    assert "OP_FANOUT_DUPLICATE_LINEAGE" not in codes
    assert "OP_LINEAGE_NOT_VIA_CHILD" not in codes
    assert report.ok, report.format()


def test_validate_handles_op_emitting_sample_batch_outputs() -> None:
    """Ops downstream of `batch` emit `SampleBatch` wrappers, not raw
    `SampleRecord` instances.  The per-record sample-identity and lineage
    checks must flatten batch outputs rather than crashing with
    `AttributeError: 'SampleBatch' object has no attribute 'meta'`.

    Regression for the ``StreamItem = SampleRecord | SampleBatch`` case
    where the validator iterated outputs assuming the `SampleRecord`
    shape.
    """

    def emit_one_batch(elems: list[SampleRecord]) -> list[SampleBatch]:
        # Wrap every input record into a single SampleBatch — sample_ids
        # and lineages are preserved, so the per-record checks (after
        # flattening) should pass cleanly.
        return [SampleBatch(records=tuple(elems))]

    pipe = _empty_pipeline().add_op(
        "wrap_into_batch",
        process_many=emit_one_batch,
        preserves_cursor_order=True,
    )
    # The bug surfaced as AttributeError raised out of the validator
    # itself, not as a graceful issue — the assert here is that
    # `validate()` returns without crashing at all.
    report = pipe.validate()
    codes = _codes(report)
    assert "OP_FABRICATES_SAMPLE_ID" not in codes
    assert "OP_LINEAGE_NOT_VIA_CHILD" not in codes
    assert "OP_FANOUT_DUPLICATE_LINEAGE" not in codes
    assert report.ok, report.format()


# ---------------------------------------------------------------------------
# Lane handling (accumulator-level)
# ---------------------------------------------------------------------------


class _LaneScramblingAccumulator(Accumulator[Any]):
    """Buggy accumulator that rewrites only ``lane_id`` on every pushed sample.

    Uses ``dataclasses.replace`` so other ``SampleMeta`` fields (including
    ``lineage``) are preserved — this isolates the failure mode to lane_id
    corruption regardless of what the synthetic input's other fields are.
    """

    def __init__(self) -> None:
        self._buf: list[Any] = []

    def push_many(self, elems: Any) -> list[ReadyBatch[Any]]:
        for elem in elems:
            elem.meta = dataclasses.replace(
                elem.meta, lane_id=(elem.meta.lane_id + 1) % 3
            )
            self._buf.append(elem)
        return []

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[Any]]:
        out = list(self._buf)
        self._buf = []
        return [(out, 0)] if out else []

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        return bool(self._buf)


def test_validate_catches_accumulator_mutating_lane_id() -> None:
    pipe = _empty_pipeline().add_op(
        "scrambler",
        process_many=lambda elems: list(elems),
        accumulator=_LaneScramblingAccumulator,
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    assert "ACC_MUTATES_LANE_ID" in _codes(report)


class _CrossPollinatingAccumulator(Accumulator[Any]):
    """Bucketing accumulator that emits batches of 3 samples regardless of lane.

    Mixed-lane input feeds in interleaved by lane (lane 0, lane 1, lane 2,
    lane 0, ...), and this accumulator emits every group of 3 as a batch
    without per-lane routing.  Result: every emitted batch contains three
    distinct lanes — a clear lane-purity violation a real bucketing
    accumulator must not produce.
    """

    def __init__(self) -> None:
        self._buf: list[Any] = []

    def push_many(self, elems: Any) -> list[ReadyBatch[Any]]:
        ready: list[ReadyBatch[Any]] = []
        for elem in elems:
            self._buf.append(elem)
            if len(self._buf) >= 3:
                ready.append((list(self._buf), 0))
                self._buf = []
        return ready

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[Any]]:
        out = list(self._buf)
        self._buf = []
        return [(out, 0)] if out else []

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        return bool(self._buf)


def test_validate_catches_lane_mixing_bucketing_accumulator() -> None:
    """A bucketing accumulator emitting cross-lane batches → ACC_BATCH_NOT_LANE_PURE."""
    pipe = _empty_pipeline().add_op(
        "cross_pollinator",
        process_many=lambda elems: list(elems),
        accumulator=_CrossPollinatingAccumulator,
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    assert "ACC_BATCH_NOT_LANE_PURE" in _codes(report)


def test_validate_skips_lane_purity_for_passthrough_shape() -> None:
    """Passthrough-shape accumulators are exempt from lane-purity (upstream's job)."""
    pipe = _empty_pipeline().add_op(
        "passthrough_op",
        process_many=lambda elems: list(elems),
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    assert "ACC_BATCH_NOT_LANE_PURE" not in _codes(report)


def test_validate_passes_lane_purity_for_lane_keyed_counting_accumulator() -> None:
    """CountingAccumulator keyed by lane must not trip the check."""
    pipe = _empty_pipeline().add_op(
        "lane_keyed",
        process_many=lambda elems: list(elems),
        accumulator=lambda: CountingAccumulator(max_batch=2),
        preserves_cursor_order=False,
    )
    report = pipe.validate()
    assert "ACC_BATCH_NOT_LANE_PURE" not in _codes(report)


# ---------------------------------------------------------------------------
# ACC_NO_PER_LANE_FLUSH / ACC_PER_LANE_FLUSH_NOT_DRAINED
# ---------------------------------------------------------------------------


def _emit_lane_pure(buf: list[Any]) -> list[ReadyBatch[Any]]:
    """Drain ``buf`` into one lane-pure batch per lane (a valid flush shape)."""
    by_lane: dict[Any, list[Any]] = {}
    for rec in buf:
        by_lane.setdefault(rec.meta.lane_id, []).append(rec)
    return [(batch, 0) for batch in by_lane.values()]


class _NoLaneIdAccumulator(Accumulator[Any]):
    """Pre-per-lane signature: ``flush``/``has_pending_data`` omit ``lane_id``.

    Otherwise correct — the runner flushes one lane at a time
    (``flush(reset=True, lane_id=lane)``), so this raises at the first epoch
    boundary even though every other contract holds.
    """

    def __init__(self) -> None:
        self._buf: list[Any] = []

    def push_many(self, elems: Any) -> list[ReadyBatch[Any]]:
        self._buf.extend(elems)
        return []

    def flush(self, *, reset: bool = False) -> list[ReadyBatch[Any]]:
        out = _emit_lane_pure(self._buf)
        self._buf = []
        return out

    def has_pending_data(self) -> bool:
        return bool(self._buf)


def test_validate_catches_accumulator_without_lane_id() -> None:
    pipe = _empty_pipeline().add_op(
        "no_lane_id",
        process_many=lambda elems: list(elems),
        accumulator=_NoLaneIdAccumulator,
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    assert "ACC_NO_PER_LANE_FLUSH" in _codes(report)


class _LaneFlushNoOpAccumulator(Accumulator[Any]):
    """Accepts ``lane_id`` but ignores it: a per-lane flush drains nothing.

    Only the all-lanes flush (``lane_id is None``) clears the buffer, so a
    per-lane flush leaves the target lane pending — exactly what the runner's
    post-flush guard rejects.
    """

    def __init__(self) -> None:
        self._buf: list[Any] = []

    def push_many(self, elems: Any) -> list[ReadyBatch[Any]]:
        self._buf.extend(elems)
        return []

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[Any]]:
        if lane_id is not None:
            return []  # bug: drops the per-lane flush on the floor
        out = _emit_lane_pure(self._buf)
        self._buf = []
        return out

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        return bool(self._buf)


def test_validate_catches_per_lane_flush_that_does_not_drain() -> None:
    pipe = _empty_pipeline().add_op(
        "lane_flush_noop",
        process_many=lambda elems: list(elems),
        accumulator=_LaneFlushNoOpAccumulator,
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    assert "ACC_PER_LANE_FLUSH_NOT_DRAINED" in _codes(report)


class _AllLaneFlushAccumulator(Accumulator[Any]):
    """Accepts ``lane_id`` but flushes *every* lane regardless.

    The target lane drains (so the not-drained check passes), but the other
    lanes are emitted and cleared too — the cross-lane corruption a per-lane
    flush must never cause.
    """

    def __init__(self) -> None:
        self._buf: list[Any] = []

    def push_many(self, elems: Any) -> list[ReadyBatch[Any]]:
        self._buf.extend(elems)
        return []

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[Any]]:
        out = _emit_lane_pure(self._buf)  # bug: ignores lane_id, drains all lanes
        self._buf = []
        return out

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        return bool(self._buf)


def test_validate_catches_per_lane_flush_that_leaks_other_lanes() -> None:
    pipe = _empty_pipeline().add_op(
        "all_lane_flush",
        process_many=lambda elems: list(elems),
        accumulator=_AllLaneFlushAccumulator,
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    assert "ACC_FLUSH_LEAKS_OTHER_LANES" in _codes(report)


# ---------------------------------------------------------------------------
# Auto-run gating via __iter__
# ---------------------------------------------------------------------------


def test_iter_raises_validation_error_for_broken_op() -> None:
    """A broken op must block __iter__ before any data flows."""
    counter = {"n": 0}

    def stateful(elems: list[Any]) -> list[Any]:
        counter["n"] += 1
        return list(elems) if counter["n"] % 2 == 0 else list(reversed(elems))

    pipe = (
        _empty_pipeline()
        .add_op("bad", process_many=stateful, preserves_cursor_order=True)
        .options(deterministic=True, prefetch_batches=0, default_stage_prefetch=0)
    )

    with pytest.raises(ValidationError) as excinfo:
        iter(pipe)
    assert "OP_NONDETERMINISTIC" in excinfo.value.report.format()


def _broken_op_pipeline() -> PublicPipeline:
    """Pipeline whose user op trips OP_NONDETERMINISTIC under validation."""
    counter = {"n": 0}

    def stateful(elems: list[Any]) -> list[Any]:
        counter["n"] += 1
        return list(elems) if counter["n"] % 2 == 0 else list(reversed(elems))

    return (
        _empty_pipeline()
        .add_op("bad", process_many=stateful, preserves_cursor_order=True)
        .options(deterministic=True, prefetch_batches=0, default_stage_prefetch=0)
    )


def test_auto_validation_warn_mode_emits_warning_and_proceeds() -> None:
    """`auto_validation="warn"` surfaces the report via warnings.warn but
    must not raise — escape hatch for false positives."""
    import warnings as _warnings

    pipe = _broken_op_pipeline().options(auto_validation="warn")

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        it = iter(pipe)
        try:
            next(it, None)
        finally:
            it.close()

    messages = [str(w.message) for w in caught]
    assert any("OP_NONDETERMINISTIC" in m for m in messages), (
        f"warn mode should surface the validation report, got: {messages}"
    )


def test_auto_validation_off_mode_skips_validation_entirely() -> None:
    """`auto_validation="off"` must not raise and must not emit the report."""
    import warnings as _warnings

    pipe = _broken_op_pipeline().options(auto_validation="off")

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        it = iter(pipe)
        try:
            next(it, None)
        finally:
            it.close()

    messages = [str(w.message) for w in caught]
    assert not any("OP_NONDETERMINISTIC" in m for m in messages), (
        f"off mode should not surface validation warnings, got: {messages}"
    )


def test_auto_validation_strict_mode_is_the_default() -> None:
    """The default of `auto_validation` is `strict` — same behavior as before."""
    pipe = _broken_op_pipeline()
    assert pipe._options.auto_validation == "strict"
    with pytest.raises(ValidationError):
        iter(pipe)


def test_auto_validation_strict_surfaces_warning_only_issues() -> None:
    """Warning-only codes (e.g. OP_REJECTS_GENERIC_PAYLOAD) must reach the
    user under strict mode.  Previously strict only raised on errors and
    swallowed every warning, so the explicit "!! THIS OP WAS NOT
    VALIDATED !!" messaging was invisible unless the user manually
    called ``pipe.validate()`` or switched to ``"warn"``."""
    import warnings as _warnings

    def needs_special_field(elems: list[Any]) -> list[Any]:
        return [e.payload["nonexistent_field"] for e in elems]

    pipe = _empty_pipeline().add_op(
        "needs_field", process_many=needs_special_field, preserves_cursor_order=True
    )
    assert pipe._options.auto_validation == "strict"

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        # Validation fires at iter() time, before any data pull.  We
        # only assert that iter() did not raise and that the warning
        # surfaced — pulling next() would re-execute process_many on
        # real input that also lacks the field, which is unrelated.
        it = iter(pipe)
        it.close()

    messages = [str(w.message) for w in caught]
    assert any("OP_REJECTS_GENERIC_PAYLOAD" in m for m in messages), (
        f"strict mode should surface warning-only issues, got: {messages}"
    )


def test_validation_caches_per_graph_generation() -> None:
    """Validation must not re-run on every iter() call for the same graph."""
    pipe = _empty_pipeline().add_op(
        "ok", process_many=lambda elems: list(elems), preserves_cursor_order=True
    )

    pipe._run_auto_validation()
    assert pipe._validated is True

    # Mutating the graph (adding another op) must invalidate the validation flag.
    pipe.add_op(
        "ok2", process_many=lambda elems: list(elems), preserves_cursor_order=True
    )
    assert pipe._validated is False


def test_options_call_invalidates_validation_cache() -> None:
    """Toggling `auto_validation` from "off" back to "strict" must re-run
    the harness.  Regression: `options()` previously only cleared
    `_runtime_spec`, so a prior iter() with `auto_validation="off"`
    left `_validated=True`, and a subsequent `options(auto_validation="strict")`
    + iter() bypassed validation entirely (the early-return on
    `_validated` short-circuited the harness)."""
    pipe = _broken_op_pipeline().options(auto_validation="off")

    # First iter under "off" — runs cleanly because the harness is skipped.
    it = iter(pipe)
    try:
        next(it, None)
    finally:
        it.close()
    assert pipe._validated is True

    # Flip back to strict.  options() must reset _validated so the harness
    # actually runs on the next iter().
    pipe.options(auto_validation="strict")
    assert pipe._validated is False, (
        "options() did not invalidate the validation cache — strict mode "
        "would silently skip validation."
    )
    with pytest.raises(ValidationError):
        iter(pipe)


def test_report_format_includes_doc_link_and_op_name() -> None:
    """Error messages must point users at the contract docs."""
    pipe = _empty_pipeline().add_op(
        "dropper",
        process_many=lambda elems: list(elems),
        accumulator=_DroppingAccumulator,
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    text = report.format()
    assert "dropper" in text
    assert "datologyai.github.io/zephon" in text
    assert "accumulators_operators" in text


# ---------------------------------------------------------------------------
# Static-analysis heuristics — run regardless of synthetic-payload acceptance
#
# Two attachment shapes are covered:
#   1. `BaseOp` subclasses attached via the instance form `add_op(MyOp())`
#      — the natural target for `self.x = ...` writes.
#   2. Bound methods threaded through the kwargs form
#      `add_op(name, process_many=instance.method, ...)` — same AST shape,
#      so the same scanner fires.
# Both cases are exercised below.
# ---------------------------------------------------------------------------


def _self_writer(self: Any, elems: list[Any]) -> list[Any]:
    """Module-level method whose source contains a self.<attr> write.

    Defined at module scope (not nested) so ``inspect.getsource`` can
    locate it — the AST scanner relies on getsource succeeding.
    """
    self.counter = getattr(self, "counter", 0) + 1
    return list(elems)


def test_writes_to_self_static_scan_flags_attribute_assignment() -> None:
    """The AST scanner fires for `self.x = ...` on a bound method passed via
    the kwargs form of add_op."""

    class _Wrapper:
        pass

    wrapper = _Wrapper()
    bound = _self_writer.__get__(wrapper, _Wrapper)

    pipe = _empty_pipeline().add_op(
        "self_assigner", process_many=bound, preserves_cursor_order=True
    )
    report = pipe.validate()
    assert "OP_WRITES_TO_SELF" in _codes(report)


def _list_appender(self: Any, elems: list[Any]) -> list[Any]:
    self._seen.append(len(elems))
    return list(elems)


def test_writes_to_self_static_scan_flags_mutator_call() -> None:
    """List/dict mutator calls on self are flagged."""

    class _Holder:
        def __init__(self) -> None:
            self._seen: list[int] = []

    holder = _Holder()
    bound = _list_appender.__get__(holder, _Holder)

    pipe = _empty_pipeline().add_op(
        "self_appender", process_many=bound, preserves_cursor_order=True
    )
    report = pipe.validate()
    assert "OP_WRITES_TO_SELF" in _codes(report)


def _self_writer_that_rejects_payload(self: Any, elems: list[Any]) -> list[Any]:
    """Combines `self.x = ...` with a payload-shape requirement.

    Used to verify the static scan still runs when the runtime probe is
    skipped via OP_REJECTS_GENERIC_PAYLOAD.
    """
    self.counter = getattr(self, "counter", 0) + 1
    return [e.payload["nonexistent_field"] for e in elems]


def test_writes_to_self_runs_even_when_op_rejects_generic_payload() -> None:
    """Static AST checks must fire even when the runtime probe is skipped."""

    class _Holder:
        pass

    bound = _self_writer_that_rejects_payload.__get__(_Holder(), _Holder)

    pipe = _empty_pipeline().add_op(
        "self_writer_picky", process_many=bound, preserves_cursor_order=True
    )
    report = pipe.validate()
    codes = _codes(report)
    # The runtime probe was skipped (this proves the AST check is independent).
    assert "OP_REJECTS_GENERIC_PAYLOAD" in codes
    # The AST scan still surfaced the suspicious write.
    assert "OP_WRITES_TO_SELF" in codes


def _uses_random_random(elems: list[Any]) -> list[Any]:
    import random

    random.random()  # noqa: returns a fresh draw — non-deterministic
    return list(elems)


def test_nondeterministic_stdlib_static_scan_flags_random() -> None:
    pipe = _empty_pipeline().add_op(
        "randy", process_many=_uses_random_random, preserves_cursor_order=True
    )
    report = pipe.validate()
    assert "OP_NONDETERMINISTIC_STDLIB_CALL" in _codes(report)


def _uses_time_time(elems: list[Any]) -> list[Any]:
    import time

    _ = time.time()
    return list(elems)


def test_nondeterministic_stdlib_static_scan_flags_time_time() -> None:
    pipe = _empty_pipeline().add_op(
        "clocky", process_many=_uses_time_time, preserves_cursor_order=True
    )
    report = pipe.validate()
    assert "OP_NONDETERMINISTIC_STDLIB_CALL" in _codes(report)


def _uses_random_but_rejects_payload(elems: list[Any]) -> list[Any]:
    import random

    random.random()
    return [e.payload["nonexistent_field"] for e in elems]


def test_nondeterministic_stdlib_runs_even_when_op_rejects_generic_payload() -> None:
    pipe = _empty_pipeline().add_op(
        "randy_picky",
        process_many=_uses_random_but_rejects_payload,
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    codes = _codes(report)
    assert "OP_REJECTS_GENERIC_PAYLOAD" in codes
    assert "OP_NONDETERMINISTIC_STDLIB_CALL" in codes


def test_state_diff_catches_closure_mutation_via_cloudpickle() -> None:
    """Cloudpickle-based state diff fires for closure-captured dict mutation."""
    seen: dict[str, int] = {}

    def closure_mutator(elems: list[Any]) -> list[Any]:
        seen["calls"] = seen.get("calls", 0) + 1
        return list(elems)

    pipe = _empty_pipeline().add_op(
        "closure_leaker",
        process_many=closure_mutator,
        preserves_cursor_order=True,
    )
    report = pipe.validate()
    assert "OP_STATE_CHANGED_DURING_CALL" in _codes(report)


def test_static_checks_are_warnings_not_errors() -> None:
    """All three heuristics report warnings; report.ok stays True."""

    def heuristic_triggerer(self: Any, elems: list[Any]) -> list[Any]:
        import random

        self.touched = True
        random.random()
        return list(elems)

    class _Holder:
        pass

    bound = heuristic_triggerer.__get__(_Holder(), _Holder)

    pipe = _empty_pipeline().add_op(
        "warn_only", process_many=bound, preserves_cursor_order=True
    )
    report = pipe.validate()
    codes = _codes(report)
    assert "OP_WRITES_TO_SELF" in codes
    assert "OP_NONDETERMINISTIC_STDLIB_CALL" in codes
    # Heuristics are warnings — they must not flip the report to not-ok on
    # their own. (OP_NONDETERMINISTIC may still fire from the runtime probe;
    # this test just confirms the static checks themselves are warning-level.)
    static_only = [
        i
        for i in report.issues
        if i.code in {"OP_WRITES_TO_SELF", "OP_NONDETERMINISTIC_STDLIB_CALL"}
    ]
    assert all(i.severity == "warning" for i in static_only)


# ---------------------------------------------------------------------------
# BaseOp subclass attachment — the natural target for the AST heuristics
# ---------------------------------------------------------------------------


class _SelfWritingOp(BaseOp):
    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True)

    def process_many(self, elems: list[Any]) -> list[Any]:
        self._counter = getattr(self, "_counter", 0) + 1
        return list(elems)


def test_static_scan_flags_self_writes_in_baseop_subclass() -> None:
    """Instance form: `BaseOp` subclass with `self.x = ...` in process_many
    must be flagged by the static AST scan."""
    pipe = _empty_pipeline().add_op(_SelfWritingOp())
    report = pipe.validate()
    assert "OP_WRITES_TO_SELF" in _codes(report)


class _PassThroughOp(BaseOp):
    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True)

    def process_many(self, elems: list[Any]) -> list[Any]:
        return list(elems)


def test_baseop_subclass_clean_op_passes_validation() -> None:
    """A clean `BaseOp` subclass attached via the instance form must not error.

    Instance-form ops without `validation_samples()` surface the
    `OP_INSTANCE_RUNTIME_CHECKS_SKIPPED` warning by design (we don't call
    `setup()` during validation, so runtime probes are gated on
    user-supplied records).  Static AST checks still run cleanly, and the
    report's `ok` flag is True (warnings don't escalate)."""
    pipe = _empty_pipeline().add_op(_PassThroughOp())
    report = pipe.validate()
    assert report.ok, f"unexpected error-severity issues: {report.format()}"
    codes = _codes(report)
    # Skipped-runtime is the only expected warning here.
    assert codes == ["OP_INSTANCE_RUNTIME_CHECKS_SKIPPED"]


def test_validator_skips_framework_builtin_ops() -> None:
    """Built-in ops (zephon.ops.*) must remain unprobed — they have their own tests."""
    pipe = _empty_pipeline().decode_text().tokenize(tokenizer_id="gpt2", field="text")
    report = pipe.validate()
    # Validator should produce no issues against built-ins on this graph.
    assert report.ok
    # And specifically should not have run static checks against framework code.
    for issue in report.issues:
        assert issue.op_name not in {"decode_text", "tokenize", "fetch"}


class _SubclassWithDefensiveSetupGuard(BaseOp):
    """Models a defensive `if self._tokenizer is None: raise RuntimeError`
    pattern.  Without the instance-form gate, the synthetic-input probe would
    call `process_many` pre-`setup` and trip the NOT-VALIDATED warning
    (`OP_REJECTS_GENERIC_PAYLOAD`) on every such op.  The gate replaces that
    generic noise with the clearer `OP_INSTANCE_RUNTIME_CHECKS_SKIPPED`
    message that explains the `validation_samples` opt-in."""

    def __init__(self) -> None:
        super().__init__()
        self._tokenizer: Any = None  # populated by setup

    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True)

    def setup(self, ctx: Any) -> None:
        super().setup(ctx)
        self._tokenizer = object()

    def process_many(self, elems: list[Any]) -> list[Any]:
        if self._tokenizer is None:
            raise RuntimeError("setup() not called")
        return list(elems)


def test_baseop_subclass_with_defensive_setup_guard_does_not_block_iteration() -> None:
    """Regression for maxi's report: a defensive `RuntimeError("setup() not called")`
    must not block iteration under `auto_validation="strict"`.  The skip gate
    keeps runtime checks from ever calling `process_many` here, and surfaces the
    clearer `OP_INSTANCE_RUNTIME_CHECKS_SKIPPED` message rather than the generic
    NOT-VALIDATED warning."""
    pipe = _empty_pipeline().add_op(_SubclassWithDefensiveSetupGuard())
    report = pipe.validate()
    codes = _codes(report)
    assert "OP_INSTANCE_RUNTIME_CHECKS_SKIPPED" in codes
    assert "OP_REJECTS_GENERIC_PAYLOAD" not in codes
    assert report.ok


def test_op_rejects_generic_payload_message_points_at_validation_samples_hook() -> None:
    """The OP_REJECTS_GENERIC_PAYLOAD warning must tell users about the
    `validation_samples` escape hatch."""

    def needs_special_field(elems: list[Any]) -> list[Any]:
        return [e.payload["nonexistent_field"] for e in elems]

    pipe = _empty_pipeline().add_op(
        "needs_field", process_many=needs_special_field, preserves_cursor_order=True
    )
    report = pipe.validate()
    msg = next(
        i.message for i in report.issues if i.code == "OP_REJECTS_GENERIC_PAYLOAD"
    )
    assert "validation_samples" in msg


# ---------------------------------------------------------------------------
# `validation_samples` — user-supplied records unlock the full check suite
# ---------------------------------------------------------------------------


def _custom_records(field: str) -> list[SampleRecord]:
    """Records carrying a non-generic payload key.  Two lanes so the
    cross-call-state probe stays meaningful."""
    records: list[SampleRecord] = []
    for lane in (0, 1):
        for offset in range(3):
            meta = SampleMeta(
                sample_id=(0, lane, offset),
                lane_id=lane,
                chunk_id=offset // 2,
                chunk_offset=offset,
            )
            records.append(
                SampleRecord(meta=meta, payload={field: f"v_{lane}_{offset}"})
            )
    return records


def test_validation_samples_kwarg_unlocks_op_level_checks_for_kwargs_form() -> None:
    """An op that reads a custom payload key is fully validated when
    `validation_samples` supplies matching records — no fallthrough to
    OP_REJECTS_GENERIC_PAYLOAD."""

    def needs_tokens(elems: list[SampleRecord]) -> list[SampleRecord]:
        for r in elems:
            assert isinstance(r.payload, dict) and "tokens" in r.payload
        return list(elems)

    pipe = _empty_pipeline().add_op(
        "tokenized",
        process_many=needs_tokens,
        preserves_cursor_order=True,
        validation_samples=lambda: _custom_records("tokens"),
    )
    report = pipe.validate()
    codes = _codes(report)
    assert "OP_REJECTS_GENERIC_PAYLOAD" not in codes
    assert report.ok


class _SubclassWithValidationSamples(BaseOp):
    """Models the canonical pattern from the docs: setup builds state and
    the user supplies records that work pre-setup so the validator can
    still probe `process_many`."""

    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True)

    def validation_samples(self) -> list[SampleRecord]:
        return _custom_records("ids")

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        for r in elems:
            assert isinstance(r.payload, dict) and "ids" in r.payload
        return list(elems)


def test_validation_samples_method_unlocks_op_level_checks_for_baseop_subclass() -> (
    None
):
    """Overriding `validation_samples()` on a BaseOp subclass has the same
    effect as the kwargs-form factory."""
    pipe = _empty_pipeline().add_op(_SubclassWithValidationSamples())
    report = pipe.validate()
    codes = _codes(report)
    assert "OP_REJECTS_GENERIC_PAYLOAD" not in codes
    assert report.ok


def test_validation_samples_factory_that_raises_surfaces_warning_and_falls_back() -> (
    None
):
    """A buggy factory must not break validation — emit
    OP_VALIDATION_SAMPLES_FACTORY_FAILED and use synthetic records."""

    def explode() -> list[SampleRecord]:
        raise RuntimeError("oops")

    def fine(elems: list[SampleRecord]) -> list[SampleRecord]:
        return list(elems)

    pipe = _empty_pipeline().add_op(
        "kvargs_factory_blows_up",
        process_many=fine,
        preserves_cursor_order=True,
        validation_samples=explode,
    )
    report = pipe.validate()
    codes = _codes(report)
    assert "OP_VALIDATION_SAMPLES_FACTORY_FAILED" in codes
    # The validator survived: report shouldn't be erroring, op-level checks
    # still ran against synthetic fallback records.
    assert report.ok


def test_validation_samples_factory_returning_wrong_type_surfaces_warning() -> None:
    """Non-list / wrong-element-type returns also fall back with a warning."""

    pipe = _empty_pipeline().add_op(
        "wrong_type",
        process_many=lambda elems: list(elems),
        preserves_cursor_order=True,
        validation_samples=lambda: "not a list of records",  # type: ignore[arg-type,return-value]
    )
    report = pipe.validate()
    codes = _codes(report)
    assert "OP_VALIDATION_SAMPLES_FACTORY_FAILED" in codes
    assert report.ok


def test_validation_samples_kwarg_rejected_with_baseop_instance_form() -> None:
    """The kwarg is forbidden when attaching a BaseOp instance — subclasses
    override the method instead."""

    class Trivial(BaseOp):
        def traits(self) -> OpTraits:
            return OpTraits(preserves_cursor_order=True)

        def process_many(self, elems: list[Any]) -> list[Any]:
            return list(elems)

    pipe = _empty_pipeline()
    with pytest.raises(ValueError, match="validation_samples"):
        pipe.add_op(Trivial(), validation_samples=lambda: _custom_records("ids"))  # type: ignore[call-overload]


def test_trait_kwargs_rejected_with_baseop_instance_form() -> None:
    """Trait kwargs (parallelism, indexable, batch_shape_sensitive,
    requires_serial_state, stall_on_epoch_boundary) belong on the op's
    ``traits()`` for the instance form.  Passing them as kwargs alongside
    a BaseOp instance previously was silently dropped — the engine would
    use the op's own traits, leaving the user thinking their kwarg took
    effect.  The instance form must reject these explicitly.  Probed with
    ``parallelism`` as a representative — all five share the same
    mismatched-list code path."""

    class _Trivial(BaseOp):
        def traits(self) -> OpTraits:
            return OpTraits(preserves_cursor_order=True)

        def process_many(self, elems: list[Any]) -> list[Any]:
            return list(elems)

    pipe = _empty_pipeline()
    with pytest.raises(ValueError, match="parallelism"):
        pipe.add_op(_Trivial(), parallelism=8)  # type: ignore[call-overload]


def test_instance_form_accepts_name_and_placement_only() -> None:
    """The static contract on the instance overload only declares ``name``
    and ``placement``; runtime must accept those and honor ``op.traits()``
    rather than any kwarg override path.  This is the only test that
    asserts ``op.traits()`` actually reaches the node for the instance
    form — happy paths elsewhere don't probe traits explicitly."""

    class _TraitsCarrying(BaseOp):
        def traits(self) -> OpTraits:
            return OpTraits(preserves_cursor_order=True, parallelism=4)

        def process_many(self, elems: list[Any]) -> list[Any]:
            return list(elems)

    pipe = _empty_pipeline().add_op(_TraitsCarrying(), name="custom", placement="auto")
    node = pipe._graph.nodes[-1]
    assert node.name == "custom"
    assert node.op.traits().parallelism == 4


def test_name_kwarg_rejected_with_kwargs_form() -> None:
    """The kwargs form takes the op name positionally; passing `name=` too was
    silently ignored (the positional won), so it must be rejected — symmetric
    with the instance form rejecting mismatched kwargs."""
    pipe = _empty_pipeline()
    with pytest.raises(ValueError, match="name="):
        pipe.add_op(  # type: ignore[call-overload]
            "real_name",
            name="other",
            process_many=lambda elems: list(elems),
            preserves_cursor_order=True,
        )


# ---------------------------------------------------------------------------
# OP_OUTPUT_NOT_STREAM_ITEM / OP_VALIDATOR_CRASHED — the harness must never
# escape iter() with a raw exception, only a ValidationReport.
# ---------------------------------------------------------------------------


def test_validate_reports_non_stream_item_output_cleanly() -> None:
    """An op returning bare payloads (not StreamItem) is reported, not crashed.

    Previously the structural checks reached into ``record.meta.sample_id``
    and died with ``AttributeError: 'dict' object has no attribute 'meta'``.
    Now it surfaces a clean OP_OUTPUT_NOT_STREAM_ITEM error and the
    structural checks that assume well-formed output are skipped (no
    spurious OP_VALIDATOR_CRASHED noise)."""
    pipe = _empty_pipeline().add_op(
        "p",
        process_many=lambda es: [e.payload for e in es],
        preserves_cursor_order=True,
    )

    report = pipe.validate()  # must not raise AttributeError
    codes = _codes(report)
    assert "OP_OUTPUT_NOT_STREAM_ITEM" in codes
    assert "OP_VALIDATOR_CRASHED" not in codes
    # Structural checks are skipped, so no determinism/identity noise.
    assert "OP_NONDETERMINISTIC" not in codes
    assert "OP_FABRICATES_SAMPLE_ID" not in codes


def test_non_stream_item_output_warn_mode_does_not_crash_iter() -> None:
    """`auto_validation="warn"` must survive a non-StreamItem op.

    The crash used to escape ``validate()`` itself, so even warn mode — the
    documented false-positive escape hatch — raised AttributeError. iter()
    must succeed and surface the report as a warning instead."""
    import warnings as _warnings

    pipe = (
        _empty_pipeline()
        .add_op(
            "p",
            process_many=lambda es: [e.payload for e in es],
            preserves_cursor_order=True,
        )
        .options(auto_validation="warn")
    )

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        it = iter(pipe)  # must not raise
        it.close()

    messages = [str(w.message) for w in caught]
    assert any("OP_OUTPUT_NOT_STREAM_ITEM" in m for m in messages), messages


def test_unexpected_check_crash_becomes_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A check that raises unexpectedly is converted into an OP_VALIDATOR_CRASHED
    error Issue rather than escaping the harness as a raw traceback."""
    import zephon.validation as _validate

    def _boom(*_args: Any, **_kwargs: Any) -> list[Any]:
        raise RuntimeError("synthetic validator bug")

    monkeypatch.setattr(_validate, "_check_op_determinism", _boom)

    pipe = _empty_pipeline().add_op(
        "stateless",
        process_many=lambda elems: list(elems),
        preserves_cursor_order=True,
    )

    report = pipe.validate()  # must not raise RuntimeError
    crashed = [i for i in report.issues if i.code == "OP_VALIDATOR_CRASHED"]
    assert crashed, _codes(report)
    assert "synthetic validator bug" in crashed[0].message


class _PayloadReadingAccumulator(Accumulator[Any]):
    """Reads payload["tokens"] — raises KeyError on the synthetic record.

    Mirrors a real ``Accumulator.reads_payload`` accumulator that the
    validator can't feed realistic records to.  push_many must blow up on the
    synthetic ``{"text", "value"}`` payload."""

    def __init__(self) -> None:
        self._buf: list[Any] = []

    def push_many(self, elems: Any) -> list[ReadyBatch[Any]]:
        for e in elems:
            _ = e.payload["tokens"]  # KeyError on synthetic input
        self._buf.extend(elems)
        return []

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[Any]]:
        out = list(self._buf)
        self._buf = []
        return [(out, 0)] if out else []

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        return bool(self._buf)


def test_payload_reading_accumulator_does_not_crash_validator() -> None:
    """An accumulator that raises on synthetic input must not escape validate().

    ``_check_acc_has_pending_data`` previously called push_many/flush
    unwrapped, so a payload-reading accumulator's KeyError propagated out of
    validate() (crashing strict and warn alike).  The conservation check
    already reports this root cause as ACC_PUSH_OR_FLUSH_RAISED — the
    pending-data check must bail quietly, not surface OP_VALIDATOR_CRASHED."""
    pipe = _empty_pipeline().add_op(
        "tok",
        process_many=lambda elems: list(elems),
        accumulator=_PayloadReadingAccumulator,
        preserves_cursor_order=False,
    )

    report = pipe.validate()  # must not raise KeyError
    codes = _codes(report)
    assert "ACC_PUSH_OR_FLUSH_RAISED" in codes
    assert "OP_VALIDATOR_CRASHED" not in codes


class _PassthroughPayloadReadingAccumulator(Accumulator[Any]):
    """Correct passthrough accumulator that reads a custom payload field.

    Emits every push immediately as one batch (so it conserves, never
    buffers, and is lane-pure by construction) but reads ``payload["tokens"]``
    — so it only works on records that carry that key."""

    def push_many(self, elems: Any) -> list[ReadyBatch[Any]]:
        elems = list(elems)
        for e in elems:
            _ = e.payload["tokens"]  # KeyError on generic synthetic input
        return [(elems, 0)] if elems else []

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[Any]]:
        return []

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        return False


def test_accumulator_probes_use_user_supplied_validation_samples() -> None:
    """The accumulator probes must feed the op's ``validation_samples`` records,
    not generic synthetic ones.

    A payload-reading accumulator that is otherwise correct would be falsely
    flagged with ACC_PUSH_OR_FLUSH_RAISED if the probes hardcoded synthetic
    records (which lack the custom payload key).  Supplying matching records
    must let the full accumulator suite run cleanly."""
    pipe = _empty_pipeline().add_op(
        "tok",
        process_many=lambda elems: list(elems),
        accumulator=_PassthroughPayloadReadingAccumulator,
        preserves_cursor_order=True,
        validation_samples=lambda: _custom_records("tokens"),
    )

    report = pipe.validate()
    codes = _codes(report)
    assert "ACC_PUSH_OR_FLUSH_RAISED" not in codes
    assert "OP_REJECTS_GENERIC_PAYLOAD" not in codes
    assert report.ok, report.format()


class _FastTok:
    name_or_path = "fake-model"
    is_fast = True
    pad_token = None
    eos_token = "</s>"
    eos_token_id = 0

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        return {
            "input_ids": [[1, 2] for _ in texts],
            "attention_mask": [[1, 1] for _ in texts],
        }


class _NoBosTok(_FastTok):
    bos_token_id = None


def _tokenize_ops(pipe: PublicPipeline) -> list[Any]:
    from zephon._internal.ops.tokenize_base import TokenizeBase

    return [n.op for n in pipe._graph.nodes if isinstance(n.op, TokenizeBase)]


def test_preflight_no_tokenize_ops_returns_empty_report() -> None:
    report = preflight_tokenizers(_empty_pipeline().decode_text())
    assert report.ok
    assert report.issues == []


def test_preflight_bad_tokenizer_reports_error_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("no repo named 'definitely-not-a-model'")

    monkeypatch.setattr("zephon._internal.ops.tokenize_base.load_hf_tokenizer", _boom)
    pipe = _empty_pipeline().tokenize(
        tokenizer_id="definitely-not-a-model",
        field="text",
        special_tokens="tokenizer_default",
    )

    report = preflight_tokenizers(pipe)
    assert not report.ok
    (issue,) = report.issues
    assert issue.severity == "error"
    assert issue.code == "TOKENIZER_PREFLIGHT_FAILED"
    assert "definitely-not-a-model" in issue.message
    assert "no repo" in issue.message


def test_preflight_uncopyable_op_becomes_issue_not_crash() -> None:
    class _Uncopyable:
        def __deepcopy__(self, memo: dict[int, Any]) -> "_Uncopyable":
            raise TypeError("cannot deepcopy this tokenizer")

        name_or_path = "uncopyable"

    pipe = _empty_pipeline().tokenize(
        tokenizer=_Uncopyable(),
        field="text",
        special_tokens="tokenizer_default",
    )
    report = preflight_tokenizers(pipe)
    (issue,) = report.issues
    assert issue.code == "TOKENIZER_PREFLIGHT_FAILED"
    assert "cannot deepcopy" in issue.message
    assert "uncopyable" in issue.message


def test_preflight_probes_a_copy_and_leaves_pipeline_ops_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[str | None] = []

    def _fake_load(tokenizer_id: str | None, **kwargs: Any) -> Any:
        loads.append(tokenizer_id)
        return _FastTok()

    monkeypatch.setattr(
        "zephon._internal.ops.tokenize_base.load_hf_tokenizer", _fake_load
    )
    pipe = _empty_pipeline().tokenize(
        tokenizer_id="fake-model",
        field="text",
        special_tokens="tokenizer_default",
    )

    report = preflight_tokenizers(pipe)
    assert report.ok, report.format()
    assert loads == ["fake-model"]

    # Preflight must not load or cache state on the pipeline-owned operator.
    (op,) = _tokenize_ops(pipe)
    assert op.tok is None
    assert op._tokenizer_instantiated is False
    assert op._setup_error is None


def test_preflight_finalize_failure_is_not_cached_on_pipeline_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Finalization failures are cached, unlike loader failures.
    monkeypatch.setattr(
        "zephon._internal.ops.tokenize_base.load_hf_tokenizer",
        lambda *a, **k: _NoBosTok(),
    )
    pipe = _empty_pipeline().tokenize(tokenizer_id="fake-model", field="text")

    report = preflight_tokenizers(pipe)
    (issue,) = report.issues
    assert issue.code == "TOKENIZER_PREFLIGHT_FAILED"
    assert "BOS" in issue.message

    (op,) = _tokenize_ops(pipe)
    assert op._setup_error is None
    assert op._tokenizer_instantiated is False


def test_preflight_aggregates_issues_in_graph_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("down")

    monkeypatch.setattr("zephon._internal.ops.tokenize_base.load_hf_tokenizer", _boom)
    pipe = (
        _empty_pipeline()
        .tokenize(
            tokenizer_id="text-model",
            field="text",
            special_tokens="tokenizer_default",
        )
        .tokenize_chat(tokenizer_id="chat-model")
    )

    report = preflight_tokenizers(pipe)
    assert [i.code for i in report.issues] == ["TOKENIZER_PREFLIGHT_FAILED"] * 2
    assert "text-model" in report.issues[0].message
    assert "chat-model" in report.issues[1].message
