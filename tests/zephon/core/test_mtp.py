# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for MTP pipeline mode."""

from __future__ import annotations

import multiprocessing
import time
from typing import Any
from unittest import mock

import cloudpickle
import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon.api import Pipeline as PublicPipeline
from zephon.core._mtp import MTPPipeline
from zephon.core.constants import SampleBatch, SampleRecord
from zephon.runners.queue import NamedQueue


def _fake_ckpt(**overrides: Any) -> dict[str, Any]:
    """Minimal structurally-valid checkpoint for tests that need restore()."""
    base: dict[str, Any] = {
        "world": {"canonical_replicas": 1},
        "progress": {"0": {"chunk_id": 0, "offset": 0}},
        "lane_next_cid": {"0": 1},
        "lane_ws_state": {"0": {}},
        "last_round_id": "test",
        "checkpoint_reload_count": 0,
    }
    base.update(overrides)
    return base


def _mk_pipe(
    n_rows: int = 10,
    chunk_size: int = 8,
    *,
    mtp: bool = True,
    **opts: Any,
) -> PublicPipeline:
    """Build a simple pipeline (MTP or inline mode)."""
    rows = [{"text": f"row-{i}"} for i in range(n_rows)]
    ds = make_inmem_dataset("test", rows)
    ws = FakeIndexableWorkSource(ds, chunk_size=chunk_size)
    if mtp:
        opts.setdefault("mtp_buffer", 4)
    return (
        PublicPipeline(ws)
        .decode_text()
        .options(
            deterministic=True,
            max_workers=1,
            default_stage_prefetch=0,
            prefetch_batches=0,
            mtp_mode=mtp,
            **opts,
        )
    )


class TestMTPBasic:
    """MTP mode produces same output as inline mode."""

    def test_mtp_yields_all_records(self) -> None:
        pipe = _mk_pipe(n_rows=5, chunk_size=5)
        records = list(pipe)
        assert len(records) == 5
        assert all(isinstance(r, SampleRecord) for r in records)

    def test_mtp_matches_inline_output(self) -> None:
        inline_records = list(_mk_pipe(mtp=False, n_rows=6, chunk_size=6))
        sub_records = list(_mk_pipe(n_rows=6, chunk_size=6))

        assert len(inline_records) == len(sub_records)
        for inline_r, sub_r in zip(inline_records, sub_records):
            assert isinstance(inline_r, SampleRecord)
            assert isinstance(sub_r, SampleRecord)
            assert inline_r.payload == sub_r.payload

    def test_mtp_with_batch(self) -> None:
        pipe = _mk_pipe(n_rows=6, chunk_size=6).batch(microbatch_size=2, drop_last=True)
        batches = list(pipe)
        assert all(isinstance(b, SampleBatch) for b in batches)
        assert all(len(b) == 2 for b in batches)


class TestMTPEarlyBreak:
    """MTP mode handles early iteration break cleanly."""

    def test_early_break_no_hang(self) -> None:
        pipe = _mk_pipe(n_rows=20, chunk_size=20)
        count = 0
        for item in pipe:
            count += 1
            if count >= 3:
                break
        assert count == 3


class TestMTPQueueStats:
    def test_none_when_inline_or_not_running(self) -> None:
        pipe = _mk_pipe(mtp=False, n_rows=4, chunk_size=4)
        assert pipe.mtp_queue_stats() is None
        list(pipe)
        assert pipe.mtp_queue_stats() is None

    def test_live_stats_then_none_after_close(self) -> None:
        pipe = _mk_pipe(n_rows=30, chunk_size=10)
        assert pipe.mtp_queue_stats() is None

        it = iter(pipe)
        next(it)
        stats = pipe.mtp_queue_stats()
        assert stats is not None
        assert stats.capacity == 4
        # depth is -1 on macOS (sem_getvalue unsupported), bounded elsewhere.
        assert stats.depth == -1 or 0 <= stats.depth <= stats.capacity

        # With the consumer stalled the producer runs ahead, so pickled
        # items accumulate in the transport buffer.
        deadline = time.monotonic() + 30
        while stats.staged_bytes == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
            stats = pipe.mtp_queue_stats()
            assert stats is not None
        assert stats.staged_bytes > 0

        it.close()
        assert pipe.mtp_queue_stats() is None


class TestCaptureFinalStateFeederError:
    """``capture_final_state`` runs in generator teardown — a data-queue
    feeder error must surface as a warning, not raise into user code."""

    def test_feeder_error_warns_and_preserves_state(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        q = NamedQueue("mtp-data-test", maxsize=4, ctx=ctx)
        try:
            # Start the feeder thread, then inject a serialization
            # failure; the FeederError sentinel makes a later get() raise
            # QueueFeederError mid-drain.
            q.put("item")
            try:
                raise TypeError("cannot pickle 'generator' object")
            except TypeError as e:
                q._on_queue_feeder_error(e, "obj")

            sp = object.__new__(MTPPipeline)
            sp._closed = False
            sp._data_q = q
            sp._main_conn = mock.Mock(**{"poll.return_value": False})
            sp._process = mock.Mock(**{"is_alive.return_value": True})
            prior_state = {"prior": True}
            sp._last_state = prior_state

            with pytest.warns(UserWarning, match="feeder error"):
                sp.capture_final_state(timeout=0.5)

            assert sp._last_state is prior_state
        finally:
            q.close()


class TestMTPCheckpoint:
    """Checkpoint and restore through MTP mode."""

    def test_checkpoint_returns_dict(self) -> None:
        pipe = _mk_pipe(n_rows=5, chunk_size=5)
        it = iter(pipe)
        try:
            next(it)
            next(it)
            ckpt = pipe.checkpoint()
            assert isinstance(ckpt, dict)
        finally:
            it.close()

    def test_restore_stashes_checkpoint(self) -> None:
        """In MTP mode, restore() stashes the checkpoint for later."""
        pipe = _mk_pipe(n_rows=5, chunk_size=5)
        ckpt = _fake_ckpt()
        pipe.restore(ckpt)
        assert pipe._pending_restore == ckpt

    def test_checkpoint_resume_matches_inline(self) -> None:
        """End-to-end: iterate K items, checkpoint, restore, iterate rest.

        If the child process has un-processed ACKs at checkpoint time,
        state_dict() would be behind and resume would replay items —
        making first + remaining != all_inline.
        """
        n = 10
        all_inline = list(_mk_pipe(mtp=False, n_rows=n, chunk_size=n))
        assert len(all_inline) == n

        # Phase 1: iterate 4 items via MTP, then checkpoint
        pipe1 = _mk_pipe(n_rows=n, chunk_size=n)
        it1 = iter(pipe1)
        first = [next(it1) for _ in range(4)]
        ckpt = pipe1.checkpoint()
        it1.close()

        # Phase 2: restore into a new MTP pipe, iterate remaining
        pipe2 = _mk_pipe(n_rows=n, chunk_size=n)
        pipe2.restore(ckpt)
        remaining = list(pipe2)

        # Verify exact payload-level match with inline ground truth
        combined = first + remaining
        assert len(combined) == len(all_inline), (
            f"Expected {len(all_inline)} items, got {len(combined)} "
            f"(first={len(first)}, remaining={len(remaining)})"
        )
        for i, (a, b) in enumerate(zip(combined, all_inline)):
            assert a.payload == b.payload, f"Payload mismatch at index {i}"

    def test_checkpoint_resume_small_buffer(self) -> None:
        """Same checkpoint-resume test with buffer=2 to stress backpressure.

        A tiny buffer means the subprocess is frequently blocked on
        data_q.put() and must drain ctrl (including ACKs) during retries.
        This maximizes the window for an ACK–CHECKPOINT ordering bug.
        """
        n = 12
        all_inline = list(_mk_pipe(mtp=False, n_rows=n, chunk_size=n))
        assert len(all_inline) == n

        pipe1 = _mk_pipe(n_rows=n, chunk_size=n, mtp_buffer=2)
        it1 = iter(pipe1)
        first = [next(it1) for _ in range(6)]
        ckpt = pipe1.checkpoint()
        it1.close()

        pipe2 = _mk_pipe(n_rows=n, chunk_size=n, mtp_buffer=2)
        pipe2.restore(ckpt)
        remaining = list(pipe2)

        combined = first + remaining
        assert len(combined) == len(all_inline), (
            f"Expected {len(all_inline)} items, got {len(combined)} "
            f"(first={len(first)}, remaining={len(remaining)})"
        )
        for i, (a, b) in enumerate(zip(combined, all_inline)):
            assert a.payload == b.payload, f"Payload mismatch at index {i}"

    def test_checkpoint_resume_near_end(self) -> None:
        """Checkpoint after consuming all-but-one item, then resume.

        Exercises the case where the child process has nearly exhausted
        its iterator at checkpoint time.
        """
        n = 8
        all_inline = list(_mk_pipe(mtp=False, n_rows=n, chunk_size=n))

        pipe1 = _mk_pipe(n_rows=n, chunk_size=n)
        it1 = iter(pipe1)
        first = [next(it1) for _ in range(n - 1)]
        ckpt = pipe1.checkpoint()
        it1.close()

        pipe2 = _mk_pipe(n_rows=n, chunk_size=n)
        pipe2.restore(ckpt)
        remaining = list(pipe2)

        combined = first + remaining
        assert len(combined) == len(all_inline)
        for i, (a, b) in enumerate(zip(combined, all_inline)):
            assert a.payload == b.payload, f"Payload mismatch at index {i}"

    def test_checkpoint_after_full_iteration(self) -> None:
        """checkpoint() after exhausting MTP iteration returns final state."""
        pipe = _mk_pipe(n_rows=5, chunk_size=5)
        items = list(pipe)
        assert len(items) == 5
        # Iteration complete, child process exited — should still work
        ckpt = pipe.checkpoint()
        assert isinstance(ckpt, dict)
        assert "progress" in ckpt

    def test_checkpoint_after_full_iteration_matches_inline(self) -> None:
        """Post-iteration checkpoint matches inline engine state."""
        n = 8
        # Get inline final state
        pipe_inline = _mk_pipe(mtp=False, n_rows=n, chunk_size=n)
        list(pipe_inline)
        ckpt_inline = pipe_inline.checkpoint()

        # Get MTP final state
        pipe_sub = _mk_pipe(n_rows=n, chunk_size=n)
        list(pipe_sub)
        ckpt_sub = pipe_sub.checkpoint()

        # Progress should match — both consumed all items
        assert ckpt_inline["progress"] == ckpt_sub["progress"]

    def test_checkpoint_after_early_break_no_prior_ckpt(self) -> None:
        """checkpoint() after break WITHOUT prior checkpoint still works.

        Exercises the drain-without-ACK path: child process is unblocked
        by draining data_q, sees CHECKPOINT, responds with state_dict.
        """
        n = 10
        pipe = _mk_pipe(n_rows=n, chunk_size=n)
        it = iter(pipe)
        first = [next(it) for _ in range(4)]
        # NO explicit checkpoint — just close the iterator
        it.close()
        # checkpoint() should return state at break point
        ckpt = pipe.checkpoint()
        assert isinstance(ckpt, dict)
        assert "progress" in ckpt

        # Verify: restore and iterate remaining matches inline
        all_inline = list(_mk_pipe(mtp=False, n_rows=n, chunk_size=n))
        pipe2 = _mk_pipe(n_rows=n, chunk_size=n)
        pipe2.restore(ckpt)
        remaining = list(pipe2)
        combined = first + remaining
        assert len(combined) == len(all_inline), (
            f"Expected {len(all_inline)} items, got {len(combined)} "
            f"(first={len(first)}, remaining={len(remaining)})"
        )
        for i, (a, b) in enumerate(zip(combined, all_inline)):
            assert a.payload == b.payload, f"Payload mismatch at index {i}"

    def test_checkpoint_after_early_break_small_buffer(self) -> None:
        """Same but with buffer=2 to stress the drain path."""
        n = 10
        pipe = _mk_pipe(n_rows=n, chunk_size=n, mtp_buffer=2)
        it = iter(pipe)
        first = [next(it) for _ in range(4)]
        it.close()
        ckpt = pipe.checkpoint()
        assert isinstance(ckpt, dict)

        all_inline = list(_mk_pipe(mtp=False, n_rows=n, chunk_size=n))
        pipe2 = _mk_pipe(n_rows=n, chunk_size=n, mtp_buffer=2)
        pipe2.restore(ckpt)
        remaining = list(pipe2)
        combined = first + remaining
        assert len(combined) == len(all_inline), (
            f"Expected {len(all_inline)} items, got {len(combined)} "
            f"(first={len(first)}, remaining={len(remaining)})"
        )
        for i, (a, b) in enumerate(zip(combined, all_inline)):
            assert a.payload == b.payload, f"Payload mismatch at index {i}"

    def test_checkpoint_after_early_break_with_prior_ckpt(self) -> None:
        """checkpoint() after break with prior checkpoint returns fresh state."""
        pipe = _mk_pipe(n_rows=10, chunk_size=10)
        it = iter(pipe)
        for _ in range(3):
            next(it)
        ckpt_mid = pipe.checkpoint()
        # Consume 2 more items after checkpoint
        next(it)
        next(it)
        it.close()
        # Post-break checkpoint should reflect 5 items, not 3
        ckpt_post = pipe.checkpoint()
        assert ckpt_post != ckpt_mid, (
            "Post-break checkpoint should be fresher than mid-iteration one"
        )


class TestCheckpointReturnsPendingRestore:
    """checkpoint() returns stashed restore when no engine/MTP process is active."""

    def test_checkpoint_returns_pending_restore_as_copy(self) -> None:
        """restore() then checkpoint() without iterating returns the stashed ckpt."""
        pipe = _mk_pipe(n_rows=5, chunk_size=5)
        ckpt = _fake_ckpt()
        pipe.restore(ckpt)
        got = pipe.checkpoint()
        assert got == ckpt
        # Must be a copy, not the same object
        assert got is not ckpt

    def test_mtp_iter_clears_pending_restore(self) -> None:
        """Starting MTP iteration consumes _pending_restore."""
        # Get a real checkpoint from an MTP pipe
        pipe1 = _mk_pipe(n_rows=3, chunk_size=3)
        items1 = list(pipe1)
        assert len(items1) == 3

        pipe2 = _mk_pipe(n_rows=3, chunk_size=3)
        it2 = iter(pipe2)
        next(it2)
        ckpt = pipe2.checkpoint()
        it2.close()

        # Restore into a new pipe — _pending_restore should be set
        pipe3 = _mk_pipe(n_rows=3, chunk_size=3)
        pipe3.restore(ckpt)
        assert pipe3._pending_restore is not None
        it3 = iter(pipe3)
        try:
            next(it3)
            # _iter_mtp passes _pending_restore to MTPPipeline and clears it
            assert pipe3._pending_restore is None
            assert pipe3._sp is not None
        finally:
            it3.close()


class TestPicklePreservesPendingRestore:
    """__getstate__ preserves _pending_restore through pickle roundtrip."""

    def test_pending_restore_survives_pickle(self) -> None:
        pipe = _mk_pipe(n_rows=5, chunk_size=5)
        ckpt = _fake_ckpt()
        pipe.restore(ckpt)
        assert pipe._pending_restore == ckpt

        pipe2 = cloudpickle.loads(cloudpickle.dumps(pipe))
        assert pipe2._pending_restore == ckpt
        # Runtime state should be stripped
        assert pipe2._engine is None
        assert pipe2._plan is None

    def test_pickle_strips_runtime_state(self) -> None:
        """Engine, plan, runtime_spec, and MTP handle are stripped."""
        pipe = _mk_pipe(mtp=False, n_rows=5, chunk_size=5)
        # Build engine by iterating
        list(pipe)
        assert pipe._engine is not None

        pipe2 = cloudpickle.loads(cloudpickle.dumps(pipe))
        assert pipe2._engine is None
        assert pipe2._plan is None
        assert pipe2._runtime_spec is None


class TestDaemonFallback:
    """mtp_mode falls back to inline in daemon processes."""

    def test_daemon_falls_back_to_inline_with_warning(self) -> None:
        import warnings

        pipe = _mk_pipe(n_rows=5, chunk_size=5)
        mock_proc = mock.MagicMock()
        mock_proc.daemon = True
        with (
            mock.patch("multiprocessing.current_process", return_value=mock_proc),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            records = list(pipe)

        assert len(records) == 5
        # No MTP process was spawned — fell back to inline
        assert pipe._sp is None
        # Engine was built in this process (inline path)
        assert pipe._engine is not None
        # Warning was emitted
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 1
        assert "daemon" in str(runtime_warnings[0].message).lower()

    def test_daemon_fallback_applies_pending_restore(self) -> None:
        """restore() checkpoint is applied when daemon causes inline fallback."""
        import warnings

        # Get a real checkpoint from an inline pipe
        pipe1 = _mk_pipe(mtp=False, n_rows=10, chunk_size=10)
        it1 = iter(pipe1)
        first_items = [next(it1) for _ in range(3)]
        ckpt = pipe1.checkpoint()
        it1.close()

        # Build an MTP pipe, restore, then iterate under daemon mock
        pipe2 = _mk_pipe(n_rows=10, chunk_size=10)
        pipe2.restore(ckpt)

        mock_proc = mock.MagicMock()
        mock_proc.daemon = True
        with (
            mock.patch("multiprocessing.current_process", return_value=mock_proc),
            warnings.catch_warnings(record=True),
        ):
            warnings.simplefilter("always")
            remaining = list(pipe2)

        # Should resume from checkpoint, not replay from start
        all_items = list(_mk_pipe(mtp=False, n_rows=10, chunk_size=10))
        assert first_items + remaining == all_items


class TestInsideWorkerCompile:
    """compile() passes inside_worker flag to resolve_runtime_spec."""

    def test_compile_passes_inside_worker(self) -> None:
        """When inside_torch_worker() is True, process runners are demoted."""
        pipe = _mk_pipe(mtp=False, n_rows=5, chunk_size=5)
        pipe._options.runner = "process"
        # Reset cached spec
        pipe._runtime_spec = None

        with mock.patch("zephon.core.engine.inside_torch_worker", return_value=True):
            spec = pipe.compile()
        # process should be demoted to threads inside a worker
        assert spec.stages[0].runner_type == "threads"

    def test_compile_without_worker_keeps_process(self) -> None:
        """When not inside a worker, process runner stays."""
        pipe = _mk_pipe(mtp=False, n_rows=5, chunk_size=5)
        pipe._options.runner = "process"
        pipe._runtime_spec = None

        with mock.patch("zephon.core.engine.inside_torch_worker", return_value=False):
            spec = pipe.compile()
        assert spec.stages[0].runner_type == "process"


class TestExplainWithoutEngine:
    """Pipeline.explain() works without building an Engine."""

    def test_explain_returns_string(self) -> None:
        pipe = _mk_pipe(mtp=False, n_rows=5, chunk_size=5)
        text = pipe.explain()
        assert isinstance(text, str)
        assert "Stage[0]" in text
        # Should NOT have built an Engine
        assert pipe._engine is None

    def test_compile_returns_runtime_spec(self) -> None:
        pipe = _mk_pipe(mtp=False, n_rows=5, chunk_size=5)
        spec = pipe.compile()
        from zephon.core.runtime_spec import RuntimeSpec

        assert isinstance(spec, RuntimeSpec)
        assert pipe._engine is None


class TestRestoreValidation:
    """restore() validates checkpoint structure eagerly."""

    def test_restore_rejects_non_dict(self) -> None:
        pipe = _mk_pipe(n_rows=5, chunk_size=5)
        with pytest.raises(TypeError, match="Expected dict"):
            pipe.restore("not a dict")  # type: ignore[arg-type]

    def test_restore_rejects_missing_keys(self) -> None:
        pipe = _mk_pipe(n_rows=5, chunk_size=5)
        with pytest.raises(ValueError, match="missing"):
            pipe.restore({"world": {"canonical_replicas": 1}})

    def test_restore_rejects_bad_world(self) -> None:
        pipe = _mk_pipe(n_rows=5, chunk_size=5)
        ckpt = _fake_ckpt()
        ckpt["world"] = "bad"
        with pytest.raises(ValueError, match="world must be a dict"):
            pipe.restore(ckpt)

    def test_restore_rejects_missing_canonical_replicas(self) -> None:
        pipe = _mk_pipe(n_rows=5, chunk_size=5)
        ckpt = _fake_ckpt()
        ckpt["world"] = {}
        with pytest.raises(ValueError, match="canonical_replicas"):
            pipe.restore(ckpt)

    def test_restore_rejects_bad_progress_entry(self) -> None:
        pipe = _mk_pipe(n_rows=5, chunk_size=5)
        ckpt = _fake_ckpt()
        ckpt["progress"] = {"0": {"chunk_id": 0}}  # missing 'offset'
        with pytest.raises(ValueError, match="missing required field 'offset'"):
            pipe.restore(ckpt)

    def test_restore_accepts_valid_checkpoint(self) -> None:
        pipe = _mk_pipe(n_rows=5, chunk_size=5)
        ckpt = _fake_ckpt()
        pipe.restore(ckpt)
        assert pipe._pending_restore == ckpt

    def test_restore_invokes_engine_migration_chain(self, monkeypatch) -> None:
        from zephon.core.checkpoint import EngineStateV1
        from zephon.core.checkpoint._migrations import (
            _MIGRATIONS,
            CURRENT_VERSIONS,
            register_migration,
        )

        monkeypatch.setitem(CURRENT_VERSIONS, "engine", 2)
        monkeypatch.setitem(_MIGRATIONS, "engine", {})

        called: list[int] = []

        def _engine_v1_to_v2(v1: EngineStateV1) -> dict:
            called.append(v1.version)
            return v1.to_dict()

        register_migration("engine", from_version=1, fn=_engine_v1_to_v2)

        pipe = _mk_pipe(n_rows=5, chunk_size=5)
        pipe.restore(_fake_ckpt())

        assert called == [1]
