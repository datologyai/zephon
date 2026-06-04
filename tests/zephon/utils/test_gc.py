# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for zephon.utils.gc module and SafeSemaphore."""

import gc
import multiprocessing as mp
import threading
from multiprocessing import util
from unittest import mock

import pytest

from zephon.utils.gc import (
    cleanup_semaphores,
    collect_with_finalizers,
    is_gil_disabled,
)
from zephon.utils.semaphore import SafeSemLock

# Use spawn context for semaphores so they have names and register with
# the resource tracker. With "fork" (default on Linux < 3.14), semaphores
# are anonymous and don't register.
_spawn_ctx = mp.get_context("spawn")


def _registry_semaphore_names() -> set[str]:
    """Semaphore names currently in the multiprocessing finalizer registry.

    Only string first-args are kept: the registry is process-global and shared
    with CPython internals — e.g. a live ``mp.Queue`` registers
    ``Queue._finalize_close`` whose first arg is its ``collections.deque``
    buffer, which is unhashable and would otherwise break the set build.
    """
    return {
        args[0]
        for f in util._finalizer_registry.values()
        if (args := getattr(f, "_args", ())) and isinstance(args[0], str)
    }


class TestIsGilDisabled:
    """Tests for is_gil_disabled()."""

    def test_returns_bool(self):
        """is_gil_disabled should return a boolean."""
        result = is_gil_disabled()
        assert isinstance(result, bool)

    def test_result_is_cached(self):
        """Repeated calls should return the same cached value."""
        result1 = is_gil_disabled()
        result2 = is_gil_disabled()
        assert result1 is result2


class TestCleanupSemaphores:
    """Tests for cleanup_semaphores()."""

    def test_empty_list_is_noop(self):
        """Empty list should not raise or modify anything."""
        registry_before = len(util._finalizer_registry)
        cleanup_semaphores([])
        assert len(util._finalizer_registry) == registry_before

    def test_none_is_handled(self):
        """None should be handled gracefully."""
        # Should not raise
        cleanup_semaphores(None)  # type: ignore

    def test_cleans_up_single_semaphore(self):
        """Single semaphore should be cleaned up from registry."""
        sem = _spawn_ctx.Semaphore(1)
        name = sem._semlock.name

        # Verify it's in the registry
        names_in_registry = _registry_semaphore_names()
        assert name in names_in_registry

        # Clean up
        cleanup_semaphores([sem])

        # Verify it's removed
        names_in_registry = _registry_semaphore_names()
        assert name not in names_in_registry

    def test_cleans_up_multiple_semaphores(self):
        """Multiple semaphores should all be cleaned up."""
        sems = [_spawn_ctx.Semaphore(1) for _ in range(5)]
        names = {s._semlock.name for s in sems}

        # Verify they're in the registry
        names_in_registry = _registry_semaphore_names()
        assert names.issubset(names_in_registry)

        # Clean up
        cleanup_semaphores(sems)

        # Verify they're all removed
        names_in_registry = _registry_semaphore_names()
        assert not names.intersection(names_in_registry)

    def test_handles_already_cleaned_semaphore(self):
        """Cleaning up an already-cleaned semaphore should not raise."""
        sem = _spawn_ctx.Semaphore(1)

        # Clean up twice - second call should be a no-op
        cleanup_semaphores([sem])
        cleanup_semaphores([sem])  # Should not raise

    def test_handles_malformed_semaphore(self):
        """Semaphores without expected structure should be skipped."""
        # Create a mock object that doesn't have _semlock
        fake_sem = object()
        cleanup_semaphores([fake_sem])  # Should not raise

    def test_handles_anonymous_semaphores_from_fork_context(self):
        """Semaphores from fork context have name=None and should be skipped gracefully."""
        fork_ctx = mp.get_context("fork")
        sem = fork_ctx.Semaphore(1)

        # Fork semaphores have name=None
        assert sem._semlock.name is None

        # Should not raise - just skips the semaphore
        cleanup_semaphores([sem])

    def test_does_not_affect_other_finalizers(self):
        """Other finalizers in the registry should not be affected."""
        # Register a custom finalizer
        marker = {"called": False}

        def custom_cleanup():
            marker["called"] = True

        custom_finalizer = util.Finalize(None, custom_cleanup, exitpriority=10)

        # Create and clean up a semaphore
        sem = _spawn_ctx.Semaphore(1)
        cleanup_semaphores([sem])

        # Custom finalizer should still be active
        assert custom_finalizer.still_active()

        # Clean up custom finalizer
        custom_finalizer.cancel()


class TestCollectWithFinalizers:
    """Tests for collect_with_finalizers()."""

    def test_calls_gc_collect_at_least_once(self):
        """Should call gc.collect() at least once."""
        with mock.patch.object(gc, "collect") as mock_collect:
            collect_with_finalizers()
            assert mock_collect.call_count >= 1

    def test_single_cycle_on_gil_python(self):
        """On GIL Python, should only call gc.collect() once by default."""
        if is_gil_disabled():
            pytest.skip("Test only applies to GIL-enabled Python")

        with mock.patch.object(gc, "collect") as mock_collect:
            collect_with_finalizers()
            assert mock_collect.call_count == 1

    def test_multiple_cycles_on_free_threaded_python(self):
        """On free-threaded Python, should call gc.collect() multiple times."""
        if not is_gil_disabled():
            pytest.skip("Test only applies to free-threaded Python")

        with mock.patch.object(gc, "collect") as mock_collect:
            collect_with_finalizers(cycles=3)
            assert mock_collect.call_count == 3

    def test_respects_cycles_parameter(self):
        """cycles parameter should control number of GC cycles."""
        if not is_gil_disabled():
            pytest.skip("Test only applies to free-threaded Python")

        with mock.patch.object(gc, "collect") as mock_collect:
            collect_with_finalizers(cycles=5)
            assert mock_collect.call_count == 5


class TestSafeSemLock:
    """Tests for SafeSemLock wrapper class."""

    def test_creation_cancels_original_finalizer(self):
        """SafeSemLock should cancel the auto-registered finalizer."""
        sem = SafeSemLock.new_semaphore(1, ctx=_spawn_ctx)
        name = sem._semlock.name

        # The original finalizer (with just the name as arg) should be cancelled
        # Our finalizer has (name, lock, state) as args
        for finalizer in util._finalizer_registry.values():
            args = getattr(finalizer, "_args", ())
            if args and args[0] == name:
                # Our finalizer has 3 args, original has 1
                assert len(args) == 3, "Original finalizer should be cancelled"
                break
        else:
            pytest.fail("No finalizer found for semaphore")

    def test_cleanup_is_idempotent(self):
        """cleanup() should be safe to call multiple times."""
        sem = SafeSemLock.new_semaphore(1, ctx=_spawn_ctx)
        # Should not raise
        sem.cleanup()
        sem.cleanup()
        sem.cleanup()

    def test_cleanup_sets_state_flag(self):
        """cleanup() should set the internal state flag."""
        sem = SafeSemLock.new_semaphore(1, ctx=_spawn_ctx)
        assert sem._state[0] is False
        sem.cleanup()
        assert sem._state[0] is True

    def test_cleanup_removes_our_finalizer(self):
        """After cleanup(), our finalizer should also be cleaned up or skipped."""
        sem = SafeSemLock.new_semaphore(1, ctx=_spawn_ctx)
        name = sem._semlock.name

        sem.cleanup()

        # When GC eventually triggers the finalizer, it should see state[0]=True
        # and skip the actual cleanup work
        assert sem._state[0] is True

    def test_semaphore_interface_works(self):
        """SafeSemLock should have working acquire/release."""
        sem = SafeSemLock.new_semaphore(1, ctx=_spawn_ctx)

        # Should work like a regular semaphore
        assert sem.acquire(block=False) is True
        assert sem.acquire(block=False) is False  # Already acquired
        sem.release()
        assert sem.acquire(block=False) is True
        sem.release()

    def test_context_manager_works(self):
        """SafeSemLock should work as a context manager."""
        sem = SafeSemLock.new_semaphore(1, ctx=_spawn_ctx)

        with sem:
            # Should be acquired
            assert sem.acquire(block=False) is False

        # Should be released
        assert sem.acquire(block=False) is True
        sem.release()

    def test_cleanup_semaphores_works_with_safe_semlock(self):
        """cleanup_semaphores() should work with SafeSemLock objects."""
        sems = [SafeSemLock.new_semaphore(1, ctx=_spawn_ctx) for _ in range(3)]

        # All should have state[0] = False
        for sem in sems:
            assert sem._state[0] is False

        # Clean up via cleanup_semaphores
        cleanup_semaphores(sems)

        # All should have state[0] = True
        for sem in sems:
            assert sem._state[0] is True

    def test_concurrent_cleanup_is_safe(self):
        """Multiple threads calling cleanup() should be safe."""
        sem = SafeSemLock.new_semaphore(1, ctx=_spawn_ctx)
        errors = []
        cleanup_count = [0]
        lock = threading.Lock()

        def cleanup_thread():
            try:
                sem.cleanup()
                with lock:
                    cleanup_count[0] += 1
            except Exception as e:
                errors.append(e)

        # Start multiple threads that all try to cleanup
        threads = [threading.Thread(target=cleanup_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have no errors
        assert errors == []
        # All threads should have completed
        assert cleanup_count[0] == 10
        # State should be True
        assert sem._state[0] is True

    def test_anonymous_semaphore_cleanup_is_noop(self):
        """SafeSemLock with anonymous semaphore should skip cleanup gracefully."""
        fork_ctx = mp.get_context("fork")
        sem = SafeSemLock.new_semaphore(1, ctx=fork_ctx)

        # Fork semaphores have name=None
        assert sem._name is None

        # Should not raise
        sem.cleanup()

    def test_semlock_property_returns_underlying_semlock(self):
        """_semlock property should return the underlying semlock."""
        sem = SafeSemLock.new_semaphore(1, ctx=_spawn_ctx)
        # Should have access to underlying _semlock for compatibility
        assert sem._semlock is not None
        assert hasattr(sem._semlock, "name")

    def test_new_lock_creates_working_lock(self):
        """new_lock() should create a working Lock with safe cleanup."""
        lock = SafeSemLock.new_lock(ctx=_spawn_ctx)
        assert lock._semlock is not None
        assert lock._name is not None

        # Should work like a regular lock
        assert lock.acquire(block=False) is True
        assert lock.acquire(block=False) is False  # Already acquired
        lock.release()

        # Cleanup should work
        lock.cleanup()
        assert lock._state[0] is True

    def test_new_bounded_semaphore_creates_working_semaphore(self):
        """new_bounded_semaphore() should create a working BoundedSemaphore."""
        sem = SafeSemLock.new_bounded_semaphore(2, ctx=_spawn_ctx)
        assert sem._semlock is not None
        assert sem._name is not None

        # Should work like a bounded semaphore with value 2
        assert sem.acquire(block=False) is True
        assert sem.acquire(block=False) is True
        assert sem.acquire(block=False) is False  # Exhausted
        sem.release()
        sem.release()

        # Cleanup should work
        sem.cleanup()
        assert sem._state[0] is True

    def test_wrap_creates_wrapper_for_existing_semaphore(self):
        """wrap() should wrap an existing semaphore with safe cleanup."""
        # Create a raw semaphore
        raw_sem = _spawn_ctx.Semaphore(1)
        name = raw_sem._semlock.name

        # Wrap it
        safe_sem = SafeSemLock.wrap(raw_sem, source="test_wrap")
        assert safe_sem._name == name
        assert safe_sem._state[0] is False

        # Should work like a semaphore
        assert safe_sem.acquire(block=False) is True
        safe_sem.release()

        # Cleanup should work
        safe_sem.cleanup()
        assert safe_sem._state[0] is True
