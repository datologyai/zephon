# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

# Tighten the resilient-worker watchdog poll for tests.  Production
# default is 5 s (near-zero overhead under steady state); the resilience
# tests run multiple crash-respawn cycles each, and 5 s × N polls
# dominates the observed elapsed time.  Setting this in *conftest.py*
# guarantees it's applied before any test module triggers
# ``import zephon.runners.process`` — earlier setdefault calls in
# individual test modules raced against alphabetical test collection
# (e.g. test_concurrent.py importing zephon first).  The 90 s
# per-test pytest-timeout is still the real hang guard.
os.environ.setdefault("ZEPHON_WATCHDOG_POLL_S", "0.1")

import multiprocessing

import pytest


def _noop() -> None:  # pragma: no cover - target for forkserver warmup
    return


@pytest.fixture(scope="session", autouse=True)
def _warm_forkserver():
    """Bring up the ``forkserver`` daemon once per pytest session.

    The first ``ctx.Process(target=...).start()`` under ``forkserver``
    pays a one-shot cost (~10-15 s on cold CI) for daemon spawn +
    importing zephon (and torch, if loaded by tests) in the daemon.
    Every subsequent spawn from the same context forks from the warm
    daemon and is near-instant.

    Resilient-worker tests trigger 2-3 spawns each, so this cold cost
    used to land *inside* their elapsed-time bounds — observed at
    ~55 s for the 3-spawn ``test_persistent_crash_deterministic_escalates``
    on 3.12 CI.  Doing one trivial warmup spawn at session scope here
    amortizes the cold cost over the whole resilient-test class.

    No-op on platforms where ``forkserver`` is unavailable (Windows).
    """
    if "forkserver" not in multiprocessing.get_all_start_methods():
        return
    ctx = multiprocessing.get_context("forkserver")
    proc = ctx.Process(target=_noop, daemon=True)
    proc.start()
    proc.join(timeout=60)


@pytest.fixture(scope="session")
def ray_init():
    """Initialize Ray once for the whole test session.

    Ray's testing-tips doc recommends reusing a single cluster across
    tests: each init/teardown is ~4–5 s and is another chance to hit
    known worker-startup SIGSEGVs in Ray's glog wrapper.
    """
    ray = pytest.importorskip("ray")

    # Disable ray's runtime env hook for uv run
    from ray._private import ray_constants

    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_cpus=2,
            object_store_memory=512_000_000,
            include_dashboard=False,
        )
    yield
    if ray.is_initialized():
        ray.shutdown()
