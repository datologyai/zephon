# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Utilities for controlling library-level threading."""

from __future__ import annotations

import os
import threading
import warnings
from types import ModuleType

from zephon.io.options import ArrowOptions, LanceOptions, VortexOptions

# Env vars that suppress internal thread pools in numerical libraries.
# Shared between suppress_library_threads() (set at runtime) and the
# Ray runner (baked into runtime_env so they're set before import time).
# Arrow, Vortex and Lance pools are configured separately when their readers
# are set up.
THREAD_SUPPRESSION_ENV_VARS: dict[str, str] = {
    "TOKENIZERS_PARALLELISM": "False",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
}


def configure_vortex_threads(vortex: ModuleType, options: VortexOptions) -> None:
    """Configure the process-wide background pool without hiding native errors."""
    count = options.worker_threads
    if count is None:
        return
    if (
        "worker_threads" not in options._provided_fields
        and "VORTEX_MAX_THREADS" in os.environ
    ):
        return
    vortex.set_worker_threads(count)


# pylance's runtime reads this when it starts, and so does obstore's.
_TOKIO_ENV = "TOKIO_WORKER_THREADS"
# Lance-only: read when pylance's CPU pool first runs, which can be much later.
_LANCE_CPU_ENV = "LANCE_CPU_THREADS"
_lance_pools_lock = threading.Lock()
_lance_pools_started = False
# Both pools' size after the first configuration; None if they differ.
_lance_pools_size: str | None = None


def configure_lance_threads(lance_file: ModuleType, options: LanceOptions) -> None:
    """Size pylance's pools once per process, before its first read.

    pylance has no setter. LANCE_CPU_THREADS stays set for the lazily started
    CPU pool; TOKIO_WORKER_THREADS is set only while an in-memory session
    starts the runtime, since obstore reads it too.
    """
    global _lance_pools_started, _lance_pools_size
    count = options.worker_threads
    if count is None:
        return
    explicit = "worker_threads" in options._provided_fields
    overrides = {
        name: str(count)
        for name in (_TOKIO_ENV, _LANCE_CPU_ENV)
        if explicit or name not in os.environ
    }
    with _lance_pools_lock:
        if _lance_pools_started:
            if explicit and str(count) != _lance_pools_size:
                warnings.warn(
                    f"lance.worker_threads={count} has no effect: pylance's "
                    + "pools are sized once per process and already were",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return
        tokio = os.environ.get(_TOKIO_ENV)
        os.environ.update(overrides)
        sizes = {os.environ.get(_TOKIO_ENV), os.environ.get(_LANCE_CPU_ENV)}
        try:
            lance_file.LanceFileSession("memory://zephon-pool-start")
        finally:
            if tokio is None:
                os.environ.pop(_TOKIO_ENV, None)
            else:
                os.environ[_TOKIO_ENV] = tokio
        _lance_pools_size = sizes.pop() if len(sizes) == 1 else None
        _lance_pools_started = True


def _reset_lance_pools_after_fork() -> None:
    # pylance rebuilds its runtime in a forked child, which must size it again.
    # The parent held the lock across the fork, so the child takes a fresh one.
    global _lance_pools_lock, _lance_pools_started, _lance_pools_size
    _lance_pools_lock = threading.Lock()
    _lance_pools_started = False
    _lance_pools_size = None


if hasattr(os, "register_at_fork"):
    # Holding the lock across the fork keeps a child from inheriting the
    # temporary TOKIO_WORKER_THREADS while another thread is starting pylance.
    os.register_at_fork(
        before=lambda: _lance_pools_lock.acquire(),
        after_in_parent=lambda: _lance_pools_lock.release(),
        after_in_child=_reset_lance_pools_after_fork,
    )


def configure_arrow_threads(pa: ModuleType, options: ArrowOptions) -> None:
    """Configure Arrow's process-wide pools, retaining the legacy env fallback."""
    counts = {"cpu_threads": options.cpu_threads, "io_threads": options.io_threads}
    legacy = os.environ.get("ZEPHON_PARQUET_ARROW_THREADS")
    if legacy is not None and options._provided_fields != counts.keys():
        warnings.warn(
            "ZEPHON_PARQUET_ARROW_THREADS is deprecated; use "
            + "io_options.arrow.cpu_threads and io_options.arrow.io_threads",
            DeprecationWarning,
            stacklevel=2,
        )
        count = int(legacy)
        for name in counts:
            if name not in options._provided_fields:
                counts[name] = count if count > 0 else None
    if counts["cpu_threads"] is not None:
        pa.set_cpu_count(counts["cpu_threads"])
    if counts["io_threads"] is not None:
        pa.set_io_thread_count(counts["io_threads"])


def suppress_library_threads() -> None:
    """Prevent numerical libraries from spawning their own thread pools.

    Sets environment variables and runtime flags that prevent libraries
    (OpenMP, MKL, BLAS, HuggingFace tokenizers, etc.) from spawning
    internal thread pools.  When N workers each spawn C library threads,
    the resulting N*C threads cause severe contention; calling this once
    at worker/actor startup avoids that.
    """
    os.environ.update(THREAD_SUPPRESSION_ENV_VARS)

    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    try:
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass
