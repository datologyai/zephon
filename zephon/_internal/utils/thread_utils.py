# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Utilities for controlling library-level threading."""

from __future__ import annotations

import contextlib
import os
import tempfile
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


# pylance reads these when it first starts its tokio and CPU pools.
_LANCE_POOL_ENV_VARS = ("TOKIO_WORKER_THREADS", "LANCE_CPU_THREADS")
_lance_pools_lock = threading.Lock()
_lance_pools_started = False
# Both pools' size after the first configuration; None if they differ.
_lance_pools_size: str | None = None


def configure_lance_threads(lance_file: ModuleType, options: LanceOptions) -> None:
    """Size pylance's pools once per process, before its first read.

    pylance has no setter and reads the sizes from the environment when it
    starts its pools, so they are set only around a throwaway open and then
    restored, since obstore reads TOKIO_WORKER_THREADS too. A runtime another
    thread starts during that open picks them up as well.
    """
    global _lance_pools_started, _lance_pools_size
    count = options.worker_threads
    if count is None:
        return
    explicit = "worker_threads" in options._provided_fields
    overrides = {
        name: str(count)
        for name in _LANCE_POOL_ENV_VARS
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
        _lance_pools_started = True
        sizes = {
            overrides.get(name, os.environ.get(name)) for name in _LANCE_POOL_ENV_VARS
        }
        _lance_pools_size = sizes.pop() if len(sizes) == 1 else None
        if not overrides:
            return
        saved = {name: os.environ.get(name) for name in overrides}
        os.environ.update(overrides)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                # Failing to open a missing file is enough to start the pools.
                with contextlib.suppress(Exception):
                    lance_file.LanceFileReader(os.path.join(tmp, "missing.lance"))
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


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
