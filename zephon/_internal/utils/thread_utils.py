# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Utilities for controlling library-level threading."""

from __future__ import annotations

import os
import sys
import threading

# Env vars that suppress internal thread pools in numerical libraries.
# Shared between suppress_library_threads() (set at runtime) and the
# Ray runner (baked into runtime_env so they're set before import time).
# ARROW_NUM_THREADS is PyArrow's pre-import default for its CPU thread pool;
# cap_arrow_threads() additionally pins the pools at runtime in case PyArrow
# was already imported before the env was set.
THREAD_SUPPRESSION_ENV_VARS: dict[str, str] = {
    "TOKENIZERS_PARALLELISM": "False",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "ARROW_NUM_THREADS": "1",
}

_vortex_thread_settings: tuple[int, int] | None = None
_vortex_thread_lock = threading.Lock()


def _reset_vortex_threads_after_fork() -> None:
    """Discard a lock that another parent thread may have held during fork."""
    global _vortex_thread_lock, _vortex_thread_settings
    _vortex_thread_lock = threading.Lock()
    _vortex_thread_settings = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_vortex_threads_after_fork)


def cap_vortex_threads() -> None:
    """Cap an imported Vortex runtime once per process and requested setting.

    ``ZEPHON_VORTEX_THREADS`` defaults to one background worker. As with the
    Arrow cap, zero leaves the library's setting untouched. The Vortex format
    also calls this before its first read, covering workers that do not call
    ``suppress_library_threads``. Other formats do not import Vortex here.
    """
    global _vortex_thread_settings
    vortex = sys.modules.get("vortex")
    set_workers = getattr(vortex, "set_worker_threads", None)
    if set_workers is None:
        # A concurrent import can register the module before exporting its API.
        # The Vortex format calls this again once that import has completed.
        return
    n = int(os.environ.get("ZEPHON_VORTEX_THREADS", "1"))
    if n < 0:
        raise ValueError("ZEPHON_VORTEX_THREADS cannot be negative")
    if n == 0:
        return
    settings = (os.getpid(), n)
    if _vortex_thread_settings != settings:
        with _vortex_thread_lock:
            if _vortex_thread_settings != settings:
                set_workers(n)
                _vortex_thread_settings = settings


def cap_arrow_threads() -> None:
    """Pin PyArrow's CPU and IO thread pools (independent of OMP/MKL).

    PyArrow keeps its own CPU (compute / Parquet column decode) and IO thread
    pools, sized to the core count by default. With several parallel fetch
    lanes each decoding a row group, those per-process pools multiply into
    oversubscription. ``ZEPHON_PARQUET_ARROW_THREADS`` sets the count (default
    1); ``0`` leaves PyArrow's own defaults untouched.
    """
    n = int(os.environ.get("ZEPHON_PARQUET_ARROW_THREADS", "1"))
    if n <= 0:
        return
    try:
        import pyarrow as pa

        pa.set_cpu_count(n)
        pa.set_io_thread_count(n)
    except Exception:
        pass


def suppress_library_threads() -> None:
    """Prevent numerical libraries from spawning their own thread pools.

    Sets environment variables and runtime flags that prevent libraries
    (OpenMP, MKL, BLAS, HuggingFace tokenizers, etc.) from spawning
    internal thread pools.  When N workers each spawn C library threads,
    the resulting N*C threads cause severe contention; calling this once
    at worker/actor startup avoids that.
    """
    os.environ.update(THREAD_SUPPRESSION_ENV_VARS)

    cap_arrow_threads()
    cap_vortex_threads()

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
