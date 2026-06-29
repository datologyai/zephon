# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Utilities for controlling library-level threading."""

from __future__ import annotations

import os

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
