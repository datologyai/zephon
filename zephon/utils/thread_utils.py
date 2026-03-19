# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Utilities for controlling library-level threading."""

from __future__ import annotations

import os

# Env vars that suppress internal thread pools in numerical libraries.
# Shared between suppress_library_threads() (set at runtime) and the
# Ray runner (baked into runtime_env so they're set before import time).
THREAD_SUPPRESSION_ENV_VARS: dict[str, str] = {
    "TOKENIZERS_PARALLELISM": "False",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
}


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
