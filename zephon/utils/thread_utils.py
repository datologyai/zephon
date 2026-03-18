# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Utilities for controlling library-level threading."""

from __future__ import annotations

import os


def suppress_library_threads() -> None:
    """Prevent numerical libraries from spawning their own thread pools.

    Sets environment variables and runtime flags that prevent libraries
    (OpenMP, MKL, BLAS, HuggingFace tokenizers, etc.) from spawning
    internal thread pools.  When N workers each spawn C library threads,
    the resulting N*C threads cause severe contention; calling this once
    at worker/actor startup avoids that.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["RAYON_NUM_THREADS"] = "1"

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
