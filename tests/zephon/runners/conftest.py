# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="module")
def ray_init():
    """Initialize Ray for testing."""
    ray = pytest.importorskip("ray")
    session_scope = os.environ.get("ZEPHON_RAY_SESSION_SCOPE") == "session"
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_cpus=2,
            object_store_memory=100_000_000,
            include_dashboard=False,
            runtime_env={"working_dir": None},
        )
    yield
    if not session_scope and ray.is_initialized():
        ray.shutdown()
