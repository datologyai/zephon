# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Work-source abstractions that feed sample identifiers to the engine."""

from zephon.work.base import WorkChunk, WorkSource
from zephon.work.static import StaticWorkSource

__all__ = [
    "WorkChunk",
    "WorkSource",
    "StaticWorkSource",
]
