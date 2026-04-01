# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Work-source abstractions that feed sample identifiers to the engine."""

from zephon.work.accumulator_mixture import AccumulatorMixtureWorkSource
from zephon.work.base import (
    ComponentOrder,
    MixtureReadConfig,
    MixtureReadMode,
    WorkChunk,
    WorkSource,
)
from zephon.work.mixture import MixtureSpec
from zephon.work.static_mixture import StaticMixtureWorkSource

__all__ = [
    "AccumulatorMixtureWorkSource",
    "ComponentOrder",
    "MixtureReadConfig",
    "MixtureReadMode",
    "WorkChunk",
    "WorkSource",
    "MixtureSpec",
    "StaticMixtureWorkSource",
]
