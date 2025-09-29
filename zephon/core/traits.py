# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Operator traits that guide planning, scheduling, and buffering."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OpTraits:
    """Static capabilities an operator advertises to the planner."""

    indexable: bool = True
    parallelism: int = 1


@dataclass
class Buffering:
    """Runtime buffering preferences exposed by the operator implementation."""

    max_batch: int = 32
    max_latency_ms: Optional[int] = 5
