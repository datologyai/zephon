# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Accumulator abstractions for deterministic parallel operator execution.

Accumulators run on the pump thread (serial) and define invocation boundaries
for parallel workers. This ensures that stateful grouping logic (batching,
packing, etc.) executes deterministically regardless of parallelism level.

The key insight is that the pump thread is already serial per operator.
By moving all cross-invocation state into accumulators, worker instances
become stateless with respect to output-producing state across invocations.
"""

from zephon.core.accumulators.base import Accumulator, ReadyBatch
from zephon.core.accumulators.counting import CountingAccumulator
from zephon.core.accumulators.passthrough import PassthroughAccumulator

__all__ = [
    "Accumulator",
    "ReadyBatch",
    "PassthroughAccumulator",
    "CountingAccumulator",
]
