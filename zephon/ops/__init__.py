# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Operator authoring kit: the contracts you subclass or pass to ``add_op()``.

Built-in operators are configured via the ``Pipeline`` builder methods
(``.shuffle()``, ``.batch()``, ``.tokenize()``, ...).
"""

from zephon.ops.accumulators import (
    Accumulator,
    CountingAccumulator,
    PassthroughAccumulator,
    ReadyBatch,
)
from zephon.ops.base import BaseOp, OpContext
from zephon.ops.children import pack_meta, spawn_child, tombstone_meta
from zephon.ops.config import PackingAlgorithm, SpanSource, SpecialTokensMode
from zephon.ops.grouping import DomainGroups
from zephon.ops.traits import OpTraits

__all__ = [
    "Accumulator",
    "BaseOp",
    "CountingAccumulator",
    "DomainGroups",
    "OpContext",
    "OpTraits",
    "PackingAlgorithm",
    "PassthroughAccumulator",
    "ReadyBatch",
    "SpanSource",
    "SpecialTokensMode",
    "pack_meta",
    "spawn_child",
    "tombstone_meta",
]
