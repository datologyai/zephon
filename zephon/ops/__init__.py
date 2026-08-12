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
from zephon.ops.base import BaseOp, OpContext, StageInfo
from zephon.ops.children import (
    pack_meta,
    spawn_child,
    tombstone_meta,
    tombstones_for_record,
)
from zephon.ops.config import (
    MissingFieldMode,
    PackingAlgorithm,
    SpanSource,
    SpecialTokensMode,
)
from zephon.ops.grouping import DomainGroups
from zephon.ops.traits import OpTraits

__all__ = [
    "Accumulator",
    "BaseOp",
    "CountingAccumulator",
    "DomainGroups",
    "MissingFieldMode",
    "OpContext",
    "OpTraits",
    "PackingAlgorithm",
    "PassthroughAccumulator",
    "ReadyBatch",
    "SpanSource",
    "SpecialTokensMode",
    "StageInfo",
    "pack_meta",
    "spawn_child",
    "tombstone_meta",
    "tombstones_for_record",
]
