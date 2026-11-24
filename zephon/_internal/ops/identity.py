# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Identity operator behind a passthrough branch of ``Pipeline.branch()``."""

from typing import Any

from zephon.ops.accumulators import Accumulator, PassthroughAccumulator
from zephon.ops.base import BaseOp
from zephon.ops.traits import OpTraits
from zephon.types import StreamItem


class Identity(BaseOp):
    """Return every record unchanged.

    ``lambda b: b`` in ``Pipeline.branch()`` gets this op, so a passthrough
    branch is an ordinary branch op with its own node.
    """

    def traits(self) -> OpTraits:
        return OpTraits(calibration_replayable=True, preserves_cursor_order=True)

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[StreamItem]:
        return PassthroughAccumulator[StreamItem]()

    def process_one(self, elem: StreamItem) -> list[StreamItem]:
        return [elem]

    def process_many(self, elems: list[StreamItem]) -> list[StreamItem]:
        return list(elems)
