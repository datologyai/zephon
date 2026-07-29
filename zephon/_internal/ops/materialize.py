# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Materialization operators that control buffering boundaries."""

from typing import Any

from zephon.ops.accumulators import Accumulator, PassthroughAccumulator
from zephon.ops.base import BaseOp
from zephon.ops.traits import OpTraits
from zephon.types import StreamItem


class Materialize(BaseOp):
    """Force evaluation of upstream iterables without altering records."""

    def traits(self) -> OpTraits:
        return OpTraits(indexable=False, preserves_cursor_order=True)

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[StreamItem]:
        return PassthroughAccumulator[StreamItem]()

    def process_one(self, elem: StreamItem) -> list[StreamItem]:
        return [elem]

    def process_many(self, elems: list[StreamItem]) -> list[StreamItem]:
        return list(elems)
