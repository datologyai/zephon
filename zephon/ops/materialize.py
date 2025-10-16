# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Materialization operators that control buffering boundaries."""

from typing import Optional

from zephon.core.constants import SampleBatch, SampleRecord
from zephon.core.op_base import DefaultFinalize, OpContext
from zephon.core.traits import Buffering, OpTraits


class Materialize(DefaultFinalize[SampleRecord | SampleBatch]):
    """Force evaluation of upstream iterables without altering records."""

    def setup(self, ctx: OpContext) -> None:
        return None

    def traits(self) -> OpTraits:
        return OpTraits(indexable=False)

    def buffering(self) -> Optional[Buffering]:
        return None

    def process_one(
        self, elem: SampleRecord | SampleBatch
    ) -> list[SampleRecord | SampleBatch]:
        return [elem]

    def process_many(
        self, elems: list[SampleRecord | SampleBatch]
    ) -> list[SampleRecord | SampleBatch]:
        return list(elems)
