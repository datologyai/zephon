# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Materialization operators that control buffering boundaries."""

from typing import Optional

from zephon.core.constants import StreamItem
from zephon.core.op_base import DefaultFinalize, DefaultSetup
from zephon.core.traits import Buffering, OpTraits


class Materialize(DefaultSetup, DefaultFinalize[StreamItem]):
    """Force evaluation of upstream iterables without altering records."""

    def traits(self) -> OpTraits:
        return OpTraits(indexable=False)

    def buffering(self) -> Optional[Buffering]:
        return None

    def process_one(self, elem: StreamItem) -> list[StreamItem]:
        return [elem]

    def process_many(self, elems: list[StreamItem]) -> list[StreamItem]:
        return list(elems)
