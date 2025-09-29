# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Batching operator for grouping sample records."""

from typing import Any, Optional

from zephon.core.constants import Element, SampleRecord
from zephon.core.op_base import DefaultFinalize, OpContext
from zephon.core.traits import Buffering, OpTraits


class Batch(DefaultFinalize):
    """Collect sample records into mini-batches."""

    def __init__(
        self, global_batch: int, dp_world: int, *, drop_last: bool = True
    ) -> None:
        if global_batch % dp_world != 0:
            msg = "global_batch must divide dp_world"
            raise ValueError(msg)
        self.local_bs = global_batch // dp_world
        self.dp_world = dp_world
        self.drop_last = drop_last
        self._buffer: list[SampleRecord] = []

    def setup(self, ctx: OpContext) -> None:
        return None

    def traits(self) -> OpTraits:
        return OpTraits(indexable=False)

    def buffering(self) -> Optional[Buffering]:
        return None

    def _collate(self, items: list[SampleRecord]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "ids": [record.meta.sample_id for record in items],
            "texts": [record.payload.get("text", "") for record in items],
        }
        first = items[0].payload
        if "input_ids" in first:
            batch["input_ids"] = [record.payload["input_ids"] for record in items]
        if "attention_mask" in first:
            batch["attention_mask"] = [
                record.payload["attention_mask"] for record in items
            ]
        return batch

    def process_one(self, elem: Element) -> list[Element]:
        assert isinstance(elem, SampleRecord)
        self._buffer.append(elem)
        if len(self._buffer) >= self.local_bs:
            output = self._collate(self._buffer[: self.local_bs])
            self._buffer = self._buffer[self.local_bs :]
            return [output]
        return []

    def process_many(self, elems: list[Element]) -> list[Element]:
        outputs: list[Element] = []
        for elem in elems:
            outputs.extend(self.process_one(elem))
        return outputs

    def finalize(self) -> list[Element]:
        if not self.drop_last and self._buffer:
            output = [self._collate(self._buffer)]
            self._buffer = []
            return output
        return []
