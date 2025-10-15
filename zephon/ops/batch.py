# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Batching operator for grouping sample records."""

from typing import Any, Optional

from zephon.core.constants import Element, SampleRecord
from zephon.core.op_base import DefaultFinalize, OpContext
from zephon.core.traits import Buffering, OpTraits


class Batch(DefaultFinalize):
    """Collect sample records into mini-batches."""

    def __init__(self, microbatch_size: int, *, drop_last: bool = True) -> None:
        if microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        self.microbatch_size = int(microbatch_size)
        self.drop_last = drop_last
        self._buffer: list[SampleRecord] = []

    def setup(self, ctx: OpContext) -> None:
        return None

    def traits(self) -> OpTraits:
        return OpTraits(indexable=False, batch_shape_sensitive=False)

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
        if len(self._buffer) >= self.microbatch_size:
            output = self._collate(self._buffer[: self.microbatch_size])
            self._buffer = self._buffer[self.microbatch_size :]
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
