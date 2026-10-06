"""Subclass BaseOp to load an expensive model once per worker in setup."""

from typing import Any

from zephon.ops import (
    Accumulator,
    BaseOp,
    CountingAccumulator,
    OpContext,
    OpTraits,
)
from zephon.types import SampleRecord


class QualityScoreOp(BaseOp):
    """Score documents with a classifier that is too costly to load per call."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self._model_name = model_name  # configuration, and picklable
        self._model = None  # built per worker in setup

    def setup(self, ctx: OpContext) -> None:
        super().setup(ctx)
        self._model = load_classifier(self._model_name)

    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True, parallelism=4)

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[Any]:
        return CountingAccumulator(max_batch=64)

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        scores = self._model.score([record.payload["text"] for record in elems])
        for record, quality in zip(elems, scores, strict=True):
            record.payload = {**record.payload, "quality": quality}
        return elems
