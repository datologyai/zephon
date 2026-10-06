"""Attach a function as an operator, with its own grouping."""

from zephon import Pipeline
from zephon.ops import CountingAccumulator
from zephon.types import SampleRecord


def score_documents(records: list[SampleRecord]) -> list[SampleRecord]:
    """Score a whole group of documents in one call of the classifier."""
    scores = classifier([record.payload["text"] for record in records])
    for record, quality in zip(records, scores, strict=True):
        record.payload = {**record.payload, "quality": quality}
    return records


# The accumulator is a factory, not an instance: Zephon rebuilds it on start
# and on every reset. CountingAccumulator buffers per lane, so a group never
# mixes lanes. Scoring only writes the field; a map_transform does the dropping.
pipeline = (
    Pipeline(work_source)
    .add_op(
        "quality_score",
        process_many=score_documents,
        accumulator=lambda: CountingAccumulator(max_batch=64),
        preserves_cursor_order=True,
        parallelism=4,
    )
    .map_transform(lambda payload: payload if payload["quality"] > 0.5 else None)
)
