"""Return the records you were given, not bare payloads."""

from zephon.types import SampleRecord


def score(records: list[SampleRecord]) -> list[SampleRecord]:
    """Rebuild the payload, and hand back the record that carried it."""
    for record in records:
        record.payload = {**record.payload, "score": len(record.payload["text"])}
    return records


def score_the_wrong_way(records: list[SampleRecord]) -> list[dict[str, object]]:
    """Zephon rejects this: the lineage left behind is what makes replay work."""
    return [{"score": len(record.payload["text"])} for record in records]
