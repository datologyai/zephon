"""Hold a record back until its partner arrives."""

from zephon import Pipeline
from zephon.types import SampleRecord


def pair_up(
    held: dict[str, SampleRecord], records: list[SampleRecord]
) -> tuple[dict[str, SampleRecord], list[SampleRecord]]:
    """Emit a record once the record with its document id has also been seen."""
    emitted = []
    for record in records:
        partner = held.pop(record.payload["doc_id"], None)
        if partner is None:
            held[record.payload["doc_id"]] = record
        else:
            emitted += [partner, record]
    return held, emitted


# push holds records back, so flush has to empty the buffer or those records
# are dropped in silence. Holding a record past a later one reorders the
# stream, hence preserves_cursor_order=False.
pipeline = Pipeline(work_source).stateful_transform(
    "pair_up",
    init_state=dict,
    push=pair_up,
    flush=lambda held: list(held.values()),
    preserves_cursor_order=False,
)
