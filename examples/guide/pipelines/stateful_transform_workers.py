"""Buffer on one thread, do the expensive work on many."""

from zephon import Pipeline
from zephon.types import SampleRecord


def encode(records: list[SampleRecord]) -> list[SampleRecord]:
    """Run on the parallel workers, once per group of records push emitted."""
    for record in records:
        record.payload = {**record.payload, "embedding": embed(record.payload["text"])}
    return records


# push and flush run single-threaded and see every record in order; transform
# is what gets fanned out. State is per lane, created by init_state for each
# one, and reset at every flush.
pipeline = Pipeline(work_source).stateful_transform(
    "pair_and_encode",
    init_state=dict,
    push=pair_up,
    flush=lambda held: list(held.values()),
    transform=encode,
    parallelism=4,
    preserves_cursor_order=False,
)
