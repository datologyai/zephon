"""Per-record randomness that survives checkpoints and topology changes."""

import numpy as np

from zephon import Pipeline
from zephon.types import SampleRecord

SEED = 1234


def random_crop(records: list[SampleRecord]) -> list[SampleRecord]:
    """Seed each record's generator from its position in its lane's curriculum."""
    for record in records:
        meta = record.meta
        # Lane, chunk, and offset are fixed by the curriculum rather than by the
        # topology, and differ on every pass, so a repeated sample draws anew.
        rng = np.random.default_rng(
            (SEED, meta.lane_id, meta.chunk_id, meta.chunk_offset, *meta.lineage)
        )
        record.payload = crop(record.payload, rng)
    return records


pipeline = (
    Pipeline(work_source)
    .add_op(
        "random_crop",
        process_many=random_crop,
        preserves_cursor_order=True,
        # Calibration replays records with placeholder metadata, which would
        # change the draws.
        calibration_replayable=False,
        parallelism=8,
    )
    .batch(microbatch_size=32)
)
