# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Time token-aware chunk allocation, which runs one deficit update per record.

Run with ``uv run python scripts/benchmark_token_allocation.py``. Reads only
in-memory rows, so it measures allocation rather than IO.
"""

import argparse
import statistics
import time

from zephon.io import Dataset, InMemoryShard
from zephon.work import StaticMixtureWorkSource, TokenEstimation


def _dataset(name: str, rows: int, words: int) -> Dataset:
    text = " ".join("w" for _ in range(words))
    return Dataset.from_dict(name, {0: InMemoryShard([{"text": text}] * rows)})


def _time_chunks(datasets: int, chunk_size: int, chunks: int) -> list[float]:
    names = [f"d{i}" for i in range(datasets)]
    work = StaticMixtureWorkSource(
        # Different document lengths give each dataset its own token cost.
        [_dataset(name, 4 * chunk_size, 5 + 3 * i) for i, name in enumerate(names)],
        dict.fromkeys(names, 1.0),
        chunk_size=chunk_size,
        seed=0,
        exhausted_policy="repeat",
        stop_after_passes=None,
        token_estimation=TokenEstimation(primer=0.25),
    )
    work.prime(counting_spec=None)
    lane = work.clone_for_lane(0, canonical_replicas=1)
    lane.next_chunk()  # builds the cost table
    times = []
    for _ in range(chunks):
        start = time.perf_counter()
        lane.next_chunk()
        times.append(time.perf_counter() - start)
    return times


def main() -> None:
    """Print per-chunk timings for each requested dataset count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--chunk-size", type=int, default=16_384)
    parser.add_argument("--chunks", type=int, default=30)
    args = parser.parse_args()
    for datasets in args.datasets:
        times = _time_chunks(datasets, args.chunk_size, args.chunks)
        print(
            f"{datasets} datasets, chunk_size={args.chunk_size}: "
            f"median {statistics.median(times) * 1e3:.2f} ms/chunk, "
            f"min {min(times) * 1e3:.2f} ms"
        )


if __name__ == "__main__":
    main()
