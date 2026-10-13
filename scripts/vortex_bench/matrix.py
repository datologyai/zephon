# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Deterministic direct-reader cases and precomputed request sequences."""

from __future__ import annotations

import random
from typing import Any

PATTERNS = (
    "contiguous",
    "contiguous_shuffled",
    "random_sorted",
    "random",
    "duplicates",
    "hotset",
)
BATCH_SIZES = {
    "fineweb_tokens": (1, 8, 64, 256, 1024),
    "fineweb_text": (1, 8, 64, 256, 1024),
    "mammoth": (1, 2, 4, 8),
}
CACHES = {
    "default": {"metadata_cache_entries": 256, "segment_cache_bytes": 64 * 1024**2},
    "none": {"metadata_cache_entries": 0, "segment_cache_bytes": 0},
    "footer_only": {"metadata_cache_entries": 256, "segment_cache_bytes": 0},
    "segments_8mib": {
        "metadata_cache_entries": 256,
        "segment_cache_bytes": 8 * 1024**2,
    },
    "segments_8mib_no_footer": {
        "metadata_cache_entries": 0,
        "segment_cache_bytes": 8 * 1024**2,
    },
    "segments_64mib_no_footer": {
        "metadata_cache_entries": 0,
        "segment_cache_bytes": 64 * 1024**2,
    },
}


def requests(
    case: dict[str, Any], shards: list[dict[str, Any]], count: int = 128
) -> list[tuple[int, list[int]]]:
    """Generate indices outside timing; never silently turn unique reads into replacement sampling."""
    rng = random.Random(case["seed"])
    result = []
    size = case["batch_size"]
    for step in range(count):
        shard = 0 if case["shard_pattern"] == "single" else step % len(shards)
        rows = shards[shard]["rows"]
        if not 0 < size <= rows:
            raise ValueError(f"Invalid batch size {size} for {rows} rows")
        pattern = case["pattern"]
        if pattern.startswith("contiguous"):
            # Advance through each file, not just repeatedly read its first rows.
            visit = step if case["shard_pattern"] == "single" else step // len(shards)
            start = (visit * size) % (rows - size + 1)
            indices = list(range(start, start + size))
            if pattern == "contiguous_shuffled":
                rng.shuffle(indices)
        elif pattern == "duplicates":
            unique = rng.sample(range(rows), max(1, size // 2))
            indices = (unique * 2 + unique[:1])[:size]
            rng.shuffle(indices)
        else:
            population = (
                rows if pattern != "hotset" else min(rows, max(size, rows // 10))
            )
            indices = rng.sample(range(population), size)
            if pattern == "random_sorted":
                indices.sort()
        result.append((shard, indices))
    return result


def cases(manifest: dict[str, Any], suite: str) -> list[dict[str, Any]]:
    """Keep the main comparison separate from candidate-only cache/thread ablations."""
    result = []

    def add(
        dataset: str,
        size: int,
        pattern: str,
        lifetime: str,
        api: str = "bulk",
        cache: str = "default",
        threads: int = 1,
        shard_pattern: str = "round_robin",
    ) -> None:
        case = dict(
            dataset=dataset,
            batch_size=size,
            pattern=pattern,
            lifetime=lifetime,
            api=api,
            cache=cache,
            threads=threads,
            shard_pattern=shard_pattern,
            seed=1729,
        )
        case["id"] = "-".join(
            str(case[key])
            for key in (
                "dataset",
                "api",
                "batch_size",
                "pattern",
                "lifetime",
                "cache",
                "threads",
                "shard_pattern",
            )
        )
        case["variants"] = (
            ["main", "candidate"] if suite == "primary" else ["candidate"]
        )
        result.append(case)

    for dataset, sizes in BATCH_SIZES.items():
        available = [s for s in manifest["shards"] if s["dataset"] == dataset]
        sizes = tuple(n for n in sizes if n <= min(s["rows"] for s in available))
        if suite == "primary":
            for size in sizes:
                for pattern in PATTERNS:
                    for lifetime in ("retained", "reopen"):
                        add(dataset, size, pattern, lifetime)
            for pattern in ("contiguous", "random", "hotset"):
                for lifetime in ("retained", "reopen"):
                    add(dataset, 1, pattern, lifetime, api="getitem")
        elif suite == "cache":
            for size in (1, sizes[-2]):
                for lifetime in ("retained", "reopen"):
                    for cache in CACHES:
                        add(dataset, size, "random", lifetime, cache=cache)
        elif suite == "threads":
            for threads in (1, 2, 4):
                for pattern in ("contiguous", "random"):
                    add(dataset, sizes[-2], pattern, "reopen", threads=threads)
        elif suite == "single_shard":
            for size in (1, sizes[-2]):
                for lifetime in ("retained", "reopen"):
                    add(dataset, size, "random", lifetime, shard_pattern="single")
        else:
            raise ValueError(suite)
    return result
