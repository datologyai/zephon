# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""One isolated Zephon revision per process; direct Vortex shard operations only.

These are intentional implementation-level imports for a maintainer benchmark.
No Pipeline, fetch op, resolver, shard download, or Zephon store is constructed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import fields
from pathlib import Path
from typing import Any

from common import ROOT, row_digest, write_json
from matrix import CACHES, requests


class Readers:
    """Exercise the actual format opener, including candidate cache ownership."""

    def __init__(
        self, variant: str, case: dict[str, Any], shards: list[dict[str, Any]]
    ) -> None:
        self.variant, self.case, self.shards = variant, case, shards
        self.stack = ExitStack()
        # Configure both revisions equally; main has no Zephon Vortex thread cap.
        os.environ["ZEPHON_VORTEX_THREADS"] = str(case["threads"])
        os.environ["VORTEX_MAX_THREADS"] = str(case["threads"])
        import pyarrow as pa
        import vortex

        import zephon
        from zephon._internal.io.formats.vortex import VortexFormat
        from zephon._internal.io.types import (
            LocalShardFile,
            LocalShardRef,
            ShardFile,
            ShardLocator,
        )

        expected_source = (ROOT / "checkouts" / variant / "zephon").resolve()
        if Path(zephon.__file__).resolve().parent != expected_source:
            raise RuntimeError(f"Wrong Zephon checkout imported: {zephon.__file__}")
        vortex.set_worker_threads(case["threads"])
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        format_handler = VortexFormat()
        if variant == "candidate":
            from zephon.io import StoreOptions, VortexOptions

            # Support the option locations used by the historical measurements
            # and the current public reader options without changing timed reads.
            limits = CACHES[case["cache"]]
            vortex_fields = {field.name for field in fields(VortexOptions)}
            store_fields = {field.name for field in fields(StoreOptions)}
            vortex_options = {"segment_cache_bytes": limits["segment_cache_bytes"]}
            store_options: dict[str, Any] = {}
            if "worker_threads" in vortex_fields:
                vortex_options["worker_threads"] = case["threads"]
            if "metadata_cache_entries" in vortex_fields:
                vortex_options["metadata_cache_entries"] = limits[
                    "metadata_cache_entries"
                ]
            elif "metadata_cache_entries" in store_fields:
                store_options["metadata_cache_entries"] = limits[
                    "metadata_cache_entries"
                ]
            else:
                from zephon.io import CacheOptions

                store_options["cache"] = CacheOptions(
                    metadata_cache_entries=limits["metadata_cache_entries"]
                )
            options = StoreOptions(
                vortex=VortexOptions(**vortex_options), **store_options
            )
            self.opener = self.stack.enter_context(
                format_handler.create_opener(None, options)
            )
        else:
            self.opener = format_handler
        self.retained: dict[int, Any] = {}
        self.references = []
        # The benchmark constructs these once, outside the timed operations.
        for index, info in enumerate(shards):
            path = ROOT / info["path"]
            extra = {"length": info["rows"]}
            locator = ShardLocator(
                dataset=info["dataset"],
                shard_id=index,
                format="vortex",
                root=str(path.parent),
                raw=ShardFile(path.name, info["bytes"], {}),
                extra=extra,
            )
            local = LocalShardRef(raw=LocalShardFile(path, info["bytes"]), extra=extra)
            self.references.append((locator, local))

    def open(self, index: int) -> Any:
        return self.opener.open_shard(*self.references[index])

    def initialize_retained(self) -> None:
        if self.case["lifetime"] == "retained":
            count = 1 if self.case["shard_pattern"] == "single" else len(self.shards)
            self.retained = {index: self.open(index) for index in range(count)}

    def read(self, shard_id: int, indices: list[int]) -> list[dict[str, Any]]:
        retained = self.case["lifetime"] == "retained"
        reader = self.retained[shard_id] if retained else self.open(shard_id)
        try:
            if self.case["api"] == "getitem":
                if len(indices) != 1:
                    raise ValueError("getitem cases must contain one requested sample")
                return [reader[indices[0]]]
            return reader.getsamples(indices)
        finally:
            if not retained:
                reader.close()

    def close(self) -> None:
        for reader in self.retained.values():
            reader.close()
        self.stack.close()


def provenance(variant: str) -> dict[str, Any]:
    """Record the checkout and installed dependency versions."""
    checkout = ROOT / "checkouts" / variant
    return {
        "variant": variant,
        "commit": subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(checkout), "status", "--porcelain"], text=True
            ).strip()
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("zephon", "vortex-data", "pyarrow", "numpy")
        },
    }


def smoke(variant: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """Check values, order, duplicates, reader lifetimes and types, without clocks."""
    import numpy as np

    checks = []
    for dataset in ("fineweb_tokens", "fineweb_text", "mammoth"):
        shards = [s for s in manifest["shards"] if s["dataset"] == dataset]
        for lifetime in ("retained", "reopen"):
            case = dict(
                dataset=dataset,
                cache="default",
                threads=1,
                lifetime=lifetime,
                api="bulk",
                shard_pattern="round_robin",
            )
            readers = Readers(variant, case, shards)
            try:
                readers.initialize_retained()
                # Each shard is checked once. Two distinct rows plus a duplicate
                # keep even Mammoth's large images bounded during preparation.
                for shard_id, shard in enumerate(shards):
                    expected = json.loads((ROOT / shard["row_hashes"]).read_text())
                    indices = [shard["rows"] - 1, 0, shard["rows"] - 1]
                    rows = readers.read(shard_id, indices)
                    if len(rows) != len(indices):
                        raise AssertionError("Wrong returned row count")
                    for row, index in zip(rows, indices):
                        if row_digest(row, shard["columns"]) != expected[index]:
                            raise AssertionError(
                                f"Wrong value/order: {variant} {shard['path']} row {index}"
                            )
                        if variant == "candidate":
                            for name, kind in shard["columns"].items():
                                if kind.startswith("array:"):
                                    if not isinstance(row[name], np.ndarray) or row[
                                        name
                                    ].dtype != np.dtype(kind.split(":", 1)[1]):
                                        raise AssertionError(
                                            f"Wrong numeric return type: {name}"
                                        )
                    del rows
                checked_sizes = []
                if dataset.startswith("fineweb_"):
                    expected = json.loads((ROOT / shards[0]["row_hashes"]).read_text())
                    for size in (64, 256):
                        request_case = dict(
                            case, batch_size=size, pattern="random", seed=1729
                        )
                        _, indices = requests(request_case, shards, count=1)[0]
                        rows = readers.read(0, indices)
                        if len(rows) != size or any(
                            row_digest(row, shards[0]["columns"]) != expected[index]
                            for row, index in zip(rows, indices)
                        ):
                            raise AssertionError(
                                f"Incorrect FineWeb request of {size} samples"
                            )
                        del rows
                        checked_sizes.append(size)
                readers.case["api"] = "getitem"
                row = readers.read(0, [0])[0]
                expected_first = json.loads(
                    (ROOT / shards[0]["row_hashes"]).read_text()
                )[0]
                if row_digest(row, shards[0]["columns"]) != expected_first:
                    raise AssertionError("getitem disagrees with original source")
                checks.append(
                    {
                        "dataset": dataset,
                        "lifetime": lifetime,
                        "shards": len(shards),
                        "batch_sizes_checked": checked_sizes,
                        "status": "passed",
                    }
                )
            finally:
                readers.close()
    return {
        "mode": "correctness_only",
        "measurements_run": False,
        **provenance(variant),
        "checks": checks,
    }


def measure(
    variant: str, case: dict[str, Any], manifest: dict[str, Any], seconds: float
) -> dict[str, Any]:
    """Time materialized row reads and release, with request generation excluded."""
    from time import perf_counter_ns

    shards = [s for s in manifest["shards"] if s["dataset"] == case["dataset"]]
    sequence = requests(case, shards, count=32)
    readers = Readers(variant, case, shards)
    samples = []
    try:
        readers.initialize_retained()
        # Warm exactly the same complete request sequence on both revisions.
        # It may exceed a configured segment cache. Never claim cold SSD IO.
        for shard_id, indices in sequence:
            rows = readers.read(shard_id, indices)
            del rows
        elapsed = 0
        count = 0
        # Finish complete passes so slower variants see the same distribution
        # of shards and indices, even when one call exceeds the target duration.
        while (
            elapsed < seconds * 1e9 or count % len(sequence) != 0 or count == 0
        ) and count < 1_000_000:
            shard_id, indices = sequence[count % len(sequence)]
            start = perf_counter_ns()
            rows = readers.read(shard_id, indices)
            if len(rows) != len(indices):
                raise AssertionError("Short read")
            del rows  # Include releasing the returned objects in both variants.
            duration = perf_counter_ns() - start
            samples.append(duration)
            elapsed += duration
            count += 1
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Sample values outside timing, from exactly the measured reader path.
        shard_id, indices = sequence[0]
        rows = readers.read(shard_id, indices)
        expected = json.loads((ROOT / shards[shard_id]["row_hashes"]).read_text())
        for row, index in zip(rows, indices):
            if row_digest(row, shards[shard_id]["columns"]) != expected[index]:
                raise AssertionError("Measured reader returned incorrect values")
        del rows
        import hashlib

        import numpy as np
        import vortex

        return {
            **provenance(variant),
            "case": case,
            "calls": count,
            "samples": count * case["batch_size"],
            "elapsed_ns": elapsed,
            "calls_per_second": count * 1e9 / elapsed,
            "samples_per_second": count * case["batch_size"] * 1e9 / elapsed,
            "latency_ns": {
                "median": float(np.median(samples)),
                "p95": float(np.quantile(samples, 0.95)),
                "p99": float(np.quantile(samples, 0.99)),
            },
            "call_durations_ns": samples,
            "process_peak_rss_bytes": int(
                rss if sys.platform == "darwin" else rss * 1024
            ),
            "vortex_workers": vortex.worker_threads(),
            "warmup_calls": len(sequence),
            "request_sequence_sha256": hashlib.sha256(
                json.dumps(sequence).encode()
            ).hexdigest(),
            "request_sequence_length": len(sequence),
            "os_page_cache": "uncontrolled; no cache flushing",
            "correctness": "passed",
        }
    finally:
        readers.close()


def main() -> None:
    """Run a correctness check or an explicitly authorized measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "measure"])
    parser.add_argument("--variant", choices=["main", "candidate"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--execute-measurements", action="store_true")
    args = parser.parse_args()
    if args.mode == "measure" and (
        not args.execute_measurements or args.case is None or args.seconds <= 0
    ):
        parser.error(
            "Measurement requires --execute-measurements, --case and positive --seconds"
        )
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    if args.mode == "smoke":
        result = smoke(args.variant, manifest)
    else:
        result = measure(
            args.variant, json.loads(args.case.read_text()), manifest, args.seconds
        )
    write_json(args.output, result)
    print(f"{args.mode}: {args.variant}: passed", flush=True)


if __name__ == "__main__":
    main()
