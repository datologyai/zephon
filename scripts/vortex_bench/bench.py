# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Prepare, validate and explicitly launch direct-reader microbenchmarks.

`plan`, `verify` and `smoke` never collect timings. Only `run` with the explicit
--execute-measurements switch can start a timed run. Nothing is scheduled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ROOT, SCRIPT_ROOT, file_digest, write_json
from matrix import cases, requests


def manifest() -> dict[str, Any]:
    """Load the prepared local shard manifest."""
    return json.loads((ROOT / "data/manifest.json").read_text())


def worker_command(variant: str, *arguments: str) -> list[str]:
    """Run one checkout in its isolated Python environment."""
    return [
        "uv",
        "run",
        "--quiet",
        "--no-project",
        "--python",
        str(ROOT / "envs" / variant / "bin/python"),
        str(SCRIPT_ROOT / "worker.py"),
        *arguments,
        "--variant",
        variant,
    ]


def verify() -> None:
    """Verify all local shards and expected payload fingerprints."""
    for shard in manifest()["shards"]:
        for key, checksum in (("path", "sha256"), ("row_hashes", "row_hashes_sha256")):
            if file_digest(ROOT / shard[key]) != shard[checksum]:
                raise ValueError(f"Checksum mismatch: {shard[key]}")
    print("All shard and expected-row checksums match.")


def source_state(variant: str) -> dict[str, Any]:
    """Fingerprint the revision and files used by a variant."""
    checkout = ROOT / "checkouts" / variant
    prefix = ["git", "-C", str(checkout)]
    names = (
        subprocess.check_output(
            [*prefix, "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
        )
        .decode()
        .split("\0")
    )
    digest = hashlib.sha256()
    for name in sorted(set(filter(None, names))):
        path = checkout / name
        digest.update(name.encode())
        digest.update(path.read_bytes() if path.is_file() else b"<deleted>")
    return {
        "commit": subprocess.check_output(
            [*prefix, "rev-parse", "HEAD"], text=True
        ).strip(),
        "status": subprocess.check_output(
            [*prefix, "status", "--porcelain"], text=True
        ),
        "source_sha256": digest.hexdigest(),
    }


def host_info() -> dict[str, Any]:
    """Record machine details without collecting reader timings."""
    command = (
        ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"]
        if sys.platform == "darwin"
        else ["lscpu", "--json"]
    )
    hardware = subprocess.run(command, text=True, capture_output=True)
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "hardware": hardware.stdout,
        "hardware_query_error": hardware.stderr if hardware.returncode else None,
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    }


def select_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Filter the deterministic matrix by command-line selections."""
    selected = cases(manifest(), args.suite)
    if args.dataset:
        selected = [case for case in selected if case["dataset"] == args.dataset]
    if args.case_id:
        selected = [case for case in selected if case["id"] == args.case_id]
    if args.batch_size:
        selected = [case for case in selected if case["batch_size"] in args.batch_size]
    if not selected:
        raise ValueError("No cases selected")
    return selected


def plan(args: argparse.Namespace) -> None:
    """Validate and save request cases without running readers."""
    selected = select_cases(args)
    data = manifest()
    for case in selected:
        shards = [s for s in data["shards"] if s["dataset"] == case["dataset"]]
        # Validate every index without opening a reader or collecting timings.
        sequence = requests(case, shards, count=32)
        for shard_id, indices in sequence:
            if not all(0 <= i < shards[shard_id]["rows"] for i in indices):
                raise ValueError(f"Invalid request: {case['id']}")
    target = ROOT / "plans" / f"{args.suite}.json"
    write_json(target, {"suite": args.suite, "cases": selected})
    print(
        f"Prepared {len(selected)} cases / {sum(len(c['variants']) for c in selected)} variant-cases; no measurements. {target}"
    )


def smoke() -> None:
    """Check both readers against source fingerprints without timing."""
    verify()
    for variant in ("main", "candidate"):
        output = ROOT / "validation" / f"{variant}.json"
        output.parent.mkdir(exist_ok=True)
        subprocess.run(
            worker_command(variant, "smoke", "--output", str(output)),
            check=True,
            cwd=ROOT,
        )
    write_json(ROOT / "validation/host.json", host_info())
    print(
        "Both revisions passed direct-reader correctness checks. No timings collected."
    )


def run(args: argparse.Namespace) -> None:
    """Run the explicitly authorized measurements and record provenance."""
    if not args.execute_measurements:
        raise ValueError(
            "Measurements are disabled unless you explicitly pass --execute-measurements"
        )
    if args.seconds <= 0 or args.repeats <= 0:
        raise ValueError("seconds and repeats must be positive")
    selected = select_cases(args)
    sources = {variant: source_state(variant) for variant in ("main", "candidate")}
    if not args.allow_dirty and any(source["status"] for source in sources.values()):
        raise ValueError(
            "Checkout is dirty; commit the experiment or use --allow-dirty to record its patch"
        )
    output = args.output.resolve()
    # Refuse to overwrite or mix iterations. Use a new folder for each experiment.
    output.mkdir(parents=True, exist_ok=False)
    verify()
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": host_info(),
        "sources": sources,
        "manifest_sha256": file_digest(ROOT / "data/manifest.json"),
        "harness_sha256": {p.name: file_digest(p) for p in SCRIPT_ROOT.glob("*.py")},
        "suite": args.suite,
        "seconds": args.seconds,
        "repeats": args.repeats,
        "cases": selected,
        "status": "running",
    }
    for variant in sources:
        metadata[variant + "_environment"] = subprocess.check_output(
            [
                "uv",
                "pip",
                "freeze",
                "--python",
                str(ROOT / "envs" / variant / "bin/python"),
            ],
            text=True,
        )
        patch = subprocess.check_output(
            ["git", "-C", str(ROOT / "checkouts" / variant), "diff", "--binary", "HEAD"]
        )
        (output / f"{variant}.patch").write_bytes(patch)
    write_json(output / "run.json", metadata)
    try:
        with (output / "results.jsonl").open("w") as stream:
            for repeat in range(args.repeats):
                order = selected.copy()
                random.Random(1729 + repeat).shuffle(order)
                for case_number, case in enumerate(order):
                    variants = case["variants"].copy()
                    if (repeat + case_number) % 2:
                        variants.reverse()
                    for variant in variants:
                        if source_state(variant) != sources[variant]:
                            raise RuntimeError(
                                f"{variant} source changed during the run"
                            )
                        case_path = output / "current-case.json"
                        result_path = output / "current-result.json"
                        write_json(case_path, case)
                        log_path = output / "worker.log"
                        command = worker_command(
                            variant,
                            "measure",
                            "--case",
                            str(case_path),
                            "--seconds",
                            str(args.seconds),
                            "--output",
                            str(result_path),
                            "--execute-measurements",
                        )
                        print(
                            f"repeat {repeat + 1}/{args.repeats}: {variant}: {case['id']}",
                            flush=True,
                        )
                        with log_path.open("a") as log:
                            subprocess.run(
                                command,
                                check=True,
                                cwd=ROOT,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                timeout=900,
                            )
                        if source_state(variant) != sources[variant]:
                            raise RuntimeError(
                                f"{variant} source changed during measurement"
                            )
                        result = json.loads(result_path.read_text())
                        result["repeat"] = repeat
                        stream.write(json.dumps(result) + "\n")
                        stream.flush()
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata["status"] = "interrupted_or_failed"
        metadata["error"] = repr(exc)
        raise
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", metadata)
    summarize(output)


def summarize(output: Path) -> None:
    """Compare median throughput for matching request sequences."""
    records = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        grouped.setdefault(record["case"]["id"], {}).setdefault(
            record["variant"], []
        ).append(record)
    rows = []
    for case_id, variants in sorted(grouped.items()):
        row: dict[str, Any] = {"case": case_id}
        for variant, results in variants.items():
            row[variant + "_samples_per_second"] = statistics.median(
                r["samples_per_second"] for r in results
            )
            row[variant + "_peak_rss_mib"] = (
                statistics.median(r["process_peak_rss_bytes"] for r in results)
                / 1024**2
            )
            row[variant + "_repeats"] = len(results)
        if "main" in variants and "candidate" in variants:
            hashes = {
                r["request_sequence_sha256"]
                for results in variants.values()
                for r in results
            }
            if len(hashes) != 1:
                raise ValueError(f"Request mismatch: {case_id}")
            row["candidate_speedup"] = (
                row["candidate_samples_per_second"] / row["main_samples_per_second"]
            )
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with (output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output / 'summary.csv'}")


def main() -> None:
    """Dispatch the requested benchmark preparation or measurement command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument(
            "--suite",
            choices=["primary", "cache", "threads", "single_shard"],
            default="primary",
        )
        command.add_argument(
            "--dataset", choices=["fineweb_tokens", "fineweb_text", "mammoth"]
        )
        command.add_argument("--case-id")
        command.add_argument(
            "--batch-size",
            type=int,
            action="append",
            help="Repeat to select multiple request sizes",
        )
        if name == "run":
            command.add_argument("--execute-measurements", action="store_true")
            command.add_argument("--seconds", type=float, default=1.0)
            command.add_argument("--repeats", type=int, default=3)
            command.add_argument("--allow-dirty", action="store_true")
            command.add_argument("--output", type=Path, required=True)
    commands.add_parser("verify")
    commands.add_parser("smoke")
    command = commands.add_parser("summarize")
    command.add_argument("output", type=Path)
    args = parser.parse_args()
    {
        "plan": lambda: plan(args),
        "run": lambda: run(args),
        "smoke": smoke,
        "verify": verify,
        "summarize": lambda: summarize(args.output),
    }[args.action]()


if __name__ == "__main__":
    main()
