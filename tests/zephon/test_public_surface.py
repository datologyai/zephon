# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Executable manifest of the public facade surface.

Locks the set of public top-level modules and each public module's ``__all__``.
A new public top-level module, or a symbol leaking into a facade, fails these
tests until the expected values here are updated consciously.

We want to keep the public interface particularly clean because agents tend to use everything that is available publicly to them. This ensures agents do not accidentally use features not intended to be touched from the outside.

This guards the curated facades, not the full ``py.typed`` surface (every
non-underscore module/symbol is importable regardless of ``__all__``). The
privacy boundary is the leading underscore — see ``test_internal_boundary.py``.
"""

import importlib
import pkgutil

import zephon

# Non-underscore top-level modules under ``zephon/`` that are supported public
# API. ``_internal`` and ``_version`` are excluded (leading underscore).
EXPECTED_PUBLIC_MODULES = {
    "zephon.build_index",
    "zephon.debug",
    "zephon.io",
    "zephon.observability",
    "zephon.ops",
    "zephon.options",
    "zephon.pipeline",
    "zephon.types",
    "zephon.validation",
    "zephon.work",
}

# Exact ``__all__`` of every public module (the root plus the namespaces).
EXPECTED_ALL = {
    "zephon": [
        "Dataset",
        "InMemoryShard",
        "MixtureSpec",
        "Pipeline",
        "SampleBatch",
        "SampleMeta",
        "SampleRecord",
        "StaticMixtureWorkSource",
        "WorkSource",
        "__version__",
    ],
    "zephon.ops": [
        "Accumulator",
        "BaseOp",
        "CountingAccumulator",
        "DomainGroups",
        "OpContext",
        "OpTraits",
        "PackingAlgorithm",
        "PassthroughAccumulator",
        "ReadyBatch",
        "SpanSource",
        "SpecialTokensMode",
        "StageInfo",
        "pack_meta",
        "spawn_child",
        "tombstone_meta",
    ],
    "zephon.work": [
        "ComponentOrder",
        "MixtureReadConfig",
        "MixtureReadMode",
        "MixtureSpec",
        "StaticMixtureWorkSource",
        "TokenEstimation",
        "WorkChunk",
        "WorkSource",
    ],
    "zephon.io": [
        "CacheOptions",
        "Dataset",
        "InMemoryShard",
        "StoreOptions",
    ],
    "zephon.types": [
        "ChunkId",
        "ChunkOffset",
        "ComponentId",
        "ContributorRef",
        "DatasetId",
        "LaneId",
        "LineageIndex",
        "LineagePath",
        "LocalSampleId",
        "SampleBatch",
        "SampleCursor",
        "SampleCursorKey",
        "SampleId",
        "SampleMeta",
        "SampleNumeric",
        "SamplePayload",
        "SamplePayloadArray",
        "SamplePayloadAtom",
        "SamplePayloadDict",
        "SampleRecord",
        "ShardId",
        "StreamItem",
    ],
    "zephon.options": [
        "IpcTransport",
        "RuntimeOptions",
    ],
    "zephon.observability": [
        "ExecutionTrackingMode",
        "FetchTimingDelta",
        "FetchTimingTotals",
        "MTPQueueStats",
        "MetricsSinkConfig",
        "MetricsSinkMode",
        "PrefetchTimingDelta",
        "PrefetchTimingTotals",
    ],
    "zephon.pipeline": [
        "Pipeline",
    ],
    "zephon.build_index": [
        "build_index",
    ],
    "zephon.validation": [
        "Issue",
        "ValidationError",
        "ValidationReport",
    ],
    "zephon.debug": [
        "dump_semaphore_leak_report",
        "dump_semaphore_registry",
        "install_debug_hooks",
    ],
}


def _public_top_level_modules() -> set[str]:
    return {
        info.name
        for info in pkgutil.iter_modules(zephon.__path__, "zephon.")
        if not info.name.rsplit(".", 1)[-1].startswith("_")
    }


def test_public_module_set_is_frozen():
    assert _public_top_level_modules() == EXPECTED_PUBLIC_MODULES


def test_expected_all_covers_every_public_module():
    # EXPECTED_ALL and EXPECTED_PUBLIC_MODULES must not drift: a module in one
    # but not the other would leave part of the surface unfrozen.
    assert set(EXPECTED_ALL) == EXPECTED_PUBLIC_MODULES | {"zephon"}


def test_public_all_snapshot():
    for name, expected in EXPECTED_ALL.items():
        mod = importlib.import_module(name)
        assert sorted(mod.__all__) == sorted(expected), (
            f"public surface of {name} changed; update EXPECTED_ALL consciously"
        )


def test_all_symbols_are_importable():
    for name, expected in EXPECTED_ALL.items():
        mod = importlib.import_module(name)
        for sym in expected:
            assert hasattr(mod, sym), f"{name}.{sym} declared in __all__ but missing"
