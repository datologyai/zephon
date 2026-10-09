# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Freeze every public module and its ``__all__``.

We want to keep the public interface particularly clean because agents tend to use everything that is available publicly to them. This ensures agents do not accidentally use features not intended to be touched from the outside.

Underscore-prefixed modules are private. Pyright also uses ``__all__`` as the
explicit re-export list for ``py.typed`` packages.
"""

import importlib
import pkgutil

import zephon

# Recursive public module set; underscore-prefixed modules are excluded.
EXPECTED_PUBLIC_MODULES = {
    "zephon.build_index",
    "zephon.debug",
    "zephon.debug.inspector",
    "zephon.debug.semaphore_registry",
    "zephon.debug.semaphore_tracker",
    "zephon.io",
    "zephon.io.dataset",
    "zephon.io.memory",
    "zephon.io.options",
    "zephon.observability",
    "zephon.observability.config",
    "zephon.observability.mtp_stats",
    "zephon.observability.stats",
    "zephon.ops",
    "zephon.ops.accumulators",
    "zephon.ops.accumulators.base",
    "zephon.ops.accumulators.counting",
    "zephon.ops.accumulators.passthrough",
    "zephon.ops.base",
    "zephon.ops.children",
    "zephon.ops.config",
    "zephon.ops.grouping",
    "zephon.ops.traits",
    "zephon.options",
    "zephon.pipeline",
    "zephon.types",
    "zephon.validation",
    "zephon.work",
    "zephon.work.base",
    "zephon.work.mixture",
    "zephon.work.static_mixture",
    "zephon.work.token_estimation",
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
        "MissingFieldMode",
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
        "tombstones_for_record",
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
        "ParquetRGCacheOptions",
        "StoreOptions",
        "VortexOptions",
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
        "FetchStageSummary",
        "FetchTimingDelta",
        "FetchTimingSummary",
        "FetchTimingTotals",
        "MTPQueueStats",
        "MetricsSinkConfig",
        "MetricsSinkMode",
        "PipelineSummary",
        "PrefetchTimingDelta",
        "PrefetchTimingSummary",
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
        "preflight_tokenizers",
    ],
    "zephon.debug": [
        "DatasetInspector",
        "dump_semaphore_leak_report",
        "dump_semaphore_registry",
        "install_debug_hooks",
    ],
    "zephon.debug.semaphore_registry": [
        "dump_semaphore_registry",
    ],
    "zephon.debug.semaphore_tracker": [
        "RegistrationInfo",
        "dump_semaphore_leak_report",
        "install_debug_hooks",
    ],
    "zephon.debug.inspector": [
        "DatasetInspector",
    ],
    "zephon.io.dataset": [
        "Dataset",
    ],
    "zephon.io.memory": [
        "InMemoryShard",
    ],
    "zephon.io.options": [
        "CacheOptions",
        "ParquetRGCacheOptions",
        "StoreOptions",
        "VortexOptions",
        "parse_size_bytes",
    ],
    "zephon.observability.config": [
        "ExecutionTrackingMode",
        "MetricsSinkConfig",
        "MetricsSinkMode",
    ],
    "zephon.observability.mtp_stats": [
        "MTPQueueStats",
    ],
    "zephon.observability.stats": [
        "FetchStageSummary",
        "FetchTimingDelta",
        "FetchTimingSummary",
        "FetchTimingTotals",
        "NodeMetricsDelta",
        "NodeSummary",
        "PipelineSummary",
        "PrefetchTimingDelta",
        "PrefetchTimingSummary",
        "PrefetchTimingTotals",
        "StageSummary",
    ],
    "zephon.ops.accumulators": [
        "Accumulator",
        "CountingAccumulator",
        "PassthroughAccumulator",
        "ReadyBatch",
    ],
    "zephon.ops.accumulators.base": [
        "Accumulator",
        "ReadyBatch",
    ],
    "zephon.ops.accumulators.counting": [
        "CountingAccumulator",
    ],
    "zephon.ops.accumulators.passthrough": [
        "PassthroughAccumulator",
    ],
    "zephon.ops.base": [
        "BaseOp",
        "OpContext",
        "StageInfo",
    ],
    "zephon.ops.children": [
        "pack_meta",
        "spawn_child",
        "tombstone_meta",
        "tombstones_for_record",
    ],
    "zephon.ops.config": [
        "MissingFieldMode",
        "PackingAlgorithm",
        "SpanSource",
        "SpecialTokensMode",
    ],
    "zephon.ops.grouping": [
        "DomainGroups",
    ],
    "zephon.ops.traits": [
        "OpTraits",
    ],
    "zephon.work.base": [
        "ComponentOrder",
        "MixtureComponent",
        "MixtureReadConfig",
        "MixtureReadMode",
        "SamplesPerComponent",
        "SourcedSampleId",
        "WorkChunk",
        "WorkSource",
    ],
    "zephon.work.mixture": [
        "MixtureSpec",
    ],
    "zephon.work.static_mixture": [
        "ShuffleBlockSpec",
        "StaticMixtureWorkSource",
    ],
    "zephon.work.token_estimation": [
        "DEFAULT_FALLBACK_TOKENS_PER_BYTE",
        "PerShardTokenCost",
        "TokenEstimation",
        "TokenRatio",
        "prime_token_ratios",
    ],
}


def _public_modules() -> set[str]:
    found: set[str] = set()

    def walk(pkg) -> None:
        for info in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
            if info.name.rsplit(".", 1)[-1].startswith("_"):
                continue
            found.add(info.name)
            if info.ispkg:
                walk(importlib.import_module(info.name))

    walk(zephon)
    return found


def test_public_module_set_is_frozen():
    assert _public_modules() == EXPECTED_PUBLIC_MODULES


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
