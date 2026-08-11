# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Enforce public import boundaries and review direct test coupling."""

import ast
import importlib
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_UNSUPPORTED_ROOTS = "_internal|api|core|runners|utils"
# A dotted reference (``zephon._internal.x``), bounded by a trailing `.` or
# word-end so doc-page names like ``zephon_api`` (underscore) don't match; or a
# ``from zephon import _internal`` style import. Private and removed roots remain
# forbidden so external surfaces cannot introduce unsupported import paths.
_FORBIDDEN = re.compile(
    rf"\bzephon\.(?:{_UNSUPPORTED_ROOTS})(?=\.|\b)"
    rf"|\bfrom\s+zephon\s+import\b[^\n]*\b(?:{_UNSUPPORTED_ROOTS})\b"
)

# Exact direct imports allowed outside the mirrored internal test tree.
_ALLOWED_INTERNAL_TEST_IMPORTS: dict[str, frozenset[str]] = {
    # Reset sticky catalog state and the mmap registry between tests.
    "tests/conftest.py": frozenset(
        {
            "zephon._internal.io.catalog:clear_registry",
            "zephon._internal.io.catalog:set_catalog_dir",
        }
    ),
    # Centralize white-box catalog setup shared by dataset and internal tests.
    "tests/_catalog_helpers.py": frozenset(
        {
            "zephon._internal.io.catalog:CatalogSet",
            "zephon._internal.io.catalog:DatasetHeader",
            "zephon._internal.io.catalog:ShardCatalog",
            "zephon._internal.io.catalog:io",
            "zephon._internal.io.catalog.builder:pack_locators",
            "zephon._internal.io.types:ShardLocator",
        }
    ),
    # HF range and revision details have no public seam.
    "tests/integration/test_hf_integration.py": frozenset(
        {
            "zephon._internal.io.storage:HFBackend",
            "zephon._internal.io.storage._hf_uri:parse_hf_uri",
            "zephon._internal.io.storage.hf:_DATASETS_SERVER_PARQUET_URL",
        }
    ),
    # Internal patches make cache interleavings deterministic; metrics stay public.
    "tests/integration/test_prefetch_integration.py": frozenset(
        {
            "zephon._internal.observability.collector:PipelineCollector",
            "zephon._internal.ops.fetch:FetchOp",
            "zephon._internal.ops.prefetch:PrefetchOp",
        }
    ),
    # The standalone op is an independent oracle for the pipeline path.
    "tests/integration/test_tokenize_chat_pipeline.py": frozenset(
        {"zephon._internal.ops.tokenize_chat:TokenizeChat"}
    ),
    # The internal tokenizer provides an independent token-count oracle; the
    # counting spec is internal and has no public re-export.
    "tests/integration/test_token_aware_mixture.py": frozenset(
        {
            "zephon._internal.ops.tokenize_chat:TokenizeChat",
            "zephon._internal.token_counting:TextTokenCountingSpec",
        }
    ),
    # Pump timing exists only at the runner-to-collector boundary.
    "tests/integration/test_pump_timing_end_to_end.py": frozenset(
        {
            "zephon._internal.observability.collector:CollectorConfig",
            "zephon._internal.observability.collector:PipelineCollector",
            "zephon._internal.runners.threads:ThreadStageRunner",
        }
    ),
    # Internal retry constants make SHM exhaustion deterministic.
    "tests/integration/test_shm_backpressure.py": frozenset(
        {"zephon._internal.utils.shm"}
    ),
    # Inspector lifecycle coverage verifies its internal cache owner is closed.
    "tests/zephon/debug/test_inspector.py": frozenset(
        {"zephon._internal.io.resolvers:CacheManager"}
    ),
    # Op classes and traits are not exposed by the fluent API.
    "tests/zephon/test_pipeline_builder.py": frozenset(
        {
            "zephon._internal.ops.pack_sequences:PackSequences",
            "zephon._internal.ops.shuffle_buffer:ShuffleBuffer",
            "zephon._internal.ops.tokenize_text:TokenizeText",
        }
    ),
    # max_buffer_size=None is visible only on the constructed op.
    "tests/zephon/test_pipeline.py": frozenset(
        {"zephon._internal.ops.ensure_mixture:EnsureMixture"}
    ),
    # Forwarded token IDs are visible only on the constructed op.
    "tests/zephon/test_pipeline_tokenize_decode.py": frozenset(
        {"zephon._internal.ops.tokenize_text:TokenizeText"}
    ),
    # Degenerate plans exercise errors unreachable through public builders.
    "tests/zephon/test_pipeline_errors.py": frozenset({"zephon._internal.graph:Plan"}),
    # Counting-spec types have no public marker.
    "tests/zephon/test_pipeline_token_priming.py": frozenset(
        {
            "zephon._internal.ops.tokenize_chat:ChatTokenCountingSpec",
            "zephon._internal.token_counting:TextTokenCountingSpec",
        }
    ),
    # Probe coverage requires enumerating internal tokenizer ops.
    "tests/zephon/test_validation.py": frozenset(
        {"zephon._internal.ops.tokenize_base:TokenizeBase"}
    ),
    # Token-mode coverage needs the internal counting spec.
    "tests/zephon/work/test_static_mixture.py": frozenset(
        {"zephon._internal.token_counting:TextTokenCountingSpec"}
    ),
    # Pin WorkChunk state to the internal checkpoint schema version.
    "tests/zephon/work/test_base.py": frozenset(
        {"zephon._internal.checkpoint:WORK_CHUNK_VERSION"}
    ),
    # Census and calibration coverage needs internal plans and oracles.
    "tests/zephon/work/test_token_estimation.py": frozenset(
        {
            "zephon._internal.io.stores.multi:build_multi_dataset_store",
            "zephon._internal.observability.size_estimator:content_bytes",
            "zephon._internal.ops.map_transform:MapTransform",
            "zephon._internal.ops.tokenize_chat:ChatTokenCountingSpec",
            "zephon._internal.token_counting:CountPlan",
            "zephon._internal.token_counting:DeliveredTokenCounter",
            "zephon._internal.token_counting:TextTokenCountingSpec",
            "zephon._internal.token_counting:_TextCounter",
            "zephon._internal.utils.tokenizer:fallback_tokenizer",
        }
    ),
}


def _consumer_files() -> list[Path]:
    files: list[Path] = []
    files.extend((_REPO / "examples").rglob("*.py"))
    files.extend((_REPO / "docs").rglob("*.md"))
    files.extend((_REPO / "docs").rglob("*.rst"))
    readme = _REPO / "README.md"
    if readme.exists():
        files.append(readme)
    files.append(_REPO / "tests" / "typing" / "public_consumer.py")
    return files


def test_consumer_surfaces_use_only_supported_paths():
    offenders: list[str] = []
    for f in _consumer_files():
        for i, line in enumerate(f.read_text().splitlines(), 1):
            m = _FORBIDDEN.search(line)
            if m:
                offenders.append(f"{f.relative_to(_REPO)}:{i}: {line.strip()}")
    assert not offenders, (
        "external surfaces must use only supported import paths:\n"
        + "\n".join(offenders)
    )


def _internal_imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "zephon._internal" or mod.startswith("zephon._internal."):
                found.update(f"{mod}:{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(
                alias.name
                for alias in node.names
                if alias.name == "zephon._internal"
                or alias.name.startswith("zephon._internal.")
            )
    return found


def _test_files_outside_internal_tree() -> list[Path]:
    internal_tree = _REPO / "tests" / "zephon" / "_internal"
    return [
        f
        for f in sorted((_REPO / "tests").rglob("*.py"))
        if internal_tree not in f.parents
    ]


def test_direct_internal_imports_outside_internal_tests_are_reviewed():
    unexpected: list[str] = []
    stale: list[str] = []
    seen: set[str] = set()
    for f in _test_files_outside_internal_tree():
        rel = f.relative_to(_REPO).as_posix()
        seen.add(rel)
        found = _internal_imports(f)
        allowed = _ALLOWED_INTERNAL_TEST_IMPORTS.get(rel, frozenset())
        unexpected.extend(f"{rel}: {imp}" for imp in sorted(found - allowed))
        stale.extend(f"{rel}: {imp}" for imp in sorted(allowed - found))
    stale.extend(
        f"{rel}: (file not found)"
        for rel in sorted(_ALLOWED_INTERNAL_TEST_IMPORTS)
        if rel not in seen
    )
    assert not unexpected, (
        "unreviewed direct zephon._internal import outside the mirrored internal "
        "test tree; extend the allowlist only for deliberate white-box coverage:\n"
        + "\n".join(unexpected)
    )
    assert not stale, (
        "stale allowlist entries (import no longer present) — remove them:\n"
        + "\n".join(stale)
    )


def test_internal_package_ships_but_is_not_exported():
    import zephon

    importlib.import_module("zephon._internal")  # must ship / be importable
    assert "_internal" not in zephon.__all__
