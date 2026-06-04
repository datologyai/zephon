import importlib.util
import stat
import sys
from pathlib import Path

import pytest


def _ensure_torch_shm_manager_is_executable() -> None:
    """Repair macOS torch SHM helper permissions for local test runs.

    Some macOS environments end up with ``torch/bin/torch_shm_manager``
    installed without execute bits, which breaks tests that exercise torch's
    shared-memory path. Fixing it here keeps the workaround scoped to tests.
    """

    if sys.platform != "darwin":
        return

    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return

    torch_dir = Path(spec.origin).resolve().parent
    manager = torch_dir / "bin" / "torch_shm_manager"
    if not manager.exists():
        return

    mode = manager.stat().st_mode
    execute_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    if mode & execute_bits:
        return

    manager.chmod(mode | execute_bits)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests marked as integration",
    )
    parser.addoption(
        "--repro-iters",
        action="store",
        type=int,
        default=4,
        help="Number of reproducibility iterations for tests that repeat pipelines",
    )


def pytest_configure(config: pytest.Config) -> None:
    _ensure_torch_shm_manager_is_executable()
    config.addinivalue_line(
        "markers", "integration: mark tests as integration (skipped by default)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(
        reason="use --run-integration to run integration tests"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def repro_iters(pytestconfig: pytest.Config) -> int:
    val = pytestconfig.getoption("--repro-iters")
    try:
        return int(val)
    except Exception:
        return 4


@pytest.fixture(autouse=True)
def _isolated_shard_catalogs(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
):
    """Isolate the shard-catalog dir and registry per test.

    ``Dataset.from_path`` + finalize/attach otherwise write content-addressed
    catalog artifacts into the *process-global* catalog dir (``$TMPDIR/zephon-
    {uid}/catalog`` or the developer's ``ZEPHON_CATALOG_DIR``), accumulating
    across runs, and the sticky ``_CATALOG_DIR``/``_REGISTRY`` globals leak
    state between tests. Point both at a per-test temp dir instead; the env var
    also propagates to spawned worker processes in integration tests. Tests
    that manage the dir themselves (tests/zephon/io/catalog) simply override.
    """
    from zephon.io.catalog import clear_registry, set_catalog_dir

    catalog_dir = tmp_path_factory.mktemp("shard-catalogs")
    monkeypatch.setenv("ZEPHON_CATALOG_DIR", str(catalog_dir))
    set_catalog_dir(None)  # re-resolve the process-global dir from the env var
    clear_registry()
    yield
    clear_registry()
