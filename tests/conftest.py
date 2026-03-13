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
