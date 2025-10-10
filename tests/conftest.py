import pytest


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
