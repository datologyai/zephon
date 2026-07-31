# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for installed-package tests."""

import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import pytest

UV = shutil.which("uv")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if proc.returncode != 0:
        raise AssertionError(
            f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return proc


def _exactly_one(paths: Iterable[Path]) -> Path:
    matches = list(paths)
    assert len(matches) == 1, f"expected exactly one artifact, got {matches}"
    return matches[0]


@pytest.fixture(scope="session")
def dist_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("dist")
    _run([UV, "build", "--out-dir", str(out)], cwd=str(REPO_ROOT))
    return out


@pytest.fixture(scope="session")
def direct_wheel(dist_dir: Path) -> Path:
    return _exactly_one(dist_dir.glob("*.whl"))


@pytest.fixture(scope="session")
def sdist_wheel(dist_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build from the sdist to verify that it contains every subpackage."""
    sdist = _exactly_one(dist_dir.glob("*.tar.gz"))
    out = tmp_path_factory.mktemp("sdist-wheel")
    _run([UV, "build", "--wheel", str(sdist), "--out-dir", str(out)])
    return _exactly_one(out.glob("*.whl"))


@pytest.fixture(params=["direct_wheel", "sdist_wheel"])
def wheel(request: pytest.FixtureRequest) -> Path:
    return request.getfixturevalue(request.param)
