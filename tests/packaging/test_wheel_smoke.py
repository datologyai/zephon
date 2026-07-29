# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the built wheel and sdist.

Build the distribution (directly and via the sdist), assert every zephon
subpackage and py.typed ship, then install into a clean venv and run a tiny
pipeline from the installed package. Guards the packages.find config so a new
subpackage (e.g. zephon._internal) cannot silently drop out of the wheel.

Gated behind --run-packaging (make packaging); skipped by default because
building a wheel and a venv is slow.
"""

import os
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pytest

UV = shutil.which("uv")

pytestmark = [
    pytest.mark.packaging,
    pytest.mark.skipif(
        UV is None, reason="uv is required to build/install distributions"
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "zephon"

# Minimal end-to-end pipeline exercising work source -> engine -> output using
# only core runtime deps (no tokenizer/torch). Run from a scratch cwd so the
# import resolves to the installed wheel, never the source tree.
_SMOKE_SCRIPT = """
import zephon
from zephon import Pipeline
from zephon.io import Dataset, InMemoryShard
from zephon.work import MixtureSpec, StaticMixtureWorkSource

assert "site-packages" in zephon.__file__, zephon.__file__

ds = Dataset.from_dict("demo", {0: InMemoryShard([{"text": f"row {i}"} for i in range(6)])})
ws = StaticMixtureWorkSource(
    [ds], mixture=MixtureSpec({"demo": 1.0}), chunk_size=1, seed=42, shuffle_shards=False
)
pipe = Pipeline(ws).decode_text().batch(microbatch_size=3, drop_last=False)
assert sum(1 for _ in pipe) >= 1
print("SMOKE OK")
"""


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


def _source_packages() -> set[str]:
    """Dotted names of every ``zephon`` package on disk (dir with __init__.py)."""
    return {
        ".".join(init.parent.relative_to(REPO_ROOT).parts)
        for init in SRC.rglob("__init__.py")
    }


def _wheel_packages(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    return {
        n[: -len("/__init__.py")].replace("/", ".")
        for n in names
        if n.endswith("/__init__.py")
    }


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
    """Wheel built from the sdist -- no git checkout, so packaging config alone
    (not the setuptools-scm git-file sweep) has to carry every subpackage."""
    sdist = _exactly_one(dist_dir.glob("*.tar.gz"))
    out = tmp_path_factory.mktemp("sdist-wheel")
    _run([UV, "build", "--wheel", str(sdist), "--out-dir", str(out)])
    return _exactly_one(out.glob("*.whl"))


@pytest.fixture(params=["direct_wheel", "sdist_wheel"])
def wheel(request: pytest.FixtureRequest) -> Path:
    return request.getfixturevalue(request.param)


def test_wheel_ships_every_subpackage(wheel: Path) -> None:
    missing = _source_packages() - _wheel_packages(wheel)
    assert not missing, f"wheel is missing subpackages: {sorted(missing)}"
    with zipfile.ZipFile(wheel) as zf:
        assert "zephon/py.typed" in zf.namelist(), (
            "py.typed marker (PEP 561) not shipped"
        )


def test_installed_wheel_imports_and_runs(wheel: Path, tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    _run([UV, "venv", "--python", sys.executable, str(venv)])
    py = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
    _run([UV, "pip", "install", "--python", str(py), str(wheel)])
    proc = _run(
        [str(py), "-c", _SMOKE_SCRIPT],
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert "SMOKE OK" in proc.stdout
