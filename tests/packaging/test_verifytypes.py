# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Check the installed wheel's public types and Torch adapter return types."""

import json
import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

from tests.packaging.conftest import REPO_ROOT, UV, _run

pytestmark = [
    pytest.mark.packaging,
    pytest.mark.skipif(
        UV is None, reason="uv is required to build/install distributions"
    ),
]

PYRIGHT = shutil.which("pyright")

_TORCH_CONSUMER = """
from typing import Any, assert_type

from torch.utils.data import Dataset, IterableDataset

from zephon import Pipeline


def check(pipeline: Pipeline) -> None:
    assert_type(pipeline.to_torch_dataset(), IterableDataset[Any])
    assert_type(pipeline.to_indexable_torch_dataset(), Dataset[Any])
"""


def _locked_torch_version() -> str:
    """Use the installed Torch version, falling back to the lockfile."""
    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        import tomllib

        lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
        for pkg in lock["package"]:
            if pkg["name"] == "torch":
                return pkg["version"]
        raise RuntimeError("torch not pinned in uv.lock")


@pytest.mark.skipif(PYRIGHT is None, reason="pyright not on PATH (dev env)")
def test_verifytypes_all_exported_symbols_complete(
    direct_wheel: Path, tmp_path: Path
) -> None:
    assert PYRIGHT is not None
    venv = tmp_path / "venv"
    _run([UV, "venv", "--python", sys.executable, str(venv)])
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    py = bin_dir / "python"
    torch_version = _locked_torch_version()
    _run(
        [
            UV,
            "pip",
            "install",
            "--python",
            str(py),
            str(direct_wheel),
            f"torch=={torch_version}",
        ]
    )

    # verifytypes follows the active environment; point it at the clean install.
    env = {
        **os.environ,
        "VIRTUAL_ENV": str(venv),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": "",
    }
    proc = subprocess.run(
        [PYRIGHT, "--verifytypes", "zephon", "--ignoreexternal", "--outputjson"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )
    report = json.loads(proc.stdout)
    completeness = report["typeCompleteness"]
    assert completeness["moduleRootDirectory"], (
        f"pyright did not resolve the installed package:\n{proc.stdout[:2000]}"
    )

    incomplete = {
        symbol["name"]
        for symbol in completeness["symbols"]
        if symbol.get("isExported") and not symbol.get("isTypeKnown")
    }
    assert not incomplete, (
        "exported symbols with unknown types (missing annotation in a public "
        "module, or an internal type reachable from a public signature):\n"
        + "\n".join(sorted(incomplete))
    )

    consumer = tmp_path / "torch_consumer.py"
    consumer.write_text(_TORCH_CONSUMER)
    _run([PYRIGHT, str(consumer)], cwd=str(tmp_path), env=env)
