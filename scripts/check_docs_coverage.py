# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Fail the docs build if a public name has no API reference entry."""

import importlib
import sys
from collections import defaultdict
from pathlib import Path

from sphinx.util.inventory import InventoryFile

# The public module list lives in the test suite; make it importable when run
# as ``python scripts/check_docs_coverage.py`` from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Report public names missing from the built docs' ``objects.inv``."""
    sys.path.insert(0, str(_PROJECT_ROOT))
    from tests.zephon.test_public_surface import EXPECTED_PUBLIC_MODULES

    build_dir = Path(sys.argv[1])
    with open(build_dir / "objects.inv", "rb") as f:
        inventory = InventoryFile.load(f, "", lambda base, uri: uri)
    documented = {
        name
        for kind, entries in inventory.items()
        if kind.startswith("py:")
        for name in entries
    }

    # One object is often exported from several public paths (zephon.ops.X and
    # zephon.ops.config.X); documenting any of them covers all of them.
    paths_by_object: dict[tuple[int, str], set[str]] = defaultdict(set)
    for module_name in EXPECTED_PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        for name in getattr(module, "__all__", []):
            if not name.startswith("__"):
                paths_by_object[id(getattr(module, name)), name].add(
                    f"{module_name}.{name}"
                )

    missing = sorted(
        min(paths) for paths in paths_by_object.values() if not paths & documented
    )
    for path in missing:
        print(f"undocumented public name: {path}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
