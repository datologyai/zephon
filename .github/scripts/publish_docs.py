"""Publish one Sphinx HTML build into the versioned docs site.

The site lives on the ``gh-pages`` branch and is deployed whole to GitHub Pages.
Each build goes into a directory named for its version:

  main/      the tip of main, republished on every push
  vX.Y.Z/    one directory per release, built from its tag

The site root holds a copy of the newest final release (no ``rc``/``b``/``a``
suffix), so the bare Pages URL, and every URL published before the docs were
versioned, shows the latest release. Until a release has been published, the
root holds ``main``.

Every page gets a ``<script>`` tag for ``_versions/switcher.js``, which reads
``versions.json`` and adds a version picker, plus a banner on any version other
than the latest release. The tag is added here rather than by Sphinx so that
builds of tags predating this script get it too.

Usage:
  publish_docs.py --site SITE_DIR --build HTML_DIR --version NAME

The script is stdlib-only.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path

SWITCHER_SRC = Path(__file__).resolve().parent / "docs_version_switcher.js"
SWITCHER_DIR = "_versions"
VERSIONS_FILE = "versions.json"
# Root entries a refresh of the root copy leaves alone, besides the version
# directories themselves.
RESERVED = frozenset({SWITCHER_DIR, VERSIONS_FILE, ".git", ".nojekyll"})

_RELEASE = re.compile(r"^v(\d+(?:\.\d+)*)(.*)$")
_SCRIPT_TAG = re.compile(r'<script data-zephon-versions="1"[^>]*></script>\n?')


def _is_release(name: str) -> bool:
    return _RELEASE.match(name) is not None


def _is_final(name: str) -> bool:
    match = _RELEASE.match(name)
    return match is not None and not match.group(2)


def _release_key(name: str) -> tuple[tuple[int, ...], int, str]:
    match = _RELEASE.match(name)
    assert match is not None
    numbers = tuple(int(part) for part in match.group(1).split("."))
    suffix = match.group(2)
    # A final release sorts after its own pre-releases: v0.2.0rc1 < v0.2.0.
    return numbers, 0 if suffix else 1, suffix


def _version_dirs(site: Path) -> list[str]:
    """Version directories on the site: ``main`` first, then releases, newest first."""
    names = [
        p.name
        for p in site.iterdir()
        if p.is_dir() and (p.name == "main" or _is_release(p.name))
    ]
    releases = sorted((n for n in names if n != "main"), key=_release_key, reverse=True)
    return (["main"] if "main" in names else []) + releases


def _inject(
    site: Path, tree: Path, version: str, skip: frozenset[str] = frozenset()
) -> None:
    """Point every page under ``tree`` at the switcher, as a page of ``version``.

    Top-level entries of ``tree`` named in ``skip`` are not walked. Any tag a
    previous publish added is replaced, so republishing is idempotent.
    """
    for page in tree.rglob("*.html"):
        rel_page = page.relative_to(tree)
        if rel_page.parts[0] in skip:
            continue
        text = page.read_text(encoding="utf-8")
        if "</head>" not in text:
            continue
        depth = len(page.relative_to(site).parts) - 1
        src = "../" * depth + f"{SWITCHER_DIR}/switcher.js"
        tag = (
            f'<script data-zephon-versions="1" defer src="{html.escape(src)}" '
            f'data-version="{html.escape(version)}" '
            f'data-page="{html.escape(rel_page.as_posix())}"></script>\n'
        )
        text = _SCRIPT_TAG.sub("", text).replace("</head>", tag + "</head>", 1)
        page.write_text(text, encoding="utf-8")


def _refresh_root(site: Path, source: str, versions: list[str]) -> None:
    """Replace the root copy with the contents of ``site/source``."""
    keep = RESERVED | set(versions)
    for entry in site.iterdir():
        if entry.name in keep:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    for entry in (site / source).iterdir():
        if entry.is_dir():
            shutil.copytree(entry, site / entry.name)
        else:
            shutil.copy2(entry, site / entry.name)


def publish(site: Path, build: Path, version: str) -> dict[str, object]:
    """Copy ``build`` into ``site/version``, then rebuild the root copy and index."""
    if version != "main" and not _is_release(version):
        raise SystemExit(f"version must be 'main' or a vX.Y.Z tag, not {version!r}")
    if not (build / "index.html").is_file():
        raise SystemExit(f"{build} holds no index.html; is it a Sphinx HTML build?")

    site.mkdir(parents=True, exist_ok=True)
    dest = site / version
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        build, dest, ignore=shutil.ignore_patterns(".buildinfo", ".doctrees")
    )
    _inject(site, dest, version)

    versions = _version_dirs(site)
    finals = [v for v in versions if _is_final(v)]
    latest = finals[0] if finals else "main"

    # Rebuilt on every publish: it is cheap, and the root then never lags a
    # republished version.
    _refresh_root(site, latest, versions)
    _inject(site, site, latest, skip=RESERVED | set(versions))

    (site / SWITCHER_DIR).mkdir(exist_ok=True)
    shutil.copy2(SWITCHER_SRC, site / SWITCHER_DIR / "switcher.js")
    # Pages is deployed from an Actions artifact, so Jekyll never runs; this
    # keeps the underscore directories served should the source become a branch.
    (site / ".nojekyll").touch()

    index = {"latest": latest, "versions": versions}
    (site / VERSIONS_FILE).write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    return index


def main() -> None:
    """Parse arguments and publish."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--site", type=Path, required=True, help="gh-pages checkout")
    parser.add_argument(
        "--build", type=Path, required=True, help="Sphinx HTML output directory"
    )
    parser.add_argument(
        "--version", required=True, help="'main' or a release tag such as v0.2.1"
    )
    args = parser.parse_args()
    index = publish(args.site, args.build, args.version)
    print(
        f"Published {args.version}; latest release is {index['latest']}; versions: {index['versions']}"
    )


if __name__ == "__main__":
    main()
