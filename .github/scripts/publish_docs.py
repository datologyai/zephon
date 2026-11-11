"""Publish one Sphinx HTML build into the versioned docs site.

The site lives on the ``gh-pages`` branch and is deployed whole to GitHub Pages.
Each build goes into a directory named for its version:

  main/      the tip of main, republished on every push
  vX.Y.Z/    one directory per release, built from its tag

The site root holds a copy of the newest final release (no ``rc``/``b``/``a``
suffix), so the bare Pages URL, and every URL published before the docs were
versioned, shows the latest release. Until a final release has been published,
the root holds ``main``, or the newest pre-release if there is no ``main``.

A release directory is named for its tag, which must match ``vN(.N)*`` with an
optional ``aN``, ``bN`` or ``rcN`` suffix (``v0.2.1``, ``v0.3.0rc1``). The docs
workflow enforces the same grammar.

Every page gets a ``<script>`` tag for ``_versions/switcher.js``, which reads
``versions.json`` and adds a version picker, plus a banner on any version other
than the latest release. The tag is added here rather than by Sphinx so that
builds of tags predating this script get it too.

Titles, descriptions, canonical URLs, and social previews are refreshed in every
published version on each publish. This also updates the stable root immediately,
without rebuilding or modifying historical release tags.

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
from html.parser import HTMLParser
from pathlib import Path
from textwrap import shorten
from urllib.parse import quote, urljoin

SWITCHER_SRC = Path(__file__).resolve().parent / "docs_version_switcher.js"
SWITCHER_DIR = "_versions"
VERSIONS_FILE = "versions.json"
DOCS_URL = "https://docs.zephon.io/"
HOME_TITLE = (
    "Zephon Docs | Elastically Deterministic Data Loading for Foundation Model Training"
)
HOME_DESCRIPTION = (
    "Build fast, reproducible data pipelines for foundation model training with Zephon. "
    "Keep global batches consistent when resuming on a different GPU count."
)
SOCIAL_IMAGE_SRC = (
    Path(__file__).resolve().parents[2] / "docs/source/_static/og-image.jpg"
)
SOCIAL_IMAGE_URL = DOCS_URL + "_static/og-image.jpg"
# Root entries a refresh of the root copy leaves alone, besides the version
# directories themselves.
RESERVED = frozenset({SWITCHER_DIR, VERSIONS_FILE, ".git", ".nojekyll"})

# Anchored with \Z, not $, which would also accept a trailing newline.
_RELEASE = re.compile(r"\Av(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?\Z")
_PRE_RANK = {"a": 0, "b": 1, "rc": 2}
_SCRIPT_TAG = re.compile(r'<script data-zephon-versions="1"[^>]*></script>\n?')
_HEAD = re.compile(r"(<head\b[^>]*>)(.*?)(</head\s*>)", re.IGNORECASE | re.DOTALL)
_AUXILIARY_DIRS = frozenset({"_static", "_sources", "_modules"})


class _HeadMetadata(HTMLParser):
    """Read metadata and locate managed tags without reserializing other HTML."""

    def __init__(self, source: str) -> None:
        super().__init__()
        self.source = source
        self.title = ""
        self.description = ""
        self.canonical = ""
        self.redirect = False
        self.redirect_target = ""
        self._title_start: int | None = None
        self._remove: list[tuple[int, int]] = []
        self.feed(source)

    def _offset(self) -> int:
        line, column = self.getpos()
        return (
            sum(len(s) for s in self.source.splitlines(keepends=True)[: line - 1])
            + column
        )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        managed = False
        if tag == "title":
            self._title_start = self._offset()
        elif tag == "meta":
            key = (values.get("name") or values.get("property") or "").lower()
            managed = key == "description" or key.startswith(("og:", "twitter:"))
            if key == "description":
                self.description = values.get("content") or ""
            if (values.get("http-equiv") or "").lower() == "refresh":
                self.redirect = True
                target = re.search(
                    r";\s*url\s*=\s*(.+)$", values.get("content") or "", re.IGNORECASE
                )
                if target:
                    self.redirect_target = target[1].strip().strip("\"'")
        elif tag == "link" and "canonical" in (values.get("rel") or "").lower().split():
            self.canonical = values.get("href") or ""
            managed = True
        if managed:
            start = self._offset()
            self._remove.append((start, start + len(self.get_starttag_text())))

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._title_start is not None:
            end = self.source.index(">", self._offset()) + 1
            self._remove.append((self._title_start, end))
            self._title_start = None

    def handle_data(self, data: str) -> None:
        if self._title_start is not None:
            self.title += data

    def without_metadata(self) -> str:
        result = self.source
        for start, end in sorted(self._remove, reverse=True):
            result = result[:start] + result[end:]
        return result.rstrip()


class _MainContent(HTMLParser):
    """Extract a page's first heading and paragraph from the Sphinx article."""

    def __init__(self, source: str) -> None:
        super().__init__()
        self.heading = ""
        self.paragraph = ""
        self._article = False
        self._capture = ""
        self.feed(source)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "article":
            self._article = True
        classes = (dict(attrs).get("class") or "").split()
        if self._article and (
            (tag == "h1" and not self.heading)
            or (tag == "p" and not self.paragraph and "admonition-title" not in classes)
        ):
            self._capture = tag

    def handle_endtag(self, tag: str) -> None:
        if tag == self._capture:
            self._capture = ""
        if tag == "article":
            self._article = False

    def handle_data(self, data: str) -> None:
        if self._capture == "h1":
            self.heading += data
        elif self._capture == "p":
            self.paragraph += data


def _metadata(text: str, rel_page: Path, prefix: str) -> str:
    """Replace metadata, keeping content and legacy redirects intact."""
    match = _HEAD.search(text)
    if match is None:
        return text
    head = _HeadMetadata(match[2])
    content = _MainContent(text)
    canonical = urljoin(DOCS_URL + prefix, quote(rel_page.as_posix()))
    # Redirect stubs retain their target, relative to the version they belong to.
    if head.redirect:
        # Use the refresh target, which remains relative after republishing.
        # An old absolute canonical could point at a former stable root copy.
        canonical = urljoin(canonical, head.redirect_target or head.canonical)
    if canonical.endswith("/index.html"):
        canonical = canonical.removesuffix("index.html")
    if head.redirect:
        title = head.title
    elif rel_page.as_posix() == "index.html":
        title = HOME_TITLE
    else:
        heading = " ".join(content.heading.split()).rstrip("¶ ")
        heading = heading or head.title.removesuffix(" - Zephon").removesuffix(
            " | Zephon Docs"
        )
        title = f"{heading} | Zephon Docs"
    tags = [
        f"<title>{html.escape(title)}</title>",
        f'<link rel="canonical" href="{html.escape(canonical, quote=True)}">',
    ]
    if not head.redirect:
        if rel_page.as_posix() == "index.html":
            description = HOME_DESCRIPTION
        else:
            description = head.description or shorten(
                content.paragraph, width=160, placeholder="…"
            )
            description = description or title
        metadata = {
            "description": description,
            "og:title": title,
            "og:description": description,
            "og:type": "website",
            "og:site_name": "Zephon Docs",
            "og:url": canonical,
            "og:image": SOCIAL_IMAGE_URL,
            "og:image:type": "image/jpeg",
            "og:image:width": "1200",
            "og:image:height": "630",
            "og:image:alt": "Zephon, open-source data loader. Same batches. Any GPU count.",
            "twitter:card": "summary_large_image",
            "twitter:title": title,
            "twitter:description": description,
            "twitter:image": SOCIAL_IMAGE_URL,
            "twitter:image:alt": "Zephon, open-source data loader. Same batches. Any GPU count.",
        }
        for key, value in metadata.items():
            attr = "property" if key.startswith("og:") else "name"
            tags.append(
                f'<meta {attr}="{key}" content="{html.escape(value, quote=True)}">'
            )
    updated_head = head.without_metadata() + "\n" + "\n".join(tags) + "\n"
    return text[: match.start(2)] + updated_head + text[match.end(2) :]


def _is_release(name: str) -> bool:
    return _RELEASE.match(name) is not None


def _is_final(name: str) -> bool:
    match = _RELEASE.match(name)
    return match is not None and match.group(2) is None


def _release_key(name: str) -> tuple[tuple[int, ...], int, int, int]:
    match = _RELEASE.match(name)
    assert match is not None
    numbers = tuple(int(part) for part in match.group(1).split("."))
    if match.group(2) is None:
        # A final release sorts after its own pre-releases: v0.2.0rc1 < v0.2.0.
        return numbers, 1, 0, 0
    return numbers, 0, _PRE_RANK[match.group(2)], int(match.group(3))


def _ordered(names: set[str]) -> list[str]:
    """``main`` first, then releases, newest first."""
    releases = sorted((n for n in names if n != "main"), key=_release_key, reverse=True)
    return (["main"] if "main" in names else []) + releases


def _version_dirs(site: Path) -> set[str]:
    return {
        p.name
        for p in site.iterdir()
        if p.is_dir() and (p.name == "main" or _is_release(p.name))
    }


def _inject(
    site: Path,
    tree: Path,
    version: str,
    latest: str,
    skip: frozenset[str] = frozenset(),
) -> None:
    """Refresh metadata and the version switcher on documentation pages.

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
        # The latest release also lives at the root; older releases and main
        # have distinct content and retain their own versioned canonicals.
        prefix = "" if tree == site or version == latest else f"{version}/"
        text = _SCRIPT_TAG.sub("", text)
        # Auxiliary HTML keeps the version picker/banner, but not SEO metadata.
        if rel_page.parts[0] not in _AUXILIARY_DIRS:
            text = _metadata(text, rel_page, prefix)
        text = text.replace("</head>", tag + "</head>", 1)
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
        raise SystemExit(
            f"version must be 'main' or a release tag such as v0.2.1 or v0.3.0rc1, "
            f"not {version!r}"
        )
    if not (build / "index.html").is_file():
        raise SystemExit(f"{build} holds no index.html; is it a Sphinx HTML build?")

    site.mkdir(parents=True, exist_ok=True)
    # Chosen before anything is written; the root source is always a version
    # that exists once this one is in place.
    versions = _ordered(_version_dirs(site) | {version})
    finals = [v for v in versions if _is_final(v)]
    latest = finals[0] if finals else versions[0]

    dest = site / version
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        build, dest, ignore=shutil.ignore_patterns(".buildinfo", ".doctrees")
    )
    # Update every existing release too, including HTML built from old tags.
    for name in versions:
        _inject(site, site / name, name, latest)

    # Rebuilt on every publish: it is cheap, and the root then never lags a
    # republished version.
    _refresh_root(site, latest, versions)
    _inject(site, site, latest, latest, skip=RESERVED | set(versions))

    image = site / "_static" / "og-image.jpg"
    image.parent.mkdir(exist_ok=True)
    shutil.copy2(SOCIAL_IMAGE_SRC, image)

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
