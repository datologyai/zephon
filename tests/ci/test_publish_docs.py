import importlib.util
import json
import shutil
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / ".github" / "scripts" / "publish_docs.py"
_spec = importlib.util.spec_from_file_location("publish_docs", _SCRIPT)
assert _spec and _spec.loader
pd = importlib.util.module_from_spec(_spec)
sys.modules["publish_docs"] = pd
_spec.loader.exec_module(pd)


def _build(tmp_path: Path, name: str, *pages: str) -> Path:
    """A minimal stand-in for a Sphinx HTML build, its pages marked with ``name``."""
    build = tmp_path / "builds" / name
    for page in ("index.html", *pages):
        path = build / page
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"<html><head></head><body>{name} {page}</body></html>")
    return build


def _publish(tmp_path: Path, version: str, *pages: str) -> dict[str, object]:
    return pd.publish(tmp_path / "site", _build(tmp_path, version, *pages), version)


def _versions(tmp_path: Path) -> dict[str, object]:
    return json.loads((tmp_path / "site" / "versions.json").read_text())


def test_root_tracks_newest_final_release(tmp_path: Path) -> None:
    _publish(tmp_path, "main")
    assert _versions(tmp_path)["latest"] == "main"
    _publish(tmp_path, "v0.2.0", "old.html")
    _publish(tmp_path, "v0.2.1")
    _publish(tmp_path, "v0.3.0rc1")

    site = tmp_path / "site"
    assert _versions(tmp_path) == {
        "latest": "v0.2.1",
        "versions": ["main", "v0.3.0rc1", "v0.2.1", "v0.2.0"],
    }
    assert "v0.2.1 index.html" in (site / "index.html").read_text()
    # Pages from an earlier root copy don't linger.
    assert not (site / "old.html").exists()
    assert (site / "v0.2.0" / "old.html").exists()


def test_prerelease_numbers_sort_numerically(tmp_path: Path) -> None:
    for version in ("v0.3.0rc9", "v0.3.0rc10", "v0.3.0b2", "v0.3.0a11", "v0.3.0"):
        _publish(tmp_path, version)
    assert _versions(tmp_path)["versions"] == [
        "v0.3.0",
        "v0.3.0rc10",
        "v0.3.0rc9",
        "v0.3.0b2",
        "v0.3.0a11",
    ]


def test_prerelease_into_empty_site_becomes_root(tmp_path: Path) -> None:
    _publish(tmp_path, "v0.3.0rc1")
    assert _versions(tmp_path) == {"latest": "v0.3.0rc1", "versions": ["v0.3.0rc1"]}
    assert "v0.3.0rc1" in (tmp_path / "site" / "index.html").read_text()


@pytest.mark.parametrize(
    "version",
    [
        "v1junk",
        "v1/../../outside",
        "v0.3.0-rc.1",
        "v0.3.0rc",
        "v0.3.0\n",
        "0.3.0",
        "latest",
        "_versions",
    ],
)
def test_rejects_names_outside_the_tag_grammar(tmp_path: Path, version: str) -> None:
    build = _build(tmp_path, "x")
    with pytest.raises(SystemExit):
        pd.publish(tmp_path / "site", build, version)
    assert not (tmp_path / "site").exists()
    assert not (tmp_path / "outside").exists()


def test_switcher_tag_points_at_site_root(tmp_path: Path) -> None:
    _publish(tmp_path, "main", "api/ops.html")
    _publish(tmp_path, "v0.2.1", "api/ops.html")
    site = tmp_path / "site"

    def tag(page: Path) -> str:
        return pd._SCRIPT_TAG.search(page.read_text()).group(0)

    assert (
        'src="_versions/switcher.js" data-version="v0.2.1" data-page="index.html"'
        in tag(site / "index.html")
    )
    assert (
        'src="../../_versions/switcher.js" data-version="main" data-page="api/ops.html"'
        in tag(site / "main" / "api" / "ops.html")
    )
    assert 'src="../_versions/switcher.js" data-version="v0.2.1"' in tag(
        site / "api" / "ops.html"
    )
    assert (site / "_versions" / "switcher.js").is_file()


def test_republishing_changes_nothing(tmp_path: Path) -> None:
    _publish(tmp_path, "main")
    _publish(tmp_path, "v0.2.1", "api/ops.html")
    site = tmp_path / "site"
    before = {p: p.read_bytes() for p in site.rglob("*") if p.is_file()}
    _publish(tmp_path, "v0.2.1", "api/ops.html")
    after = {p: p.read_bytes() for p in site.rglob("*") if p.is_file()}
    assert before == after
    for page in site.rglob("*.html"):
        assert page.read_text().count("data-zephon-versions") == 1


class _Metadata(HTMLParser):
    def __init__(self, page: Path) -> None:
        super().__init__()
        self.values: dict[str, str] = {}
        self._head = False
        self._title = False
        self.feed(page.read_text())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "head":
            self._head = True
        if not self._head:
            return
        values = dict(attrs)
        key = ""
        value = ""
        if tag == "meta":
            key = values.get("name") or values.get("property") or ""
            value = values.get("content") or ""
        elif tag == "link" and values.get("rel") == "canonical":
            key, value = "canonical", values.get("href") or ""
        elif tag == "title":
            self._title = True
            key = "title"
        if key:
            assert key not in self.values, f"Duplicate {key} in page"
            self.values[key] = value

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            self._head = False
        if tag == "title":
            self._title = False

    def handle_data(self, data: str) -> None:
        if self._head and self._title:
            self.values["title"] += data


def test_publish_retrofits_all_existing_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stand in for releases published before metadata support existed.
    site = tmp_path / "site"
    for version in ("v0.1.0", "v0.2.1"):
        shutil.copytree(_build(tmp_path, version), site / version)
    _publish(tmp_path, "main")

    indexes = [site / "index.html"] + [
        site / version / "index.html" for version in ("main", "v0.1.0", "v0.2.1")
    ]
    for page in indexes:
        meta = _Metadata(page).values
        assert meta["title"] == pd.HOME_TITLE
        assert meta["description"] == pd.HOME_DESCRIPTION
        assert meta["og:title"] == meta["twitter:title"] == meta["title"]
        assert (
            meta["og:description"] == meta["twitter:description"] == meta["description"]
        )
        assert meta["og:url"] == meta["canonical"]
        assert meta["og:image"] == meta["twitter:image"] == pd.SOCIAL_IMAGE_URL
        assert meta["twitter:card"] == "summary_large_image"
    assert (
        site / "_static/og-image.jpg"
    ).read_bytes() == pd.SOCIAL_IMAGE_SRC.read_bytes()

    # A later policy change reaches the stable root and untouched releases too.
    monkeypatch.setattr(
        pd, "HOME_DESCRIPTION", "Updated Zephon documentation description."
    )
    _publish(tmp_path, "main")
    assert all(
        _Metadata(page).values["description"] == pd.HOME_DESCRIPTION for page in indexes
    )


def test_canonicals_change_when_a_new_release_becomes_stable(tmp_path: Path) -> None:
    for version in ("v0.1.0", "main", "v0.2.1"):
        _publish(tmp_path, version, "datasets/index.html", "api/pipeline.html")
    site = tmp_path / "site"
    for prefix, canonical_prefix in (
        ("", ""),
        ("v0.2.1/", ""),
        ("v0.1.0/", "v0.1.0/"),
        ("main/", "main/"),
    ):
        for page, canonical_page in (
            ("index.html", ""),
            ("datasets/index.html", "datasets/"),
            ("api/pipeline.html", "api/pipeline.html"),
        ):
            meta = _Metadata(site / prefix / page).values
            assert meta["canonical"] == pd.DOCS_URL + canonical_prefix + canonical_page
            assert meta["og:url"] == meta["canonical"]


def test_page_metadata_uses_article_content_and_escapes_html(tmp_path: Path) -> None:
    build = _build(tmp_path, "main", "guide.html")
    page = build / "guide.html"
    page.write_text(
        """<html><head><title>Old title - Zephon</title>
<meta name="description" content="">
<link href="https://old.example/" rel="canonical">
<meta property="og:title" content="Old preview">
<meta name="custom" content="keep this">
<script>const example = '<meta name="description" content="example">';</script>
</head><body><nav><p>Ignore navigation</p></nav><article role="main">
<h1>Datasets &amp; <code>"mixtures"</code><a class="headerlink">¶</a></h1>
<p>Keep <strong>global batches</strong> stable &amp; reproducible.</p>
</article></body></html>"""
    )
    pd.publish(tmp_path / "site", build, "main")
    output = tmp_path / "site/main/guide.html"
    meta = _Metadata(output).values
    assert meta["title"] == 'Datasets & "mixtures" | Zephon Docs'
    assert meta["description"] == "Keep global batches stable & reproducible."
    assert meta["og:description"] == meta["description"]
    assert meta["custom"] == "keep this"
    assert "<h1>Datasets &amp;" in output.read_text()
    assert 'content="example">\';</script>' in output.read_text()


def test_redirects_keep_their_targets_and_embeds_are_untouched(tmp_path: Path) -> None:
    build = _build(tmp_path, "v0.1.0", "api/old.html", "_static/embeds/demo.html")
    redirect = (
        "<html><head><title>Moved to Pipeline</title>"
        '<link rel="canonical" href="pipeline.html">'
        '<meta http-equiv="refresh" content="0; url=pipeline.html">'
        '<script>location.replace("pipeline.html" + location.hash);</script>'
        "</head><body>Moved</body></html>"
    )
    (build / "api/old.html").write_text(redirect)
    pd.publish(tmp_path / "site", build, "v0.1.0")
    _publish(tmp_path, "v0.2.1")
    site = tmp_path / "site"
    output = site / "v0.1.0/api/old.html"
    meta = _Metadata(output).values
    assert meta["canonical"] == pd.DOCS_URL + "v0.1.0/api/pipeline.html"
    assert meta["title"] == "Moved to Pipeline"
    assert 'content="0; url=pipeline.html"' in output.read_text()
    assert 'location.replace("pipeline.html" + location.hash)' in output.read_text()
    assert (site / "v0.1.0/_static/embeds/demo.html").read_bytes() == (
        build / "_static/embeds/demo.html"
    ).read_bytes()
