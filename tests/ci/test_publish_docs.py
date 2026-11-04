import importlib.util
import json
import sys
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
