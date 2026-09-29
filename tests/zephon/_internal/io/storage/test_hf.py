# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the hf:// storage backend."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import zephon._internal.io.storage.hf as hf_mod
from zephon._internal.io.storage import HFBackend, RouterStorageBackend
from zephon._internal.io.storage._hf_resolve import Resolution
from zephon._internal.io.storage._hf_uri import parse_hf_uri

requests = pytest.importorskip("requests")

# Some test environments (e.g. the free-threaded Python 3.14t CI job) don't
# ship a huggingface_hub wheel; skip the download tests that patch its internals.
requires_hf_hub = pytest.mark.skipif(
    importlib.util.find_spec("huggingface_hub") is None,
    reason="huggingface_hub is not installed in this environment",
)

SOURCE = "1" * 40
CONVERSION = "2" * 40
FROZEN_ORIGINAL = f"hf://org/repo@{SOURCE}~original/default/train"
FROZEN_PARQUET = f"hf://org/repo@{CONVERSION}~parquet/default/train"
_API = "https://huggingface.co/api/datasets/org/repo"
_RESOLVE = "https://huggingface.co/datasets/org/repo/resolve"
_PQ_NAME = (
    "default%2Ftrain%2F0000.parquet"  # conversion file default/train/0000.parquet
)


class _Resp:
    def __init__(
        self,
        status: int = 200,
        *,
        json: Any = None,
        headers: Mapping[str, str] | None = None,
        content: bytes = b"",
        links: Mapping[str, Any] | None = None,
        redirect: bool = False,
    ) -> None:
        self.status_code = status
        self._json = json
        self.headers = dict(headers or {})
        self.content = content
        self.links = dict(links or {})
        self.is_redirect = redirect

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


Route = _Resp | Callable[[dict[str, Any]], _Resp]


class _FakeHTTP:
    """Routes ``requests.Session`` calls by ``(method, url)`` and records them."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Route] = {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def on(self, method: str, url: str, route: Route) -> None:
        self.routes[(method, url)] = route

    def dispatch(self, method: str, url: str, kwargs: dict[str, Any]) -> _Resp:
        self.calls.append((method, url, kwargs))
        route = self.routes.get((method, url))
        if route is None:
            raise AssertionError(f"unexpected request: {method} {url}")
        return route(kwargs) if callable(route) else route

    def urls(self, method: str | None = None) -> list[str]:
        return [url for m, url, _ in self.calls if method in (None, m)]


@pytest.fixture(autouse=True)
def _fresh_memos() -> Iterator[None]:
    for memo in (hf_mod._frozen_roots, hf_mod._listings, hf_mod._sizes):
        memo.clear()
    yield
    for memo in (hf_mod._frozen_roots, hf_mod._listings, hf_mod._sizes):
        memo.clear()


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> _FakeHTTP:
    fake = _FakeHTTP()
    for method in ("get", "post", "head"):
        verb = method.upper()
        monkeypatch.setattr(
            requests.Session,
            method,
            lambda _self, url, _verb=verb, **kwargs: fake.dispatch(_verb, url, kwargs),
        )
    monkeypatch.setattr(HFBackend, "_get_token", lambda self: None)
    return fake


def _stub_resolution(
    monkeypatch: pytest.MonkeyPatch, frozen: str, files: Mapping[str, int]
) -> list[tuple[str | None, bool]]:
    """Make ``resolve`` return ``frozen``/``files``; records ``(fmt, allow_partial)``."""
    seen: list[tuple[str | None, bool]] = []

    def fake(
        hub: Any, parts: Any, fmt: str | None, *, allow_partial: bool
    ) -> Resolution:
        seen.append((fmt, allow_partial))
        return Resolution(parse_hf_uri(frozen), dict(files), "stub")

    monkeypatch.setattr(hf_mod, "resolve", fake)
    return seen


# ---------------------------------------------------------------------------
# HubClient calls
# ---------------------------------------------------------------------------


class TestHubClient:
    def test_commit_for_requests_only_the_sha(self, http: _FakeHTTP) -> None:
        http.on("GET", f"{_API}/revision/v1.0", _Resp(json={"sha": SOURCE}))
        assert HFBackend().commit_for("org/repo", "v1.0") == SOURCE
        assert http.calls[0][2]["params"] == {"expand[]": "sha"}

    @pytest.mark.parametrize(
        ("status", "error"), [(404, FileNotFoundError), (401, PermissionError)]
    )
    def test_commit_for_maps_http_errors(
        self, http: _FakeHTTP, status: int, error: type[Exception]
    ) -> None:
        http.on("GET", f"{_API}/revision/main", _Resp(status))
        with pytest.raises(error):
            HFBackend().commit_for("org/repo", "main")

    def test_conversion_commit(self, http: _FakeHTTP) -> None:
        refs = {
            "branches": [],
            "converts": [{"name": "parquet", "targetCommit": CONVERSION}],
        }
        http.on("GET", f"{_API}/refs", _Resp(json=refs))
        assert HFBackend().conversion_commit("org/repo") == CONVERSION

    def test_conversion_commit_missing(self, http: _FakeHTTP) -> None:
        http.on("GET", f"{_API}/refs", _Resp(json={"branches": [], "converts": []}))
        assert HFBackend().conversion_commit("org/repo") is None

    def test_parquet_export_returns_the_source_revision(self, http: _FakeHTTP) -> None:
        payload = {"parquet_files": []}
        http.on(
            "GET",
            hf_mod._DATASETS_SERVER_PARQUET_URL,
            _Resp(json=payload, headers={"X-Revision": SOURCE}),
        )
        assert HFBackend().parquet_export("org/repo", "default") == (payload, SOURCE)
        assert http.calls[0][2]["params"] == {
            "dataset": "org/repo",
            "config": "default",
        }

    def test_list_tree_follows_pagination(self, http: _FakeHTTP) -> None:
        first = f"{_API}/tree/{CONVERSION}/default"
        second = f"{first}?cursor=abc"
        http.on(
            "GET",
            first,
            _Resp(
                json=[
                    {"type": "file", "path": "default/README.md", "size": 1},
                    {"type": "directory", "path": "default/train-part0"},
                ],
                links={"next": {"url": second}},
            ),
        )
        http.on(
            "GET",
            second,
            _Resp(json=[{"type": "directory", "path": "default/train-part1"}]),
        )
        files, directories = HFBackend().list_tree("org/repo", CONVERSION, "default")
        assert files == {"README.md": 1}
        assert directories == ["train-part0", "train-part1"]

    def test_large_selections_list_the_common_directory_recursively(
        self, http: _FakeHTTP
    ) -> None:
        wanted = {f"data/{i // 60}/{i:04d}.jsonl.zst": i for i in range(101)}
        entries = [
            {"type": "file", "path": path, "size": size}
            for path, size in wanted.items()
        ]
        entries.insert(0, {"type": "directory", "path": "data/0"})
        entries.insert(1, {"type": "file", "path": "data/README.md", "size": 1})
        first = f"{_API}/tree/{SOURCE}/data"
        second = f"{first}?cursor=next"

        def page(kwargs: dict[str, Any]) -> _Resp:
            assert kwargs["params"] == {"recursive": "true", "expand": "false"}
            return _Resp(json=entries[:60], links={"next": {"url": second}})

        http.on("GET", first, page)
        http.on("GET", second, _Resp(json=entries[60:]))

        backend = HFBackend()
        assert backend.file_sizes("org/repo", SOURCE, list(wanted)) == wanted
        assert http.urls() == [first, second]

    def test_tree_listing_stops_once_every_size_is_known(self, http: _FakeHTTP) -> None:
        wanted = {f"data/{i:04d}.parquet": i for i in range(101)}
        first = f"{_API}/tree/{SOURCE}/data"

        def unreachable(kwargs: dict[str, Any]) -> _Resp:
            raise ConnectionError("the next page must not be requested")

        http.on(
            "GET",
            first,
            _Resp(
                json=[
                    {"type": "file", "path": p, "size": s} for p, s in wanted.items()
                ],
                links={"next": {"url": f"{first}?cursor=next"}},
            ),
        )
        http.on("GET", f"{first}?cursor=next", unreachable)

        assert HFBackend().file_sizes("org/repo", SOURCE, list(wanted)) == wanted
        assert http.urls() == [first]

    def test_small_selections_use_one_paths_info_request(self, http: _FakeHTTP) -> None:
        """A small split never lists the (possibly huge) directory it shares."""

        def paths_info(kwargs: dict[str, Any]) -> _Resp:
            paths = kwargs["data"]["paths"]
            if len(paths) > 100:  # the Hub rejects more than 100 paths
                return _Resp(400)
            known = {"data/validation.parquet": 3}
            return _Resp(
                json=[
                    {"type": "file", "path": p, "size": known[p]}
                    for p in paths
                    if p in known
                ]
            )

        http.on("POST", f"{_API}/paths-info/{SOURCE}", paths_info)
        backend = HFBackend()
        assert backend.file_sizes("org/repo", SOURCE, ["data/validation.parquet"]) == {
            "data/validation.parquet": 3
        }
        assert http.urls() == [f"{_API}/paths-info/{SOURCE}"]

        with pytest.raises(FileNotFoundError, match="1 of 2 files missing"):
            backend.file_sizes(
                "org/repo", SOURCE, ["data/validation.parquet", "gone.parquet"]
            )

    def test_empty_selection_needs_no_request(self, http: _FakeHTTP) -> None:
        assert HFBackend().file_sizes("org/repo", SOURCE, []) == {}
        assert http.calls == []


# ---------------------------------------------------------------------------
# canonical_root and listings
# ---------------------------------------------------------------------------


class TestCanonicalRoot:
    def test_frozen_uri_needs_no_requests(self, http: _FakeHTTP) -> None:
        assert HFBackend().canonical_root(FROZEN_ORIGINAL + "/") == FROZEN_ORIGINAL
        assert http.calls == []

    def test_resolves_once_and_memoizes_across_backends(
        self, http: _FakeHTTP, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _stub_resolution(
            monkeypatch, FROZEN_ORIGINAL, {"data/a.parquet": 3, "b c.parquet": 4}
        )
        assert HFBackend().canonical_root("hf://org/repo/train") == FROZEN_ORIGINAL
        assert HFBackend().canonical_root("hf://org/repo/train") == FROZEN_ORIGINAL
        assert len(seen) == 1

        # The listing from resolution is reused: no request, flat unique names.
        assert HFBackend().listdir(FROZEN_ORIGINAL) == [
            "b%20c.parquet",
            "data%2Fa.parquet",
        ]
        assert http.calls == []

    def test_fmt_and_partial_opt_in_reach_resolution(
        self, http: _FakeHTTP, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _stub_resolution(
            monkeypatch, FROZEN_PARQUET, {"default/train/0000.parquet": 1}
        )
        monkeypatch.setenv("ZEPHON_HF_ALLOW_PARTIAL", "1")
        backend = HFBackend()
        backend.canonical_root("hf://org/repo/train", "parquet")
        backend.canonical_root("hf://org/repo/train")  # a different memo key
        assert seen == [("parquet", True), (None, True)]

    def test_overlong_names_are_rejected(
        self, http: _FakeHTTP, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_resolution(monkeypatch, FROZEN_ORIGINAL, {"d/" * 90 + "x.parquet": 1})
        with pytest.raises(ValueError, match="too long to cache"):
            HFBackend().canonical_root("hf://org/repo/train")

    def test_logs_the_resolution(
        self,
        http: _FakeHTTP,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _stub_resolution(monkeypatch, FROZEN_ORIGINAL, {"a.parquet": 1})
        with caplog.at_level(logging.INFO, logger=hf_mod.__name__):
            HFBackend().canonical_root("hf://org/repo/train")
        assert FROZEN_ORIGINAL in caplog.text


class TestListing:
    def test_frozen_listing_without_memo_relists(
        self, http: _FakeHTTP, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            hf_mod, "list_frozen", lambda hub, parts: {"default/train/0000.parquet": 5}
        )
        backend = HFBackend()
        assert backend.listdir(FROZEN_PARQUET) == [_PQ_NAME]
        assert list(backend.walk(FROZEN_PARQUET)) == [(_PQ_NAME, 5)]

    def test_listdir_of_unfrozen_uri_resolves_first(
        self, http: _FakeHTTP, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_resolution(monkeypatch, FROZEN_PARQUET, {"default/train/0000.parquet": 1})
        assert HFBackend().listdir("hf://org/repo@~parquet/train") == [_PQ_NAME]

    def test_listdir_on_file_path_raises(self, http: _FakeHTTP) -> None:
        with pytest.raises(NotADirectoryError):
            HFBackend().listdir(f"{FROZEN_PARQUET}/{_PQ_NAME}")

    def test_walk_on_file_or_missing_split_yields_nothing(
        self, http: _FakeHTTP, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing(hub: Any, parts: Any) -> dict[str, int]:
            raise FileNotFoundError("gone")

        monkeypatch.setattr(hf_mod, "list_frozen", missing)
        backend = HFBackend()
        assert list(backend.walk(f"{FROZEN_PARQUET}/{_PQ_NAME}")) == []
        assert list(backend.walk(FROZEN_PARQUET)) == []
        assert backend.exists(FROZEN_PARQUET) is False

    def test_exists_for_a_listed_split(
        self, http: _FakeHTTP, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(hf_mod, "list_frozen", lambda hub, parts: {})
        assert HFBackend().exists(FROZEN_PARQUET) is True


# ---------------------------------------------------------------------------
# File access under a frozen root
# ---------------------------------------------------------------------------


class TestFileAccess:
    def test_split_path(self) -> None:
        parts, name = HFBackend._split_path(FROZEN_ORIGINAL)
        assert (parts.uri(), name) == (FROZEN_ORIGINAL, None)
        parts, name = HFBackend._split_path(f"{FROZEN_ORIGINAL}/data%2Fa.parquet")
        assert (parts.uri(), name) == (FROZEN_ORIGINAL, "data%2Fa.parquet")
        # Without a frozen parent, a trailing segment is a split, not a file.
        parts, name = HFBackend._split_path("hf://org/repo/plain_text/train")
        assert (parts.config, parts.split, name) == ("plain_text", "train", None)

    def test_uploaded_file_names_decode_to_repo_paths(self, http: _FakeHTTP) -> None:
        url = f"{_RESOLVE}/{SOURCE}/data/a%20b.parquet"
        http.on("HEAD", url, _Resp(302, headers={"X-Linked-Size": "12"}, redirect=True))
        size = HFBackend().stat(f"{FROZEN_ORIGINAL}/data%2Fa%20b.parquet")["size"]
        assert size == 12

    def test_conversion_names_live_in_the_conversion_dir(self, http: _FakeHTTP) -> None:
        url = f"{_RESOLVE}/{CONVERSION}/default/train/0000.parquet"
        http.on("HEAD", url, _Resp(302, headers={"X-Linked-Size": "7"}, redirect=True))
        assert HFBackend().stat(f"{FROZEN_PARQUET}/{_PQ_NAME}")["size"] == 7

    def test_stat_prefers_the_memoized_listing(self, http: _FakeHTTP) -> None:
        hf_mod._listings[FROZEN_PARQUET] = {_PQ_NAME: 99}
        backend = HFBackend()
        assert backend.stat(f"{FROZEN_PARQUET}/{_PQ_NAME}")["size"] == 99
        assert (
            backend.exists(f"{FROZEN_PARQUET}/default%2Ftrain%2F0001.parquet") is False
        )
        assert http.calls == []

    def test_non_lfs_files_report_content_length(self, http: _FakeHTTP) -> None:
        url = f"{_RESOLVE}/{SOURCE}/a.jsonl"
        http.on("HEAD", url, _Resp(200, headers={"Content-Length": "5"}))
        assert HFBackend().stat(f"{FROZEN_ORIGINAL}/a.jsonl")["size"] == 5

    def test_redirect_without_linked_size_is_followed(self, http: _FakeHTTP) -> None:
        url = f"{_RESOLVE}/{SOURCE}/a.jsonl"

        def head(kwargs: dict[str, Any]) -> _Resp:
            if kwargs["allow_redirects"]:
                return _Resp(200, headers={"Content-Length": "6"})
            return _Resp(302, redirect=True)

        http.on("HEAD", url, head)
        assert HFBackend().stat(f"{FROZEN_ORIGINAL}/a.jsonl")["size"] == 6

    def test_head_sizes_are_cached(self, http: _FakeHTTP) -> None:
        url = f"{_RESOLVE}/{SOURCE}/a.jsonl"
        http.on("HEAD", url, _Resp(200, headers={"Content-Length": "5"}))
        backend = HFBackend()
        backend.exists(f"{FROZEN_ORIGINAL}/a.jsonl")
        backend.stat(f"{FROZEN_ORIGINAL}/a.jsonl")
        assert len(http.urls("HEAD")) == 1

    def test_missing_file(self, http: _FakeHTTP) -> None:
        http.on("HEAD", f"{_RESOLVE}/{SOURCE}/gone.parquet", _Resp(404))
        backend = HFBackend()
        assert backend.exists(f"{FROZEN_ORIGINAL}/gone.parquet") is False
        with pytest.raises(FileNotFoundError):
            backend.stat(f"{FROZEN_ORIGINAL}/gone.parquet")

    def test_non_data_names_are_absent_without_requests(self, http: _FakeHTTP) -> None:
        backend = HFBackend()
        assert backend.exists(f"{FROZEN_ORIGINAL}/index.json") is False
        with pytest.raises(FileNotFoundError):
            backend.stat(f"{FROZEN_ORIGINAL}/_index.json")
        assert http.calls == []

    def test_stat_on_directory_raises(self, http: _FakeHTTP) -> None:
        with pytest.raises(IsADirectoryError):
            HFBackend().stat(FROZEN_ORIGINAL)

    def test_read_range_issues_an_identity_range_request(self, http: _FakeHTTP) -> None:
        url = f"{_RESOLVE}/{CONVERSION}/default/train/0000.parquet"
        http.on("GET", url, _Resp(206, content=b"hello"))
        data = HFBackend().read_range(f"{FROZEN_PARQUET}/{_PQ_NAME}", 10, length=10)
        headers = http.calls[0][2]["headers"]
        assert data == b"hello"
        assert headers["Range"] == "bytes=10-19"
        assert headers["Accept-Encoding"] == "identity"

    def test_read_range_to_end_uses_the_size(self, http: _FakeHTTP) -> None:
        hf_mod._listings[FROZEN_PARQUET] = {_PQ_NAME: 100}
        url = f"{_RESOLVE}/{CONVERSION}/default/train/0000.parquet"
        http.on("GET", url, _Resp(206, content=b"tail"))
        assert HFBackend().read_range(f"{FROZEN_PARQUET}/{_PQ_NAME}", 90) == b"tail"
        assert http.calls[0][2]["headers"]["Range"] == "bytes=90-99"

    def test_read_range_slices_when_server_ignores_range(self, http: _FakeHTTP) -> None:
        url = f"{_RESOLVE}/{CONVERSION}/default/train/0000.parquet"
        http.on("GET", url, _Resp(200, content=bytes(range(100))))
        data = HFBackend().read_range(f"{FROZEN_PARQUET}/{_PQ_NAME}", 10, length=5)
        assert data == bytes(range(10, 15))

    @pytest.mark.parametrize(
        ("status", "error"), [(404, FileNotFoundError), (403, PermissionError)]
    )
    def test_read_range_maps_http_errors(
        self, http: _FakeHTTP, status: int, error: type[Exception]
    ) -> None:
        url = f"{_RESOLVE}/{CONVERSION}/default/train/0000.parquet"
        http.on("GET", url, _Resp(status))
        with pytest.raises(error):
            HFBackend().read_range(f"{FROZEN_PARQUET}/{_PQ_NAME}", 0, length=10)


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


@requires_hf_hub
class TestDownload:
    _NAME = "data%2Fa.parquet"
    _URL = f"{_RESOLVE}/{SOURCE}/data/a.parquet"

    @pytest.fixture(autouse=True)
    def _sized(self, http: _FakeHTTP) -> None:
        http.on(
            "HEAD",
            self._URL,
            _Resp(302, headers={"X-Linked-Size": "11"}, redirect=True),
        )

    def test_worker_download_needs_no_listing(
        self, http: _FakeHTTP, tmp_path: Path
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_http_get(url: str, temp_file: Any, **kwargs: Any) -> None:
            seen["url"] = url
            temp_file.write(b"hello world")

        dst = tmp_path / "out.parquet"
        with patch("huggingface_hub.file_download.http_get", side_effect=fake_http_get):
            HFBackend().download(f"{FROZEN_ORIGINAL}/{self._NAME}", str(dst))

        assert dst.read_bytes() == b"hello world"
        assert seen["url"] == self._URL
        assert http.urls() == [self._URL]  # one HEAD, no Hub API call
        assert not (tmp_path / "out.parquet.incomplete").exists()

    def test_resumes_from_existing_incomplete(self, tmp_path: Path) -> None:
        dst = tmp_path / "out.parquet"
        (tmp_path / "out.parquet.incomplete").write_bytes(b"hello ")
        seen: dict[str, int] = {}

        def fake_http_get(
            url: str, temp_file: Any, *, resume_size: int = 0, **kwargs: Any
        ) -> None:
            seen["resume_size"] = resume_size
            temp_file.write(b"world")

        with patch("huggingface_hub.file_download.http_get", side_effect=fake_http_get):
            HFBackend().download(f"{FROZEN_ORIGINAL}/{self._NAME}", str(dst))

        assert seen["resume_size"] == 6
        assert dst.read_bytes() == b"hello world"

    def test_promotes_already_complete_incomplete(self, tmp_path: Path) -> None:
        dst = tmp_path / "out.parquet"
        (tmp_path / "out.parquet.incomplete").write_bytes(b"hello world")
        http_get = MagicMock()
        with patch("huggingface_hub.file_download.http_get", new=http_get):
            HFBackend().download(f"{FROZEN_ORIGINAL}/{self._NAME}", str(dst))
        assert dst.read_bytes() == b"hello world"
        assert http_get.call_count == 0

    def test_404_removes_incomplete(self, tmp_path: Path) -> None:
        from huggingface_hub.utils import EntryNotFoundError

        def fake_http_get(*args: Any, **kwargs: Any) -> None:
            kwargs["temp_file"].write(b"partial")
            raise EntryNotFoundError("not found")

        dst = tmp_path / "out.parquet"
        with patch("huggingface_hub.file_download.http_get", side_effect=fake_http_get):
            with pytest.raises(FileNotFoundError):
                HFBackend().download(f"{FROZEN_ORIGINAL}/{self._NAME}", str(dst))
        assert not (tmp_path / "out.parquet.incomplete").exists()

    def test_403_maps_to_permission_error(self, tmp_path: Path) -> None:
        from huggingface_hub.utils import HfHubHTTPError

        forbidden = MagicMock()
        forbidden.status_code = 403

        def fake_http_get(*args: Any, **kwargs: Any) -> None:
            raise HfHubHTTPError("forbidden", response=forbidden)

        with patch("huggingface_hub.file_download.http_get", side_effect=fake_http_get):
            with pytest.raises(PermissionError):
                HFBackend().download(
                    f"{FROZEN_ORIGINAL}/{self._NAME}", str(tmp_path / "o")
                )

    def test_transient_failure_keeps_incomplete(self, tmp_path: Path) -> None:
        from huggingface_hub.utils import HfHubHTTPError

        bad_gateway = MagicMock()
        bad_gateway.status_code = 502

        def fake_http_get(*args: Any, **kwargs: Any) -> None:
            kwargs["temp_file"].write(b"partial")
            raise HfHubHTTPError("bad gateway", response=bad_gateway)

        dst = tmp_path / "out.parquet"
        with patch("huggingface_hub.file_download.http_get", side_effect=fake_http_get):
            with pytest.raises(HfHubHTTPError):
                HFBackend().download(f"{FROZEN_ORIGINAL}/{self._NAME}", str(dst))
        assert (tmp_path / "out.parquet.incomplete").read_bytes() == b"partial"

    def test_missing_source_raises_before_downloading(
        self, http: _FakeHTTP, tmp_path: Path
    ) -> None:
        http.on("HEAD", self._URL, _Resp(404))
        http_get = MagicMock()
        with patch("huggingface_hub.file_download.http_get", new=http_get):
            with pytest.raises(FileNotFoundError):
                HFBackend().download(
                    f"{FROZEN_ORIGINAL}/{self._NAME}", str(tmp_path / "o")
                )
        assert http_get.call_count == 0


# ---------------------------------------------------------------------------
# Session, read-only ops, routing
# ---------------------------------------------------------------------------


def test_session_retries_transient_errors_including_paths_info() -> None:
    retry = (
        HFBackend()._get_session().get_adapter("https://huggingface.co/").max_retries
    )
    assert {429, 500, 502, 503, 504} <= set(retry.status_forcelist)
    assert {"GET", "HEAD", "POST"} <= set(retry.allowed_methods)
    assert retry.total >= 1


def test_write_operations_and_globs_are_unsupported() -> None:
    backend = HFBackend()
    with pytest.raises(NotImplementedError):
        backend.put(f"{FROZEN_ORIGINAL}/x.parquet", b"")
    with pytest.raises(NotImplementedError):
        backend.delete(f"{FROZEN_ORIGINAL}/x.parquet")
    with pytest.raises(NotImplementedError):
        backend.mkdir(FROZEN_ORIGINAL)
    with pytest.raises(NotImplementedError):
        backend.glob("hf://org/repo/*")


class TestRouterRoutesHFScheme:
    def test_router_selects_and_caches_hf_backend(self) -> None:
        router = RouterStorageBackend()
        backend = router._backend_for("hf://org/repo/train")
        assert isinstance(backend, HFBackend)
        assert router._backend_for("hf://other/repo/test") is backend
        assert router.is_cloud_path("hf://org/repo/train") is True

    def test_router_canonical_root_delegates_to_hf(
        self, http: _FakeHTTP, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _stub_resolution(monkeypatch, FROZEN_ORIGINAL, {"a.parquet": 1})
        router = RouterStorageBackend()
        assert (
            router.canonical_root("hf://org/repo/train", fmt="parquet")
            == FROZEN_ORIGINAL
        )
        assert seen == [("parquet", False)]


# ---------------------------------------------------------------------------
# Memo keys, concurrency and an end-to-end read
# ---------------------------------------------------------------------------


def test_equivalent_uris_share_one_resolution(
    http: _FakeHTTP, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _stub_resolution(monkeypatch, FROZEN_ORIGINAL, {"a.parquet": 1})
    backend = HFBackend()
    assert backend.canonical_root("hf://org/repo/train") == FROZEN_ORIGINAL
    assert backend.canonical_root("hf://org/repo@main/train/") == FROZEN_ORIGINAL
    assert len(seen) == 1


def test_partial_opt_in_is_part_of_the_memo_key(
    http: _FakeHTTP, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _stub_resolution(monkeypatch, FROZEN_ORIGINAL, {"a.parquet": 1})
    monkeypatch.setenv("ZEPHON_HF_ALLOW_PARTIAL", "1")
    HFBackend().canonical_root("hf://org/repo/train")
    monkeypatch.delenv("ZEPHON_HF_ALLOW_PARTIAL")
    HFBackend().canonical_root("hf://org/repo/train")
    assert seen == [(None, True), (None, False)]


_WAIT_SECONDS = 5.0


def _run_racing_callers(
    monkeypatch: pytest.MonkeyPatch, release: Any, calls: list[Callable[[], Any]]
) -> list[Any]:
    """Run ``calls`` on threads while the first same-key miss is held open.

    The first caller that enters the stubbed slow path blocks on ``release``;
    the test releases it only after every other caller has asked for the same
    key's lock, so a missing per-key lock deterministically duplicates work.
    """
    import threading

    requested = threading.Semaphore(0)
    real_key_lock = hf_mod._key_lock

    def observed_key_lock(key: Any) -> Any:
        requested.release()
        return real_key_lock(key)

    monkeypatch.setattr(hf_mod, "_key_lock", observed_key_lock)
    results: list[Any] = [None] * len(calls)
    errors: list[BaseException] = []

    def run(index: int, call: Callable[[], Any]) -> None:
        try:
            results[index] = call()
        except BaseException as exc:  # surfaced on the test thread below
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(i, call), daemon=True)
        for i, call in enumerate(calls)
    ]
    for thread in threads:
        thread.start()
    for _ in calls:  # every caller reached the same-key lock
        assert requested.acquire(timeout=_WAIT_SECONDS)
    release.set()
    for thread in threads:
        thread.join(timeout=_WAIT_SECONDS)
        assert not thread.is_alive(), "caller deadlocked"
    if errors:
        raise errors[0]
    return results


def test_concurrent_resolutions_of_one_uri_share_the_work(
    http: _FakeHTTP, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    release = threading.Event()
    resolutions: list[str] = []

    def held_resolve(
        hub: Any, parts: Any, fmt: Any, *, allow_partial: bool
    ) -> Resolution:
        resolutions.append(parts.uri())
        if len(resolutions) == 1:
            assert release.wait(_WAIT_SECONDS)
        return Resolution(parse_hf_uri(FROZEN_ORIGINAL), {"a.parquet": 1}, "stub")

    monkeypatch.setattr(hf_mod, "resolve", held_resolve)
    results = _run_racing_callers(
        monkeypatch,
        release,
        [
            lambda: HFBackend().canonical_root("hf://org/repo/train"),
            lambda: HFBackend().canonical_root("hf://org/repo@main/train/"),
        ],
    )
    assert resolutions == ["hf://org/repo@main/train"]
    assert results == [FROZEN_ORIGINAL, FROZEN_ORIGINAL]


def test_concurrent_listings_of_one_root_share_the_work(
    http: _FakeHTTP, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    release = threading.Event()
    listings: list[str] = []

    def held_list(hub: Any, parts: Any) -> dict[str, int]:
        listings.append(parts.uri())
        if len(listings) == 1:
            assert release.wait(_WAIT_SECONDS)
        return {"default/train/0000.parquet": 1}

    monkeypatch.setattr(hf_mod, "list_frozen", held_list)
    results = _run_racing_callers(
        monkeypatch,
        release,
        [lambda: HFBackend().listdir(FROZEN_PARQUET) for _ in range(2)],
    )
    assert listings == [FROZEN_PARQUET]
    assert results == [[_PQ_NAME], [_PQ_NAME]]


@requires_hf_hub
def test_uploaded_compressed_jsonl_end_to_end(
    http: _FakeHTTP, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolution, discovery, the shard cache and JSONL decoding, offline."""
    import gzip
    import json

    import zephon._internal.io.storage._hf_resolve as resolve_mod
    from zephon._internal.io.storage._hf_resolve import _Uploaded
    from zephon._internal.io.stores.multi import build_multi_dataset_store
    from zephon.io.dataset import Dataset
    from zephon.io.options import CacheOptions, StoreOptions

    rows = [{"i": i, "text": f"row {i}"} for i in range(3)]
    blob = gzip.compress(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
    path = "data/train.jsonl.gz"
    http.on("GET", f"{_API}/revision/main", _Resp(json={"sha": SOURCE}))
    http.on(
        "POST",
        f"{_API}/paths-info/{SOURCE}",
        _Resp(json=[{"type": "file", "path": path, "size": len(blob)}]),
    )
    monkeypatch.setattr(
        resolve_mod,
        "_uploaded_files",
        lambda repo_id, commit, config, split: _Uploaded("default", (path,), None),
    )
    downloads: list[str] = []

    def fake_http_get(url: str, temp_file: Any, **kwargs: Any) -> None:
        downloads.append(url)
        temp_file.write(blob)

    with patch("huggingface_hub.file_download.http_get", side_effect=fake_http_get):
        dataset = Dataset.from_path("hf-jsonl", "hf://org/repo/train")
        assert dataset.path == FROZEN_ORIGINAL
        assert dataset.backend["kind"] == "jsonl"
        assert dataset.total() == 3

        store = build_multi_dataset_store(
            {0: dataset},
            options=StoreOptions(
                cache=CacheOptions(enabled=True, root=tmp_path / "cache")
            ),
        )
        try:
            shard, _ = store.for_dataset(0).open(0)
            assert shard.getsamples([2, 0])[0] == [rows[2], rows[0]]
            fetched = len(downloads)
            again, _ = store.for_dataset(0).open(0)
            assert again.getsamples([1])[0] == [rows[1]]
            assert len(downloads) == fetched  # served from the shard cache
        finally:
            store.close()

    assert set(downloads) == {f"{_RESOLVE}/{SOURCE}/data/train.jsonl.gz"}
