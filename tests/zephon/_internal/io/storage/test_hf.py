# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the HuggingFace URI grammar and the hf:// storage backend."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pytest

from zephon._internal.io.storage import HFBackend, RouterStorageBackend
from zephon._internal.io.storage._hf_uri import HFUriParts, parse_hf_uri

_DATASETS_SERVER_HOST = "datasets-server.huggingface.co"

# Some test environments (e.g. the free-threaded Python 3.14t CI job) don't
# ship a huggingface_hub wheel. The unit tests that mock its internals would
# fail at ``patch`` time with ModuleNotFoundError; skip those cleanly while
# leaving the pure-requests tests (URI parsing, listdir, range, router, walk)
# running.
requires_hf_hub = pytest.mark.skipif(
    importlib.util.find_spec("huggingface_hub") is None,
    reason="huggingface_hub is not installed in this environment",
)


def _is_datasets_server_url(url: str) -> bool:
    """Whitelist the dataset-server host exactly (no substring bypass)."""
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.netloc == _DATASETS_SERVER_HOST


class TestParseHFUri:
    """Unit tests for ``parse_hf_uri``."""

    def test_minimal(self) -> None:
        parts = parse_hf_uri("hf://org/repo/train")
        assert parts == HFUriParts(
            repo_id="org/repo", revision="main", config=None, split="train"
        )

    def test_with_revision(self) -> None:
        parts = parse_hf_uri("hf://org/repo@v1.2/train")
        assert parts.revision == "v1.2"
        assert parts.repo_id == "org/repo"

    def test_with_sha_revision(self) -> None:
        sha = "abc123def456"
        parts = parse_hf_uri(f"hf://org/repo@{sha}/train")
        assert parts.revision == sha

    def test_with_config(self) -> None:
        parts = parse_hf_uri("hf://org/repo/plain_text/train")
        assert parts == HFUriParts(
            repo_id="org/repo",
            revision="main",
            config="plain_text",
            split="train",
        )

    def test_full(self) -> None:
        parts = parse_hf_uri("hf://HuggingFaceH4/ultrachat_200k@main/default/train_sft")
        assert parts == HFUriParts(
            repo_id="HuggingFaceH4/ultrachat_200k",
            revision="main",
            config="default",
            split="train_sft",
        )

    def test_trailing_slash_tolerated(self) -> None:
        parts = parse_hf_uri("hf://org/repo/train/")
        assert parts.split == "train"

    @pytest.mark.parametrize(
        "bad_uri",
        [
            "s3://bucket/key",  # wrong scheme
            "hf://",  # empty body
            "hf://org/repo",  # missing split
            "hf://org",  # missing name + split
            "hf://org/repo/a/b/c",  # too many segments
            "hf://org/repo@bad rev/train",  # revision contains whitespace
            "hf://org/repo@/train",  # empty revision
            "hf:///repo/train",  # empty org
            "hf://org//train",  # empty name
        ],
    )
    def test_rejects_malformed(self, bad_uri: str) -> None:
        with pytest.raises(ValueError):
            parse_hf_uri(bad_uri)


class TestHFBackendSplitPath:
    """File/directory disambiguation for ``hf://`` paths."""

    def test_split_only_is_directory(self) -> None:
        parts, filename = HFBackend._split_path("hf://org/repo/train")
        assert filename is None
        assert parts.split == "train"
        assert parts.config is None

    def test_config_split_is_directory(self) -> None:
        parts, filename = HFBackend._split_path("hf://org/repo/plain_text/train")
        assert filename is None
        assert parts.config == "plain_text"
        assert parts.split == "train"

    def test_split_plus_file_is_file(self) -> None:
        parts, filename = HFBackend._split_path(
            "hf://org/repo/train/train-00000-of-00002.parquet"
        )
        assert filename == "train-00000-of-00002.parquet"
        assert parts.config is None
        assert parts.split == "train"

    def test_config_split_plus_file_is_file(self) -> None:
        parts, filename = HFBackend._split_path(
            "hf://org/repo/plain_text/train/0000.parquet"
        )
        assert filename == "0000.parquet"
        assert parts.config == "plain_text"
        assert parts.split == "train"

    def test_revision_carried_through(self) -> None:
        parts, filename = HFBackend._split_path("hf://org/repo@v1/train/0000.parquet")
        assert filename == "0000.parquet"
        assert parts.revision == "v1"

    def test_rejects_non_hf_uri(self) -> None:
        with pytest.raises(ValueError):
            HFBackend._split_path("s3://bucket/key")


def _mock_parquet_response(items: list[dict[str, Any]]) -> MagicMock:
    """Build a fake ``requests.get`` response object for ``/parquet``."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "parquet_files": items,
        "pending": [],
        "failed": [],
        "partial": False,
    }
    resp.raise_for_status = MagicMock()
    return resp


def _shard_item(
    *,
    config: str = "default",
    split: str = "train",
    filename: str = "0000.parquet",
    size: int = 1024,
    url: str | None = None,
) -> dict[str, Any]:
    if url is None:
        # Real /parquet endpoint returns canonical resolve URLs with the
        # auto-converted-parquet ref URL-encoded into the path segment.
        url = (
            f"https://huggingface.co/datasets/org/repo/resolve/"
            f"refs%2Fconvert%2Fparquet/{config}/{split}/{filename}"
        )
    return {
        "dataset": "org/repo",
        "config": config,
        "split": split,
        "url": url,
        "filename": filename,
        "size": size,
    }


class TestHFBackendListing:
    """The ``/parquet`` endpoint drives listdir / stat / read_range."""

    def test_listdir_returns_filenames_for_split(self) -> None:
        backend = HFBackend()
        items = [
            _shard_item(split="train", filename="0000.parquet"),
            _shard_item(split="train", filename="0001.parquet"),
            _shard_item(split="test", filename="test.parquet"),
        ]
        with patch(
            "requests.Session.get", return_value=_mock_parquet_response(items)
        ) as get:
            names = backend.listdir("hf://org/repo/train")

        assert names == ["0000.parquet", "0001.parquet"]
        # One call to /parquet for the whole listing.
        assert get.call_count == 1

    def test_listdir_filters_by_config(self) -> None:
        backend = HFBackend()
        items = [
            _shard_item(config="default", filename="d-0.parquet"),
            _shard_item(config="other", filename="o-0.parquet"),
        ]
        with patch("requests.Session.get", return_value=_mock_parquet_response(items)):
            names = backend.listdir("hf://org/repo/default/train")
        assert names == ["d-0.parquet"]

    def test_listdir_missing_split_raises(self) -> None:
        backend = HFBackend()
        items = [_shard_item(split="train", filename="0000.parquet")]
        with patch("requests.Session.get", return_value=_mock_parquet_response(items)):
            with pytest.raises(FileNotFoundError):
                backend.listdir("hf://org/repo/validation")

    def test_listing_cached_per_repo_revision_config_split(self) -> None:
        backend = HFBackend()
        items = [
            _shard_item(filename="0000.parquet"),
            _shard_item(filename="0001.parquet", size=2048),
        ]
        with patch(
            "requests.Session.get", return_value=_mock_parquet_response(items)
        ) as get:
            backend.listdir("hf://org/repo/train")
            backend.stat("hf://org/repo/train/0000.parquet")
            backend.stat("hf://org/repo/train/0001.parquet")
        assert get.call_count == 1

    def test_stat_reports_size_from_listing(self) -> None:
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", size=4242)]
        with patch("requests.Session.get", return_value=_mock_parquet_response(items)):
            stats = backend.stat("hf://org/repo/train/0000.parquet")
        assert stats["size"] == 4242

    def test_stat_missing_file_raises(self) -> None:
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet")]
        with patch("requests.Session.get", return_value=_mock_parquet_response(items)):
            with pytest.raises(FileNotFoundError):
                backend.stat("hf://org/repo/train/nope.parquet")

    def test_exists_directory_and_file(self) -> None:
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet")]
        with patch("requests.Session.get", return_value=_mock_parquet_response(items)):
            assert backend.exists("hf://org/repo/train") is True
            assert backend.exists("hf://org/repo/train/0000.parquet") is True
            assert backend.exists("hf://org/repo/train/missing.parquet") is False

    def test_exists_returns_false_for_unknown_dataset(self) -> None:
        backend = HFBackend()
        resp = MagicMock()
        resp.status_code = 404
        resp.raise_for_status = MagicMock()
        with patch("requests.Session.get", return_value=resp):
            assert backend.exists("hf://org/missing/train") is False
            assert backend.exists("hf://org/missing/train/index.json") is False

    def test_read_range_issues_http_range(self) -> None:
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", size=1024)]
        listing_resp = _mock_parquet_response(items)

        range_resp = MagicMock()
        range_resp.status_code = 206
        range_resp.content = b"hello"
        range_resp.raise_for_status = MagicMock()

        def fake_get(url: str, **kwargs: Any) -> Any:
            if _is_datasets_server_url(url):
                return listing_resp
            assert "Range" in kwargs.get("headers", {})
            assert kwargs["headers"]["Range"] == "bytes=10-19"
            return range_resp

        with patch("requests.Session.get", side_effect=fake_get):
            data = backend.read_range("hf://org/repo/train/0000.parquet", 10, length=10)
        assert data == b"hello"

    def test_read_range_to_end_uses_size_from_listing(self) -> None:
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", size=100)]
        listing_resp = _mock_parquet_response(items)

        range_resp = MagicMock()
        range_resp.status_code = 206
        range_resp.content = b"tail"
        range_resp.raise_for_status = MagicMock()

        seen: dict[str, str] = {}

        def fake_get(url: str, **kwargs: Any) -> Any:
            if _is_datasets_server_url(url):
                return listing_resp
            seen["range"] = kwargs["headers"]["Range"]
            return range_resp

        with patch("requests.Session.get", side_effect=fake_get):
            data = backend.read_range("hf://org/repo/train/0000.parquet", 90)
        assert data == b"tail"
        assert seen["range"] == "bytes=90-99"

    @requires_hf_hub
    def test_download_streams_to_local_file(self, tmp_path: Path) -> None:
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", size=11)]
        listing_resp = _mock_parquet_response(items)

        def fake_http_get(
            url: str, temp_file: Any, *, resume_size: int = 0, **kwargs: Any
        ) -> None:
            temp_file.write(b"hello world")

        dst = tmp_path / "out.parquet"
        with patch("requests.Session.get", return_value=listing_resp):
            with patch(
                "huggingface_hub.file_download.http_get", side_effect=fake_http_get
            ):
                backend.download("hf://org/repo/train/0000.parquet", str(dst))
        assert dst.read_bytes() == b"hello world"
        assert not (tmp_path / "out.parquet.incomplete").exists()

    def test_walk_yields_filenames_and_sizes(self) -> None:
        backend = HFBackend()
        items = [
            _shard_item(split="train", filename="0000.parquet", size=10),
            _shard_item(split="train", filename="0001.parquet", size=20),
            _shard_item(split="test", filename="test.parquet", size=99),
        ]
        with patch("requests.Session.get", return_value=_mock_parquet_response(items)):
            entries = list(backend.walk("hf://org/repo/train"))
        assert entries == [("0000.parquet", 10), ("0001.parquet", 20)]

    def test_walk_on_file_path_yields_nothing(self) -> None:
        backend = HFBackend()
        assert list(backend.walk("hf://org/repo/train/0000.parquet")) == []

    def test_walk_missing_split_yields_nothing(self) -> None:
        backend = HFBackend()
        items = [_shard_item(split="train", filename="0000.parquet")]
        with patch("requests.Session.get", return_value=_mock_parquet_response(items)):
            assert list(backend.walk("hf://org/repo/validation")) == []

    def test_listdir_on_file_path_raises(self) -> None:
        backend = HFBackend()
        with pytest.raises(NotADirectoryError):
            backend.listdir("hf://org/repo/train/0000.parquet")

    def test_stat_on_directory_raises(self) -> None:
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet")]
        with patch("requests.Session.get", return_value=_mock_parquet_response(items)):
            with pytest.raises(IsADirectoryError):
                backend.stat("hf://org/repo/train")

    def test_write_operations_not_supported(self) -> None:
        backend = HFBackend()
        with pytest.raises(NotImplementedError):
            backend.put("hf://org/repo/train/x.parquet", b"")
        with pytest.raises(NotImplementedError):
            backend.delete("hf://org/repo/train/x.parquet")
        with pytest.raises(NotImplementedError):
            backend.mkdir("hf://org/repo/train")


class TestHFBackendErrorMapping:
    """Per-file fetches must map HTTP errors to protocol exceptions."""

    def test_read_range_maps_404_to_filenotfound(self) -> None:
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", size=1024)]
        listing_resp = _mock_parquet_response(items)

        gone_resp = MagicMock()
        gone_resp.status_code = 404
        gone_resp.raise_for_status = MagicMock(side_effect=AssertionError("unmapped"))

        def fake_get(url: str, **kwargs: Any) -> Any:
            if _is_datasets_server_url(url):
                return listing_resp
            return gone_resp

        with patch("requests.Session.get", side_effect=fake_get):
            with pytest.raises(FileNotFoundError):
                backend.read_range("hf://org/repo/train/0000.parquet", 0, length=10)

    def test_read_range_maps_403_to_permissionerror(self) -> None:
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", size=1024)]
        listing_resp = _mock_parquet_response(items)

        forbidden = MagicMock()
        forbidden.status_code = 403
        forbidden.raise_for_status = MagicMock(side_effect=AssertionError("unmapped"))

        def fake_get(url: str, **kwargs: Any) -> Any:
            if _is_datasets_server_url(url):
                return listing_resp
            return forbidden

        with patch("requests.Session.get", side_effect=fake_get):
            with pytest.raises(PermissionError):
                backend.read_range("hf://org/repo/train/0000.parquet", 0, length=10)

    @requires_hf_hub
    def test_download_maps_404_to_filenotfound(self, tmp_path: Path) -> None:
        from huggingface_hub.utils import EntryNotFoundError

        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", size=1024)]
        listing_resp = _mock_parquet_response(items)

        def fake_http_get(*args: Any, **kwargs: Any) -> None:
            raise EntryNotFoundError("not found")

        with patch("requests.Session.get", return_value=listing_resp):
            with patch(
                "huggingface_hub.file_download.http_get", side_effect=fake_http_get
            ):
                with pytest.raises(FileNotFoundError):
                    backend.download(
                        "hf://org/repo/train/0000.parquet", str(tmp_path / "out")
                    )

    @requires_hf_hub
    def test_download_maps_hub_http_403_to_permissionerror(
        self, tmp_path: Path
    ) -> None:
        from huggingface_hub.utils import HfHubHTTPError

        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", size=1024)]
        listing_resp = _mock_parquet_response(items)

        forbidden_resp = MagicMock()
        forbidden_resp.status_code = 403

        def fake_http_get(*args: Any, **kwargs: Any) -> None:
            err = HfHubHTTPError("forbidden", response=forbidden_resp)
            raise err

        with patch("requests.Session.get", return_value=listing_resp):
            with patch(
                "huggingface_hub.file_download.http_get", side_effect=fake_http_get
            ):
                with pytest.raises(PermissionError):
                    backend.download(
                        "hf://org/repo/train/0000.parquet", str(tmp_path / "out")
                    )


@requires_hf_hub
class TestHFBackendDownloadResume:
    """``.incomplete`` sidecar drives resume across retried calls."""

    def test_resumes_from_existing_incomplete(self, tmp_path: Path) -> None:
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", size=11)]
        listing_resp = _mock_parquet_response(items)

        dst = tmp_path / "out.parquet"
        (tmp_path / "out.parquet.incomplete").write_bytes(b"hello ")

        seen: dict[str, int] = {}

        def fake_http_get(
            url: str, temp_file: Any, *, resume_size: int = 0, **kwargs: Any
        ) -> None:
            seen["resume_size"] = resume_size
            temp_file.write(b"world")

        with patch("requests.Session.get", return_value=listing_resp):
            with patch(
                "huggingface_hub.file_download.http_get", side_effect=fake_http_get
            ):
                backend.download("hf://org/repo/train/0000.parquet", str(dst))

        assert seen["resume_size"] == 6
        assert dst.read_bytes() == b"hello world"
        assert not (tmp_path / "out.parquet.incomplete").exists()

    def test_promotes_already_complete_incomplete(self, tmp_path: Path) -> None:
        """A ``.incomplete`` that's already the expected size is just renamed."""
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", size=11)]
        listing_resp = _mock_parquet_response(items)

        dst = tmp_path / "out.parquet"
        (tmp_path / "out.parquet.incomplete").write_bytes(b"hello world")

        http_get_calls = MagicMock()
        with patch("requests.Session.get", return_value=listing_resp):
            with patch("huggingface_hub.file_download.http_get", new=http_get_calls):
                backend.download("hf://org/repo/train/0000.parquet", str(dst))

        assert dst.read_bytes() == b"hello world"
        assert http_get_calls.call_count == 0


class TestHFBackendRangeIdentityEncoding:
    """``read_range`` must defeat CDN gzip to keep byte offsets meaningful."""

    def test_read_range_sets_accept_encoding_identity(self) -> None:
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", size=1024)]
        listing_resp = _mock_parquet_response(items)

        range_resp = MagicMock()
        range_resp.status_code = 206
        range_resp.content = b"abc"
        range_resp.raise_for_status = MagicMock()

        seen_headers: dict[str, str] = {}

        def fake_get(url: str, **kwargs: Any) -> Any:
            if _is_datasets_server_url(url):
                return listing_resp
            seen_headers.update(kwargs.get("headers") or {})
            return range_resp

        with patch("requests.Session.get", side_effect=fake_get):
            backend.read_range("hf://org/repo/train/0000.parquet", 0, length=3)

        assert seen_headers.get("Accept-Encoding") == "identity"


@requires_hf_hub
class TestHFBackendDownloadSidecarCleanup:
    """The ``.incomplete`` sidecar should not leak on terminal failures."""

    def test_404_removes_incomplete(self, tmp_path: Path) -> None:
        from huggingface_hub.utils import EntryNotFoundError

        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", size=1024)]
        listing_resp = _mock_parquet_response(items)

        def fake_http_get(*args: Any, **kwargs: Any) -> None:
            # Touch the sidecar to simulate partial writes before the error.
            kwargs["temp_file"].write(b"partial")
            raise EntryNotFoundError("not found")

        dst = tmp_path / "out.parquet"
        with patch("requests.Session.get", return_value=listing_resp):
            with patch(
                "huggingface_hub.file_download.http_get", side_effect=fake_http_get
            ):
                with pytest.raises(FileNotFoundError):
                    backend.download("hf://org/repo/train/0000.parquet", str(dst))
        assert not (tmp_path / "out.parquet.incomplete").exists()

    def test_transient_failure_keeps_incomplete(self, tmp_path: Path) -> None:
        from huggingface_hub.utils import HfHubHTTPError

        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", size=1024)]
        listing_resp = _mock_parquet_response(items)

        bad_gateway_resp = MagicMock()
        bad_gateway_resp.status_code = 502

        def fake_http_get(*args: Any, **kwargs: Any) -> None:
            kwargs["temp_file"].write(b"partial")
            raise HfHubHTTPError("bad gateway", response=bad_gateway_resp)

        dst = tmp_path / "out.parquet"
        with patch("requests.Session.get", return_value=listing_resp):
            with patch(
                "huggingface_hub.file_download.http_get", side_effect=fake_http_get
            ):
                with pytest.raises(HfHubHTTPError):
                    backend.download("hf://org/repo/train/0000.parquet", str(dst))
        # Sidecar preserved so a follow-up call can resume.
        assert (tmp_path / "out.parquet.incomplete").read_bytes() == b"partial"


class TestHFBackendRevisionPinning:
    """``@<rev>`` rebuilds shard URLs via ``hf_hub_url`` for real pinning."""

    # /parquet returns resolve URLs with ``refs/convert/parquet`` URL-encoded
    # into the path (verified live against rajpurkar/squad).
    _PARQUET_URL = (
        "https://huggingface.co/datasets/org/repo/resolve/"
        "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    )

    def test_default_revision_uses_parquet_endpoint_url(self) -> None:
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", url=self._PARQUET_URL)]
        with patch("requests.Session.get", return_value=_mock_parquet_response(items)):
            shards = backend._list_shards(parse_hf_uri("hf://org/repo/train"))
        assert shards["0000.parquet"].url == self._PARQUET_URL

    def test_explicit_main_revision_treated_as_default(self) -> None:
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", url=self._PARQUET_URL)]
        with patch("requests.Session.get", return_value=_mock_parquet_response(items)):
            shards = backend._list_shards(parse_hf_uri("hf://org/repo@main/train"))
        assert shards["0000.parquet"].url == self._PARQUET_URL

    def test_explicit_tag_rebuilds_resolve_url(self) -> None:
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", url=self._PARQUET_URL)]
        with patch("requests.Session.get", return_value=_mock_parquet_response(items)):
            shards = backend._list_shards(parse_hf_uri("hf://org/repo@v1.0/train"))
        # Rebuilt as a /resolve/<rev>/<path> URL preserving the path-in-repo.
        assert shards["0000.parquet"].url == (
            "https://huggingface.co/datasets/org/repo/resolve/v1.0/"
            "default/train/0000.parquet"
        )

    def test_explicit_sha_rebuilds_resolve_url(self) -> None:
        backend = HFBackend()
        sha = "abc123def4567890"
        items = [_shard_item(filename="0000.parquet", url=self._PARQUET_URL)]
        with patch("requests.Session.get", return_value=_mock_parquet_response(items)):
            shards = backend._list_shards(parse_hf_uri(f"hf://org/repo@{sha}/train"))
        assert shards["0000.parquet"].url == (
            f"https://huggingface.co/datasets/org/repo/resolve/{sha}/"
            "default/train/0000.parquet"
        )

    def test_unparseable_parquet_url_under_explicit_rev_raises(self) -> None:
        """If /parquet returns a non-HF URL we can't rebuild a pinned URL."""
        backend = HFBackend()
        items = [
            _shard_item(
                filename="0000.parquet",
                url="https://cdn.example.com/mirror/0000.parquet",
            )
        ]
        with patch("requests.Session.get", return_value=_mock_parquet_response(items)):
            with pytest.raises(RuntimeError, match="Cannot pin to revision"):
                backend._list_shards(parse_hf_uri("hf://org/repo@v1.0/train"))

    def test_404_under_explicit_rev_hints_at_layout_mismatch(self) -> None:
        """When a rebuilt @<rev> URL 404s, the error message tells the user why."""
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", url=self._PARQUET_URL)]
        listing_resp = _mock_parquet_response(items)

        gone = MagicMock()
        gone.status_code = 404
        gone.raise_for_status = MagicMock()

        def fake_get(url: str, **kwargs: Any) -> Any:
            if _is_datasets_server_url(url):
                return listing_resp
            return gone

        with patch("requests.Session.get", side_effect=fake_get):
            with pytest.raises(FileNotFoundError) as exc:
                backend.read_range(
                    "hf://org/repo@v1.0/train/0000.parquet", 0, length=10
                )
        message = str(exc.value)
        assert "@'v1.0'" in message
        assert "different layout" in message or "refs/convert/parquet" in message
        assert "Drop the @<revision> suffix" in message

    def test_revisions_cache_independently(self) -> None:
        """Two different revisions issue distinct /parquet calls and produce distinct URLs."""
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", url=self._PARQUET_URL)]
        with patch(
            "requests.Session.get", return_value=_mock_parquet_response(items)
        ) as get:
            v1 = backend._list_shards(parse_hf_uri("hf://org/repo@v1/train"))
            v2 = backend._list_shards(parse_hf_uri("hf://org/repo@v2/train"))
        assert get.call_count == 2
        assert "/resolve/v1/" in v1["0000.parquet"].url
        assert "/resolve/v2/" in v2["0000.parquet"].url


class TestHFBackendPendingFailedGuard:
    """``pending``/``failed`` entries for the requested split must block."""

    def _resp(
        self,
        *,
        items: list[dict[str, Any]] | None = None,
        pending: list[dict[str, str]] | None = None,
        failed: list[dict[str, str]] | None = None,
        partial: bool = False,
    ) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "parquet_files": items or [_shard_item(filename="0000.parquet")],
            "pending": pending or [],
            "failed": failed or [],
            "partial": partial,
        }
        resp.raise_for_status = MagicMock()
        return resp

    def test_failed_for_requested_split_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ZEPHON_HF_ALLOW_PARTIAL", raising=False)
        backend = HFBackend()
        resp = self._resp(
            failed=[{"dataset": "org/repo", "config": "default", "split": "train"}],
        )
        with patch("requests.Session.get", return_value=resp):
            with pytest.raises(RuntimeError, match="failed conversion"):
                backend.listdir("hf://org/repo/train")

    def test_failed_for_other_split_does_not_raise(self) -> None:
        backend = HFBackend()
        resp = self._resp(
            failed=[{"dataset": "org/repo", "config": "default", "split": "test"}],
        )
        with patch("requests.Session.get", return_value=resp):
            assert backend.listdir("hf://org/repo/train") == ["0000.parquet"]

    def test_pending_for_requested_split_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ZEPHON_HF_ALLOW_PARTIAL", raising=False)
        backend = HFBackend()
        resp = self._resp(
            pending=[{"dataset": "org/repo", "config": "default", "split": "train"}],
        )
        with patch("requests.Session.get", return_value=resp):
            with pytest.raises(RuntimeError, match="pending conversion"):
                backend.listdir("hf://org/repo/train")

    def test_pending_override_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZEPHON_HF_ALLOW_PARTIAL", "1")
        backend = HFBackend()
        resp = self._resp(
            pending=[{"dataset": "org/repo", "config": "default", "split": "train"}],
        )
        with patch("requests.Session.get", return_value=resp):
            assert backend.listdir("hf://org/repo/train") == ["0000.parquet"]

    def test_failed_overrides_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``failed`` is fatal even with the partial-view escape hatch."""
        monkeypatch.setenv("ZEPHON_HF_ALLOW_PARTIAL", "1")
        backend = HFBackend()
        resp = self._resp(
            failed=[{"dataset": "org/repo", "config": "default", "split": "train"}],
        )
        with patch("requests.Session.get", return_value=resp):
            with pytest.raises(RuntimeError, match="failed conversion"):
                backend.listdir("hf://org/repo/train")


class TestHFBackendRangeQuirks:
    """``read_range`` must defend against servers that ignore ``Range``."""

    def test_read_range_slices_when_server_returns_200(self) -> None:
        backend = HFBackend()
        items = [_shard_item(filename="0000.parquet", size=100)]
        listing_resp = _mock_parquet_response(items)

        # CDN returned the whole body with status 200 instead of 206.
        whole_resp = MagicMock()
        whole_resp.status_code = 200
        whole_resp.content = bytes(range(100))
        whole_resp.raise_for_status = MagicMock()

        def fake_get(url: str, **kwargs: Any) -> Any:
            if _is_datasets_server_url(url):
                return listing_resp
            return whole_resp

        with patch("requests.Session.get", side_effect=fake_get):
            data = backend.read_range("hf://org/repo/train/0000.parquet", 10, length=5)
        assert data == bytes(range(10, 15))


class TestHFBackendRetries:
    """The session must mount a retry adapter for 429/5xx."""

    def test_session_has_retry_adapter_on_https(self) -> None:
        backend = HFBackend()
        session = backend._get_session()
        adapter = session.get_adapter("https://huggingface.co/")
        # urllib3 Retry is exposed on the adapter as ``max_retries``.
        retry = adapter.max_retries
        assert 500 in retry.status_forcelist
        assert 503 in retry.status_forcelist
        assert 429 in retry.status_forcelist
        assert retry.total >= 1


class TestHFBackendPartialPayload:
    """``partial=true`` from the Datasets Server is a foot-gun by default."""

    def _resp_with_partial(self, partial: bool) -> MagicMock:
        items = [_shard_item(filename="0000.parquet")]
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "parquet_files": items,
            "pending": [],
            "failed": [],
            "partial": partial,
        }
        resp.raise_for_status = MagicMock()
        return resp

    def test_partial_raises_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ZEPHON_HF_ALLOW_PARTIAL", raising=False)
        backend = HFBackend()
        with patch("requests.Session.get", return_value=self._resp_with_partial(True)):
            with pytest.raises(RuntimeError, match="partial"):
                backend.listdir("hf://org/repo/train")

    def test_partial_allowed_with_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ZEPHON_HF_ALLOW_PARTIAL", "1")
        backend = HFBackend()
        with patch("requests.Session.get", return_value=self._resp_with_partial(True)):
            assert backend.listdir("hf://org/repo/train") == ["0000.parquet"]


class TestRouterRoutesHFScheme:
    """``RouterStorageBackend`` should dispatch ``hf://`` to ``HFBackend``."""

    def test_router_selects_hf_backend(self) -> None:
        router = RouterStorageBackend()
        backend = router._backend_for("hf://org/repo/train")
        assert isinstance(backend, HFBackend)

    def test_router_caches_hf_backend_instance(self) -> None:
        router = RouterStorageBackend()
        a = router._backend_for("hf://org/repo/train")
        b = router._backend_for("hf://other/repo/test")
        assert a is b

    def test_hf_is_cloud_path(self) -> None:
        router = RouterStorageBackend()
        assert router.is_cloud_path("hf://org/repo/train") is True
