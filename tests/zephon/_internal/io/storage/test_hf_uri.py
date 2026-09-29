# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``hf://`` URI grammar."""

from __future__ import annotations

import pytest

from zephon._internal.io.storage._hf_uri import (
    SOURCE_ORIGINAL,
    SOURCE_PARQUET,
    HFUriParts,
    parse_hf_uri,
)

_COMMIT = "0123456789abcdef0123456789abcdef01234567"


class TestParseHFUri:
    def test_minimal(self) -> None:
        assert parse_hf_uri("hf://org/repo/train") == HFUriParts(
            repo_id="org/repo", revision="main", config=None, split="train"
        )

    def test_with_revision(self) -> None:
        parts = parse_hf_uri("hf://org/repo@v1.2/train")
        assert parts.revision == "v1.2"
        assert parts.repo_id == "org/repo"

    def test_with_config(self) -> None:
        assert parse_hf_uri("hf://org/repo/plain_text/train") == HFUriParts(
            repo_id="org/repo", revision="main", config="plain_text", split="train"
        )

    def test_dotted_config(self) -> None:
        parts = parse_hf_uri("hf://allenai/c4/en.noblocklist/train")
        assert (parts.config, parts.split) == ("en.noblocklist", "train")

    def test_trailing_slash_tolerated(self) -> None:
        assert parse_hf_uri("hf://org/repo/train/").split == "train"

    @pytest.mark.parametrize(
        ("uri", "revision", "source"),
        [
            ("hf://org/repo@~parquet/train", "main", SOURCE_PARQUET),
            ("hf://org/repo@~original/train", "main", SOURCE_ORIGINAL),
            ("hf://org/repo@v1.0~original/train", "v1.0", SOURCE_ORIGINAL),
            (f"hf://org/repo@{_COMMIT}~parquet/cfg/train", _COMMIT, SOURCE_PARQUET),
        ],
    )
    def test_source_marker(self, uri: str, revision: str, source: str) -> None:
        parts = parse_hf_uri(uri)
        assert (parts.revision, parts.source) == (revision, source)

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
            "hf://org/repo@main~bogus/train",  # unknown source
            "hf://org/repo@~/train",  # empty source
            "hf://org/repo@v1~parquet~x/train",  # one marker at most
        ],
    )
    def test_rejects_malformed(self, bad_uri: str) -> None:
        with pytest.raises(ValueError):
            parse_hf_uri(bad_uri)


class TestFrozen:
    @pytest.mark.parametrize(
        ("uri", "frozen"),
        [
            (f"hf://org/repo@{_COMMIT}~original/cfg/train", True),
            (f"hf://org/repo@{_COMMIT}~parquet/cfg/partial-train", True),
            (f"hf://org/repo@{_COMMIT}~original/train", False),  # no config
            (f"hf://org/repo@{_COMMIT}/cfg/train", False),  # no source
            ("hf://org/repo@main~original/cfg/train", False),  # not a commit
            (f"hf://org/repo@{_COMMIT[:12]}~original/cfg/train", False),  # short sha
        ],
    )
    def test_is_frozen(self, uri: str, frozen: bool) -> None:
        assert parse_hf_uri(uri).is_frozen is frozen

    @pytest.mark.parametrize(
        "uri",
        [
            f"hf://org/repo@{_COMMIT}~original/cfg/train",
            "hf://org/repo@main/train",
            "hf://org/repo@v1~parquet/cfg/train",
        ],
    )
    def test_uri_round_trips(self, uri: str) -> None:
        parts = parse_hf_uri(uri)
        assert parts.uri() == uri
        assert parse_hf_uri(parts.uri()) == parts
