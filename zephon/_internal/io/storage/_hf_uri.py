# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace dataset URI grammar.

Defines the ``hf://`` URI shape and a parser. Internal to
:mod:`zephon._internal.io.storage` — the only consumer is
:class:`zephon._internal.io.storage.hf.HFBackend`. The leading underscore in the
module name signals this is not part of the public storage API.

URI grammar:
    hf://{org}/{name}[@[{revision}][~{source}]]/[{config}/]{split}

``source`` pins which files back the split: ``original`` (the files uploaded to
the repo) or ``parquet`` (HuggingFace's automatic Parquet conversion). Without
it the backend picks the uploaded files when Zephon can read their format and
falls back to the conversion otherwise.

A *frozen* URI names a 40-hex commit, a source and a config. It is what
:meth:`~zephon._internal.io.storage.hf.HFBackend.canonical_root` returns, and it
lists identically in every process:

    hf://{org}/{name}@{commit}~original/{config}/{split}
    hf://{org}/{name}@{conversion commit}~parquet/{config}/{conversion dir}

Examples:
    hf://HuggingFaceFW/fineweb-edu/train
    hf://rajpurkar/squad@main/plain_text/train
    hf://allenai/c4@~parquet/en/validation
"""

import re
from dataclasses import dataclass

HF_URI_SCHEME = "hf://"
SOURCE_ORIGINAL = "original"
SOURCE_PARQUET = "parquet"
_SOURCES = (SOURCE_ORIGINAL, SOURCE_PARQUET)

_SAFE_REVISION = re.compile(r"^[\w.\-]+$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class HFUriParts:
    """Parsed components of an ``hf://`` URI."""

    repo_id: str
    revision: str
    config: str | None
    split: str
    source: str | None = None

    @property
    def is_frozen(self) -> bool:
        """Whether the URI pins a commit, a source and a config."""
        return (
            self.source is not None
            and self.config is not None
            and COMMIT_RE.match(self.revision) is not None
        )

    def uri(self) -> str:
        """Render the parts back into an ``hf://`` URI."""
        revision = self.revision + (f"~{self.source}" if self.source else "")
        tail = f"{self.config}/{self.split}" if self.config else self.split
        return f"{HF_URI_SCHEME}{self.repo_id}@{revision}/{tail}"


def parse_hf_uri(uri: str) -> HFUriParts:
    """Parse an ``hf://{org}/{name}[@[{revision}][~{source}]]/[{config}/]{split}`` URI.

    Args:
        uri: The URI to parse.

    Returns:
        Parsed URI components. ``revision`` defaults to ``"main"`` when omitted.

    Raises:
        ValueError: If the URI does not match the documented grammar.
    """
    if not uri.startswith(HF_URI_SCHEME):
        raise ValueError(f"Expected hf:// URI, got: {uri!r}")

    body = uri[len(HF_URI_SCHEME) :].strip("/")
    if not body:
        raise ValueError(f"Empty hf:// URI: {uri!r}")

    segments = body.split("/")
    if len(segments) < 3:
        raise ValueError(
            f"hf:// URI must be hf://org/name[@rev]/[config/]split, got: {uri!r}"
        )

    org = segments[0]
    name_and_rev = segments[1]
    rest = segments[2:]

    source: str | None = None
    if "@" in name_and_rev:
        name, spec = name_and_rev.split("@", 1)
        if not spec:
            raise ValueError(f"Empty revision in hf:// URI: {uri!r}")
        revision, marker, source_name = spec.partition("~")
        if marker:
            if source_name not in _SOURCES:
                raise ValueError(
                    f"Unknown hf:// source {source_name!r} in {uri!r}; "
                    f"expected one of {', '.join('~' + s for s in _SOURCES)}"
                )
            source = source_name
        revision = revision or "main"
    else:
        name, revision = name_and_rev, "main"

    if not org or not name:
        raise ValueError(f"Malformed repo_id in hf:// URI: {uri!r}")
    if not _SAFE_REVISION.match(revision):
        raise ValueError(
            f"Revision must match [A-Za-z0-9._-]+, got {revision!r} in {uri!r}"
        )

    if len(rest) == 1:
        config: str | None = None
        split = rest[0]
    elif len(rest) == 2:
        config = rest[0]
        split = rest[1]
    else:
        raise ValueError(f"hf:// URI has too many path segments after repo_id: {uri!r}")

    if not split or config == "":
        raise ValueError(f"hf:// URI missing split or config: {uri!r}")

    return HFUriParts(
        repo_id=f"{org}/{name}",
        revision=revision,
        config=config,
        split=split,
        source=source,
    )


__all__ = [
    "COMMIT_RE",
    "HF_URI_SCHEME",
    "SOURCE_ORIGINAL",
    "SOURCE_PARQUET",
    "HFUriParts",
    "parse_hf_uri",
]
