# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace dataset URI grammar.

Defines the ``hf://`` URI shape and a parser. Internal to
:mod:`zephon.io.storage` — the only consumer is
:class:`zephon.io.storage.hf.HFBackend`. The leading underscore in the
module name signals this is not part of the public storage API.

URI grammar:
    hf://{org}/{name}[@{revision}]/[{config}/]{split}

Examples:
    hf://HuggingFaceFW/fineweb-edu/train
    hf://HuggingFaceH4/ultrachat_200k/train_sft
    hf://rajpurkar/squad@main/plain_text/train
"""

import re
from dataclasses import dataclass

HF_URI_SCHEME = "hf://"

_SAFE_REVISION = re.compile(r"^[\w.\-]+$")


@dataclass(frozen=True)
class HFUriParts:
    """Parsed components of an ``hf://`` URI."""

    repo_id: str
    revision: str
    config: str | None
    split: str


def parse_hf_uri(uri: str) -> HFUriParts:
    """Parse an ``hf://{org}/{name}[@{revision}]/[{config}/]{split}`` URI.

    Args:
        uri: The URI to parse.

    Returns:
        Parsed URI components. ``revision`` defaults to ``"main"`` when omitted.

    Raises:
        ValueError: If the URI does not match the documented grammar.

    Note:
        ``@<revision>`` is honored by :class:`~zephon.io.storage.hf.HFBackend`
        for parquet-native repos: the backend rebuilds per-shard URLs via
        ``huggingface_hub.hf_hub_url(..., revision=<rev>)`` so reads are
        pinned to that revision. For non-parquet-native repos the
        user-specified revision won't have parquet at the auto-converted
        paths and reads will fail with ``FileNotFoundError`` at the
        revision-pinned URL — by design, to avoid silently substituting
        ``refs/convert/parquet`` bytes for the user's chosen revision.
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

    if "@" in name_and_rev:
        name, revision = name_and_rev.split("@", 1)
    else:
        name, revision = name_and_rev, "main"

    if not org or not name:
        raise ValueError(f"Malformed repo_id in hf:// URI: {uri!r}")
    if not revision or not _SAFE_REVISION.match(revision):
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

    if not split:
        raise ValueError(f"hf:// URI missing split: {uri!r}")

    return HFUriParts(
        repo_id=f"{org}/{name}", revision=revision, config=config, split=split
    )


__all__ = ["HF_URI_SCHEME", "HFUriParts", "parse_hf_uri"]
