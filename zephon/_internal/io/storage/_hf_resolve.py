# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Resolve an ``hf://`` URI to the files that back one split.

A split is backed either by the files uploaded to the repo or by HuggingFace's
automatic Parquet conversion (branch ``refs/convert/parquet``). The uploaded
files win when Zephon can read their format; otherwise the conversion is used
if it is complete and was built from the requested commit. The choice is frozen
into the returned URI so every later process lists the same files.

Everything here speaks repo paths; :mod:`~zephon._internal.io.storage.hf` maps
them to virtual-directory names.
"""

from __future__ import annotations

import collections
import dataclasses
import os
import re
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol

from zephon._internal.io.storage._hf_uri import (
    SOURCE_ORIGINAL,
    SOURCE_PARQUET,
    HFUriParts,
)
from zephon._internal.io.suffixes import REMOTE_SCAN_FORMATS, format_of

# How ``datasets`` refuses repos that ship a loading script.
_SCRIPT_REJECTION = "Dataset scripts are no longer supported"


class HubClient(Protocol):
    """The HuggingFace Hub and Dataset Viewer calls resolution needs."""

    def commit_for(self, repo_id: str, revision: str) -> str:
        """Return the 40-hex commit ``revision`` names."""
        ...

    def conversion_commit(self, repo_id: str) -> str | None:
        """Return the head commit of ``refs/convert/parquet``, if it exists."""
        ...

    def parquet_export(
        self, repo_id: str, config: str | None
    ) -> tuple[Mapping[str, Any], str | None]:
        """Return the ``/parquet`` payload and its ``X-Revision`` header."""
        ...

    def list_tree(
        self, repo_id: str, commit: str, path: str
    ) -> tuple[dict[str, int], list[str]]:
        """Return ``({file name: size}, [subdirectory names])`` directly under ``path``."""
        ...

    def file_sizes(
        self, repo_id: str, commit: str, paths: Sequence[str]
    ) -> dict[str, int]:
        """Return ``{repo path: size}`` for ``paths`` at ``commit``."""
        ...


@dataclass(frozen=True)
class Resolution:
    """A frozen URI, the repo files (``{path: size}``) behind it, and why."""

    parts: HFUriParts
    files: dict[str, int]
    summary: str


@dataclass(frozen=True)
class _Uploaded:
    config: str | None
    paths: tuple[str, ...]
    reason: str | None  # why these files cannot be read, when they cannot


def resolve(
    hub: HubClient, parts: HFUriParts, fmt: str | None, *, allow_partial: bool
) -> Resolution:
    """Pick the files backing ``parts`` and freeze the choice.

    ``fmt`` restricts the pick to sources in that format. Only definite
    rejections (an unreadable format, a script repo, a missing or incomplete
    conversion) move on to the next source; auth and network errors propagate.

    Raises:
        ValueError: if neither source is usable; the message gives each reason.
    """
    commit = hub.commit_for(parts.repo_id, parts.revision)
    rejected: list[str] = []
    config = parts.config

    # The uploaded files back the original source, and ``datasets``' default
    # config picks the logical dataset for both sources.
    uploaded: _Uploaded | None = None
    if parts.source != SOURCE_PARQUET or config is None:
        uploaded = _uploaded_files(parts.repo_id, commit, config, parts.split)
        config = uploaded.config or config

    if uploaded is not None and parts.source != SOURCE_PARQUET:
        kind, reason = _readable_kind(uploaded)
        if reason is None and fmt is not None and kind != fmt:
            reason = f"they are {kind}, not the requested {fmt}"
        if reason is None:
            assert uploaded.config is not None and kind is not None
            frozen = HFUriParts(
                repo_id=parts.repo_id,
                revision=commit,
                config=uploaded.config,
                split=parts.split,
                source=SOURCE_ORIGINAL,
            )
            files = hub.file_sizes(parts.repo_id, commit, uploaded.paths)
            return Resolution(frozen, files, f"{len(files)} uploaded {kind} files")
        rejected.append(f"uploaded files: {reason}")

    if parts.source != SOURCE_ORIGINAL:
        if fmt not in (None, "parquet"):
            rejected.append(
                f"parquet conversion: it is parquet, not the requested {fmt}"
            )
        else:
            conversion = _conversion(
                hub, parts.repo_id, commit, config, parts.split, allow_partial
            )
            if isinstance(conversion, Resolution):
                summary = "; ".join([conversion.summary, *rejected])
                return dataclasses.replace(conversion, summary=summary)
            rejected.append(f"parquet conversion: {conversion}")

    raise ValueError(
        f"No readable source for {parts.uri()} at commit {commit}:\n  - "
        + "\n  - ".join(rejected)
    )


def list_frozen(hub: HubClient, parts: HFUriParts) -> dict[str, int]:
    """Return ``{repo path: size}`` for a frozen URI without re-selecting."""
    assert parts.is_frozen and parts.config is not None
    if parts.source == SOURCE_PARQUET:
        try:
            files = _conversion_files(
                hub, parts.repo_id, parts.revision, parts.config, parts.split
            )
        except FileNotFoundError:
            files = {}
        if not files:
            raise FileNotFoundError(
                f"{parts.uri()}: no {parts.split!r} under {parts.config!r} at commit "
                f"{parts.revision}. A ~parquet revision must be a "
                "refs/convert/parquet commit."
            )
        return files

    uploaded = _uploaded_files(parts.repo_id, parts.revision, parts.config, parts.split)
    if uploaded.reason is not None:
        raise RuntimeError(f"Cannot list {parts.uri()}: {uploaded.reason}")
    return hub.file_sizes(parts.repo_id, parts.revision, uploaded.paths)


def _builder_path(datasets_version: str, repo_id: str) -> str:
    """The ``load_dataset_builder`` path that reaches ``repo_id`` on the Hub only."""
    major, minor = (int(part) for part in datasets_version.split(".")[:2])
    if (major, minor) >= (4, 8):
        # Never loads a same-named local directory in the working directory.
        return f"hf://datasets/{repo_id}"
    # Older ``datasets`` reach the Hub only through a bare ``org/name``, which
    # a same-named local path would shadow.
    if os.path.exists(repo_id):
        raise RuntimeError(
            f"Local path {repo_id!r} would shadow hf://{repo_id} with datasets "
            f"{datasets_version}; run from another directory or upgrade to "
            "datasets>=4.8"
        )
    return repo_id


def _uploaded_files(
    repo_id: str, commit: str, config: str | None, split: str
) -> _Uploaded:
    """Resolve the uploaded files of ``split`` the way ``datasets`` would."""
    try:
        # Deferred: ``storage/__init__`` imports the HF backend in every
        # process, and ``datasets`` pulls in pandas, dill and multiprocess.
        # Only resolution needs it; workers download by name.
        import datasets
    except ModuleNotFoundError as exc:
        if exc.name != "datasets":
            raise  # installed but broken: never a reason to switch sources
        return _Uploaded(config, (), "the `datasets` package is not installed")
    from datasets.data_files import EmptyDatasetError
    from datasets.exceptions import DataFilesNotFoundError

    try:
        builder = datasets.load_dataset_builder(
            _builder_path(datasets.__version__, repo_id), name=config, revision=commit
        )
    except (DataFilesNotFoundError, EmptyDatasetError) as exc:
        return _Uploaded(config, (), f"`datasets` finds no data files ({exc})")
    except RuntimeError as exc:
        if not str(exc).startswith(_SCRIPT_REJECTION):
            raise
        return _Uploaded(config, (), str(exc))

    resolved_config = builder.config.name
    data_files = {
        str(name): files for name, files in (builder.config.data_files or {}).items()
    }
    if split not in data_files:
        raise FileNotFoundError(
            f"hf://{repo_id} config {resolved_config!r} has no split {split!r}; "
            f"available: {sorted(data_files)}"
        )

    pinned = f"hf://datasets/{repo_id}@{commit}/"
    paths: list[str] = []
    for url in data_files[split]:
        if url.startswith(pinned):
            paths.append(url[len(pinned) :])
        elif url.startswith(f"hf://datasets/{repo_id}@"):
            raise RuntimeError(
                f"`datasets` resolved {url!r} at a commit other than {commit}"
            )
        else:
            return _Uploaded(
                resolved_config,
                (),
                f"data files like {url} live outside hf://datasets/{repo_id}",
            )

    repeated = [path for path, count in collections.Counter(paths).items() if count > 1]
    if repeated:
        # ``datasets`` reads a repeated file once per listing; one shard per
        # file would drop those rows.
        return _Uploaded(
            resolved_config,
            tuple(sorted(paths)),
            f"the split lists files more than once (e.g. {sorted(repeated)[0]})",
        )
    return _Uploaded(resolved_config, tuple(sorted(paths)), None)


def _readable_kind(uploaded: _Uploaded) -> tuple[str | None, str | None]:
    """Return ``(format, None)`` if Zephon reads the files, else ``(None, reason)``."""
    if uploaded.reason is not None:
        return None, uploaded.reason
    if not uploaded.paths:
        return None, "the split has no files"
    kinds = {format_of(PurePosixPath(path).name) for path in uploaded.paths}
    if None in kinds:
        examples = sorted(
            PurePosixPath(path).name
            for path in uploaded.paths
            if format_of(PurePosixPath(path).name) is None
        )[:3]
        return None, f"Zephon reads no format for files like {', '.join(examples)}"
    if len(kinds) > 1:
        return None, f"they mix formats ({', '.join(sorted(str(k) for k in kinds))})"
    kind = kinds.pop()
    assert kind is not None
    if kind not in REMOTE_SCAN_FORMATS:
        return None, f"{kind} discovery only works on local files"
    return kind, None


def _conversion(
    hub: HubClient,
    repo_id: str,
    commit: str,
    config: str | None,
    split: str,
    allow_partial: bool,
) -> Resolution | str:
    """Return the conversion's files, or why the conversion cannot be used."""
    try:
        payload, built_from = hub.parquet_export(repo_id, config)
    except FileNotFoundError:
        return "HuggingFace has no parquet conversion of this dataset"
    if built_from != commit:
        # HF converts only the latest commit; never serve it for another.
        return f"it was built from commit {built_from or 'unknown'}, not {commit}"

    entries = [
        item
        for item in payload.get("parquet_files", [])
        if item.get("split") == split
        and (config is None or item.get("config") == config)
    ]
    configs = sorted({str(item.get("config")) for item in entries})
    if not configs:
        return f"it has no split {split!r}" + (
            f" in config {config!r}" if config else ""
        )
    if len(configs) > 1:
        raise ValueError(
            f"hf://{repo_id} has split {split!r} in configs {configs}; "
            "name one in the URI (hf://org/name/<config>/<split>)"
        )
    chosen = configs[0]

    incomplete, fatal = _incomplete_reason(payload, chosen, split)
    if incomplete is not None and (fatal or not allow_partial):
        return incomplete

    expected = {
        _conversion_path(repo_id, str(item.get("url"))): int(item["size"])
        for item in entries
    }
    bases = {_split_base(split, path, chosen) for path in expected}
    if len(bases) != 1 or None in bases:
        raise RuntimeError(
            f"Unexpected conversion layout for hf://{repo_id} split {split!r}: "
            f"{sorted(expected)[:3]}"
        )
    base = bases.pop()
    assert base is not None
    if base.startswith("partial-") and not allow_partial:
        return (
            "it is partial (HF converts only the first ~5 GB of large "
            "datasets); set ZEPHON_HF_ALLOW_PARTIAL=1 to use it anyway"
        )

    conversion_commit = hub.conversion_commit(repo_id)
    if conversion_commit is None:
        return "the refs/convert/parquet branch does not exist"
    listed = _conversion_files(hub, repo_id, conversion_commit, chosen, base)
    if listed != expected:  # the conversion moved between the two reads
        return "HF's parquet conversion changed while resolving; retry"
    # The Viewer response and the branch head are read separately, and an
    # update can keep every path and size. HF converts only main's head, so
    # an unchanged head proves the pinned conversion is of ``commit``.
    head = hub.commit_for(repo_id, "main")
    if head != commit:
        return (
            f"main moved to {head} while resolving, and HF converts only "
            "main's latest commit"
        )

    frozen = HFUriParts(
        repo_id=repo_id,
        revision=conversion_commit,
        config=chosen,
        split=base,
        source=SOURCE_PARQUET,
    )
    return Resolution(
        frozen, listed, f"{len(listed)} files from HF's parquet conversion"
    )


def _conversion_files(
    hub: HubClient, repo_id: str, commit: str, config: str, base: str
) -> dict[str, int]:
    """List the conversion files of split directory ``base`` under ``config``.

    HF moves a split with more than 10,000 files into ``{base}-part{k}``
    directories (each restarting at ``0000.parquet``) and never keeps a bare
    ``{base}`` next to them.
    """
    _, subdirs = hub.list_tree(repo_id, commit, config)
    if base in subdirs:
        directories = [base]
    else:
        part = re.compile(re.escape(base) + r"-part\d+")
        directories = sorted(d for d in subdirs if part.fullmatch(d))
    files: dict[str, int] = {}
    for directory in directories:
        listed, _ = hub.list_tree(repo_id, commit, f"{config}/{directory}")
        files.update(
            {f"{config}/{directory}/{name}": size for name, size in listed.items()}
        )
    return files


def _split_base(split: str, repo_path: str, config: str) -> str | None:
    """Return the split directory (``[partial-]{split}``) holding ``repo_path``."""
    parent = PurePosixPath(repo_path).parent
    if parent.parent != PurePosixPath(config):
        return None
    for base in (split, f"partial-{split}"):
        if parent.name == base or re.fullmatch(
            re.escape(base) + r"-part\d+", parent.name
        ):
            return base
    return None


def _incomplete_reason(
    payload: Mapping[str, Any], config: str, split: str
) -> tuple[str | None, bool]:
    """Describe a failed (fatal) or pending conversion of ``config``/``split``."""

    def touches(item: Mapping[str, Any]) -> bool:
        return item.get("config") in (None, config) and item.get("split") in (
            None,
            split,
        )

    if any(touches(item) for item in payload.get("failed") or []):
        return "HF failed to convert this split", True
    if any(touches(item) for item in payload.get("pending") or []):
        return (
            "it is still being converted; retry later or set "
            "ZEPHON_HF_ALLOW_PARTIAL=1 to use what exists"
        ), False
    return None, False


def _conversion_path(repo_id: str, url: str) -> str:
    """Return the repo path (``{config}/{dir}/{file}``) of a ``/parquet`` resolve URL."""
    path = urllib.parse.urlparse(url).path
    prefix = f"/datasets/{repo_id}/resolve/"
    if not path.startswith(prefix) or path.count("/") < prefix.count("/") + 3:
        raise RuntimeError(f"Unexpected /parquet file URL shape: {url!r}")
    # Drop the URL-encoded ``refs%2Fconvert%2Fparquet`` segment, keep the path.
    _ref, repo_path = path[len(prefix) :].split("/", 1)
    return urllib.parse.unquote(repo_path)


__all__ = ["HubClient", "Resolution", "list_frozen", "resolve"]
