# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for choosing and freezing the files behind an ``hf://`` split."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import zephon._internal.io.storage._hf_resolve as resolve_mod
from zephon._internal.io.storage._hf_resolve import _Uploaded, list_frozen, resolve
from zephon._internal.io.storage._hf_uri import (
    SOURCE_ORIGINAL,
    SOURCE_PARQUET,
    parse_hf_uri,
)

SOURCE = "1" * 40
CONVERSION = "2" * 40
_REPO = "org/repo"


def _item(
    filename: str,
    *,
    size: int = 10,
    config: str = "default",
    split: str = "train",
    directory: str | None = None,
) -> dict[str, Any]:
    directory = directory or split
    return {
        "dataset": _REPO,
        "config": config,
        "split": split,
        "filename": filename,
        "size": size,
        "url": (
            f"https://huggingface.co/datasets/{_REPO}/resolve/"
            f"refs%2Fconvert%2Fparquet/{config}/{directory}/{filename}"
        ),
    }


def _export(
    *items: dict[str, Any],
    pending: Sequence[Mapping[str, Any]] = (),
    failed: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "parquet_files": list(items),
        "pending": list(pending),
        "failed": list(failed),
        "partial": False,
    }


class _FakeHub:
    """In-memory HubClient.

    ``trees`` maps ``(commit, directory)`` to its files; subdirectories are
    derived from the keys.
    """

    def __init__(
        self,
        *,
        export: Mapping[str, Any] | None = None,
        built_from: str | None = SOURCE,
        trees: Mapping[tuple[str, str], Any] | None = None,
        sizes: Mapping[str, int] | None = None,
        main_heads: Sequence[str] = (),
    ) -> None:
        self.export = export
        self.built_from = built_from
        self.trees = dict(trees or {})
        self.sizes = dict(sizes or {})
        self.calls: list[str] = []
        self.main_heads = list(main_heads)  # later ``main`` lookups, in order

    def commit_for(self, repo_id: str, revision: str) -> str:
        self.calls.append(f"commit_for:{revision}")
        first = not any(call.startswith("commit_for") for call in self.calls[:-1])
        if revision == "main" and not first and self.main_heads:
            return self.main_heads.pop(0)
        return SOURCE

    def conversion_commit(self, repo_id: str) -> str | None:
        self.calls.append("conversion_commit")
        return CONVERSION

    def parquet_export(
        self, repo_id: str, config: str | None
    ) -> tuple[Mapping[str, Any], str | None]:
        self.calls.append(f"parquet_export:{config}")
        if self.export is None:
            raise FileNotFoundError("no conversion")
        return self.export, self.built_from

    def list_tree(
        self, repo_id: str, commit: str, path: str
    ) -> tuple[dict[str, int], list[str]]:
        self.calls.append(f"list_tree:{commit[:1]}:{path}")
        prefix = f"{path}/"
        subdirs = sorted(
            {
                key[len(prefix) :].split("/")[0]
                for tree_commit, key in self.trees
                if tree_commit == commit and key.startswith(prefix)
            }
        )
        listing = self.trees.get((commit, path))
        if listing is None and not subdirs:
            raise FileNotFoundError(path)
        return dict(listing or {}), subdirs

    def file_sizes(
        self, repo_id: str, commit: str, paths: Sequence[str]
    ) -> dict[str, int]:
        self.calls.append("file_sizes")
        return {path: self.sizes.get(path, 1) for path in paths}


@pytest.fixture
def uploaded(monkeypatch: pytest.MonkeyPatch):
    """Stub ``datasets``-backed resolution; call with the ``_Uploaded`` to return."""
    seen: list[tuple[str, str, str | None, str]] = []

    def install(result: _Uploaded) -> list[tuple[str, str, str | None, str]]:
        def fake(
            repo_id: str, commit: str, config: str | None, split: str
        ) -> _Uploaded:
            seen.append((repo_id, commit, config, split))
            return result

        monkeypatch.setattr(resolve_mod, "_uploaded_files", fake)
        return seen

    return install


def _conversion_hub(**kwargs: Any) -> _FakeHub:
    items = [_item("0000.parquet", size=7), _item("0001.parquet", size=8)]
    return _FakeHub(
        export=_export(*items),
        trees={(CONVERSION, "default/train"): {"0000.parquet": 7, "0001.parquet": 8}},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# resolve(): which source wins
# ---------------------------------------------------------------------------


def test_readable_uploaded_files_win(uploaded) -> None:
    uploaded(_Uploaded("default", ("data/a.parquet", "data/b.parquet"), None))
    hub = _FakeHub(sizes={"data/a.parquet": 3, "data/b.parquet": 4})

    result = resolve(
        hub, parse_hf_uri("hf://org/repo/train"), None, allow_partial=False
    )

    assert result.parts.uri() == f"hf://org/repo@{SOURCE}~original/default/train"
    assert result.files == {"data/a.parquet": 3, "data/b.parquet": 4}
    assert not any(call.startswith("parquet_export") for call in hub.calls)


def test_compressed_jsonl_uploads_are_readable(uploaded) -> None:
    uploaded(_Uploaded("default", ("shard_0.jsonl.zst",), None))
    result = resolve(
        _FakeHub(), parse_hf_uri("hf://org/repo/train"), None, allow_partial=False
    )
    assert result.parts.source == SOURCE_ORIGINAL


@pytest.mark.parametrize(
    ("paths", "reason"),
    [
        (("en/c4-train.00000.json.gz",), "no format"),  # JSON is not JSONL
        (("a.parquet", "b.jsonl"), "mix formats"),
        (("a.vortex",), "only works on local files"),
        ((), "no files"),
    ],
)
def test_unreadable_uploads_fall_back_to_conversion(
    uploaded, paths: tuple[str, ...], reason: str
) -> None:
    uploaded(_Uploaded("default", paths, None))
    hub = _conversion_hub(built_from="0" * 40)  # conversion unusable too

    with pytest.raises(ValueError) as exc:
        resolve(hub, parse_hf_uri("hf://org/repo/train"), None, allow_partial=False)

    assert reason in str(exc.value)
    assert "parquet conversion:" in str(exc.value)


def test_conversion_used_when_uploads_unreadable(uploaded) -> None:
    uploaded(_Uploaded("default", ("x.json.gz",), None))
    hub = _conversion_hub()

    result = resolve(
        hub, parse_hf_uri("hf://org/repo/train"), None, allow_partial=False
    )

    assert result.parts.uri() == f"hf://org/repo@{CONVERSION}~parquet/default/train"
    assert result.files == {
        "default/train/0000.parquet": 7,
        "default/train/0001.parquet": 8,
    }


def test_conversion_of_another_commit_is_never_served(uploaded) -> None:
    uploaded(_Uploaded("default", ("x.csv",), None))
    hub = _conversion_hub(built_from="9" * 40)

    with pytest.raises(ValueError, match="built from commit 9{40}"):
        resolve(hub, parse_hf_uri("hf://org/repo@v1/train"), None, allow_partial=False)


def test_missing_conversion_is_a_rejection(uploaded) -> None:
    uploaded(_Uploaded("default", ("x.csv",), None))
    with pytest.raises(ValueError, match="has no parquet conversion"):
        resolve(
            _FakeHub(), parse_hf_uri("hf://org/repo/train"), None, allow_partial=False
        )


@pytest.mark.parametrize("allow_partial", [False, True])
def test_partial_conversion_needs_opt_in(uploaded, allow_partial: bool) -> None:
    uploaded(_Uploaded("default", ("x.json.gz",), None))
    items = [_item("0000.parquet", directory="partial-train")]
    hub = _FakeHub(
        export=_export(*items),
        trees={(CONVERSION, "default/partial-train"): {"0000.parquet": 10}},
    )
    parts = parse_hf_uri("hf://org/repo/train")

    if not allow_partial:
        with pytest.raises(ValueError, match="ZEPHON_HF_ALLOW_PARTIAL"):
            resolve(hub, parts, None, allow_partial=False)
        return
    result = resolve(hub, parts, None, allow_partial=True)
    assert result.parts.split == "partial-train"


def test_pending_conversion_needs_opt_in(uploaded) -> None:
    uploaded(_Uploaded("default", ("x.json.gz",), None))
    pending = [{"config": "default", "split": "train"}]
    hub = _conversion_hub()
    assert hub.export is not None
    hub.export = {**hub.export, "pending": pending}

    with pytest.raises(ValueError, match="still being converted"):
        resolve(hub, parse_hf_uri("hf://org/repo/train"), None, allow_partial=False)
    assert resolve(hub, parse_hf_uri("hf://org/repo/train"), None, allow_partial=True)


def test_failed_conversion_is_fatal_even_with_opt_in(uploaded) -> None:
    uploaded(_Uploaded("default", ("x.json.gz",), None))
    hub = _conversion_hub()
    assert hub.export is not None
    # A config-level failure (split=None) covers every split of the config.
    hub.export = {**hub.export, "failed": [{"config": "default", "split": None}]}

    with pytest.raises(ValueError, match="failed to convert"):
        resolve(hub, parse_hf_uri("hf://org/repo/train"), None, allow_partial=True)


def test_fmt_parquet_prefers_conversion_over_jsonl_upload(uploaded) -> None:
    uploaded(_Uploaded("default", ("data.jsonl",), None))
    result = resolve(
        _conversion_hub(),
        parse_hf_uri("hf://org/repo/train"),
        "parquet",
        allow_partial=False,
    )
    assert result.parts.source == SOURCE_PARQUET


def test_fmt_parquet_keeps_parquet_upload(uploaded) -> None:
    uploaded(_Uploaded("default", ("data.parquet",), None))
    result = resolve(
        _FakeHub(), parse_hf_uri("hf://org/repo/train"), "parquet", allow_partial=False
    )
    assert result.parts.source == SOURCE_ORIGINAL


def test_fmt_without_matching_source_raises(uploaded) -> None:
    uploaded(_Uploaded("default", ("data.parquet",), None))
    hub = _conversion_hub()

    with pytest.raises(ValueError) as exc:
        resolve(hub, parse_hf_uri("hf://org/repo/train"), "jsonl", allow_partial=False)

    assert "not the requested jsonl" in str(exc.value)
    assert not any(call.startswith("parquet_export") for call in hub.calls)


def test_forced_original_never_consults_conversion(uploaded) -> None:
    uploaded(_Uploaded("default", ("x.json.gz",), None))
    hub = _conversion_hub()

    with pytest.raises(ValueError, match="uploaded files"):
        resolve(
            hub,
            parse_hf_uri("hf://org/repo@~original/train"),
            None,
            allow_partial=False,
        )
    assert not any(call.startswith("parquet_export") for call in hub.calls)


def test_forced_parquet_skips_uploaded_files(uploaded) -> None:
    seen = uploaded(_Uploaded("default", ("data.parquet",), None))
    result = resolve(
        _conversion_hub(),
        parse_hf_uri("hf://org/repo@~parquet/default/train"),
        None,
        allow_partial=False,
    )
    assert result.parts.source == SOURCE_PARQUET
    assert seen == []


def test_conversion_config_inferred_when_datasets_is_missing(uploaded) -> None:
    uploaded(_Uploaded(None, (), "the `datasets` package is not installed"))
    items = [_item("0000.parquet", config="plain", size=5)]
    hub = _FakeHub(
        export=_export(items[0]),
        trees={(CONVERSION, "plain/train"): {"0000.parquet": 5}},
    )

    result = resolve(
        hub, parse_hf_uri("hf://org/repo/train"), None, allow_partial=False
    )

    assert result.parts.config == "plain"
    assert "parquet_export:None" in hub.calls


def test_split_in_several_conversion_configs_needs_a_config(uploaded) -> None:
    uploaded(_Uploaded(None, (), "the `datasets` package is not installed"))
    hub = _FakeHub(
        export=_export(_item("a.parquet", config="a"), _item("b.parquet", config="b"))
    )

    with pytest.raises(ValueError, match="name one in the URI"):
        resolve(hub, parse_hf_uri("hf://org/repo/train"), None, allow_partial=False)


def test_conversion_that_moves_while_resolving_is_rejected(uploaded) -> None:
    uploaded(_Uploaded("default", ("x.csv",), None))
    hub = _conversion_hub()
    hub.trees[(CONVERSION, "default/train")] = {
        "0000.parquet": 1
    }  # not what /parquet said

    with pytest.raises(ValueError, match="changed while resolving"):
        resolve(hub, parse_hf_uri("hf://org/repo/train"), None, allow_partial=False)


class _OfflineHub(_FakeHub):
    def parquet_export(
        self, repo_id: str, config: str | None
    ) -> tuple[Mapping[str, Any], str | None]:
        raise ConnectionError("offline")


def test_network_errors_propagate(uploaded) -> None:
    uploaded(_Uploaded("default", ("x.csv",), None))
    with pytest.raises(ConnectionError):
        resolve(
            _OfflineHub(),
            parse_hf_uri("hf://org/repo/train"),
            None,
            allow_partial=False,
        )


# ---------------------------------------------------------------------------
# list_frozen(): relisting never re-selects
# ---------------------------------------------------------------------------


def test_list_frozen_conversion_reads_only_the_tree(uploaded) -> None:
    seen = uploaded(_Uploaded("default", ("data.parquet",), None))
    hub = _conversion_hub()
    parts = parse_hf_uri(f"hf://org/repo@{CONVERSION}~parquet/default/train")

    assert list_frozen(hub, parts) == {
        "default/train/0000.parquet": 7,
        "default/train/0001.parquet": 8,
    }
    assert hub.calls == ["list_tree:2:default", "list_tree:2:default/train"]
    assert seen == []


def test_list_frozen_conversion_needs_a_conversion_commit(uploaded) -> None:
    parts = parse_hf_uri(f"hf://org/repo@{SOURCE}~parquet/default/train")
    with pytest.raises(FileNotFoundError, match="refs/convert/parquet commit"):
        list_frozen(_FakeHub(), parts)


def test_list_frozen_original_uses_the_pinned_commit(uploaded) -> None:
    seen = uploaded(_Uploaded("default", ("data/a.parquet",), None))
    parts = parse_hf_uri(f"hf://org/repo@{SOURCE}~original/default/train")

    assert list_frozen(_FakeHub(sizes={"data/a.parquet": 9}), parts) == {
        "data/a.parquet": 9
    }
    assert seen == [(_REPO, SOURCE, "default", "train")]


def test_list_frozen_original_without_datasets_raises(uploaded) -> None:
    uploaded(_Uploaded("default", (), "the `datasets` package is not installed"))
    parts = parse_hf_uri(f"hf://org/repo@{SOURCE}~original/default/train")
    with pytest.raises(RuntimeError, match="not installed"):
        list_frozen(_FakeHub(), parts)


# ---------------------------------------------------------------------------
# _uploaded_files(): the datasets glue
# ---------------------------------------------------------------------------


def _fake_builder(
    monkeypatch: pytest.MonkeyPatch, config: Any = None, error: Exception | None = None
) -> list[str]:
    """Stub ``load_dataset_builder``; returns the paths it was asked to load."""
    datasets = pytest.importorskip("datasets")
    loaded: list[str] = []

    def fake_load(
        path: str, name: str | None = None, revision: str | None = None
    ) -> Any:
        loaded.append(path)
        if error is not None:
            raise error
        return SimpleNamespace(config=config)

    monkeypatch.setattr(datasets, "load_dataset_builder", fake_load)
    return loaded


def _data_files(splits: Mapping[str, list[str]]) -> Any:
    from datasets.data_files import DataFilesDict, DataFilesList

    return DataFilesDict(
        {
            split: DataFilesList(urls, origin_metadata=[()] * len(urls))
            for split, urls in splits.items()
        }
    )


def _pinned(*names: str) -> list[str]:
    return [f"hf://datasets/{_REPO}@{SOURCE}/{name}" for name in names]


def _json_config(**kwargs: Any) -> Any:
    from datasets.packaged_modules.json.json import JsonConfig

    splits = kwargs.pop("splits", {"train": _pinned("b.jsonl", "a.jsonl")})
    return JsonConfig(name="default", data_files=_data_files(splits), **kwargs)


def test_uploaded_files_strips_the_pinned_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("datasets")
    _fake_builder(monkeypatch, _json_config())
    result = resolve_mod._uploaded_files(_REPO, SOURCE, None, "train")
    assert result == _Uploaded("default", ("a.jsonl", "b.jsonl"), None)


def test_uploaded_files_missing_split_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("datasets")
    _fake_builder(monkeypatch, _json_config())
    with pytest.raises(FileNotFoundError, match="available: \\['train'\\]"):
        resolve_mod._uploaded_files(_REPO, SOURCE, None, "test")


def test_uploaded_files_outside_the_commit_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("datasets")
    stray = {"train": [f"hf://datasets/{_REPO}@{'3' * 40}/a.jsonl"]}
    _fake_builder(monkeypatch, _json_config(splits=stray))
    with pytest.raises(RuntimeError, match="at a commit other than"):
        resolve_mod._uploaded_files(_REPO, SOURCE, None, "train")


def test_script_repos_and_empty_repos_are_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("datasets")
    from datasets.exceptions import DataFilesNotFoundError

    _fake_builder(
        monkeypatch, error=RuntimeError("Dataset scripts are no longer supported")
    )
    assert "scripts" in (
        resolve_mod._uploaded_files(_REPO, SOURCE, None, "train").reason or ""
    )

    _fake_builder(
        monkeypatch, error=DataFilesNotFoundError("No (supported) data files found")
    )
    assert "no data files" in (
        resolve_mod._uploaded_files(_REPO, SOURCE, None, "train").reason or ""
    )


def test_missing_repo_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("datasets")
    from datasets.exceptions import DatasetNotFoundError

    _fake_builder(monkeypatch, error=DatasetNotFoundError("nope"))
    with pytest.raises(DatasetNotFoundError):
        resolve_mod._uploaded_files(_REPO, SOURCE, None, "train")


def test_missing_datasets_package_is_a_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "datasets", None)
    result = resolve_mod._uploaded_files(_REPO, SOURCE, "cfg", "train")
    assert result == _Uploaded("cfg", (), "the `datasets` package is not installed")


def test_conversion_path_parses_resolve_urls() -> None:
    url = _item("0000.parquet", config="en", directory="partial-train")["url"]
    assert resolve_mod._conversion_path(_REPO, url) == "en/partial-train/0000.parquet"
    with pytest.raises(RuntimeError, match="Unexpected /parquet file URL"):
        resolve_mod._conversion_path(_REPO, "https://cdn.example.com/0000.parquet")


# ---------------------------------------------------------------------------
# Uploads that cannot back a split, configs, conversion layouts
# ---------------------------------------------------------------------------


def test_repeated_data_files_fall_back_to_the_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("datasets")
    splits = {"train": _pinned("a.jsonl", "a.jsonl")}  # overlapping globs
    _fake_builder(monkeypatch, _json_config(splits=splits))

    result = resolve(
        _conversion_hub(),
        parse_hf_uri("hf://org/repo/train"),
        None,
        allow_partial=False,
    )

    assert result.parts.source == SOURCE_PARQUET
    assert "more than once (e.g. a.jsonl)" in result.summary


def test_external_data_files_fall_back_to_the_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("datasets")
    splits = {"train": ["https://data.example.org/train.jsonl"]}
    _fake_builder(monkeypatch, _json_config(splits=splits))

    auto = resolve(
        _conversion_hub(),
        parse_hf_uri("hf://org/repo/train"),
        None,
        allow_partial=False,
    )
    assert auto.parts.source == SOURCE_PARQUET

    with pytest.raises(ValueError, match="live outside hf://datasets/org/repo"):
        resolve(
            _conversion_hub(),
            parse_hf_uri("hf://org/repo@~original/train"),
            None,
            allow_partial=False,
        )


def test_forced_parquet_uses_the_datasets_default_config(uploaded) -> None:
    seen = uploaded(_Uploaded("default", ("x.jsonl",), None))
    items = [
        _item("0000.parquet", config="default"),
        _item("0000.parquet", config="other"),
    ]
    hub = _FakeHub(
        export=_export(*items),
        trees={(CONVERSION, "default/train"): {"0000.parquet": 10}},
    )

    result = resolve(
        hub, parse_hf_uri("hf://org/repo@~parquet/train"), None, allow_partial=False
    )

    assert result.parts.config == "default"
    assert seen == [(_REPO, SOURCE, None, "train")]


def test_forced_parquet_with_a_config_skips_datasets(uploaded) -> None:
    seen = uploaded(_Uploaded("default", ("x.jsonl",), None))
    resolve(
        _conversion_hub(),
        parse_hf_uri("hf://org/repo@~parquet/default/train"),
        None,
        allow_partial=False,
    )
    assert seen == []


def test_broken_datasets_install_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def failing_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "datasets":
            raise ModuleNotFoundError("No module named 'pandas'", name="pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(ModuleNotFoundError, match="pandas"):
        resolve_mod._uploaded_files(_REPO, SOURCE, None, "train")


def test_unrelated_builder_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("datasets")
    _fake_builder(monkeypatch, error=RuntimeError("something else broke"))
    with pytest.raises(RuntimeError, match="something else broke"):
        resolve_mod._uploaded_files(_REPO, SOURCE, None, "train")


def test_successful_fallback_keeps_the_uploaded_reason(uploaded) -> None:
    uploaded(
        _Uploaded("default", ("data.jsonl",), "the split lists files more than once")
    )
    result = resolve(
        _conversion_hub(),
        parse_hf_uri("hf://org/repo/train"),
        None,
        allow_partial=False,
    )
    assert result.summary == (
        "2 files from HF's parquet conversion; "
        "uploaded files: the split lists files more than once"
    )


def _part_hub(directory_prefix: str = "") -> _FakeHub:
    """A conversion split across ``train-part0``/``train-part1``, reusing basenames."""
    directories = [f"{directory_prefix}train-part0", f"{directory_prefix}train-part1"]
    items = [
        _item("0000.parquet", size=i + 1, directory=d)
        for i, d in enumerate(directories)
    ]
    return _FakeHub(
        export=_export(*items),
        trees={
            (CONVERSION, f"default/{d}"): {"0000.parquet": i + 1}
            for i, d in enumerate(directories)
        },
    )


def test_conversion_split_across_part_directories(uploaded) -> None:
    uploaded(_Uploaded("default", ("x.csv",), None))
    hub = _part_hub()

    result = resolve(
        hub, parse_hf_uri("hf://org/repo/train"), None, allow_partial=False
    )

    assert result.parts.uri() == f"hf://org/repo@{CONVERSION}~parquet/default/train"
    assert result.files == {
        "default/train-part0/0000.parquet": 1,
        "default/train-part1/0000.parquet": 2,
    }
    assert list_frozen(hub, result.parts) == result.files


def test_partial_conversion_split_across_part_directories(uploaded) -> None:
    uploaded(_Uploaded("default", ("x.csv",), None))
    result = resolve(
        _part_hub("partial-"),
        parse_hf_uri("hf://org/repo/train"),
        None,
        allow_partial=True,
    )
    assert result.parts.split == "partial-train"
    assert len(result.files) == 2


def test_a_split_named_like_a_part_is_not_merged() -> None:
    hub = _FakeHub(
        trees={
            (CONVERSION, "default/train"): {"0000.parquet": 1},
            (CONVERSION, "default/train-part1"): {"0000.parquet": 2},  # its own split
        }
    )
    parts = parse_hf_uri(f"hf://org/repo@{CONVERSION}~parquet/default/train")
    assert list_frozen(hub, parts) == {"default/train/0000.parquet": 1}


# ---------------------------------------------------------------------------
# Hub-only builder and conversion provenance
# ---------------------------------------------------------------------------


def test_builder_is_loaded_from_the_hub_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare ``org/repo`` would load a same-named local directory instead."""
    datasets = pytest.importorskip("datasets")
    loaded = _fake_builder(monkeypatch, _json_config())
    resolve_mod._uploaded_files(_REPO, SOURCE, None, "train")
    assert loaded == [resolve_mod._builder_path(datasets.__version__, _REPO)]


@pytest.mark.parametrize("version", ["4.8.0", "4.10.1", "5.0.0.dev0"])
def test_builder_path_is_hub_only_from_4_8(
    version: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / _REPO).mkdir(parents=True)  # ignored by the hf://datasets/ form
    assert resolve_mod._builder_path(version, _REPO) == f"hf://datasets/{_REPO}"


@pytest.mark.parametrize("version", ["4.4.0", "4.7.0"])
def test_builder_path_is_bare_before_4_8(
    version: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``hf://datasets/`` paths don't reach the Hub before ``datasets`` 4.8."""
    monkeypatch.chdir(tmp_path)
    assert resolve_mod._builder_path(version, _REPO) == _REPO


def test_builder_path_refuses_a_shadowing_local_dir_before_4_8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / _REPO).mkdir(parents=True)
    with pytest.raises(RuntimeError, match="would shadow hf://org/repo"):
        resolve_mod._builder_path("4.7.0", _REPO)


def test_conversion_is_rejected_when_main_moves_while_pinning(uploaded) -> None:
    """Same paths and sizes can hide a conversion of a newer commit."""
    uploaded(_Uploaded("default", ("x.csv",), None))
    hub = _conversion_hub(main_heads=["3" * 40])

    with pytest.raises(ValueError, match="main moved to 3{40} while resolving"):
        resolve(hub, parse_hf_uri("hf://org/repo/train"), None, allow_partial=False)


def test_conversion_accepted_when_main_is_unchanged(uploaded) -> None:
    uploaded(_Uploaded("default", ("x.csv",), None))
    hub = _conversion_hub(main_heads=[SOURCE])
    result = resolve(
        hub, parse_hf_uri("hf://org/repo/train"), None, allow_partial=False
    )
    assert result.parts.source == SOURCE_PARQUET
    assert hub.calls[-1] == "commit_for:main"  # checked after the listing
