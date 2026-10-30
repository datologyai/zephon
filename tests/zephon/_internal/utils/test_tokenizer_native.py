# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Native tokenizers v1 loading, metadata, and encoding contract."""

from __future__ import annotations

import builtins
import json
import pickle
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pytest

try:
    _TOKENIZERS_MAJOR = int(version("tokenizers").split(".", 1)[0])
except (PackageNotFoundError, ValueError):
    _TOKENIZERS_MAJOR = 0

if _TOKENIZERS_MAJOR < 1:
    pytest.skip("requires tokenizers v1", allow_module_level=True)

pytest.importorskip("tokenizers")

from zephon._internal.ops.tokenize_text import TokenizeText
from zephon._internal.utils.tokenizer import load_text_tokenizer
from zephon._internal.utils.tokenizer_native import (
    NativeTokenizer,
    load_native_tokenizer,
)
from zephon.ops.base import OpContext
from zephon.ops.config import SpecialTokensMode
from zephon.types import SampleMeta, SampleRecord


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _update_json(path: Path, **changes: Any) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(changes)
    _write_json(path, value)


@pytest.fixture
def tokenizer_dir(tmp_path: Path) -> Path:
    """An independent serialized pipeline, with no Transformers construction."""
    vocabulary = {
        "<bos>": 0,
        "<eos>": 1,
        "<pad>": 2,
        "<unk>": 3,
        "hello": 4,
        "world": 5,
        "data": 6,
        "loader": 7,
        "!": 8,
    }
    added_tokens = [
        {
            "id": vocabulary[token],
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for token in ("<bos>", "<eos>", "<pad>")
    ]
    single = [
        {"SpecialToken": {"id": "<bos>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<eos>", "type_id": 0}},
    ]
    _write_json(
        tmp_path / "tokenizer.json",
        {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": added_tokens,
            "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": {
                "type": "TemplateProcessing",
                "single": single,
                "pair": [
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"Sequence": {"id": "B", "type_id": 1}},
                ],
                "special_tokens": {
                    token: {
                        "id": token,
                        "ids": [vocabulary[token]],
                        "tokens": [token],
                    }
                    for token in ("<bos>", "<eos>")
                },
            },
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": vocabulary,
                "unk_token": "<unk>",
            },
        },
    )
    _write_json(
        tmp_path / "tokenizer_config.json",
        {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"},
    )
    return tmp_path


def test_native_roles_and_single_or_ragged_batch(tokenizer_dir: Path) -> None:
    tok = NativeTokenizer(str(tokenizer_dir))
    assert (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id) == (0, 1, 2)
    assert tok("hello world", add_special_tokens=False) == {
        "input_ids": [4, 5],
        "attention_mask": [1, 1],
    }
    assert tok(["hello world", "hello", ""], add_special_tokens=False) == {
        "input_ids": [[4, 5], [4], []],
        "attention_mask": [[1, 1], [1], []],
    }
    assert tok("hello")["input_ids"] == [0, 4, 1]


def test_native_padding_truncation_and_sides(tokenizer_dir: Path) -> None:
    _update_json(
        tokenizer_dir / "tokenizer_config.json",
        padding_side="left",
        truncation_side="left",
    )
    tok = NativeTokenizer(str(tokenizer_dir))
    assert tok(["hello world", "hello"], add_special_tokens=False, padding=True) == {
        "input_ids": [[4, 5], [2, 4]],
        "attention_mask": [[1, 1], [0, 1]],
    }
    assert tok(
        "hello world data", add_special_tokens=False, truncation=True, max_length=2
    )["input_ids"] == [5, 6]


@pytest.mark.parametrize("eos_token", ["!", "<new-eos>"])
def test_native_eos_override_existing_or_new(
    tokenizer_dir: Path, eos_token: str
) -> None:
    tok = NativeTokenizer(str(tokenizer_dir), eos_token=eos_token)
    expected_id = 8 if eos_token == "!" else 9
    assert tok.eos_token == eos_token
    assert tok.eos_token_id == expected_id
    assert tok(eos_token, add_special_tokens=False)["input_ids"] == [expected_id]
    # Role overrides do not rewrite the serialized postprocessor.
    assert tok("hello")["input_ids"] == [0, 4, 1]


def test_native_role_precedence_and_added_token_objects(tokenizer_dir: Path) -> None:
    _update_json(
        tokenizer_dir / "tokenizer.json",
        role_to_token={"eos_token": "<bos>", "pad_token": "<eos>"},
    )
    _write_json(
        tokenizer_dir / "special_tokens_map.json",
        {"eos_token": "<pad>", "pad_token": "!"},
    )
    _write_json(
        tokenizer_dir / "tokenizer_config.json",
        {"eos_token": {"content": "<eos>", "special": True}},
    )
    tok = NativeTokenizer(str(tokenizer_dir))
    assert tok.eos_token_id == 1
    assert tok.pad_token_id == 8


def test_native_explicit_null_prevents_other_role_fallbacks(
    tokenizer_dir: Path,
) -> None:
    _update_json(tokenizer_dir / "tokenizer_config.json", eos_token=None)
    _write_json(tokenizer_dir / "special_tokens_map.json", {"eos_token": "<eos>"})
    _write_json(tokenizer_dir / "config.json", {"eos_token_id": 1})
    assert NativeTokenizer(str(tokenizer_dir)).eos_token_id is None


def test_native_model_config_scalar_fallback_ignores_multiple_eos(
    tokenizer_dir: Path,
) -> None:
    _write_json(tokenizer_dir / "tokenizer_config.json", {})
    _write_json(
        tokenizer_dir / "config.json",
        {"bos_token_id": 0, "eos_token_id": [1, 8], "pad_token_id": 2},
    )
    tok = NativeTokenizer(str(tokenizer_dir))
    assert tok.bos_token_id == 0
    assert tok.eos_token_id is None
    assert tok.pad_token_id == 2


def test_native_role_to_token_supplies_metadata(tokenizer_dir: Path) -> None:
    _write_json(tokenizer_dir / "tokenizer_config.json", {})
    _update_json(
        tokenizer_dir / "tokenizer.json",
        role_to_token={"bos_token": "<bos>", "eos_token": "<eos>"},
    )
    tok = NativeTokenizer(str(tokenizer_dir))
    assert (tok.bos_token_id, tok.eos_token_id) == (0, 1)


def test_native_missing_declared_role_is_rejected(tokenizer_dir: Path) -> None:
    _update_json(tokenizer_dir / "tokenizer_config.json", eos_token="<missing>")
    with pytest.raises(ValueError, match="eos_token"):
        NativeTokenizer(str(tokenizer_dir))


@pytest.mark.parametrize(("token_id", "content"), [(9, "<missing>"), (6, "data")])
def test_native_config_only_added_tokens_are_rejected(
    tokenizer_dir: Path, token_id: int, content: str
) -> None:
    _update_json(
        tokenizer_dir / "tokenizer_config.json",
        added_tokens_decoder={str(token_id): {"content": content, "special": True}},
    )
    with pytest.raises(ValueError, match="tokenizer_config.json"):
        NativeTokenizer(str(tokenizer_dir))


def test_native_rejects_added_token_id_renumbering(tokenizer_dir: Path) -> None:
    path = tokenizer_dir / "tokenizer.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["added_tokens"].append(
        {
            "id": 17,
            "content": "<gapped>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    )
    _write_json(path, spec)
    with pytest.raises(ValueError):
        NativeTokenizer(str(tokenizer_dir))


def test_native_ignores_saved_padding_and_truncation_when_disabled(
    tokenizer_dir: Path,
) -> None:
    _update_json(
        tokenizer_dir / "tokenizer.json",
        padding={
            "strategy": {"Fixed": 8},
            "direction": "Right",
            "pad_to_multiple_of": None,
            "pad_id": 2,
            "pad_type_id": 0,
            "pad_token": "<pad>",
        },
        truncation={
            "direction": "Right",
            "max_length": 1,
            "strategy": "LongestFirst",
            "stride": 0,
        },
    )
    tok = NativeTokenizer(str(tokenizer_dir))
    assert tok(["hello world data", ""], add_special_tokens=False) == {
        "input_ids": [[4, 5, 6], []],
        "attention_mask": [[1, 1, 1], []],
    }


def test_native_pickle_preserves_new_eos_and_mutable_pad(tokenizer_dir: Path) -> None:
    tok = NativeTokenizer(str(tokenizer_dir), eos_token="<new-eos>")
    tok.pad_token = tok.eos_token
    serialized = pickle.dumps(tok)
    for path in tokenizer_dir.iterdir():
        path.unlink()
    restored = pickle.loads(serialized)
    assert restored.eos_token_id == tok.eos_token_id == 9
    assert restored.pad_token_id == 9
    expected = {
        "input_ids": [[4, 5], [4, 9]],
        "attention_mask": [[1, 1], [1, 0]],
    }
    assert (
        restored(["hello world", "hello"], add_special_tokens=False, padding=True)
        == expected
    )
    restored_again = pickle.loads(pickle.dumps(restored))
    assert restored_again.pad_token_id == restored_again.eos_token_id == 9
    assert (
        restored_again(["hello world", "hello"], add_special_tokens=False, padding=True)
        == expected
    )


def test_native_loader_local_and_use_fast_false(tokenizer_dir: Path) -> None:
    assert load_native_tokenizer(str(tokenizer_dir)).eos_token_id == 1
    with pytest.raises(ValueError, match="use_fast"):
        load_native_tokenizer(str(tokenizer_dir), use_fast=False)


def test_native_loader_downloads_tokenizer_files(
    tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def download(repo_id: str, **kwargs: Any) -> str:
        calls.append((repo_id, kwargs))
        return str(tokenizer_dir)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    tok = load_native_tokenizer("example/fixture-tokenizer")
    assert tok.eos_token_id == 1
    assert calls[0][0] == "example/fixture-tokenizer"
    assert "tokenizer.json" in calls[0][1]["allow_patterns"]


def test_text_loader_resolves_cloud_without_transformers(
    tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved: list[str] = []

    def resolve(source: str) -> str:
        resolved.append(source)
        return str(tokenizer_dir)

    import_module = builtins.__import__

    def without_transformers(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".", 1)[0] == "transformers":
            raise AssertionError("native text tokenization imported Transformers")
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(
        "zephon._internal.utils.tokenizer_cloud.resolve_tokenizer_id_with_retry",
        resolve,
    )
    monkeypatch.setattr(
        "zephon._internal.utils.tokenizer.suppress_library_threads", lambda: None
    )
    monkeypatch.setattr(builtins, "__import__", without_transformers)
    tok = load_text_tokenizer("s3://bucket/tokenizer/")
    assert resolved == ["s3://bucket/tokenizer/"]
    assert tok("hello", add_special_tokens=False)["input_ids"] == [4]


def test_native_loader_missing_tokenizer_json_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tokenizer.json"):
        load_native_tokenizer(str(tmp_path))


def _record(text: str, index: int = 0) -> SampleRecord:
    return SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, index), lane_id=0, chunk_id=0),
        payload={"text": text},
    )


def _list(value: Any) -> list[int]:
    return value.tolist() if hasattr(value, "tolist") else list(value)


@pytest.mark.parametrize("special_tokens", ["bos_eos", "none", "tokenizer_default"])
@pytest.mark.parametrize("return_tensors", [None, "np", "pt"])
def test_native_loaded_op_ragged_backends_and_counter(
    tokenizer_dir: Path,
    special_tokens: SpecialTokensMode,
    return_tensors: str | None,
) -> None:
    if return_tensors == "pt":
        pytest.importorskip("torch")
    texts = ["hello world data", "hello", ""]
    expected = [[4, 5, 6], [4], []]
    if special_tokens != "none":
        expected = [[0, *ids, 1] for ids in expected]
    op = TokenizeText(
        tokenizer_id=str(tokenizer_dir),
        field="text",
        special_tokens=special_tokens,
        return_tensors=return_tensors,
    )
    counter = op.token_counting_spec().build_counter().plan([])
    op.setup(OpContext({}))
    assert op.tok is None
    records = op.process_many([_record(text, i) for i, text in enumerate(texts)])
    for record, text, ids in zip(records, texts, expected):
        assert isinstance(record.payload, dict)
        actual = record.payload["input_ids"]
        assert _list(actual) == ids
        assert _list(record.payload["attention_mask"]) == [1] * len(ids)
        assert (counter.count({"text": text}) or 0) == len(ids)
        if return_tensors is None:
            assert isinstance(actual, list)
        else:
            assert "int64" in str(actual.dtype)


@pytest.mark.parametrize("special_tokens", ["bos_eos", "tokenizer_default"])
@pytest.mark.parametrize("split", [False, True])
def test_native_loaded_op_truncate_split_padding_and_counter(
    tokenizer_dir: Path, special_tokens: SpecialTokensMode, split: bool
) -> None:
    text = "hello world data"
    op = TokenizeText(
        tokenizer_id=str(tokenizer_dir),
        field="text",
        special_tokens=special_tokens,
        max_length=3 if split else 4,
        truncation=not split,
        split_long_samples=split,
        padding="max_length",
        return_tensors="np",
    )
    with pytest.warns(UserWarning, match="padding=.*ignored"):
        counter = op.token_counting_spec().build_counter().plan([])
    op.setup(OpContext({}))
    records = op.process_one(_record(text))
    ids = [_list(record.payload["input_ids"]) for record in records]
    masks = [_list(record.payload["attention_mask"]) for record in records]
    assert ids == ([[0, 4, 5], [6, 1, 2]] if split else [[0, 4, 5, 1]])
    assert masks == ([[1, 1, 1], [1, 1, 0]] if split else [[1, 1, 1, 1]])
    assert counter.count({"text": text}) == sum(sum(mask) for mask in masks)
    assert [record.meta.lineage for record in records] == (
        [(0,), (1,)] if split else [()]
    )


def test_native_counter_honors_implicit_model_max_length(tokenizer_dir: Path) -> None:
    _update_json(tokenizer_dir / "tokenizer_config.json", model_max_length=3)
    op = TokenizeText(
        tokenizer_id=str(tokenizer_dir),
        field="text",
        special_tokens="tokenizer_default",
        truncation=True,
    )
    counter = op.token_counting_spec().build_counter().plan([])
    op.setup(OpContext({}))
    text = "hello world data"
    (record,) = op.process_one(_record(text))
    assert record.payload["input_ids"] == [0, 4, 1]
    assert counter.count({"text": text}) == len(record.payload["input_ids"]) == 3


@pytest.mark.parametrize("runner", ["threads", "process"])
def test_native_public_pipeline_and_preflight_leave_tokenizer_lazy(
    tokenizer_dir: Path, runner: str
) -> None:
    from zephon import Pipeline
    from zephon.io import Dataset, InMemoryShard
    from zephon.work import StaticMixtureWorkSource

    dataset = Dataset.from_dict(
        "native-text", {0: InMemoryShard([{"text": "hello world"}, {"text": ""}])}
    )
    source = StaticMixtureWorkSource([dataset], {dataset.name: 1.0}, chunk_size=2)
    pipeline = Pipeline(source).tokenize(
        tokenizer_id=str(tokenizer_dir), field="text", return_tensors="np"
    )
    pipeline = pipeline.options(runner=runner, max_workers=1, canonical_replicas=1)
    assert pipeline.preflight_tokenizers().ok
    (op,) = [
        node.op for node in pipeline._graph.nodes if isinstance(node.op, TokenizeText)
    ]
    assert op.tok is None
    assert not op._tokenizer_instantiated
    records = list(pipeline)
    assert [_list(record.payload["input_ids"]) for record in records] == [
        [0, 4, 5, 1],
        [0, 1],
    ]
