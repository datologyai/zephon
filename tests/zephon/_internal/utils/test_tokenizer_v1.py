# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tokenizers v1 loading, metadata, and encoding contract."""

from __future__ import annotations

import builtins
import inspect
import json
import os
import pickle
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.tokenizers import write_wordlevel_tokenizer

try:
    _TOKENIZERS_MAJOR = int(version("tokenizers").split(".", 1)[0])
except (PackageNotFoundError, ValueError):
    _TOKENIZERS_MAJOR = 0

if _TOKENIZERS_MAJOR < 1:
    pytest.skip("requires tokenizers v1", allow_module_level=True)

pytest.importorskip("tokenizers")

from zephon._internal.utils.tokenizer_v1 import (
    TokenizerV1,
    load_v1_tokenizer,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _update_json(path: Path, **changes: Any) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(changes)
    _write_json(path, value)


@pytest.fixture
def tokenizer_dir(tmp_path: Path) -> Path:
    return write_wordlevel_tokenizer(tmp_path)


def test_v1_workaround_is_still_needed(tokenizer_dir: Path) -> None:
    try:
        version("transformers")
    except PackageNotFoundError:
        pytest.skip("requires real Transformers alongside tokenizers v1")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        output = tok("hello world")
    except (ImportError, AttributeError, TypeError):
        return  # Transformers still rejects v1 or requires its removed APIs.
    assert output["input_ids"] == [0, 4, 5, 1]
    assert tok.eos_token_id == 1
    pytest.fail(
        "Transformers now loads and tokenizes correctly with tokenizers v1; "
        "remove the v1 workaround and its compatibility test"
    )


def _transformers_reference() -> None:
    """Build saved tokenizers and reference outputs in an isolated v0 interpreter."""
    import json
    import sys
    from importlib.metadata import version
    from pathlib import Path

    from tokenizers import (
        AddedToken,
        Tokenizer,
        models,
        normalizers,
        pre_tokenizers,
        processors,
        trainers,
    )
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    assert int(version("tokenizers").split(".", 1)[0]) == 0, "reference requires v0"
    root, family = Path(sys.argv[1]), sys.argv[2]
    cases = json.load(sys.stdin)
    tokenizer = Tokenizer(getattr(models, family)(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = (
        pre_tokenizers.ByteLevel(add_prefix_space=False)
        if family == "BPE"
        else pre_tokenizers.Whitespace()
    )
    trainer = getattr(trainers, ("Bpe" if family == "BPE" else family) + "Trainer")(
        vocab_size=320,
        special_tokens=["<bos>", "<eos>", "<pad>", "<unk>"],
        **(
            {"initial_alphabet": pre_tokenizers.ByteLevel.alphabet()}
            if family == "BPE"
            else {}
        ),
    )
    tokenizer.train_from_iterator(
        ["hello world data loader ! café Ελληνικά 中文"], trainer=trainer
    )
    tokenizer.add_tokens(
        [
            AddedToken("  ", normalized=False),
            AddedToken("<tag>", lstrip=True, rstrip=True),
        ]
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>", special_tokens=[("<bos>", 0), ("<eos>", 1)]
    )
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        model_max_length=8,
    )
    results = {}
    for side in ("left", "right"):
        wrapped.padding_side = wrapped.truncation_side = side
        folder = root / side
        wrapped.save_pretrained(folder)
        loaded = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        results[side] = {
            "roles": [loaded.bos_token_id, loaded.eos_token_id, loaded.pad_token_id],
            "outputs": [dict(loaded(texts, **kwargs)) for texts, kwargs in cases],
        }
    print(json.dumps(results))


@pytest.mark.parametrize("family", ["BPE", "WordPiece", "WordLevel"])
def test_v1_matches_transformers_saved_tokenizers(tmp_path: Path, family: str) -> None:
    """Compare supported saved configurations, without model-specific class defaults."""
    reference_python = os.environ.get("TRANSFORMERS_REFERENCE_PYTHON")
    if not reference_python:
        pytest.skip(
            "set TRANSFORMERS_REFERENCE_PYTHON to a Transformers/v0 interpreter"
        )
    texts = ["hello world", "", "hello  world", "cafe\u0301 中文", "hello <tag> world"]
    cases = [
        ("hello <eos> world", {"add_special_tokens": False}),
        (texts, {"add_special_tokens": False}),
        (texts, {"padding": True}),
        (texts, {"padding": "max_length", "max_length": 12}),
        (["hello world data loader ! " * 4, "hello"], {"truncation": True}),
        (texts, {"truncation": True, "max_length": 5, "padding": "max_length"}),
    ]
    # Execute only the helper's source so the v0 environment needs no Zephon or
    # pytest dependencies. Both loaders read the same files without patched APIs.
    script = inspect.getsource(_transformers_reference) + "\n_transformers_reference()"
    reference = subprocess.run(
        [reference_python, "-c", script, str(tmp_path), family],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert reference.returncode == 0, reference.stderr
    for side, expected in json.loads(reference.stdout).items():
        tok = load_v1_tokenizer(str(tmp_path / side))
        assert [tok.bos_token_id, tok.eos_token_id, tok.pad_token_id] == expected[
            "roles"
        ]
        for (texts, kwargs), output in zip(cases, expected["outputs"]):
            # Transformers also returns token_type_ids, which Zephon does not emit.
            assert tok(texts, **kwargs) == {
                key: output[key] for key in ("input_ids", "attention_mask")
            }, (family, side, texts, kwargs)


def test_v1_roles_and_single_or_ragged_batch(tokenizer_dir: Path) -> None:
    tok = TokenizerV1(str(tokenizer_dir))
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


def test_v1_added_tokens_preserve_whitespace_and_normalization(
    tokenizer_dir: Path,
) -> None:
    path = tokenizer_dir / "tokenizer.json"
    spec = json.loads(path.read_text())
    spec["normalizer"] = {"type": "NFC"}
    for token_id, (content, normalized, strip) in enumerate(
        [
            (" ", False, False),
            ("  ", False, False),
            ("\t\n", False, False),
            ("é", True, False),
            ("<strip>", False, True),
        ],
        start=9,
    ):
        spec["added_tokens"].append(
            {
                "id": token_id,
                "content": content,
                "single_word": False,
                "lstrip": strip,
                "rstrip": strip,
                "normalized": normalized,
                "special": False,
            }
        )
    _write_json(path, spec)
    tok = TokenizerV1(str(tokenizer_dir))
    assert tok(
        ["hello  world", "hello\t\nworld", "e\u0301", "hello\t<strip>\tworld"],
        add_special_tokens=False,
    )["input_ids"] == [[4, 10, 5], [4, 11, 5], [12], [4, 13, 5]]


@pytest.mark.parametrize("single", [False, True])
def test_v1_numpy_rows_match_ragged_lists(tokenizer_dir: Path, single: bool) -> None:
    import numpy as np

    tok = TokenizerV1(str(tokenizer_dir))
    texts = "" if single else ["hello world", "hello", ""]
    expected = tok(texts, add_special_tokens=False)
    actual = tok(texts, add_special_tokens=False, return_numpy=True)
    for field in ("input_ids", "attention_mask"):
        rows = [actual[field]] if single else actual[field]
        expected_rows = [expected[field]] if single else expected[field]
        for row, expected_row in zip(rows, expected_rows):
            assert isinstance(row, np.ndarray)
            assert row.dtype == np.uint32
            assert row.shape == (len(expected_row),)
            assert row.tolist() == expected_row


def test_v1_padding_truncation_and_sides(tokenizer_dir: Path) -> None:
    _update_json(
        tokenizer_dir / "tokenizer_config.json",
        padding_side="left",
        truncation_side="left",
    )
    tok = TokenizerV1(str(tokenizer_dir))
    assert tok(["hello world", "hello"], add_special_tokens=False, padding=True) == {
        "input_ids": [[4, 5], [2, 4]],
        "attention_mask": [[1, 1], [0, 1]],
    }
    assert tok(
        "hello world data", add_special_tokens=False, truncation=True, max_length=2
    )["input_ids"] == [5, 6]


def test_v1_eos_override_uses_existing_added_token(tokenizer_dir: Path) -> None:
    tok = TokenizerV1(str(tokenizer_dir), eos_token="<pad>")
    assert (tok.eos_token, tok.eos_token_id) == ("<pad>", 2)
    assert tok("hello <pad> world", add_special_tokens=False)["input_ids"] == [4, 2, 5]
    # Role overrides do not rewrite the serialized postprocessor.
    assert tok("hello")["input_ids"] == [0, 4, 1]


@pytest.mark.parametrize(
    ("model_max_length", "max_length", "expected"), [(3, None, 3), (1, 1, 2)]
)
def test_v1_implicit_limit_and_mandatory_specials(
    tokenizer_dir: Path, model_max_length: int, max_length: int | None, expected: int
) -> None:
    _update_json(
        tokenizer_dir / "tokenizer_config.json", model_max_length=model_max_length
    )
    tok = TokenizerV1(str(tokenizer_dir))
    ids = tok("hello world data", truncation=True, max_length=max_length)["input_ids"]
    assert len(ids) == expected


@pytest.mark.parametrize("eos_token", ["!", "<new-eos>"])
def test_v1_eos_override_requires_existing_added_token(
    tokenizer_dir: Path, eos_token: str
) -> None:
    with pytest.raises(ValueError, match="existing added token"):
        TokenizerV1(str(tokenizer_dir), eos_token=eos_token)


@pytest.mark.parametrize("modern", [False, True])
def test_v1_role_precedence_matches_transformers(
    tokenizer_dir: Path, modern: bool
) -> None:
    _write_json(
        tokenizer_dir / "special_tokens_map.json",
        {"eos_token": {"content": "<pad>", "special": True}, "pad_token": "<bos>"},
    )
    config: dict[str, Any] = {
        "bos_token": "<eos>",
        "eos_token": "<eos>",
        "pad_token": None,
    }
    if modern:
        config["added_tokens_decoder"] = {"2": {"content": "<pad>", "special": True}}
    _write_json(tokenizer_dir / "tokenizer_config.json", config)
    tok = TokenizerV1(str(tokenizer_dir))
    assert tok.bos_token_id == 1
    # Legacy maps override config; modern files ignore the map entirely.
    assert (tok.eos_token_id, tok.pad_token_id) == ((1, None) if modern else (2, 0))


def test_v1_explicit_null_prevents_other_role_fallbacks(
    tokenizer_dir: Path,
) -> None:
    _update_json(
        tokenizer_dir / "tokenizer_config.json", eos_token=None, added_tokens_decoder={}
    )
    _write_json(tokenizer_dir / "special_tokens_map.json", {"eos_token": "<eos>"})
    _write_json(tokenizer_dir / "config.json", {"eos_token_id": 1})
    assert TokenizerV1(str(tokenizer_dir)).eos_token_id is None


def test_v1_model_config_scalar_fallback_ignores_multiple_eos(
    tokenizer_dir: Path,
) -> None:
    _write_json(tokenizer_dir / "tokenizer_config.json", {})
    _write_json(
        tokenizer_dir / "config.json",
        {"bos_token_id": 0, "eos_token_id": [1, 8], "pad_token_id": 2},
    )
    tok = TokenizerV1(str(tokenizer_dir))
    assert tok.bos_token_id == 0
    assert tok.eos_token_id is None
    assert tok.pad_token_id == 2


@pytest.mark.parametrize("pad_id", [99, 4])
def test_v1_scalar_pad_requires_existing_added_id(
    tokenizer_dir: Path, pad_id: int
) -> None:
    _write_json(tokenizer_dir / "tokenizer_config.json", {"eos_token": "<eos>"})
    _write_json(tokenizer_dir / "config.json", {"pad_token_id": pad_id})
    # An unmapped ID must not silently become None and fall back to EOS padding.
    with pytest.raises(ValueError, match="pad_token_id.*existing added token"):
        TokenizerV1(str(tokenizer_dir))


def test_v1_role_metadata_follows_tokenizer_schema(tokenizer_dir: Path) -> None:
    from tokenizers import Tokenizer

    _write_json(tokenizer_dir / "tokenizer_config.json", {})
    _update_json(
        tokenizer_dir / "tokenizer.json",
        role_to_token={"bos_token": "<bos>", "eos_token": "<eos>"},
    )
    # rc3 accepts this optional metadata; the newer editable-spec API rejects it.
    # Preserve that tokenizer error rather than silently rewriting the saved pipeline.
    try:
        Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    except ValueError:
        with pytest.raises(ValueError, match="unknown keys.*role_to_token"):
            TokenizerV1(str(tokenizer_dir))
    else:
        tok = TokenizerV1(str(tokenizer_dir))
        assert (tok.bos_token_id, tok.eos_token_id) == (0, 1)


def test_v1_missing_declared_role_is_rejected(tokenizer_dir: Path) -> None:
    _update_json(tokenizer_dir / "tokenizer_config.json", eos_token="<missing>")
    with pytest.raises(ValueError, match="eos_token"):
        TokenizerV1(str(tokenizer_dir))


@pytest.mark.parametrize(("token_id", "content"), [(9, "<missing>"), (6, "data")])
def test_v1_config_only_added_tokens_are_rejected(
    tokenizer_dir: Path, token_id: int, content: str
) -> None:
    _update_json(
        tokenizer_dir / "tokenizer_config.json",
        added_tokens_decoder={str(token_id): {"content": content, "special": True}},
    )
    with pytest.raises(ValueError, match="tokenizer_config.json"):
        TokenizerV1(str(tokenizer_dir))


def test_v1_rejects_added_token_id_renumbering(tokenizer_dir: Path) -> None:
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
    with pytest.raises(ValueError, match="changed the ID of '<gapped>'"):
        TokenizerV1(str(tokenizer_dir))


def test_v1_compiles_editable_specs_at_load_and_restore(
    tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tokenizers

    spec = tokenizers.Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    tokenizer = spec.compile() if hasattr(spec, "compile") else spec
    compiled: list[bool] = []

    class EditableSpec:
        def compile(self) -> Any:
            compiled.append(True)
            return tokenizer

    class Tokenizer:
        @staticmethod
        def from_file(path: Any) -> EditableSpec:
            return EditableSpec()

    monkeypatch.setattr(tokenizers, "Tokenizer", Tokenizer)
    tok = pickle.loads(pickle.dumps(TokenizerV1(str(tokenizer_dir))))
    assert compiled == [True, True]
    assert tok("hello world", add_special_tokens=False)["input_ids"] == [4, 5]


def test_v1_guards_only_versions_with_upstream_fork_bug(
    tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zephon._internal.utils import tokenizer_v1

    monkeypatch.setattr(tokenizer_v1, "_TOKENIZER_PID", os.getpid() + 1)
    if version("tokenizers") in {"1.0.0rc1", "1.0.0rc2", "1.0.0rc3"}:
        with pytest.raises(ValueError, match="spawn or forkserver"):
            TokenizerV1(str(tokenizer_dir))
    else:
        assert TokenizerV1(str(tokenizer_dir)).eos_token_id == 1


def test_v1_ignores_saved_padding_and_truncation_when_disabled(
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
    tok = TokenizerV1(str(tokenizer_dir))
    assert tok(["hello world data", ""], add_special_tokens=False) == {
        "input_ids": [[4, 5, 6], []],
        "attention_mask": [[1, 1, 1], []],
    }


@pytest.mark.parametrize("cache_change", ["replace", "delete"])
def test_v1_pickle_preserves_loaded_snapshot_and_mutable_pad(
    tokenizer_dir: Path, cache_change: str
) -> None:
    tok = TokenizerV1(str(tokenizer_dir), eos_token="<bos>")
    tok.pad_token = tok.eos_token
    if cache_change == "replace":
        path = tokenizer_dir / "tokenizer.json"
        replacement = json.loads(path.read_text(encoding="utf-8"))
        replacement["model"]["vocab"].update(hello=5, world=4)
        _write_json(path, replacement)
    else:
        for path in tokenizer_dir.iterdir():
            path.unlink()
    # Capture must happen at loading, before a cloud cache changes or disappears.
    serialized = pickle.dumps(tok)
    restored = pickle.loads(serialized)
    assert restored.eos_token_id == tok.eos_token_id == 0
    assert restored.pad_token_id == 0
    expected = {
        "input_ids": [[4, 5], [4, 0]],
        "attention_mask": [[1, 1], [1, 0]],
    }
    assert (
        restored(["hello world", "hello"], add_special_tokens=False, padding=True)
        == expected
    )
    restored_again = pickle.loads(pickle.dumps(restored))
    assert restored_again.pad_token_id == restored_again.eos_token_id == 0
    assert (
        restored_again(["hello world", "hello"], add_special_tokens=False, padding=True)
        == expected
    )


def test_v1_loader_rejects_use_fast_false(tokenizer_dir: Path) -> None:
    with pytest.raises(ValueError, match="use_fast"):
        load_v1_tokenizer(str(tokenizer_dir), use_fast=False)


def test_v1_loader_only_requires_hub_dependency_for_hub_ids(
    tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__

    def without_hub(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] == "huggingface_hub":
            raise ImportError("No module named 'huggingface_hub'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_hub)
    assert load_v1_tokenizer(str(tokenizer_dir)).eos_token_id == 1
    with pytest.raises(ImportError, match="huggingface_hub.*local tokenizer directory"):
        load_v1_tokenizer("example/fixture-tokenizer")


def test_v1_loader_downloads_tokenizer_files(
    tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def download(repo_id: str, **kwargs: Any) -> str:
        calls.append((repo_id, kwargs))
        return str(tokenizer_dir)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    tok = load_v1_tokenizer("example/fixture-tokenizer")
    assert tok.eos_token_id == 1
    assert calls[0][0] == "example/fixture-tokenizer"
    assert "tokenizer.json" in calls[0][1]["allow_patterns"]


def test_v1_loader_missing_tokenizer_json_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tokenizer.json"):
        load_v1_tokenizer(str(tmp_path))
