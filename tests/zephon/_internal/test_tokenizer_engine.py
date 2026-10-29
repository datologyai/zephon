"""TokenizeText through the installed tokenizer engine.

The main job runs this module with HF (tokenizers 0.x) and the tokenizers-v1 job
with tokenizers>=1: the ``ENGINE`` tests assert literals both must produce.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from zephon._internal.ops.tokenize_text import TokenizeText
from zephon._internal.utils.tokenizer import (
    Refused,
    installed_tokenizer_loader,
    load_v1_tokenizer,
)
from zephon.ops.base import OpContext, StageInfo
from zephon.types import SampleMeta, SampleRecord

# The free-threaded main job installs neither tokenizers nor transformers.
pytest.importorskip("tokenizers")

VOCAB = ["<s>", "</s>", "<pad>", "<unk>", "hello", "world", "data", "loader", "!"]
SPECIALS = VOCAB[:3]
SPECIAL_PROPS = {"single_word": False, "lstrip": False, "rstrip": False}
SPECIAL_PROPS |= {"normalized": False, "special": True}

# Read from the environment, not from the dispatch under test.
V1 = int(importlib.metadata.version("tokenizers").split(".")[0]) >= 1
ENGINE = pytest.mark.skipif(
    not V1 and importlib.util.find_spec("transformers") is None,
    reason="transformers is not installed",
)


# What tokenizer_default adds: <s> text </s>.
TEMPLATE = {
    "type": "TemplateProcessing",
    "single": [
        {"SpecialToken": {"id": "<s>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "</s>", "type_id": 0}},
    ],
    "pair": [
        {"Sequence": {"id": "A", "type_id": 0}},
        {"Sequence": {"id": "B", "type_id": 1}},
    ],
    "special_tokens": {
        t: {"id": t, "ids": [i], "tokens": [t]} for i, t in enumerate(SPECIALS[:2])
    },
}


def write_tokenizer(
    folder: Path, config: dict[str, Any] | None = None, **tokenizer_config: Any
) -> str:
    """A word-level tokenizer folder as HF saves one, specials ``<s> </s> <pad>``."""
    folder.mkdir(exist_ok=True)
    added = [{"id": i, "content": t, **SPECIAL_PROPS} for i, t in enumerate(SPECIALS)]
    model = {"type": "WordLevel", "vocab": {t: i for i, t in enumerate(VOCAB)}}
    spec = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": TEMPLATE,
        "decoder": None,
        "model": {**model, "unk_token": "<unk>"},
    }
    tc = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        **tokenizer_config,
    }
    (folder / "tokenizer.json").write_text(json.dumps(spec))
    (folder / "tokenizer_config.json").write_text(json.dumps(tc))
    if config is not None:
        (folder / "config.json").write_text(json.dumps(config))
    return str(folder)


def byte_alphabet() -> list[str]:
    """GPT-2's byte-to-unicode alphabet, which ByteLevel BPE vocabularies need."""
    keep = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1)]
    keep += range(ord("®"), ord("ÿ") + 1)
    rest = len([b for b in range(256) if b not in keep])
    return [chr(b) for b in keep] + [chr(256 + i) for i in range(rest)]


def write_bpe(
    folder: Path, model_type: str, tokens: list[str], merges: list[tuple[str, str]]
) -> str:
    """A BPE folder whose tokenizer.json pipeline is deliberately not its class's."""
    folder.mkdir()
    model = {"type": "BPE", "vocab": {t: i for i, t in enumerate(tokens)}}
    spec = {
        "version": "1.0",
        "added_tokens": [],
        "pre_tokenizer": {"type": "Whitespace"},
        "model": {**model, "merges": [list(m) for m in merges]},
    }
    (folder / "tokenizer.json").write_text(json.dumps(spec))
    (folder / "tokenizer_config.json").write_text("{}")
    (folder / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(folder)


def records(folder: str, texts: list[str], **op_kwargs: Any) -> list[dict[str, Any]]:
    op = TokenizeText(tokenizer_id=folder, field="text", **op_kwargs)
    op.setup(
        OpContext(
            {"record_node_metrics": lambda *a, **k: None},
            StageInfo(stage_index=0, stage_name="stage0", op_index=0),
        )
    )
    batch = [
        SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, i), lane_id=0, chunk_id=0),
            payload={"text": t},
        )
        for i, t in enumerate(texts)
    ]
    return [rec.payload for rec in op.process_many(batch)]


TEXTS = ["hello world", "hello </s> data loader!", ""]


@ENGINE
@pytest.mark.parametrize(
    "op_kwargs,tokenizer_config,expected",
    [
        (
            {"truncation": True, "max_length": 4},
            {},
            [[0, 4, 5, 1], [0, 4, 1, 1], [0, 1]],
        ),
        (
            {"truncation": True, "max_length": 4},
            {"truncation_side": "left"},
            [[0, 4, 5, 1], [0, 7, 8, 1], [0, 1]],
        ),
        ({"special_tokens": "none"}, {}, [[4, 5], [4, 1, 6, 7, 8], []]),
        (
            {"special_tokens": "none"},
            {"split_special_tokens": True},
            [[4, 5], [4, 3, 3, 3, 6, 7, 8], []],
        ),
    ],
    ids=["bos_eos_truncated", "truncation_side_left", "none", "split_special"],
)
def test_records_are_identical_on_either_engine(
    tmp_path: Path,
    op_kwargs: dict[str, Any],
    tokenizer_config: dict[str, Any],
    expected: list[list[int]],
) -> None:
    folder = write_tokenizer(tmp_path / "tok", **tokenizer_config)
    got = records(folder, TEXTS, **op_kwargs)
    assert got == [
        {"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in expected
    ]


PADDED = [[0, 4, 5, 6, 7, 1], [0, 4, 1, 2, 2, 2]]


@ENGINE
@pytest.mark.parametrize(
    "op_kwargs,tokenizer_config,expected,mask",
    [
        ({"padding": True}, {}, PADDED, [[1] * 6, [1, 1, 1, 0, 0, 0]]),
        (
            {"padding": True},
            {"padding_side": "left"},
            [PADDED[0], [2, 2, 2, 0, 4, 1]],
            [[1] * 6, [0, 0, 0, 1, 1, 1]],
        ),
        ({"truncation": True, "max_length": 4}, {}, [[0, 4, 5, 1], [0, 4, 1]], None),
        ({"truncation": True}, {"model_max_length": 3}, [[0, 4, 1]] * 2, None),
        (
            {"padding": "max_length", "max_length": 6, "return_tensors": "np"},
            {},
            PADDED,
            [[1] * 6, [1, 1, 1, 0, 0, 0]],
        ),
    ],
    ids=["longest", "padding_side_left", "truncated", "model_max_length", "np"],
)
def test_tokenizer_default_records_are_identical_on_either_engine(
    tmp_path: Path,
    op_kwargs: dict[str, Any],
    tokenizer_config: dict[str, Any],
    expected: list[list[int]],
    mask: list[list[int]] | None,
) -> None:
    folder = write_tokenizer(tmp_path / "tok", **tokenizer_config)
    texts = ["hello world data loader", "hello"]
    got = records(folder, texts, special_tokens="tokenizer_default", **op_kwargs)
    masks = mask or [[1] * len(ids) for ids in expected]
    np_out = op_kwargs.get("return_tensors") == "np"
    for rec, ids, m in zip(got, expected, masks):
        assert {k: type(v) for k, v in rec.items()} == dict.fromkeys(
            ("input_ids", "attention_mask"), np.ndarray if np_out else list
        )
        assert [
            np.asarray(rec["input_ids"]).tolist(),
            np.asarray(rec["attention_mask"]).tolist(),
        ] == [ids, m]
        if np_out:
            assert rec["input_ids"].dtype == np.int64


@ENGINE
def test_mlx_tensors_are_identical_on_either_engine(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    tok = installed_tokenizer_loader()(write_tokenizer(tmp_path / "tok"))
    out = tok(["hello world", "hello"], padding=True, return_tensors="mlx")
    assert {k: (v.dtype, v.tolist()) for k, v in out.items()} == {
        "input_ids": (mx.int32, [[0, 4, 5, 1], [0, 4, 1, 2]]),
        "attention_mask": (mx.int32, [[1, 1, 1, 1], [1, 1, 1, 0]]),
    }


@ENGINE
def test_model_input_names_without_mask_emit_none(tmp_path: Path) -> None:
    folder = write_tokenizer(tmp_path / "tok", model_input_names=["input_ids"])
    got = records(folder, ["hello"], special_tokens="none")
    assert got == [{"input_ids": [4]}]


@ENGINE
@pytest.mark.parametrize("key", ["truncation_side", "padding_side"])
@pytest.mark.parametrize("side", [None, "Left"])
def test_invalid_side_raises(tmp_path: Path, key: str, side: str | None) -> None:
    folder = write_tokenizer(tmp_path / "tok", **{key: side})
    with pytest.raises(ValueError, match="side should be selected between"):
        installed_tokenizer_loader()(folder)


@ENGINE
@pytest.mark.parametrize(
    "tokenizer_config,eos_token,text,expected",
    [
        ({"bos_token": "hello"}, None, "hello world", [4, 1, 2, [4, 5]]),
        (
            {"additional_special_tokens": ["<extra>"]},
            None,
            "data<extra>loader",
            [0, 1, 2, [6, 9, 7]],
        ),
        ({}, "<new>", "hello<new>", [0, 9, 2, [4, 9]]),
        (
            {
                "added_tokens_decoder": {
                    str(i): {"content": t, "special": True, "single_word": t == "</s>"}
                    for i, t in enumerate(SPECIALS)
                }
            },
            None,
            "hello</s> </s> x",
            [0, 1, 2, [4, 3, 3, 3, 1, 3]],
        ),
    ],
    ids=["bos_vocab_word", "extra_special", "eos_override_new", "single_word_flip"],
)
def test_tokens_hf_adds_are_identical_on_either_engine(
    tmp_path: Path,
    tokenizer_config: dict[str, Any],
    eos_token: str | None,
    text: str,
    expected: list[Any],
) -> None:
    folder = write_tokenizer(tmp_path / "tok", **tokenizer_config)
    tok = installed_tokenizer_loader()(folder, eos_token=eos_token)
    ids = tok([text], add_special_tokens=False)["input_ids"][0]
    assert [tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, ids] == expected


@ENGINE
def test_rebuilt_classes_are_identical_on_either_engine(tmp_path: Path) -> None:
    qwen = write_bpe(
        tmp_path / "qwen",
        "qwen2",
        ["<|endoftext|>", *byte_alphabet(), "he", "hel", "Ġw"],
        [("h", "e"), ("he", "l"), ("Ġ", "w")],
    )
    # Qwen2Tokenizer re-adds its specials with default properties: rstrip is reset.
    rstrip = {"content": "<|endoftext|>", "special": True, "rstrip": True}
    (tmp_path / "qwen" / "tokenizer_config.json").write_text(
        json.dumps({"added_tokens_decoder": {"0": rstrip}})
    )
    bytes_ = [f"<0x{b:02X}>" for b in range(256)]
    gemma = write_bpe(
        tmp_path / "gemma",
        "gemma3",
        [
            "<pad>",
            "<eos>",
            "<bos>",
            "<unk>",
            "<mask>",
            *bytes_,
            "▁",
            *"helow",
            "he",
            "▁w",
        ],
        [("h", "e"), ("▁", "w")],
    )
    got = {}
    for name, folder in (("qwen", qwen), ("gemma", gemma)):
        tok = installed_tokenizer_loader()(folder)
        texts = ["hello world", "hé", "<|endoftext|> he"]
        ids = tok(texts, add_special_tokens=False)["input_ids"]
        padded = tok(["he", "hello"], add_special_tokens=False, padding=True)
        got[name] = [tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, ids, padded]
    assert got == {
        "qwen": [
            None,
            0,
            0,
            [[258, 76, 79, 259, 79, 82, 76, 68], [72, 128, 103], [0, 221, 257]],
            {
                "input_ids": [[257, 0, 0], [258, 76, 79]],
                "attention_mask": [[1, 0, 0], [1] * 3],
            },
        ],
        "gemma": [
            2,
            1,
            0,
            [
                [267, 264, 264, 265, 268, 265, 119, 264, 105],
                [262, 200, 174],
                [
                    65,
                    129,
                    263,
                    115,
                    105,
                    265,
                    107,
                    121,
                    263,
                    125,
                    121,
                    129,
                    67,
                    261,
                    267,
                ],
            ],
            {
                "input_ids": [[0, 0, 0, 267], [267, 264, 264, 265]],
                "attention_mask": [[0, 0, 0, 1], [1] * 4],
            },
        ],
    }


@ENGINE
def test_rebuilt_class_takes_padding_side_from_tokenizer_json(tmp_path: Path) -> None:
    # tokenizer.json's padding direction overrides GemmaTokenizer's "left".
    bytes_ = [f"<0x{b:02X}>" for b in range(256)]
    folder = write_bpe(
        tmp_path / "gemma",
        "gemma3",
        ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", *bytes_, "▁", *"helow", "he"],
        [("h", "e")],
    )
    spec = json.loads((tmp_path / "gemma" / "tokenizer.json").read_text())
    spec["padding"] = {"strategy": "BatchLongest", "direction": "Right", "pad_id": 0}
    spec["padding"] |= {
        "pad_to_multiple_of": None,
        "pad_type_id": 0,
        "pad_token": "<pad>",
    }
    (tmp_path / "gemma" / "tokenizer.json").write_text(json.dumps(spec))
    tok = installed_tokenizer_loader()(folder)
    out = tok(["he", "hello"], add_special_tokens=False, padding=True)
    assert out["input_ids"] == [[267, 0, 0, 0], [267, 264, 264, 265]]


@ENGINE
def test_special_token_given_as_plain_dict_raises(tmp_path: Path) -> None:
    folder = write_tokenizer(tmp_path / "tok", bos_token={"content": "<s>"})
    with pytest.raises(TypeError):
        installed_tokenizer_loader()(folder)


@ENGINE
def test_special_token_ids_and_eos_override(tmp_path: Path) -> None:
    folder = write_tokenizer(tmp_path / "tok")
    load = installed_tokenizer_loader()
    tok = load(folder)
    tok.pad_token = "</s>"
    assert pickle.loads(pickle.dumps(tok)).pad_token_id == 1
    tok = load(folder)
    overridden = pickle.loads(pickle.dumps(load(folder, eos_token="<pad>")))
    assert [tok.bos_token_id, tok.eos_token_id, tok.pad_token_id] == [0, 1, 2]
    assert overridden.eos_token_id == 2
    assert overridden(["hello </s>"], add_special_tokens=False)["input_ids"] == [[4, 1]]


@ENGINE
@pytest.mark.parametrize("with_decoder,bos_id", [(False, 2), (True, 0)])
def test_special_tokens_map_counts_only_without_added_tokens_decoder(
    tmp_path: Path, with_decoder: bool, bos_id: int
) -> None:
    decoder = {str(i): {"content": t, **SPECIAL_PROPS} for i, t in enumerate(SPECIALS)}
    folder = write_tokenizer(
        tmp_path / "tok", **({"added_tokens_decoder": decoder} if with_decoder else {})
    )
    (tmp_path / "tok" / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": {"content": "<pad>"}})
    )
    assert installed_tokenizer_loader()(folder).bos_token_id == bos_id


def _install_tokenizers(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    real = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: version if name == "tokenizers" else real(name),
    )


@pytest.mark.parametrize(
    "version,engine", [("0.23.3", "hf"), ("1.0.0-rc.3", "v1"), ("1.2.0", "v1")]
)
def test_engine_is_chosen_where_the_tokenizer_loads(
    monkeypatch: pytest.MonkeyPatch, version: str, engine: str
) -> None:
    for name in ("hf", "v1"):
        monkeypatch.setattr(
            f"zephon._internal.utils.tokenizer.load_{name}_tokenizer",
            lambda *a, name=name, **k: name,
        )
    _install_tokenizers(monkeypatch, "0.23.3" if engine == "v1" else "1.0.0")
    op = TokenizeText(tokenizer_id="some/dir", field="text")
    _install_tokenizers(monkeypatch, version)
    counter_tokenizer = op.token_counting_spec().build_counter().tokenizer
    assert [op._load_tokenizer(), counter_tokenizer] == [engine, engine]


def test_v1_resolves_hub_ids_and_falls_back_like_hf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = write_tokenizer(tmp_path / "tok")
    calls: list[tuple[str, list[str]]] = []

    def snapshot_download(repo_id: str, allow_patterns: list[str]) -> str:
        calls.append((repo_id, allow_patterns))
        return folder

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    tok = load_v1_tokenizer("org/name")
    assert [c[0] for c in calls] == ["org/name"]
    assert {"tokenizer.json", "tokenizer_config.json", "config.json"} <= set(
        calls[0][1]
    )
    assert tok.name_or_path == folder
    assert type(load_v1_tokenizer(None)).__name__ == "_Tokenizer"


@pytest.mark.parametrize(
    "config,tokenizer_config,refused",
    [
        ({"model_type": "siglip2"}, {}, "does not replicate"),
        (
            {"model_type": "llama"},
            {"tokenizer_class": "LlamaTokenizerFast"},
            "does not replicate",
        ),
        ({}, {"tokenizer_class": None}, "does not replicate"),
        ({}, {"auto_map": {"AutoTokenizer": ["x.T", None]}}, "does not replicate"),
        ({}, {"fix_mistral_regex": True}, "fix_mistral_regex"),
    ],
    ids=[
        "generic_class_overridden",
        "model_class",
        "no_class",
        "remote_code",
        "fix_mistral_regex",
    ],
)
def test_v1_refuses_what_hf_would_change(
    tmp_path: Path,
    config: dict[str, Any],
    tokenizer_config: dict[str, Any],
    refused: str,
) -> None:
    folder = write_tokenizer(tmp_path / "tok", config, **tokenizer_config)
    with pytest.raises(Refused, match=refused):
        load_v1_tokenizer(folder)


def test_v1_refuses_a_folder_without_tokenizer_json(tmp_path: Path) -> None:
    with pytest.raises(Refused, match="no tokenizer.json"):
        load_v1_tokenizer(str(tmp_path))


@pytest.mark.skipif(not V1, reason="needs tokenizers>=1")
def test_v1_serves_backend_types_whatever_the_hub_class(tmp_path: Path) -> None:
    folder = write_tokenizer(
        tmp_path / "tok",
        {"model_type": "deepseek_v3"},
        tokenizer_class="LlamaTokenizerFast",
    )
    assert load_v1_tokenizer(folder)(["hello"], add_special_tokens=False) == {
        "input_ids": [[4]],
        "attention_mask": [[1]],
    }


@pytest.mark.skipif(not V1, reason="needs tokenizers>=1")
@pytest.mark.parametrize(
    "kwargs",
    [{"return_tensors": "tf"}, {"return_offsets_mapping": True}],
    ids=["tf", "other"],
)
def test_v1_refuses_call_options(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    tok = load_v1_tokenizer(write_tokenizer(tmp_path / "tok"))
    with pytest.raises(Refused):
        tok(["hello"], **kwargs)
