"""Build the committed tokenizer folders and record HF's outputs on them.

Run in the main test environment (tokenizers 0.x + transformers), from the
repository root::

    uv run python -m tests.helpers.generate_tokenizer_reference

The folders are trained with tokenizers 0.x's training API, which 1.x removed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.helpers.tokenizer_reference import (
    DEEPSEEK_LIKE_NORMALIZED,
    DEEPSEEK_LIKE_SPECIALS,
    DIRS,
    LLAMA3_LIKE_SPECIALS,
    REFERENCE_PATH,
    compute,
)

# Split regexes copied from the real tokenizer.json files as JSON, so every
# character matches: the tokenizers 1.0 engine without ``fancy-regex`` rejects
# the escaped spellings (``\"``, ``\r``) of DeepSeek's pattern.
_DEEPSEEK_SPLITS: list[str] = json.loads(
    r"""[
    "\\p{N}{1,3}",
    "[一-龥぀-ゟ゠-ヿ]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
]"""
)
_LLAMA3_SPLITS: list[str] = json.loads(
    r"""[
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
]"""
)


def _byte_level_bpe(splits: list[str], trainer_specials: tuple[str, ...]) -> Any:
    """A byte-level BPE trained on a small multilingual corpus."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(Regex(p), "isolated") for p in splits]
        + [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]
    )
    tok.decoder = decoders.ByteLevel()
    corpus = [
        "The quick brown fox jumps over the lazy dog, again and again.",
        "Tokenizers split text into pieces; numbers like 12345 split in threes.",
        "It's what they'll say: we've done it, I'm sure, you'd agree.",
        "数据加载器 日本語のテキスト カタカナ emoji 😀👍🏽 and café naïve.",
        "def main():\n    return {'key': [1, 2, 3]}  # code\r\n",
    ] * 50
    tok.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=600,
            special_tokens=list(trainer_specials),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    return tok


def _added_token(content: str, *, normalized: bool) -> dict[str, object]:
    return {
        "content": content,
        "lstrip": False,
        "normalized": normalized,
        "rstrip": False,
        "single_word": False,
    }


def _dumps(value: Any, depth: int = 0) -> str:
    """JSON with objects expanded one key per line and anything else on one line."""
    if not isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    pad = " " * (depth + 1)
    items = ",\n".join(
        f"{pad}{json.dumps(k, ensure_ascii=False)}: {_dumps(v, depth + 1)}"
        for k, v in value.items()
    )
    return "{\n" + items + "\n" + " " * depth + "}"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_dumps(value) + "\n", "utf-8")


def build_deepseek_like(root: Path) -> None:
    """A tokenizer folder with DeepSeek V4's structure and a small trained vocab.

    Same pre-tokenizer, specials at ids 0-2, ``normalized`` added tokens, and a
    ``tokenizer_config.json`` (no ``special_tokens_map.json``) that stores
    tokens as AddedToken dicts with pad set to the EOS token.
    """
    from tokenizers import AddedToken, processors

    tok = _byte_level_bpe(_DEEPSEEK_SPLITS, DEEPSEEK_LIKE_SPECIALS)
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.add_tokens(
        [
            AddedToken(t, normalized=True, special=False)
            for t in DEEPSEEK_LIKE_NORMALIZED
        ]
    )
    _write_json(root / "tokenizer.json", json.loads(tok.to_str()))
    bos, eos, _ = DEEPSEEK_LIKE_SPECIALS

    def added(content: str) -> dict[str, object]:
        return {"__type": "AddedToken", **_added_token(content, normalized=True)}

    _write_json(
        root / "tokenizer_config.json",
        {
            "add_bos_token": False,
            "add_eos_token": False,
            "bos_token": added(bos),
            "eos_token": added(eos),
            "pad_token": added(eos),
            "clean_up_tokenization_spaces": False,
            "model_max_length": 1048576,
            "tokenizer_class": "PreTrainedTokenizerFast",
        },
    )


def build_llama3_like(root: Path) -> None:
    """A tokenizer folder with Llama-3.2-1B's structure and a small trained vocab.

    Same pre-tokenizer, ``ignore_merges``, BOS template, specials appended after
    the BPE vocab, and both ``tokenizer_config.json`` (plain strings) and
    ``special_tokens_map.json`` (AddedToken dicts) naming bos/eos/pad.
    """
    from tokenizers import AddedToken, processors

    tok = _byte_level_bpe(_LLAMA3_SPLITS, ())
    tok.add_special_tokens(
        [AddedToken(t, normalized=False, special=True) for t in LLAMA3_LIKE_SPECIALS]
    )
    bos, eos, pad = LLAMA3_LIKE_SPECIALS[:3]
    tok.post_processor = processors.Sequence(
        [
            processors.ByteLevel(trim_offsets=False),
            processors.TemplateProcessing(
                single=f"{bos} $A",
                pair=f"{bos} $A {bos}:1 $B:1",
                special_tokens=[(bos, tok.token_to_id(bos))],
            ),
        ]
    )
    spec = json.loads(tok.to_str())
    spec["model"]["ignore_merges"] = True
    _write_json(root / "tokenizer.json", spec)
    _write_json(
        root / "tokenizer_config.json",
        {
            "add_bos_token": True,
            "bos_token": bos,
            "clean_up_tokenization_spaces": True,
            "eos_token": eos,
            "model_input_names": ["input_ids", "attention_mask"],
            "model_max_length": 131072,
            "pad_token": pad,
            "padding_side": "left",
            "tokenizer_class": "PreTrainedTokenizer",
            "unk_token": None,
        },
    )
    _write_json(
        root / "special_tokens_map.json",
        {
            f"{k}_token": _added_token(t, normalized=False)
            for k, t in zip(("bos", "eos", "pad"), (bos, eos, pad))
        },
    )


def main() -> None:
    for family, build in (
        ("deepseek", build_deepseek_like),
        ("llama3", build_llama3_like),
    ):
        DIRS[family].mkdir(parents=True, exist_ok=True)
        build(DIRS[family])
    _write_json(REFERENCE_PATH, compute("hf"))


if __name__ == "__main__":
    main()
