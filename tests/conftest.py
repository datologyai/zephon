import importlib.util
import json
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from tests._helpers import (
    DEEPSEEK_LIKE_NORMALIZED,
    DEEPSEEK_LIKE_SPECIALS,
    LLAMA3_LIKE_SPECIALS,
)


def _ensure_torch_shm_manager_is_executable() -> None:
    """Repair macOS torch SHM helper permissions for local test runs.

    Some macOS environments end up with ``torch/bin/torch_shm_manager``
    installed without execute bits, which breaks tests that exercise torch's
    shared-memory path. Fixing it here keeps the workaround scoped to tests.
    """

    if sys.platform != "darwin":
        return

    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return

    torch_dir = Path(spec.origin).resolve().parent
    manager = torch_dir / "bin" / "torch_shm_manager"
    if not manager.exists():
        return

    mode = manager.stat().st_mode
    execute_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    if mode & execute_bits:
        return

    manager.chmod(mode | execute_bits)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests marked as integration",
    )
    parser.addoption(
        "--run-packaging",
        action="store_true",
        default=False,
        help="Run tests marked as packaging (build + clean-venv install)",
    )
    parser.addoption(
        "--repro-iters",
        action="store",
        type=int,
        default=4,
        help="Number of reproducibility iterations for tests that repeat pipelines",
    )


def pytest_configure(config: pytest.Config) -> None:
    _ensure_torch_shm_manager_is_executable()
    config.addinivalue_line(
        "markers", "integration: mark tests as integration (skipped by default)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    skip_integration = (
        None
        if config.getoption("--run-integration")
        else pytest.mark.skip(reason="use --run-integration to run integration tests")
    )
    skip_packaging = (
        None
        if config.getoption("--run-packaging")
        else pytest.mark.skip(reason="use --run-packaging to run packaging tests")
    )
    for item in items:
        if skip_integration is not None and "integration" in item.keywords:
            item.add_marker(skip_integration)
        if skip_packaging is not None and "packaging" in item.keywords:
            item.add_marker(skip_packaging)


@pytest.fixture(scope="session")
def repro_iters(pytestconfig: pytest.Config) -> int:
    val = pytestconfig.getoption("--repro-iters")
    try:
        return int(val)
    except Exception:
        return 4


@pytest.fixture(autouse=True)
def _isolated_shard_catalogs(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
):
    """Isolate the shard-catalog dir and registry per test.

    ``Dataset.from_path`` + finalize/attach otherwise write content-addressed
    catalog artifacts into the *process-global* catalog dir (``$TMPDIR/zephon-
    {uid}/catalog`` or the developer's ``ZEPHON_CATALOG_DIR``), accumulating
    across runs, and the sticky ``_CATALOG_DIR``/``_REGISTRY`` globals leak
    state between tests. Point both at a per-test temp dir instead; the env var
    also propagates to spawned worker processes in integration tests. Tests
    that manage the dir themselves (tests/zephon/io/catalog) simply override.
    """
    from zephon._internal.io.catalog import clear_registry, set_catalog_dir

    catalog_dir = tmp_path_factory.mktemp("shard-catalogs")
    monkeypatch.setenv("ZEPHON_CATALOG_DIR", str(catalog_dir))
    set_catalog_dir(None)  # re-resolve the process-global dir from the env var
    clear_registry()
    yield
    clear_registry()


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


@pytest.fixture(scope="session")
def deepseek_like_tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
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
    root = tmp_path_factory.mktemp("deepseek_like_tokenizer")
    tok.save(str(root / "tokenizer.json"))
    bos, eos, _ = DEEPSEEK_LIKE_SPECIALS

    def added(content: str) -> dict[str, object]:
        return {"__type": "AddedToken", **_added_token(content, normalized=True)}

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "bos_token": added(bos),
        "eos_token": added(eos),
        "pad_token": added(eos),
        "clean_up_tokenization_spaces": False,
        "model_max_length": 1048576,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    (root / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def llama3_like_tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
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
    root = tmp_path_factory.mktemp("llama3_like_tokenizer")
    spec = json.loads(tok.to_str())
    spec["model"]["ignore_merges"] = True
    (root / "tokenizer.json").write_text(json.dumps(spec), encoding="utf-8")
    config = {
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
    }
    (root / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    special_map = {
        f"{k}_token": _added_token(t, normalized=False)
        for k, t in zip(("bos", "eos", "pad"), (bos, eos, pad))
    }
    (root / "special_tokens_map.json").write_text(
        json.dumps(special_map), encoding="utf-8"
    )
    return root
