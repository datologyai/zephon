# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer contract shared across pipeline ops and calibration.

Defines the minimal :class:`TokenizerLike` interface (a structural subset of a
Hugging Face tokenizer), the batch/output type aliases,
:func:`fallback_tokenizer` — a deterministic, dependency-free reference
implementation used when no tokenizer is configured and throughout the tests —
:func:`load_hf_tokenizer`, the retrying HF loader shared by the tokenize ops,
and :func:`installed_tokenizer_loader`, which serves ``TokenizeText`` with
:func:`load_v1_tokenizer` instead when ``tokenizers>=1`` is installed.

Chat helpers select assistant messages, map generated ranges to token masks,
and isolate HF's compiled-template caches between threads.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
import threading
import traceback
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
    Union,
    cast,
)

import numpy as np
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from zephon._internal.utils.thread_utils import suppress_library_threads
from zephon._internal.utils.torch_compat import _tensor_lock_ctx
from zephon.ops.config import LossScope

if TYPE_CHECKING:
    import tensorflow as tf
    import torch
    from transformers import BatchEncoding, PreTrainedTokenizerFast

TokenBatch: TypeAlias = Union[
    "np.ndarray",
    "torch.Tensor",
    "tf.Tensor",
    Sequence[int],
    Sequence[Sequence[int]],
]
TokenizerOutput: TypeAlias = Mapping[str, TokenBatch]


class TokenizerLike(Protocol):
    """Minimal interface used for tokenization."""

    name_or_path: str | None
    pad_token: int | str | None
    eos_token: int | str | None

    def __call__(
        self, texts: Sequence[str] | str, **kwargs: Any
    ) -> TokenizerOutput: ...


def encode_rendered_text(
    tokenizer: PreTrainedTokenizerFast,
    text: str,
    *,
    max_length: int | None = None,
    return_offsets_mapping: bool = False,
    add_special_tokens: bool = False,
    return_special_tokens_mask: bool = False,
) -> BatchEncoding:
    """Encode text, optionally letting the tokenizer postprocessor insert tokens.

    The optional special-token mask identifies tokens inserted by the tokenizer.
    Special tokens already present in the text count as input content.
    """
    kwargs: dict[str, Any] = {"add_special_tokens": add_special_tokens}
    if max_length is not None:
        kwargs.update(truncation=True, max_length=max_length)
    if return_offsets_mapping:
        kwargs["return_offsets_mapping"] = True
    if return_special_tokens_mask:
        kwargs["return_special_tokens_mask"] = True
    return tokenizer(text, **kwargs)


def thread_salted_template(template: str) -> str:
    """Return ``template`` plus a per-thread no-op salt expression.

    HF caches compiled chat templates process-wide by template text, and masked
    renders mutate AssistantTracker state on the cached object; the salt gives
    each thread its own template. An expression tag survives ``lstrip_blocks``
    untouched, and jinja's trailing-newline strip is replicated here, so the
    salted render is byte-identical.
    """
    for newline in ("\r\n", "\r", "\n"):
        if template.endswith(newline):
            template = template[: -len(newline)]
            break
    return template + f'{{{{ "zephon-render-slot-{threading.get_ident()}" and "" }}}}'


def selected_assistant_indices(
    messages: Sequence[Mapping[str, Any]], scope: LossScope
) -> set[int]:
    """Select raw assistant indices before rendering or truncation."""
    assistants = {
        i for i, message in enumerate(messages) if message.get("role") == "assistant"
    }
    if scope == "after_last_user":
        users = [
            i for i, message in enumerate(messages) if message.get("role") == "user"
        ]
        return {i for i in assistants if users and i > users[-1]}
    return assistants


def generation_mask(
    encoding: Any, spans: Sequence[tuple[int, int, int, int]], owners: set[int]
) -> np.ndarray:
    """Map selected generated ranges with HF's endpoint and early-break rules."""
    mask = np.zeros(len(encoding["input_ids"]), dtype=np.uint8)
    for owner, start, end, _ in spans:
        start_token = encoding.char_to_token(0, start)
        end_token = encoding.char_to_token(0, end - 1)
        if start_token is None:
            break
        if owner in owners:
            stop = end_token + 1 if end_token else len(mask)
            mask[start_token:stop] = 1
    return mask


def fallback_tokenizer() -> TokenizerLike:
    """Build a deterministic, dependency-free tokenizer.

    Reserves ``pad=0, bos=1, eos=2`` and hashes whitespace-split words to
    content ids, so it needs no model download. Used as the runtime fallback
    when no tokenizer is configured and as a stable stand-in in tests.
    """

    class _Tokenizer:
        name_or_path: str | None = "__fallback__"
        # Distinct ids so a bracketed sample reads naturally as
        # [bos=1, ..., eos=2] padded with pad=0.
        pad_token: int | str | None = 0
        pad_token_id: int | None = 0
        bos_token: int | str | None = 1
        bos_token_id: int | None = 1
        eos_token: int | str | None = 2
        eos_token_id: int | None = 2

        def __call__(
            self,
            texts: Sequence[str] | str,
            **kwargs: Any,
        ) -> TokenizerOutput:
            truncation = bool(kwargs.get("truncation", False))
            max_length = kwargs.get("max_length", None)
            padding = kwargs.get("padding", False)
            return_tensors = kwargs.get("return_tensors", None)

            max_len_int = int(max_length) if max_length is not None else None

            def encode_single(value: str) -> dict[str, list[int]]:
                # Offset content ids past the reserved range (pad=0, bos=1,
                # eos=2) so a hash collision can't masquerade as a special
                # token. PYTHONHASHSEED randomization across runs means
                # without the offset, content ids of 1 or 2 are possible and
                # tests like ``ids[0] != _BOS`` become flaky.
                tokens = [abs(hash(word)) % 10000 + 10 for word in value.split()]
                if truncation and max_len_int is not None:
                    tokens = tokens[:max_len_int]
                return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

            single_input = isinstance(texts, str)
            items = [texts] if single_input else list(texts)

            encoded = [encode_single(item) for item in items]
            input_ids: list[list[int]] = [item["input_ids"] for item in encoded]
            attention_masks: list[list[int]] = [
                item["attention_mask"] for item in encoded
            ]

            if padding:
                target = max(len(ids) for ids in input_ids) if input_ids else 0
                if padding == "max_length" and max_len_int is not None:
                    target = max_len_int

                pad_val = 0
                if isinstance(self.pad_token, int):
                    pad_val = self.pad_token
                elif self.pad_token is not None:
                    pad_val = 0

                def _pad(seq: list[int]) -> list[int]:
                    if len(seq) >= target:
                        return seq
                    return seq + [pad_val] * (target - len(seq))

                input_ids = [_pad(ids) for ids in input_ids]
                attention_masks = [_pad(mask) for mask in attention_masks]

            result: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
            }

            if single_input:
                result = {
                    "input_ids": input_ids[0] if input_ids else [],
                    "attention_mask": attention_masks[0] if attention_masks else [],
                }

            if return_tensors is None:
                return cast(TokenizerOutput, result)

            if return_tensors == "pt":
                try:  # pragma: no cover
                    import torch
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(
                        "return_tensors='pt' requested but torch is not available"
                    ) from exc
                with _tensor_lock_ctx():
                    return {key: torch.tensor(value) for key, value in result.items()}

            if return_tensors in ("np", "numpy"):
                try:  # pragma: no cover
                    import numpy as np
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(
                        "return_tensors='np' requested but numpy is not available"
                    ) from exc
                return {key: np.asarray(value) for key, value in result.items()}

            if return_tensors == "tf":
                try:  # pragma: no cover
                    import tensorflow as tf
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(
                        "return_tensors='tf' requested but tensorflow is not available"
                    ) from exc
                return {
                    key: tf.convert_to_tensor(value) for key, value in result.items()
                }

            raise ValueError(
                f"fallback tokenizer does not support return_tensors='{return_tensors}'"
            )

    return _Tokenizer()


# Transformers replaces its module with a lazy module during import. Concurrent
# cold imports can receive the original module without AutoTokenizer, even with
# the GIL enabled. Serialize imports, but keep tokenizer loading parallel below.
_IMPORT_LOCK = threading.Lock()


def load_hf_tokenizer(
    tokenizer_id: str | None,
    *,
    use_fast: bool | None = True,
    eos_token: str | None = None,
) -> TokenizerLike:
    """Load an HF tokenizer by id, with cloud resolution and retries.

    ``None`` / ``"__fallback__"`` return the dependency-free stub. Cloud URIs
    are resolved to a local cache path before loading. Transient load failures
    are retried with backoff; ``use_fast`` is dropped and the load retried once
    if the tokenizer class rejects the kwarg. ``eos_token`` is forwarded to
    ``from_pretrained``; HF derives ``eos_token_id`` from that string.
    """
    # Before importing hf tokenizers we tell it we handle the parallelism and
    # not hf tokenizers. This avoids unforeseen effects when running multiple
    # op instances.
    suppress_library_threads()

    if tokenizer_id in (None, "__fallback__"):
        if eos_token is not None:
            raise ValueError(
                "eos_token cannot be applied to the fallback tokenizer; "
                "pass a real tokenizer_id"
            )
        return fallback_tokenizer()

    # Deferred: tokenizer_cloud pulls the io/storage stack into this otherwise light module.
    from zephon._internal.utils.tokenizer_cloud import resolve_tokenizer_id_with_retry

    with _IMPORT_LOCK:
        from transformers import AutoTokenizer

    # Resolve before the HF retry — nesting would re-list the bucket on every
    # HF retry attempt.
    load_id = resolve_tokenizer_id_with_retry(tokenizer_id)

    @retry(
        wait=wait_random_exponential(multiplier=2, max=15),
        stop=stop_after_attempt(5),
        retry=retry_if_not_exception_type((TypeError, ValueError)),
        reraise=True,
    )
    def _load_with_retry(model_id: str | None, **k: Any) -> Any:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **k)
        except Exception as e:
            print(
                f"Error while instantiating tokenizer:\n\n{traceback.format_exc()}\n\n Will retry after some wait (unless this is the last iteration).",
                file=sys.stderr,
            )
            raise e
        return tokenizer

    kwargs: dict[str, Any] = {}
    if use_fast is not None:
        kwargs["use_fast"] = use_fast
    if eos_token is not None:
        kwargs["eos_token"] = eos_token
    try:
        return cast(TokenizerLike, _load_with_retry(load_id, **kwargs))
    except TypeError as exc:
        # Some tokenizers may not accept the `use_fast` kwarg; retry without
        # it so we surface the original failure instead of a signature
        # mismatch.
        if "use_fast" in kwargs and "use_fast" in str(exc):
            kwargs = dict(kwargs)
            kwargs.pop("use_fast", None)
            return cast(TokenizerLike, _load_with_retry(load_id, **kwargs))
        raise


class Refused(ValueError):
    """A tokenizer or call the tokenizers>=1 path cannot serve identically to HF."""


# The class AutoTokenizer (transformers 5.21) loads a folder with: TokenizersBackend
# runs tokenizer.json, the _REBUILT classes rebuild its pipeline, others are refused.
# Exhaustively checked against AutoTokenizer.
_GENERIC_CLASSES = set(
    "PreTrainedTokenizerFast PreTrainedTokenizer TokenizersBackend "
    "BloomTokenizerFast BloomTokenizer".split()
)
# Model types whose registered class overrides a generic hub class (None: refused).
_GENERIC_OVERRIDES: dict[str, str | None] = {"qwen2": "Qwen2Tokenizer"} | dict.fromkeys(
    "qwen3_5 qwen3_5_moe qwen3_5_moe_text qwen4_exp siglip2 hyperclovax_vision_v2".split()
)
# Model types registered to TokenizersBackend, which overrides any hub class.
_BACKEND_TYPES = set(
    "mistral mixtral deepseek_v2 deepseek_v3 olmo2 olmo3 phi3 granite".split()
)
_CLASS_TYPES = {
    "Qwen2Tokenizer": set(
        "qwen2 qwen2_5_omni qwen2_5_vl qwen2_audio qwen2_moe qwen2_vl qwen3 qwen3_asr "
        "qwen3_moe qwen3_next qwen3_omni_moe qwen3_vl qwen3_vl_moe".split()
    ),
    "GemmaTokenizer": set(
        "gemma gemma2 gemma3 gemma3_text gemma3n gemma3n_text".split()
    ),
}
_NAMED = tuple(f"{k}_token" for k in "bos eos unk sep pad cls mask".split())
_QWEN2_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _qwen2(spec: dict[str, Any], init: dict[str, Any]) -> dict[str, str | None]:
    """Qwen2Tokenizer.__init__'s pipeline; returns its default special tokens."""
    spec["normalizer"] = {"type": "NFC"}
    byte_level = {"type": "ByteLevel", "trim_offsets": True, "use_regex": False}
    split = {"type": "Split", "pattern": {"Regex": _QWEN2_SPLIT}}
    spec["pre_tokenizer"] = {
        "type": "Sequence",
        "pretokenizers": [
            {**split, "behavior": "Isolated", "invert": False},
            {**byte_level, "add_prefix_space": bool(init.get("add_prefix_space"))},
        ],
    }
    spec["model"] |= {"dropout": None, "unk_token": None, "fuse_unk": False}
    spec["model"] |= {"continuing_subword_prefix": "", "end_of_word_suffix": ""}
    spec["model"] |= {"byte_fallback": False, "ignore_merges": False}
    endoftext = "<|endoftext|>"
    return {"unk_token": endoftext, "eos_token": endoftext, "pad_token": endoftext}


def _gemma(spec: dict[str, Any], init: dict[str, Any]) -> dict[str, str | None]:
    """GemmaTokenizer.__init__'s pipeline; returns its default special tokens."""
    unk = init.get("unk_token", "<unk>")
    spec["normalizer"] = {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
    spec["pre_tokenizer"] = {"type": "Split", "pattern": {"String": " "}}
    spec["pre_tokenizer"] |= {"behavior": "MergedWithPrevious", "invert": False}
    # str(): GemmaTokenizer passes str(unk_token), so a null unk becomes "None".
    spec["model"] |= {
        "dropout": None,
        "unk_token": str(_content(unk)),
        "fuse_unk": True,
    }
    spec["model"] |= {"continuing_subword_prefix": None, "end_of_word_suffix": None}
    spec["model"] |= {"byte_fallback": True, "ignore_merges": False}
    names = ("unk", "bos", "eos", "pad", "mask")
    return {f"{n}_token": f"<{n}>" for n in names}


_REBUILT = {"Qwen2Tokenizer": _qwen2, "GemmaTokenizer": _gemma}
# Class attributes the rebuilt classes set over PreTrainedTokenizerBase's "right".
_PADDING_SIDE = {"GemmaTokenizer": "left"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _content(token: Any) -> Any:
    # HF stores a special token as a plain string or an AddedToken dict.
    return token["content"] if isinstance(token, dict) else token


def _side(name: str, value: Any) -> str:
    if value not in ("right", "left"):
        raise ValueError(
            f"{name} side should be selected between 'right' and 'left', "
            + f"current value: {value}"
        )
    return value


def _resolve_class(
    tc: dict[str, Any], config: dict[str, Any], root: Path
) -> str | None:
    hub = tc.get("tokenizer_class") or config.get("tokenizer_class")
    model_type = config.get("model_type")
    mistral_common = (root / "tekken.json").exists() and importlib.util.find_spec(
        "mistral_common"
    )
    if "auto_map" in tc or "model_name" in config or mistral_common:
        return None
    if hub in _GENERIC_CLASSES:
        return _GENERIC_OVERRIDES.get(model_type, "TokenizersBackend")
    if hub is not None and model_type in _BACKEND_TYPES:
        return "TokenizersBackend"
    return next(
        (
            cls
            for cls, types in _CLASS_TYPES.items()
            if model_type in types and hub in (None, cls, f"{cls}Fast")
        ),
        None,
    )


def _added_token(value: Any, special: bool) -> dict[str, Any]:
    """An AddedToken as HF builds it from a config str (``special`` given) or dict."""
    given = (
        {"content": value, "special": special}
        if isinstance(value, str)
        else {k: v for k, v in value.items() if k != "__type"}
    )
    is_special = given.get("special", False)
    defaults = {"single_word": False, "lstrip": False, "rstrip": False}
    return defaults | {"normalized": not is_special, "special": is_special} | given


def _add(
    added: list[dict[str, Any]], vocab: dict[str, int], token: dict[str, Any]
) -> None:
    """Mirror tokenizers' AddedVocabulary.add_tokens for one token on that list."""
    if not token["content"]:
        return
    for entry in added:
        if entry["content"] == token["content"]:
            # Properties are replaced; the special set only grows.
            entry |= token | {"special": entry["special"] or token["special"]}
            return
    if token["content"] in vocab:
        new_id = vocab[token["content"]]
    else:
        top = max((e["id"] for e in added), default=-1)
        new_id = top + 1 if top >= len(vocab) else len(vocab)
    added.append({"id": new_id} | token)


def _hf_tokenizer_json(
    root: Path, tc: dict[str, Any], eos_token: str | None
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """The tokenizer.json transformers 5.21 effectively runs for ``root``.

    Returns it with the named special tokens HF registers and the resolved class.
    """
    spec = _read_json(root / "tokenizer.json")
    config = _read_json(root / "config.json")
    cls = _resolve_class(tc, config, root)
    if cls is None:
        raise Refused(
            f"HF loads {root} (tokenizer_class="
            + f"{tc.get('tokenizer_class') or config.get('tokenizer_class')!r}, "
            + f"model_type={config.get('model_type')!r}) with a class this path "
            + "does not replicate"
        )
    # special_tokens_map.json counts only without added_tokens_decoder, its dicts
    # as special AddedTokens.
    decoder = tc.get("added_tokens_decoder")
    stm = {
        k: v | {"special": True}
        if isinstance(v, dict) and k != "extra_special_tokens"
        else v
        for k, v in _read_json(root / "special_tokens_map.json").items()
    }
    init = dict(tc) if decoder is not None else tc | stm
    # HF rejects a plain-dict special from tokenizer_config (special_tokens_map's
    # dicts become AddedTokens).
    if bad := [
        k
        for k in _NAMED
        if isinstance(init.get(k), dict)
        and init[k] is tc.get(k)
        and "__type" not in tc[k]
    ]:
        raise TypeError(f"special tokens {bad} must be strings or AddedToken objects")
    if "additional_special_tokens" in init and not init.get("extra_special_tokens"):
        init["extra_special_tokens"] = init.pop("additional_special_tokens")
    if eos_token is not None:
        init["eos_token"] = eos_token
    vocab = spec["model"]["vocab"]
    if not isinstance(vocab, dict):
        vocab = {piece: i for i, (piece, _) in enumerate(vocab)}

    # Declared added tokens: added_tokens_decoder, or else added_tokens.json then
    # tokenizer.json's added tokens.
    if decoder is not None:
        declared = {int(i): _added_token(t, False) for i, t in decoder.items()}
    else:
        extra = init.get("extra_special_tokens") or []
        named_contents = {_content(init[k]) for k in _NAMED if init.get(k)}
        named_contents |= {_content(t) for t in extra}
        declared = {
            i: _added_token(c, c in named_contents)
            | {"normalized": c not in named_contents}
            for c, i in _read_json(root / "added_tokens.json").items()
        }
        declared |= {
            t["id"]: {k: v for k, v in t.items() if k != "id"}
            for t in spec["added_tokens"]
        }

    defaults: dict[str, str | None] = {}
    if cls in _REBUILT:
        # The class builds a fresh backend: no added tokens of its own.
        spec["added_tokens"] = []
        defaults = _REBUILT[cls](spec, init)
    elif spec.get("padding"):
        init.setdefault("pad_token", spec["padding"]["pad_token"])

    named = {k: init.get(k, defaults.get(k)) for k in _NAMED}
    # Custom *_token keys: AddedToken-valued ones register before tokenizer_config's
    # strings ({**auto, **explicit}); a dict without "__type" is never registered.
    custom = {
        k: v
        for k, v in init.items()
        if k.endswith("_token")
        and k not in _NAMED
        and (
            isinstance(v, str) or (isinstance(v, dict) and ("__type" in v or k in stm))
        )
    }
    named |= {k: v for k, v in custom.items() if not isinstance(tc.get(k), str)}
    named |= {k: v for k, v in custom.items() if isinstance(tc.get(k), str)}
    extra = init.get("extra_special_tokens") or []
    if isinstance(extra, dict):
        named |= extra
        extra = []
    by_content = {t["content"]: t for t in declared.values()}
    named_tokens = {
        k: by_content.get(_content(v)) or _added_token(v, True)
        for k, v in named.items()
        if v is not None
    }
    for token in named_tokens.values():
        token["special"] = True

    # HF re-adds every declared token, then the named and extra ones it lacks.
    encoder = {t["content"] for t in spec["added_tokens"]} | by_content.keys()
    queue = [declared[i] for i in sorted(declared)]
    queue += [t for t in named_tokens.values() if t["content"] not in encoder]
    queue += [
        _added_token(t, isinstance(t, str)) for t in extra if _content(t) not in encoder
    ]
    if cls == "Qwen2Tokenizer":
        # Qwen2Tokenizer.__init__ ends by re-adding every special token as special.
        specials = [t["content"] for t in named_tokens.values()] + [
            _content(t) for t in extra
        ]
        queue += [_added_token(c, True) for c in specials]
    for token in queue:
        _add(spec["added_tokens"], vocab, token)
    return spec, {k: t["content"] for k, t in named_tokens.items()}, cls


class V1Tokenizer:
    """``TokenizerLike`` over a tokenizers 1.x ``Tokenizer``, identical to HF's.

    Builds the tokenizer.json HF effectively runs (its class's pipeline and the
    tokens HF adds) and returns what HF does for ``TokenizeText``'s calls, one
    ``encode`` per text (``encode_batch`` ignores ``RAYON_NUM_THREADS`` and
    spreads over every core). Tokenizers HF loads with a class this does not
    replicate raise :class:`Refused`.
    """

    def __init__(self, local_dir: str, eos_token: str | None = None) -> None:
        import tempfile

        from tokenizers import Tokenizer

        root = Path(local_dir)
        if not (root / "tokenizer.json").exists():
            raise Refused(f"{local_dir} has no tokenizer.json")
        tc = _read_json(root / "tokenizer_config.json")
        if unsupported := [
            k for k in ("fix_mistral_regex", "fast_tokenizer_files") if tc.get(k)
        ]:
            raise Refused(f"tokenizer_config.json sets {unsupported}")
        spec, special, cls = _hf_tokenizer_json(root, tc, eos_token)

        self._ids = {t["content"]: t["id"] for t in spec["added_tokens"]}
        self.bos_token: str | None = special.get("bos_token")
        self.eos_token: int | str | None = special.get("eos_token")
        self.pad_token: int | str | None = special.get("pad_token")
        truncation = spec.get("truncation") or {}
        padding = spec.get("padding") or {}
        self._truncation_side = _side(
            "Truncation",
            tc.get("truncation_side", truncation.get("direction", "Right").lower()),
        )
        self._padding_side = _side(
            "Padding",
            tc.get(
                "padding_side",
                padding.get("direction", _PADDING_SIDE.get(cls, "right")).lower(),
            ),
        )
        model_max_length = tc.get("model_max_length")
        # HF treats a length above 1e20 as unset.
        self._model_max_length = (
            model_max_length
            if isinstance(model_max_length, int) and model_max_length <= 10**20
            else None
        )
        self._split_special = bool(tc.get("split_special_tokens"))
        self._mask = "attention_mask" in tc.get("model_input_names", ["attention_mask"])
        self.name_or_path: str | None = local_dir
        self._eos_override = eos_token
        # tokenizers 1.x loads only from a file.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            self._tokenizer = Tokenizer.from_file(str(path))

    def _id(self, token: int | str | None) -> int | None:
        return None if token is None else self._ids[str(token)]

    @property
    def bos_token_id(self) -> int | None:
        return self._id(self.bos_token)

    @property
    def eos_token_id(self) -> int | None:
        return self._id(self.eos_token)

    @property
    def pad_token_id(self) -> int | None:
        return self._id(self.pad_token)

    def __reduce__(
        self,
    ) -> tuple[type[V1Tokenizer], tuple[str | None, str | None], dict[str, Any]]:
        # A tokenizers Tokenizer is not picklable; rebuild from the directory.
        state = {"pad_token": self.pad_token}
        return (V1Tokenizer, (self.name_or_path, self._eos_override), state)

    def __call__(
        self,
        texts: Sequence[str] | str,
        *,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> TokenizerOutput:
        from tokenizers import Truncation

        if kwargs or return_tensors not in (None, "np", "pt", "mlx"):
            raise Refused(
                f"the tokenizers>=1 path does not support {return_tensors=}, {kwargs}"
            )
        if truncation and max_length is None:
            max_length = self._model_max_length
        limit = (
            Truncation(max_length, direction=self._truncation_side)
            if truncation and max_length is not None
            else None
        )
        single = isinstance(texts, str)
        # padding=None and an explicit truncation override tokenizer.json's own
        # blocks, as HF's call options do.
        ids = [
            self._tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                encode_special_tokens=self._split_special,
                padding=None,
                truncation=limit,
            ).ids
            for text in ([texts] if single else texts)
        ]
        masks = [[1] * len(i) for i in ids]
        if padding not in (False, "do_not_pad"):
            pad = self.pad_token_id
            if pad is None:
                raise ValueError(
                    "Asking to pad but the tokenizer does not have a padding token."
                )
            width = (
                max_length if padding == "max_length" else max(map(len, ids), default=0)
            )
            assert width is not None
            left = self._padding_side == "left"
            ids = [
                [pad] * (width - len(i)) + i if left else i + [pad] * (width - len(i))
                for i in ids
            ]
            masks = [
                [0] * (width - len(m)) + m if left else m + [0] * (width - len(m))
                for m in masks
            ]
        out: dict[str, Any] = {"input_ids": ids, "attention_mask": masks}
        if not self._mask:
            del out["attention_mask"]
        if return_tensors is not None:
            return {k: _as_tensor(v, return_tensors) for k, v in out.items()}
        return {k: v[0] for k, v in out.items()} if single else out


def _as_tensor(rows: list[list[int]], return_tensors: str) -> Any:
    """Mirror HF's ``BatchEncoding.convert_to_tensors`` for "np", "pt" and "mlx"."""
    if return_tensors == "mlx":
        import mlx.core as mx

        return mx.array(rows, dtype=None if any(rows) else mx.int32)
    if return_tensors == "pt":
        import torch

        return torch.tensor(rows, dtype=torch.int64)
    if len({len(r) for r in rows}) > 1:
        return np.asarray([np.asarray(r) for r in rows], dtype=object)
    return np.asarray(rows, dtype=np.int64)


def load_v1_tokenizer(
    tokenizer_id: str | None,
    *,
    use_fast: bool | None = True,
    eos_token: str | None = None,
) -> TokenizerLike:
    """Load a :class:`V1Tokenizer` from a local folder, cloud URI or Hub repo.

    ``use_fast`` is accepted and ignored, as transformers 5 ignores it.
    """
    if tokenizer_id in (None, "__fallback__"):
        return load_hf_tokenizer(tokenizer_id, eos_token=eos_token)
    suppress_library_threads()

    from zephon._internal.utils.tokenizer_cloud import resolve_tokenizer_id_with_retry

    local_dir = cast(str, resolve_tokenizer_id_with_retry(tokenizer_id))
    if not Path(local_dir).is_dir():
        # The files AutoTokenizer reads, at one pinned revision; huggingface_hub
        # (a tokenizers dependency) honours the same token, cache and offline env.
        from huggingface_hub import snapshot_download

        files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        files += ["added_tokens.json", "config.json"]
        if importlib.util.find_spec("mistral_common"):
            files.append("tekken.json")
        download = retry(
            wait=wait_random_exponential(multiplier=2, max=15),
            stop=stop_after_attempt(5),
            retry=retry_if_not_exception_type((TypeError, ValueError)),
            reraise=True,
        )(snapshot_download)
        local_dir = download(local_dir, allow_patterns=files)
    return V1Tokenizer(local_dir, eos_token=eos_token)


def installed_tokenizer_loader() -> Callable[..., TokenizerLike]:
    """The v1 loader when tokenizers>=1 is installed, else the HF loader.

    Read from package metadata so the driver never imports the native library
    before workers fork.
    """
    try:
        version = importlib.metadata.version("tokenizers")
    except importlib.metadata.PackageNotFoundError:
        return load_hf_tokenizer
    return load_v1_tokenizer if int(version.split(".")[0]) >= 1 else load_hf_tokenizer


__all__ = [
    "Refused",
    "TokenBatch",
    "TokenizerLike",
    "TokenizerOutput",
    "V1Tokenizer",
    "fallback_tokenizer",
    "installed_tokenizer_loader",
    "load_hf_tokenizer",
    "load_v1_tokenizer",
]
