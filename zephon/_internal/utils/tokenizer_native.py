# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Bridge tokenizers v1's saved pipeline to Zephon's text-tokenizer interface.

The serialized pipeline is authoritative: this module does not reconstruct
Transformers tokenizer classes or their model-specific defaults.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _token_content(value: Any) -> Any:
    return value.get("content") if isinstance(value, dict) else value


class NativeTokenizer:
    """Execute tokenizer.json and expose only the interface used by text ops."""

    name_or_path: str | None

    def __init__(self, folder: str, eos_token: str | None = None) -> None:
        from tokenizers import Tokenizer

        root = Path(folder)
        source = root / "tokenizer.json"
        if not source.is_file():
            raise ValueError(
                f"tokenizers v1 requires {source}; use a repository containing "
                "tokenizer.json or install Transformers with tokenizers <1"
            )
        spec = _read_json(source)
        config = _read_json(root / "special_tokens_map.json")
        config.update(_read_json(root / "tokenizer_config.json"))
        roles = (spec.get("role_to_token") or {}) | config
        model_config = _read_json(root / "config.json")
        vocab = spec["model"]["vocab"]
        self._ids = (
            dict(vocab)
            if isinstance(vocab, dict)
            else {token: i for i, (token, _) in enumerate(vocab)}
        )
        added = spec.setdefault("added_tokens", [])
        self._ids.update({token["content"]: token["id"] for token in added})

        # Config-only additions need an updated pipeline, rather than silently
        # assigning different IDs from those expected by the model.
        declared_added = {token["content"]: token["id"] for token in added}
        for token_id, token in (config.get("added_tokens_decoder") or {}).items():
            if declared_added.get(token["content"]) != int(token_id):
                raise ValueError(
                    "tokenizer_config.json declares tokens absent from tokenizer.json; "
                    "save an up-to-date tokenizer.json before using tokenizers v1"
                )

        # The existing EOS override may introduce a token. v1 has no add_tokens
        # API, so append that one declaration to a temporary copy of the spec.
        if eos_token is not None and not any(t["content"] == eos_token for t in added):
            token_id = self._ids.get(eos_token, max(self._ids.values(), default=-1) + 1)
            added.append(
                dict(
                    id=token_id,
                    content=eos_token,
                    special=True,
                    normalized=False,
                    single_word=False,
                    lstrip=False,
                    rstrip=False,
                )
            )
            self._ids[eos_token] = token_id
            with tempfile.TemporaryDirectory(prefix="zephon-tokenizer-") as tmp:
                modified = Path(tmp) / "tokenizer.json"
                modified.write_text(json.dumps(spec), encoding="utf-8")
                native = Tokenizer.from_file(modified)
        else:
            native = Tokenizer.from_file(source)
        compile_spec = getattr(native, "compile", None)
        self._engine = compile_spec() if compile_spec is not None else native

        # rc3 can silently renumber added tokens (e.g. IDs with gaps). Verify
        # actual encoding, not decode spelling: Metaspace may change spelling.
        if added:
            encodings = self._engine.encode_batch(
                [token["content"] for token in added],
                add_special_tokens=False,
                encode_special_tokens=False,
                padding=None,
                truncation=None,
            )
            for token, encoding in zip(added, encodings):
                if encoding.ids != [token["id"]]:
                    raise ValueError(
                        f"tokenizers v1 changed the ID of {token['content']!r}: "
                        f"expected {[token['id']]}, got {encoding.ids}"
                    )

        def role(name: str) -> str | int | None:
            key = f"{name}_token"
            if key in roles:
                value = _token_content(roles[key])
            else:
                # Do not choose an arbitrary EOS from a generation-config list.
                value = model_config.get(f"{name}_token_id")
                if type(value) is not int:
                    value = None
            if value is not None and (
                (isinstance(value, str) and value not in self._ids)
                or (isinstance(value, int) and value not in self._ids.values())
                or (not isinstance(value, str) and type(value) is not int)
            ):
                raise ValueError(f"{key}={value!r} is absent from tokenizer.json")
            return value

        self.bos_token = role("bos")
        self.eos_token = eos_token if eos_token is not None else role("eos")
        self.pad_token = role("pad")
        if self.pad_token is None and "pad_token" not in roles and native.padding:
            # A saved padding ID need not have a vocabulary entry for its string.
            self.pad_token = native.padding.pad_id
        self._padding_side = config.get(
            "padding_side", native.padding.direction if native.padding else "right"
        )
        self._truncation_side = config.get(
            "truncation_side",
            native.truncation.direction if native.truncation else "right",
        )
        self._max_length = config.get("model_max_length")
        if not isinstance(self._max_length, int) or self._max_length > 10**20:
            self._max_length = None
        self._split_special = config.get("split_special_tokens", False)
        self._folder = str(root.resolve())
        self._files: dict[str, bytes] | None = None
        self._eos_override = eos_token
        self.name_or_path = folder
        # Do not retain a second copy of a potentially very large vocabulary.
        self._ids = {
            token: self._ids[token]
            for token in (self.bos_token, self.eos_token, self.pad_token)
            if isinstance(token, str)
        }

    @property
    def bos_token_id(self) -> int | None:
        """Return the declared BOS ID."""
        return self._token_id(self.bos_token)

    @property
    def eos_token_id(self) -> int | None:
        """Return the declared EOS ID."""
        return self._token_id(self.eos_token)

    @property
    def pad_token_id(self) -> int | None:
        """Resolve the current padding token, including Zephon's EOS fallback."""
        return self._token_id(self.pad_token)

    def _token_id(self, token: str | int | None) -> int | None:
        if token is None:
            return None
        return token if isinstance(token, int) else self._ids.get(token)

    def __reduce__(self) -> Any:
        # rc3's native pickle can renumber tokens. Carry the source files so
        # live tokenizers also work on workers without this machine's cache.
        files = (
            self._files
            if self._files is not None
            else {
                name: (Path(self._folder) / name).read_bytes()
                for name in _FILES
                if (Path(self._folder) / name).is_file()
            }
        )
        return (
            _restore_native_tokenizer,
            (files, self._eos_override),
            {"pad_token": self.pad_token, "name_or_path": self.name_or_path},
        )

    def __call__(
        self,
        texts: str | Sequence[str],
        *,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return ragged lists; Zephon owns conversion to tensor backends."""
        from tokenizers import Padding, Truncation

        if kwargs:
            raise TypeError(f"Unsupported tokenizer arguments: {sorted(kwargs)}")
        if padding not in (False, True, "longest", "max_length", "do_not_pad"):
            raise ValueError(f"Unsupported padding: {padding!r}")
        length = max_length if max_length is not None else self._max_length
        limit = (
            Truncation(length, direction=self._truncation_side)
            if truncation and length is not None
            else None
        )
        native_padding = None
        if padding not in (False, "do_not_pad"):
            if self.pad_token_id is None:
                raise ValueError(
                    "Padding requires a PAD token or an EOS token to use as padding"
                )
            if padding == "max_length" and length is None:
                raise ValueError("padding='max_length' requires max_length")
            native_padding = Padding(
                direction=self._padding_side,
                pad_id=self.pad_token_id,
                pad_token=str(self.pad_token),
                length=length if padding == "max_length" else None,
            )
        single = isinstance(texts, str)
        encodings = self._engine.encode_batch(
            [texts] if single else list(texts),
            add_special_tokens=add_special_tokens,
            encode_special_tokens=self._split_special,
            padding=native_padding,
            truncation=limit,
        )
        result = {
            "input_ids": [encoding.ids for encoding in encodings],
            "attention_mask": [encoding.attention_mask for encoding in encodings],
        }
        return {key: rows[0] for key, rows in result.items()} if single else result


def _restore_native_tokenizer(
    files: dict[str, bytes], eos_token: str | None
) -> NativeTokenizer:
    with tempfile.TemporaryDirectory(prefix="zephon-tokenizer-") as tmp:
        for name, data in files.items():
            (Path(tmp) / name).write_bytes(data)
        tokenizer = NativeTokenizer(tmp, eos_token)
    tokenizer._files = files
    return tokenizer


def load_native_tokenizer(
    tokenizer_id: str,
    *,
    use_fast: bool | None = True,
    eos_token: str | None = None,
) -> NativeTokenizer:
    """Resolve local/cloud/Hub files and load a native text tokenizer."""
    from zephon._internal.utils.thread_utils import suppress_library_threads
    from zephon._internal.utils.tokenizer_cloud import resolve_tokenizer_id_with_retry

    if use_fast is False:
        raise ValueError(
            "tokenizers v1 does not support use_fast=False; install tokenizers <1 for slow tokenizers"
        )
    suppress_library_threads()
    folder = resolve_tokenizer_id_with_retry(tokenizer_id)
    assert folder is not None
    if not Path(folder).is_dir():
        from huggingface_hub import snapshot_download

        download = retry(
            wait=wait_random_exponential(multiplier=2, max=15),
            stop=stop_after_attempt(5),
            retry=retry_if_not_exception_type((TypeError, ValueError)),
            reraise=True,
        )(snapshot_download)
        folder = download(
            tokenizer_id,
            allow_patterns=list(_FILES),
        )
    tokenizer = NativeTokenizer(folder, eos_token)
    tokenizer.name_or_path = tokenizer_id
    return tokenizer
