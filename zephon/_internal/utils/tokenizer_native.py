# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""The native v1 text adapter: saved pipeline, declared roles, ragged outputs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
)
_ENGINE_PID: int | None = None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _load_engine(source: bytes) -> tuple[Any, dict[str, Any]]:
    """Load the exact saved bytes and check rc3's added-token ID assignments."""
    global _ENGINE_PID
    from tokenizers import Tokenizer, __version__

    if _ENGINE_PID not in (None, os.getpid()) and __version__ in (
        "1.0.0rc3",
        "1.0.0-rc.3",
    ):
        raise ValueError(
            "Native tokenizers initialized before fork can deadlock; use spawn or forkserver"
        )
    _ENGINE_PID = os.getpid()
    spec = json.loads(source)
    with tempfile.TemporaryDirectory(prefix="zephon-tokenizer-") as tmp:
        path = Path(tmp) / "tokenizer.json"
        path.write_bytes(source)
        native = Tokenizer.from_file(path)
    # Released rc3 encodes directly; upstream's editable spec requires compile().
    compile_spec = getattr(native, "compile", None)
    engine = compile_spec() if compile_spec else native
    added = spec.get("added_tokens") or []
    encodings = engine.encode_batch(
        [t["content"] for t in added],
        add_special_tokens=False,
        encode_special_tokens=False,
        padding=None,
        truncation=None,
    )
    for token, encoding in zip(added, encodings):
        if encoding.ids != [token["id"]]:
            raise ValueError(
                f"tokenizers v1 changed the ID of {token['content']!r}: expected {[token['id']]}, got {encoding.ids}"
            )
    return engine, spec


class NativeTokenizer:
    """Run tokenizer.json without reconstructing Transformers model classes."""

    name_or_path: str | None
    _supports_numpy_rows = True

    def __init__(self, folder: str, eos_token: str | None = None) -> None:
        root = Path(folder)
        path = root / "tokenizer.json"
        if not path.is_file():
            raise ValueError(
                f"tokenizers v1 requires {path}; use a repository with tokenizer.json"
            )
        # Capture once: cloud cache replacement/deletion must not change a pickle.
        self._source = path.read_bytes()
        self._engine, spec = _load_engine(self._source)
        self._ids = {t["content"]: t["id"] for t in spec.get("added_tokens") or []}
        config = _read_json(root / "tokenizer_config.json")
        if "added_tokens_decoder" not in config:
            config |= _read_json(root / "special_tokens_map.json")
        for token_id, token in (config.get("added_tokens_decoder") or {}).items():
            if self._ids.get(token["content"]) != int(token_id):
                raise ValueError(
                    "tokenizer_config.json declares added tokens absent from tokenizer.json; save an up-to-date tokenizer.json"
                )
        roles = (spec.get("role_to_token") or {}) | config
        if eos_token is not None:
            roles["eos_token"] = eos_token
        padding = spec.get("padding") or {}
        if padding and "pad_token" not in roles:
            roles["pad_token"] = padding["pad_token"]
            self._ids.setdefault(padding["pad_token"], padding["pad_id"])
        model = _read_json(root / "config.json")
        by_id = {token_id: token for token, token_id in self._ids.items()}

        def role(key: str) -> str | None:
            token = roles.get(key)
            if isinstance(token, dict):
                token = token.get("content")
            if key not in roles:
                token_id = model.get(f"{key}_id")
                # Multiple generation stop IDs do not declare one tokenizer EOS.
                if key == "eos_token" and isinstance(token_id, list):
                    return None
                if token_id is not None:
                    token = by_id.get(token_id) if type(token_id) is int else None
                    if token is None:
                        raise ValueError(
                            f"v1 {key}_id={token_id!r} must identify an existing added token of tokenizer.json"
                        )
            if token is not None and (
                not isinstance(token, str) or token not in self._ids
            ):
                raise ValueError(
                    f"v1 {key}={token!r} must name an existing added token of tokenizer.json"
                )
            return token

        self.bos_token = role("bos_token")
        self.eos_token = role("eos_token")
        self.pad_token = role("pad_token")
        truncation = spec.get("truncation") or {}
        self._padding_side = config.get(
            "padding_side", padding.get("direction", "right").lower()
        )
        self._truncation_side = config.get(
            "truncation_side", truncation.get("direction", "right").lower()
        )
        length = config.get("model_max_length")
        self._max_length = (
            length if type(length) is int and 0 < length < 10**20 else None
        )
        self._split_special = bool(config.get("split_special_tokens", False))
        self.name_or_path = folder

    @property
    def bos_token_id(self) -> int | None:
        """Return the declared BOS ID."""
        return self._ids.get(self.bos_token)

    @property
    def eos_token_id(self) -> int | None:
        """Return the declared EOS ID."""
        return self._ids.get(self.eos_token)

    @property
    def pad_token_id(self) -> int | None:
        """Follow pad_token, including Zephon's EOS-as-padding fallback."""
        return self._ids.get(self.pad_token)

    def __getstate__(self) -> dict[str, Any]:
        # rc3's native pickle can change IDs; restore from our captured JSON.
        return {**vars(self), "_engine": None}

    def __setstate__(self, state: dict[str, Any]) -> None:
        vars(self).update(state)
        self._engine, _ = _load_engine(self._source)

    def __call__(
        self,
        texts: str | Sequence[str],
        *,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        """Return ragged lists or NumPy rows; the op owns tensor conversion."""
        from tokenizers import Padding, Truncation

        if padding not in (False, True, "longest", "max_length", "do_not_pad"):
            raise ValueError(f"Unsupported padding: {padding!r}")
        if return_tensors not in (None, "np"):
            raise ValueError(
                "Native tokenizer supports list or NumPy rows; use the op for other tensor backends"
            )
        length = self._max_length if max_length is None else max_length
        pad = None
        if padding and padding != "do_not_pad":
            if self.pad_token_id is None:
                raise ValueError(
                    "Padding requires a PAD token or an EOS token to use as padding"
                )
            if padding == "max_length" and length is None:
                raise ValueError("padding='max_length' requires max_length")
            pad = Padding(
                direction=self._padding_side,
                pad_id=self.pad_token_id,
                pad_token=self.pad_token,
                length=length if padding == "max_length" else None,
            )
        single = isinstance(texts, str)
        encodings = self._engine.encode_batch(
            [texts] if single else texts,
            add_special_tokens=add_special_tokens,
            encode_special_tokens=self._split_special,
            padding=pad,
            truncation=Truncation(length, direction=self._truncation_side)
            if truncation and length is not None
            else None,
        )
        if return_tensors:
            ids = [enc.ids_array.astype(np.int64) for enc in encodings]
            masks = [enc.attention_mask_array.astype(np.int64) for enc in encodings]
        else:
            ids = [enc.ids for enc in encodings]
            masks = [enc.attention_mask for enc in encodings]
        return {
            "input_ids": ids[0] if single else ids,
            "attention_mask": masks[0] if single else masks,
        }


def load_native_tokenizer(
    tokenizer_id: str,
    *,
    use_fast: bool | None = True,
    eos_token: str | None = None,
) -> NativeTokenizer:
    """Load resolved local files or the tokenizer files of a normal Hub ID."""
    if use_fast is False:
        raise ValueError("tokenizers v1 does not support use_fast=False")
    folder = tokenizer_id
    if not Path(folder).is_dir():
        from huggingface_hub import snapshot_download

        folder = snapshot_download(tokenizer_id, allow_patterns=list(_FILES))
    tokenizer = NativeTokenizer(folder, eos_token)
    tokenizer.name_or_path = tokenizer_id
    return tokenizer
