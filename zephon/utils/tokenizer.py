# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer contract shared across pipeline ops and calibration.

Defines the minimal :class:`TokenizerLike` interface (a structural subset of a
Hugging Face tokenizer), the batch/output type aliases,
:func:`fallback_tokenizer` — a deterministic, dependency-free reference
implementation used when no tokenizer is configured and throughout the tests —
and :func:`load_hf_tokenizer`, the retrying HF loader shared by the tokenize
ops.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import traceback
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
    Union,
    cast,
)

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from zephon.utils.thread_utils import suppress_library_threads
from zephon.utils.torch_compat import _gil_disabled, _tensor_lock_ctx

if TYPE_CHECKING:
    import numpy as np
    import tensorflow as tf
    import torch

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


# Only needed for free-threaded builds (e.g., CPython 3.13t/3.14t) where the GIL is absent.
_IMPORT_LOCK: threading.Lock | None = threading.Lock() if _gil_disabled() else None


def load_hf_tokenizer(
    tokenizer_id: str | None, *, use_fast: bool | None = True
) -> TokenizerLike:
    """Load an HF tokenizer by id, with cloud resolution and retries.

    ``None`` / ``"__fallback__"`` return the dependency-free stub. Cloud URIs
    are resolved to a local cache path before loading. Transient load failures
    are retried with backoff; ``use_fast`` is dropped and the load retried once
    if the tokenizer class rejects the kwarg.
    """
    # Before importing hf tokenizers we tell it we handle the parallelism and
    # not hf tokenizers. This avoids unforeseen effects when running multiple
    # op instances.
    suppress_library_threads()

    if tokenizer_id in (None, "__fallback__"):
        return fallback_tokenizer()

    # Deferred: tokenizer_cloud pulls the io/storage stack into this otherwise light module.
    from zephon.utils.tokenizer_cloud import resolve_tokenizer_id_with_retry

    import_lock_ctx = (
        _IMPORT_LOCK if _IMPORT_LOCK is not None else contextlib.nullcontext()
    )
    with import_lock_ctx:
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


__all__ = [
    "TokenBatch",
    "TokenizerLike",
    "TokenizerOutput",
    "fallback_tokenizer",
    "load_hf_tokenizer",
]
