# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer operators that prepare text fields for model consumption."""

from __future__ import annotations

import logging
import os
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

from zephon.core.children import spawn_child
from zephon.core.constants import (
    SampleMeta,
    SamplePayload,
    SamplePayloadDict,
    SampleRecord,
)
from zephon.core.op_base import DefaultFinalize, DefaultSetup, OpContext
from zephon.core.traits import Buffering, OpTraits

if TYPE_CHECKING:
    import numpy as np
    import tensorflow as tf
    import torch

log = logging.getLogger(__name__)

TokenSeq: TypeAlias = Union[
    "np.ndarray", "torch.Tensor", "tf.Tensor", Sequence[int], list[int]
]
# Bound TypeVar ensures "Tensor in -> Tensor out" relationship
T_TokenSeq = TypeVar("T_TokenSeq", bound=TokenSeq)

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


class TokenizeText(DefaultSetup, DefaultFinalize[SampleRecord]):
    """Tokenize text fields using a provided or auto-resolved tokenizer."""

    def __init__(
        self,
        tokenizer: TokenizerLike | None = None,
        tokenizer_id: str | None = None,
        *,
        field: str = "text",
        add_attention_mask: bool = True,
        max_length: int | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        return_tensors: str | None = None,
        split_long_samples: bool = False,
        use_fast: bool | None = True,
        buffering: Optional[Buffering] = None,
        preserve_upstream_payload: bool = False,
    ) -> None:
        DefaultSetup.__init__(self)

        if split_long_samples and truncation:
            raise ValueError("split_long_samples is mutually exclusive with truncation")
        if split_long_samples and not max_length:
            raise ValueError("split_long_samples requires max_length")

        self.tok = tokenizer
        self.tokenizer_id = tokenizer_id
        self.field = field
        self.add_attention_mask = add_attention_mask
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.split_long_samples = split_long_samples
        self.use_fast = use_fast
        self.preserve_upstream_payload = preserve_upstream_payload
        self._buffering = buffering or Buffering(max_batch=64, max_latency_ms=3)
        self._warned_non_mapping = False
        self._warned_preserve_non_mapping = False
        # Cache kwargs to avoid building dict per batch
        self._cached_kwargs: dict[str, Any] = {}
        self._tokenizer_instantiated = False

    def setup(
        self,
        ctx: OpContext,
        stage_index: int,
        stage_name: str,
        op_index: int,
        collect_stats: bool,
    ) -> None:
        DefaultSetup.setup(self, ctx, stage_index, stage_name, op_index, collect_stats)
        # We do NOT set up the tokenizer here to avoid problems in multiprocessing:
        # hf tokenizers don't like if we fork after creating the object

        # Pre-compute tokenizer kwargs (Performance Optimization)
        self._cached_kwargs = {"add_special_tokens": True}
        if self.split_long_samples:
            self._cached_kwargs["padding"] = False
            self._cached_kwargs["truncation"] = False
        else:
            self._cached_kwargs["padding"] = self.padding
            self._cached_kwargs["truncation"] = self.truncation
            if self.max_length is not None:
                self._cached_kwargs["max_length"] = self.max_length
            if self.return_tensors is not None:
                self._cached_kwargs["return_tensors"] = self.return_tensors

    def _setup_tokenizer(self) -> None:
        if self.tok is None:
            # Before importing hf tokenizers we tell it we handle the parallelism
            # and not hf tokenizers. This avoids unforeseen effects when running
            # multiple op instances.
            os.environ["TOKENIZERS_PARALLELISM"] = "False"
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["RAYON_NUM_THREADS"] = "1"

            if self.tokenizer_id in (None, "__fallback__"):
                self.tok = _fallback_tokenizer()
            else:
                from transformers import AutoTokenizer

                kwargs: dict[str, Any] = {}
                if self.use_fast is not None:
                    kwargs["use_fast"] = self.use_fast
                try:
                    self.tok = AutoTokenizer.from_pretrained(
                        self.tokenizer_id, **kwargs
                    )
                except TypeError as exc:
                    # Some tokenizers may not accept the `use_fast` kwarg;
                    # retry without it so we surface the original failure instead of
                    # a signature mismatch.
                    if "use_fast" in kwargs and "use_fast" in str(exc):
                        kwargs = dict(kwargs)
                        kwargs.pop("use_fast", None)
                        self.tok = AutoTokenizer.from_pretrained(
                            self.tokenizer_id, **kwargs
                        )

                    else:
                        raise

        self._tokenizer_instantiated = True

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, preserves_cursor_order=True, parallelism=4)

    def buffering(self) -> Optional[Buffering]:
        return self._buffering

    def _extract_text(self, payload: SamplePayload) -> tuple[str, SamplePayloadDict]:
        if isinstance(payload, dict):
            # Safe cast: we expect the user to provide string fields as configured
            text_value = cast(str, payload.get(self.field, ""))
            if self.preserve_upstream_payload:
                return text_value, payload
            # Fresh dict: drop upstream keys entirely
            return text_value, {}
        if self.preserve_upstream_payload and not self._warned_preserve_non_mapping:
            log.warning(
                "preserve_upstream_payload=True cannot be honored for non-mapping payloads "
                + "(got %s); tokenizing the value directly",
                type(payload).__name__,
            )
            self._warned_preserve_non_mapping = True
        elif self.field and not self._warned_non_mapping:
            log.warning(
                "TokenizeText expected mapping payloads for field '%s'; "
                + "got %s, tokenizing the value directly",
                self.field,
                type(payload).__name__,
            )
            self._warned_non_mapping = True
        text = str(payload)
        return text, {}

    def _set_payload_tensors(
        self, payload: SamplePayloadDict, ids: TokenSeq, mask: TokenSeq | None
    ) -> None:
        """Helper to centralize casting logic and keep _process clean."""
        # We must cast because SamplePayload definition is strict (Union),
        # but Tensors are handled as 'Any' or specific array types at runtime.
        payload["input_ids"] = cast(SamplePayload, ids)
        if self.add_attention_mask and mask is not None:
            payload["attention_mask"] = cast(SamplePayload, mask)

    def _process(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        if not self._tokenizer_instantiated:
            self._setup_tokenizer()

        if self.tok is None:
            msg = "Tokenizer not initialised"
            raise RuntimeError(msg)

        texts: list[str] = []
        metas: list[SampleMeta] = []
        payloads: list[SamplePayloadDict] = []

        for elem in elems:
            text, payload = self._extract_text(elem.payload)
            texts.append(text)
            metas.append(elem.meta)
            payloads.append(payload)

        if not texts:
            return []

        encoded = self._tokenize_texts(texts)

        # Use type ignores here as TokenBatch is complex;
        # _normalize_batch handles the runtime safety.
        raw_input_ids: TokenBatch = encoded.get("input_ids", [])  # type: ignore[assignment]
        raw_attention_mask: TokenBatch | None = encoded.get("attention_mask")  # type: ignore[assignment]

        input_ids = self._normalize_batch(raw_input_ids, len(metas))

        attention_mask: Sequence[TokenSeq] | None = None
        if raw_attention_mask is not None:
            attention_mask = self._normalize_batch(raw_attention_mask, len(metas))

        # 3. FAST PATH: In-Place Reuse (Zero Allocation)
        if not self.split_long_samples:
            if attention_mask is None:
                # Zip input elements directly to modify them
                for elem, payload, ids in zip(elems, payloads, input_ids):
                    self._set_payload_tensors(payload, ids, None)
                    elem.payload = payload
            else:
                for elem, payload, ids, mask in zip(
                    elems, payloads, input_ids, attention_mask
                ):
                    self._set_payload_tensors(payload, ids, mask)
                    elem.payload = payload

            # Return the original list structure
            return elems

        # 4. Slower path: Splitting logic required (Must allocate new records)
        results: list[SampleRecord] = []
        for idx, meta in enumerate(metas):
            payload = payloads[idx]
            ids = input_ids[idx]
            mask = attention_mask[idx] if attention_mask is not None else None
            results.extend(self._maybe_split_outputs(meta, payload, ids, mask))
        return results

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        return self._process([elem])

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        return self._process(elems)

    def resolved_tokenizer_id(self) -> str | None:
        if self.tokenizer_id is not None:
            return self.tokenizer_id
        name = getattr(self.tok, "name_or_path", None)
        return "__fallback__" if name is None else str(name)

    def _ensure_padding_token(self) -> None:
        if not self.padding:
            return
        tokenizer = self.tok
        if tokenizer is None:
            return
        if getattr(tokenizer, "pad_token", None) is not None:
            return
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            return
        tokenizer.pad_token = eos_token

    def _tokenize_texts(self, texts: list[str]) -> TokenizerOutput:
        tokenizer = self.tok
        if tokenizer is None:
            msg = "Tokenizer not initialised"
            raise RuntimeError(msg)
        self._ensure_padding_token()
        return tokenizer(texts, **self._cached_kwargs)

    def _convert_tensor(self, value: TokenSeq) -> TokenSeq:
        backend = self.return_tensors
        if backend is None:
            return value
        if backend == "pt":
            torch = self._lazy_import("torch")
            return (
                value
                if hasattr(value, "shape") and "torch" in type(value).__module__
                else torch.tensor(value)
            )
        if backend in ("np", "numpy"):
            np = self._lazy_import("numpy")
            return (
                value
                if hasattr(value, "shape") and "numpy" in type(value).__module__
                else np.asarray(value)
            )
        if backend == "tf":
            tf = self._lazy_import("tensorflow")
            return (
                value
                if hasattr(value, "shape") and "tensorflow" in type(value).__module__
                else tf.convert_to_tensor(value)
            )
        raise ValueError(f"Unsupported return_tensors backend: {backend}")

    def _lazy_import(self, module: str) -> ModuleType:
        try:
            return __import__(module)
        except Exception as exc:
            raise RuntimeError(
                f"return_tensors='{self.return_tensors}' requested but {module} is unavailable"
            ) from exc

    def _normalize_batch(
        self, batch: TokenBatch, batch_size: int
    ) -> Sequence[TokenSeq]:
        """Ensure tokenizer outputs align with requested batch size.

        This handles edge cases where batch_size=1 causes some backends
        to squeeze dimensions, or where we need to ensure iterability.
        """
        # 1. Handle Lists (Fastest check for common HF output)
        if isinstance(batch, list):
            if len(batch) == batch_size:
                return cast(Sequence[TokenSeq], batch)
            # Mismatch implies it's a single unbatched sequence needing wrapping
            return [cast(TokenSeq, batch)]

        # 2. Handle Tuples
        if isinstance(batch, tuple):
            if len(batch) == batch_size:
                return cast(Sequence[TokenSeq], batch)
            return [cast(TokenSeq, batch)]

        # 3. Handle Tensors (Attributes check)
        shape = getattr(batch, "shape", None)

        if shape is not None:
            # Check dimension 0. Safety: len(shape) check handles 0-d scalars.
            if len(shape) > 0 and shape[0] == batch_size:
                return cast(Sequence[TokenSeq], batch)

            # If shape mismatch or scalar, wrap it in a list.
            # This is backend-agnostic (works for torch/tf/np) and functionally
            # equivalent to unsqueeze(0) when iterated in zip().
            return [cast(TokenSeq, batch)]

        # 4. Fallback (Unknown type, possibly just a Sequence of ints)
        if len(batch) == batch_size:  # type: ignore
            return cast(Sequence[TokenSeq], batch)
        return [cast(TokenSeq, batch)]

    def _pad_token_id(self) -> int:
        tok = self.tok
        if tok is None:
            return 0
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tok, "pad_token", None)
        try:
            return int(pad_id) if pad_id is not None else 0
        except Exception:
            return 0

    def _split_segments(
        self, ids: T_TokenSeq, mask: T_TokenSeq | None
    ) -> list[tuple[T_TokenSeq, T_TokenSeq | None]]:
        max_len = self.max_length
        assert max_len is not None
        pad_id = self._pad_token_id()

        module = type(ids).__module__

        # Dispatch based on module string to avoid strict isinstance imports
        if "torch" in module:
            ids_t = cast("torch.Tensor", ids)
            mask_t = cast("torch.Tensor | None", mask)

            result_torch = self._split_segments_torch(ids_t, mask_t, max_len, pad_id)
            return cast(list[tuple[T_TokenSeq, T_TokenSeq | None]], result_torch)

        if "numpy" in module:
            ids_np = cast("np.ndarray", ids)
            mask_np = cast("np.ndarray | None", mask)

            result_np = self._split_segments_numpy(ids_np, mask_np, max_len, pad_id)
            return cast(list[tuple[T_TokenSeq, T_TokenSeq | None]], result_np)

        if "tensorflow" in module:
            ids_tf = cast("tf.Tensor", ids)
            mask_tf = cast("tf.Tensor | None", mask)

            result_tf = self._split_segments_tf(ids_tf, mask_tf, max_len, pad_id)
            return cast(list[tuple[T_TokenSeq, T_TokenSeq | None]], result_tf)

        # Standard Python List/Sequence path
        # Explicitly cast to Sequence to satisfy helper
        ids_std = cast(Sequence[int], ids)
        mask_std = cast(Sequence[int] | None, mask)

        result_std = self._split_segments_std(ids_std, mask_std, max_len, pad_id)
        return cast(list[tuple[T_TokenSeq, T_TokenSeq | None]], result_std)

    def _split_segments_std(
        self,
        ids: Sequence[int],
        mask: Sequence[int] | None,
        max_len: int,
        pad_id: int,
    ) -> list[tuple[list[int], list[int] | None]]:
        # Performance: Avoid copy if already list
        ids_list = ids if isinstance(ids, list) else list(ids)

        mask_list: list[int] | None = None
        if mask is not None:
            mask_list = mask if isinstance(mask, list) else list(mask)

        if mask_list is None and self.add_attention_mask:
            mask_list = [1] * len(ids_list)

        segments: list[tuple[list[int], list[int] | None]] = []
        if not ids_list:
            empty_ids: list[int] = []
            empty_mask: list[int] | None = [] if mask_list is not None else None
            if self.padding:
                empty_ids = [pad_id] * max_len
                if self.add_attention_mask:
                    empty_mask = [0] * max_len
            segments.append((empty_ids, empty_mask))
            return segments

        for start in range(0, len(ids_list), max_len):
            # Slicing creates a copy, which is efficient for ints.
            seg_ids = ids_list[start : start + max_len]
            seg_mask = (
                mask_list[start : start + max_len] if mask_list is not None else None
            )
            if self.padding and len(seg_ids) < max_len:
                pad_len = max_len - len(seg_ids)
                # Performance: Use extend instead of '+' to modify the slice list in-place
                seg_ids.extend([pad_id] * pad_len)
                if seg_mask is not None:
                    seg_mask.extend([0] * pad_len)
            segments.append((seg_ids, seg_mask))
        return segments

    def _split_segments_torch(
        self,
        ids: "torch.Tensor",
        mask: "torch.Tensor | None",
        max_len: int,
        pad_id: int,
    ) -> list[tuple["torch.Tensor", "torch.Tensor | None"]]:
        import torch

        segments: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        if mask is None and self.add_attention_mask:
            mask = torch.ones_like(ids, dtype=ids.dtype)

        total = ids.shape[0]
        if total == 0:
            seg_ids = ids
            seg_mask = mask
            if self.padding:
                seg_ids = torch.full(
                    (max_len,), pad_id, dtype=ids.dtype, device=ids.device
                )
                if self.add_attention_mask:
                    seg_mask = torch.zeros_like(seg_ids)
            segments.append((seg_ids, seg_mask))
            return segments

        for start in range(0, total, max_len):
            end = min(total, start + max_len)
            seg_ids = ids[start:end]
            seg_mask = mask[start:end] if mask is not None else None
            seg_len = end - start
            if self.padding and seg_len < max_len:
                pad_len = max_len - seg_len
                pad = torch.full((pad_len,), pad_id, dtype=ids.dtype, device=ids.device)
                seg_ids = torch.cat((seg_ids, pad), dim=-1)
                if seg_mask is not None:
                    pad_mask = torch.zeros(
                        (pad_len,), dtype=seg_mask.dtype, device=seg_ids.device
                    )
                    seg_mask = torch.cat((seg_mask, pad_mask), dim=-1)
            segments.append((seg_ids, seg_mask))
        return segments

    def _split_segments_numpy(
        self,
        ids: "np.ndarray",
        mask: "np.ndarray | None",
        max_len: int,
        pad_id: int,
    ) -> list[tuple["np.ndarray", "np.ndarray | None"]]:
        import numpy as np

        segments: list[tuple[np.ndarray, np.ndarray | None]] = []
        if mask is None and self.add_attention_mask:
            mask = np.ones_like(ids, dtype=ids.dtype)

        total = ids.shape[0]
        if total == 0:
            seg_ids = ids
            seg_mask = mask
            if self.padding:
                seg_ids = np.full((max_len,), pad_id, dtype=ids.dtype)
                if self.add_attention_mask:
                    seg_mask = np.zeros((max_len,), dtype=ids.dtype)
            segments.append((seg_ids, seg_mask))
            return segments

        for start in range(0, total, max_len):
            end = min(total, start + max_len)
            seg_ids = ids[start:end]
            seg_mask = mask[start:end] if mask is not None else None
            seg_len = end - start
            if self.padding and seg_len < max_len:
                pad_len = max_len - seg_len
                seg_ids = np.concatenate(
                    (seg_ids, np.full((pad_len,), pad_id, dtype=ids.dtype))
                )
                if seg_mask is not None:
                    seg_mask = np.concatenate(
                        (seg_mask, np.zeros((pad_len,), dtype=seg_mask.dtype))
                    )
            segments.append((seg_ids, seg_mask))
        return segments

    def _split_segments_tf(
        self,
        ids: "tf.Tensor",
        mask: "tf.Tensor | None",
        max_len: int,
        pad_id: int,
    ) -> list[tuple["tf.Tensor", "tf.Tensor | None"]]:
        import tensorflow as tf

        segments: list[tuple[tf.Tensor, tf.Tensor | None]] = []
        total_dim = ids.shape[0]
        total = (
            int(total_dim) if total_dim is not None else int(tf.shape(ids)[0].numpy())
        )
        if mask is None and self.add_attention_mask:
            mask = tf.ones_like(ids)

        if total == 0:
            seg_ids = ids
            seg_mask = mask
            if self.padding:
                seg_ids = tf.fill((max_len,), tf.cast(pad_id, ids.dtype))
                if self.add_attention_mask:
                    seg_mask = tf.zeros((max_len,), dtype=ids.dtype)
            segments.append((seg_ids, seg_mask))
            return segments

        for start in range(0, total, max_len):
            end = min(total, start + max_len)
            seg_ids = ids[start:end]
            seg_mask = mask[start:end] if mask is not None else None
            seg_len = end - start
            if self.padding and seg_len < max_len:
                pad_len = max_len - seg_len
                pad = tf.fill((pad_len,), tf.cast(pad_id, ids.dtype))
                seg_ids = tf.concat((seg_ids, pad), axis=-1)
                if seg_mask is not None:
                    pad_mask = tf.zeros((pad_len,), dtype=seg_mask.dtype)
                    seg_mask = tf.concat((seg_mask, pad_mask), axis=-1)
            segments.append((seg_ids, seg_mask))
        return segments

    def _maybe_split_outputs(
        self,
        meta: SampleMeta,
        payload: SamplePayloadDict,
        ids: T_TokenSeq,
        mask: T_TokenSeq | None,
    ) -> list[SampleRecord]:
        if not self.split_long_samples or self.max_length is None:
            self._set_payload_tensors(payload, ids, mask)
            return [SampleRecord(meta=meta, payload=payload)]

        segments = self._split_segments(ids, mask)

        if len(segments) == 1:
            seg_ids, seg_mask = segments[0]
            # Convert backend if needed (e.g. return_tensors='pt')
            conv_ids = self._convert_tensor(seg_ids)
            conv_mask = self._convert_tensor(seg_mask) if seg_mask is not None else None
            self._set_payload_tensors(payload, conv_ids, conv_mask)
            return [SampleRecord(meta=meta, payload=payload)]

        records: list[SampleRecord] = []
        last_idx = len(segments) - 1
        for idx, (seg_ids, seg_mask) in enumerate(segments):
            child_meta = spawn_child(meta, idx, is_last_child=(idx == last_idx))
            child_payload = dict(payload)

            conv_ids = self._convert_tensor(seg_ids)
            conv_mask = self._convert_tensor(seg_mask) if seg_mask is not None else None

            self._set_payload_tensors(child_payload, conv_ids, conv_mask)
            records.append(SampleRecord(meta=child_meta, payload=child_payload))
        return records


def _fallback_tokenizer() -> TokenizerLike:
    class _Tokenizer:
        name_or_path: str | None = "__fallback__"
        pad_token: int | str | None = None
        eos_token: int | str | None = 0

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
                tokens = [abs(hash(word)) % 10000 + 1 for word in value.split()]
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
