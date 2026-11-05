# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Route each record to text or chat tokenization within one operator."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from zephon._internal.ops.tokenize_base import TokenizeBase
from zephon._internal.ops.tokenize_chat import ChatTokenCountingSpec, TokenizeChat
from zephon._internal.ops.tokenize_text import TokenizeText
from zephon._internal.token_counting import (
    CountPlan,
    DeliveredTokenCounter,
    TextTokenCountingSpec,
    TokenCountingSpec,
)
from zephon._internal.utils.tokenizer import TokenizerLike
from zephon.ops.base import OpContext
from zephon.ops.config import (
    ChatSpecialTokensMode,
    LossScope,
    SpanSource,
    SpecialTokensMode,
)
from zephon.ops.traits import OpTraits
from zephon.types import SampleMeta, SampleRecord


def _is_chat(payload: Any, chat_path: tuple[str, ...]) -> bool:
    """Return whether ``chat_path`` resolves to a non-null value in ``payload``.

    Null counts as absent so datasets with a merged schema, where every row has
    both columns, route by whichever column is filled.
    """
    value: Any = payload
    for key in chat_path:
        if not isinstance(value, Mapping):
            return False
        value = value.get(key)
    return value is not None


def _text_loss_mask(length: int) -> np.ndarray:
    """Supervise every text token except the first, like chat ``all_tokens``."""
    mask = np.ones(length, dtype=np.uint8)
    if length:
        mask[0] = 0
    return mask


def _is_first_child(meta: SampleMeta, parent: SampleMeta) -> bool:
    return (
        meta.lineage == (*parent.lineage, 0)
        and meta.chunk_id == parent.chunk_id
        and meta.chunk_offset == parent.chunk_offset
        and meta.sample_id == parent.sample_id
    )


class TokenizeMixed(TokenizeBase):
    """Tokenize a stream that mixes plain-text documents and chat conversations.

    Records whose payload has a non-null ``chat_field`` go through
    ``TokenizeChat``; all others go through ``TokenizeText``. Each path keeps
    the settings and defaults of its standalone operator, including special
    tokens and overflow handling, and both share one tokenizer. Both emit
    ``{"input_ids": int64[n], <mask_field_out>: uint8[n]}`` so records from
    either path can be packed and batched together. Text records supervise every
    token except the first, the rule chat applies so packed training never
    supervises a prediction across a document boundary. Neither path emits an
    ``attention_mask``; ``to_training(return_padding_mask=True)`` marks padding.

    Text records are tokenized as one batch per call; chat records are processed
    individually, as in ``TokenizeChat``. Outputs keep the input order.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike | None = None,
        tokenizer_id: str | None = None,
        *,
        eos_token: str | None = None,
        max_length: int | None = None,
        truncation: bool = False,
        split_long_samples: bool = False,
        text_field: str = "text",
        text_special_tokens: SpecialTokensMode = "bos_eos",
        text_bos_token_id: int | None = None,
        text_eos_token_id: int | None = None,
        chat_field: str = "messages",
        chat_special_tokens: ChatSpecialTokensMode | None = None,
        chat_bos_token_id: int | None = None,
        chat_eos_token_id: int | None = None,
        chat_template: str | os.PathLike[str] | None = None,
        apply_chat_template: bool = True,
        span_source: SpanSource = "auto",
        supervise_assistant_header: bool = False,
        loss_scope: LossScope = "all_assistant",
        chat_template_kwargs: Mapping[str, Any] | None = None,
        tools_field: str = "tools",
        enable_thinking_field: str = "enable_thinking",
        mask_field_out: str = "loss_mask",
        max_batch: int = 64,
        max_latency_ms: int | None = 3,
    ) -> None:
        """Configure both tokenization paths around one shared tokenizer.

        Options prefixed ``text_`` or ``chat_`` mean what the same option means
        on ``TokenizeText`` or ``TokenizeChat`` and keep its default.

        Args:
            tokenizer: Pre-instantiated fast Hugging Face tokenizer.
            tokenizer_id: Tokenizer to load lazily in each worker.
            eos_token: EOS string override applied when loading ``tokenizer_id``.
            max_length: Output limit shared by both paths. Text applies it only
                with ``truncation`` or ``split_long_samples``; chat truncates to
                it unless ``split_long_samples`` is set.
            truncation: Truncate text to ``max_length``, as in ``TokenizeText``.
                Chat already truncates whenever it does not split.
            split_long_samples: Split both text and conversations into
                ``max_length`` chunks instead of truncating them.
            text_field: Dot path of the text in plain-text records.
            text_special_tokens: ``TokenizeText`` special-token mode.
            text_bos_token_id: BOS ID for the text path's bracket modes.
            text_eos_token_id: EOS ID for the text path's bracket modes.
            chat_field: Dot path of the messages; a non-null value selects chat.
            chat_special_tokens: ``TokenizeChat`` special-token mode.
            chat_bos_token_id: Outer BOS ID for the chat path.
            chat_eos_token_id: Outer EOS ID for the chat path.
            chat_template: Chat template name, Jinja text or file path.
            apply_chat_template: Render chat records with the template.
            span_source: How chat assistant spans are found.
            supervise_assistant_header: Include assistant headers in chat targets.
            loss_scope: Which chat tokens are supervised.
            chat_template_kwargs: Static chat-template render arguments.
            tools_field: Per-record tools path for chat rendering.
            enable_thinking_field: Per-record thinking flag path for chat rendering.
            mask_field_out: Output field holding the loss mask on both paths.
            max_batch: Records accumulated per call.
            max_latency_ms: Accumulator flush latency outside deterministic mode.
        """
        if text_field == chat_field:
            raise ValueError("text_field and chat_field must differ")
        super().__init__(
            tokenizer,
            tokenizer_id,
            use_fast=True,
            eos_token=eos_token,
            max_batch=max_batch,
            max_latency_ms=max_latency_ms,
        )
        self.text_field = text_field
        self.chat_field = chat_field
        self.max_length = max_length
        self.mask_field_out = mask_field_out
        self._chat_path: tuple[str, ...] = tuple(chat_field.split("."))

        # Sub-operators share this operator's tokenizer: a pre-instantiated one
        # is passed through, and a lazily loaded one is injected at setup.
        batching: dict[str, Any] = {
            "max_batch": max_batch,
            "max_latency_ms": max_latency_ms,
        }
        self.text_op = TokenizeText(
            tokenizer,
            tokenizer_id,
            field=text_field,
            add_attention_mask=False,
            max_length=max_length,
            truncation=truncation,
            split_long_samples=split_long_samples,
            special_tokens=text_special_tokens,
            bos_token_id=text_bos_token_id,
            eos_token_id=text_eos_token_id,
            **batching,
        )
        self.chat_op = TokenizeChat(
            tokenizer,
            tokenizer_id,
            field=chat_field,
            max_length=max_length,
            split_long_samples=split_long_samples,
            special_tokens=chat_special_tokens,
            bos_token_id=chat_bos_token_id,
            eos_token_id=chat_eos_token_id,
            chat_template=chat_template,
            apply_chat_template=apply_chat_template,
            span_source=span_source,
            supervise_assistant_header=supervise_assistant_header,
            loss_scope=loss_scope,
            chat_template_kwargs=chat_template_kwargs,
            tools_field=tools_field,
            enable_thinking_field=enable_thinking_field,
            mask_field_out=mask_field_out,
            **batching,
        )

    # Setup

    def setup(self, ctx: OpContext) -> None:
        super().setup(ctx)
        self.text_op.setup(ctx)
        self.chat_op.setup(ctx)

    def teardown(self) -> None:
        self.text_op.teardown()
        self.chat_op.teardown()

    def _finalize_setup(self) -> None:
        # This runs under TokenizeBase's non-reentrant setup lock on free-threaded
        # Python, so finalize the text path directly instead of re-entering its
        # _setup_tokenizer. A failure is cached on this operator.
        self.text_op.tok = self.tok
        self.text_op._eos_token = self._eos_token
        self.text_op._finalize_setup()
        self.text_op._tokenizer_instantiated = True
        # The chat path finishes setup on its first chat record, so text-only
        # streams never prepare or verify chat templates. Sharing the tokenizer
        # keeps it from loading its own copy without the EOS override.
        self.chat_op.tok = self.tok

    def _preflight_setup(self) -> None:
        # Preflight checks every configured path, including an unused chat path.
        self._setup_tokenizer()
        self.chat_op._setup_tokenizer()

    def _cache_preflight_result(self, probe: TokenizeBase) -> None:
        if isinstance(probe, TokenizeMixed):
            self.chat_op._cache_preflight_result(probe.chat_op)

    # Record processing

    def _finish_text(self, record: SampleRecord) -> SampleRecord:
        payload = cast(Mapping[str, Any], record.payload)
        ids = np.asarray(payload["input_ids"], dtype=np.int64)
        record.payload = {
            "input_ids": ids,
            self.mask_field_out: _text_loss_mask(len(ids)),
        }
        return record

    def _text_outputs(self, inputs: list[SampleRecord]) -> list[list[SampleRecord]]:
        """Tokenize text records as one batch and group the outputs per input.

        ``TokenizeText`` emits each input's outputs contiguously and in order:
        the input's own metadata, or split children numbered from zero.
        """
        metas = [record.meta for record in inputs]
        groups: list[list[SampleRecord]] = []
        for record in self.text_op.process_many(inputs):
            nxt = metas[len(groups)] if len(groups) < len(metas) else None
            if nxt is not None and (
                record.meta is nxt or _is_first_child(record.meta, nxt)
            ):
                groups.append([])
            elif not groups:
                raise RuntimeError("TokenizeMixed: unexpected first text output")
            groups[-1].append(self._finish_text(record))
        if len(groups) != len(inputs):
            raise RuntimeError(
                f"TokenizeMixed: text tokenization produced {len(groups)} output "
                + f"groups for {len(inputs)} inputs"
            )
        return groups

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        if not self._tokenizer_instantiated:
            self._setup_tokenizer()
        per_input: list[list[SampleRecord]] = [[] for _ in elems]
        text_indices: list[int] = []
        for index, record in enumerate(elems):
            if record.meta.tombstone:
                per_input[index] = [record]
            elif _is_chat(record.payload, self._chat_path):
                per_input[index] = self.chat_op.process_one(record)
            else:
                text_indices.append(index)
        if text_indices:
            groups = self._text_outputs([elems[i] for i in text_indices])
            for index, group in zip(text_indices, groups):
                per_input[index] = group
        return [record for group in per_input for record in group]

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        return self.process_many([elem])

    # Op plumbing

    def traits(self) -> OpTraits:
        return OpTraits(
            calibration_replayable=True,
            indexable=True,
            preserves_cursor_order=True,
            parallelism=4,
        )

    def token_counting_spec(self) -> MixedTokenCountingSpec:
        # Sub-operators never load a tokenizer, so carry the EOS override here.
        return MixedTokenCountingSpec(
            tokenizer=self.tok,
            tokenizer_id=self.tokenizer_id,
            chat=replace(
                self.chat_op.token_counting_spec(),
                tokenizer=self.tok,
                eos_token=self.eos_token,
            ),
            text=replace(
                TextTokenCountingSpec.from_op(self.text_op),
                tokenizer=self.tok,
                eos_token=self.eos_token,
            ),
            chat_field=self.chat_field,
        )


# ---------------------------------------------------------------------------
# Token-counting spec (priming calibration)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixedTokenCountingSpec(TokenCountingSpec):
    """Count each calibration payload with the path it would take."""

    chat: ChatTokenCountingSpec | None = None
    text: TextTokenCountingSpec | None = None
    chat_field: str = "messages"

    def build_counter(self) -> DeliveredTokenCounter:
        assert self.chat is not None and self.text is not None
        return _MixedCounter(
            self.chat.build_counter(),
            self.text.build_counter(),
            tuple(self.chat_field.split(".")),
        )


@dataclass(frozen=True)
class _MixedPlan(CountPlan):
    chat: CountPlan
    text: CountPlan
    chat_path: tuple[str, ...]

    @property
    def description(self) -> str:
        return f"mixed ({self.chat.description}; {self.text.description})"

    def count(self, payload: Any) -> int | None:
        if _is_chat(payload, self.chat_path):
            return self.chat.count(payload)
        return self.text.count(payload)


@dataclass(frozen=True)
class _MixedCounter(DeliveredTokenCounter):
    chat: DeliveredTokenCounter
    text: DeliveredTokenCounter
    chat_path: tuple[str, ...]

    def plan(self, sample_payloads: list[Any]) -> CountPlan:
        is_chat = [_is_chat(payload, self.chat_path) for payload in sample_payloads]
        return _MixedPlan(
            self.chat.plan([p for p, c in zip(sample_payloads, is_chat) if c]),
            self.text.plan([p for p, c in zip(sample_payloads, is_chat) if not c]),
            self.chat_path,
        )


__all__ = ["MixedTokenCountingSpec", "TokenizeMixed"]
