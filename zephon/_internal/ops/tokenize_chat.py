# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Chat-template tokenizer operator producing token ids and a loss mask."""

from __future__ import annotations

import logging
import os
import re
import warnings
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Mapping,
    Sequence,
    cast,
    get_args,
)

import numpy as np

from zephon._internal.ops._one_to_one import (
    REQUIRE_ONE_TO_ONE_RECORDS,
    one_to_one_violation,
    record_identity,
)
from zephon._internal.ops.tokenize_base import _MISSING, TokenizeBase
from zephon._internal.token_counting import (
    CountPlan,
    DeliveredTokenCounter,
    FatalCountError,
    TokenCountingSpec,
)
from zephon._internal.utils.tokenizer import (
    TokenizerLike,
    encode_rendered_text,
    generation_mask,
    selected_assistant_indices,
    thread_salted_template,
)
from zephon.ops.base import OpContext
from zephon.ops.children import tombstones_for_record
from zephon.ops.config import LossScope, SpanSource
from zephon.ops.traits import OpTraits
from zephon.types import SamplePayloadDict, SampleRecord
from zephon.validation import ValidationError

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from zephon._internal.utils.chat_template import (
        TemplateSetupResult,
        TemplateVerifier,
    )

log = logging.getLogger(__name__)

_VALID_SPAN_SOURCES: tuple[str, ...] = get_args(SpanSource)

# Same detection HF uses to decide whether a template records assistant spans.
_GENERATION_TAG_RE = re.compile(r"\{%-?\s*generation\s*-?%\}")

# User template kwargs may not override operator-owned render controls.
_RESERVED_RENDER_KWARGS = frozenset(
    {
        "tokenize",
        "return_dict",
        "return_assistant_tokens_mask",
        "add_generation_prompt",
        "chat_template",
        "tools",
        "truncation",
        "max_length",
        "return_tensors",
        "return_attention_mask",
        "continue_final_message",
        "tokenizer_kwargs",
        "padding",
        "messages",
        "return_offsets_mapping",
    }
)


def _mask_from_spans(
    offsets: Sequence[tuple[int, int]], spans: Sequence[tuple[int, int]]
) -> "np.ndarray":
    """Mark tokens whose character range overlaps any supervised span.

    A token straddling a span boundary is supervised. Zero-width offsets
    never overlap. Generation-tag masks use HF's endpoint rule separately.
    """
    mask = np.zeros(len(offsets), dtype=np.uint8)
    if not len(offsets) or not spans:
        return mask
    bounds = np.asarray(offsets, dtype=np.int64).reshape(len(offsets), 2)
    starts, ends = bounds[:, 0], bounds[:, 1]
    for a, b in spans:
        mask[(starts < b) & (ends > a)] = 1
    return mask


def _last_span_only(mask: "np.ndarray") -> "np.ndarray":
    """Zero every supervised run except the last contiguous one."""
    supervised = np.flatnonzero(mask)
    if supervised.size == 0:
        return mask
    end = int(supervised[-1]) + 1
    gaps = np.flatnonzero(mask[:end] == 0)
    start = int(gaps[-1]) + 1 if gaps.size else 0
    out = np.zeros_like(mask)
    out[start:end] = 1
    return out


def _resolve_loss_scope(
    loss_scope: LossScope | None,
    loss_on_last_turn_only: bool | None,
    *,
    stacklevel: int,
) -> LossScope:
    """Resolve the deprecated boolean and warn at the calling API boundary."""
    if loss_scope is not None and loss_on_last_turn_only is not None:
        raise ValueError("supply loss_scope or loss_on_last_turn_only, not both")
    if loss_on_last_turn_only is not None:
        if type(loss_on_last_turn_only) is not bool:
            raise TypeError("loss_on_last_turn_only must be a bool")
        warnings.warn(
            "loss_on_last_turn_only is deprecated; use loss_scope='last_span' "
            + "or loss_scope='all_assistant'",
            FutureWarning,
            stacklevel=stacklevel,
        )
        loss_scope = "last_span" if loss_on_last_turn_only else "all_assistant"
    if loss_scope is None:
        loss_scope = "all_assistant"
    if loss_scope not in get_args(LossScope):
        raise ValueError(f"invalid loss_scope: {loss_scope!r}")
    return loss_scope


class TokenizeChat(TokenizeBase):
    """Tokenize chat conversations into ids plus a supervised-token loss mask.

    Renders each conversation to one flat string, tokenizes it once, and marks
    assistant tokens via character spans. Span provenance:

    * ``generation_tags``: HF supplies the assistant mask for ``all_assistant``
      and ``last_span``. ``after_last_user`` tracks which input message owns
      each generation block and maps selected ranges through HF's encoding.
    * ``prefix_diff``: spans are reconstructed by prefix-diffing partial
      renders. Auto-selected for untagged templates with ``all_assistant`` or
      ``last_span`` and guarded against position-dependent rendering.
    * ``apply_chat_template=False``: message contents are concatenated
      verbatim, with EOS appended to assistant turns, so spans are exact.

    Output payload: ``{"input_ids": int64[n], <mask_field_out>: uint8[n]}``,
    unshifted and aligned, ``n <= max_length``. ``mask[i] == 1`` means token
    ``i`` is trained as a target; the emit-time shift consumes ``mask[1:]``.

    ``mask[0]`` must stay zero so packed training can mask cross-document
    labels. Degenerate or fully unsupervised samples become tombstones.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike | None = None,
        tokenizer_id: str | None = None,
        *,
        eos_token: str | None = None,
        field: str = "messages",
        max_length: int | None = None,
        chat_template: str | os.PathLike[str] | None = None,
        apply_chat_template: bool = True,
        span_source: SpanSource = "auto",
        loss_scope: LossScope | None = None,
        loss_on_last_turn_only: bool | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        tools_field: str = "tools",
        enable_thinking_field: str = "enable_thinking",
        mask_field_out: str = "loss_mask",
        preserve_upstream_payload: bool = False,
        max_batch: int = 64,
        max_latency_ms: int | None = 3,
    ) -> None:
        """Tokenize chat conversations with a chat template and loss mask.

        Args:
            tokenizer: Pre-instantiated HF-compatible **fast** tokenizer; takes
                precedence over ``tokenizer_id``.
            tokenizer_id: HF model id passed to ``AutoTokenizer.from_pretrained``
                (always loaded with ``use_fast=True`` — offset mapping and
                assistant-token masks require a fast tokenizer).
            eos_token: Optional EOS special-token override applied when
                loading ``tokenizer_id``. HF looks ``eos_token_id`` up from
                the ``eos_token`` string, so the override moves both the
                string the no-template path appends and the id the EOS
                finalizer checks. Only valid together with ``tokenizer_id``
                — a pre-instantiated ``tokenizer`` already carries its EOS.
            field: Dot-separated payload path of the messages list (each
                message a mapping with ``role`` and ``content``).
            max_length: Cap on output length in tokens, applied to the full
                rendered conversation (SFT passes ``seq_len + 1``). Over-long
                conversations are truncated mid-content — surviving assistant
                tokens stay supervised; a lost EOS is not re-appended.
            chat_template: Template override — an inline jinja string, or a
                path to a template file (read eagerly so the template travels
                with the pickled op). ``None`` uses the tokenizer's own
                template.
            apply_chat_template: ``False`` selects the no-template path (see
                class docstring); ``chat_template`` and a non-``"auto"``
                ``span_source`` are then rejected.
            span_source: ``"auto"`` picks ``generation_tags`` when the
                resolved template has ``{% generation %}`` tags, else
                ``prefix_diff`` (with a setup-time warning). Pin explicitly
                for reproducibility or testing. ``after_last_user`` requires
                explicit ``prefix_diff`` when the template is untagged.
            loss_scope: ``all_assistant`` selects every assistant span;
                ``last_span`` selects the final contiguous supervised run;
                ``after_last_user`` selects every assistant after the final user.
                Defaults to ``all_assistant``. With generation tags,
                ``after_last_user`` verifies ownership and supervision automatically.
                Verification adds rendering and tokenization at setup and for new
                message/tool formats or non-object tool argument values; successful
                checks are reused. Samples with omitted selected content are
                dropped and logged; other verification failures raise.
            loss_on_last_turn_only: Deprecated alias: True selects ``last_span``;
                False selects ``all_assistant``. Cannot accompany ``loss_scope``.
            chat_template_kwargs: Static extra kwargs forwarded to
                ``apply_chat_template`` (template context variables).
            tools_field: Dot-separated payload path of per-sample tools passed
                to the render; missing means no tools.
            enable_thinking_field: Dot-separated payload path of a per-sample
                ``enable_thinking`` flag; missing means not forwarded.
            mask_field_out: Output payload key for the loss mask.
            preserve_upstream_payload: Keep all upstream payload keys;
                otherwise the output dict contains only ``input_ids`` and the
                mask field.
            max_batch: Accumulator batch size.
            max_latency_ms: Accumulator latency bound (non-deterministic runs).
        """
        loss_scope = _resolve_loss_scope(
            loss_scope, loss_on_last_turn_only, stacklevel=3
        )
        if span_source not in _VALID_SPAN_SOURCES:
            raise ValueError(
                f"span_source must be one of {_VALID_SPAN_SOURCES}, got {span_source!r}"
            )
        if not apply_chat_template:
            if span_source != "auto":
                raise ValueError(
                    "span_source only applies to the template path; with "
                    + "apply_chat_template=False spans are exact by construction"
                )
            if chat_template is not None:
                raise ValueError(
                    "chat_template has no effect with apply_chat_template=False"
                )
        if max_length is not None and max_length < 2:
            raise ValueError(
                f"max_length={max_length} cannot hold a shifted training pair; "
                + "need at least 2 tokens"
            )
        if not field:
            raise ValueError("field must name the messages payload path")
        if not mask_field_out or mask_field_out == "input_ids":
            raise ValueError("mask_field_out must be a non-empty key != 'input_ids'")
        if chat_template_kwargs:
            reserved = _RESERVED_RENDER_KWARGS.intersection(chat_template_kwargs)
            if reserved:
                raise ValueError(
                    "chat_template_kwargs may not override operator-owned "
                    + f"apply_chat_template kwargs: {sorted(reserved)!r}"
                )

        TokenizeBase.__init__(
            self,
            tokenizer,
            tokenizer_id,
            # Offset mapping / assistant-token masks require a fast tokenizer.
            use_fast=True,
            eos_token=eos_token,
            max_batch=max_batch,
            max_latency_ms=max_latency_ms,
        )
        self.field = field
        self._field_path: tuple[str, ...] = tuple(field.split("."))
        self.max_length = max_length
        self.apply_chat_template = apply_chat_template
        self.chat_template = self._read_template(chat_template)
        self.span_source: SpanSource = span_source
        self.loss_scope: LossScope = loss_scope
        self._template_verifier: TemplateVerifier | None = None
        self._template_setup_result: TemplateSetupResult | None = None
        self.chat_template_kwargs: dict[str, Any] = dict(chat_template_kwargs or {})
        self.tools_field = tools_field
        self.enable_thinking_field = enable_thinking_field
        self._tools_path: tuple[str, ...] = tuple(tools_field.split("."))
        self._thinking_path: tuple[str, ...] = tuple(enable_thinking_field.split("."))
        self.mask_field_out = mask_field_out
        self.preserve_upstream_payload = preserve_upstream_payload

        self._resolved_span_source: SpanSource | None = None
        self._template_str: str | None = None
        self._eos_id: int | None = None
        self._eos_str: str | None = None
        # Logging-only latch; outputs stay independent across process_many calls.
        self._warned_unsupervised_drop = False
        self._require_one_to_one = False
        # Logging-only counts; sample delivery does not depend on their values.
        self._content_loss_checked = 0
        self._content_loss_dropped = 0

    def __getstate__(self) -> dict[str, Any]:
        """Serialize successful setup checks, excluding compiled Jinja state."""
        state = dict(self.__dict__)
        if self._template_verifier is not None:
            # Workers recompile Jinja and match the portable result against their
            # configuration before reusing checks. Loaded tokenizers are retained.
            state["_template_verifier"] = None
            state["_tokenizer_instantiated"] = False
        return state

    def _cache_preflight_result(self, probe: TokenizeBase) -> None:
        """Retain verified setup without copying the probe's loaded tokenizer."""
        if isinstance(probe, TokenizeChat):
            self._template_setup_result = probe._template_setup_result

    @staticmethod
    def _read_template(template: str | os.PathLike[str] | None) -> str | None:
        if template is None:
            return None
        if not isinstance(template, os.PathLike):
            try:
                if not os.path.isfile(template):
                    return template
            except (OSError, ValueError):
                return template
        with open(template, encoding="utf-8") as fh:
            return fh.read()

    # Template resolution runs inside TokenizeBase's lazy setup envelope.
    def _finalize_setup(self) -> None:
        tok = self.tok
        assert tok is not None

        eos_id = getattr(tok, "eos_token_id", None)
        self._eos_id = int(eos_id) if eos_id is not None else None
        eos_str = getattr(tok, "eos_token", None)
        self._eos_str = eos_str if isinstance(eos_str, str) else None

        # Every span source maps char spans to tokens via offsets; the tagged
        # path also needs HF's assistant-token mask support.
        if not getattr(tok, "is_fast", False):
            raise ValueError(
                "TokenizeChat requires a fast tokenizer (offset mapping / "
                + "assistant token masks); the loaded tokenizer is not fast"
            )

        if not self.apply_chat_template:
            if self._eos_str is None:
                raise ValueError(
                    "apply_chat_template=False requires a tokenizer with a "
                    + "string eos_token: the no-template path appends it to "
                    + "every assistant turn"
                )
            return

        template = self.chat_template
        if template is None:
            template = getattr(tok, "chat_template", None)
        if not isinstance(template, str) or not template:
            raise ValueError(
                "no chat template available: the tokenizer defines none and "
                + "no chat_template override was passed. Pass chat_template=..., "
                + "or set apply_chat_template=False for raw-content SFT."
            )
        self._template_str = template

        has_tags = _GENERATION_TAG_RE.search(template) is not None
        if self.span_source == "generation_tags" and not has_tags:
            raise ValueError(
                "span_source='generation_tags' but the resolved chat template "
                + "has no {% generation %} tags; add tags to the template or "
                + "use span_source='prefix_diff'"
            )
        if self.span_source == "auto":
            self._resolved_span_source = (
                "generation_tags" if has_tags else "prefix_diff"
            )
            if not has_tags and self.loss_scope == "after_last_user":
                raise ValueError(
                    "after_last_user requires generation tags or explicit span_source='prefix_diff'"
                )
            if not has_tags:
                log.warning(
                    "TokenizeChat: chat template has no {%% generation %%} tags; "
                    + "falling back to prefix-diff span reconstruction. This is "
                    + "guarded per sample and hard-errors on position-dependent "
                    + "templates — prefer a tagged template variant."
                )
        else:
            self._resolved_span_source = self.span_source

        if (
            self._resolved_span_source == "generation_tags"
            and self.loss_scope == "after_last_user"
        ):
            from zephon._internal.utils.chat_template import TemplateVerifier

            verifier = TemplateVerifier(
                tok, template, self.chat_template_kwargs, self.loss_scope
            )
            report = verifier.setup_report(cached=self._template_setup_result)
            if not report.ok:
                raise ValidationError(report)
            self._template_setup_result = verifier.setup_result
            self._template_verifier = verifier

        if self._resolved_span_source == "prefix_diff":
            # The generation prompt anchors assistant content, excluding role headers.
            probe = [{"role": "user", "content": ""}]
            renders = [
                cast(Any, tok).apply_chat_template(
                    probe,
                    tokenize=False,
                    add_generation_prompt=flag,
                    chat_template=template,
                    **self.chat_template_kwargs,
                )
                for flag in (True, False)
            ]
            if renders[0] == renders[1]:
                raise ValueError(
                    "TokenizeChat: the chat template ignores "
                    + "add_generation_prompt, so prefix-diff cannot locate where "
                    + "assistant content starts (the role header would be "
                    + "supervised). Provide a {% generation %}-tagged template "
                    + "variant via chat_template=."
                )

    # Payload access

    def _extract_messages(
        self, payload: Any, sample_id: Any
    ) -> list[Mapping[str, Any]]:
        messages = self._lookup_path(payload, self._field_path)
        if messages is _MISSING:
            raise ValueError(
                f"TokenizeChat: payload of sample {sample_id!r} has "
                + f"no {self.field!r} field"
            )
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise TypeError(
                f"TokenizeChat: field {self.field!r} of sample "
                + f"{sample_id!r} is {type(messages).__name__}, "
                + "expected a list of role/content messages"
            )
        for message in messages:
            if not isinstance(message, Mapping) or "role" not in message:
                raise TypeError(
                    f"TokenizeChat: sample {sample_id!r} contains a "
                    + "message without a 'role' mapping"
                )
        if self.loss_scope == "after_last_user":
            for message in messages:
                content = message.get("content")
                if isinstance(content, list) and any(
                    not isinstance(part, Mapping) or part.get("type") != "text"
                    for part in content
                ):
                    raise ValueError("after_last_user requires text-only content parts")
                if message.get("role") != "assistant" and message.get("tool_calls"):
                    raise ValueError("tool_calls must belong to assistant messages")
        return list(messages)

    def _render_kwargs(self, payload: Any) -> dict[str, Any]:
        kwargs = dict(self.chat_template_kwargs)
        tools = self._lookup_path(payload, self._tools_path)
        if tools is not _MISSING and tools is not None:
            kwargs["tools"] = tools
        thinking = self._lookup_path(payload, self._thinking_path)
        if thinking is not _MISSING and thinking is not None:
            kwargs["enable_thinking"] = thinking
        return kwargs

    # Span-source implementations return unshifted (ids, mask) pairs.

    def _via_generation_tags(
        self, messages: list[Mapping[str, Any]], render_kwargs: dict[str, Any]
    ) -> tuple[list[int], "np.ndarray"]:
        tok = cast(Any, self.tok)
        assert self._template_str is not None
        call_kwargs: dict[str, Any] = {
            "tokenize": True,
            "return_dict": True,
            "return_assistant_tokens_mask": True,
            "add_generation_prompt": False,
            "chat_template": thread_salted_template(self._template_str),
            **render_kwargs,
        }
        if self.max_length is not None:
            call_kwargs["truncation"] = True
            call_kwargs["max_length"] = self.max_length
        out = tok.apply_chat_template(messages, **call_kwargs)
        ids = out["input_ids"]
        # HF versions disagree on flat versus batch-of-one shape.
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        mask_raw = out["assistant_masks"]
        if mask_raw and isinstance(mask_raw[0], list):
            mask_raw = mask_raw[0]
        return ids, np.asarray(mask_raw, dtype=np.uint8)

    def _via_owned_generation_tags(
        self,
        verifier: TemplateVerifier,
        messages: list[Mapping[str, Any]],
        render_kwargs: dict[str, Any],
        sample_id: Any,
    ) -> tuple[list[int], "np.ndarray"]:
        verifier.verify_input_structure(
            messages,
            render_kwargs,
            context=f"loss_scope={self.loss_scope!r}, template={verifier.template.identity}, sample={sample_id!r}",
        )
        tok = cast("PreTrainedTokenizerFast", self.tok)
        text, spans = verifier.template.render(
            messages, {**tok.special_tokens_map, **render_kwargs}
        )
        selected = selected_assistant_indices(messages, self.loss_scope)
        rendered_owners = {owner for owner, start, end, _ in spans if end > start}
        for owner in selected - rendered_owners:
            message = messages[owner]
            if any(
                message.get(key)
                for key in (
                    "content",
                    "reasoning_content",
                    "reasoning",
                    "thinking",
                    "tool_calls",
                )
            ):
                raise ValueError(
                    f"TokenizeChat: loss_scope={self.loss_scope!r}, template="
                    + f"{verifier.template.identity}: nonempty selected assistant "
                    + f"{owner} emitted no generation range; sample={sample_id!r}"
                )
        encoding = encode_rendered_text(tok, text, max_length=self.max_length)
        return encoding["input_ids"], generation_mask(encoding, spans, selected)

    def _via_prefix_diff(
        self, messages: list[Mapping[str, Any]], render_kwargs: dict[str, Any]
    ) -> tuple[list[int], "np.ndarray"]:
        tok = cast(Any, self.tok)

        def render(msgs: list[Mapping[str, Any]], generation_prompt: bool) -> str:
            return tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=generation_prompt,
                chat_template=self._template_str,
                **render_kwargs,
            )

        full = render(messages, False)
        selected = selected_assistant_indices(messages, self.loss_scope)
        spans: list[tuple[int, int]] = []
        for i, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            prefix = render(messages[:i], True)
            upto = render(messages[: i + 1], False)
            if not (full.startswith(upto) and upto.startswith(prefix)):
                raise ValueError(
                    "TokenizeChat: chat template renders position-dependently "
                    + "(partial renders are not prefixes of the full render), so "
                    + "prefix-diff span reconstruction cannot produce a correct "
                    + "loss mask. Provide a {% generation %}-tagged template "
                    + "variant via chat_template=."
                )
            if i in selected:
                spans.append((len(prefix), len(upto)))
        return self._encode_with_spans(full, spans)

    def _via_no_template(
        self, messages: list[Mapping[str, Any]]
    ) -> tuple[list[int], "np.ndarray"]:
        eos = self._eos_str
        assert eos is not None
        parts: list[str] = []
        spans: list[tuple[int, int]] = []
        cursor = 0
        selected = selected_assistant_indices(messages, self.loss_scope)
        for i, message in enumerate(messages):
            content = message.get("content", "")
            if isinstance(content, list):
                texts: list[str] = []
                for item in content:
                    if item.get("type") != "text":
                        continue
                    if "text" not in item:
                        raise ValueError(
                            "TokenizeChat: content part with type='text' "
                            + "is missing its 'text' key"
                        )
                    texts.append(item["text"])
                content = "".join(texts)
            if not isinstance(content, str):
                content = str(content)
            if message.get("role") == "assistant":
                parts.append(content)
                parts.append(eos)
                if i in selected:
                    spans.append((cursor, cursor + len(content) + len(eos)))
                cursor += len(content) + len(eos)
            else:
                parts.append(content)
                cursor += len(content)
        return self._encode_with_spans("".join(parts), spans)

    def _encode_with_spans(
        self, text: str, spans: list[tuple[int, int]]
    ) -> tuple[list[int], "np.ndarray"]:
        enc = encode_rendered_text(
            cast("PreTrainedTokenizerFast", self.tok),
            text,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        return enc["input_ids"], _mask_from_spans(enc["offset_mapping"], spans)

    # Record processing

    def _ids_and_mask(
        self, payload: Any, sample_id: Any
    ) -> tuple[list[int], "np.ndarray"]:
        if not self._tokenizer_instantiated:
            self._setup_tokenizer()
        messages = self._extract_messages(payload, sample_id)
        if not messages:
            return [], np.zeros(0, dtype=np.uint8)

        if not self.apply_chat_template:
            ids, mask = self._via_no_template(messages)
        else:
            render_kwargs = self._render_kwargs(payload)
            if self._resolved_span_source == "generation_tags":
                if self._template_verifier is not None:
                    ids, mask = self._via_owned_generation_tags(
                        self._template_verifier, messages, render_kwargs, sample_id
                    )
                else:
                    ids, mask = self._via_generation_tags(messages, render_kwargs)
            else:
                ids, mask = self._via_prefix_diff(messages, render_kwargs)

        if self.loss_scope == "last_span":
            mask = _last_span_only(mask)

        # Templates may omit the final EOS. Copying the previous mask bit
        # trains assistant-final EOS, leaves user-final EOS unsupervised, and
        # preserves all-zero masks for the unsupervised-drop path.
        if (
            self.apply_chat_template
            and self._eos_id is not None
            and ids
            and ids[-1] != self._eos_id
            and (self.max_length is None or len(ids) < self.max_length)
        ):
            ids.append(self._eos_id)
            mask = np.append(mask, mask[-1])

        return ids, mask

    def _tokenize_delivery(
        self, payload: Any, sample_id: Any
    ) -> (
        tuple[list[int], "np.ndarray"]
        | Literal["degenerate", "unsupervised", "content_loss"]
    ):
        """Filter samples consistently during execution and calibration."""
        # Setup failures are fatal; only per-sample content omissions are dropped.
        if not self._tokenizer_instantiated:
            self._setup_tokenizer()
        checks_content = self._template_verifier is not None
        if checks_content:
            self._content_loss_checked += 1
        try:
            ids, mask = self._ids_and_mask(payload, sample_id)
        except ValidationError as error:
            codes = {issue.code for issue in error.report.errors} - {
                "CHAT_INPUT_CONTEXT"
            }
            if not checks_content or codes != {"CHAT_REQUIRED_CONTENT"}:
                raise
            self._content_loss_dropped += 1
            level = (
                logging.ERROR
                if self._content_loss_dropped <= 5
                or self._content_loss_dropped % 100 == 0
                else logging.DEBUG
            )
            log.log(
                level,
                "TokenizeChat: dropping sample %r because selected assistant "
                + "content was omitted; content-loss drops=%d/%d (%.2f%%) "
                + "in this operator instance. %s",
                sample_id,
                self._content_loss_dropped,
                self._content_loss_checked,
                100 * self._content_loss_dropped / self._content_loss_checked,
                error,
            )
            return "content_loss"
        if len(ids) < 2:
            return "degenerate"
        if not mask.any():
            return "unsupervised"
        if mask[0] == 1:
            raise ValueError(
                f"TokenizeChat: sample {sample_id!r} starts with a "
                + "supervised token (mask[0] == 1). Packed-window training masks "
                + "cross-document boundary labels only because every "
                + "conversation's first token is unsupervised; a supervised "
                + "first token would train a cross-document prediction. Start "
                + "conversations with system/user content (or a BOS token)."
            )
        return ids, mask

    def count_delivered_tokens(self, payload: Any) -> int:
        """Count sequence tokens delivered for a raw payload.

        This includes template and unsupervised tokens. Dropped samples count
        as zero so their scheduled bytes remain part of calibration.
        """
        result = self._tokenize_delivery(payload, "<calibration>")
        if isinstance(result, str):
            return 0
        ids, _ = result
        return len(ids)

    def _process_record(self, record: SampleRecord) -> list[SampleRecord]:
        if record.meta.tombstone:
            return [record]

        source = record.meta.cursor
        result = self._tokenize_delivery(record.payload, source)

        if isinstance(result, str):
            if self._require_one_to_one:
                raise one_to_one_violation(
                    "tokenize_chat",
                    f"dropped a {result.replace('_', ' ')} sample",
                    record_identity(record),
                )
            if result == "content_loss":
                return tombstones_for_record(record)
            if result == "degenerate":
                log.debug(
                    "TokenizeChat: dropping degenerate sample %r (<2 tokens)",
                    record.meta.cursor,
                )
            elif not self._warned_unsupervised_drop:
                self._warned_unsupervised_drop = True
                log.warning(
                    "TokenizeChat: dropping sample %r — no supervised tokens "
                    + "(all-zero loss mask, e.g. truncation swallowed every "
                    + "assistant token). Further drops are logged at DEBUG.",
                    record.meta.cursor,
                )
            else:
                log.debug(
                    "TokenizeChat: dropping unsupervised sample %r",
                    record.meta.cursor,
                )
            return tombstones_for_record(record)

        ids, mask = result
        upstream = record.payload if isinstance(record.payload, Mapping) else None
        payload: SamplePayloadDict = (
            dict(upstream) if (self.preserve_upstream_payload and upstream) else {}
        )
        payload["input_ids"] = np.asarray(ids, dtype=np.int64)
        payload[self.mask_field_out] = mask
        record.payload = payload
        return [record]

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        return self._process_record(elem)

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        results: list[SampleRecord] = []
        for elem in elems:
            results.extend(self._process_record(elem))
        return results

    # Op plumbing

    def setup(self, ctx: OpContext) -> None:
        super().setup(ctx)
        self._require_one_to_one = bool(ctx.get(REQUIRE_ONE_TO_ONE_RECORDS))

    def traits(self) -> OpTraits:
        return OpTraits(
            calibration_replayable=True,
            indexable=True,
            preserves_cursor_order=True,
            one_to_one_records=True,
            parallelism=4,
        )

    def token_counting_spec(self) -> "ChatTokenCountingSpec":
        return ChatTokenCountingSpec.from_op(self)


# ---------------------------------------------------------------------------
# Token-counting spec (priming calibration)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatTokenCountingSpec(TokenCountingSpec):
    """Serializable ``TokenizeChat`` settings that affect token counts."""

    field: str = "messages"
    max_length: int | None = None
    chat_template: str | None = None
    apply_chat_template: bool = True
    span_source: SpanSource = "auto"
    loss_scope: LossScope = "all_assistant"
    chat_template_kwargs: tuple[tuple[str, Any], ...] = ()
    tools_field: str = "tools"
    enable_thinking_field: str = "enable_thinking"
    # Moves the appended/checked EOS, so it changes delivered counts.
    eos_token: str | None = None

    @classmethod
    def from_op(cls, op: TokenizeChat) -> "ChatTokenCountingSpec":
        return cls(
            tokenizer=op.tok,
            tokenizer_id=op.tokenizer_id,
            field=op.field,
            max_length=op.max_length,
            chat_template=op.chat_template,
            apply_chat_template=op.apply_chat_template,
            span_source=op.span_source,
            loss_scope=op.loss_scope,
            chat_template_kwargs=tuple(sorted(op.chat_template_kwargs.items())),
            tools_field=op.tools_field,
            enable_thinking_field=op.enable_thinking_field,
            eos_token=op.eos_token,
        )

    def build_counter(self) -> DeliveredTokenCounter:
        return _ChatCounter(
            TokenizeChat(
                self.tokenizer,
                self.tokenizer_id,
                eos_token=self.eos_token,
                field=self.field,
                max_length=self.max_length,
                chat_template=self.chat_template,
                apply_chat_template=self.apply_chat_template,
                span_source=self.span_source,
                loss_scope=self.loss_scope,
                chat_template_kwargs=dict(self.chat_template_kwargs),
                tools_field=self.tools_field,
                enable_thinking_field=self.enable_thinking_field,
            )
        )


@dataclass(frozen=True)
class _ChatPlan(CountPlan):
    op: TokenizeChat

    @property
    def description(self) -> str:
        return f"chat, field: {self.op.field}"

    def count(self, payload: Any) -> int | None:
        # Errors on the shared delivery path are deterministic — the run hits
        # them too — so mark them fatal rather than have priming sample around.
        try:
            return self.op.count_delivered_tokens(payload)
        except Exception as exc:
            raise FatalCountError(f"chat delivery failed: {exc}") from exc


@dataclass(frozen=True)
class _ChatCounter(DeliveredTokenCounter):
    op: TokenizeChat

    def plan(self, sample_payloads: list[Any]) -> CountPlan:
        # Chat payloads do not fit text/pretokenized voting; use delivery directly.
        return _ChatPlan(self.op)
