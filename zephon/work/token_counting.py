# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Delivered-token counting for token-aware mixture priming."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import cloudpickle
import numpy as np

from zephon.utils.tokenizer import load_hf_tokenizer

if TYPE_CHECKING:
    from zephon.ops.tokenize_text import TokenizeText

#: Text keys tried in priority order when no field is configured.
_COMMON_TEXT_KEYS = ("text", "content", "document", "body", "markdown", "raw_content")

#: Token-array keys tried the same way for already-tokenized payloads.
_COMMON_TOKEN_KEYS = ("input_ids", "tokens", "token_ids")


# ---------------------------------------------------------------------------
# Calibration payload heuristics
# ---------------------------------------------------------------------------


def _lookup_field_path(payload: Any, path: tuple[str, ...] | None) -> Any:
    value = payload
    for key in path or ():  # falsy path resolves to the payload itself
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _as_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return None
    return None


def choose_text_field(payload: Any) -> str | None:
    """Choose a likely text field, preferring known body keys."""
    if not isinstance(payload, Mapping):
        return None
    for key in _COMMON_TEXT_KEYS:
        if key in payload and _as_text(payload[key]) is not None:
            return key
    candidates = {
        key: text
        for key, value in payload.items()
        if isinstance(key, str) and (text := _as_text(value)) is not None
    }
    if not candidates:
        return None
    if len(candidates) == 1:
        return next(iter(candidates))
    return max(candidates, key=lambda k: (len(candidates[k]), k))


def extract_text(payload: Any, path: tuple[str, ...] | None = None) -> str | None:
    """Extract calibration text from a configured path or text-like payload."""
    if path:
        text = _as_text(_lookup_field_path(payload, path))
        if text is not None:
            return text
    return _as_text(payload)


def _token_array_length(value: Any) -> int | None:
    if isinstance(value, np.ndarray):
        return int(value.size) if np.issubdtype(value.dtype, np.integer) else None
    if isinstance(value, (list, tuple)):
        head = value[0] if value else None
        return (
            len(value) if isinstance(head, int) and not isinstance(head, bool) else None
        )
    is_float = getattr(value, "is_floating_point", None)  # torch.Tensor, duck-typed
    if callable(is_float) and hasattr(value, "numel"):
        try:
            if not is_float() and not getattr(value, "is_complex", lambda: False)():
                return int(value.numel())
        except Exception:  # noqa: BLE001 — unknown tensor-likes fall through to text
            return None
    return None


def _pretokenized_field(
    sample_payload: Any, field_path: tuple[str, ...] | None
) -> tuple[str, ...] | None:
    """Find a token-array path; ``()`` denotes the payload itself."""
    if field_path is not None:
        found = _token_array_length(_lookup_field_path(sample_payload, field_path))
        return field_path if found is not None else None
    if _token_array_length(sample_payload) is not None:
        return ()
    if isinstance(sample_payload, Mapping):
        for key in _COMMON_TOKEN_KEYS:
            if _token_array_length(sample_payload.get(key)) is not None:
                return (key,)
    return None


def _text_field_vote(
    payload: Any, field_path: tuple[str, ...] | None
) -> tuple[str, ...] | None:
    """Return a text-field vote; ``()`` denotes the payload itself."""
    if field_path is not None:
        found = _as_text(_lookup_field_path(payload, field_path)) is not None
        return field_path if found else None
    chosen = choose_text_field(payload)
    if chosen is not None:
        return (chosen,)
    return () if _as_text(payload) is not None else None


def _common_key_rank(path: tuple[str, ...]) -> int:
    if len(path) == 1 and path[0] in _COMMON_TEXT_KEYS:
        return len(_COMMON_TEXT_KEYS) - _COMMON_TEXT_KEYS.index(path[0])
    return 0


def _field_label(path: tuple[str, ...] | None) -> str:
    return ".".join(path) if path else "payload directly"


def _count_raw_tokens(tokenizer: Any, text: str, spec: "TextTokenCountingSpec") -> int:
    add_specials = spec.special_tokens == "tokenizer_default"
    result = tokenizer(text, add_special_tokens=add_specials)
    input_ids = result["input_ids"] if isinstance(result, Mapping) else result.input_ids
    return len(input_ids)


# ---------------------------------------------------------------------------
# Counting contract
# ---------------------------------------------------------------------------


class FatalCountError(ValueError):
    """A deterministic delivery error that execution will hit too.

    Calibration shares the op's delivery path, so this is not a measurement miss
    to sample around — the same row would crash the run. Priming re-raises it
    instead of falling back, surfacing the problem upfront rather than mid-run.
    """


class CountPlan(ABC):
    """Per-dataset strategy mapping one calibration payload to delivered tokens."""

    # Whether a counting error invalidates the dataset instead of one payload.
    abort_on_error: ClassVar[bool] = False

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def count(self, payload: Any) -> int | None:
        """Count delivered tokens, or return ``None`` when unmeasurable.

        Zero is a measurement and contributes to both ratio and coverage.
        """


class DeliveredTokenCounter(ABC):
    """Build dataset-specific counting plans within a worker."""

    @abstractmethod
    def plan(self, sample_payloads: list[Any]) -> CountPlan:
        pass


@dataclass(frozen=True)
class TokenCountingSpec(ABC):
    """Serializable settings for building a delivered-token counter."""

    tokenizer: Any | None = None
    tokenizer_id: str | None = None

    @abstractmethod
    def build_counter(self) -> DeliveredTokenCounter:
        pass

    def __getstate__(self) -> dict[str, Any]:
        # std pickle cannot carry every live tokenizer; cloudpickle can.
        state = dict(self.__dict__)
        if state["tokenizer"] is not None:
            state["tokenizer"] = cloudpickle.dumps(state["tokenizer"])
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if isinstance(state.get("tokenizer"), bytes):
            state = {**state, "tokenizer": cloudpickle.loads(state["tokenizer"])}
        for key, value in state.items():
            object.__setattr__(self, key, value)


# ---------------------------------------------------------------------------
# Text counting (also the default when the pipeline has no tokenize op)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextTokenCountingSpec(TokenCountingSpec):
    """Token-counting settings for ``TokenizeText`` and default calibration."""

    field: str | None = None
    max_length: int | None = None
    truncation: bool = False
    split_long_samples: bool = False
    special_tokens: str = "bos_eos"
    use_fast: bool | None = True

    @classmethod
    def from_op(cls, op: "TokenizeText") -> "TextTokenCountingSpec":
        if op.padding:
            warnings.warn(
                f"[zephon] TokenizeText padding={op.padding!r} is ignored by "
                "token-aware mixture calibration: the estimate counts the "
                "training tokens a sample maps to, and padding is packing waste "
                "no dataset owns.",
                stacklevel=2,
            )
        return cls(
            tokenizer=op.tok,
            tokenizer_id=op.tokenizer_id,
            field=op.field,
            max_length=op.max_length,
            truncation=op.truncation,
            split_long_samples=op.split_long_samples,
            special_tokens=op.special_tokens,
            use_fast=op.use_fast,
        )

    @property
    def num_specials(self) -> int:
        prepend = 1 if self.special_tokens in ("bos_eos", "bos") else 0
        append = 1 if self.special_tokens in ("bos_eos", "eos") else 0
        return prepend + append

    def delivered_tokens(self, raw_tokens: int) -> int:
        """Compute delivery length from a content-token count.

        In ``tokenizer_default`` mode, ``raw_tokens`` already includes
        tokenizer-added special tokens.
        """
        if self.special_tokens == "tokenizer_default":
            if self.truncation and self.max_length is not None:
                return min(raw_tokens, self.max_length)
            return raw_tokens

        specials = self.num_specials
        if self.split_long_samples:
            # Splitting preserves token mass; BOS/EOS land on the first/last chunk.
            return raw_tokens + specials
        if self.truncation and self.max_length is not None:
            return min(raw_tokens, self.max_length - specials) + specials
        return raw_tokens + specials

    def build_counter(self) -> DeliveredTokenCounter:
        tokenizer = self.tokenizer
        if tokenizer is None:
            tokenizer = load_hf_tokenizer(self.tokenizer_id, use_fast=self.use_fast)
        return _TextCounter(tokenizer, self)


@dataclass(frozen=True)
class _PretokenizedPlan(CountPlan):
    path: tuple[str, ...] | None

    @property
    def description(self) -> str:
        return f"pretokenized, field: {_field_label(self.path)}"

    def count(self, payload: Any) -> int | None:
        length = _token_array_length(_lookup_field_path(payload, self.path))
        # An empty token array is unmeasurable, not a zero-yield sample.
        return length if length else None


@dataclass(frozen=True)
class _TextPlan(CountPlan):
    tokenizer: Any
    spec: TextTokenCountingSpec
    path: tuple[str, ...] | None

    @property
    def description(self) -> str:
        return f"text, field: {_field_label(self.path)}"

    def count(self, payload: Any) -> int | None:
        text = extract_text(payload, self.path)
        if text is None:
            return None
        delivered = self.spec.delivered_tokens(
            _count_raw_tokens(self.tokenizer, text, self.spec)
        )
        # Empty text is unmeasurable, not a zero-yield sample.
        return delivered or None


@dataclass(frozen=True)
class _TextCounter(DeliveredTokenCounter):
    tokenizer: Any
    spec: TextTokenCountingSpec

    def plan(self, sample_payloads: list[Any]) -> CountPlan:
        """Choose a plan by vote, preferring exact token arrays on ties."""
        field_path = tuple(self.spec.field.split(".")) if self.spec.field else None
        token_votes: Counter[tuple[str, ...]] = Counter()
        text_votes: Counter[tuple[str, ...]] = Counter()
        for payload in sample_payloads:
            token_path = _pretokenized_field(payload, field_path)
            if token_path is not None:
                token_votes[token_path] += 1
                continue
            text_path = _text_field_vote(payload, field_path)
            if text_path is not None:
                text_votes[text_path] += 1
        if token_votes and sum(token_votes.values()) >= sum(text_votes.values()):
            return _PretokenizedPlan(token_votes.most_common(1)[0][0])
        if field_path is not None:
            return _TextPlan(self.tokenizer, self.spec, field_path)
        if not text_votes:
            return _TextPlan(self.tokenizer, self.spec, None)
        best = max(text_votes, key=lambda p: (text_votes[p], _common_key_rank(p)))
        return _TextPlan(self.tokenizer, self.spec, best)


__all__ = [
    "CountPlan",
    "DeliveredTokenCounter",
    "TextTokenCountingSpec",
    "TokenCountingSpec",
    "choose_text_field",
    "extract_text",
]
