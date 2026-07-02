# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Token-cost model for the token-aware mixture work source.

The work source rations sample *pointers* and never sees payloads, so a token
mixture can only be enforced through an estimated per-sample token cost:
``shard_avg_bytes x tokens_per_byte``. The byte term comes from the node-local
shard catalog (``raw_bytes`` / ``num_rows``, mmap-backed, zero new I/O); the
per-dataset ratio is supplied by a separate calibration pass.

The estimate is heuristic and best-effort: it only needs to be good enough that
residual mixture drift stays small (the bounded ``ensure_mixture`` buffer
absorbs zero-mean noise). Users override it via pinned ratios or a ``measure``
callable.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args

import numpy as np

from zephon.core.constants import SampleId
from zephon.io.dataset import Dataset
from zephon.ops.tokenize_text import TokenizeText
from zephon.utils.tokenizer import fallback_tokenizer
from zephon.utils.tokenizer_cloud import resolve_tokenizer_id_with_retry

logger = logging.getLogger(__name__)

#: Standard ~4 bytes/token BPE heuristic for English prose.
DEFAULT_FALLBACK_TOKENS_PER_BYTE = 0.25

#: Text keys tried in priority order when no field is configured.
_COMMON_TEXT_KEYS = ("text", "content", "document", "body", "markdown", "raw_content")

#: Token-array keys tried the same way for already-tokenized payloads.
_COMMON_TOKEN_KEYS = ("input_ids", "tokens", "token_ids")

RatioSource = Literal["measured", "pinned", "fallback"]


@dataclass(frozen=True)
class TokenRatio:
    """A primed tokens-per-byte ratio for one dataset, with provenance."""

    tokens_per_byte: float
    source: RatioSource

    def __post_init__(self) -> None:
        if self.source not in get_args(RatioSource):
            raise ValueError(f"unknown token-ratio source {self.source!r}")
        ratio = self.tokens_per_byte
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"token ratio must be finite and positive, got {ratio!r}")

    def to_state(self) -> list[Any]:
        """Checkpoint shape: ``[ratio, source]`` (msgpack/json friendly)."""
        return [float(self.tokens_per_byte), self.source]

    @classmethod
    def from_state(cls, raw: Any) -> "TokenRatio":
        ratio, source = raw
        return cls(tokens_per_byte=float(ratio), source=source)


@dataclass(frozen=True)
class TokenEstimation:
    """Configuration for token-cost estimation (token-aware mixtures).

    Passing an instance of this to a work source is what selects the
    token-aware allocation mode; leaving it ``None`` keeps sample-based mixing.

    Attributes:
        primer: How per-dataset tokens/byte ratios are obtained.

            * ``"measure"`` (default): calibration fetch + tokenize at prime
              time, deterministic in ``(seed, datasets, tokenize config)``.
            * a ``{dataset_name: tokens_per_byte}`` mapping: pins the listed
              datasets and *measures the rest* (partial pins merge over
              measured values).
            * a single ``float``: one global tokens/byte for every dataset,
              no measurement (and no tokenizer needed).
        measure: Escape hatch replacing text extraction + tokenization
            entirely: a callable mapping a fetched payload to its delivered
            token count (weird schemas, VLM cost units). Only consulted when
            a dataset is actually measured.
        calibration_samples: Records measured per dataset (size-proportional
            draws for catalog-backed datasets, scattered offsets otherwise).
        calibration_shards_min: Shards scanned per dataset when its shards
            are homogeneous. The count is chosen per dataset from the
            catalog's per-shard bytes/row spread (free metadata): census
            error scales as CV/sqrt(shards), so heterogeneous datasets
            automatically scan up to ``calibration_shards_max`` while uniform
            ones stay at the minimum. Each scanned shard costs a download +
            decode at prime time.
        calibration_shards_max: Upper bound for the adaptive shard count
            (also used when there is no catalog to read the spread from).
        fallback_tokens_per_byte: Ratio used when a dataset cannot be
            measured (no text found, measurement failed, or ``primer`` is a
            float). See :data:`DEFAULT_FALLBACK_TOKENS_PER_BYTE`.
    """

    primer: Literal["measure"] | Mapping[str, float] | float = "measure"
    measure: Callable[[Any], int] | None = None
    calibration_samples: int = 2048
    calibration_shards_min: int = 4
    calibration_shards_max: int = 16
    fallback_tokens_per_byte: float = DEFAULT_FALLBACK_TOKENS_PER_BYTE

    def __post_init__(self) -> None:
        if isinstance(self.primer, str):
            if self.primer != "measure":
                raise ValueError(
                    f"primer must be 'measure', a per-dataset mapping, or a "
                    f"float; got {self.primer!r}"
                )
        elif isinstance(self.primer, bool):
            raise ValueError("primer must not be a bool")
        elif isinstance(self.primer, (int, float)):
            if not math.isfinite(self.primer) or self.primer <= 0:
                raise ValueError(
                    f"primer as a global ratio must be finite and positive, "
                    f"got {self.primer!r}"
                )
        elif isinstance(self.primer, Mapping):
            for name, value in self.primer.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(
                        f"primer[{name!r}]={value!r} must be a finite positive "
                        f"tokens/byte ratio"
                    )
        else:
            raise ValueError(
                f"primer must be 'measure', a per-dataset mapping, or a float; "
                f"got {type(self.primer).__name__}"
            )
        if self.calibration_samples <= 0:
            raise ValueError("calibration_samples must be positive")
        if self.calibration_shards_min <= 0:
            raise ValueError("calibration_shards_min must be positive")
        if self.calibration_shards_max < self.calibration_shards_min:
            raise ValueError("calibration_shards_max must be >= calibration_shards_min")
        if (
            not math.isfinite(self.fallback_tokens_per_byte)
            or self.fallback_tokens_per_byte <= 0
        ):
            raise ValueError("fallback_tokens_per_byte must be finite and positive")


@dataclass(frozen=True)
class TokenizeProfile:
    """The ``TokenizeText`` settings that affect delivered-token counts."""

    tokenizer: Any | None = None
    tokenizer_id: str | None = None
    field: str | None = None
    max_length: int | None = None
    truncation: bool = False
    split_long_samples: bool = False
    special_tokens: str = "bos_eos"

    @classmethod
    def from_op(cls, op: TokenizeText) -> "TokenizeProfile":
        """Capture the token-count-relevant fields of a ``TokenizeText`` op."""
        return cls(
            tokenizer=op.tok,
            tokenizer_id=op.tokenizer_id,
            field=op.field,
            max_length=op.max_length,
            truncation=op.truncation,
            split_long_samples=op.split_long_samples,
            special_tokens=op.special_tokens,
        )

    @property
    def num_specials(self) -> int:
        prepend = 1 if self.special_tokens in ("bos_eos", "bos") else 0
        append = 1 if self.special_tokens in ("bos_eos", "eos") else 0
        return prepend + append

    def delivered_tokens(self, raw_tokens: int) -> int:
        """Closed-form delivered-token count from a raw content-token count.

        ``raw_tokens`` excludes operator-added specials in bracket modes, and
        is the template-included count for ``tokenizer_default`` (where HF owns
        specials and truncates the whole sequence to ``max_length``).
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


# ---------------------------------------------------------------------------
# Byte sources
# ---------------------------------------------------------------------------


class PerShardByteSize:
    """Per-shard average bytes/sample for one dataset.

    ``mean_bytes`` is the dataset-wide fallback for shards absent from
    ``shard_avg_bytes``.
    """

    def __init__(self, shard_avg_bytes: dict[int, float], mean_bytes: float) -> None:
        self.shard_avg_bytes = shard_avg_bytes
        self.mean_bytes = max(1.0, float(mean_bytes))


def build_byte_source(dataset: Dataset) -> PerShardByteSize | None:
    """Build per-shard byte sizes from Dataset byte accessors."""
    ids = dataset.ids()
    rows = dataset.counts()
    raw = dataset.raw_bytes()
    total_rows = int(rows.sum())
    total_bytes = int(raw.sum())
    if total_rows <= 0 or total_bytes <= 0:
        return None
    shard_avg = {
        sid: b / n
        for sid, n, b in zip(ids.tolist(), rows.tolist(), raw.tolist())
        if n > 0 and b > 0
    }
    return PerShardByteSize(shard_avg, total_bytes / total_rows)


class PerShardTokenCost:
    """Per-(dataset, shard) token costs from byte metadata and ratios.

    Built lazily per process; checkpoints carry only compact primed ratios.
    """

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        ratios: Mapping[str, TokenRatio],
    ) -> None:
        self._cost_by_shard: dict[str, dict[int, float]] = {}
        self._mean_cost: dict[str, float] = {}
        missing: list[str] = []
        for name, dataset in datasets.items():
            ratio = ratios[name].tokens_per_byte
            source = build_byte_source(dataset)
            if source is None:
                missing.append(name)
                self._cost_by_shard[name] = {}
                self._mean_cost[name] = max(1.0, ratio)
                continue
            self._cost_by_shard[name] = {
                shard_id: max(1.0, avg * ratio)
                for shard_id, avg in source.shard_avg_bytes.items()
            }
            self._mean_cost[name] = max(1.0, source.mean_bytes * ratio)
        if missing:
            warnings.warn(
                f"[zephon] token-aware mixture: no byte metadata for datasets "
                f"{sorted(missing)}; falling back to a flat per-dataset "
                f"estimate. Between-dataset correction still applies, but "
                f"shard-level variation is invisible for these datasets.",
                RuntimeWarning,
                stacklevel=2,
            )

    def cost(self, name: str, sample_id: SampleId) -> float:
        """Estimated delivered tokens for one drawn sample (>= 1)."""
        got = self._cost_by_shard[name].get(sample_id[1])
        if got is None:
            return self._mean_cost[name]
        return got

    def mean_cost(self, name: str) -> float:
        """Mean estimated tokens/sample for the dataset (>= 1)."""
        return self._mean_cost[name]


# ---------------------------------------------------------------------------
# Text extraction (calibration)
# ---------------------------------------------------------------------------


def _lookup_field_path(payload: Any, path: tuple[str, ...] | None) -> Any:
    value = payload
    for key in path or ():  # falsy path resolves to the payload itself
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _as_text(value: Any) -> str | None:
    """Return str values directly and UTF-8 bytes as text."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return None
    return None


def choose_text_field(payload: Any) -> str | None:
    """Pick a likely text field from a mapping payload.

    Known body keys win; otherwise choose the only or longest text-like value.
    """
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
    """Return the element count for integer token arrays, else ``None``.

    Handles numpy arrays, integer torch tensors (duck-typed), and int lists.
    """
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
    """Return the token-array field path; ``()`` means the payload itself."""
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


# ---------------------------------------------------------------------------
# Tokenizer plumbing (calibration)
# ---------------------------------------------------------------------------


def _instantiate_tokenizer(tokenize_profile: TokenizeProfile) -> Any:
    """Build the calibration tokenizer the same way ``TokenizeText`` would."""
    if tokenize_profile.tokenizer is not None:
        return tokenize_profile.tokenizer
    if tokenize_profile.tokenizer_id in (None, "__fallback__"):
        return fallback_tokenizer()
    # transformers is an optional dep; imported only when loading a real model.
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    load_id = resolve_tokenizer_id_with_retry(tokenize_profile.tokenizer_id)
    return AutoTokenizer.from_pretrained(load_id)


def _count_raw_tokens(
    tokenizer: Any, text: str, tokenize_profile: TokenizeProfile
) -> int:
    """Count tokenizer output in the units expected by ``delivered_tokens``."""
    add_specials = tokenize_profile.special_tokens == "tokenizer_default"
    result = tokenizer(text, add_special_tokens=add_specials)
    input_ids = result["input_ids"] if isinstance(result, Mapping) else result.input_ids
    return len(input_ids)
