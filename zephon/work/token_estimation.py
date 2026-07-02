# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Token-cost model for the token-aware mixture work source.

The work source rations sample *pointers* and never sees payloads, so a token
mixture can only be enforced through an estimated per-sample token cost:
``shard_avg_bytes x tokens_per_byte``. The byte term comes from the node-local
shard catalog (``raw_bytes`` / ``num_rows``, mmap-backed, zero new I/O); the
per-dataset ratio comes from the calibration pass (:func:`prime_token_ratios`).

The estimate is heuristic and best-effort: it only needs to be good enough that
residual mixture drift stays small (the bounded ``ensure_mixture`` buffer
absorbs zero-mean noise). Users override it via pinned ratios or a ``measure``
callable.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import random
import time
import warnings
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import cloudpickle
import numpy as np
from filelock import FileLock

from zephon.core.constants import SampleId
from zephon.io import build_multi_dataset_store
from zephon.io.catalog import resolve_catalog_dir, set_catalog_dir
from zephon.io.dataset import Dataset
from zephon.io.options import StoreOptions
from zephon.observability.size_estimator import content_bytes
from zephon.ops.tokenize_text import TokenizeText
from zephon.utils.atomic import atomic_write_bytes
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

    def __getstate__(self) -> dict[str, Any]:
        # std pickle (e.g. DataLoader spawn) cannot carry a lambda/closure
        # measure; wrap it in cloudpickle bytes so every transport works.
        state = dict(self.__dict__)
        if state["measure"] is not None:
            state["measure"] = cloudpickle.dumps(state["measure"])
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if isinstance(state.get("measure"), bytes):
            state = {**state, "measure": cloudpickle.loads(state["measure"])}
        for key, value in state.items():
            object.__setattr__(self, key, value)


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


# ---------------------------------------------------------------------------
# Priming
# ---------------------------------------------------------------------------


@dataclass
class _DatasetMeasurement:
    """Outcome of measuring one dataset."""

    ratio: TokenRatio
    reason: str | None = None
    retryable: bool = False  # fallback was a transient I/O failure, not cacheable


def _fetch_payloads(
    store: Any, dataset_id: int, sample_ids: list[SampleId]
) -> dict[SampleId, Any]:
    """Fetch calibration payloads keyed by sample id (per-shard, mirrors FetchOp)."""
    by_shard: dict[int, list[SampleId]] = {}
    for sid in sample_ids:
        by_shard.setdefault(sid[1], []).append(sid)
    view = store.for_dataset(dataset_id)
    payloads: dict[SampleId, Any] = {}
    for shard_id, sids in by_shard.items():
        shard, _ = view.open(shard_id)
        ordered = sorted(sids, key=lambda s: s[2])
        got = shard.getsamples([s[2] for s in ordered])
        rows = got[0] if isinstance(got, tuple) else got
        payloads.update(zip(ordered, rows))
    return payloads


#: Row cap per shard for a measured payload census (no format metadata).
_CENSUS_MAX_ROWS_PER_SHARD = 65536
_CENSUS_BATCH = 2048


@dataclass(frozen=True)
class _CalibrationScan:
    """Payload-byte census over selected calibration shards.

    ``sizes`` feeds PPS selection; payload/raw totals rebase measured payload
    bytes onto the raw-byte basis used by the cost model.
    """

    sizes: dict[tuple[int, int], int]
    payload_total: float
    raw_total: float


#: Target relative error for the payload/raw census: CV / sqrt(scanned shards).
_CENSUS_TARGET_REL_ERR = 0.03


def _calibration_shard_count(cv: float, lo: int, hi: int) -> int:
    """Shards needed so census error (~cv/sqrt(n)) meets the target, clamped."""
    if cv <= 0:
        return lo
    needed = math.ceil((cv / _CENSUS_TARGET_REL_ERR) ** 2)
    return max(lo, min(hi, needed))


def _shard_bytes_cv(rows: np.ndarray, raw: np.ndarray) -> float:
    """Row-weighted CV of per-shard bytes/row from catalog metadata."""
    valid = rows > 0
    if not valid.any():
        return 0.0
    per_row = raw[valid].astype(float) / rows[valid].astype(float)
    weights = rows[valid].astype(float)
    mean = float(np.average(per_row, weights=weights))
    if mean <= 0:
        return 0.0
    var = float(np.average((per_row - mean) ** 2, weights=weights))
    return math.sqrt(var) / mean


def _census_shard(
    view: Any, shard_id: int, count: int
) -> tuple[dict[tuple[int, int], int], float]:
    """Decode evenly-spaced rows of one shard: per-record sizes + mean bytes/row.

    Even spacing stays order-neutral at bounded cost; sizes seed PPS and the
    mean scales to a shard payload-byte total.
    """
    step = max(1, math.ceil(count / _CENSUS_MAX_ROWS_PER_SHARD))
    offsets = list(range(0, count, step))
    shard, _ = view.open(shard_id)
    sizes: dict[tuple[int, int], int] = {}
    size_sum = 0
    for start in range(0, len(offsets), _CENSUS_BATCH):
        chunk = offsets[start : start + _CENSUS_BATCH]
        got = shard.getsamples(chunk)
        batch = got[0] if isinstance(got, tuple) else got
        for offset, payload in zip(chunk, batch):
            size = content_bytes(payload)
            sizes[(shard_id, offset)] = size
            size_sum += size
    return sizes, size_sum / len(offsets)


# Per-dataset scan fanout. Composes with MAX_PRIME_PROCS: up to 128 concurrent
# shard reads per node, and the per-node cache means every node primes at once.
MAX_SCAN_THREADS = 8


def _scan_calibration_shards(
    dataset: Dataset,
    store: Any,
    dataset_id: int,
    shards_min: int,
    shards_max: int,
    seed: int,
) -> _CalibrationScan | None:
    """Census selected shards for PPS sizes and payload/raw totals."""
    ids = dataset.ids()
    rows = dataset.counts()
    raw = dataset.raw_bytes()
    if ids.size == 0:
        return None

    cv = _shard_bytes_cv(rows, raw)
    n_shards = min(_calibration_shard_count(cv, shards_min, shards_max), int(ids.size))
    logger.info(
        "calibration for %s: shard bytes/row CV %.1f%% -> scanning %d shards",
        dataset.name,
        100 * cv,
        n_shards,
    )
    rng = random.Random((seed << 16) ^ (dataset_id + 1))
    positions = sorted(rng.sample(range(int(ids.size)), n_shards))

    view = store.for_dataset(dataset_id)
    work = [
        (int(ids[pos]), n, float(raw[pos]))
        for pos in positions
        if (n := int(rows[pos])) > 0
    ]
    sizes: dict[tuple[int, int], int] = {}
    payload_total = 0.0
    raw_total = 0.0
    if work:
        # Overlap shard I/O and GIL-released decodes; aggregate on this thread.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(MAX_SCAN_THREADS, len(work))
        ) as pool:
            futures = {
                pool.submit(_census_shard, view, shard_id, count): (count, rb)
                for shard_id, count, rb in work
            }
            for fut in concurrent.futures.as_completed(futures):
                count, rb = futures[fut]
                shard_sizes, mean_bytes_per_row = fut.result()
                sizes.update(shard_sizes)
                payload_total += mean_bytes_per_row * count
                raw_total += rb
    if payload_total <= 0 or raw_total <= 0:
        return None
    return _CalibrationScan(sizes, payload_total, raw_total)


def _pps_select(
    dataset_id: int,
    sizes: Mapping[tuple[int, int], int],
    n_draws: int,
    seed: int,
) -> tuple[list[SampleId], list[int]]:
    """Draw records with probability proportional to payload size.

    PPS avoids under-sampling rare huge records in heavy-tailed datasets. Returns
    unique records in fetch order plus their draw counts; deterministic in
    ``(seed, dataset_id)``.
    """
    # Sort to hide nondeterministic census-completion order from seeded draws.
    records = sorted(rec for rec, s in sizes.items() if s > 0)
    if not records:
        return [], []
    weights = np.array([sizes[rec] for rec in records], dtype=float)
    probs = weights / weights.sum()
    rng = np.random.default_rng((seed << 24) ^ (dataset_id + 1))
    drawn = rng.choice(len(records), size=n_draws, replace=True, p=probs)
    counts = np.bincount(drawn, minlength=len(records))
    ordered = sorted(
        (records[i], int(counts[i])) for i in range(len(records)) if counts[i] > 0
    )
    return (
        [(dataset_id, sid, offset) for (sid, offset), _ in ordered],
        [c for _, c in ordered],
    )


def _hansen_hurwitz_ratio(measured: list[tuple[int, int, int]]) -> float:
    """Hansen-Hurwitz byte-weighted tokens/payload-byte estimate."""
    weighted = 0.0
    total_draws = 0
    for tokens, nbytes, draws in measured:
        weighted += draws * tokens / nbytes
        total_draws += draws
    return weighted / total_draws


def _fallback(
    estimation: TokenEstimation,
    scan: _CalibrationScan | None,
    reason: str,
    *,
    retryable: bool = False,
) -> _DatasetMeasurement:
    """Fallback ratio, rebased onto raw-byte costs when a census succeeded.

    ``retryable`` marks transient I/O fallbacks that should not be cached.
    """
    tokens_per_byte = estimation.fallback_tokens_per_byte
    if scan is not None:
        tokens_per_byte *= scan.payload_total / scan.raw_total
    return _DatasetMeasurement(
        TokenRatio(tokens_per_byte, "fallback"), reason=reason, retryable=retryable
    )


@dataclass(frozen=True)
class _DeliveryPlan:
    """How one dataset's calibration tokens are counted."""

    mode: Literal["measure callable", "pretokenized", "text"]
    path: tuple[str, ...] | None

    @property
    def field_label(self) -> str:
        return ".".join(self.path) if self.path else "payload directly"

    @property
    def description(self) -> str:
        if self.mode == "measure callable":
            return self.mode
        return f"{self.mode}, field: {self.field_label}"


# Plan from a payload prefix so one anomalous record cannot steer the dataset's
# measurement mode; gate on measured draw mass so a plan/data mismatch is loud.
_PLAN_SAMPLE_COUNT = 8
_MIN_PLAN_COVERAGE = 0.5


def _text_field_vote(
    payload: Any, field_path: tuple[str, ...] | None
) -> tuple[str, ...] | None:
    """The text path this payload could be measured by; ``None`` abstains.

    ``()`` votes for tokenizing the payload itself (plain string/bytes rows).
    """
    if field_path is not None:
        found = _as_text(_lookup_field_path(payload, field_path)) is not None
        return field_path if found else None
    chosen = choose_text_field(payload)
    if chosen is not None:
        return (chosen,)
    return () if _as_text(payload) is not None else None


def _common_key_rank(path: tuple[str, ...]) -> int:
    """Tie-break rank: common body keys beat heuristic picks, in list order."""
    if len(path) == 1 and path[0] in _COMMON_TEXT_KEYS:
        return len(_COMMON_TEXT_KEYS) - _COMMON_TEXT_KEYS.index(path[0])
    return 0


def _plan_delivery(
    estimation: TokenEstimation,
    field_path: tuple[str, ...] | None,
    sample_payloads: list[Any],
) -> _DeliveryPlan:
    """Choose the token-counting strategy by vote over a payload prefix.

    Undecidable payloads abstain; ties prefer pretokenized (exact counts).
    """
    if estimation.measure is not None:
        return _DeliveryPlan("measure callable", None)
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
        return _DeliveryPlan("pretokenized", token_votes.most_common(1)[0][0])
    if field_path is not None:
        return _DeliveryPlan("text", field_path)
    if not text_votes:
        return _DeliveryPlan("text", None)
    best = max(text_votes, key=lambda p: (text_votes[p], _common_key_rank(p)))
    return _DeliveryPlan("text", best)


def _measure_dataset(
    dataset: Dataset,
    dataset_id: int,
    store: Any,
    estimation: TokenEstimation,
    tokenize_profile: TokenizeProfile | None,
    tokenizer: Any,
    seed: int,
) -> _DatasetMeasurement:
    """Measure one dataset's tokens/byte ratio from calibration samples."""
    try:
        scan = _scan_calibration_shards(
            dataset,
            store,
            dataset_id,
            estimation.calibration_shards_min,
            estimation.calibration_shards_max,
            seed,
        )
    except Exception as exc:  # noqa: BLE001 — priming is best-effort by contract
        return _fallback(
            estimation, None, f"calibration scan failed: {exc}", retryable=True
        )
    if scan is None:
        return _fallback(estimation, None, "dataset is empty or unsized")

    sample_ids, draw_counts = _pps_select(
        dataset_id, scan.sizes, estimation.calibration_samples, seed
    )
    if not sample_ids:
        return _fallback(estimation, scan, "dataset is empty")
    try:
        payloads = _fetch_payloads(store, dataset_id, sample_ids)
    except Exception as exc:  # noqa: BLE001 — priming is best-effort by contract
        return _fallback(
            estimation, scan, f"calibration fetch failed: {exc}", retryable=True
        )
    if not payloads:
        return _fallback(
            estimation, scan, "calibration fetch was empty", retryable=True
        )
    assert payloads.keys() == set(sample_ids), (
        f"fetch returned {len(payloads)} payloads for {len(sample_ids)} ids"
    )

    profile = tokenize_profile or TokenizeProfile()
    field_path = tuple(profile.field.split(".")) if profile.field else None
    plan = _plan_delivery(
        estimation,
        field_path,
        [payloads[sid] for sid in sample_ids[:_PLAN_SAMPLE_COUNT]],
    )
    logger.info("calibrating %s as %s", dataset.name, plan.description)
    measured: list[tuple[int, int, int]] = []
    shards_seen: set[int] = set()
    tokenizer_errors = 0
    last_tokenizer_error: Exception | None = None
    for sid, draws in zip(sample_ids, draw_counts):
        payload = payloads[sid]
        if estimation.measure is not None:
            try:
                delivered = int(estimation.measure(payload))
            except Exception as exc:  # noqa: BLE001 — user callable, stay best-effort
                return _fallback(estimation, scan, f"measure callable failed: {exc}")
        elif plan.mode == "pretokenized":
            delivered = _token_array_length(_lookup_field_path(payload, plan.path))
            if delivered is None:
                continue
        else:
            text = extract_text(payload, plan.path)
            if text is None:
                continue
            try:
                delivered = profile.delivered_tokens(
                    _count_raw_tokens(tokenizer, text, profile)
                )
            except Exception as exc:  # noqa: BLE001 — one bad doc must not kill the prime
                tokenizer_errors += 1
                last_tokenizer_error = exc
                continue
        if delivered > 0:
            # Reuse the census size; PPS only ever selects records with bytes > 0.
            measured.append((delivered, scan.sizes[(sid[1], sid[2])], draws))
            shards_seen.add(sid[1])

    if tokenizer_errors:
        logger.warning(
            "skipped %d calibration samples for %s that failed to tokenize (last: %s)",
            tokenizer_errors,
            dataset.name,
            last_tokenizer_error,
        )
    total_draws = sum(draw_counts)
    measured_draws = sum(draws for _, _, draws in measured)
    if measured_draws < _MIN_PLAN_COVERAGE * total_draws:
        reason = (
            f"calibration plan ({plan.description}) measured"
            f" only {measured_draws}/{total_draws} draws"
        )
        if tokenizer_errors:
            reason += (
                f" (tokenizer failed on {tokenizer_errors} samples,"
                f" last: {last_tokenizer_error})"
            )
        return _fallback(estimation, scan, reason)

    # Rebase the payload-byte ratio onto the raw on-disk bytes the cost charges.
    ratio = _hansen_hurwitz_ratio(measured) * scan.payload_total / scan.raw_total
    if not math.isfinite(ratio) or ratio <= 0:
        return _fallback(estimation, scan, f"measured ratio {ratio!r} is not positive")
    logger.info(
        "primed %s: %.5f tokens/byte over %d samples in %d shards",
        dataset.name,
        ratio,
        len(measured),
        len(shards_seen),
    )
    return _DatasetMeasurement(TokenRatio(ratio, "measured"))


# Processes isolate the GIL-bound census; also capped by core count at pool
# construction. Composes with MAX_SCAN_THREADS.
MAX_PRIME_PROCS = 16

# Built once per worker process in the initializer.
_prime_worker: dict[str, Any] = {}


def _init_prime_worker(
    measured_datasets: Mapping[int, Dataset],
    store_options: Any,
    estimation: TokenEstimation,
    tokenize_profile: TokenizeProfile | None,
    seed: int,
) -> None:
    """Pool initializer: attach catalogs, load the tokenizer, stash worker state."""
    set_catalog_dir(store_options)
    _prime_worker.update(
        datasets=measured_datasets,
        store=build_multi_dataset_store(measured_datasets, options=store_options),
        estimation=estimation,
        profile=tokenize_profile,
        seed=seed,
        tokenizer=(
            None
            if estimation.measure is not None
            else _instantiate_tokenizer(tokenize_profile or TokenizeProfile())
        ),
    )


def _measure_in_worker(dataset_id: int) -> _DatasetMeasurement:
    """Measure one dataset using the worker-local store + tokenizer (pool task)."""
    w = _prime_worker
    return _measure_dataset(
        w["datasets"][dataset_id],
        dataset_id,
        w["store"],
        w["estimation"],
        w["profile"],
        w["tokenizer"],
        w["seed"],
    )


def _scan_cost(dataset: Dataset) -> float:
    """LPT ordering key: mean shard bytes (the per-shard download the census pays)."""
    raw = dataset.raw_bytes()
    return float(raw.mean()) if raw.size else 0.0


def _run_census(
    measured_datasets: Mapping[int, Dataset],
    store_options: StoreOptions,
    estimation: TokenEstimation,
    tokenize_profile: TokenizeProfile | None,
    seed: int,
    mp_context: Any,
) -> dict[str, _DatasetMeasurement]:
    """Measure every dataset across the bounded process pool.

    Heaviest datasets run first so large shard downloads overlap smaller work.
    """
    order = sorted(
        measured_datasets.items(), key=lambda kv: _scan_cost(kv[1]), reverse=True
    )
    workers = min(MAX_PRIME_PROCS, len(order), os.cpu_count() or 1)
    logger.info(
        "token-aware priming: measuring %d dataset(s) across %d process(es) "
        "(%d samples, %d-%d scanned shards each) — downloads/decodes calibration "
        "shards and runs the tokenizer; expect minutes, longer on cold caches",
        len(order),
        workers,
        estimation.calibration_samples,
        estimation.calibration_shards_min,
        estimation.calibration_shards_max,
    )
    t_all = time.perf_counter()
    out: dict[str, _DatasetMeasurement] = {}
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context or mp.get_context("spawn"),
            initializer=_init_prime_worker,
            # TokenEstimation/TokenizeProfile carry cloudpickle hooks for their
            # callable/tokenizer fields, so plain initargs survive spawn.
            initargs=(
                measured_datasets,
                store_options,
                estimation,
                tokenize_profile,
                seed,
            ),
        ) as pool:
            pending = {
                pool.submit(_measure_in_worker, dataset_id): dataset.name
                for dataset_id, dataset in order
            }
            for fut in concurrent.futures.as_completed(pending):
                name = pending[fut]
                try:
                    out[name] = fut.result()
                except Exception as exc:  # noqa: BLE001 — priming is best-effort by contract
                    logger.warning(
                        "token-aware priming: measuring %s crashed; using fallback",
                        name,
                        exc_info=True,
                    )
                    out[name] = _fallback(
                        estimation, None, f"measurement crashed: {exc}", retryable=True
                    )
    except Exception as exc:  # noqa: BLE001 — priming is best-effort by contract
        # Parent-side pool failure, e.g. census inputs that defeat pickling at
        # worker spawn: fall back for every dataset not already measured.
        logger.warning(
            "token-aware priming: census pool failed; using fallback",
            exc_info=True,
        )
        for _, dataset in order:
            out.setdefault(
                dataset.name,
                _fallback(
                    estimation, None, f"census pool failed: {exc}", retryable=True
                ),
            )
    logger.info(
        "token-aware priming: measured %d dataset(s) in %.1fs total",
        len(order),
        time.perf_counter() - t_all,
    )
    return out


# Node-local single-flight for the calibration census.
#
# Ranks on the same node would otherwise repeat the same shard downloads and
# tokenization. The fingerprint covers every input, so peers can safely block on
# a file lock and reuse the byte-identical ratios written by the first rank.
_PRIME_CACHE_VERSION = 1
_PRIME_CACHE_SUBDIR = "token_ratios"


def _dataset_content_key(dataset: Dataset) -> str:
    """Stable content identity for a dataset's census inputs."""
    handle = dataset.catalog_handle
    if handle is not None and handle.fingerprint is not None:
        return handle.fingerprint
    return hashlib.sha256(
        dataset.ids().tobytes()
        + dataset.counts().tobytes()
        + dataset.raw_bytes().tobytes()
    ).hexdigest()


def _pickle_fingerprint(obj: Any) -> str:
    """Content hash via cloudpickle; sentinel when the object cannot serialize."""
    try:
        return hashlib.sha256(cloudpickle.dumps(obj)).hexdigest()
    except Exception:  # noqa: BLE001 — priming is best-effort by contract
        return "__unpicklable__"


def _prime_cache_key(
    names: list[str],
    by_name: Mapping[str, Dataset],
    estimation: TokenEstimation,
    tokenize_profile: TokenizeProfile | None,
    seed: int,
) -> str:
    """Fingerprint every input the measured ratios depend on."""
    profile = tokenize_profile or TokenizeProfile()
    parts = [
        str(_PRIME_CACHE_VERSION),
        str(seed),
        _pickle_fingerprint(estimation),
        repr(
            (
                profile.tokenizer_id,
                # Live tokenizer state shapes counts; id-less instances must
                # not alias each other.
                None
                if profile.tokenizer is None
                else _pickle_fingerprint(profile.tokenizer),
                profile.field,
                profile.max_length,
                profile.truncation,
                profile.split_long_samples,
                profile.special_tokens,
            )
        ),
        *(f"{name}={_dataset_content_key(by_name[name])}" for name in sorted(names)),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _read_prime_cache(path: Path, key: str) -> dict[str, _DatasetMeasurement] | None:
    """Load a cached census result, or ``None`` on miss / mismatch / corruption."""
    try:
        doc = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("v") != _PRIME_CACHE_VERSION:
        return None
    if doc.get("key") != key:
        return None
    try:
        return {
            name: _DatasetMeasurement(TokenRatio.from_state(entry[:2]), entry[2])
            for name, entry in doc["ratios"].items()
        }
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _write_prime_cache(
    path: Path, key: str, measured: Mapping[str, _DatasetMeasurement]
) -> None:
    doc = {
        "v": _PRIME_CACHE_VERSION,
        "key": key,
        "ratios": {
            name: [*m.ratio.to_state(), m.reason] for name, m in measured.items()
        },
    }
    atomic_write_bytes(path, json.dumps(doc).encode("utf-8"), fsync=True)


def prime_token_ratios(
    *,
    datasets: list[Dataset],
    dataset_ids: Mapping[str, int],
    estimation: TokenEstimation,
    tokenize_profile: TokenizeProfile | None,
    io_options: Any = None,
    seed: int = 0,
    mp_context: Any = None,
) -> dict[str, TokenRatio]:
    """Derive per-dataset tokens/byte ratios.

    Pinned ratios skip measurement. Unmeasurable datasets get the fallback ratio
    with a warning because mixed measured/fallback results can skew weights.
    """
    by_name = {ds.name: ds for ds in datasets if ds.name in dataset_ids}

    ratios: dict[str, TokenRatio] = {}
    pinned: Mapping[str, float] = {}
    if isinstance(estimation.primer, Mapping):
        unknown = set(estimation.primer) - set(dataset_ids)
        if unknown:
            raise ValueError(
                f"token_estimation.primer pins unknown datasets: {sorted(unknown)}. "
                f"Known datasets: {sorted(dataset_ids)}"
            )
        pinned = estimation.primer
    elif isinstance(estimation.primer, (int, float)):
        ratio = TokenRatio(float(estimation.primer), "pinned")
        return dict.fromkeys(dataset_ids, ratio)

    for name, value in pinned.items():
        ratios[name] = TokenRatio(float(value), "pinned")

    to_measure = [name for name in dataset_ids if name not in ratios]
    fallback_reasons: dict[str, str] = {}
    if to_measure:
        store_options = StoreOptions.from_any(io_options)
        set_catalog_dir(store_options)

        names: list[str] = []
        for name in to_measure:
            if name in by_name:
                names.append(name)
            else:
                ratios[name] = TokenRatio(
                    estimation.fallback_tokens_per_byte, "fallback"
                )
                fallback_reasons[name] = "dataset descriptor not found"

        if names:
            measured_datasets = {dataset_ids[name]: by_name[name] for name in names}
            # Build catalogs once so workers mmap-attach instead of repeating discovery.
            build_multi_dataset_store(measured_datasets, options=store_options)

            key = _prime_cache_key(names, by_name, estimation, tokenize_profile, seed)
            cache_path = (
                resolve_catalog_dir(store_options) / _PRIME_CACHE_SUBDIR / f"{key}.json"
            )
            measured = _read_prime_cache(cache_path, key)
            if measured is not None:
                logger.info(
                    "token-aware priming: reusing node-local ratios for %d dataset(s)",
                    len(names),
                )
            else:
                lock_path = cache_path.with_suffix(".lock")
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                # First rank measures; peers block, then re-check the cache.
                # Tradeoff: transient failures are not cached, so a persistent
                # failure (bad credentials) re-runs the census serially on every
                # local rank. TODO(MaxiBoether): short-TTL negative cache.
                with FileLock(str(lock_path)):
                    measured = _read_prime_cache(cache_path, key)
                    if measured is None:
                        measured = _run_census(
                            measured_datasets,
                            store_options,
                            estimation,
                            tokenize_profile,
                            seed,
                            mp_context,
                        )
                        # Do not cache transient fallbacks; retrying later is cheaper.
                        transient = sorted(
                            n for n, m in measured.items() if m.retryable
                        )
                        if transient:
                            logger.warning(
                                "token-aware priming: transient failure measuring "
                                "%s; not caching so a later prime retries",
                                transient,
                            )
                        else:
                            _write_prime_cache(cache_path, key, measured)
                    else:
                        logger.info(
                            "token-aware priming: reusing ratios primed by a peer "
                            "on this node"
                        )
            for name, outcome in measured.items():
                ratios[name] = outcome.ratio
                if outcome.reason is not None:
                    fallback_reasons[name] = outcome.reason

    if fallback_reasons:
        # Scan-rebased fallbacks differ from the raw constant; quote each.
        details = "; ".join(
            f"{name}: {ratios[name].tokens_per_byte:.6f} tokens/byte ({reason})"
            for name, reason in sorted(fallback_reasons.items())
        )
        warnings.warn(
            f"[zephon] token-aware mixture priming fell back for "
            f"{len(fallback_reasons)} dataset(s): {details}. Mixing measured "
            f"and fallback ratios mis-weights the fallback datasets by the "
            f"tokens/byte spread — pin ratios via "
            f"TokenEstimation(primer={{...}}) or provide a measure= callable "
            f"for these datasets.",
            RuntimeWarning,
            stacklevel=2,
        )

    for name, ratio in ratios.items():
        logger.info(
            "token-aware priming: dataset %s -> %.6f tokens/byte (%s)",
            name,
            ratio.tokens_per_byte,
            ratio.source,
        )
    return ratios


__all__ = [
    "DEFAULT_FALLBACK_TOKENS_PER_BYTE",
    "PerShardTokenCost",
    "TokenEstimation",
    "TokenRatio",
    "TokenizeProfile",
    "prime_token_ratios",
]
