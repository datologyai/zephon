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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, get_args

import cloudpickle
import numpy as np
from filelock import FileLock

from zephon._internal.io.catalog import (
    resolve_catalog_dir as _resolve_catalog_dir,
)
from zephon._internal.io.catalog import (
    set_catalog_dir as _set_catalog_dir,
)
from zephon._internal.io.stores import (
    build_multi_dataset_store as _build_multi_dataset_store,
)
from zephon._internal.observability.size_estimator import (
    content_bytes as _content_bytes,
)
from zephon._internal.token_counting import (
    CountPlan as _CountPlan,
)
from zephon._internal.token_counting import (
    DeliveredTokenCounter as _DeliveredTokenCounter,
)
from zephon._internal.token_counting import (
    FatalCountError as _FatalCountError,
)
from zephon._internal.token_counting import (
    TextTokenCountingSpec as _TextTokenCountingSpec,
)
from zephon._internal.token_counting import (
    TokenCountingSpec as _TokenCountingSpec,
)
from zephon._internal.utils.atomic import atomic_write_bytes as _atomic_write_bytes
from zephon.io.dataset import Dataset
from zephon.io.options import StoreOptions
from zephon.ops.base import OpContext, StageInfo
from zephon.types import SampleId, SampleMeta, SampleRecord

_logger = logging.getLogger(__name__)

#: Standard ~4 bytes/token BPE heuristic for English prose.
DEFAULT_FALLBACK_TOKENS_PER_BYTE: float = 0.25

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

    ``primer`` — how per-dataset tokens/byte ratios are obtained:

    * ``measure`` (default): calibration fetch + tokenize at prime
      time, deterministic in ``(seed, datasets, tokenize config)``.
    * a ``{dataset_name: tokens_per_byte}`` mapping: pins the listed
      datasets and *measures the rest* (partial pins merge over
      measured values).
    * a single ``float``: one global tokens/byte for every dataset,
      no measurement (and no tokenizer needed).

    ``measure`` — escape hatch replacing text extraction + tokenization
    entirely: a callable mapping a fetched payload to its delivered token
    count (weird schemas, VLM cost units). Only consulted when a dataset is
    actually measured.

    ``calibration_samples`` — records measured per dataset
    (size-proportional draws for catalog-backed datasets, scattered offsets
    otherwise).

    ``calibration_shards_min`` — shards scanned per dataset when its shards
    are homogeneous. The count is chosen per dataset from the catalog's
    per-shard bytes/row spread (free metadata): census error scales as
    CV/sqrt(shards), so heterogeneous datasets automatically scan up to
    ``calibration_shards_max`` while uniform ones stay at the minimum. Each
    scanned shard costs a download + decode at prime time.

    ``calibration_shards_max`` — upper bound for the adaptive shard count
    (also used when there is no catalog to read the spread from).

    ``fallback_tokens_per_byte`` — ratio used when a dataset cannot be
    measured (no text found, measurement failed, or ``primer`` is a float).
    See :data:`DEFAULT_FALLBACK_TOKENS_PER_BYTE`.
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


# ---------------------------------------------------------------------------
# Byte sources
# ---------------------------------------------------------------------------


class _PerShardByteSize:
    """Per-shard average bytes/sample for one dataset.

    ``mean_bytes`` is the dataset-wide fallback for shards absent from
    ``shard_avg_bytes``.
    """

    def __init__(self, shard_avg_bytes: dict[int, float], mean_bytes: float) -> None:
        self.shard_avg_bytes = shard_avg_bytes
        self.mean_bytes = max(1.0, float(mean_bytes))


def _build_byte_source(dataset: Dataset) -> _PerShardByteSize | None:
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
    return _PerShardByteSize(shard_avg, total_bytes / total_rows)


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
            source = _build_byte_source(dataset)
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
# Pre-tokenize replay for calibration
# ---------------------------------------------------------------------------


class _PreTokenizeReplay:
    """Replay safe per-record ops before tokenization during calibration.

    Calibration fetches raw rows, so it replays these ops to count what the
    tokenizer sees; dropped rows contribute zero tokens. The pipeline allowlist
    excludes ops that need accumulator state or ``OpContext`` services.
    """

    def __init__(self, ops: Sequence[Any]) -> None:
        self.ops: tuple[Any, ...] = tuple(ops)
        self._setup_done = False

    def apply(self, payload: Any) -> list[Any]:
        """Map one raw payload to the payloads the tokenize op would see."""
        if not self._setup_done:
            # Ops arrive as pre-setup cloudpickle copies, like runner workers.
            ctx = OpContext(
                {}, StageInfo(stage_index=0, stage_name="calibration", op_index=0)
            )
            for op in self.ops:
                op.setup(ctx)
            self._setup_done = True
        records = [
            SampleRecord(
                meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
                payload=payload,
            )
        ]
        for op in self.ops:
            # Strip drop tombstones between ops: not every op tolerates them,
            # and calibration has no chunk accounting for them to serve.
            records = [
                rec for rec in op.process_many(records) if not rec.meta.tombstone
            ]
            if not records:
                break
        return [rec.payload for rec in records]


@dataclass(frozen=True)
class _UnreplayableOp:
    """Marks a fetch -> tokenize span that cannot be replayed for calibration.

    Measuring through it would count raw rows the tokenize op never sees, so
    priming refuses; pinned ratios and ``measure=`` are unaffected.
    """

    op_name: str


def _count_replayed(plan: _CountPlan, payloads: Sequence[Any]) -> int | None:
    """Count the payloads produced by replaying one raw pointer."""
    total = 0
    for payload in payloads:
        got = plan.count(payload)
        if got is None:
            return None
        total += got
    return total


def _count_delivered(
    plan: _CountPlan, replay: _PreTokenizeReplay | None, payload: Any
) -> int | None:
    """Delivered tokens for one raw pointer, or ``None`` when unmeasurable."""
    if replay is None:
        return plan.count(payload)
    return _count_replayed(plan, replay.apply(payload))


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
            size = _content_bytes(payload)
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
    _logger.info(
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


# Plan from a payload prefix so one anomalous record cannot steer the dataset's
# measurement mode; gate on measured draw mass so a plan/data mismatch is loud.
_PLAN_SAMPLE_COUNT = 8
_MIN_PLAN_COVERAGE = 0.5


@dataclass(frozen=True)
class _MeasurePlan(_CountPlan):
    measure: Callable[[Any], int]
    # A user measure defines the unit, so an exception invalidates calibration.
    abort_on_error: ClassVar[bool] = True

    @property
    def description(self) -> str:
        return "measure callable"

    def count(self, payload: Any) -> int | None:
        delivered = int(self.measure(payload))
        # Non-positive results are unmeasurable, not zero-yield samples.
        return delivered if delivered > 0 else None


def _measure_dataset(
    dataset: Dataset,
    dataset_id: int,
    store: Any,
    estimation: TokenEstimation,
    counter: _DeliveredTokenCounter | None,
    seed: int,
    pre_tokenize_replay: _PreTokenizeReplay | None = None,
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

    replayed_prefix: dict[SampleId, list[Any]] = {}
    replay_errors: dict[SampleId, Exception] = {}
    if estimation.measure is not None:
        plan: _CountPlan = _MeasurePlan(estimation.measure)
    else:
        assert counter is not None, "no measure callable, so a counter is required"
        prefix_ids = sample_ids[:_PLAN_SAMPLE_COUNT]
        prefix: list[Any] = [payloads[sid] for sid in prefix_ids]
        if pre_tokenize_replay is not None:
            prefix = []
            for sid in prefix_ids:
                try:
                    replayed = pre_tokenize_replay.apply(payloads[sid])
                except Exception as exc:  # noqa: BLE001 — handled during measurement
                    replay_errors[sid] = exc
                else:
                    replayed_prefix[sid] = replayed
                    prefix.extend(replayed)
        plan = counter.plan(prefix)
    _logger.info("calibrating %s as %s", dataset.name, plan.description)
    measured: list[tuple[int, int, int]] = []
    shards_seen: set[int] = set()
    count_errors = 0
    last_count_error: Exception | None = None
    for sid, draws in zip(sample_ids, draw_counts):
        try:
            if sid in replay_errors:
                raise replay_errors.pop(sid)
            if sid in replayed_prefix:
                delivered = _count_replayed(plan, replayed_prefix.pop(sid))
            else:
                delivered = _count_delivered(plan, pre_tokenize_replay, payloads[sid])
        except _FatalCountError:
            # Structural: execution rejects this row too. Surface it instead of
            # sampling around it, even when coverage would otherwise pass.
            raise
        except Exception as exc:  # noqa: BLE001 — the plan chooses abort or skip
            if plan.abort_on_error:
                return _fallback(estimation, scan, f"{plan.description} failed: {exc}")
            count_errors += 1
            last_count_error = exc
            continue
        if delivered is not None:
            # Zero counts are genuine measurements (e.g. chat drops): the
            # bytes are scheduled either way, so zeros belong in the ratio.
            # Reuse the census size; PPS only ever selects records with bytes > 0.
            measured.append((delivered, scan.sizes[(sid[1], sid[2])], draws))
            shards_seen.add(sid[1])

    if count_errors:
        _logger.warning(
            "skipped %d calibration samples for %s that failed to count (last: %s)",
            count_errors,
            dataset.name,
            last_count_error,
        )
    total_draws = sum(draw_counts)
    measured_draws = sum(draws for _, _, draws in measured)
    if measured_draws < _MIN_PLAN_COVERAGE * total_draws:
        reason = (
            f"calibration plan ({plan.description}) measured"
            f" only {measured_draws}/{total_draws} draws"
        )
        if count_errors:
            reason += (
                f" (counting failed on {count_errors} samples,"
                f" last: {last_count_error})"
            )
        return _fallback(estimation, scan, reason)

    # Rebase the payload-byte ratio onto the raw on-disk bytes the cost charges.
    ratio = _hansen_hurwitz_ratio(measured) * scan.payload_total / scan.raw_total
    if not math.isfinite(ratio) or ratio <= 0:
        return _fallback(estimation, scan, f"measured ratio {ratio!r} is not positive")
    _logger.info(
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
    counting_spec: _TokenCountingSpec | None,
    seed: int,
    pre_tokenize_replay: _PreTokenizeReplay | None = None,
) -> None:
    _set_catalog_dir(store_options)
    _prime_worker.update(
        datasets=measured_datasets,
        store=_build_multi_dataset_store(measured_datasets, options=store_options),
        estimation=estimation,
        seed=seed,
        pre_tokenize_replay=pre_tokenize_replay,
        counter=(
            None
            if estimation.measure is not None
            else (counting_spec or _TextTokenCountingSpec()).build_counter()
        ),
    )


def _ensure_imports(dataset: Dataset) -> None:
    """Finish format-specific cold imports before calibration fans out."""
    if dataset.backend["kind"] == "litdata":
        from zephon._internal.io.formats import litdata_support

        litdata_support.ensure_litdata_deps()


def _measure_in_worker(dataset_id: int) -> _DatasetMeasurement:
    w = _prime_worker
    dataset = w["datasets"][dataset_id]
    _ensure_imports(dataset)
    return _measure_dataset(
        dataset,
        dataset_id,
        w["store"],
        w["estimation"],
        w["counter"],
        w["seed"],
        pre_tokenize_replay=w["pre_tokenize_replay"],
    )


def _scan_cost(dataset: Dataset) -> float:
    """LPT ordering key: mean shard bytes (the per-shard download the census pays)."""
    raw = dataset.raw_bytes()
    return float(raw.mean()) if raw.size else 0.0


def _run_census(
    measured_datasets: Mapping[int, Dataset],
    store_options: StoreOptions,
    estimation: TokenEstimation,
    counting_spec: _TokenCountingSpec | None,
    seed: int,
    mp_context: Any,
    pre_tokenize_replay: _PreTokenizeReplay | None = None,
) -> dict[str, _DatasetMeasurement]:
    """Measure every dataset across the bounded process pool.

    Heaviest datasets run first so large shard downloads overlap smaller work.
    """
    order = sorted(
        measured_datasets.items(), key=lambda kv: _scan_cost(kv[1]), reverse=True
    )
    workers = min(MAX_PRIME_PROCS, len(order), os.cpu_count() or 1)
    _logger.info(
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
            # Cloudpickle hooks let callable/tokenizer fields survive spawn.
            initargs=(
                measured_datasets,
                store_options,
                estimation,
                counting_spec,
                seed,
                pre_tokenize_replay,
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
                except _FatalCountError:
                    raise
                except Exception as exc:  # noqa: BLE001 — priming is best-effort by contract
                    _logger.warning(
                        "token-aware priming: measuring %s crashed; using fallback",
                        name,
                        exc_info=True,
                    )
                    out[name] = _fallback(
                        estimation, None, f"measurement crashed: {exc}", retryable=True
                    )
    except _FatalCountError:
        raise
    except Exception as exc:  # noqa: BLE001 — priming is best-effort by contract
        # Parent-side pool failure, e.g. census inputs that defeat pickling at
        # worker spawn: fall back for every dataset not already measured.
        _logger.warning(
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
    _logger.info(
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
_PRIME_CACHE_VERSION = 2
_PRIME_CACHE_SUBDIR = "token_ratios"


def _dataset_content_key(dataset: Dataset) -> str:
    """Stable content identity for a dataset's census inputs."""
    handle = dataset._catalog_handle
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
    counting_spec: _TokenCountingSpec | None,
    seed: int,
    pre_tokenize_replay: _PreTokenizeReplay | None = None,
) -> str:
    """Fingerprint every input the measured ratios depend on."""
    # Include spec type and state so distinct live tokenizers cannot alias.
    # Serialization failures also prevent spawn, and fallbacks are not cached.
    parts = [
        str(_PRIME_CACHE_VERSION),
        str(seed),
        _pickle_fingerprint(estimation),
        _pickle_fingerprint(counting_spec or _TextTokenCountingSpec()),
        _pickle_fingerprint(pre_tokenize_replay),
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
    _atomic_write_bytes(path, json.dumps(doc).encode("utf-8"), fsync=True)


def prime_token_ratios(
    *,
    datasets: list[Dataset],
    dataset_ids: Mapping[str, int],
    estimation: TokenEstimation,
    counting_spec: _TokenCountingSpec | None,
    io_options: Any = None,
    seed: int = 0,
    mp_context: Any = None,
    pre_tokenize_replay: _PreTokenizeReplay | _UnreplayableOp | None = None,
) -> dict[str, TokenRatio]:
    """Derive per-dataset tokens/byte ratios.

    Pinned ratios skip measurement. Unmeasurable datasets get the fallback ratio
    with a warning because mixed measured/fallback results can skew weights.
    ``pre_tokenize_replay`` applies the pipeline's pre-tokenize map ops to each
    calibration payload; an unreplayable-op marker rejects measurement outright
    (pins and ``measure=`` still work).
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
        _set_catalog_dir(store_options)

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
            if estimation.measure is not None:
                # measure= owns raw payload -> count; skip the replay so the
                # callable is not fed already-normalized payloads.
                pre_tokenize_replay = None
            elif isinstance(pre_tokenize_replay, _UnreplayableOp):
                raise ValueError(
                    f"token-aware mixture priming cannot calibrate through "
                    f"this pipeline: op {pre_tokenize_replay.op_name!r} "
                    f"between fetch and the tokenize op cannot be replayed "
                    f"outside the engine, so calibration would count raw rows "
                    f"the tokenize op never sees. Provide "
                    f"TokenEstimation(measure=...) or pin ratios via "
                    f"TokenEstimation(primer={{...}})."
                )
            measured_datasets = {dataset_ids[name]: by_name[name] for name in names}
            # Build catalogs once so workers mmap-attach instead of repeating discovery.
            _build_multi_dataset_store(measured_datasets, options=store_options)

            key = _prime_cache_key(
                names, by_name, estimation, counting_spec, seed, pre_tokenize_replay
            )
            cache_path = (
                _resolve_catalog_dir(store_options)
                / _PRIME_CACHE_SUBDIR
                / f"{key}.json"
            )
            measured = _read_prime_cache(cache_path, key)
            if measured is not None:
                _logger.info(
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
                            counting_spec,
                            seed,
                            mp_context,
                            pre_tokenize_replay,
                        )
                        # Do not cache transient fallbacks; retrying later is cheaper.
                        transient = sorted(
                            n for n, m in measured.items() if m.retryable
                        )
                        if transient:
                            _logger.warning(
                                "token-aware priming: transient failure measuring "
                                "%s; not caching so a later prime retries",
                                transient,
                            )
                        else:
                            _write_prime_cache(cache_path, key, measured)
                    else:
                        _logger.info(
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

    # TODO: Broadcast or publish one authoritative ratio map (for example via
    # the cloud/checkpoint directory) so every rank and node uses identical costs.
    for name, ratio in ratios.items():
        _logger.info(
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
    "prime_token_ratios",
]
