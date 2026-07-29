# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Streaming mixture enforcement operator using Smooth Weighted Round Robin."""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from zephon._internal.op_base import DefaultSetup
from zephon._internal.utils.length_extraction import extract_length
from zephon._internal.utils.swrr import SmoothWeightedRoundRobin
from zephon.ops.accumulators import Accumulator, ReadyBatch
from zephon.ops.base import OpContext
from zephon.ops.children import tombstones_for_record
from zephon.ops.traits import OpTraits
from zephon.types import ChunkId, ComponentId, LaneId, SampleRecord

logger = logging.getLogger(__name__)


@dataclass
class EnsureMixtureConfig:
    """Configuration for EnsureMixture operator and its accumulator.

    Contains only picklable fields. Context callbacks are passed directly
    to the accumulator constructor to support process runner pickling.
    """

    # Set during __init__
    weight_by: Callable[[SampleRecord], float] | Literal["samples", "auto"] | str
    warn_tolerance: float | None
    warn_warmup: float  # Min emitted weight before ratio warnings are enabled
    mixture_override: dict[str, float] | None  # component_name -> target weight
    # Samples to buffer before force-emitting (enables reordering).  When
    # ``None`` the buffer is unbounded — the operator only emits while SWRR's
    # ideal component is available, never compromising the target mixture.
    max_buffer_size: int | None
    drain_target_ratio: (
        float  # When forced to emit, drain to this fraction of max_buffer_size
    )
    obsolete_drain_rate: float  # Fraction of emissions reserved for obsolete components

    # Explicit mixture override converted to int form during setup
    mixture_override_by_id: dict[int, float] | None = None

    def get_weight(self, record: SampleRecord) -> float:
        """Get the weight for a sample.

        - Callable: custom weight function
        - "samples": weight = 1.0
        - "auto": auto-detect token field (input_ids, tokens, etc.)
        - Other string: explicit field name

        Raises ValueError if token field cannot be found/detected.
        """
        if callable(self.weight_by):
            return float(self.weight_by(record))

        if self.weight_by == "samples":
            return 1.0

        # Token-based: "auto" = auto-detect, other string = explicit field
        field = None if self.weight_by == "auto" else self.weight_by
        return float(extract_length(record, field))


@dataclass
class _MultiComponentSample:
    """A multi-component (packed) sample with precomputed contributions."""

    record: SampleRecord
    contributions: dict[int, float]  # component_id -> weight


@dataclass
class _LaneState:
    """Per-lane state for SWRR reordering with buffering.

    Single-component samples use per-component buffers (original design).
    Multi-component (packed) samples go into a separate buffer and use
    benefit scoring for selection.
    """

    # Per-component buffers for single-component samples: component_id -> deque[(record, weight)]
    # This preserves the original SWRR selection logic for the common case.
    buffers: dict[int, deque[tuple[SampleRecord, float]]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    # Separate buffer for multi-component (packed) samples.
    # These require benefit scoring since they contribute to multiple components.
    multi_component_buffer: deque[_MultiComponentSample] = field(default_factory=deque)
    # SWRR selector for this lane (created lazily with first chunk's mixture)
    swrr: SmoothWeightedRoundRobin[int] | None = None
    # Current chunk being processed (for mixture lookup)
    current_chunk_id: int | None = None
    # Track total samples currently buffered (for max_buffer_size check)
    total_buffered: int = 0
    # Accumulated weight for ratio deviation checking
    emitted_by_component: dict[int, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    total_emitted: float = 0.0
    # Counter for obsolete interleaving: counts emissions since last obsolete drain
    emissions_since_obsolete_drain: int = 0


class EnsureMixtureAccumulator(Accumulator[SampleRecord]):
    """Adaptive accumulator that reorders samples to maintain target mixture ratios.

    Uses adaptive buffering: emit immediately when SWRR's ideal component is
    available, buffer when the desired component isn't present yet. Falls back
    to emitting best-available when max_buffer_size is reached.

    The SWRR algorithm tracks deficit (target - actual) and always selects the
    component with highest deficit, ensuring smooth convergence to target ratios.

    Emits batched per-lane to minimize scheduling overhead.
    """

    def __init__(
        self,
        config: EnsureMixtureConfig,
        get_chunk_mixture: Callable[[LaneId, ChunkId], dict[int, float] | None]
        | None = None,
        get_component_name: Callable[[ComponentId], str | None] | None = None,
    ) -> None:
        self._config = config
        self._get_chunk_mixture = get_chunk_mixture
        self._get_component_name = get_component_name
        self._lanes: dict[int, _LaneState] = defaultdict(_LaneState)
        self._total_discarded: int = 0
        #: Strict mode discards on roughly every flush in a skewed stream, so
        #: only the first one warns; the rest drop to DEBUG to avoid log spam.
        self._discard_warned: bool = False

    #: Cap on tombstones per ReadyBatch when discarding. The strict-mode buffer
    #: is unbounded, so a discard can dump a whole flush window at once; splitting
    #: it keeps any one discard from becoming a single oversized worker dispatch.
    _TOMBSTONE_BATCH = 1024

    def _discard_lane_buffers(
        self, state: _LaneState
    ) -> tuple[list[ReadyBatch[SampleRecord]], int]:
        """Discard a lane's buffers, emitting tombstones that close offsets.

        Discarded records carried the closing contributors for their base
        offsets; dropping them silently would leave those offsets open
        forever — the epoch never completes and per-epoch eviction halts at
        the first discard. A tombstone per closing contributor keeps eviction
        moving, the same bookkeeping pack_sequences uses for its dropped
        records.

        Records are freed as they are walked rather than held alongside the
        full tombstone list, which on the unbounded buffer would double peak
        memory. Returns the batches and the discarded-record count.
        """
        dropped = state.total_buffered
        ready: list[ReadyBatch[SampleRecord]] = []
        batch: list[SampleRecord] = []

        def take(record: SampleRecord) -> None:
            batch.extend(tombstones_for_record(record))
            while len(batch) >= self._TOMBSTONE_BATCH:
                ready.append((batch[: self._TOMBSTONE_BATCH], 0))
                del batch[: self._TOMBSTONE_BATCH]

        for buf in state.buffers.values():
            while buf:
                record, _ = buf.popleft()
                take(record)
        while state.multi_component_buffer:
            take(state.multi_component_buffer.popleft().record)

        if batch:
            ready.append((batch, 0))

        state.buffers.clear()
        state.multi_component_buffer.clear()
        state.total_buffered = 0
        return ready, dropped

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        """Return True if buffered records remain (in ``lane_id`` if given)."""
        if lane_id is None:
            states: Iterable[_LaneState] = self._lanes.values()
        else:
            state = self._lanes.get(lane_id)
            states = (state,) if state is not None else ()
        return any(
            any(buf for buf in s.buffers.values()) or s.multi_component_buffer
            for s in states
        )

    def _can_emit_optimal(self, state: _LaneState) -> bool:
        """Check if SWRR's ideal choice is available in the buffer.

        Returns True if we can emit without compromising mixture quality.
        This enables adaptive buffering: emit early when possible, buffer when needed.

        For multi-component samples: we also check if any packed sample contributes
        to the ideal component.
        """
        if state.swrr is None:
            # No target mixture configured - emit anything available
            return True

        available = {cid for cid, buf in state.buffers.items() if buf}
        has_multi = bool(state.multi_component_buffer)

        if not available and not has_multi:
            return False

        # What does SWRR ideally want (ignoring availability)?
        ideal = state.swrr.peek()

        # Check single-component buffers first (original logic)
        if ideal in available:
            return True

        # Check if any multi-component sample contributes to the ideal component
        for multi in state.multi_component_buffer:
            if ideal in multi.contributions:
                return True

        return False

    @property
    def reads_payload(self) -> bool:
        return True

    def push_many(
        self, elems: Sequence[SampleRecord]
    ) -> list[ReadyBatch[SampleRecord]]:
        """Accumulate records and emit adaptively via SWRR.

        Uses adaptive buffering: emit immediately when SWRR's ideal component is
        available, buffer when the desired component isn't present yet. Falls back
        to emitting best-available when max_buffer_size is reached.

        Emits batched per-lane to minimize scheduling overhead.
        """
        ready: list[ReadyBatch[SampleRecord]] = []
        affected_lanes: set[int] = set()

        # Phase 1: Buffer all incoming records
        for elem in elems:
            lane_id = elem.meta.lane_id
            chunk_id = elem.meta.chunk_id

            state = self._lanes[lane_id]

            # Update SWRR if chunk changed (mixture might have changed)
            if state.current_chunk_id != chunk_id:
                self._update_lane_swrr(lane_id, chunk_id, state)
                state.current_chunk_id = chunk_id

            # Compute per-component contributions for this sample.
            contributions = self._compute_contributions(elem)

            # Route to appropriate buffer based on number of components.
            if len(contributions) == 1:
                # Single-component sample: use per-component buffer (original logic)
                component_id = next(iter(contributions.keys()))
                weight = contributions[component_id]
                state.buffers[component_id].append((elem, weight))
            else:
                # Multi-component sample (packed): use separate buffer
                state.multi_component_buffer.append(
                    _MultiComponentSample(record=elem, contributions=contributions)
                )

            state.total_buffered += 1
            affected_lanes.add(lane_id)

        # Phase 2: Emit batched per-lane (adaptive)
        for lane_id in affected_lanes:
            state = self._lanes[lane_id]
            emitted_batch: list[SampleRecord] = []

            # Compute drain target: when forced to emit, drain to this level.
            max_buf = self._config.max_buffer_size
            if max_buf is None:
                # Unbounded: never force-drain (drain_target unused here).
                started_forced_drain = False
                drain_target = 0
            else:
                drain_target = int(self._config.drain_target_ratio * max_buf)
                started_forced_drain = state.total_buffered >= max_buf

            # Greedy emit: while SWRR is happy OR buffer needs draining
            while self._has_buffered_samples(state):
                is_optimal = self._can_emit_optimal(state)

                # If we started forced draining, continue until we reach drain_target
                is_forced_drain = (
                    started_forced_drain and state.total_buffered > drain_target
                )

                if not is_optimal and not is_forced_drain:
                    break

                # Emit one sample. Obsolete components are drained automatically
                # at obsolete_drain_rate, interleaved with normal emissions.
                record = self._emit_one(lane_id, state)
                if record is not None:
                    state.total_buffered -= 1
                    emitted_batch.append(record)
                else:
                    break

            if emitted_batch:
                ready.append((emitted_batch, 0))

        return ready

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[SampleRecord]]:
        """Emit remaining buffered records for the given lane(s).

        Bounded mode drains the buffer in SWRR order (lossless), accelerating
        obsolete-component draining so nothing is stranded. Unbounded mode
        (``max_buffer_size=None``) instead discards the buffer on every flush —
        mid-stream sentinel (``reset=True``) and terminal close
        (``reset=False``) alike — since it only holds the surplus SWRR withheld
        to keep the mixture exact; draining it would skew the output.

        ``lane_id=None`` flushes every lane; otherwise only that lane is
        flushed and (when ``reset=True``) reset, leaving the others untouched.
        """
        if lane_id is None:
            lane_ids = list(self._lanes)
        else:
            lane_ids = [lane_id] if lane_id in self._lanes else []

        ready: list[ReadyBatch[SampleRecord]] = []
        discarded = 0
        discarded_lanes = 0

        for lid in lane_ids:
            state = self._lanes[lid]

            if self._config.max_buffer_size is None:
                batches, dropped = self._discard_lane_buffers(state)
                ready.extend(batches)
                if dropped:
                    discarded += dropped
                    discarded_lanes += 1
            else:
                emitted_batch: list[SampleRecord] = []

                while self._has_buffered_samples(state):
                    # During flush, accelerate obsolete draining by resetting the counter
                    # This ensures obsolete samples are drained promptly
                    obsolete = self._get_obsolete_components(state)
                    if obsolete:
                        # Force obsolete drain by setting counter to threshold
                        if self._config.obsolete_drain_rate > 0:
                            drain_interval = int(1.0 / self._config.obsolete_drain_rate)
                            state.emissions_since_obsolete_drain = drain_interval

                    record = self._emit_one(lid, state)
                    if record is not None:
                        state.total_buffered -= 1
                        emitted_batch.append(record)
                    else:
                        break

                if emitted_batch:
                    ready.append((emitted_batch, 0))

            # Mid-stream flush: reset emission history so the next epoch starts
            # with clean SWRR state, identical to a freshly constructed
            # accumulator.  Done even when the buffer was already empty — a stale
            # SWRR deficit would otherwise perturb the next epoch on replay.  We
            # null out swrr (not just reset()) so _update_lane_swrr builds a fresh
            # instance with sorted _order — otherwise update_target() on the old
            # instance keeps stale _order/_index entries, which can diverge
            # tie-breaking if mixture components change across epochs.
            if reset:
                state.emitted_by_component.clear()
                state.total_emitted = 0.0
                state.emissions_since_obsolete_drain = 0
                state.current_chunk_id = None
                state.swrr = None

        if discarded:
            self._total_discarded += discarded
            # Skewed strict-mode runs discard on roughly every flush, so warn
            # once and log the rest at DEBUG. The cumulative count is per
            # accumulator (one per rank), not run-wide across ranks.
            if not self._discard_warned:
                self._discard_warned = True
                logger.warning(
                    "EnsureMixture strict mode discarded %d record(s) across %d "
                    "lane(s) to keep the mixture exact (%d dropped cumulatively "
                    "on this rank). Raise flush_every_k_chunks to discard less "
                    "often; further discards on this rank are logged at DEBUG.",
                    discarded,
                    discarded_lanes,
                    self._total_discarded,
                )
            else:
                logger.debug(
                    "EnsureMixture strict mode discarded %d record(s) across %d "
                    "lane(s) to keep the mixture exact (%d dropped cumulatively "
                    "on this rank).",
                    discarded,
                    discarded_lanes,
                    self._total_discarded,
                )

        return ready

    def _has_buffered_samples(self, state: _LaneState) -> bool:
        """Check if lane has any buffered samples."""
        return any(buf for buf in state.buffers.values()) or bool(
            state.multi_component_buffer
        )

    def _compute_contributions(self, record: SampleRecord) -> dict[int, float]:
        """Compute per-component contribution weights for a sample.

        For weight="samples": uses component_sample_counts directly.
        For token-based modes: uses component_token_counts if available,
        otherwise distributes the total token count proportionally by sample counts.

        Returns a dict mapping component_id -> weight for this sample.
        """
        meta = record.meta

        if self._config.weight_by == "samples":
            # Sample-weighted: each original sample counts as 1
            return {
                cid: float(count) for cid, count in meta.component_sample_counts.items()
            }

        # Token-weighted mode
        if meta.component_token_counts is not None:
            # Token counts already computed (e.g., packing after tokenization)
            return {
                cid: float(count) for cid, count in meta.component_token_counts.items()
            }

        # Token counts not available - distribute total tokens by sample count ratio.
        # This happens when packing occurred before tokenization.
        total_weight = self._config.get_weight(record)
        total_samples = sum(meta.component_sample_counts.values())

        if total_samples == 0:
            # Shouldn't happen, but handle gracefully
            return {0: total_weight}

        return {
            cid: total_weight * (count / total_samples)
            for cid, count in meta.component_sample_counts.items()
        }

    def _update_lane_swrr(self, lane_id: int, chunk_id: int, state: _LaneState) -> None:
        """Update or create SWRR selector for a lane based on chunk mixture."""
        target: dict[int, float] = {}

        # Explicit mixture override (already in int form) overrides chunk mixture
        if self._config.mixture_override_by_id:
            target = dict(self._config.mixture_override_by_id)
        # Fall back to chunk mixture
        elif self._get_chunk_mixture is not None:
            chunk_mixture = self._get_chunk_mixture(lane_id, chunk_id)
            if chunk_mixture:
                target = dict(chunk_mixture)

        if not target:
            # No mixture info - will use uniform weights over available components
            state.swrr = None
            return

        # Normalize target weights
        total = sum(target.values())
        if total > 0:
            target = {k: v / total for k, v in target.items()}

        # Component order for deterministic tie-breaking
        order = sorted(target.keys())

        if state.swrr is None:
            state.swrr = SmoothWeightedRoundRobin(target, order)
        else:
            # Update existing SWRR with new targets (preserves emission history)
            state.swrr.update_target(target)

    def _get_obsolete_components(self, state: _LaneState) -> set[int]:
        """Get components with buffered samples that are not in the current target.

        These are components from a previous mixture that still have samples
        waiting to be emitted. They should be drained gradually over time.
        """
        if state.swrr is None:
            return set()

        target_components = set(state.swrr.target_ratios.keys())
        buffered_components = {cid for cid, buf in state.buffers.items() if buf}
        return buffered_components - target_components

    def _should_drain_obsolete(self, state: _LaneState) -> bool:
        """Check if it's time to drain an obsolete sample based on the drain rate.

        Returns True if we have obsolete samples AND we've emitted enough
        normal samples since the last obsolete drain.
        """
        if self._config.obsolete_drain_rate <= 0:
            return False

        obsolete = self._get_obsolete_components(state)
        if not obsolete:
            return False

        # Drain interval: emit one obsolete for every N emissions
        # e.g., rate=0.1 means drain every 10 emissions
        drain_interval = int(1.0 / self._config.obsolete_drain_rate)
        return state.emissions_since_obsolete_drain >= drain_interval

    def _emit_one(self, lane_id: int, state: _LaneState) -> SampleRecord | None:
        """Emit one sample from the lane using SWRR selection.

        Selection priority:
        0. Periodically drain obsolete components (based on obsolete_drain_rate)
        1. If ideal component has single-component sample, emit it (original behavior)
        2. If ideal component can be served by multi-component sample, use benefit
           scoring to pick the best one
        3. Fall back to best available single-component sample
        4. Fall back to best available multi-component sample

        This preserves the original SWRR logic for single-component samples while
        gradually draining obsolete components and properly integrating
        multi-component (packed) samples.
        """
        if not self._has_buffered_samples(state):
            return None

        available_single = {cid for cid, buf in state.buffers.items() if buf}
        has_multi = bool(state.multi_component_buffer)

        if not available_single and not has_multi:
            return None

        # Priority 0: Periodically drain obsolete components
        # Obsolete components are those with buffered samples but no longer in the
        # current mixture target. We interleave them at obsolete_drain_rate to
        # prevent samples from being stuck indefinitely.
        if self._should_drain_obsolete(state):
            obsolete = self._get_obsolete_components(state)
            if obsolete:
                # Pick the obsolete component with smallest ID for determinism
                chosen = min(obsolete)
                buf = state.buffers[chosen]
                record, weight = buf.popleft()

                # Don't update SWRR - obsolete components aren't in the target
                # But do track for ratio monitoring
                state.emitted_by_component[chosen] += weight
                state.total_emitted += weight
                # Reset the counter after draining
                state.emissions_since_obsolete_drain = 0
                return record

        # Get the ideal component (what SWRR wants, ignoring availability)
        ideal = state.swrr.peek() if state.swrr else None

        # Priority 1: Ideal component has single-component sample (original behavior)
        if ideal is not None and ideal in available_single:
            buf = state.buffers[ideal]
            record, weight = buf.popleft()

            if state.swrr is not None:
                state.swrr.record(ideal, weight)

            state.emitted_by_component[ideal] += weight
            state.total_emitted += weight
            state.emissions_since_obsolete_drain += 1
            self._check_ratios(lane_id, state)
            return record

        # Priority 2: Multi-component sample that contributes to ideal component
        if has_multi and ideal is not None:
            # Find multi-component samples that contribute to the ideal component
            candidates = [
                (idx, m)
                for idx, m in enumerate(state.multi_component_buffer)
                if ideal in m.contributions
            ]

            if candidates:
                # Use benefit scoring among candidates
                deficits = state.swrr.get_deficits() if state.swrr else {}

                best_idx, best_multi = max(
                    candidates,
                    key=lambda pair: sum(
                        contrib * deficits.get(cid, 0.0)
                        for cid, contrib in pair[1].contributions.items()
                    ),
                )

                # Remove from buffer (need to find actual index in deque)
                del state.multi_component_buffer[best_idx]

                if state.swrr is not None:
                    state.swrr.record_multi(best_multi.contributions)

                for cid, weight in best_multi.contributions.items():
                    state.emitted_by_component[cid] += weight
                    state.total_emitted += weight

                state.emissions_since_obsolete_drain += 1
                self._check_ratios(lane_id, state)
                return best_multi.record

        # Priority 3: Best available single-component sample
        if available_single:
            if state.swrr is not None:
                chosen = state.swrr.select(available_single)
            else:
                chosen = min(available_single)

            if chosen is None:
                chosen = min(available_single)

            buf = state.buffers[chosen]
            record, weight = buf.popleft()

            if state.swrr is not None:
                state.swrr.record(chosen, weight)

            state.emitted_by_component[chosen] += weight
            state.total_emitted += weight
            state.emissions_since_obsolete_drain += 1
            self._check_ratios(lane_id, state)
            return record

        # Priority 4: Best available multi-component sample (benefit scoring)
        if has_multi:
            deficits = state.swrr.get_deficits() if state.swrr else {}

            best_idx = max(
                range(len(state.multi_component_buffer)),
                key=lambda idx: sum(
                    contrib * deficits.get(cid, 0.0)
                    for cid, contrib in state.multi_component_buffer[
                        idx
                    ].contributions.items()
                ),
            )

            multi = state.multi_component_buffer[best_idx]
            del state.multi_component_buffer[best_idx]

            if state.swrr is not None:
                state.swrr.record_multi(multi.contributions)

            for cid, weight in multi.contributions.items():
                state.emitted_by_component[cid] += weight
                state.total_emitted += weight

            state.emissions_since_obsolete_drain += 1
            self._check_ratios(lane_id, state)
            return multi.record

        return None

    def _check_ratios(self, lane_id: int, state: _LaneState) -> None:
        """Check if actual ratios deviate from target and warn if so."""
        if self._config.warn_tolerance is None:
            return

        if state.total_emitted < self._config.warn_warmup:
            return

        if state.swrr is None:
            return

        actual_ratios = state.swrr.get_actual_ratios()
        target = state.swrr.target_ratios

        for comp_id, target_ratio in target.items():
            actual = actual_ratios.get(comp_id, 0.0)
            deviation = abs(actual - target_ratio)
            if deviation > self._config.warn_tolerance:
                comp_name = "unknown"
                if self._get_component_name is not None:
                    name = self._get_component_name(comp_id)
                    if name is not None:
                        comp_name = name

                warnings.warn(
                    f"[zephon] Mixture ratio drift for lane {lane_id}, "
                    + f"component '{comp_name}' (id={comp_id}): "
                    + f"actual={actual:.3f}, target={target_ratio:.3f}, "
                    + f"deviation={deviation:.3f} "
                    + f"(warn_tolerance={self._config.warn_tolerance})",
                    RuntimeWarning,
                    stacklevel=4,
                )


class EnsureMixture(DefaultSetup):
    """Enforce mixture ratios using adaptive Smooth Weighted Round Robin.

    This operator reorders samples to match target mixture proportions using
    adaptive buffering: emit immediately when SWRR's ideal component is available,
    buffer when the desired component isn't present yet. Falls back to emitting
    best-available when max_buffer_size is reached.

    Component identity comes from SampleMeta.component_sample_counts, which maps
    component_id to sample count. For single-component samples, this is ``{cid: 1}``.
    For packed samples, it may contain multiple components. Target weights are
    derived from the chunk's mixture unless explicitly provided.

    The operator uses SWRR (Smooth Weighted Round Robin) for selection, which
    tracks deficit (target - actual) and always picks the component most
    "owed" samples. This ensures smooth, deterministic convergence.

    Adaptive behavior:
    - Interleaved input (A,B,A,B,...): Low latency, immediate emission
    - Sequential input (A,A,...,B,B,...): Buffers until desired component arrives
    - Single component: No buffering, immediate passthrough

    Args:
        max_buffer_size: Samples to buffer while waiting for the component the
            target needs next. Default 1000; when the buffer fills the operator
            force-emits, so the mixture may drift if a component stays scarce.
            ``None`` removes the cap and instead *discards* the surplus it cannot
            place on-target, on every flush (epoch boundaries and end of stream):
            the mixture stays exact, at the cost of dropping data. Useful when a
            component is rare in the stream (e.g. on-the-fly tokenization of a
            small, non-repeating dataset). Memory is then bounded only by
            ``flush_every_k_chunks``, which also bounds how much is dropped: a
            smaller value flushes (and discards the unplaced surplus) sooner,
            leaving the scarce component less time to absorb the buffer.
        drain_target_ratio: When forced to emit (buffer hits max_buffer_size), drain
            the buffer down to this fraction of max_buffer_size before stopping.
            Default is 0.8 (drain to 80% of max). This ensures meaningful progress
            when suboptimal emissions are required, rather than emitting just one
            sample per push_many call.  Ignored when ``max_buffer_size=None``.
        obsolete_drain_rate: Fraction of emissions reserved for draining obsolete
            components (those no longer in the current mixture target). Default is
            0.1 (10%), meaning 1 in every 10 emissions drains an obsolete sample
            if available. Set to 0 to disable. This ensures samples from previous
            mixtures are gradually drained rather than stuck indefinitely.
        weight_by: How to compute sample weights. Options:
            - "auto" (default): Auto-detect token field from common names
              (input_ids, tokens, token_ids, ids). Raises if not found.
            - "samples": Each sample has weight 1.
            - Explicit field name (e.g., "input_ids"): Use that field's length.
            - Callable: Custom function taking SampleRecord, returning float.
        warn_tolerance: If set, warn when mixture drift exceeds this value (0.05 = ±5%).
            If None (default), no warnings are emitted.
        warn_warmup: Minimum total emitted weight before ratio warnings are enabled.
            Default is 1000. This prevents spurious warnings during early sampling
            when ratios naturally fluctuate.
        mixture_override: Explicit mixture target {component_name: float}. If None,
            derived from chunk mixture via engine context. Names are converted to
            component IDs.
        parallelism: Parallelism level for this operator.
    """

    def __init__(
        self,
        *,
        max_buffer_size: int | None = 1000,
        drain_target_ratio: float = 0.8,
        obsolete_drain_rate: float = 0.1,
        weight_by: Callable[[SampleRecord], float]
        | Literal["samples", "auto"]
        | str = "auto",
        warn_tolerance: float | None = None,
        warn_warmup: float = 1000,
        mixture_override: Mapping[str, float] | None = None,
        parallelism: int = 1,
    ) -> None:
        DefaultSetup.__init__(self)

        if max_buffer_size is not None and max_buffer_size <= 0:
            raise ValueError("max_buffer_size must be positive (or None for unbounded)")
        if not 0 < drain_target_ratio < 1:
            raise ValueError("drain_target_ratio must be between 0 and 1 (exclusive)")
        if not 0 <= obsolete_drain_rate <= 1:
            raise ValueError("obsolete_drain_rate must be between 0 and 1")
        if warn_tolerance is not None and (warn_tolerance < 0 or warn_tolerance > 1):
            raise ValueError("warn_tolerance must be between 0 and 1")
        if warn_warmup < 0:
            raise ValueError("warn_warmup must be non-negative")

        self._parallelism = parallelism

        # Create shared config object - accumulator gets a reference to this,
        # and we populate context callbacks during setup/accumulator
        self._config = EnsureMixtureConfig(
            weight_by=weight_by,
            warn_tolerance=warn_tolerance,
            warn_warmup=warn_warmup,
            mixture_override=dict(mixture_override)
            if mixture_override is not None
            else None,
            max_buffer_size=max_buffer_size,
            drain_target_ratio=drain_target_ratio,
            obsolete_drain_rate=obsolete_drain_rate,
        )

    @property
    def weight_by(
        self,
    ) -> Callable[[SampleRecord], float] | Literal["samples", "auto"] | str:
        """Configured sample/token weighting policy."""
        return self._config.weight_by

    def setup(
        self,
        ctx: OpContext,
        stage_index: int,
        stage_name: str,
        op_index: int,
        collect_stats: bool,
    ) -> None:
        """Initialize operator with context from engine."""
        DefaultSetup.setup(self, ctx, stage_index, stage_name, op_index, collect_stats)
        self._populate_config_from_ctx(ctx._services)

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=False,
            preserves_cursor_order=False,
            batch_shape_sensitive=False,
            requires_serial_state=False,
            parallelism=self._parallelism,
            stall_on_epoch_boundary=False,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        # Populate mixture_override_by_id if not yet done (e.g., after pickling for workers)
        self._populate_config_from_ctx(ctx)
        return EnsureMixtureAccumulator(
            self._config,
            get_chunk_mixture=ctx.get("get_chunk_mixture"),
            get_component_name=ctx.get("get_component_name"),
        )

    def _populate_config_from_ctx(self, ctx: dict[str, Any]) -> None:
        """Populate config fields from context dictionary.

        Called from both setup() and accumulator() to handle the case where
        the operator is pickled for worker processes (which creates a fresh
        config without mixture_override_by_id populated).
        """
        # Convert explicit mixture override from names to IDs (idempotent)
        if (
            self._config.mixture_override
            and self._config.mixture_override_by_id is None
        ):
            get_component_id = ctx.get("get_component_id")
            if get_component_id is not None:
                self._config.mixture_override_by_id = {
                    get_component_id(name): weight
                    for name, weight in self._config.mixture_override.items()
                }

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        """Pass through - accumulator handles reordering."""
        return [elem]

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        """Pass through - accumulator handles reordering."""
        return list(elems)
