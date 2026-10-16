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

from zephon._internal.flush_context import in_shutdown_flush
from zephon._internal.utils.length_extraction import extract_length
from zephon._internal.utils.swrr import SmoothWeightedRoundRobin
from zephon.ops.accumulators import Accumulator, ReadyBatch
from zephon.ops.base import BaseOp, OpContext
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
    # Buffer limit before forced emission. None waits for SWRR's next component
    # or an obsolete drain slot, with no memory cap.
    max_buffer_size: int | None
    drain_target_ratio: (
        float  # When forced to emit, drain to this fraction of max_buffer_size
    )
    obsolete_drain_rate: float  # One obsolete record per int(1 / rate) normal emissions

    # Explicit mixture override converted to int form during setup
    mixture_override_by_id: dict[int, float] | None = None
    continues_after_exhaustion: frozenset[int] = frozenset()

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


@dataclass(slots=True)
class _BufferedRecord:
    """A buffered record with precomputed component contributions."""

    record: SampleRecord
    contributions: dict[int, float]  # component_id -> weight


@dataclass
class _LaneState:
    """Per-lane state for SWRR reordering with buffering.

    Single-component samples use per-component buffers.
    Multi-component (packed) samples go into a separate buffer and use
    benefit scoring for selection.
    """

    # Per-component buffers for single-component samples: component_id -> deque[(record, weight)]
    buffers: dict[int, deque[tuple[SampleRecord, float]]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    # Separate buffer for multi-component (packed) samples.
    # These require benefit scoring since they contribute to multiple components.
    multi_component_buffer: deque[_BufferedRecord] = field(default_factory=deque)
    # Shared FIFO and drain allowance for obsolete records. Reclassification
    # appends component queues then packs, not global arrival order.
    obsolete_buffer: deque[_BufferedRecord] = field(default_factory=deque)
    # Availability index: avoid scanning packed records on every selection.
    multi_component_counts: dict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    # SWRR selector for this lane (created lazily with first chunk's mixture)
    swrr: SmoothWeightedRoundRobin[int] | None = None
    # Highest chunk id adopted this epoch; late records cannot restore older targets.
    target_chunk_id: int | None = None
    # Track total samples currently buffered (for max_buffer_size check)
    total_buffered: int = 0
    # Warmup spans target changes, including temporary absence of a target.
    total_emitted: float = 0.0
    # Counter for obsolete interleaving: counts emissions since last obsolete drain
    emissions_since_obsolete_drain: int = 0
    # Exhausted components listed in continues_after_exhaustion.
    source_exhausted: set[int] = field(default_factory=set)


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
        self._obsolete_interval = (
            int(1.0 / config.obsolete_drain_rate)
            if config.obsolete_drain_rate
            else None
        )
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
        while state.obsolete_buffer:
            take(state.obsolete_buffer.popleft().record)

        if batch:
            ready.append((batch, 0))

        state.buffers.clear()
        state.multi_component_counts.clear()
        state.total_buffered = 0
        return ready, dropped

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        """Return True if buffered records remain (in ``lane_id`` if given)."""
        if lane_id is None:
            states: Iterable[_LaneState] = self._lanes.values()
        else:
            state = self._lanes.get(lane_id)
            states = (state,) if state is not None else ()
        return any(self._has_buffered_samples(s) for s in states)

    def _can_emit_optimal(self, state: _LaneState, ideal: int | None) -> bool:
        """Check if SWRR's ideal choice is available in the buffer.

        Returns True if we can emit without compromising mixture quality.
        This enables adaptive buffering: emit early when possible, buffer when needed.

        For multi-component samples: we also check if any packed sample contributes
        to the ideal component.
        """
        if state.swrr is None:
            # No target mixture configured - emit anything available
            return True

        return ideal is not None and bool(
            state.buffers.get(ideal) or state.multi_component_counts.get(ideal)
        )

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

            # Targets are per chunk. Reordering must not restore older targets.
            if state.target_chunk_id is None or chunk_id > state.target_chunk_id:
                self._update_lane_swrr(lane_id, chunk_id, state)
                state.target_chunk_id = chunk_id

            # Compute per-component contributions for this sample.
            contributions = self._compute_contributions(elem)

            self._buffer_record(state, elem, contributions)

            state.total_buffered += 1
            affected_lanes.add(lane_id)

        # Phase 2: Select from the buffered records, batching output per lane.
        for lane_id in affected_lanes:
            ready.extend(self._emit_ready(lane_id, self._lanes[lane_id]))
        return ready

    def _emit_ready(
        self, lane_id: int, state: _LaneState
    ) -> list[ReadyBatch[SampleRecord]]:
        """Emit ready records, allowing forced output under buffer pressure."""
        emitted: list[SampleRecord] = []
        max_buf = self._config.max_buffer_size
        force = max_buf is not None and state.total_buffered >= max_buf
        drain_target = (
            int(self._config.drain_target_ratio * max_buf) if max_buf is not None else 0
        )
        # Buffer pressure permits suboptimal choices until the lower threshold
        # is reached. After that, continue only with normal selection/drain slots.
        while self._has_buffered_samples(state):
            record = self._emit_one(
                lane_id, state, force=force and state.total_buffered > drain_target
            )
            if record is None:
                break
            state.total_buffered -= 1
            emitted.append(record)
        return [(emitted, 0)] if emitted else []

    def on_source_exhausted(
        self, lane_id: int, component_id: int
    ) -> list[ReadyBatch[SampleRecord]]:
        """Remove an exhausted component from the target if source policy allows."""
        if component_id not in self._config.continues_after_exhaustion:
            return []
        state = self._lanes[lane_id]
        if component_id in state.source_exhausted:
            return []
        state.source_exhausted.add(component_id)
        if state.swrr is not None:
            self._set_target(state, state.swrr.target_ratios)
        # Release buffered survivors now; there may be no further input.
        return self._emit_ready(lane_id, state)

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[SampleRecord]]:
        """Emit remaining buffered records for the given lane(s).

        Bounded mode emits everything, following selection priorities before
        draining any obsolete tail, which may form a burst. Strict mode
        (``max_buffer_size=None``) discards everything still buffered, both at
        epoch boundaries and upstream close; emitting that surplus would skew
        the target mixture.

        ``lane_id=None`` flushes all lanes. Otherwise only that lane is affected,
        including its state reset when ``reset=True``.
        """
        if lane_id is None:
            lane_ids = list(self._lanes)
        else:
            lane_ids = [lane_id] if lane_id in self._lanes else []

        ready: list[ReadyBatch[SampleRecord]] = []
        discarded = 0
        discarded_lanes = 0
        discarded_obsolete = 0

        for lid in lane_ids:
            state = self._lanes[lid]

            if self._config.max_buffer_size is None:
                discarded_obsolete += len(state.obsolete_buffer)
                batches, dropped = self._discard_lane_buffers(state)
                ready.extend(batches)
                if dropped:
                    discarded += dropped
                    discarded_lanes += 1
            else:
                emitted_batch: list[SampleRecord] = []

                while self._has_buffered_samples(state):
                    record = self._emit_one(lid, state, force=True)
                    if record is not None:
                        state.total_buffered -= 1
                        emitted_batch.append(record)
                    else:
                        break

                if emitted_batch:
                    ready.append((emitted_batch, 0))

            # Recreate SWRR even with empty buffers: replay needs fresh emission
            # history and tie ordering. Clearing counters alone keeps the old
            # component order. Source exhaustion survives the reset.
            if reset:
                state.total_emitted = 0.0
                state.emissions_since_obsolete_drain = 0
                # A stalled upstream flush may split a chunk's output. Allow
                # the next record to reload its target even with the same chunk id.
                state.target_chunk_id = None
                state.swrr = None

        if discarded:
            self._total_discarded += discarded
            # Skewed strict-mode runs discard on roughly every flush, so warn
            # once and log the rest at DEBUG. The cumulative count is per
            # accumulator (one per rank), not run-wide across ranks.
            # An early-close drain is never consumed, so don't spend the warning on it.
            if not self._discard_warned and not in_shutdown_flush():
                self._discard_warned = True
                advice = (
                    "Raise flush_every_k_chunks to give buffered records more time. "
                )
                if discarded_obsolete:
                    advice += (
                        "For obsolete tails, consider raising obsolete_drain_rate. "
                    )
                logger.warning(
                    "EnsureMixture strict mode discarded %d record(s) across %d "
                    "lane(s), including %d obsolete record(s), to respect the "
                    "target and drain allowance (%d dropped cumulatively "
                    "on this rank). %sFurther discards on this rank are logged at DEBUG.",
                    discarded,
                    discarded_lanes,
                    discarded_obsolete,
                    self._total_discarded,
                    advice,
                )
            else:
                logger.debug(
                    "EnsureMixture strict mode discarded %d record(s) across %d "
                    "lane(s), including %d obsolete record(s), to respect the "
                    "target and drain allowance (%d dropped cumulatively "
                    "on this rank).",
                    discarded,
                    discarded_lanes,
                    discarded_obsolete,
                    self._total_discarded,
                )

        return ready

    def _has_buffered_samples(self, state: _LaneState) -> bool:
        """Check if lane has any buffered samples."""
        return any(buf for buf in state.buffers.values()) or bool(
            state.multi_component_buffer or state.obsolete_buffer
        )

    def _compute_contributions(self, record: SampleRecord) -> dict[int, float]:
        """Compute per-component contribution weights for a sample.

        For weight="samples": uses component_sample_counts directly.
        For token-based modes: uses component_token_counts if available,
        otherwise distributes the total token count proportionally by sample counts.

        Returns a dict mapping component_id -> weight for this sample. Keep zero
        weights so zero-weight records retain their component identity for selection.
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
        """Adopt the operator override, or the chunk target if no override is set."""
        target = self._config.mixture_override_by_id
        if not target and self._get_chunk_mixture is not None:
            target = self._get_chunk_mixture(lane_id, chunk_id)
        self._set_target(state, target or None)

    def _set_target(self, state: _LaneState, target: dict[int, float] | None) -> None:
        """Filter out exhausted components and update buffer eligibility."""
        previous = state.swrr.target_ratios if state.swrr is not None else None
        if target is None:
            state.swrr = None  # No target configured: unrestricted output.
        else:
            target = {
                cid: weight
                for cid, weight in target.items()
                if weight > 0 and cid not in state.source_exhausted
            }
            # Keep an empty SWRR target distinct from unrestricted output.
            if state.swrr is None:
                # Sorted component IDs make initial tie-breaking deterministic.
                state.swrr = SmoothWeightedRoundRobin(target, sorted(target))
            else:
                # Preserve emission history and tie order across target updates.
                state.swrr.update_target(target)

        old_components = previous.keys() if previous is not None else None
        new_components = target.keys() if target is not None else None
        if not state.total_buffered or old_components == new_components:
            return

        if old_components is not None and new_components is not None:
            removed = old_components - new_components
            added = new_components - old_components
            if not (
                any(
                    state.buffers.get(cid) or state.multi_component_counts.get(cid)
                    for cid in removed
                )
                or (added and state.obsolete_buffer)
            ):
                return

        # Preserve the existing obsolete FIFO ahead of newly obsolete records.
        # No full-buffer scans on ordinary emissions or weight-only updates.
        single, multi, obsolete = (
            state.buffers,
            state.multi_component_buffer,
            state.obsolete_buffer,
        )
        state.buffers = defaultdict(deque)
        state.multi_component_buffer = deque()
        state.obsolete_buffer = deque()
        state.multi_component_counts.clear()
        while obsolete:
            item = obsolete.popleft()
            self._buffer_record(state, item.record, item.contributions)
        for cid, buf in single.items():
            while buf:
                record, weight = buf.popleft()
                self._buffer_record(state, record, {cid: weight})
        while multi:
            item = multi.popleft()
            self._buffer_record(state, item.record, item.contributions)

    def _buffer_record(
        self, state: _LaneState, record: SampleRecord, contributions: dict[int, float]
    ) -> None:
        """Keep obsolete records out of normal selection.

        With a target, any positive off-target contribution or the absence of
        all target components makes a record obsolete, even at zero weight.
        """
        target = state.swrr.target_ratios if state.swrr is not None else None
        if len(contributions) == 1:
            cid, weight = next(iter(contributions.items()))
            if target is None or cid in target:
                state.buffers[cid].append((record, weight))
                return
            # Even zero-weight records need a drain slot if their component is
            # absent from the target: normal selection can never choose them.
        elif target is None or (
            any(cid in target for cid in contributions)
            and not any(
                weight > 0 and cid not in target
                for cid, weight in contributions.items()
            )
        ):
            state.multi_component_buffer.append(_BufferedRecord(record, contributions))
            for cid in contributions:
                state.multi_component_counts[cid] += 1
            return
        state.obsolete_buffer.append(_BufferedRecord(record, contributions))

    def _should_drain_obsolete(self, state: _LaneState) -> bool:
        """Whether an obsolete record is waiting and its drain slot is due."""
        # TODO: Explore smooth decay of retired content instead of a fixed interval.
        return bool(state.obsolete_buffer) and (
            self._obsolete_interval is not None
            and state.emissions_since_obsolete_drain >= self._obsolete_interval
        )

    def _take_multi(self, state: _LaneState, index: int) -> _BufferedRecord:
        """Remove a packed record and update the component-availability index."""
        multi = state.multi_component_buffer[index]
        del state.multi_component_buffer[index]
        for cid in multi.contributions:
            state.multi_component_counts[cid] -= 1
            if not state.multi_component_counts[cid]:
                del state.multi_component_counts[cid]
        return multi

    def _emit_single(
        self, lane_id: int, state: _LaneState, component_id: int
    ) -> SampleRecord:
        """Account for a single-component emission without allocating a packed wrapper."""
        record, weight = state.buffers[component_id].popleft()
        if state.swrr is not None:
            state.swrr.record(component_id, weight)
        state.total_emitted += weight
        self._finish_emission(lane_id, state, obsolete=False)
        return record

    def _finish_emission(
        self, lane_id: int, state: _LaneState, *, obsolete: bool
    ) -> None:
        """Count output records toward the next drain slot, regardless of weight."""
        if obsolete:
            state.emissions_since_obsolete_drain = 0
        elif self._obsolete_interval is not None:
            # Save at most one slot while obsolete records are absent.
            state.emissions_since_obsolete_drain = min(
                state.emissions_since_obsolete_drain + 1, self._obsolete_interval
            )
        self._check_ratios(lane_id, state)

    def _emit_one(
        self, lane_id: int, state: _LaneState, *, force: bool = False
    ) -> SampleRecord | None:
        """Select a record using the priorities below.

        Priorities 3 and 4 require forced emission or an unconfigured target;
        otherwise, wait for the ideal component after checking the drain slot.
        Forced emission can bypass drain spacing to bound memory or finish a
        lossless flush. Strict flushes discard buffered records instead.
        """
        # Priority 0: One shared FIFO slot for obsolete records, including mixed packs.
        obsolete = self._should_drain_obsolete(state)
        if obsolete:
            item = state.obsolete_buffer.popleft()
        else:
            # What does SWRR ideally want (ignoring availability)?
            ideal = state.swrr.peek() if state.swrr is not None else None
            # A live but unavailable ideal component still blocks normal output.
            # Exhaustion removes its target share; mere scarcity does not.
            if not force and not self._can_emit_optimal(state, ideal):
                return None

            available = {cid for cid, buf in state.buffers.items() if buf}
            if ideal is not None and ideal in available:
                # Priority 1: Prefer the ideal component's single-record queue.
                return self._emit_single(lane_id, state, ideal)
            else:
                # Priority 2: Packs serving the ideal component take precedence
                # over single-component fallbacks.
                candidates = [
                    (idx, packed)
                    for idx, packed in enumerate(state.multi_component_buffer)
                    if ideal is not None and ideal in packed.contributions
                ]
                if not candidates and available:
                    # Priority 3: Best available single-component fallback.
                    chosen = (
                        state.swrr.select(available) if state.swrr is not None else None
                    )
                    if chosen is None:
                        chosen = min(available)
                    return self._emit_single(lane_id, state, chosen)
                elif candidates or state.multi_component_buffer:
                    if not candidates:
                        # Priority 4: No ideal-serving pack or single record is
                        # available; score all remaining on-target packs.
                        candidates = list(enumerate(state.multi_component_buffer))
                    # Score packs by how much they serve underrepresented components,
                    # using the same deficits as single-component selection.
                    deficits = (
                        state.swrr.get_deficits() if state.swrr is not None else {}
                    )
                    best_idx, _ = max(
                        candidates,
                        key=lambda pair: sum(
                            weight * deficits.get(cid, 0.0)
                            for cid, weight in pair[1].contributions.items()
                        ),
                    )
                    item = self._take_multi(state, best_idx)
                elif force and state.obsolete_buffer:
                    # Priority 5: Force-drain obsolete records when no on-target
                    # records remain, even without live data to interleave.
                    item = state.obsolete_buffer.popleft()
                    obsolete = True
                else:
                    return None

        # Credit every contribution, including the live part of a retired pack.
        # SWRR excludes off-target mass when calculating survivor deficits.
        if state.swrr is not None:
            state.swrr.record_multi(item.contributions)
        state.total_emitted += sum(item.contributions.values())
        self._finish_emission(lane_id, state, obsolete=obsolete)
        return item.record

    def _check_ratios(self, lane_id: int, state: _LaneState) -> None:
        """Check if actual ratios deviate from target and warn if so."""
        if self._config.warn_tolerance is None:
            return

        if state.total_emitted < self._config.warn_warmup:
            return

        if state.swrr is None:
            return

        # Monitoring retains all contributions. Compare ratios within the current
        # target so off-target emissions do not dilute them.
        actual_ratios = state.swrr.get_actual_ratios()
        target = state.swrr.target_ratios
        actual_total = sum(actual_ratios.get(cid, 0.0) for cid in target)
        if not actual_total:
            return
        actual_ratios = {
            cid: actual_ratios.get(cid, 0.0) / actual_total for cid in target
        }

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


class EnsureMixture(BaseOp):
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

    EOD removes components listed by ``WorkSource.continues_after_exhaustion()``
    from the target, including explicit overrides, and renormalizes the remaining
    weights. ``StaticMixtureWorkSource`` lists none. Late records follow
    ``obsolete_drain_rate`` in both modes.

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
            the mixture is enforced at the cost of dropping data. Useful when a
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
        obsolete_drain_rate: Emit one obsolete record per ``int(1 / rate)`` normal
            output records, regardless of token count. The effective share is
            ``1 / (int(1 / rate) + 1)``: default 0.1 means one in eleven;
            values above 0.5 alternate (50%); 0 disables periodic drainage.
            A record is obsolete if it has any positive off-target contribution
            or no component in the target. Its on-target contributions still
            count toward the mixture. At most one slot is saved while the queue
            is empty. Bounded forced drains and flushes may exceed this rate;
            strict flushes discard leftovers.
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
            component IDs. EOD handling also applies to this override.
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
        super().__init__()

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

    def setup(self, ctx: OpContext) -> None:
        """Initialize operator with context from engine."""
        super().setup(ctx)
        self._populate_config_from_ctx(ctx._services)

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=False,
            preserves_cursor_order=False,
            batch_shape_sensitive=False,
            requires_serial_state=False,
            parallelism=self._parallelism,
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
        self._config.continues_after_exhaustion = ctx.get(
            "continues_after_exhaustion", frozenset()
        )
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
