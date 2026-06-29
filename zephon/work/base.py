# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Abstract base definitions for work sources and chunks."""

import copy
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

from zephon.core.checkpoint import (
    WORK_CHUNK_VERSION,
    WorkChunkStateV2,
)
from zephon.core.constants import SampleId
from zephon.io.dataset import Dataset
from zephon.utils.swrr import swrr_iterate
from zephon.work.mixture import MixtureSpec

MixtureComponent = str
SourcedSampleId = tuple[SampleId, MixtureComponent]
SamplesPerComponent = MutableMapping[MixtureComponent, list[SampleId]]


class MixtureReadMode(str, Enum):
    """Supported sampling strategies when iterating over a ``WorkChunk``."""

    WEIGHTED_RANDOM = "weighted_random"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"


class ComponentOrder(str, Enum):
    """How to traverse samples within an individual mixture component."""

    AS_IS = "as_is"
    SHUFFLE = "shuffle"


@dataclass(frozen=True)
class MixtureReadConfig:
    """Reader configuration controlling deterministic traversal of a chunk."""

    mode: MixtureReadMode = MixtureReadMode.WEIGHTED_ROUND_ROBIN
    seed: int | None = None
    precompute: bool = False  # Whether to pre-compute sample order. Might cause performance spikes when requesting the first item.
    within_component: ComponentOrder = (
        ComponentOrder.AS_IS
    )  # how to yield samples within the same component.


@dataclass
class _Bucket:
    name: str
    items: Sequence[SampleId]  # for trivial concatenate
    it: Iterator[SampleId]  # for streaming
    weight: float


@dataclass
class WorkChunk:
    """Bundle of sample identifiers handed to the engine for processing.

    ``components`` stores each mixture component (for example, ``"German"``) and
    the ordered sample identifiers that belong to it.  Mapping insertion order is used
    as a stable tie-breaker whenever behaviour depends on component ordering.

    ``target_mixture``, when set, is the per-component target the engine should
    deliver to downstream mixture correctors (``ensure_mixture``) *instead of*
    the counted composition. Token-aware work sources stamp the user's token
    mixture here because their chunks deliberately carry a different sample
    composition (long-doc components contribute fewer pointers); the counted
    :attr:`mixture` stays composition-derived for within-chunk interleaving.
    """

    components: SamplesPerComponent
    seed: int | None = None
    target_mixture: Mapping[str, float] | None = None

    ### INTERNAL ATTRIBUTES ###
    _order_cache: list[SourcedSampleId] | None = field(
        init=False, default=None, repr=False
    )
    _order_cache_key: tuple | None = field(init=False, default=None, repr=False)
    _component_order: tuple[MixtureComponent, ...] = field(init=False, repr=False)
    _total_samples: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._component_order = tuple(self.components.keys())
        self._total_samples = sum(len(items) for items in self.components.values())
        if self.target_mixture is not None:
            # Normalize to canonical ratios (matches the `mixture` property) and
            # validate (non-empty, positive) in one step.
            self.target_mixture = MixtureSpec(self.target_mixture).normalized

    def __len__(self) -> int:
        return self._total_samples

    def __iter__(self) -> Iterator[SourcedSampleId]:
        yield from self.iter_samples()

    @property
    def mixture(self) -> Mapping[str, float]:
        """Return normalized mixture weights for the *active* components."""
        comps = [c for c in self._component_order if self.components.get(c)]
        if not comps:
            return {}

        # Otherwise derive from counts (skip empty buckets so MixtureSpec stays >0).
        counts = {c: len(self.components[c]) for c in comps}
        return MixtureSpec(counts).normalized

    def _resolve_config(self, config: MixtureReadConfig | None) -> MixtureReadConfig:
        """Return an effective config with defaults applied (no mutation of input)."""
        base = config or MixtureReadConfig()
        # Fill seed from chunk if missing.
        seed = base.seed if base.seed is not None else self.seed
        # If you want a dynamic default (None for single-component), keep mode as-is here
        # and change MixtureReadConfig.mode to Optional[MixtureReadMode] with default None.
        return MixtureReadConfig(
            mode=base.mode,
            seed=seed,
            precompute=base.precompute,
            within_component=base.within_component,
        )

    def iter_samples(
        self, config: MixtureReadConfig | None = None
    ) -> Iterator[SourcedSampleId]:
        """Iterate samples in mixture order, yielding (sample_id, component_name) tuples."""
        cfg = self._resolve_config(config)

        if cfg.precompute:
            yield from self.materialize_order(cfg)
            return

        if self._total_samples == 0:
            return

        yield from self._iter_streaming(cfg)

    def materialize_order(self, config: MixtureReadConfig) -> list[SourcedSampleId]:
        key = (config.mode, config.seed, config.within_component)

        if self._order_cache is not None and self._order_cache_key == key:
            return self._order_cache

        # Populate cache
        no_precompute_cfg = MixtureReadConfig(
            mode=config.mode,
            seed=config.seed,
            precompute=False,
            within_component=config.within_component,
        )  # avoid recursion
        order = list(self.iter_samples(no_precompute_cfg))
        self._order_cache = order
        self._order_cache_key = key

        return self._order_cache

    def sample_at(
        self, index: int, config: MixtureReadConfig | None = None
    ) -> SourcedSampleId:
        cfg = self._resolve_config(config)
        return self.materialize_order(cfg)[index]

    def _iter_streaming(self, config: MixtureReadConfig) -> Iterator[SourcedSampleId]:
        if not self._total_samples:
            return

        buckets = self._build_buckets(config.seed, config.within_component)
        if not buckets:
            return

        # Single-bucket fast path (covers both None/WRR/Random cases)
        if len(buckets) == 1:
            name = buckets[0].name
            for sample_id in buckets[0].items:
                yield (sample_id, name)
            return

        if config.mode is MixtureReadMode.WEIGHTED_RANDOM:
            yield from self._emit_weighted_random(buckets, config.seed)
        elif config.mode is MixtureReadMode.WEIGHTED_ROUND_ROBIN:
            yield from self._emit_weighted_round_robin(buckets)
        else:
            raise ValueError(f"Unsupported mixture read mode: {config.mode}")

    def _build_buckets(
        self,
        seed: int | None,
        within_component: ComponentOrder,
    ) -> list[_Bucket]:
        from random import Random

        mix = self.mixture
        if not mix:
            return []

        rng = Random()
        buckets: list[_Bucket] = []
        for pos, name in enumerate(self._component_order):
            items = self.components.get(name, [])
            if not items:
                continue
            seq: list[SampleId]

            if within_component is ComponentOrder.SHUFFLE:
                seq = list(items)
                effective_seed = seed if seed is not None else self.seed
                if effective_seed is not None:
                    rng.seed((effective_seed << 16) + pos)
                rng.shuffle(seq)
            else:
                seq = items

            buckets.append(
                _Bucket(name=name, items=seq, it=iter(seq), weight=mix[name])
            )

        return buckets

    # further modes: just random next sample (random without weights), trivial round robin
    def _emit_weighted_random(
        self, buckets: list[_Bucket], seed: int | None
    ) -> Iterator[SourcedSampleId]:
        from random import Random

        active = list(buckets)
        if not active:
            return

        weights = [b.weight for b in active]
        rng = Random(seed)

        while active:
            # pick a bucket index according to its weight
            idx = rng.choices(range(len(active)), weights=weights, k=1)[0]
            b = active[idx]
            try:
                yield (next(b.it), b.name)
            except StopIteration:
                # drop exhausted bucket and its weight
                del active[idx]
                del weights[idx]

    def _emit_weighted_round_robin(
        self, buckets: list[_Bucket]
    ) -> Iterator[SourcedSampleId]:
        """Smooth Weighted Round Robin (SWRR) using shared implementation.

        Delegates to swrr_iterate() which provides deterministic, proportional
        emission matching target weights over time.
        """
        yield from swrr_iterate(
            components={b.name: b.items for b in buckets},
            weights={b.name: b.weight for b in buckets},
            order=[b.name for b in buckets],
        )

    def state_dict(self) -> dict[str, Any]:
        """Portable, JSON-friendly snapshot of this chunk (always the current version)."""
        comps_serial: list[tuple[str, list[list[int]]]] = []
        for name in self._component_order:
            items = self.components.get(name, [])
            comps_serial.append((name, [list(sid) for sid in items]))

        state = WorkChunkStateV2(
            version=WORK_CHUNK_VERSION,
            seed=None if self.seed is None else int(self.seed),
            components=comps_serial,
            component_order=list(self._component_order),
            total_samples=int(self._total_samples),
            target_mixture=(
                None
                if self.target_mixture is None
                else {k: float(v) for k, v in self.target_mixture.items()}
            ),
        )
        return state.to_dict()

    @classmethod
    def from_state(cls, payload: Mapping[str, Any]) -> "WorkChunk":
        """Rebuild a WorkChunk from state_dict()."""
        ckpt = WorkChunkStateV2.load(payload)

        comps: dict[str, list[SampleId]] = {}
        for name, items in ckpt.components:
            restored: list[SampleId] = []
            for raw in items:
                if len(raw) != 3:
                    raise ValueError(f"Bad SampleId for component {name}: {raw!r}")
                a, b, c = int(raw[0]), int(raw[1]), int(raw[2])
                restored.append((a, b, c))
            comps[name] = restored

        chunk = cls(
            components=comps,
            seed=ckpt.seed,
            target_mixture=ckpt.target_mixture,
        )

        if (
            ckpt.component_order
            and tuple(ckpt.component_order) != chunk._component_order
        ):
            ordered = {name: comps[name] for name in ckpt.component_order}
            chunk.components = ordered
            chunk.__post_init__()

        if (
            ckpt.total_samples is not None
            and ckpt.total_samples != chunk._total_samples
        ):
            raise ValueError(
                f"WorkChunk total_samples mismatch: payload={ckpt.total_samples}, "
                f"computed={chunk._total_samples}"
            )

        return chunk


class WorkSource(ABC):
    """Abstract producer of `WorkChunk` instances for the engine."""

    def __init__(self):
        self._lane: int | None = None
        self._canon: int | None = None
        self._cloned = False

    def next_chunk(self) -> WorkChunk | None:
        raise NotImplementedError()

    def state_dict(self) -> dict:
        return {
            "lane_id": self._lane,
            "canonical_replicas": self._canon,
            "chunk_size_hint": self.chunk_size_hint(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._verify_base_state(int(state["lane_id"]), int(state["canonical_replicas"]))

    def _verify_base_state(self, lane_id: int, canonical_replicas: int) -> None:
        """Verify lane/canonical-replicas identity against this instance.

        Subclasses that route state through a typed schema should call this
        directly (passing schema fields) instead of ``super().load_state_dict(state)``
        so a future rename in the schema does not desync from the base.
        """
        if lane_id != self._lane:
            raise RuntimeError("Lane mismatch loading LaneWorkSource state.")
        if canonical_replicas != self._canon:
            raise RuntimeError("canonical_replicas changed; migration required.")

    def _bind_lane(self, lane_id: int, canonical_replicas: int) -> None:
        """Bind the clone to a specific lane.

        Update any internal seed/cursors using lane_id if needed.
        """
        if not self._cloned:
            raise RuntimeError(
                "State Error: The WorkSource should have been cloned internally before binding it."
            )

        self._lane = int(lane_id)
        self._canon = int(canonical_replicas)

    def clone_for_lane(self, lane_id: int, canonical_replicas: int) -> "WorkSource":
        """Default clone strategy: config-based if available, else deepcopy."""
        if self._cloned:
            raise RuntimeError(
                "State Error: clone_for_lane should only be called on user-defined WorkSource instances."
            )
        ws = copy.deepcopy(self)
        ws._cloned = True
        ws._bind_lane(lane_id, canonical_replicas)

        # TODO(MaxiBoether): Implement an alternative to deepcopy if worksources support it.
        # try:
        #    cfg = self.config_dict()
        #    ws = type(self).from_config(cfg)
        # except Exception:
        # fall back to deepcopy; ensure your subclass is deepcopy-safe
        #    ws = copy.deepcopy(self)
        return ws

    def supports_indexing(self) -> bool:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def sample_id_at(self, index: int) -> SampleId:
        raise NotImplementedError()

    @property
    def datasets_by_id(self) -> Mapping[int, Dataset]:
        raise NotImplementedError()

    def chunk_size_hint(self) -> int | None:
        """Return fixed chunk size if constant.

        Used for deterministic resume validation. Default None.
        """
        return None
