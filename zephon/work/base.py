# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Abstract base definitions for work sources and chunks."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Mapping, MutableMapping, Protocol, Sequence

from zephon.core.constants import SampleId
from zephon.io.dataset import Dataset
from zephon.work.mixture import MixtureSpec

MixtureComponent = str
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
    """

    components: SamplesPerComponent
    seed: int | None = None

    ### INTERNAL ATTRIBUTES ###
    _order_cache: list[SampleId] | None = field(init=False, default=None, repr=False)
    _order_cache_key: tuple | None = field(init=False, default=None, repr=False)
    _component_order: tuple[MixtureComponent, ...] = field(init=False, repr=False)
    _total_samples: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._component_order = tuple(self.components.keys())
        self._total_samples = sum(len(items) for items in self.components.values())

    def __len__(self) -> int:
        return self._total_samples

    def __iter__(self) -> Iterator[SampleId]:
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
    ) -> Iterator[SampleId]:
        cfg = self._resolve_config(config)

        if cfg.precompute:
            yield from self.materialize_order(cfg)
            return

        if self._total_samples == 0:
            return

        yield from self._iter_streaming(cfg)

    def materialize_order(self, config: MixtureReadConfig) -> list[SampleId]:
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
    ) -> SampleId:
        cfg = self._resolve_config(config)
        return self.materialize_order(cfg)[index]

    def _iter_streaming(self, config: MixtureReadConfig) -> Iterator[SampleId]:
        if not self._total_samples:
            return

        buckets = self._build_buckets(config.seed, config.within_component)
        if not buckets:
            return

        # Single-bucket fast path (covers both None/WRR/Random cases)
        if len(buckets) == 1:
            yield from buckets[0].items
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
    ) -> Iterator[SampleId]:
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
                yield next(b.it)
            except StopIteration:
                # drop exhausted bucket and its weight
                del active[idx]
                del weights[idx]

    def _emit_weighted_round_robin(self, buckets: list[_Bucket]) -> Iterator[SampleId]:
        """
        Smooth Weighted Round Robin (SWRR), streaming.

        Intuition:
        - Each component 'c' has a fixed weight W[c].
        - We maintain a score C[c] (starts at 0). On each step:
            1) For all active c: C[c] += W[c]
            2) Pick component k with maximum C[k] (ties → earlier component wins)
            3) Emit one sample from k
            4) C[k] -= sum(W[active])   # subtract *total* so k's score drops back

        Properties:
        - Over time, emits in proportion to weights.
        - Deterministic given insertion order and inputs.
        - O(N) work per emitted sample (N = #active components).

        Edge cases:
        - If a bucket runs out of items, we remove it and decrease the total weight.
        - Floating point drift can make 'total' slightly negative; we clamp to 0.0.
        - If total reaches 0 (e.g., only empty/removed weights remain), we just drain
            the remaining iterators in insertion order.
        """
        # Filter/fast paths
        active = [b for b in buckets if b.weight > 0.0]
        if not active:
            return
        if len(active) == 1:
            yield from active[0].items
            return

        # Stable tie-breaker: remember original order index
        index = {b.name: i for i, b in enumerate(active)}

        # Per-component state
        weights = {b.name: float(b.weight) for b in active}
        iters = {b.name: b.it for b in active}
        current = {b.name: 0.0 for b in active}

        total = sum(weights.values())
        # (Should be > 0.0 because we filtered, assert defensively)
        assert total > 0.0, "No positive weights in WRR"

        while weights:
            # 1) Everyone accrues their weight
            for name in list(weights.keys()):
                current[name] += weights[name]

            # 2) Pick the argmax score; break ties by original order (smaller index wins)
            chosen_name = max(weights.keys(), key=lambda n: (current[n], -index[n]))

            # 3) Emit one item; reduce its score by the total
            current[chosen_name] -= total
            it = iters[chosen_name]
            try:
                yield next(it)
                continue  # still active → next round
            except StopIteration:
                # 4) Exhausted: remove chosen bucket
                w = weights.pop(chosen_name)
                iters.pop(chosen_name, None)
                current.pop(chosen_name, None)
                total -= w

                # Guard against tiny negative due to FP rounding
                if total < 0.0:
                    total = 0.0

                # If no total weight left, just drain what's left linearly
                if total == 0.0:
                    for n in sorted(iters.keys(), key=lambda n: index[n]):
                        for x in iters[n]:
                            yield x
                    break

    def state_dict(self) -> dict[str, Any]:
        """Portable, JSON-friendly snapshot of this chunk."""
        # Serialize components as an ordered list of (name, items-as-lists)
        comps_serial: list[tuple[str, list[list[int]]]] = []
        for name in self._component_order:
            items = self.components.get(name, [])
            # Each SampleId is a tuple[int,int,int] → store as [int,int,int]
            comps_serial.append((name, [list(sid) for sid in items]))

        return {
            "version": 1,
            "seed": None if self.seed is None else int(self.seed),
            "components": comps_serial,  # preserves insertion order
            "component_order": list(self._component_order),  # redundant but explicit
            # Optional sanity field — consumers may ignore
            "total_samples": int(self._total_samples),
        }

    @classmethod
    def from_state(cls, payload: Mapping[str, Any]) -> "WorkChunk":
        """Rebuild a WorkChunk from state_dict()."""
        version = int(payload.get("version", 0))
        if version != 1:
            raise ValueError(f"Unsupported WorkChunk state version: {version}")

        comps_in: Sequence[tuple[str, Sequence[Sequence[int]]]] = payload["components"]

        # Use a plain dict; insertion order matches iteration order in Python 3.7+
        comps: dict[str, list[SampleId]] = {}

        for name, items in comps_in:
            restored: list[SampleId] = []
            for raw in items:
                if len(raw) != 3:
                    raise ValueError(f"Bad SampleId for component {name}: {raw!r}")
                # IMPORTANT: build a fixed-length tuple to avoid tuple[int, ...]
                a, b, c = int(raw[0]), int(raw[1]), int(raw[2])
                restored.append((a, b, c))  # this is SampleId
            comps[name] = restored

        seed = payload.get("seed", None)
        chunk = cls(components=comps, seed=None if seed is None else int(seed))

        # Optional: honor serialized order explicitly (should already match)
        co = payload.get("component_order")
        if co is not None and tuple(co) != chunk._component_order:
            # rebuild in the serialized order using a new dict literal to set insertion order
            ordered = {name: comps[name] for name in co}
            chunk.components = ordered
            chunk.__post_init__()  # recompute internal caches

        return chunk


class WorkSource(Protocol):
    """Protocol for producing work chunks and supporting random access."""

    def next_chunk_for(
        self,
        lane: int,
        *,
        worker_id: int = 0,
        workers_per_rank: int = 1,
        canonical_replicas: int = 1,  # work source might need to know this for total number of lanes.
    ) -> WorkChunk | None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    def supports_indexing(self) -> bool: ...

    def __len__(self) -> int: ...

    def sample_id_at(self, index: int) -> SampleId: ...

    @property
    def datasets_by_id(self) -> Mapping[int, Dataset]: ...
