# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Abstract base definitions for work sources and chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Mapping, MutableMapping, Protocol, Sequence

from zephon.core.constants import SampleId
from zephon.io.dataset import Dataset
from zephon.work.mixture import MixtureSpec

MixtureComponent = str
SamplesPerComponent = MutableMapping[MixtureComponent, list[SampleId]]
MixtureWeights = dict[MixtureComponent, float]


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

    mode: MixtureReadMode | None = None
    seed: int | None = None
    precompute: bool = False  # Whether to pre-compute sample order. Might cause performance spikes when requesting the first item.
    within_component: ComponentOrder = (
        ComponentOrder.AS_IS
    )  # how to yield samples within the same component.


@dataclass(frozen=True)
class _ResolvedMixtureConfig:  # TODO(MaxiBoether): can we dynamically define this class
    """Internal variant of ``MixtureReadConfig`` with defaults resolved."""

    mode: MixtureReadMode | None
    seed: int | None
    precompute: bool
    within_component: ComponentOrder


@dataclass
class WorkChunk:
    """Bundle of sample identifiers handed to the engine for processing.

    ``components`` stores each mixture component (for example, ``"German"``) and
    the ordered sample identifiers that belong to it.  The insertion order of the
    mapping is used as a stable tie-breaker whenever behaviour depends on
    component ordering (for instance, within the weighted round-robin iterator).
    """

    components: SamplesPerComponent
    seed: int | None = None
    explicit_mixture: MixtureWeights | None = None

    ### INTERNAL ATTRIBUTES ###
    _mixture_cache: MixtureWeights | None = field(init=False, default=None, repr=False)
    _order_cache: list[SampleId] | None = field(init=False, default=None, repr=False)
    _order_cache_config: _ResolvedMixtureConfig | None = field(
        init=False, default=None, repr=False
    )
    _shards_cache: set[tuple[int, int]] | None = field(
        init=False, default=None, repr=False
    )
    _component_order: tuple[MixtureComponent, ...] = field(init=False, repr=False)
    _total_samples: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        components: dict[MixtureComponent, list[SampleId]] = {}
        for (
            component,
            sample_ids,
        ) in (
            self.components.items()
        ):  # TODO(MaxiBoether): why? doesnt this literally clone the list
            components[component] = list(sample_ids)
        self.components = components
        # NOTE: component iteration order follows the original mapping insertion
        # order emitted by the WorkSource.  Consumers rely on this for
        # deterministic tie-breaking when weights alone are insufficient.
        self._component_order = tuple(self.components.keys())
        self._total_samples = sum(len(items) for items in self.components.values())

    def __len__(self) -> int:
        return self._total_samples

    def __iter__(self) -> Iterator[SampleId]:
        yield from self.iter_samples()

    @property
    def mixture(self) -> MixtureWeights:
        if self._mixture_cache is None:
            self._mixture_cache = self._compute_mixture()
        return dict(self._mixture_cache)  # returning a copy necessary?

    @property
    def shards(self) -> set[tuple[int, int]]:
        if self._shards_cache is None:
            shard_keys: set[tuple[int, int]] = set()
            for sample_ids in self.components.values():
                shard_keys.update(
                    (sample_id[0], sample_id[1]) for sample_id in sample_ids
                )
            self._shards_cache = shard_keys
        return set(self._shards_cache)

    @property
    def sample_ids(self) -> list[SampleId]:
        return self.materialize_order()

    def iter_samples(
        self, config: MixtureReadConfig | None = None
    ) -> Iterator[SampleId]:
        resolved = self._resolve_config(config)  # should we cache this?
        if resolved.precompute:
            yield from self.materialize_order(config)
            return
        yield from self._iter_streaming(resolved)

    def materialize_order(
        self, config: MixtureReadConfig | None = None
    ) -> list[SampleId]:  # maybe this should be a cached property instead?
        resolved = self._resolve_config(config, force_precompute=True)
        if self._order_cache is not None and self._order_cache_config == resolved:
            return list(self._order_cache)  # copy necesary?
        order = list(self._iter_streaming(resolved))
        self._order_cache = order
        self._order_cache_config = resolved
        return list(order)

    def sample_at(
        self, index: int, config: MixtureReadConfig | None = None
    ) -> SampleId:
        order = self.materialize_order(config)
        return order[index]

    def _resolve_config(
        self,
        config: MixtureReadConfig | None,
        *,
        force_precompute: bool = False,
    ) -> _ResolvedMixtureConfig:
        default_mode = (
            MixtureReadMode.WEIGHTED_ROUND_ROBIN
            if self._has_multiple_components
            else None
        )
        cfg = config or MixtureReadConfig()
        mode = cfg.mode if cfg.mode is not None else default_mode
        seed = cfg.seed if cfg.seed is not None else self.seed
        precompute = cfg.precompute or force_precompute
        return _ResolvedMixtureConfig(
            mode=mode,
            seed=seed,
            precompute=precompute,
            within_component=cfg.within_component,
        )

    def _iter_streaming(self, config: _ResolvedMixtureConfig) -> Iterator[SampleId]:
        if not self._total_samples:
            return
        buckets = self._prepare_buckets(config)
        if not buckets:
            return
        if config.mode is None:
            if not self._has_multiple_components:
                print(
                    "You have more than 1 component. Are you sure you want trivial emitting?"
                )
            yield from self._emit_trivial(buckets)
        elif config.mode is MixtureReadMode.WEIGHTED_RANDOM:
            yield from self._emit_weighted_random(buckets, config.seed)
        elif config.mode is MixtureReadMode.WEIGHTED_ROUND_ROBIN:
            yield from self._emit_weighted_round_robin(buckets)
        else:
            raise ValueError(f"Unsupported mixture read mode: {config.mode}")

    def _prepare_buckets(
        self, config: _ResolvedMixtureConfig
    ) -> list[tuple[MixtureComponent, Sequence[SampleId], float]]:
        buckets: list[tuple[MixtureComponent, Sequence[SampleId], float]] = []
        mixture = self.mixture
        seed = config.seed
        for position, component in enumerate(self._component_order):
            items = self.components.get(component, [])
            if not items:
                continue
            if config.within_component is ComponentOrder.SHUFFLE:
                from random import Random

                effective_seed = seed if seed is not None else self.seed
                rand = Random()
                if effective_seed is not None:
                    rand.seed((effective_seed << 16) + position)
                shuffled = list(items)
                rand.shuffle(shuffled)
                entries: Sequence[SampleId] = shuffled
            else:
                entries = items
            weight = mixture.get(component, 0.0)
            buckets.append((component, entries, weight))
        return buckets

    def _emit_trivial(
        self, buckets: list[tuple[MixtureComponent, Sequence[SampleId], float]]
    ) -> Iterator[SampleId]:
        for _, items, _ in buckets:
            yield from items

    # further modes: just random next sample (random without weights), trivial round robin
    def _emit_weighted_random(
        self,
        buckets: list[tuple[MixtureComponent, Sequence[SampleId], float]],
        seed: int | None,
    ) -> Iterator[SampleId]:
        from random import Random

        rng = Random(seed)
        positions = {component: 0 for component, _, _ in buckets}
        active = [
            (component, list(items), weight)
            for component, items, weight in buckets
            if items and weight > 0.0
        ]
        if not active:
            yield from self._emit_trivial(buckets)
            return
        total_weight = sum(weight for _, _, weight in active)
        while active:
            # this is basically a random choice with weights we can simplify this.
            pick = rng.random() * total_weight
            cumulative = 0.0
            chosen_index = 0
            for idx, (component, _, weight) in enumerate(active):
                cumulative += weight
                if pick <= cumulative:
                    chosen_index = idx
                    break
            # we could similar to mixtera have one iterator per bucket and just yield the next one instead of keeping track of offsets here
            component, items, weight = active[chosen_index]
            offset = positions[component]
            yield items[offset]
            offset += 1
            positions[component] = offset
            if offset >= len(items):
                total_weight -= weight
                del active[chosen_index]
                positions.pop(component, None)
                if total_weight <= 0:
                    active = []
            if not active:
                break

    def _emit_weighted_round_robin(
        self,
        buckets: list[tuple[MixtureComponent, Sequence[SampleId], float]],
    ) -> Iterator[SampleId]:
        import math

        positions = {component: 0 for component, _, _ in buckets}
        active_components: list[tuple[MixtureComponent, Sequence[SampleId], float]] = [
            (component, items, weight)
            for component, items, weight in buckets
            if items and weight > 0.0
        ]
        if not active_components:
            yield from self._emit_trivial(buckets)
            return
        weights = {component: weight for component, _, weight in active_components}
        items_lookup = {component: items for component, items, _ in active_components}
        current = dict.fromkeys(weights, 0.0)
        total_weight = sum(weights.values())
        order = [component for component, _, _ in buckets if component in weights]
        if total_weight <= 0:
            yield from self._emit_trivial(buckets)
            return
        while weights:
            chosen_component: MixtureComponent | None = None
            chosen_value = -math.inf
            for component in order:
                if component not in weights:
                    continue
                current_value = current[component] + weights[component]
                current[component] = current_value
                if current_value > chosen_value:
                    chosen_component = component
                    chosen_value = current_value
            assert chosen_component is not None
            current[chosen_component] -= total_weight
            items = items_lookup[chosen_component]
            offset = positions[chosen_component]
            yield items[offset]
            offset += 1
            positions[chosen_component] = offset
            if offset >= len(items):
                weight = weights.pop(chosen_component)
                current.pop(chosen_component, None)
                total_weight -= weight
                items_lookup.pop(chosen_component, None)
                active_components = [
                    bucket
                    for bucket in active_components
                    if bucket[0] != chosen_component
                ]
                if total_weight <= 0:
                    for component in order:
                        if component in weights:
                            remaining = items_lookup.get(
                                component, self.components[component]
                            )
                            yield from remaining[positions[component] :]
                    break

    def _compute_mixture(self) -> MixtureWeights:
        if self.explicit_mixture is not None:
            spec = MixtureSpec(self.explicit_mixture)
            components = list(self._component_order)
            spec.validate_for(components)
            mixture = spec.normalized_for(components)
            # Ensure zero-weight components don't carry samples.
            zero_components = {
                component
                for component in components
                if mixture.get(component, 0.0) == 0.0 and self.components.get(component)
            }
            if zero_components:
                missing = ", ".join(sorted(zero_components))
                raise ValueError(
                    "Explicit mixture weight must be positive for components with samples: "
                    + missing
                )
            return {
                component: mixture[component]
                for component in components
                if component in mixture and mixture[component] > 0.0
            }  # i dont think this manual handling here is necessary if we assume that normalize already validates and fixes the mixture.
        counts = {
            component: len(self.components.get(component, ()))
            for component in self._component_order
        }
        total_count = sum(counts.values())
        if total_count == 0:
            return {}
        return {
            component: counts[component] / total_count
            for component in self._component_order
            if counts[component] > 0
        }  # cant we use the normalize logic here as well? i think we can drastically simplify this.

    @property  # cached?
    def _has_multiple_components(self) -> bool:
        active = sum(1 for items in self.components.values() if items)
        return active > 1


class WorkSource(Protocol):
    """Protocol for producing work chunks and supporting random access."""

    def next_chunk(self) -> WorkChunk | None: ...

    def checkpoint(self) -> bytes: ...

    def restore(self, state: bytes) -> None: ...

    def supports_indexing(self) -> bool: ...

    def __len__(self) -> int: ...

    def sample_id_at(self, index: int) -> SampleId: ...

    @property  # TODO(MaxiBoether): Can we use this decorator in a Protocol?
    def datasets_by_id(self) -> Mapping[int, Dataset]: ...
