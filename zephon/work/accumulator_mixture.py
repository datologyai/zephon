# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Accumulator-based mixture work source with Bresenham-style quota allocation."""

import math
import warnings
from typing import Any, Mapping

from zephon.core.constants import SampleId
from zephon.io.dataset import Dataset
from zephon.work.base import WorkChunk, WorkSource
from zephon.work.mixture import MixtureSpec
from zephon.work.static_mixture import _DatasetCursor, _DatasetKnobs


class AccumulatorMixtureWorkSource(WorkSource):
    """Emit SampleId triples using Bresenham-style fractional accumulators.

    Unlike ``StaticMixtureWorkSource`` which assigns a fixed per-chunk quota to
    every dataset (guaranteeing at least 1 sample per component per chunk), this
    source allows *sparse* chunks where a dataset may contribute 0 samples.
    Over many chunks the running average converges to the exact requested mixture
    weights with no ``1/chunk_size`` granularity limitation.

    Trade-offs vs ``StaticMixtureWorkSource``:

    * **Pro:** Arbitrary mixture precision (no ``1/chunk_size`` floor).
    * **Pro:** ``chunk_size`` can be smaller than the number of components.
    * **Con:** Individual chunks may not reflect the target mixture — only the
      running total converges.
    * **Con:** An extra ``dict[str, float]`` of accumulator state must be
      checkpointed.

    The emitted identifiers have the shape ``(dataset_id, shard_id, sample_idx)``
    where ``dataset_id`` is the position of the dataset in the constructor list.
    """

    def __init__(
        self,
        datasets: list[Dataset],
        mixture: MixtureSpec | Mapping[str, float],
        chunk_size: int = 1024,
        seed: int = 0,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = False,
        shuffle_block_size: int | None = None,
        exhausted_policy: str = "stop",
        reshuffle_on_repeat: bool = True,
        max_repeats: int | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not datasets:
            raise ValueError("At least one dataset must be provided")
        super().__init__()

        self._datasets: list[Dataset] = list(datasets)
        dataset_names = [ds.name for ds in self._datasets]

        mixture_spec = MixtureSpec(mixture) if isinstance(mixture, Mapping) else mixture
        mixture_spec.validate_for(dataset_names)

        self._dataset_ids: dict[str, int] = {}
        self._datasets_by_id: dict[int, Dataset] = {}
        self._cursors: dict[str, _DatasetCursor] = {}
        self._weights: dict[str, float] = dict(mixture_spec.normalized)
        self._global_chunk_index = 0

        self._knobs = _DatasetKnobs(
            seed=seed,
            shuffle_shards=shuffle_shards,
            shuffle_within_shard=shuffle_within_shard,
            shuffle_block_size=shuffle_block_size,
        )

        for dataset_id, ds in enumerate(self._datasets):
            self._dataset_ids[ds.name] = dataset_id
            self._datasets_by_id[dataset_id] = ds
            cursor = _DatasetCursor(dataset_id, ds.shard_index, self._knobs)

            if cursor.remaining <= 0:
                self._weights.pop(ds.name)
                continue
            self._cursors[ds.name] = cursor

        if not self._weights:
            raise ValueError("No datasets with positive weight and non-empty cursors")

        self._chunk_size = chunk_size
        self._component_order: list[str] = [
            ds.name for ds in self._datasets if ds.name in self._weights
        ]
        if not self._component_order:
            raise ValueError("No active mixture components available")

        self._accumulators: dict[str, float] = dict.fromkeys(self._component_order, 0.0)

        self._seed = seed
        valid_policies = {"stop", "redistribute", "repeat"}
        if exhausted_policy not in valid_policies:
            raise ValueError(
                "exhausted_policy must be one of 'stop', 'redistribute', 'repeat'"
            )
        self._exhausted_policy = exhausted_policy
        self._reshuffle_on_repeat = reshuffle_on_repeat
        self._max_repeats = max_repeats

        if max_repeats is not None and exhausted_policy != "repeat":
            warnings.warn(
                f"max_repeats={max_repeats} has no effect with "
                f"exhausted_policy='{exhausted_policy}'",
                RuntimeWarning,
                stacklevel=2,
            )

        # Guard: with repeat policy, each dataset must have at least 1 sample
        # so that cursor.reset() can always make progress.
        if exhausted_policy == "repeat":
            for name in self._component_order:
                cursor = self._cursors[name]
                if cursor._total_samples < 1:
                    raise ValueError(
                        f"Dataset '{name}' has 0 samples but repeat policy "
                        f"requires at least 1. "
                    )

        self.total_samples: int | float = (
            float("inf") if self._exhausted_policy == "repeat" else len(self)
        )

    # ------------------------------------------------------------------
    # Cloning
    # ------------------------------------------------------------------

    def clone_for_lane(self, lane_id: int, canonical_replicas: int) -> WorkSource:
        """Lightweight clone that avoids deepcopying large cursor buffers."""
        if self._cloned:
            raise RuntimeError(
                "State Error: clone_for_lane should only be called on "
                "user-defined WorkSource instances."
            )

        clone = object.__new__(type(self))
        WorkSource.__init__(clone)

        clone._datasets = list(self._datasets)
        clone._dataset_ids = dict(self._dataset_ids)
        clone._datasets_by_id = dict(self._datasets_by_id)
        clone._weights = dict(self._weights)
        clone._component_order = list(self._component_order)
        clone._chunk_size = self._chunk_size
        clone._seed = self._seed
        clone._global_chunk_index = self._global_chunk_index
        clone._exhausted_policy = self._exhausted_policy
        clone._reshuffle_on_repeat = self._reshuffle_on_repeat
        clone._max_repeats = self._max_repeats
        clone._knobs = self._knobs

        clone._cursors = {name: cur._clone() for name, cur in self._cursors.items()}
        clone._accumulators = dict(self._accumulators)

        clone.total_samples = self.total_samples

        clone._cloned = True
        clone._bind_lane(lane_id, canonical_replicas)
        return clone

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def chunk_size_hint(self) -> int | None:
        return self._chunk_size

    @property
    def datasets_by_id(self) -> Mapping[int, Dataset]:
        """Mapping of dataset_id to its Dataset descriptor."""
        return dict(self._datasets_by_id)

    @property
    def dataset_ids(self) -> Mapping[str, int]:
        """Mapping of dataset name to the stable dataset_id used in SampleIds."""
        return dict(self._dataset_ids)

    # ------------------------------------------------------------------
    # Accumulator-based quota computation
    # ------------------------------------------------------------------

    def _compute_quotas_from_accumulators(self) -> dict[str, int]:
        """Advance accumulators and return per-component quotas for this chunk.

        Uses Bresenham-style fractional accumulation:

        1. Add ``weight * chunk_size`` to each accumulator.
        2. Take ``int(accum)`` (truncation toward zero) as the base quota;
           subtract it so the accumulator holds only the remainder.
        3. Apply a largest-remainder correction so ``sum(quotas) == chunk_size``
           exactly, and **debit/credit each correction back into the
           accumulator** so that future chunks account for the actual
           (corrected) allocation.

        Accumulators may temporarily go negative after a +1 correction; they
        recover naturally as ``weight * chunk_size`` is added each chunk.
        Negative accumulators sort last in the deficit correction, preventing
        back-to-back over-allocation.
        """
        quotas: dict[str, int] = {}
        total = 0

        for name in self._component_order:
            self._accumulators[name] += self._weights[name] * self._chunk_size
            q = int(self._accumulators[name])  # truncation toward zero
            self._accumulators[name] -= q
            quotas[name] = q
            total += q

        deficit = self._chunk_size - total

        if deficit > 0:
            # Give extra slots to components with largest fractional remainder.
            order = sorted(
                enumerate(self._component_order),
                key=lambda pair: (-self._accumulators[pair[1]], pair[0]),
            )
            for _, name in order:
                if deficit <= 0:
                    break
                quotas[name] += 1
                deficit -= 1

                # debit each correction back into the accumulator
                self._accumulators[name] -= 1.0

        elif deficit < 0:
            # Remove slots from components with smallest fractional remainder.
            order = sorted(
                enumerate(self._component_order),
                key=lambda pair: (self._accumulators[pair[1]], pair[0]),
            )
            for _, name in order:
                if deficit >= 0:
                    break
                if quotas[name] > 0:
                    quotas[name] -= 1
                    deficit += 1

                    # credit each correction back into the accumulator
                    self._accumulators[name] += 1.0

        if sum(quotas.values()) != self._chunk_size:
            raise RuntimeError(
                "Accumulator quota correction failed to hit chunk_size "
                f"(got {sum(quotas.values())}, expected {self._chunk_size})"
            )

        return quotas

    # ------------------------------------------------------------------
    # Chunk production
    # ------------------------------------------------------------------

    def next_chunk(self) -> WorkChunk | None:
        """Return the next chunk for this lane.

        Uses the same compute-everywhere-then-discard strategy as
        ``StaticMixtureWorkSource``: every lane advances all cursors and
        accumulators for every global chunk, then discards chunks that don't
        belong to this lane.
        """
        assert self._lane is not None, (
            "Please assign lane for WorkSource before requesting chunk."
        )
        assert self._canon is not None, (
            "Please assign lane for WorkSource before requesting chunk."
        )
        while True:
            chunk = self._next_chunk()
            if chunk is None:
                return None
            g = self._global_chunk_index
            self._global_chunk_index += 1
            chunk_lane = g % self._canon
            if chunk_lane != (self._lane % self._canon):
                continue
            return chunk

    def _next_chunk(self) -> WorkChunk | None:
        if self._exhausted_policy == "redistribute":
            raise NotImplementedError("Exhausted policy 'redistribute' not implemented")

        # Compute quotas, then verify cursors can fulfill them.
        # On exhaustion with repeat policy: rollback accumulators, reset the
        # exhausted cursor, and retry.
        while True:
            saved_accumulators = dict(self._accumulators)
            quotas = self._compute_quotas_from_accumulators()

            needs_retry = False
            for name in self._component_order:
                quota = quotas[name]
                cursor = self._cursors[name]
                if cursor.remaining < quota:
                    if self._exhausted_policy == "repeat":
                        if (
                            self._max_repeats is not None
                            and cursor._epoch >= self._max_repeats
                        ):
                            return None
                        cursor.reset(
                            reshuffle=self._reshuffle_on_repeat,
                            base_knobs=self._knobs,
                        )
                        self._accumulators = saved_accumulators
                        needs_retry = True
                        break
                    else:
                        return None  # "stop" policy

            if needs_retry:
                continue
            break

        components: dict[str, list[SampleId]] = {}
        for name in self._component_order:
            quota = quotas[name]
            if quota <= 0:
                continue
            cursor = self._cursors[name]
            samples = cursor.next_many(quota)
            if len(samples) != quota:
                raise RuntimeError(
                    f"Cursor for component '{name}' returned {len(samples)}"
                    f" samples, expected {quota}"
                )
            components[name] = samples

        if not components:
            return None

        if self._exhausted_policy != "repeat":
            self.total_samples = len(self)

        return WorkChunk(
            components=components,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Estimate of remaining samples across all chunks that can be yielded.

        Uses ``weight * chunk_size`` (the average per-chunk quota) as the
        divisor.  This may slightly overestimate since a component could receive
        a larger-than-average quota in the next chunk, but it is close to
        correct and consistent with accumulator convergence.  The actual
        termination is governed by ``_next_chunk``.
        """
        if hasattr(self, "_exhausted_policy"):
            if self._exhausted_policy == "redistribute":
                raise NotImplementedError(
                    "Exhausted policy 'redistribute' not implemented"
                )
            if self._exhausted_policy == "repeat":
                raise TypeError(
                    "len() is not defined for an infinite work source "
                    "(exhausted_policy='repeat')"
                )

        chunks_possible: float = math.inf
        for name in self._component_order:
            avg_quota = self._weights[name] * self._chunk_size
            if avg_quota <= 0:
                continue
            cursor = self._cursors[name]
            available = cursor.remaining / avg_quota
            chunks_possible = min(chunks_possible, available)
            if chunks_possible <= 0:
                return 0
        if chunks_possible is math.inf:
            return 0
        return int(chunks_possible) * self._chunk_size

    # ------------------------------------------------------------------
    # Indexing (not supported)
    # ------------------------------------------------------------------

    def sample_id_at(self, index: int) -> SampleId:
        raise NotImplementedError("AccumulatorMixtureWorkSource is not yet indexable")

    def supports_indexing(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Checkpoint / restore
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        base = super().state_dict()
        cursor_states = {
            name: cur.checkpoint_state() for name, cur in self._cursors.items()
        }
        return base | {
            "version": 1,
            "type": "AccumulatorMixtureWorkSource",
            "seed": int(self._seed),
            "chunk_size": int(self._chunk_size),
            "knobs": {
                "shuffle_shards": self._knobs.shuffle_shards,
                "shuffle_within_shard": self._knobs.shuffle_within_shard,
                "shuffle_block_size": self._knobs.shuffle_block_size,
            },
            "global_chunk_index": int(self._global_chunk_index),
            "weights": dict(self._weights),
            "component_order": list(self._component_order),
            "dataset_ids": dict(self._dataset_ids),
            "cursor_states": cursor_states,
            "cursor_positions": {
                name: int(cur_state["position"])
                for name, cur_state in cursor_states.items()
            },
            "cursor_epochs": {
                name: int(cur_state["epoch"])
                for name, cur_state in cursor_states.items()
            },
            "accumulators": {name: float(v) for name, v in self._accumulators.items()},
            "exhausted_policy": self._exhausted_policy,
            "reshuffle_on_repeat": self._reshuffle_on_repeat,
            "max_repeats": self._max_repeats,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)

        if int(state.get("version", 0)) != 1:
            raise RuntimeError(
                "Unsupported AccumulatorMixtureWorkSource checkpoint version"
            )

        self._seed = int(state["seed"])
        self._chunk_size = int(state["chunk_size"])
        knobs = _DatasetKnobs(
            seed=self._seed,
            shuffle_shards=bool(state["knobs"]["shuffle_shards"]),
            shuffle_within_shard=bool(state["knobs"]["shuffle_within_shard"]),
            shuffle_block_size=state["knobs"]["shuffle_block_size"],
        )
        self._knobs = knobs

        ckpt_reshuffle = state.get("reshuffle_on_repeat", self._reshuffle_on_repeat)
        ckpt_max_repeats = state.get("max_repeats", self._max_repeats)

        if ckpt_reshuffle != self._reshuffle_on_repeat:
            raise RuntimeError(
                f"Checkpoint has reshuffle_on_repeat={ckpt_reshuffle} but "
                f"the current instance was constructed with "
                f"reshuffle_on_repeat={self._reshuffle_on_repeat}. "
                f"These must match for deterministic continuation."
            )
        if ckpt_max_repeats != self._max_repeats:
            raise RuntimeError(
                f"Checkpoint has max_repeats={ckpt_max_repeats} but "
                f"the current instance was constructed with "
                f"max_repeats={self._max_repeats}. "
                f"These must match for deterministic continuation."
            )
        self._reshuffle_on_repeat = ckpt_reshuffle
        self._max_repeats = ckpt_max_repeats
        cursor_states: dict[str, Any] = state.get("cursor_states", {})
        if not cursor_states:
            raise RuntimeError(
                "AccumulatorMixtureWorkSource checkpoints require cursor_states"
            )

        # Rebuild cursors deterministically and set positions.
        self._cursors.clear()
        self._dataset_ids.clear()
        self._datasets_by_id.clear()

        for dataset_id, ds in enumerate(self._datasets):
            self._dataset_ids[ds.name] = dataset_id
            self._datasets_by_id[dataset_id] = ds
            cur = _DatasetCursor(dataset_id, ds.shard_index, knobs)
            if ds.name not in cursor_states:
                raise RuntimeError(
                    f"Checkpoint missing cursor_states entry for dataset '{ds.name}'"
                )
            cur.restore_checkpoint_state(
                cursor_states[ds.name],
                reshuffle=self._reshuffle_on_repeat,
                base_knobs=knobs,
            )
            self._cursors[ds.name] = cur

        self._weights = dict(state["weights"])
        self._component_order = list(state["component_order"])
        self._global_chunk_index = int(state["global_chunk_index"])

        # Restore accumulators.
        if "accumulators" not in state:
            raise RuntimeError(
                "AccumulatorMixtureWorkSource checkpoints require accumulators"
            )
        self._accumulators = {
            str(k): float(v) for k, v in state["accumulators"].items()
        }

        self.total_samples = (
            float("inf") if self._exhausted_policy == "repeat" else len(self)
        )
