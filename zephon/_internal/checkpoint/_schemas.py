r"""Typed dataclass schemas for Zephon checkpoint state.

Each versioned checkpoint component (Engine, WorkChunk, StaticMixtureWorkSource,
_DatasetCursor) has a corresponding dataclass that defines the canonical shape of
its serialised state. The dataclass is used on **both** the write path
(``state_dict()`` builds an instance, ``instance.to_dict()`` serialises it) and
the read path (``Schema.load()``/``Schema.from_dict()`` reconstructs from a raw dict).

Validation model — each schema is its own version's validator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A v\\_N schema captures the union of all on-disk shapes ever written with
version=N via two field categories:

* **Required (no default)** — fields that EVERY historical v\\_N checkpoint has
  written. Absence is corruption, not "old format". Missing keys raise
  ``TypeError`` at construction.
* **Defaulted** — fields that may legitimately be absent from older v\\_N
  checkpoints (added later within the same version). Defaults document the
  historical absence.

Semantic invariants (e.g. nested dict shape, value ranges) live in
``__post_init__`` so they run after every construction — including after a
v\\_{N-1} → v\\_N migration. Validation = schema construction + __post_init__,
no separate validator function.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import re
import types
import typing
from dataclasses import dataclass, field
from typing import Any, ClassVar, Union, cast, get_args, get_origin

from typing_extensions import Self  # typing.Self is 3.11+; project supports 3.10

# ---------------------------------------------------------------------------
# CheckpointMixin — generic from_dict() via field introspection
# ---------------------------------------------------------------------------


def _is_optional(annotation: Any) -> bool:
    """True if the annotation is ``X | None`` / ``Optional[X]`` / ``Union[..., None]``."""
    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, types.UnionType):
        return type(None) in get_args(annotation)
    return annotation is type(None)


def _auto_coerce(annotation: Any) -> type | None:
    """Return a coercion callable for primitive type annotations, else None."""
    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            annotation = args[0]
    # bool before int: bool is a subclass of int, so check it first.
    if annotation is bool:
        return bool
    if annotation is int:
        return int
    if annotation is float:
        return float
    if annotation is str:
        return str
    return None


class CheckpointMixin:
    """Mixin providing ``load()`` and ``from_dict()`` for checkpoint dataclasses.

    ``from_dict`` iterates over ``dataclasses.fields()``, takes values from the
    raw dict when present, applies type coercion (auto-detected from the
    annotation for primitives, or overridden via ``_COERCE``), and lets missing
    keys fall through to the field defaults declared on the dataclass.
    Fields *without* a default raise ``TypeError`` when absent — that is the
    structural validation.

    Subclasses must set ``_COMPONENT`` to the migration registry key
    (e.g. ``"engine"``, ``"cursor"``); this is enforced at class-creation
    time so a missing value fails immediately, not at ``load()``.
    """

    #: Migration registry key — must be set by each schema subclass.
    _COMPONENT: ClassVar[str] = ""

    #: Escape hatch for fields that need custom coercion beyond the automatic
    #: primitive handling. Map *field name* -> *callable(value) -> coerced*.
    _COERCE: ClassVar[dict[str, Any]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "_COMPONENT", ""):
            raise TypeError(
                f"{cls.__name__} must set _COMPONENT to the migration registry "
                f"key (e.g. 'engine', 'cursor'); CheckpointMixin subclasses "
                f"cannot rely on the empty default."
            )

    @classmethod
    def load(cls, raw: dict[str, Any]) -> Self:
        """Canonical read-path entry point: migrate + structure into a typed instance.

        The migration framework validates each intermediate version by
        constructing its schema, so a v_{N-1} → v_N migration only ever
        receives a clean v_{N-1} instance. The final structuring step is
        ``from_dict`` on the current-version schema (this class).

        ``load()`` does not mutate *raw*: ``migrate()`` takes a shallow copy
        of the top level. Nested mutable values are still shared — see
        :mod:`zephon._internal.checkpoint._migrations` for the mutation contract.
        """
        from zephon._internal.checkpoint._migrations import migrate

        migrated = migrate(cls._COMPONENT, raw)
        return cls.from_dict(migrated)

    def to_dict(self, *, strip_none: bool = False) -> dict[str, Any]:
        """Shallow dataclass-to-dict conversion (mirror of ``from_dict``).

        Unlike ``dataclasses.asdict()`` nested containers are referenced, not
        deep-copied; the write path relies on this. Pass ``strip_none=True``
        to omit keys whose value is ``None`` (e.g. legacy back-compat where an
        older reader rejects an unknown ``null`` field).
        """
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(cast(Any, self))}
        if strip_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d

    @classmethod
    @functools.cache
    def _field_specs(cls) -> tuple[tuple[str, bool, Any, Any], ...]:
        """Cached per-field metadata for ``from_dict``.

        Returns a tuple of ``(name, required, hint, coerce_fn)``. Cached because
        ``typing.get_type_hints`` resolves forward references on every call —
        on the WorkChunk hot path that's once per inflight chunk.
        """
        hints = typing.get_type_hints(cls)
        # cls is always a @dataclass subclass; the mixin itself is never instantiated.
        specs: list[tuple[str, bool, Any, Any]] = []
        for f in dataclasses.fields(cast(Any, cls)):
            hint = hints.get(f.name)
            coerce_fn = cls._COERCE.get(f.name) or _auto_coerce(hint)
            required = (
                f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING
            )
            specs.append((f.name, required, hint, coerce_fn))
        return tuple(specs)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        """Construct an instance from a raw dict (already at the current version).

        Collects all structural errors (missing required keys, ``None`` for
        non-Optional fields, coercion failures) and raises a single
        ``ValueError`` listing them. Semantic invariants live in
        ``__post_init__`` and likewise raise once with every failure.
        """
        kwargs: dict[str, Any] = {}
        errors: list[str] = []
        for name, required, hint, coerce_fn in cls._field_specs():
            if name not in raw:
                if required:
                    errors.append(f"{name}: missing (required, no default)")
                continue
            val = raw[name]
            if val is None and not _is_optional(hint):
                errors.append(f"{name}: must not be None (annotation {hint!r})")
                continue
            if coerce_fn is not None and val is not None:
                try:
                    val = coerce_fn(val)
                except (TypeError, ValueError) as exc:
                    errors.append(
                        f"{name}: coercion to {coerce_fn.__name__} "
                        f"failed for {val!r} ({exc})"
                    )
                    continue
            kwargs[name] = val

        if errors:
            raise ValueError(
                f"{cls.__name__} cannot be constructed from raw dict:\n  - "
                + "\n  - ".join(errors)
            )

        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Component: _DatasetCursor
# ---------------------------------------------------------------------------

CURSOR_VERSION = 1


@dataclass(frozen=True)
class CursorStateV1(CheckpointMixin):
    """Checkpoint schema for ``_DatasetCursor`` (version 1).

    The ``block_rng_snapshot`` sub-dict, when present, has the shape
    ``{"rng_state": <random.Random state>, "block_count": int}``.
    """

    _COMPONENT: ClassVar[str] = "cursor"
    version: int = 1
    position: int = 0
    epoch: int = 0
    block_rng_snapshot: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Component: WorkChunk
# ---------------------------------------------------------------------------

WORK_CHUNK_VERSION = 3


@dataclass(frozen=True)
class WorkChunkStateV1(CheckpointMixin):
    """Checkpoint schema for ``WorkChunk.state_dict()`` (version 1).

    ``components`` is an ordered list of ``(component_name, [[dataset_id,
    shard_id, sample_idx], ...])``. Insertion order encodes component
    precedence.
    """

    _COMPONENT: ClassVar[str] = "work_chunk"
    # msgpack decodes arrays as lists; restore tuple form so the annotation holds.
    _COERCE: ClassVar[dict[str, Any]] = {
        "components": lambda v: [(e[0], e[1]) for e in v],
    }
    # Always written by every historical v1 producer.
    components: list[tuple[str, list[list[int]]]]
    component_order: list[str]
    # Defaulted: tolerant of historical absence.
    version: int = 1
    seed: int | None = None
    # Sanity field; None when absent on disk so callers can skip the check.
    # WorkChunk.from_state verifies this against the recomputed total.
    total_samples: int | None = None


@dataclass(frozen=True, kw_only=True)
class WorkChunkStateV2(WorkChunkStateV1):
    """Checkpoint schema for ``WorkChunk.state_dict()`` (version 2).

    v2 adds ``target_mixture`` — the worksource-declared per-component target
    delivered to ``ensure_mixture`` (token-aware mode stamps the user's token
    mixture here; the counted sample composition deliberately differs from it).
    Required but nullable: sample-mode chunks write it as ``None`` and the
    v1->v2 migration fills the same for pre-existing checkpoints.
    """

    version: int = 2
    target_mixture: dict[str, float] | None

    def __post_init__(self) -> None:
        if self.target_mixture is None:
            return
        errors: list[str] = []
        if not isinstance(self.target_mixture, dict):
            errors.append(
                f"target_mixture must be a dict, got "
                f"{type(self.target_mixture).__name__}"
            )
        elif not self.target_mixture:
            errors.append(
                "target_mixture must be non-empty when set (use None instead)"
            )
        else:
            for name, weight in self.target_mixture.items():
                if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                    errors.append(
                        f"target_mixture[{name!r}] must be a number, "
                        f"got {type(weight).__name__}"
                    )
                elif weight <= 0:
                    errors.append(
                        f"target_mixture[{name!r}]={weight!r} must be positive"
                    )
        if errors:
            raise ValueError(
                "WorkChunkStateV2 invariants violated:\n  - " + "\n  - ".join(errors)
            )


@dataclass(frozen=True, kw_only=True)
class WorkChunkStateV3(WorkChunkStateV2):
    """Add replayable per-component source-exhaustion announcements."""

    version: int = 3
    source_exhausted: list[str]

    def __post_init__(self) -> None:
        super().__post_init__()
        names = self.source_exhausted
        if not isinstance(names, list) or any(
            not isinstance(n, str) or not n for n in names
        ):
            raise ValueError(
                "source_exhausted must be a list of non-empty component names"
            )
        if len(set(names)) != len(names):
            raise ValueError("source_exhausted must not contain duplicate components")


# ---------------------------------------------------------------------------
# Component: StaticMixtureWorkSource
# ---------------------------------------------------------------------------

STATIC_MIXTURE_VERSION = 5


@dataclass(frozen=True)
class StaticMixtureStateV1(CheckpointMixin):
    """Checkpoint schema for ``StaticMixtureWorkSource`` (version 1).

    .. note:: **v1 is a union of all shapes ever written with version=1.**

       The version field has been ``1`` since inception, but fields were added
       over time without bumping it. The schema reflects this:

       * **Required** fields are those every historical v1 checkpoint wrote.
       * **Defaulted** fields are those that may be absent in older v1
         checkpoints; the default documents the historical absence.

       When v2 is introduced, the v1→v2 migration can normalise the union
       (e.g. populate ``cursor_states`` from ``cursor_positions`` +
       ``cursor_epochs``, make previously-defaulted fields required, or drop
       fields that are no longer written).

    Includes the base ``WorkSource`` fields (``lane_id``,
    ``canonical_replicas``, ``chunk_size_hint``) which are inherited from
    ``WorkSource.state_dict()``.

    ``knobs["shuffle_block_size"]`` is always a concrete ``int | None`` in v1
    and applies uniformly to every cursor.
    """

    _COMPONENT: ClassVar[str] = "static_mixture"
    # Always written by every historical v1 producer.
    lane_id: int
    canonical_replicas: int
    chunk_size_hint: int | None
    seed: int
    chunk_size: int
    knobs: dict[str, Any]
    global_chunk_index: int
    weights: dict[str, float]
    component_order: list[str]
    dataset_ids: dict[str, int]
    cursor_positions: dict[str, int]
    cursor_epochs: dict[str, int]
    # Defaulted: absent in older v1 checkpoints. Each default documents the
    # historical absence; see the union-of-shapes note above.
    version: int = 1
    # exhausted_policy was added by PR #186 (2026-03-08). Pre-#186 v1
    # checkpoints lack the field; "stop" is the original behaviour.
    exhausted_policy: str = "stop"
    cursor_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    reshuffle_on_repeat: bool = False
    max_repeats: int | None = None
    # Absent in legacy_fixed mode; absent in pre-accumulator-era v1.
    accumulators: dict[str, float] | None = None
    # Absent in pre-AllocationStrategy v1 checkpoints; loaders fall back to
    # inferring from "accumulators" key presence (see load_state_dict).
    allocation_mode: str | None = None


_VALID_ALLOCATION_MODES = frozenset({"accumulator", "legacy_fixed"})
#: v5 adds "token_aware"; earlier schemas reject it as corrupt.
_VALID_ALLOCATION_MODES_V5 = _VALID_ALLOCATION_MODES | {"token_aware"}
_REQUIRED_V2_KNOB_KEYS = frozenset({"shuffle_shards", "shuffle_within_shard"})

#: Provenance tags a primed tokens/byte ratio may carry in v5 checkpoints.
VALID_TOKEN_RATIO_SOURCES = frozenset({"measured", "pinned", "fallback"})


def _static_mixture_block_invariant_errors(
    *,
    shuffle_block_size_spec: int | str | None,
    knobs: dict[str, Any],
    allocation_mode: str,
    cursor_block_sizes: dict[str, int | None],
    dataset_ids: dict[str, int],
    valid_allocation_modes: frozenset[str] = _VALID_ALLOCATION_MODES,
) -> list[str]:
    """Structural invariants shared by ``StaticMixtureStateV2`` and ``V3``+.

    These fields are identical across v2 (scalar policies) and v3 (per-dataset
    policies); only the exhaustion-policy knobs differ. Returns a list of error
    strings (empty == valid) so each caller raises with its own version tag.
    ``valid_allocation_modes`` lets a later schema widen the accepted set (v5
    adds ``token_aware``) while sharing the rest of the checks.
    """
    errors: list[str] = []

    # shuffle_block_size_spec — int (not bool), one of two sentinel strings,
    # or None. The annotation alone is too loose; reject other types here.
    spec = shuffle_block_size_spec
    if isinstance(spec, bool):
        errors.append(
            f"shuffle_block_size_spec must be int, 'auto', 'global', or None; "
            f"got bool ({spec!r})"
        )
    elif isinstance(spec, str):
        if spec not in ("auto", "global"):
            errors.append(
                f"shuffle_block_size_spec={spec!r} is not a valid sentinel "
                f"(expected 'auto', 'global', int, or None)"
            )
    elif spec is not None and not isinstance(spec, int):
        errors.append(
            f"shuffle_block_size_spec must be int, 'auto', 'global', or None; "
            f"got {type(spec).__name__} ({spec!r})"
        )

    # knobs must carry the orthogonal shuffle toggles and must NOT carry
    # the stale shuffle_block_size key — that field moved out in v2.
    if not isinstance(knobs, dict):
        errors.append(f"knobs must be a dict, got {type(knobs).__name__}")
    else:
        missing_knobs = _REQUIRED_V2_KNOB_KEYS - set(knobs)
        if missing_knobs:
            errors.append(f"knobs missing required keys {sorted(missing_knobs)}")
        if "shuffle_block_size" in knobs:
            errors.append(
                "knobs must not carry 'shuffle_block_size' in v2 — it lives "
                "in cursor_block_sizes/shuffle_block_size_spec now"
            )

    if allocation_mode not in valid_allocation_modes:
        errors.append(
            f"allocation_mode={allocation_mode!r} is not one of "
            f"{sorted(valid_allocation_modes)}"
        )

    if not isinstance(cursor_block_sizes, dict):
        errors.append(
            f"cursor_block_sizes must be a dict, got {type(cursor_block_sizes).__name__}"
        )
    else:
        for name, value in cursor_block_sizes.items():
            # bool first — it is an int subclass.
            if isinstance(value, bool):
                errors.append(
                    f"cursor_block_sizes[{name!r}] must be int or None, got bool"
                )
            elif value is not None and not isinstance(value, int):
                errors.append(
                    f"cursor_block_sizes[{name!r}] must be int or None, "
                    f"got {type(value).__name__}"
                )
        if isinstance(dataset_ids, dict):
            missing = set(dataset_ids) - set(cursor_block_sizes)
            extra = set(cursor_block_sizes) - set(dataset_ids)
            if missing:
                errors.append(
                    f"cursor_block_sizes missing entries for {sorted(missing)} "
                    f"(must match dataset_ids keys)"
                )
            if extra:
                errors.append(
                    f"cursor_block_sizes has unknown entries {sorted(extra)} "
                    f"(must match dataset_ids keys)"
                )
    return errors


@dataclass(frozen=True, kw_only=True)
class StaticMixtureStateV2(CheckpointMixin):
    """Checkpoint schema for ``StaticMixtureWorkSource`` (version 2).

    The resolved per-dataset block size lives in ``cursor_block_sizes`` (always
    ``int | None``, never a sentinel) and is the source of truth on restore;
    ``shuffle_block_size_spec`` records the user-given spec verbatim (``int |
    None`` or the ``"auto"`` / ``"global"`` sentinels). ``knobs`` no longer
    carries ``shuffle_block_size`` — it was promoted to its own field, and
    ``__post_init__`` rejects checkpoints that still carry the stale key.
    """

    _COMPONENT: ClassVar[str] = "static_mixture"
    lane_id: int
    canonical_replicas: int
    chunk_size_hint: int | None
    seed: int
    chunk_size: int
    knobs: dict[str, Any]
    global_chunk_index: int
    weights: dict[str, float]
    component_order: list[str]
    dataset_ids: dict[str, int]
    cursor_positions: dict[str, int]
    cursor_epochs: dict[str, int]
    cursor_states: dict[str, dict[str, Any]]
    cursor_block_sizes: dict[str, int | None]
    shuffle_block_size_spec: int | str | None
    exhausted_policy: str
    reshuffle_on_repeat: bool
    max_repeats: int | None
    allocation_mode: str
    accumulators: dict[str, float] | None
    version: int = 2

    def __post_init__(self) -> None:
        errors = _static_mixture_block_invariant_errors(
            shuffle_block_size_spec=self.shuffle_block_size_spec,
            knobs=self.knobs,
            allocation_mode=self.allocation_mode,
            cursor_block_sizes=self.cursor_block_sizes,
            dataset_ids=self.dataset_ids,
        )
        if errors:
            raise ValueError(
                "StaticMixtureStateV2 invariants violated:\n  - "
                + "\n  - ".join(errors)
            )


def _static_mixture_policy_invariant_errors(
    *,
    component_order: list[str],
    exhausted_policy: dict[str, str],
    reshuffle_on_repeat: dict[str, bool],
    max_repeats: dict[str, int | None],
) -> list[str]:
    """Per-dataset exhaustion-policy invariants for v3.

    Each knob must be a dict keyed by exactly ``component_order``, and each
    policy value must be a recognised policy.
    """
    errors: list[str] = []
    expected = set(component_order)
    for fname, value in (
        ("exhausted_policy", exhausted_policy),
        ("reshuffle_on_repeat", reshuffle_on_repeat),
        ("max_repeats", max_repeats),
    ):
        if not isinstance(value, dict):
            errors.append(
                f"{fname} must be a per-dataset dict, got {type(value).__name__}"
            )
        elif set(value) != expected:
            errors.append(
                f"{fname} keys {sorted(value)} must match component_order "
                f"{sorted(expected)}"
            )

    valid_policies = {"stop", "redistribute", "repeat"}
    if isinstance(exhausted_policy, dict):
        for name, policy in exhausted_policy.items():
            if policy not in valid_policies:
                errors.append(
                    f"exhausted_policy[{name!r}]={policy!r} must be one of "
                    f"{sorted(valid_policies)}"
                )
    return errors


@dataclass(frozen=True, kw_only=True)
class StaticMixtureStateV3(CheckpointMixin):
    """Checkpoint schema for ``StaticMixtureWorkSource`` (version 3).

    The three exhaustion-policy knobs (``exhausted_policy`` /
    ``reshuffle_on_repeat`` / ``max_repeats``) are per-dataset dicts keyed by
    dataset name, letting datasets in one mixture use different policies.
    ``stop_after_passes`` is the global termination floor (``None`` == no floor).

    A standalone class, not a ``StaticMixtureStateV2`` subclass: re-typing those
    knobs from v2's scalars cannot be expressed as a field override.
    """

    _COMPONENT: ClassVar[str] = "static_mixture"
    lane_id: int
    canonical_replicas: int
    chunk_size_hint: int | None
    seed: int
    chunk_size: int
    knobs: dict[str, Any]
    global_chunk_index: int
    weights: dict[str, float]
    component_order: list[str]
    dataset_ids: dict[str, int]
    cursor_positions: dict[str, int]
    cursor_epochs: dict[str, int]
    cursor_states: dict[str, dict[str, Any]]
    cursor_block_sizes: dict[str, int | None]
    shuffle_block_size_spec: int | str | None
    exhausted_policy: dict[str, str]
    reshuffle_on_repeat: dict[str, bool]
    max_repeats: dict[str, int | None]
    stop_after_passes: int | None
    allocation_mode: str
    accumulators: dict[str, float] | None
    version: int = 3

    # Allocation modes this schema accepts; v5 overrides to add "token_aware".
    _ALLOCATION_MODES: ClassVar[frozenset[str]] = _VALID_ALLOCATION_MODES

    def __post_init__(self) -> None:
        errors = _static_mixture_block_invariant_errors(
            shuffle_block_size_spec=self.shuffle_block_size_spec,
            knobs=self.knobs,
            allocation_mode=self.allocation_mode,
            cursor_block_sizes=self.cursor_block_sizes,
            dataset_ids=self.dataset_ids,
            valid_allocation_modes=self._ALLOCATION_MODES,
        )
        errors.extend(
            _static_mixture_policy_invariant_errors(
                component_order=self.component_order,
                exhausted_policy=self.exhausted_policy,
                reshuffle_on_repeat=self.reshuffle_on_repeat,
                max_repeats=self.max_repeats,
            )
        )
        # Mirror the constructor's stop_after_passes interaction rules: load
        # paths adopt these fields directly (no constructor), so the schema is
        # the only gate against a config the public API would reject.
        if self.stop_after_passes is not None:
            if self.stop_after_passes < 1:
                errors.append(
                    f"stop_after_passes must be a positive int or None, got "
                    f"{self.stop_after_passes!r}"
                )
            if isinstance(self.exhausted_policy, dict):
                non_repeat = sorted(
                    ds for ds, pol in self.exhausted_policy.items() if pol != "repeat"
                )
                if non_repeat:
                    errors.append(
                        f"stop_after_passes={self.stop_after_passes} requires every "
                        f"dataset to repeat, but {non_repeat} have a non-'repeat' "
                        f"exhausted_policy"
                    )
            if isinstance(self.max_repeats, dict):
                capped = sorted(
                    ds for ds, mr in self.max_repeats.items() if mr is not None
                )
                if capped:
                    errors.append(
                        f"stop_after_passes={self.stop_after_passes} cannot be "
                        f"combined with max_repeats, but {capped} are capped"
                    )
        if errors:
            raise ValueError(
                "StaticMixtureStateV3 invariants violated:\n  - "
                + "\n  - ".join(errors)
            )


@dataclass(frozen=True, kw_only=True)
class StaticMixtureStateV4(StaticMixtureStateV3):
    """Checkpoint schema for ``StaticMixtureWorkSource`` (version 4).

    Adds ``lane_assignment``, the chunk->lane routing mode: ``"permute"`` is the
    per-block seeded permutation that breaks cadence-sharding resonance (see
    ``StaticMixtureWorkSource._lane_for_chunk``); ``"modulo"`` is plain
    ``g % canonical_replicas``. An additive change, so it extends v3 by
    inheritance. Required, no default — the v3 -> v4 migration fills pre-fix
    checkpoints with ``"modulo"`` so a resumed run replays the routing it used.
    """

    lane_assignment: str
    version: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lane_assignment not in ("modulo", "permute"):
            raise ValueError(
                "StaticMixtureStateV4 invariants violated:\n  - "
                f"lane_assignment={self.lane_assignment!r} must be "
                "'modulo' or 'permute'"
            )


@dataclass(frozen=True, kw_only=True)
class StaticMixtureStateV5(StaticMixtureStateV4):
    """StaticMixture checkpoint schema with token-aware allocation fields."""

    # Widen the inherited mode set for the shared invariant check.
    _ALLOCATION_MODES: ClassVar[frozenset[str]] = _VALID_ALLOCATION_MODES_V5
    token_deficits: dict[str, float] | None
    token_ratios: dict[str, Any] | None
    version: int = 5

    def __post_init__(self) -> None:
        super().__post_init__()
        errors: list[str] = []

        token_aware = self.allocation_mode == "token_aware"
        if token_aware:
            if self.accumulators is not None:
                errors.append(
                    "accumulators must be None in token_aware mode, got "
                    f"{type(self.accumulators).__name__}"
                )
            if not isinstance(self.token_deficits, dict):
                errors.append(
                    "token_deficits must be a per-component dict in token_aware "
                    f"mode, got {type(self.token_deficits).__name__}"
                )
            elif set(self.token_deficits) != set(self.component_order):
                errors.append(
                    f"token_deficits keys {sorted(self.token_deficits)} must "
                    f"match component_order {sorted(self.component_order)}"
                )
            else:
                for name, deficit in self.token_deficits.items():
                    if (
                        isinstance(deficit, bool)
                        or not isinstance(deficit, (int, float))
                        or not math.isfinite(deficit)
                    ):
                        errors.append(
                            f"token_deficits[{name!r}]={deficit!r} must be "
                            "a finite number"
                        )
            # token_ratios=None means unprimed; token_deficits stays required.
            if self.token_ratios is None:
                pass
            elif not isinstance(self.token_ratios, dict):
                errors.append(
                    "token_ratios must be a per-dataset dict or None in "
                    f"token_aware mode, got {type(self.token_ratios).__name__}"
                )
            elif set(self.token_ratios) != set(self.component_order):
                errors.append(
                    f"token_ratios keys {sorted(self.token_ratios)} must "
                    f"match component_order {sorted(self.component_order)}"
                )
            else:
                for name, entry in self.token_ratios.items():
                    if (
                        not isinstance(entry, (list, tuple))
                        or len(entry) != 2
                        or isinstance(entry[0], bool)
                        or not isinstance(entry[0], (int, float))
                        or not math.isfinite(entry[0])
                        or entry[0] <= 0
                        or entry[1] not in VALID_TOKEN_RATIO_SOURCES
                    ):
                        errors.append(
                            f"token_ratios[{name!r}]={entry!r} must be "
                            f"[positive ratio, source in "
                            f"{sorted(VALID_TOKEN_RATIO_SOURCES)}]"
                        )
        else:
            if self.token_deficits is not None:
                errors.append(
                    "token_deficits must be None outside token_aware mode, got "
                    f"{type(self.token_deficits).__name__}"
                )
            if self.token_ratios is not None:
                errors.append(
                    "token_ratios must be None outside token_aware mode, got "
                    f"{type(self.token_ratios).__name__}"
                )

        if errors:
            raise ValueError(
                "StaticMixtureStateV5 invariants violated:\n  - "
                + "\n  - ".join(errors)
            )


# ---------------------------------------------------------------------------
# Component: Engine (top-level checkpoint)
# ---------------------------------------------------------------------------

ENGINE_VERSION = 2


@dataclass(frozen=True)
class EngineStateV1(CheckpointMixin):
    """Checkpoint schema for the top-level engine state (version 1).

    .. note:: **v1 is a union of all shapes ever written with version=1.**

       The version field has been ``1`` since the engine's first release, but
       fields were added incrementally. The schema reflects this:

       * **Required** fields are those every historical v1 checkpoint wrote.
       * **Defaulted** fields may be absent in older v1 checkpoints; the
         default documents the historical absence.

       Semantic invariants (canonical_replicas int-castable, progress entries
       have chunk_id/offset, etc.) are enforced in ``__post_init__`` so they
       run on every construction — including after a future v1→v2 migration.
    """

    _COMPONENT: ClassVar[str] = "engine"
    # Always written by every historical v1 producer.
    world: dict[str, Any]
    progress: dict[str, Any]
    lane_next_cid: dict[str, Any]
    lane_ws_state: dict[str, Any]
    last_round_id: str | None
    checkpoint_reload_count: int
    # Defaulted: may be absent in older v1 checkpoints.
    version: int = 1
    inflight: dict[str, Any] = field(default_factory=dict)
    # work_source is written for debugging and never read back.
    work_source: dict[str, Any] | None = None
    work_config: dict[str, Any] | None = None
    rr_next_idx: dict[str, int] = field(default_factory=dict)
    replay_cursors: dict[str, Any] = field(default_factory=dict)
    epoch_boundaries: dict[str, Any] = field(default_factory=dict)
    lane_emitted: dict[str, Any] = field(default_factory=dict)  # empty = unknown

    def __post_init__(self) -> None:
        errors = _engine_state_errors(self)
        if errors:
            raise ValueError(
                "EngineStateV1 invariants violated:\n  - " + "\n  - ".join(errors)
            )


def _engine_state_errors(state: EngineStateV1 | EngineStateV2) -> list[str]:
    """Invariants shared by every engine schema version."""
    # Accumulate every invariant violation so a malformed checkpoint
    # surfaces all issues at once instead of forcing fix-and-retry cycles.
    errors: list[str] = []

    for name in ("world", "progress", "lane_next_cid", "lane_ws_state"):
        val = getattr(state, name)
        if not isinstance(val, dict):
            errors.append(f"{name} must be a dict, got {type(val).__name__}")

    # Validate before v1 migration can coerce values or collapse peer keys.
    if not isinstance(state.rr_next_idx, dict):
        errors.append(
            f"rr_next_idx must be a dict, got {type(state.rr_next_idx).__name__}"
        )
    else:
        for key, idx in state.rr_next_idx.items():
            if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
                errors.append(
                    f"rr_next_idx[{key!r}] must be a non-negative integer "
                    f"(not bool), got {idx!r}"
                )

    if isinstance(state.world, dict):
        if "canonical_replicas" not in state.world:
            errors.append("world missing 'canonical_replicas'")
        else:
            try:
                int(state.world["canonical_replicas"])
            except (TypeError, ValueError):
                errors.append(
                    f"world['canonical_replicas'] must be int-like, "
                    f"got {state.world['canonical_replicas']!r}"
                )

    if isinstance(state.progress, dict):
        for lane_key, entry in state.progress.items():
            if not isinstance(entry, dict):
                errors.append(
                    f"progress[{lane_key!r}] must be a dict, got {type(entry).__name__}"
                )
                continue
            for fld in ("chunk_id", "offset"):
                if fld not in entry:
                    errors.append(
                        f"progress[{lane_key!r}] missing required field {fld!r}"
                    )

    return errors


# "{worker}/{active_workers}:{comma-separated owned lanes}". v1 prefixed this
# with "{global_rank}:", which DP peers could not share.
_RR_KEY = re.compile(r"\d+/\d+:(\d+(,\d+)*)?")


@dataclass(frozen=True, kw_only=True)
class EngineStateV2(CheckpointMixin):
    """Checkpoint schema for the top-level engine state (version 2).

    ``rr_next_idx`` is keyed by lane set (see ``_RR_KEY``) instead of by
    global rank, so every TP/PP peer of a DP group restores the same tail
    round-robin pointer. A standalone class, not an ``EngineStateV1``
    subclass: v2 drops v1's historical field defaults.
    """

    _COMPONENT: ClassVar[str] = "engine"
    world: dict[str, Any]
    progress: dict[str, Any]
    lane_next_cid: dict[str, Any]
    lane_ws_state: dict[str, Any]
    last_round_id: str | None
    checkpoint_reload_count: int
    inflight: dict[str, Any]
    # work_source is written for debugging and never read back.
    work_source: dict[str, Any] | None
    work_config: dict[str, Any] | None
    rr_next_idx: dict[str, int]
    replay_cursors: dict[str, Any]
    epoch_boundaries: dict[str, Any]
    lane_emitted: dict[str, Any]  # empty = unknown
    version: int = 2

    def __post_init__(self) -> None:
        errors = _engine_state_errors(self)
        if isinstance(self.rr_next_idx, dict):
            for key, idx in self.rr_next_idx.items():
                match = _RR_KEY.fullmatch(key) if isinstance(key, str) else None
                if match is None:
                    errors.append(
                        f"rr_next_idx key {key!r} is not '{{worker}}/{{active}}:{{lanes}}'"
                    )
                    continue
                lanes = match.group(1)
                lane_count = len(lanes.split(",")) if lanes else 0
                # Idle workers may retain an empty-lane entry, always at zero.
                if isinstance(idx, int) and idx >= max(1, lane_count):
                    errors.append(
                        f"rr_next_idx[{key!r}]={idx} is out of range for "
                        f"{lane_count} owned lanes (expected "
                        f"0 <= index < {max(1, lane_count)})"
                    )
        if errors:
            raise ValueError(
                "EngineStateV2 invariants violated:\n  - " + "\n  - ".join(errors)
            )
