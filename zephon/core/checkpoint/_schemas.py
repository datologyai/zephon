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
        :mod:`zephon.core.checkpoint._migrations` for the mutation contract.
        """
        from zephon.core.checkpoint._migrations import migrate

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

WORK_CHUNK_VERSION = 1


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


# ---------------------------------------------------------------------------
# Component: StaticMixtureWorkSource
# ---------------------------------------------------------------------------

STATIC_MIXTURE_VERSION = 1


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


# ---------------------------------------------------------------------------
# Component: Engine (top-level checkpoint)
# ---------------------------------------------------------------------------

ENGINE_VERSION = 1


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

    def __post_init__(self) -> None:
        # Accumulate every invariant violation so a malformed checkpoint
        # surfaces all issues at once instead of forcing fix-and-retry cycles.
        errors: list[str] = []

        for name in ("world", "progress", "lane_next_cid", "lane_ws_state"):
            val = getattr(self, name)
            if not isinstance(val, dict):
                errors.append(f"{name} must be a dict, got {type(val).__name__}")

        if isinstance(self.world, dict):
            if "canonical_replicas" not in self.world:
                errors.append("world missing 'canonical_replicas'")
            else:
                try:
                    int(self.world["canonical_replicas"])
                except (TypeError, ValueError):
                    errors.append(
                        f"world['canonical_replicas'] must be int-like, "
                        f"got {self.world['canonical_replicas']!r}"
                    )

        if isinstance(self.progress, dict):
            for lane_key, entry in self.progress.items():
                if not isinstance(entry, dict):
                    errors.append(
                        f"progress[{lane_key!r}] must be a dict, "
                        f"got {type(entry).__name__}"
                    )
                    continue
                for fld in ("chunk_id", "offset"):
                    if fld not in entry:
                        errors.append(
                            f"progress[{lane_key!r}] missing required field {fld!r}"
                        )

        if errors:
            raise ValueError(
                "EngineStateV1 invariants violated:\n  - " + "\n  - ".join(errors)
            )
