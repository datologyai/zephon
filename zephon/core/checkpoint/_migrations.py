"""Migration registry for versioned checkpoint components.

Each component has a chain of migration functions that upgrade a raw dict
from version N to version N+1.

Validation model
~~~~~~~~~~~~~~~~

The schema *of each version* is that version's validator. Before each
migration step, :func:`migrate` constructs the v_N schema from the raw dict
(``from_dict`` + ``__post_init__``); a malformed v_N raises immediately and
the migration never runs against garbage.

Migration signature
~~~~~~~~~~~~~~~~~~~

Migration functions take the *typed* v_N instance and return a v_{N+1} dict::

    def _engine_v1_to_v2(v1: EngineStateV1) -> dict[str, Any]:
        d = v1.to_dict()
        d["lane_state"] = d.pop("lane_ws_state")
        return d

The returned dict is validated by constructing the v_{N+1} schema on the
next loop iteration (or by the caller's ``Schema.load`` at the chain's end).

Nested untyped subtrees
~~~~~~~~~~~~~~~~~~~~~~~

Top-level fields are typed, but several values are intentionally opaque
(``EngineStateV1.world``, ``progress[lane]``, ``inflight[lane][cid]``,
``lane_ws_state[lane]`` etc. are all ``dict[str, Any]``). The migration
registry has no visibility into renames *inside* those dicts; a v_N → v_{N+1}
migration that renames a key inside ``world`` must walk into the dict
explicitly and reassign it. See the mutation contract below for ownership
when doing so.

Schema evolution: adding or changing a field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A version's class is the complete, self-contained validator for that version's
on-disk shape. Two rules keep the contract sharp and adding a field cheap.

**No field defaults except ``version``.** A static default would let a new
field skip its migration — ``from_dict`` fills the absent key — but we
deliberately do not take that shortcut: it makes a version a *union* of shapes,
where a missing key reads as the default instead of as corruption (the drift
``StaticMixtureStateV1`` accumulated and v2 undid). Add new fields as
**required** and fill them in the migration into that version; a nullable field
is still required (present, value may be ``None``). Only the legacy v1 union
keeps its historical defaults.

**Bump and migrate, cheaply via inheritance.** Schemas are
``@dataclass(frozen=True, kw_only=True)`` and each version inherits the
previous, so an additive change is a few lines, not a field-list copy::

    @dataclass(frozen=True, kw_only=True)
    class StaticMixtureStateV4(StaticMixtureStateV3):
        new_field: SomeType   # required; kw_only lets it follow the version default
        version: int = 4

Pair it with a ``_..._v3_to_v4`` migration that fills ``new_field``. ``kw_only``
is load-bearing: without it a required field after the inherited ``version``
default raises "non-default argument follows default argument".

**Copy instead of inherit when you cannot extend additively.** A field cannot
be re-typed in a subclass (an incompatible override under static type checking)
and inheritance cannot drop one, so **re-typing** (v2 scalars -> v3 per-dataset
dicts) or **removing** a field needs a standalone schema copy.

**Migration required whenever** existing dicts cannot pass
``CurrentSchema.from_dict()`` (including ``__post_init__``): adding a required
field, renaming (``from_dict`` drops the unknown old key and defaults the new
one — silently losing data), removing, re-typing beyond ``_auto_coerce``,
restructuring (merging fields, nesting, computing from siblings), or tightening
a ``__post_init__`` invariant old data fails.

**``__post_init__`` holds the semantic invariants** — value ranges, nested
shapes, key sets matching ``component_order``. It runs on every construction,
including after a migration, so a structurally valid but semantically broken
result is still caught; accumulate all violations and raise once. Because state
can be loaded *without* the public constructor (e.g. ``load_state_dict``
adopting checkpoint fields directly), ``__post_init__`` is the load-time trust
boundary: keep its invariants in sync with the constructor's, including
cross-field interaction rules, or a malformed checkpoint loads a config the
constructor would have rejected.

Mutation contract
~~~~~~~~~~~~~~~~~

:func:`migrate` shallow-copies *state* at the top level. Nested mutable
values stay shared with the caller; :func:`to_dict` is also shallow.
Migrations that need to mutate a nested subtree must :func:`copy.deepcopy`
that subtree first, or assemble the v_{N+1} dict from fresh values.
"""

from __future__ import annotations

from typing import Any, Callable

from zephon.core.checkpoint._schemas import (
    CURSOR_VERSION,
    ENGINE_VERSION,
    STATIC_MIXTURE_VERSION,
    WORK_CHUNK_VERSION,
    CheckpointMixin,
    CursorStateV1,
    EngineStateV1,
    StaticMixtureStateV1,
    StaticMixtureStateV2,
    StaticMixtureStateV3,
    WorkChunkStateV1,
)


def _static_mixture_v1_to_v2(v1: StaticMixtureStateV1) -> dict[str, Any]:
    """Fan v1's single ``knobs["shuffle_block_size"]`` out into per-cursor values.

    Pre-strategy v1 checkpoints lack ``allocation_mode``; here it is inferred
    from the presence of ``accumulators`` so v2 can require an explicit value.
    """
    v1_block_size = v1.knobs.get("shuffle_block_size")

    d = v1.to_dict()
    d["knobs"] = {k: v for k, v in v1.knobs.items() if k != "shuffle_block_size"}
    d["cursor_block_sizes"] = dict.fromkeys(v1.dataset_ids, v1_block_size)
    d["shuffle_block_size_spec"] = v1_block_size
    if v1.allocation_mode is None:
        d["allocation_mode"] = (
            "accumulator" if v1.accumulators is not None else "legacy_fixed"
        )
    return d


def _static_mixture_v2_to_v3(v2: StaticMixtureStateV2) -> dict[str, Any]:
    """Broadcast v2's scalar exhaustion-policy knobs to per-dataset dicts.

    v2 stored ``exhausted_policy`` / ``reshuffle_on_repeat`` / ``max_repeats``
    as single scalars applied uniformly to every dataset; v3 keys them by
    dataset name. Each scalar is fanned out across ``component_order``. v3 also
    introduces ``stop_after_passes`` (a required field); v2 had no global floor, so it
    is filled with ``None``.
    """
    d = v2.to_dict()
    names = v2.component_order
    d["exhausted_policy"] = dict.fromkeys(names, v2.exhausted_policy)
    d["reshuffle_on_repeat"] = dict.fromkeys(names, v2.reshuffle_on_repeat)
    d["max_repeats"] = dict.fromkeys(names, v2.max_repeats)
    d["stop_after_passes"] = None
    return d


#: Migration functions take a validated v_N instance and return a v_{N+1} dict.
MigrationFn = Callable[[Any], dict[str, Any]]

#: Component name -> {from_version: migration_fn}. Chains are applied in
#: order: v1->v2, v2->v3, etc.
_MIGRATIONS: dict[str, dict[int, MigrationFn]] = {
    "engine": {},
    "work_chunk": {},
    "static_mixture": {
        1: _static_mixture_v1_to_v2,
        2: _static_mixture_v2_to_v3,
    },
    "cursor": {},
}

#: Component name -> {version: schema class}. Used to validate each version
#: in the migration chain. Each version's entry is constructed from the raw
#: dict via ``from_dict``; failure raises immediately, before migration.
_SCHEMAS: dict[str, dict[int, type[CheckpointMixin]]] = {
    "engine": {1: EngineStateV1},
    "work_chunk": {1: WorkChunkStateV1},
    "static_mixture": {
        1: StaticMixtureStateV1,
        2: StaticMixtureStateV2,
        3: StaticMixtureStateV3,
    },
    "cursor": {1: CursorStateV1},
}

CURRENT_VERSIONS: dict[str, int] = {
    "engine": ENGINE_VERSION,
    "work_chunk": WORK_CHUNK_VERSION,
    "static_mixture": STATIC_MIXTURE_VERSION,
    "cursor": CURSOR_VERSION,
}


def register_migration(
    component: str,
    from_version: int,
    fn: MigrationFn,
) -> None:
    """Register a migration for *component* from *from_version* to from_version + 1."""
    if component not in _MIGRATIONS:
        raise ValueError(f"Unknown checkpoint component: {component!r}")
    if from_version in _MIGRATIONS[component]:
        raise ValueError(
            f"Migration already registered for {component} v{from_version}"
        )
    _MIGRATIONS[component][from_version] = fn


def migrate(component: str, state: dict[str, Any]) -> dict[str, Any]:
    """Apply the full migration chain for *component*, returning the upgraded dict.

    If the dict has no ``version`` field it is treated as version 1.

    Raises:
        RuntimeError: if the version is newer than the running code supports.
        ValueError: if no migration is registered for an intermediate version.
        TypeError / ValueError: if an intermediate version's schema rejects
            the input (i.e. the migration was about to receive garbage).
    """
    if component not in _MIGRATIONS:
        raise ValueError(f"Unknown checkpoint component: {component!r}")
    if not isinstance(state, dict):
        raise TypeError(
            f"Expected dict checkpoint for {component!r}, got {type(state).__name__}"
        )

    current = CURRENT_VERSIONS[component]
    version = int(state.get("version", 1))

    if version > current:
        raise RuntimeError(
            f"Checkpoint {component} version {version} is newer than the "
            f"supported version {current}. Upgrade Zephon to load this "
            f"checkpoint."
        )

    state = dict(state)  # shallow copy: protect caller's top-level keys
    chain = _MIGRATIONS[component]
    schemas = _SCHEMAS[component]
    while version < current:
        if version not in chain:
            raise ValueError(
                f"No migration registered for {component} v{version} -> v{version + 1}"
            )
        if version not in schemas:
            raise RuntimeError(
                f"No schema registered for {component} v{version} in _SCHEMAS. "
                f"When introducing v{version + 1}, the v{version} schema class "
                f"must remain in _SCHEMAS[{component!r}] so the migration "
                f"framework can validate the v_N input before applying the "
                f"v_N -> v_{{N+1}} migration."
            )
        instance = schemas[version].from_dict(state)  # validates v_{version}
        state = chain[version](instance)
        version += 1
        state["version"] = version

    return state


def current_version(component: str) -> int:
    """Return the current schema version for *component*."""
    return CURRENT_VERSIONS[component]
