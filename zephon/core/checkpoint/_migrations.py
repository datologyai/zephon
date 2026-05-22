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

When to use a migration vs. a dataclass default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The schema is a contract. A migration is needed whenever existing on-disk
dicts cannot pass ``CurrentSchema.from_dict()`` (including ``__post_init__``).
Defaults are the concession the contract makes for one specific case:
"field missing -> use this value". Anything stricter requires a migration.

**No migration needed** when:

- Adding a new field that can carry a sensible static default. Old
  checkpoints get the default automatically via ``from_dict()``.

**Migration required** when:

- Adding a new *required* field (no default). Existing checkpoints would
  raise ``TypeError`` at construction; the migration fills the value in.
- Renaming a field. ``from_dict`` ignores unknown keys and defaults the
  new name, so a rename without a migration silently drops data.
- Removing a field that something downstream depended on.
- Changing a field's type beyond what ``_auto_coerce`` handles.
- Restructuring data (merging fields, nesting, computing from siblings).
- Tightening a ``__post_init__`` invariant in a way old checkpoints
  don't satisfy — the migration fixes the data first.

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
    WorkChunkStateV1,
)

#: Migration functions take a validated v_N instance and return a v_{N+1} dict.
MigrationFn = Callable[[Any], dict[str, Any]]

#: Component name -> {from_version: migration_fn}. Chains are applied in
#: order: v1->v2, v2->v3, etc.
_MIGRATIONS: dict[str, dict[int, MigrationFn]] = {
    "engine": {},
    "work_chunk": {},
    "static_mixture": {},
    "cursor": {},
}

#: Component name -> {version: schema class}. Used to validate each version
#: in the migration chain. Each version's entry is constructed from the raw
#: dict via ``from_dict``; failure raises immediately, before migration.
_SCHEMAS: dict[str, dict[int, type[CheckpointMixin]]] = {
    "engine": {1: EngineStateV1},
    "work_chunk": {1: WorkChunkStateV1},
    "static_mixture": {1: StaticMixtureStateV1},
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
