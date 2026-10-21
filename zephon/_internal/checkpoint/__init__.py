"""Checkpoint versioning, schemas, migrations, and aggregation codec."""

from zephon._internal.checkpoint._codec import AggregationCodec
from zephon._internal.checkpoint._migrations import (
    CURRENT_VERSIONS,
    MigrationFn,
    current_version,
    migrate,
    register_migration,
)
from zephon._internal.checkpoint._schemas import (
    CURSOR_VERSION,
    ENGINE_VERSION,
    STATIC_MIXTURE_VERSION,
    VALID_TOKEN_RATIO_SOURCES,
    WORK_CHUNK_VERSION,
    CheckpointMixin,
    CursorStateV1,
    EngineStateV1,
    EngineStateV2,
    StaticMixtureStateV1,
    StaticMixtureStateV2,
    StaticMixtureStateV3,
    StaticMixtureStateV4,
    StaticMixtureStateV5,
    WorkChunkStateV1,
    WorkChunkStateV2,
    WorkChunkStateV3,
)

__all__ = [
    # Schemas
    "CheckpointMixin",
    "CursorStateV1",
    "WorkChunkStateV1",
    "WorkChunkStateV2",
    "WorkChunkStateV3",
    "StaticMixtureStateV1",
    "StaticMixtureStateV2",
    "StaticMixtureStateV3",
    "StaticMixtureStateV4",
    "StaticMixtureStateV5",
    "EngineStateV1",
    "EngineStateV2",
    "VALID_TOKEN_RATIO_SOURCES",
    # Version constants
    "CURSOR_VERSION",
    "WORK_CHUNK_VERSION",
    "STATIC_MIXTURE_VERSION",
    "ENGINE_VERSION",
    # Migrations
    "migrate",
    "register_migration",
    "current_version",
    "CURRENT_VERSIONS",
    "MigrationFn",
    # Codec
    "AggregationCodec",
]
