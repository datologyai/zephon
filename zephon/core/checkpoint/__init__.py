"""Checkpoint versioning, schemas, migrations, and aggregation codec."""

from zephon.core.checkpoint._codec import AggregationCodec
from zephon.core.checkpoint._migrations import (
    CURRENT_VERSIONS,
    MigrationFn,
    current_version,
    migrate,
    register_migration,
)
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
    StaticMixtureStateV4,
    WorkChunkStateV1,
)

__all__ = [
    # Schemas
    "CheckpointMixin",
    "CursorStateV1",
    "WorkChunkStateV1",
    "StaticMixtureStateV1",
    "StaticMixtureStateV2",
    "StaticMixtureStateV3",
    "StaticMixtureStateV4",
    "EngineStateV1",
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
