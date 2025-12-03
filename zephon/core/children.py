# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Helpers for child/contributor-aware metadata construction."""

from collections.abc import Iterable
from typing import Any

from zephon.core.constants import ContributorRef, SampleCursor, SampleMeta


def spawn_child(
    parent: SampleMeta,
    child_idx: int,
    *,
    is_last_child: bool = False,
    tags: dict[str, Any] | None = None,
) -> SampleMeta:
    """Create metadata for a child derived from ``parent``.

    - Assigns deterministic child lineage for identity/replay.
    - Contributors:
        * If the parent has no contributors (single-base record), emit one
          contributor tied to the child cursor and honor ``is_last_child``.
        * If the parent already has contributors (e.g., packed input), propagate
          them to the child; if ``is_last_child=True``, mark all propagated
          contributors as closing.
    - Raises if you try to mark a child as closing a base offset that is already
      marked closed by the parent metadata (e.g., parent contributors already
      contain an ``is_last_child=True`` ref or the parent is a tombstone).
    """
    if is_last_child:
        if parent.tombstone:
            raise ValueError("Cannot spawn a closing child from tombstone metadata")
        if parent.contributors and any(
            ref.is_last_child for ref in parent.contributors
        ):
            raise ValueError(
                "Cannot mark child as last: parent metadata already marks the base offset closed"
            )

    new_lineage = parent.lineage + (int(child_idx),)
    if parent.contributors:
        refs = tuple(
            ContributorRef(
                cursor=ref.cursor,
                is_last_child=(ref.is_last_child or is_last_child),
            )
            for ref in parent.contributors
        )
    else:
        cursor = SampleCursor(
            parent.chunk_id, parent.chunk_offset, parent.sample_id, new_lineage
        )
        refs = (ContributorRef(cursor=cursor, is_last_child=is_last_child),)

    child_tags = dict(parent.tags) if tags is None else dict(tags)
    child_tags.pop("_tombstone", None)
    meta = SampleMeta(
        sample_id=parent.sample_id,
        lane_id=parent.lane_id,
        chunk_id=parent.chunk_id,
        chunk_offset=parent.chunk_offset,
        lineage=new_lineage,
        tags=child_tags,
    ).with_contributors(refs)
    return meta


def pack_meta(
    primary_cursor: SampleCursor,
    contributors: Iterable[ContributorRef],
    *,
    lane_id: int,
    tags: dict[str, Any] | None = None,
) -> SampleMeta:
    """Build metadata for a packed record that merges multiple contributors.

    ``primary_cursor`` is the replay identity for the packed record and must be
    unique per lane. ``contributors`` lists all contributors included in the pack;
    any contributor that completes a base offset must set ``is_last_child=True``.
    """
    tags = {} if tags is None else dict(tags)
    tags.pop("_tombstone", None)
    meta = SampleMeta(
        sample_id=primary_cursor.sample_id,
        lane_id=lane_id,
        chunk_id=primary_cursor.chunk_id,
        chunk_offset=primary_cursor.chunk_offset,
        lineage=primary_cursor.lineage,
        tags=tags,
    ).with_contributors(tuple(contributors))
    return meta


def tombstone_meta(ref: ContributorRef, lane_id: int) -> SampleMeta:
    """Emit a tombstone that marks a base offset complete without payload."""
    if not ref.is_last_child:
        raise ValueError(
            "tombstone_meta requires a closing contributor (is_last_child=True)"
        )
    meta = SampleMeta(
        sample_id=ref.cursor.sample_id,
        lane_id=lane_id,
        chunk_id=ref.cursor.chunk_id,
        chunk_offset=ref.cursor.chunk_offset,
        lineage=ref.cursor.lineage,
        tags={"_tombstone": True},
    ).with_contributors((ref,))
    return meta
