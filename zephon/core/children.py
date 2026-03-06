# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Helpers for child/contributor-aware metadata construction."""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, Callable

from zephon.core.constants import ContributorRef, SampleCursor, SampleMeta, SampleRecord


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
        # Preserve component contribution tracking from parent.
        # Children inherit the same component membership as their parent.
        component_sample_counts=dict(parent.component_sample_counts),
        component_token_counts=(
            dict(parent.component_token_counts)
            if parent.component_token_counts is not None
            else None
        ),
        lineage=new_lineage,
        tags=child_tags,
    ).with_contributors(refs)
    return meta


def pack_meta(
    primary_cursor: SampleCursor,
    contributors: Iterable[ContributorRef],
    *,
    lane_id: int,
    component_sample_counts: dict[int, int],
    component_token_counts: dict[int, int] | None = None,
    tags: dict[str, Any] | None = None,
) -> SampleMeta:
    """Build metadata for a packed record that merges multiple contributors.

    ``primary_cursor`` is the replay identity for the packed record and must be
    unique per lane. ``contributors`` lists all contributors included in the pack;
    any contributor that completes a base offset must set ``is_last_child=True``.

    ``component_sample_counts`` aggregates how many original samples from each
    component are included in this pack. For example, if packing 3 samples from
    component 0 and 2 from component 1, this would be ``{0: 3, 1: 2}``.

    ``component_token_counts`` optionally provides token counts per component,
    computed when packing happens after tokenization. If packing before tokenize,
    pass None and ensure_mixture will fall back to distributing by sample counts.
    """
    tags = {} if tags is None else dict(tags)
    tags.pop("_tombstone", None)
    meta = SampleMeta(
        sample_id=primary_cursor.sample_id,
        lane_id=lane_id,
        chunk_id=primary_cursor.chunk_id,
        chunk_offset=primary_cursor.chunk_offset,
        component_sample_counts=component_sample_counts,
        component_token_counts=component_token_counts,
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


def tombstones_for_record(record: SampleRecord) -> list[SampleRecord]:
    """Emit tombstone records for every closing contributor in *record*.

    Each record tracks which chunk offsets it contributes to via
    ``contribution_refs()``.  Only refs with ``is_last_child=True`` need a
    tombstone — intermediate children (from split/spawn) don't close the
    base offset, so the engine doesn't need a signal for them.
    """
    tombstones: list[SampleRecord] = []
    for ref in record.meta.contribution_refs():
        if ref.is_last_child:
            tombstones.append(
                SampleRecord(
                    meta=tombstone_meta(ref, record.meta.lane_id),
                    payload=None,
                )
            )
    return tombstones


def collect_pack_contributions(
    samples: Sequence[SampleRecord],
    length_fn: Callable[[SampleRecord], int],
) -> tuple[list[ContributorRef], dict[int, int], dict[int, int]]:
    """Gather contributors and aggregate component counts from packed samples.

    Returns ``(contributors, component_sample_counts, component_token_counts)``
    ready for ``pack_meta()``.  Used by ``PackSequences`` and any external
    packing operator (e.g. legacy shuffle+pack).
    """
    contributors: list[ContributorRef] = []
    component_sample_counts: dict[int, int] = defaultdict(int)
    component_token_counts: dict[int, int] = defaultdict(int)

    for sample in samples:
        contributors.extend(sample.meta.contribution_refs())
        seq_len = length_fn(sample)
        for cid, count in sample.meta.component_sample_counts.items():
            component_sample_counts[cid] += count
        if sample.meta.component_token_counts is not None:
            for cid, tokens in sample.meta.component_token_counts.items():
                component_token_counts[cid] += tokens
        else:
            total_samples = sum(sample.meta.component_sample_counts.values())
            for cid, count in sample.meta.component_sample_counts.items():
                component_token_counts[cid] += round(seq_len * count / total_samples)

    return contributors, dict(component_sample_counts), dict(component_token_counts)
