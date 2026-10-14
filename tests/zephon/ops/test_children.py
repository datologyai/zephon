import pytest

from zephon.ops.children import pack_meta, spawn_child, tombstone_meta
from zephon.types import ContributorRef, SampleCursor, SampleMeta


def test_contribution_refs_default_to_meta_cursor() -> None:
    meta = SampleMeta(sample_id=(0, 0, 0), lane_id=1, chunk_id=2, chunk_offset=3)
    refs = meta.contribution_refs()
    assert len(refs) == 1
    assert refs[0].cursor == meta.cursor
    assert refs[0].is_last_child is True


def test_spawn_child_builds_child_lineage_and_ref() -> None:
    parent = SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=1, chunk_offset=0)
    child = spawn_child(parent, 5, is_last_child=True)
    assert child.lineage[-1] == 5
    assert len(child.contributors) == 1
    ref = child.contributors[0]
    assert ref.cursor == SampleCursor(1, 0, parent.sample_id, child.lineage)
    assert ref.is_last_child is True


@pytest.mark.parametrize("closing", [False, True])
def test_nested_children_inherit_parent_completion(closing: bool) -> None:
    base = SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=1, chunk_offset=0)
    parent = spawn_child(base, 3, is_last_child=closing)
    first = spawn_child(parent, 0)
    last = spawn_child(parent, 1, is_last_child=True)
    assert first.lineage == (3, 0)
    assert last.lineage == (3, 1)
    assert not first.contributors[0].is_last_child
    assert last.contributors == parent.contributors


def test_spawn_child_preserves_mixed_contributor_flags() -> None:
    refs = (
        ContributorRef(SampleCursor(1, 0, (0, 0, 1)), False),
        ContributorRef(SampleCursor(1, 1, (0, 0, 2)), True),
    )
    parent = SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=1).with_contributors(
        refs
    )
    assert not any(r.is_last_child for r in spawn_child(parent, 0).contributors)
    assert spawn_child(parent, 1, is_last_child=True).contributors == refs


def test_spawn_child_rejects_closing_tombstone() -> None:
    parent = SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=1).with_tombstone(True)
    with pytest.raises(ValueError, match="tombstone"):
        spawn_child(parent, 0, is_last_child=True)


def test_pack_and_tombstone_helpers() -> None:
    base_cursor = SampleCursor(3, 1, (0, 0, 7), (9,))
    refs = (
        ContributorRef(cursor=SampleCursor(3, 1, (0, 0, 7), (1,)), is_last_child=False),
        ContributorRef(cursor=SampleCursor(4, 0, (0, 0, 8), (2,)), is_last_child=True),
    )
    # Packing two samples from component 0
    packed = pack_meta(
        base_cursor,
        refs,
        lane_id=2,
        component_sample_counts={0: 2},
        tags={"kind": "packed"},
    )
    assert packed.cursor == base_cursor
    assert packed.contributors == refs
    assert packed.tags["kind"] == "packed"
    assert packed.component_sample_counts == {0: 2}

    tombstone = tombstone_meta(refs[1], lane_id=2)
    assert tombstone.tombstone
    assert tombstone.contributors[0].cursor == refs[1].cursor
    assert tombstone.contributors[0].is_last_child is True


def test_tombstone_meta_requires_closing_contributor() -> None:
    ref = ContributorRef(
        cursor=SampleCursor(1, 0, (0, 0, 1), (0,)), is_last_child=False
    )
    with pytest.raises(ValueError):
        tombstone_meta(ref, lane_id=0)
