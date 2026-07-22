# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DomainGroups (grouping mixing domains for homogeneous packing)."""

from __future__ import annotations

import pickle

import cloudpickle
import pytest

from zephon.ops.grouping import DomainGroups


def test_to_member_map_flattens_grouped_only() -> None:
    groups = DomainGroups({"code": ["python", "java"], "web": ["c4"]})
    # Ungrouped components are absent (packing treats them as singleton domains).
    assert groups.to_member_map() == {"python": "code", "java": "code", "c4": "web"}
    # Independent copy: mutating the result does not corrupt the DomainGroups.
    groups.to_member_map()["python"] = "web"
    assert groups.to_member_map()["python"] == "code"


@pytest.mark.parametrize(
    "dumps", [pickle.dumps, cloudpickle.dumps], ids=["pickle", "cloudpickle"]
)
def test_round_trip_preserves_member_map(dumps) -> None:
    # The op carries DomainGroups to worker processes, so it must survive both
    # pickle and cloudpickle (the process-runner's path).
    groups = DomainGroups({"code": ["python", "java"], "web": ["c4"]})
    restored = pickle.loads(dumps(groups))
    assert restored.to_member_map() == groups.to_member_map()


def test_snapshots_input_against_later_mutation() -> None:
    members = ["python", "java"]
    src = {"code": members}
    groups = DomainGroups(src)
    members.append("rust")  # mutate the caller's list after construction
    src["web"] = ["c4"]  # mutate the caller's mapping after construction
    assert groups.to_member_map() == {"python": "code", "java": "code"}


def test_validate_against_known_components() -> None:
    groups = DomainGroups({"code": ["python", "java"]})
    groups.validate_against(["python", "java", "wiki"])  # ok: members are known
    with pytest.raises(ValueError, match="not in the mixture"):
        groups.validate_against(["python", "wiki"])  # 'java' missing


def test_component_in_two_groups_raises() -> None:
    with pytest.raises(ValueError, match="multiple groups"):
        DomainGroups({"code": ["python", "shared"], "web": ["shared"]})


def test_empty_group_or_name_raises() -> None:
    with pytest.raises(ValueError, match="no member components"):
        DomainGroups({"code": []})
    with pytest.raises(ValueError, match="non-empty"):
        DomainGroups({"": ["python"]})


def test_empty_grouping_rejected() -> None:
    # An empty grouping would silently self-group every component; require None.
    with pytest.raises(ValueError, match="at least one group"):
        DomainGroups({})


def test_non_string_group_name_rejected() -> None:
    with pytest.raises(ValueError, match="group name must be a non-empty string"):
        DomainGroups({5: ["python"]})  # type: ignore[dict-item]


def test_non_string_or_empty_member_rejected() -> None:
    with pytest.raises(ValueError, match="member must be a non-empty string"):
        DomainGroups({"code": ["python", ""]})  # empty member
    with pytest.raises(ValueError, match="member must be a non-empty string"):
        DomainGroups({"code": [1, 2]})  # type: ignore[list-item]


def test_duplicate_member_within_group_raises() -> None:
    with pytest.raises(ValueError, match="listed twice"):
        DomainGroups({"code": ["python", "python"]})


def test_bare_string_members_rejected() -> None:
    # A bare string would otherwise split into single-char "components".
    with pytest.raises(ValueError, match="not a bare string"):
        DomainGroups({"code": "python"})
