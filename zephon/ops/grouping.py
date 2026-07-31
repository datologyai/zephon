# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Grouping of mixing domains for ``homogeneity="group"`` packing.

Grouping only *widens* which mixture components may share a packed sample; it
does not change how they are mixed. See :class:`DomainGroups`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


class DomainGroups:
    """Groups of mixing-domain (component) names that may share a packed sample.

    Members are mixture-component names (the source's ``component_ids``) — dataset
    names in ``StaticMixtureWorkSource``, but not necessarily so (a hierarchical
    source could map several datasets to one component). Validated at construction
    into an owned map, so mutating the caller's input afterward can't change it.

    Args:
        groups: Maps a group name to the component names it contains. A component
            may appear in at most one group. Components absent from every group
            form their own singleton group.

    Example:
        >>> groups = DomainGroups({"code": ["python", "java"], "web": ["c4"]})
        >>> groups.to_member_map()
        {'python': 'code', 'java': 'code', 'c4': 'web'}
    """

    def __init__(self, groups: Mapping[str, Sequence[str]]) -> None:
        if not groups:
            # A degenerate empty grouping would silently pack every component as
            # its own domain; express "no grouping" as None instead.
            raise ValueError(
                "DomainGroups requires at least one group; use None for no grouping."
            )
        member_to_group: dict[str, str] = {}
        for group_name, members in groups.items():
            if not isinstance(group_name, str) or not group_name:
                raise ValueError(
                    f"DomainGroups: group name must be a non-empty string, got "
                    f"{group_name!r}"
                )
            if isinstance(members, str):
                # A bare string is itself a Sequence[str] of chars; require a
                # non-string sequence so it isn't split into single-char members.
                raise ValueError(
                    f"DomainGroups: group {group_name!r} members must be a sequence "
                    f"(e.g. a list) of component names, not a bare string {members!r}."
                )
            member_list = list(members)
            if not member_list:
                raise ValueError(
                    f"DomainGroups: group {group_name!r} has no member components"
                )
            seen: set[str] = set()
            for name in member_list:
                if not isinstance(name, str) or not name:
                    raise ValueError(
                        f"DomainGroups: group {group_name!r} member must be a "
                        f"non-empty string, got {name!r}"
                    )
                if name in seen:
                    raise ValueError(
                        f"DomainGroups: component {name!r} listed twice in group "
                        f"{group_name!r}"
                    )
                seen.add(name)
                prior = member_to_group.get(name)
                if prior is not None and prior != group_name:
                    raise ValueError(
                        f"DomainGroups: component {name!r} appears in multiple groups "
                        f"({prior!r} and {group_name!r}); a component belongs to at "
                        "most one packing domain"
                    )
                member_to_group[name] = group_name
        # Owned dict built from the caller's input, so later mutation of the
        # caller's mapping/lists can't change us.
        self._member_to_group = member_to_group

    def to_member_map(self) -> dict[str, str]:
        """Flat ``{component: group}`` map (grouped only) — the projection packing consumes."""
        return dict(self._member_to_group)

    def validate_against(self, known_components: Iterable[str]) -> None:
        """Raise if any grouped component is absent from ``known_components``."""
        known = set(known_components)
        unknown = sorted(name for name in self._member_to_group if name not in known)
        if unknown:
            raise ValueError(
                "DomainGroups references components not in the mixture: "
                f"{unknown}. Known components: {sorted(known)}"
            )


__all__ = ["DomainGroups"]
