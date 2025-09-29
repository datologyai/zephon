# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Abstract operator contracts shared by the planner and runtime."""

from typing import Any, Optional, Protocol

from zephon.core.constants import Element
from zephon.core.traits import Buffering, OpTraits


class OpContext:
    """Container exposing runner-provided services to operator instances."""

    def __init__(self, services: dict[str, Any]):
        self._services = services

    def get(self, key: str, default: Any | None = None) -> Any:
        """Fetch a service by name, returning ``default`` when unavailable."""
        return self._services.get(key, default)


class Op(Protocol):
    """Protocol that Zephon operators must satisfy to plug into the pipeline."""

    def setup(self, ctx: OpContext) -> None: ...

    def traits(self) -> OpTraits: ...

    def buffering(self) -> Optional[Buffering]: ...

    def process_one(self, elem: Element) -> list[Element]: ...

    def process_many(self, elems: list[Element]) -> list[Element]: ...

    def finalize(self) -> list[Element]:
        """Emit any buffered outputs once the upstream iterator is exhausted."""
        ...


class DefaultFinalize:
    """Mixin providing a no-op ``finalize`` implementation."""

    def finalize(self) -> list[Element]:
        """Return an empty list when the operator has no buffered tail."""
        return []
