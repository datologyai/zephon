# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Internal `BaseOp` subclass assembled from `Pipeline.add_op` kwargs."""

import inspect
from typing import Any, Callable, Optional

from zephon.core.accumulators import Accumulator
from zephon.core.constants import SampleRecord
from zephon.core.op_base import BaseOp
from zephon.core.traits import OpTraits


class _FunctionalOp(BaseOp):
    """Internal Op assembled from `Pipeline.add_op` kwargs.

    Holds an `OpTraits` value, an `Accumulator` factory, and the user-supplied
    `process_many` (and optional `process_one`) callables. The runner sees
    this as an ordinary `BaseOp` subclass.

    The accumulator factory may be either `Callable[[], Accumulator]` or
    `Callable[*, deterministic, ctx], Accumulator]` — the constructor
    inspects the signature once and routes `accumulator()` calls accordingly.
    Lets users pick between deterministic-aware accumulators (e.g. switching
    `CountingAccumulator(max_latency_ms=...)` on the deterministic flag) and
    the simpler 0-arg form.
    """

    def __init__(
        self,
        *,
        process_many_fn: Callable[[list[Any]], list[Any]],
        accumulator_factory: Callable[..., Accumulator[Any]],
        process_one_fn: Optional[Callable[[Any], list[Any]]],
        op_traits: OpTraits,
        validation_samples_factory: Optional[Callable[[], list[SampleRecord]]] = None,
    ) -> None:
        super().__init__()
        self._process_many_fn = process_many_fn
        self._accumulator_factory = accumulator_factory
        self._process_one_fn = process_one_fn
        self._traits = op_traits
        self._validation_samples_factory = validation_samples_factory
        self._factory_takes_kwargs = _accumulator_factory_takes_kwargs(
            accumulator_factory
        )

    def traits(self) -> OpTraits:
        return self._traits

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[Any]:
        if self._factory_takes_kwargs:
            return self._accumulator_factory(deterministic=deterministic, ctx=ctx)
        return self._accumulator_factory()

    def process_one(self, elem: Any) -> list[Any]:
        if self._process_one_fn is not None:
            return self._process_one_fn(elem)
        return self._process_many_fn([elem])

    def process_many(self, elems: list[Any]) -> list[Any]:
        return self._process_many_fn(elems)

    def validation_samples(self) -> Optional[list[SampleRecord]]:
        if self._validation_samples_factory is None:
            return None
        return self._validation_samples_factory()


def _accumulator_factory_takes_kwargs(factory: Callable[..., Any]) -> bool:
    """Decide whether ``factory`` accepts the ``deterministic``/``ctx`` kwargs.

    True when the factory declares any parameter (positional, keyword, or
    ``**kwargs``).  False when the factory takes nothing.  Falls back to
    False for callables that ``inspect.signature`` can't introspect (some
    C builtins, certain wrappers).
    """
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False
    return bool(params)
