# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Operator authoring contracts: :class:`BaseOp` and :class:`OpContext`."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from zephon._internal.op_base import DefaultSetup
from zephon.ops.accumulators import Accumulator, PassthroughAccumulator
from zephon.ops.traits import OpTraits
from zephon.types import SampleRecord


class OpContext:
    """Container exposing runner-provided services to operator instances."""

    def __init__(self, services: dict[str, Any]):
        self._services = services

    def get(self, key: str, default: Any | None = None) -> Any:
        """Fetch a service by name, returning ``default`` when unavailable."""
        return self._services.get(key, default)


class BaseOp(DefaultSetup, ABC):
    """Base class for operators — both built-in and user-authored.

    Attach a subclass to a pipeline via the instance form of
    :py:meth:`zephon.Pipeline.add_op`::

        pipeline.add_op(MyOp(...))

    Built-in operators subclass `BaseOp` and are attached internally;
    the same class hierarchy is what user code subclasses for ops that
    need full lifecycle control (``__init__`` / ``setup`` / ``traits`` /
    ``accumulator`` / ``process_many``).  The kwargs form of ``add_op``
    is the lighter alternative for stateless transforms — it wraps user
    callables in an internal `BaseOp` subclass, so the same contract
    documented here applies either way.

    **Operators are stateless.**  Instance attributes set in ``__init__``
    are read-only configuration or shared resources (a codec table, a
    threshold, a tokenizer name).  ``process_many`` must not depend on or
    mutate state carried across calls — any cross-invocation or per-lane
    state (buffers, counters, open bins) lives in the operator's
    accumulator, not on the op instance.  This is the invariant that
    lets the runtime fan ``process_many`` out across parallel workers
    deterministically.

    Lifecycle: ``__init__`` vs ``setup``
    ------------------------------------
    ``__init__`` runs **once on the user's main process**: store config
    and lightweight references, and leave runtime fields defaulted to
    ``None``. The op instance is then deep-copied per parallel worker;
    on the process and Ray runners it is also cloudpickled across the
    process boundary, so anything assigned in ``__init__`` must survive
    cloudpickle on those runners.

    ``setup`` runs **once per worker after the deep copy** (and, on
    cross-process runners, after the cloudpickle round trip). It is the
    only place to read services from :class:`OpContext`, construct heavy
    or non-picklable resources (tokenizers, model handles, open file
    descriptors), and reset per-worker state. Resources built in
    ``setup`` never need to survive pickling.

    The convention in this repo is to construct tokenizers (and similar
    heavy resources) in ``setup``, not ``__init__``:

        class MyOp(BaseOp):
            def __init__(self, tokenizer_name: str):
                super().__init__()
                self._tokenizer_name = tokenizer_name  # picklable config
                self._tokenizer = None  # built per-worker in setup()

            def traits(self) -> OpTraits:
                return OpTraits(preserves_cursor_order=True)

            def setup(self, ctx, stage_index, stage_name, op_index, collect_stats):
                super().setup(ctx, stage_index, stage_name, op_index, collect_stats)
                self._tokenizer = load_tokenizer(self._tokenizer_name)

            def process_many(self, elems):
                # Records in, records out: rebuild payload, preserve meta.
                for e in elems:
                    e.payload = {**e.payload, "ids": self._tokenizer.encode(e.payload["text"])}
                return elems

        pipeline.add_op(MyOp("gpt2"))

    Defaults supplied:

    - ``setup`` — inherits `DefaultSetup`, which records stage metadata on
      the instance (``stage_index``, ``stage_name``, etc.). Override to
      construct per-worker resources, and call ``super().setup(...)`` to
      preserve the stage metadata bookkeeping.
    - ``accumulator`` — returns `PassthroughAccumulator` so each upstream
      micro-batch is forwarded as one ready batch. Override to enable
      size-based per-lane batching via `CountingAccumulator` or any custom
      `Accumulator`.
    - ``process_one`` — wraps the element in a single-item list and routes
      it through ``process_many``. Override only when the single-element
      fast path differs (rare).

    Abstract — every subclass must implement:

    - ``process_many`` — the workhorse transform.
    - ``traits`` — must return an `OpTraits` with ``preserves_cursor_order``
      declared; the subclass must declare ``True`` (1:1 maps, payload
      transforms, non-reordering filters) or ``False`` (reorders, shuffles,
      packs).

    Instantiating a subclass that doesn't override both ``process_many``
    and ``traits`` raises ``TypeError`` at construction time.
    """

    @abstractmethod
    def traits(self) -> OpTraits: ...

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[Any]:
        return PassthroughAccumulator[Any]()

    def process_one(self, elem: Any) -> list[Any]:
        return self.process_many([elem])

    @abstractmethod
    def process_many(self, elems: list[Any]) -> list[Any]: ...

    def validation_samples(self) -> Optional[list[SampleRecord]]:
        """Optional records for the validation harness.

        Override to opt this op into the validator's runtime check
        suite.  Returning a small batch (~12 records across a couple of
        lanes) lets the harness run determinism, cross-call-state,
        self-mutation, state-diff, and sample-identity probes against
        ``process_many``.

        Why instance-form ops are gated this way: the validator never
        invokes ``setup``, so probing a `BaseOp` subclass whose
        ``process_many`` depends on setup-built resources (the typical
        pattern — tokenizers, model handles, etc.) would raise
        spuriously.  Returning ``None`` (the default) tells the
        validator to skip the runtime checks for this op and surface a
        single ``OP_INSTANCE_RUNTIME_CHECKS_SKIPPED`` warning instead
        of crashing the pipeline under ``auto_validation="strict"``.
        Static-AST checks (self-writes, non-deterministic stdlib calls)
        run unconditionally either way.

        Returning an empty list, a list of the wrong type, or raising
        surfaces an ``OP_VALIDATION_SAMPLES_FACTORY_FAILED`` warning
        and falls back to the skip path — the validator never dies on
        a buggy factory.

        Tip: include at least two distinct ``lane_id`` values so the
        cross-call-state probe (which derives an A-vs-B sample-id
        variant from your records) and the lane-purity accumulator
        checks remain meaningful.  Records must work *without*
        ``setup`` having run — pre-tokenize / pre-encode your payload
        rather than relying on the resources the real ``setup`` would
        build.
        """
        return None
