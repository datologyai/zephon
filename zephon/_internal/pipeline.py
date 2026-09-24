# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""User-facing pipeline wrapper that layers ergonomics atop core planning."""

import functools
import importlib.util as _importlib_util
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    Optional,
    Protocol,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    overload,
)

from zephon._internal.engine import Engine
from zephon._internal.functional_op import _FunctionalOp
from zephon._internal.graph import Graph, Node, Plan
from zephon._internal.ops.batch import Batch
from zephon._internal.ops.decode_text import DecodeText
from zephon._internal.ops.ensure_mixture import EnsureMixture
from zephon._internal.ops.fetch import FetchOp
from zephon._internal.ops.map_transform import MapBatchTransform, MapTransform
from zephon._internal.ops.materialize import Materialize
from zephon._internal.ops.pack_sequences import PackSequences
from zephon._internal.ops.prefetch import PrefetchOp
from zephon._internal.ops.shuffle_buffer import ShuffleBuffer
from zephon._internal.ops.tokenize_base import TokenizeBase
from zephon._internal.ops.tokenize_chat import TokenizeChat
from zephon._internal.ops.tokenize_text import TokenizeText
from zephon._internal.planner import Planner
from zephon._internal.runtime_spec import (
    RuntimeSpec,
    resolve_mtp_buffer,
    resolve_prefetch_batches,
    resolve_runtime_spec,
)
from zephon._internal.stream import EngineSample
from zephon._internal.utils import buffered_iterable
from zephon._internal.utils.torch_compat import detect_loader_kind
from zephon.io.options import StoreOptions
from zephon.observability import ExecutionTrackingMode, MetricsSinkConfig
from zephon.observability.mtp_stats import MTPQueueStats
from zephon.observability.stats import (
    FetchTimingSummary,
    PipelineSummary,
    PrefetchTimingSummary,
)
from zephon.ops.accumulators import Accumulator, PassthroughAccumulator
from zephon.ops.base import BaseOp
from zephon.ops.config import (
    MissingFieldMode,
    PackingAlgorithm,
    SpanSource,
    SpecialTokensMode,
)
from zephon.ops.grouping import DomainGroups
from zephon.ops.traits import OpTraits
from zephon.options import RuntimeOptions
from zephon.types import SampleBatch, SampleId, SamplePayload, SampleRecord, StreamItem
from zephon.validation import ValidationError, preflight_tokenizers
from zephon.work import WorkSource
from zephon.work.token_estimation import _PreTokenizeReplay, _UnreplayableOp

# TypeVar for stateful_transform state type
_S = TypeVar("_S")

_OpT = TypeVar("_OpT")

# Ops calibration can replay standalone: pure per-record, no accumulator
# state, no OpContext services.
_PRE_TOKENIZE_REPLAYABLE_OPS: tuple[type, ...] = (DecodeText, MapTransform)

# TypeVar for Pipeline methods that mutate the graph or cached plan state
_PipelineMethod = TypeVar("_PipelineMethod", bound=Callable[..., Any])


# ---------- Private protocol fallbacks (non-public => no D101) ----------
class _IterableDatasetProto(Protocol):
    def __iter__(self) -> Iterator[Any]: ...


class _DatasetProto(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...


# Type checkers expose Torch types; runtime without Torch uses protocols.
if TYPE_CHECKING:
    from torch.utils.data import Dataset as _TorchDataset
    from torch.utils.data import IterableDataset as _TorchIterableDataset

    _TorchIterableDatasetType: TypeAlias = _TorchIterableDataset[Any]
    _TorchDatasetType: TypeAlias = _TorchDataset[Any]

    from zephon.validation import ValidationReport
else:
    _TorchIterableDatasetType: TypeAlias = _IterableDatasetProto
    _TorchDatasetType: TypeAlias = _DatasetProto

# ---------- Runtime base (used for inheritance / isinstance) ----------
_RTIterableDatasetBase: type[Any]
try:
    from torch.utils.data import IterableDataset as _RTIterableDatasetBase
except Exception:
    # Provide a concrete runtime base so type checkers accept the subclass definition.
    _RTIterableDatasetBase = object  # type: ignore[assignment]


class _TorchPipelineIterableDataset(_RTIterableDatasetBase):
    """Adapter that just yields from the Zephon Pipeline; no sharding here."""

    def __init__(self, pipeline: "Pipeline", *, stateful: bool = False) -> None:
        self._pipeline = pipeline
        self._stateful = stateful
        # Only used when stateful=True.
        self._pending_ckpt: dict[str, Any] | None = None

    def __iter__(self) -> Iterator[Any]:
        # Defer engine construction & (if stateful) applying checkpoint until we’re in the worker.
        if self._stateful and self._pending_ckpt is not None:
            # restore() will internally _ensure() and apply the checkpoint.
            self._pipeline.restore(self._pending_ckpt)
            self._pending_ckpt = None

        if detect_loader_kind() == "unknown":
            print(
                "Warning! You seem to be using neither torchdata.StatefulDataloader nor torch.DataLoader. You might want to consider iterating over the Pipeline directly, as the TorchDataset wrapper is mostly used as a tool to integrate with legacy setups that require the DataLoader class."
            )

        # Worker partitioning handled by Engine/WorkSource internally.
        yield from self._pipeline

    def state_dict(self) -> dict[str, Any]:
        if not self._stateful:
            raise AttributeError(
                "This dataset is not stateful. Pass stateful=True in to_torch_dataset()."
            )

        dl_kind = detect_loader_kind()
        if dl_kind == "vanilla":
            raise NotImplementedError(
                "Obtaining correct state via the torch.DataLoader is not supported. If you need to checkpoint state, please use the torchdata.StatefulDataloader instead."
            )
        elif dl_kind == "unknown":
            print(
                "Warning! You seem to be using neither torchdata.StatefulDataloader nor torch.DataLoader. You might want to consider iterating over the Pipeline directly, as the TorchDataset wrapper is mostly used as a tool to integrate with legacy setups that require the DataLoader class."
            )

        # If an active iteration exists (inline engine or subprocess), use pipeline.checkpoint().
        if self._pipeline._engine is not None or self._pipeline._sp is not None:
            return {"engine": self._pipeline.checkpoint()}
        # If iteration hasn’t begun in this process, return whatever pending state we have (or None).
        return {"engine": self._pending_ckpt}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        if not self._stateful:
            raise AttributeError(
                "This dataset is not stateful. Pass stateful=True in to_torch_dataset()."
            )
        # Do NOT call pipeline.restore() here — this might be running in the parent.
        # Just stash it; __iter__ in the worker will apply it before building the engine.
        self._pending_ckpt = sd.get("engine")


def _mutates_graph(method: _PipelineMethod) -> _PipelineMethod:
    """Decorator for Pipeline methods that mutate the graph or cached plan state.

    Automatically calls ``_invalidate_plan()`` before the method body so that
    the plan/engine cache is always cleared when the graph changes.  Applying
    this decorator makes the intent explicit and is enforced by the structural
    test ``test_all_graph_mutating_methods_invalidate_plan``.
    """

    @functools.wraps(method)
    def wrapper(self: "Pipeline", *args: Any, **kwargs: Any) -> Any:
        self._invalidate_plan()
        return method(self, *args, **kwargs)

    wrapper._mutates_graph = True  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


class Pipeline:
    """Fluent builder that compiles user ops into an executable pipeline."""

    def __init__(self, work_source: WorkSource) -> None:
        self.ws = work_source
        self._graph = Graph()
        self._plan: Plan | None = None
        self._runtime_spec: RuntimeSpec | None = None
        self._engine: Engine | None = None
        self._sp: Any = None  # MTPPipeline handle (lazy import)
        self._pending_restore: dict[str, Any] | None = None
        self._last_state: dict[str, Any] | None = None
        self._iterating: bool = False
        self._validated: bool = False
        self._options = RuntimeOptions()
        self._fetch_node: Node[FetchOp] = self._graph.add(
            "fetch", FetchOp(), placement="local"
        )
        self._tail: Node[Any] = self._fetch_node
        # Optional prefetch node inserted before fetch
        self._prefetch_node: Node[PrefetchOp] | None = None

    @_mutates_graph
    def fetch_parallelism(
        self, parallelism: int | None, max_batch: int | None = None
    ) -> "Pipeline":
        """Override the implicit FetchOp parallelism and batch size.

        Args:
            parallelism: Number of parallel fetch workers. If None, uses default from traits.
            max_batch: Maximum batch size for fetch accumulator. If None, uses default (64).
        """
        if parallelism is None:
            parallelism = max(1, self._fetch_node.op.traits().parallelism)
        elif parallelism < 1:
            raise ValueError("Fetch parallelism must be >= 1.")
        self._fetch_node.parallelism = parallelism

        if max_batch is not None:
            if max_batch < 1:
                raise ValueError("Fetch max_batch must be >= 1.")
            self._fetch_node.op._max_batch = max_batch

        return self

    def fetch(
        self, parallelism: int | None = None, max_batch: int | None = None
    ) -> "Pipeline":
        """Configure fetch operator parallelism and batch size.

        Args:
            parallelism: Number of parallel fetch workers. If None, uses default.
            max_batch: Maximum batch size for fetch accumulator. If None, uses default (64).
        """
        return self.fetch_parallelism(parallelism, max_batch)

    @_mutates_graph
    def prefetch(
        self,
        buffer_size: int = 1024,
        parallelism: int | None = None,
        *,
        placement: str = "local",
    ) -> "Pipeline":
        """Add a prefetch operator to warm the cache before fetching.

        The prefetch operator looks ahead in the sample stream and downloads
        shards to the local cache before they're needed by FetchOp. This
        significantly reduces fetch latency when loading from remote storage (S3, GCS).

        The prefetch node is inserted before the fetch node in the pipeline.

        Args:
            buffer_size: Number of samples to buffer for lookahead (default: 1024).
                Larger values provide more prefetch opportunities but use more memory.
            parallelism: Number of concurrent worker threads for downloads (default: 4).
                Higher values increase download parallelism.
            placement: Placement hint for the prefetch operator (default: "local").

        Returns:
            Self for method chaining.

        Example:
            >>> pipeline = (
            ...     Pipeline(work_source)
            ...     .prefetch(buffer_size=2048, parallelism=8)
            ...     .decode_text()
            ...     .tokenize(field="text")
            ...     .batch(32)
            ... )

            Note: The fetch operator is automatically added by Pipeline.__init__,
            so you don't need to call .fetch() explicitly. The prefetch operator
            is inserted before the implicit fetch operator.

        Note:
            Prefetch is most effective with:
            - Remote storage (S3, GCS) with high download latency
            - Sequential or predictable shard access patterns
            - Large shards where download time is significant

            For local storage or already-cached data, prefetch has minimal benefit.
        """
        if self._prefetch_node is not None:
            raise RuntimeError("prefetch() can only be called once per pipeline")

        op = PrefetchOp(buffer_size=buffer_size)

        # Insert prefetch node before fetch node in the graph
        # The prefetch node has no inputs (connects to work source output)
        # We need to insert it BEFORE the fetch node in the nodes list
        # Use provided parallelism or fall back to operator's default
        node_parallelism = (
            parallelism if parallelism is not None else op.traits().parallelism
        )
        prefetch_node = Node(
            name="prefetch",
            op=op,
            inputs=[],  # No inputs - reads from work source
            placement=placement,
            parallelism=max(1, node_parallelism),
        )
        self._prefetch_node = prefetch_node

        # Insert prefetch at the beginning (before fetch)
        self._graph.nodes.insert(0, prefetch_node)

        # Update fetch node to depend on prefetch
        self._fetch_node.inputs = [prefetch_node]

        return self

    @_mutates_graph
    def decode_text(
        self, parallelism: Optional[int] = None, **kwargs: Any
    ) -> "Pipeline":
        node = self._graph.add(
            "decode_text",
            DecodeText(**kwargs),
            self._tail,
            placement="local",
            parallelism=parallelism,
        )
        self._tail = node
        return self

    @_mutates_graph
    def map_transform(
        self,
        transform_fn: Callable[[SamplePayload], SamplePayload | None],
        *,
        drop_none: bool = True,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> "Pipeline":
        """Add a map-style transformation operator.

        Applies a user-provided transformation function to each sample's payload.
        Supports lambda functions, closures, and nested functions seamlessly with
        multiprocessing-based runners (uses cloudpickle for serialization).

        Args:
            transform_fn: Callable that transforms the payload.
                If it returns None and drop_none=True, the sample is filtered out.
                Supports lambdas, closures, and nested functions.
            drop_none: If True, drop samples where transform_fn returns None.
            placement: Placement hint for this operator.
            parallelism: Override default parallelism for this operator.

        Returns:
            Self for method chaining.

        Example:
            >>> pipeline.map_transform(lambda x: {"value": x["value"] * 2})
        """
        op = MapTransform(transform_fn, drop_none=drop_none)
        node = self._graph.add(
            "map_transform",
            op,
            self._tail,
            placement=placement,
            parallelism=parallelism,
        )
        self._tail = node
        return self

    @_mutates_graph
    def map_batch(
        self,
        transform_fn: Callable[[SampleBatch], SampleBatch | None],
        *,
        drop_none: bool = True,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> "Pipeline":
        """Add a batch-level map transformation operator.

        Applies a user-provided transformation function to each ``SampleBatch``.
        This operator must be placed **after** a ``batch()`` call in the pipeline.
        For per-sample transforms (before batching), use ``map_transform()`` instead.

        Args:
            transform_fn: Callable that transforms a ``SampleBatch``.
                If it returns None and drop_none=True, the batch is filtered out.
                Supports lambdas, closures, and nested functions.
            drop_none: If True, drop batches where transform_fn returns None.
            placement: Placement hint for this operator.
            parallelism: Override default parallelism for this operator.

        Returns:
            Self for method chaining.

        Example:
            >>> pipeline.batch(32).map_batch(lambda b: process(b))
        """
        op = MapBatchTransform(transform_fn, drop_none=drop_none)
        node = self._graph.add(
            "map_batch",
            op,
            self._tail,
            placement=placement,
            parallelism=parallelism,
        )
        self._tail = node
        return self

    @_mutates_graph
    def stateful_transform(
        self,
        name: str,
        *,
        init_state: Callable[[], _S],
        push: Callable[[_S, list[SampleRecord]], tuple[_S, list[SampleRecord]]],
        flush: Optional[Callable[[_S], list[SampleRecord]]] = None,
        should_flush: Optional[Callable[[_S], bool]] = None,
        transform: Optional[Callable[[list[SampleRecord]], list[SampleRecord]]] = None,
        placement: str = "auto",
        parallelism: int = 1,
        indexable: bool = False,
        preserves_cursor_order: bool = True,
    ) -> "Pipeline":
        """Add a stateful transformation with custom accumulation logic.

        This is a higher-level alternative to subclassing ``BaseOp`` directly.
        State management runs on the pump thread (serial); the optional transform
        runs in parallel workers for expensive computation.

        Execution model:

        - push/flush: Run on pump thread (serial) for state management
        - transform: Runs in parallel workers for expensive per-item processing

        This split enables patterns like "deduplicate (serial) then encode (parallel)".

        State is partitioned per lane — each lane keeps its own state instance
        and your callbacks see one lane at a time, so an epoch-boundary flush
        resets only that lane (required for deterministic replay when one engine
        owns several lanes).  The per-lane lifecycle:

        1. A lane's state is lazily initialized on its first record via init_state()
        2. push(state, items) is called with one lane's records -> (new_state, outputs)
        3. If should_flush returns True, that lane's flush runs and its state resets
        4. On stream end (all lanes) or a lane's epoch boundary, flush() emits that
           lane's remaining buffered items
        5. Each output item goes through transform (if provided) in parallel

        Args:
            name: Operator name for debugging/metrics.
            init_state: Factory that creates initial state (called lazily per lane).
            push: Called with (lane_state, one lane's records) -> (new_state,
                outputs).  Outputs are emitted immediately; state carries forward
                for that lane.
            flush: Optional. Emits a lane's remaining buffered items at
                end-of-stream.  In non-monotonic pipelines, also called per lane
                mid-stream at that lane's epoch boundary; the lane's state is
                re-initialized afterward.
            should_flush: Optional. If returns True, triggers early flush and state reset.
            transform: Optional. Batch-level transform that runs in parallel workers.
                Receives the full batch from the accumulator, preserving batch structure
                for efficient GPU processing, vectorized ops, etc.
            placement: Placement hint for this operator.
            parallelism: Worker parallelism. Use >1 when transform is expensive.
            indexable: Whether this operator preserves indexability (default False).
                Set True only if the transform is 1:1 and deterministic.
            preserves_cursor_order: Whether outputs maintain monotone cursor order.
                Set False when the transform reorders or packs items (e.g.,
                shuffle, bin-packing). This affects which chunk-eviction path
                the engine uses for the entire plan.

        Returns:
            Self for method chaining.

        Example - Custom batching by token count:
            >>> def accumulate_by_tokens(state, items, max_tokens=4096):
            ...     buffer, token_count = state["buffer"], state["tokens"]
            ...     outputs = []
            ...     for item in items:
            ...         item_tokens = len(item.payload["token_ids"])
            ...         if token_count + item_tokens > max_tokens and buffer:
            ...             outputs.extend(buffer)
            ...             buffer, token_count = [], 0
            ...         buffer.append(item)
            ...         token_count += item_tokens
            ...     return {"buffer": buffer, "tokens": token_count}, outputs
            ...
            >>> pipeline.stateful_transform(
            ...     "batch_by_tokens",
            ...     init_state=lambda: {"buffer": [], "tokens": 0},
            ...     push=lambda s, items: accumulate_by_tokens(s, items),
            ...     flush=lambda s: s["buffer"] if s["buffer"] else [],
            ... )

        Example - Deduplicate (serial) then batch encode (parallel):
            >>> def batch_encode(records):
            ...     # Process whole batch efficiently (e.g., GPU batching)
            ...     for r in records:
            ...         r.payload["encoded"] = encode(r.payload["text"])
            ...     return records
            ...
            >>> pipeline.stateful_transform(
            ...     "dedupe_and_encode",
            ...     init_state=lambda: set(),
            ...     push=lambda seen, items: (
            ...         seen | {i.payload["id"] for i in items},
            ...         [i for i in items if i.payload["id"] not in seen]
            ...     ),
            ...     transform=batch_encode,  # processes whole batch in parallel
            ...     parallelism=8,
            ... )
        """
        from zephon._internal.ops.stateful_transform import StatefulTransformOp

        op = StatefulTransformOp(
            init_state=init_state,
            push_fn=push,
            flush_fn=flush,
            should_flush_fn=should_flush,
            transform_fn=transform,
            parallelism=parallelism,
            indexable=indexable,
            preserves_cursor_order=preserves_cursor_order,
        )
        node = self._graph.add(
            name,
            op,
            self._tail,
            placement=placement,
            parallelism=parallelism,
        )
        self._tail = node
        return self

    @overload
    def add_op(
        self,
        op: BaseOp,
        /,
        *,
        name: Optional[str] = None,
        placement: str = "auto",
    ) -> "Pipeline": ...

    @overload
    def add_op(
        self,
        name: str,
        /,
        *,
        process_many: Callable[[list[Any]], list[Any]],
        preserves_cursor_order: bool,
        accumulator: Optional[Callable[..., "Accumulator[Any]"]] = None,
        process_one: Optional[Callable[[Any], list[Any]]] = None,
        validation_samples: Optional[Callable[[], list[SampleRecord]]] = None,
        parallelism: int = 1,
        placement: str = "auto",
        indexable: bool = False,
        batch_shape_sensitive: bool = False,
        requires_serial_state: bool = False,
    ) -> "Pipeline": ...

    @_mutates_graph
    def add_op(
        self,
        name_or_op: Union[str, BaseOp],
        /,
        *,
        name: Optional[str] = None,
        process_many: Optional[Callable[[list[Any]], list[Any]]] = None,
        preserves_cursor_order: Optional[bool] = None,
        accumulator: Optional[Callable[..., "Accumulator[Any]"]] = None,
        process_one: Optional[Callable[[Any], list[Any]]] = None,
        validation_samples: Optional[Callable[[], list[SampleRecord]]] = None,
        parallelism: Optional[int] = None,
        placement: str = "auto",
        indexable: Optional[bool] = None,
        batch_shape_sensitive: Optional[bool] = None,
        requires_serial_state: Optional[bool] = None,
    ) -> "Pipeline":
        """Append a custom operator — two forms.

        **Instance form**: ``pipe.add_op(op, name=..., placement=...)``.
        Pass a prebuilt :class:`BaseOp` subclass instance.  Use this when
        the op needs full lifecycle control — ``__init__`` for
        configuration, ``setup`` for per-worker resource construction
        (tokenizers, model handles) and :class:`OpContext` services,
        ``traits`` for any combination of `OpTraits` fields.  The
        framework deep-copies the instance per parallel worker, so
        ``self.*`` attributes set in ``setup`` are isolated per worker
        on every runner (including the thread runner).  This matches
        the lifecycle built-in operators use.  Traits come from
        ``op.traits()`` rather than kwargs.

        **Kwargs form**: ``pipe.add_op(name, *, process_many=..., preserves_cursor_order=..., ...)``.
        Convenience for stateless transforms.  The framework builds an
        internal `BaseOp` subclass from the supplied callables +
        accumulator factory + traits.  ``process_many`` must not carry
        state between calls; any per-lane or cross-invocation state
        belongs in the accumulator.  Read-only shared resources (codec
        tables, thresholds) can be captured in a closure over the
        callables.  See the "Accumulators and Operators" page in the
        documentation for the operator/accumulator split.

        Args:
            op: Instance form. The :class:`BaseOp` instance to attach.
            name: Operator name used in plan graphs, metrics, and logs.
                The instance form defaults to the class name of ``op``.
            placement: Placement hint passed to the planner (``"auto"``,
                ``"local"``, or a runner-specific tag).
            process_many: Kwargs form, required. Callable applied to each ready batch
                from the accumulator. Receives a list of upstream items,
                returns the list of downstream items. Runs in parallel
                workers when ``parallelism > 1``.
            accumulator: Kwargs form. Optional factory returning a fresh `Accumulator`
                instance. The framework calls the factory once at runner
                setup and again on ``reset_buffers`` between runs, so a
                factory (rather than an instance) is required for
                correctness. Two signatures are supported and auto-detected:

                - ``Callable[[], Accumulator]`` — simplest form. Use
                  ``lambda: CountingAccumulator(max_batch=N)`` for
                  size-based per-lane batching (it always groups by lane).
                - ``Callable[*, deterministic, ctx], Accumulator]`` —
                  for accumulators whose construction depends on the
                  deterministic mode (e.g. enabling latency-based flush
                  only when ``deterministic`` is False) or runtime
                  context. Example::

                      accumulator=lambda *, deterministic, ctx: (
                          CountingAccumulator(
                              max_batch=N,
                              max_latency_ms=None if deterministic else 3,
                          )
                      )

                Defaults to a `PassthroughAccumulator` factory — each
                upstream micro-batch becomes one ready batch as-is.
            process_one: Kwargs form. Optional fast path for single-element
                processing.
                If omitted, the framework wraps each element in a list and
                routes it through ``process_many``.
            validation_samples: Kwargs form. Optional factory returning a list of
                :class:`~zephon.types.SampleRecord` instances for the
                validation harness.  Override when ``process_many``
                requires payload fields beyond the validator's synthetic
                ``{'text': str, 'value': int}`` — letting the harness
                run the full op-level + state-diff suite instead of
                degrading to ``OP_REJECTS_GENERIC_PAYLOAD``.  Buggy
                factories surface ``OP_VALIDATION_SAMPLES_FACTORY_FAILED``
                and the validator falls back to synthetic records.
                Include at least two distinct ``lane_id`` values so the
                cross-call-state probe stays meaningful.
            parallelism: Kwargs form. Number of worker invocations to run in
                parallel for this op.  Default 1 (serial).  Increase when
                ``process_many`` is CPU/GPU-bound.
            preserves_cursor_order: Kwargs form, required. True when ``process_many``
                emits records whose ``chunk_id`` order matches their
                inputs (1:1 maps, payload transforms, non-reordering
                filters). False when the op reorders, shuffles, or
                packs. The planner picks a different eviction strategy
                based on this trait — getting it wrong corrupts
                checkpoint semantics silently. See the Checkpointing
                page in the documentation.
            indexable: Kwargs form. Whether the op preserves indexability through the
                plan.  Default False; set True only if the transform is
                1:1 and deterministic.
            batch_shape_sensitive: Kwargs form. Set True when ``process_many``'s output
                can depend on how inputs are grouped into micro-batches
                (per-batch RNG, statistics, etc.). In deterministic mode
                this disables latency-based accumulator flushing for
                stages containing this op, preserving strong determinism
                at the cost of some throughput.
            requires_serial_state: Kwargs form. Set True when the op's accumulator
                holds cross-invocation state that cannot be sharded
                across parallel worker instances. In deterministic mode
                the planner pins ``parallelism=1`` for ops with this
                trait.

        Returns:
            Self for method chaining.

        Examples::

            # Instance form — class with setup/traits/accumulator overrides.
            class Tokenize(BaseOp):
                def __init__(self, name):
                    super().__init__()
                    self._name = name
                    self._tokenizer = None
                def traits(self):
                    return OpTraits(preserves_cursor_order=True, parallelism=4)
                def setup(self, ctx):
                    super().setup(ctx)
                    self._tokenizer = load_tokenizer(self._name)
                def process_many(self, elems):
                    return [self._tokenizer.encode(e) for e in elems]

            pipeline.add_op(Tokenize("gpt2"))

            # Kwargs form — stateless windowed transform.
            from zephon.ops import CountingAccumulator
            pipeline.add_op(
                "windowed_transform",
                process_many=lambda elems: [...],
                accumulator=lambda: CountingAccumulator(max_batch=64),
                parallelism=4,
                preserves_cursor_order=True,
            )
        """
        if isinstance(name_or_op, BaseOp):
            mismatched = [
                kw
                for kw, val in (
                    ("process_many", process_many),
                    ("preserves_cursor_order", preserves_cursor_order),
                    ("accumulator", accumulator),
                    ("process_one", process_one),
                    ("validation_samples", validation_samples),
                    ("parallelism", parallelism),
                    ("indexable", indexable),
                    ("batch_shape_sensitive", batch_shape_sensitive),
                    ("requires_serial_state", requires_serial_state),
                )
                if val is not None
            ]
            if mismatched:
                raise ValueError(
                    "add_op(op) with a BaseOp instance does not accept "
                    f"{', '.join(mismatched)}; traits, callables, and "
                    "validation_samples come from the op itself "
                    "(override `traits()` / `validation_samples()` on "
                    "your subclass).  Use the kwargs form add_op(name, "
                    "...) instead if you want to supply callables."
                )
            instance_op = name_or_op
            node_name = name if name is not None else type(instance_op).__name__
            if not node_name:
                raise ValueError("add_op() requires a non-empty name")
            node = self._graph.add(
                node_name, instance_op, self._tail, placement=placement
            )
            self._tail = node
            return self

        op_name = name_or_op
        if name is not None:
            raise ValueError(
                "add_op(name, ...) takes the op name positionally; the "
                f"`name=` keyword is only for the instance form. Got positional "
                f"{op_name!r} and name={name!r} — drop the `name=` kwarg."
            )
        if not op_name:
            raise ValueError("add_op() requires a non-empty name")
        if process_many is None:
            raise ValueError("add_op(name, ...) requires `process_many`")
        if preserves_cursor_order is None:
            raise ValueError(
                "add_op(name, ...) requires `preserves_cursor_order` "
                "(True for 1:1 maps and non-reordering filters; False for "
                "reorder/shuffle/pack ops)"
            )

        functional_op = _FunctionalOp(
            process_many_fn=process_many,
            accumulator_factory=(
                accumulator if accumulator is not None else PassthroughAccumulator
            ),
            process_one_fn=process_one,
            validation_samples_factory=validation_samples,
            op_traits=OpTraits(
                preserves_cursor_order=preserves_cursor_order,
                parallelism=1 if parallelism is None else parallelism,
                indexable=False if indexable is None else indexable,
                batch_shape_sensitive=(
                    False if batch_shape_sensitive is None else batch_shape_sensitive
                ),
                requires_serial_state=(
                    False if requires_serial_state is None else requires_serial_state
                ),
            ),
        )
        node = self._graph.add(
            op_name,
            functional_op,
            self._tail,
            placement=placement,
            parallelism=parallelism,
        )
        self._tail = node
        return self

    @_mutates_graph
    def tokenize(
        self,
        tokenizer: Any | None = None,
        tokenizer_id: str | None = None,
        *,
        field: str,
        missing_field: MissingFieldMode = "error",
        add_attention_mask: bool = True,
        max_length: int | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        return_tensors: str | None = None,
        split_long_samples: bool = False,
        use_fast: bool | None = True,
        preserve_upstream_payload: bool = False,
        special_tokens: SpecialTokensMode = "bos_eos",
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> "Pipeline":
        op = TokenizeText(
            tokenizer,
            tokenizer_id,
            field=field,
            missing_field=missing_field,
            add_attention_mask=add_attention_mask,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            split_long_samples=split_long_samples,
            use_fast=use_fast,
            preserve_upstream_payload=preserve_upstream_payload,
            special_tokens=special_tokens,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        node = self._graph.add(
            "tokenize", op, self._tail, placement=placement, parallelism=parallelism
        )
        self._tail = node
        return self

    @_mutates_graph
    def tokenize_chat(
        self,
        tokenizer: Any | None = None,
        tokenizer_id: str | None = None,
        *,
        eos_token: str | None = None,
        field: str = "messages",
        max_length: int | None = None,
        chat_template: str | os.PathLike[str] | None = None,
        apply_chat_template: bool = True,
        span_source: SpanSource = "auto",
        loss_on_last_turn_only: bool = False,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        tools_field: str = "tools",
        enable_thinking_field: str = "enable_thinking",
        mask_field_out: str = "loss_mask",
        preserve_upstream_payload: bool = False,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> "Pipeline":
        """Tokenize chat conversations into ids plus a loss mask.

        Full contract: one-shot
        render+tokenize with assistant spans from ``{% generation %}`` tags,
        a guarded prefix-diff fallback for untagged templates, or exact spans
        in the no-template path (``apply_chat_template=False``).
        """
        op = TokenizeChat(
            tokenizer,
            tokenizer_id,
            eos_token=eos_token,
            field=field,
            max_length=max_length,
            chat_template=chat_template,
            apply_chat_template=apply_chat_template,
            span_source=span_source,
            loss_on_last_turn_only=loss_on_last_turn_only,
            chat_template_kwargs=chat_template_kwargs,
            tools_field=tools_field,
            enable_thinking_field=enable_thinking_field,
            mask_field_out=mask_field_out,
            preserve_upstream_payload=preserve_upstream_payload,
        )
        node = self._graph.add(
            "tokenize_chat",
            op,
            self._tail,
            placement=placement,
            parallelism=parallelism,
        )
        self._tail = node
        return self

    @_mutates_graph
    def shuffle(
        self,
        buffer_size: int | None = None,
        *,
        seed: int = 0,
        algorithm: Literal["streaming", "block", "block_warmup"] = "streaming",
        warmup_growth: float = 1.5,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> "Pipeline":
        """Insert a deterministic shuffle buffer.

        Args:
            buffer_size: Number of samples to buffer for shuffling. If None,
                a default of 8192 is used.
            seed: RNG seed for the shuffle.
            algorithm: ``"streaming"`` uses a deterministic reservoir (default).
                ``"block"`` uses legacy tumbling blocks. ``"block_warmup"`` uses
                tumbling blocks with a post-flush block-size ramp.
            warmup_growth: Multiplicative block-size growth per step for
                ``algorithm="block_warmup"`` (ignored otherwise). Default 1.5;
                lower values ramp more smoothly and reach the full window later.
            placement: Placement hint for this operator.
            parallelism: Override default parallelism for this operator.

        Returns:
            Self for method chaining.
        """
        op = ShuffleBuffer(
            buffer_size=buffer_size,
            seed=seed,
            algorithm=algorithm,
            warmup_growth=warmup_growth,
        )
        node = self._graph.add(
            "shuffle_buffer",
            op,
            self._tail,
            placement=placement,
            parallelism=parallelism,
        )
        self._tail = node
        return self

    @_mutates_graph
    def ensure_mixture(
        self,
        *,
        max_buffer_size: Optional[int] = 1000,
        drain_target_ratio: float = 0.8,
        obsolete_drain_rate: float = 0.1,
        weight_by: Union[
            Callable[[SampleRecord], float], Literal["samples", "auto"], str
        ] = "auto",
        warn_tolerance: Optional[float] = None,
        mixture: Optional[dict[str, float]] = None,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> "Pipeline":
        """Enforce mixture ratios using adaptive Smooth Weighted Round Robin.

        Place this operator after tokenization (for token-level) or after filtering
        (for sample-level). The operator automatically detects token fields from
        common names (input_ids, tokens, token_ids, ids).

        Uses adaptive buffering: emit immediately when SWRR's ideal component is
        available, buffer when the desired component isn't present yet. Falls back
        to emitting best-available when max_buffer_size is reached.

        The SWRR algorithm tracks deficit (target - actual) and always picks the
        component most "owed" samples, ensuring smooth, deterministic convergence.

        Args:
            max_buffer_size: Samples to hold while waiting for the component the
                target mixture needs next. Default 1000; when the buffer fills, the
                operator emits what it has, so the output mixture may drift if a
                component stays scarce. Pass ``None`` to instead **drop** the surplus
                it cannot place on-target — at every epoch boundary and at end of
                stream. This makes the mixture exact but discards data, so use it only
                when an exact mixture matters more than keeping every sample. How much
                is dropped depends on the stream's skew and on ``flush_every_k_chunks``
                (smaller → more dropped). ensure_mixture is always non-monotonic, so
                that cadence defaults to 8; in strict mode the buffer is then discarded
                on roughly every flush. If drops are too frequent, raise
                ``flush_every_k_chunks`` above the default to give scarce components
                more time to absorb the buffer (at the cost of holding more in memory
                between flushes).
            drain_target_ratio: When forced to emit (buffer hits max_buffer_size), drain
                the buffer down to this fraction of max_buffer_size before stopping.
                Default is 0.8 (drain to 80% of max).  Ignored when ``max_buffer_size=None``.
            obsolete_drain_rate: Fraction of emissions reserved for draining obsolete
                components (those no longer in the current mixture target). Default is
                0.1 (10%), meaning 1 in every 10 emissions drains an obsolete sample.
            weight_by: How to compute sample weights. Options:

                - ``auto`` (default): Auto-detect token field from common names
                  (input_ids, tokens, token_ids, ids). Raises if not found.
                - ``samples``: Each sample has weight 1.
                - Explicit field name (e.g., ``input_ids``): Use that field's length.
                - Callable: Custom function taking SampleRecord, returning float.
            warn_tolerance: If set, warn when mixture drift exceeds this value (0.05 = ±5%).
                If None (default), no warnings are emitted.
            mixture: Explicit mixture target {component_name: float}. If None, derived
                from chunk mixture via engine context.
            placement: Placement hint for this operator.
            parallelism: Override default parallelism for this operator.

        Returns:
            Self for method chaining.

        Examples:
            ::

                # Token-level (default) - place after tokenize
                pipeline.fetch().tokenize(tokenizer_id="gpt2", field="text").ensure_mixture()

                # Sample-level enforcement (after filter)
                pipeline.fetch().filter(...).ensure_mixture(weight_by="samples")

                # With explicit token field
                pipeline.fetch().tokenize(tokenizer_id="gpt2", field="text").ensure_mixture(
                    weight_by="input_ids"
                )

                # With warnings for drift (warn if >5% deviation)
                pipeline.fetch().tokenize(tokenizer_id="gpt2", field="text").ensure_mixture(
                    warn_tolerance=0.05
                )

                # Explicit mixture target (override chunk mixture)
                pipeline.fetch().tokenize(tokenizer_id="gpt2", field="text").ensure_mixture(
                    mixture={"English": 0.7, "German": 0.3}
                )
        """
        from zephon._internal.ops.ensure_mixture import EnsureMixture

        op = EnsureMixture(
            max_buffer_size=max_buffer_size,
            drain_target_ratio=drain_target_ratio,
            obsolete_drain_rate=obsolete_drain_rate,
            weight_by=weight_by,
            warn_tolerance=warn_tolerance,
            mixture_override=mixture,
            parallelism=parallelism or 1,
        )
        node = self._graph.add(
            "ensure_mixture",
            op,
            self._tail,
            placement=placement,
            parallelism=parallelism,
        )
        self._tail = node
        return self

    @_mutates_graph
    def materialize(
        self, placement: str = "auto", parallelism: Optional[int] = None
    ) -> "Pipeline":
        node = self._graph.add(
            "materialize",
            Materialize(),
            self._tail,
            placement=placement,
            parallelism=parallelism,
        )
        self._tail = node
        return self

    @_mutates_graph
    def batch(
        self,
        microbatch_size: int,
        *,
        drop_last: bool = True,
        placement: str = "auto",
    ) -> "Pipeline":
        op = Batch(microbatch_size, drop_last=drop_last)
        node = self._graph.add("batch", op, self._tail, placement=placement)
        self._tail = node
        return self

    def _resolve_pack_groups(
        self,
        homogeneity: Literal["none", "group", "full"],
        groups: DomainGroups | Mapping[str, Sequence[str]] | None,
    ) -> DomainGroups | None:
        """Coerce/validate the packing ``groups`` argument.

        Rejects a homogeneity/groups mismatch, requires groups in group mode, and
        validates members against the source's component vocabulary — the same
        names packing resolves records to at runtime.
        """
        if groups is not None and homogeneity != "group":
            raise ValueError(
                f"groups is only valid with homogeneity='group', not {homogeneity!r}."
            )
        if groups is None:
            if homogeneity == "group":
                raise ValueError(
                    "homogeneity='group' requires groups (pass groups=...)."
                )
            return None
        spec = groups if isinstance(groups, DomainGroups) else DomainGroups(groups)
        # component_ids() is the vocabulary packing resolves members against at
        # runtime; ``dataset_ids`` is a StaticMixture-only property.
        components = self.ws.component_ids()
        spec.validate_against(components)
        # Ungrouped components pack as their own singleton domains by design;
        # warn so an accidentally-omitted member isn't silently self-grouped.
        ungrouped = sorted(set(components) - set(spec.to_member_map()))
        if ungrouped:
            warnings.warn(
                f"homogeneity='group': components {ungrouped} are in no group and "
                "will pack as their own singleton domains; add them to a group if "
                "that was unintended.",
                stacklevel=4,  # warn -> _resolve_pack_groups -> pack_* -> @_mutates_graph -> caller
            )
        return spec

    @_mutates_graph
    def pack_sequences(
        self,
        max_length: int,
        *,
        num_bins: int | None = None,
        max_sequences_per_bin: int | None = None,
        algorithm: PackingAlgorithm = "first_fit",
        tokens_field: str = "auto",
        length_fn: Callable[[SampleRecord], int] | None = None,
        drop_oversized: bool | None = None,
        min_sequence_length: int = 1,
        shuffle_strategy: Literal["random", "length", None] = None,
        shuffle_seed: Optional[int] = None,
        flush_strategy: Literal["fifo", "fullest"] = "fifo",
        pack_payloads: str | Callable[[list[Any]], Any] = "keep_list",
        candidate_pool_size: int | None = None,
        max_candidate_age: int | None = None,
        homogeneity: Literal["none", "group", "full"] = "none",
        groups: DomainGroups | Mapping[str, Sequence[str]] | None = None,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> "Pipeline":
        """Add a sequence-packing operator that preserves segment boundaries.

        Each bin is emitted as ``{"packed_samples": [seg0, seg1, ...]}`` — an
        ordered list of constituent records (first_fit/best_fit) or record slices
        (wrap/best_fit_wrap). Each list element remains a distinct segment;
        buffered algorithms may reorder segments, and best_fit_wrap may emit a
        suffix before its remaining prefix. ``pack_payloads`` merges that list
        (default keeps it as-is). For flat, tensor-ready training records with
        ``positions``, use :meth:`pack_flat` instead.

        Args:
            max_length: Maximum length for packed bins.
            max_sequences_per_bin: Optional positive bound on segments per bin;
                split fragments count once per bin. Capped wrap emits partial
                bins, including the final remainder.
            num_bins: Number of bins to maintain per packing group. Required for
                first_fit/best_fit and not allowed for wrapping algorithms.
            algorithm: ``"first_fit"`` (default), ``"best_fit"``, ``"wrap"``, or
                ``"best_fit_wrap"`` (candidate-based buffered wrapping with
                length-based selection and at most one split per bin).
            tokens_field: Token field to slice (``"auto"`` or an explicit name);
                consumed by wrap/best_fit_wrap. first/best keep whole payloads.
            length_fn: Optional callable measuring packing length, for
                first_fit/best_fit only (e.g. a precomputed ``length`` field with
                no token field to slice). ``None`` measures
                ``len(payload[tokens_field])``. Not allowed with wrap or
                best_fit_wrap (length is the sliced field's length).
            drop_oversized: Whether first/best should drop records longer than
                ``max_length``. Defaults to True for first/best and False for
                wrap/best_fit_wrap, which split instead.
            min_sequence_length: Remaining capacity below which a bin is emitted.
            shuffle_strategy: Strategy for ordering sequences before packing
                ("random", "length", or None).
            shuffle_seed: Seed for random shuffling when shuffle_strategy="random".
            flush_strategy: "fifo" (default) flushes oldest bins first; "fullest"
                flushes bins with the smallest remaining capacity first.
            pack_payloads: How to merge the segment list. "keep_list" (default),
                "torch_tensor", "numpy_array", or a custom callable taking list[Any].
            candidate_pool_size: best_fit_wrap only — candidate lookahead per
                packing group. May be exceeded until the pool contains
                ``max_length`` tokens or reaches the segment cap. Defaults to 1024.
            max_candidate_age: best_fit_wrap only — candidate arrivals before a
                still-buffered record is force-placed. Defaults to
                ``8 * candidate_pool_size``.
            homogeneity: If ``"full"``, each packed sample stays within a single
                mixing domain (mixture component); ``"group"`` keeps it within a
                single ``groups`` group of domains; ``"none"`` (default) mixes
                freely. See :meth:`PackSequences.__init__`.
            groups: Required for ``homogeneity="group"`` — a
                :class:`zephon.ops.DomainGroups` or ``{group: [component, ...]}``
                mapping naming which mixing domains may share a packed sample.
                Members are mixture-component names (the source's
                ``component_ids``), validated against them at build time.
            placement: Placement strategy for this operator.
            parallelism: Worker count for materializing packed-bin payloads.
                Bin assignment remains serial, so output is unchanged.
        """
        op = PackSequences(
            max_length=max_length,
            max_sequences_per_bin=max_sequences_per_bin,
            num_bins=num_bins,
            length_fn=length_fn,
            algorithm=algorithm,
            output="envelope",
            tokens_field=tokens_field,
            drop_oversized=drop_oversized,
            min_sequence_length=min_sequence_length,
            shuffle_strategy=shuffle_strategy,
            shuffle_seed=shuffle_seed,
            flush_strategy=flush_strategy,
            pack_payloads=pack_payloads,
            candidate_pool_size=candidate_pool_size,
            max_candidate_age=max_candidate_age,
            homogeneity=homogeneity,
            groups=self._resolve_pack_groups(homogeneity, groups),
        )
        node = self._graph.add(
            "pack_sequences",
            op,
            self._tail,
            placement=placement,
            parallelism=parallelism,
        )
        self._tail = node
        return self

    @_mutates_graph
    def pack_flat(
        self,
        max_length: int,
        *,
        num_bins: int | None = None,
        max_sequences_per_bin: int | None = None,
        algorithm: PackingAlgorithm = "first_fit",
        tokens_field: str = "auto",
        pad_token_id: Optional[int] = None,
        emit_positions: bool = True,
        drop_oversized: bool | None = None,
        min_sequence_length: int = 1,
        shuffle_strategy: Literal["random", "length", None] = None,
        shuffle_seed: Optional[int] = None,
        flush_strategy: Literal["fifo", "fullest"] = "fifo",
        candidate_pool_size: int | None = None,
        max_candidate_age: int | None = None,
        homogeneity: Literal["none", "group", "full"] = "none",
        groups: DomainGroups | Mapping[str, Sequence[str]] | None = None,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> "Pipeline":
        """Add a sequence-packing operator that emits flat training records.

        Each bin is emitted flat as ``{tokens_field: concat[+pad], "positions"?}``
        — no ``packed_samples`` — so ``SampleBatch.to_training`` consumes it
        directly (``positions`` is surfaced automatically). Uncapped ``wrap``
        emits only full bins. Capped wrap and the other algorithms pad partial
        bins, including their final remainder, to ``max_length`` with
        ``pad_token_id``.
        Buffered algorithms may reorder segments, and best_fit_wrap may emit a
        suffix before its remaining prefix. When ``emit_positions=True``, positions
        still reset at every emitted segment; they do not restore source order.
        Only the token field and length-aligned sliceable fields survive; scalar
        and non-aligned payload is dropped (use :meth:`pack_sequences` to keep it).

        Args:
            max_length: Fixed length of every emitted record.
            max_sequences_per_bin: Optional positive bound on real segments per
                output bin, excluding padding. Split fragments count once per bin.
            num_bins: Number of bins to maintain per packing group. Required for
                first_fit/best_fit and not allowed for wrapping algorithms.
            algorithm: ``"first_fit"`` (default), ``"best_fit"``, ``"wrap"``, or
                ``"best_fit_wrap"`` (candidate-based buffered wrapping with
                length-based selection and at most one split per bin).
            tokens_field: Token field to concatenate (``"auto"`` or explicit name).
            pad_token_id: Fill value for the token field when padding partial
                bins (aligned fields pad with 0). Required for first_fit,
                best_fit, best_fit_wrap, and capped wrap; unused for uncapped
                wrap. Any embeddable id works —
                ``to_training`` masks the pad tail from the loss by position
                (``meta.padding_length``), not by id.
            emit_positions: Include the ``positions`` array marking document
                boundaries (``cumsum(positions == 0) - 1`` → doc ids). Defaults to
                True; set False for classic concatenated blocks.
            drop_oversized: Whether first/best should drop records longer than
                ``max_length``. Defaults to True for first/best and False for
                wrap/best_fit_wrap, which split instead.
            min_sequence_length: Remaining capacity below which a bin is emitted.
            shuffle_strategy: Strategy for ordering sequences before packing.
            shuffle_seed: Seed for random shuffling when shuffle_strategy="random".
            flush_strategy: "fifo" (default) or "fullest" when num_bins is reached.
            candidate_pool_size: best_fit_wrap only — candidate lookahead per
                packing group. May be exceeded until the pool contains
                ``max_length`` tokens or reaches the segment cap. Defaults to 1024.
            max_candidate_age: best_fit_wrap only — candidate arrivals before a
                still-buffered record is force-placed. Defaults to
                ``8 * candidate_pool_size``.
            homogeneity: If ``"full"``, each packed sample stays within a single
                mixing domain (mixture component); ``"group"`` keeps it within a
                single ``groups`` group of domains; ``"none"`` (default) mixes
                freely. See :meth:`PackSequences.__init__`.
            groups: Required for ``homogeneity="group"`` — a
                :class:`zephon.ops.DomainGroups` or ``{group: [component, ...]}``
                mapping naming which mixing domains may share a packed sample.
                Members are mixture-component names (the source's
                ``component_ids``), validated against them at build time.
            placement: Placement strategy for this operator.
            parallelism: Worker count for materializing packed-bin payloads.
                Bin assignment remains serial, so output is unchanged.
        """
        op = PackSequences(
            max_length=max_length,
            max_sequences_per_bin=max_sequences_per_bin,
            num_bins=num_bins,
            algorithm=algorithm,
            output="flat",
            tokens_field=tokens_field,
            drop_oversized=drop_oversized,
            min_sequence_length=min_sequence_length,
            shuffle_strategy=shuffle_strategy,
            shuffle_seed=shuffle_seed,
            flush_strategy=flush_strategy,
            emit_positions=emit_positions,
            pad_token_id=pad_token_id,
            candidate_pool_size=candidate_pool_size,
            max_candidate_age=max_candidate_age,
            homogeneity=homogeneity,
            groups=self._resolve_pack_groups(homogeneity, groups),
        )
        node = self._graph.add(
            "pack_flat", op, self._tail, placement=placement, parallelism=parallelism
        )
        self._tail = node
        return self

    def enable_observability(
        self,
        tracking: ExecutionTrackingMode | str = ExecutionTrackingMode.NODES,
        *,
        sink: MetricsSinkConfig | None = None,
    ) -> "Pipeline":
        """Enable runtime metrics collection for the compiled pipeline."""
        mode = ExecutionTrackingMode.from_value(tracking)
        self._options.execution_tracking = mode
        if sink is None:
            sink = MetricsSinkConfig()
        self._options.metrics_sink_config = sink
        return self

    # Internal/testing helper: insert a small deterministic delay stage.
    @_mutates_graph
    def _delay(
        self,
        *,
        max_delay_ms: float = 2.0,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> "Pipeline":
        from zephon._internal.ops.delay import (
            DelayById,
        )  # lazy import; not part of public __all__

        op = DelayById(max_delay_ms=max_delay_ms)
        node = self._graph.add(
            "delay",
            op,
            self._tail,
            placement=placement,
            parallelism=parallelism,
        )
        self._tail = node
        return self

    def options(self, **hints: Any) -> "Pipeline":
        # TODO(MaxiBoether): Support in addition to dict options just typed options using dataclasses.
        for key, value in hints.items():
            if not hasattr(self._options, key):
                warnings.warn(
                    f"Unknown pipeline option '{key}' ignored. "
                    f"Valid options are attributes of RuntimeOptions.",
                    stacklevel=2,
                )
                continue
            if key == "io_options":
                new_opts = StoreOptions.from_any(value)
                self._options.io_options = self._options.io_options.merge(new_opts)
            else:
                setattr(self._options, key, value)
        # Invalidate cached RuntimeSpec — options may have changed runner
        # selection, parallelism, or other compile-time decisions.
        self._runtime_spec = None
        # Invalidate the validation cache: toggling auto_validation
        # between "off" and "strict"/"warn" needs to re-run the harness
        self._validated = False
        return self

    def _invalidate_plan(self) -> None:
        """Invalidate cached plan and derived state when the graph is mutated."""
        self._plan = None
        self._runtime_spec = None
        self._validated = False
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def _ensure_plan(self) -> None:
        """Build Plan only (cheap graph analysis, no Engine)."""
        if self._plan is None:
            self._plan = Planner().make_plan(self._graph)

    def _compile(self) -> RuntimeSpec:
        """Compile pipeline into a RuntimeSpec. Cheap, no Engine construction.

        Always re-evaluates ``inside_torch_worker()`` because the worker
        context can change between calls (e.g. parent process vs forked
        DataLoader worker).  ``resolve_runtime_spec`` is a pure function
        with negligible cost — no I/O or resource allocation.
        """
        from zephon._internal.engine import inside_torch_worker

        self._ensure_plan()
        assert self._plan is not None
        self._runtime_spec = resolve_runtime_spec(
            self._plan,
            self._options,
            inside_worker=inside_torch_worker(),
        )
        return self._runtime_spec

    def _ensure(self) -> None:
        spec = self._compile()
        assert self._plan is not None
        if self._engine is None or self._engine._closed:
            self._engine = Engine(self._plan, self._options, self.ws, spec)

    def _iter_ops(self, op_type: type[_OpT]) -> Iterator[tuple[int, _OpT]]:
        for index, node in enumerate(self._graph.nodes):
            if isinstance(node.op, op_type):
                yield index, node.op

    def _find_op(self, op_type: type[_OpT]) -> tuple[int, _OpT] | None:
        return next(self._iter_ops(op_type), None)

    def _validate_token_mixture_graph(self) -> tuple[int, TokenizeBase] | None:
        """Validate token-mode placement and return the first tokenize op."""
        tokenizes = list(self._iter_ops(TokenizeBase))
        tokenize = tokenizes[0] if tokenizes else None
        if len(tokenizes) > 1:
            warnings.warn(
                f"token-aware mixture priming calibrates with the first of "
                f"{len(tokenizes)} tokenize ops; if a later one determines "
                f"delivered token counts, pin ratios via "
                f"TokenEstimation(primer=...).",
                RuntimeWarning,
                stacklevel=2,
            )
        tokenize_index = tokenize[0] if tokenize is not None else None
        for index, op in self._iter_ops(EnsureMixture):
            if op.weight_by == "samples":
                raise ValueError(
                    "token-aware mixtures require ensure_mixture to weigh in "
                    "token units, but weight_by='samples' was configured. The "
                    "work source deliberately emits a token-balanced (sample-"
                    "skewed) stream; enforcing the token target in sample "
                    "units would fight it. Use weight_by='auto' (default), a "
                    "token field name, or a token-counting callable."
                )
            if tokenize_index is not None and index < tokenize_index:
                raise ValueError(
                    "token-aware mixtures require ensure_mixture to run "
                    "after tokenize: before tokenization there are no token "
                    "counts, so the operator would enforce the token target "
                    "in sample units against a deliberately sample-skewed "
                    "stream."
                )
        return tokenize

    def _build_pre_tokenize_replay(
        self, tokenize_index: int
    ) -> _PreTokenizeReplay | _UnreplayableOp | None:
        """Collect the fetch -> tokenize ops calibration must replay.

        Calibration fetches raw store rows; ops ahead of the tokenize op
        can change what it counts. An op outside the replayable allowlist
        yields a marker that priming rejects if it actually measures.
        """
        fetch_index = self._graph.nodes.index(self._fetch_node)
        ops: list[Any] = []
        for node in self._graph.nodes[fetch_index + 1 : tokenize_index]:
            if not isinstance(node.op, _PRE_TOKENIZE_REPLAYABLE_OPS):
                return _UnreplayableOp(type(node.op).__name__)
            ops.append(node.op)
        return _PreTokenizeReplay(ops) if ops else None

    def _prime_worksource(self) -> None:
        """Prime token-aware work sources before worker pickling.

        Restores keep checkpointed ratios instead of remeasuring, so a
        tokenizer or data change cannot silently shift the mixture.
        """
        if not self.ws.requires_token_priming:
            return
        # Restores still need graph validation even though ratios come from
        # the checkpoint.
        tokenize = self._validate_token_mixture_graph()
        if self._pending_restore is not None:
            return
        if self._engine is not None and not self._engine._closed:
            # After restore, the live clones already carry checkpointed ratios.
            return
        self.ws.prime(
            io_options=self._options.io_options,
            counting_spec=(
                tokenize[1].token_counting_spec() if tokenize is not None else None
            ),
            pre_tokenize_replay=(
                self._build_pre_tokenize_replay(tokenize[0])
                if tokenize is not None
                else None
            ),
            mp_context=self._options.mp_context,
        )

    def to_torch_dataset(self, stateful: bool = True) -> _TorchIterableDatasetType:
        if _importlib_util.find_spec("torch.utils.data") is None:
            raise RuntimeError("to_torch_dataset requires 'torch' to be installed.")
        # Restores are applied in workers, so the driver cannot know one is
        # coming: a resume discards these ratios, costing one wasted census
        # on cold nodes (warm nodes hit the prime cache).
        self._prime_worksource()
        return _TorchPipelineIterableDataset(self, stateful=stateful)

    def to_indexable_torch_dataset(self) -> _TorchDatasetType:
        if not self.is_indexable:
            raise RuntimeError(
                "Pipeline is not indexable; cannot build a Map-style Dataset."
            )
        try:
            from torch.utils.data import Dataset
        except ModuleNotFoundError as exc:
            msg = "to_indexable_torch_dataset requires 'torch' to be installed."
            raise RuntimeError(msg) from exc

        pipeline = self

        class _Dataset(Dataset):
            def __len__(self) -> int:
                return len(pipeline.ws)

            def __getitem__(self, index: int) -> Any:
                sample_id = pipeline.ws.sample_id_at(index)
                return pipeline._eval_one(sample_id)

        return _Dataset()

    @property
    def is_indexable(self) -> bool:
        self._ensure_plan()
        assert self._plan is not None
        supports = getattr(self.ws, "supports_indexing", lambda: False)()
        return self._plan.indexable and supports

    def validate(self, *, strict: bool = False) -> "ValidationReport":
        """Run the custom-op validation harness against the current graph.

        Walks every user-supplied operator (those built by :meth:`add_op`)
        and runs a smoke test that checks: ``process_many`` is stateless
        across calls, the accumulator conserves samples on push+flush,
        ``flush(reset=True)`` leaves the accumulator in a fresh state, and
        ``has_pending_data()`` mirrors reality.  See the
        ``zephon.validation`` module for the full check list and the
        "Accumulators and Operators" page in the published documentation
        for the contracts being checked.

        Built-in ops are skipped (framework code, tested separately); a
        graph with no user ops returns an empty report immediately.

        Args:
            strict: If True, raise :class:`ValidationError` when any error
                is found.  Default False — caller inspects ``report.ok``.

        Returns:
            A :class:`ValidationReport` with structured issues.
        """
        from zephon.validation import (
            ValidationError as _ValidationError,
        )
        from zephon.validation import (
            validate_pipeline as _validate_pipeline,
        )

        report = _validate_pipeline(self)
        if strict and not report.ok:
            raise _ValidationError(report)
        return report

    def preflight_tokenizers(self, *, strict: bool = True) -> "ValidationReport":
        """Load and validate every tokenizer configured on the pipeline.

        Each tokenizer operator is probed through a deep copy, leaving the
        pipeline's operator state unchanged. This may download tokenizer files.

        Args:
            strict: If True, raise :class:`ValidationError` when any tokenizer
                fails to initialize.

        Returns:
            A :class:`ValidationReport` containing one
            ``TOKENIZER_PREFLIGHT_FAILED`` error per failure.
        """
        report = preflight_tokenizers(self)
        if strict and not report.ok:
            raise ValidationError(report)
        return report

    def _run_auto_validation(self) -> None:
        """Run the validation harness once per graph generation.

        The check is cheap (sub-ms per user op, none if there are no user
        ops) and only runs on the first ``iter()`` after a graph mutation.

        Behavior is governed by :attr:`RuntimeOptions.auto_validation`:

        - ``"strict"`` (default) surfaces every non-empty report via
          :func:`warnings.warn` and additionally raises
          :class:`ValidationError` on any error-severity issue.
        - ``"warn"`` runs the validator and surfaces the report via
          :func:`warnings.warn` without raising — escape hatch when a
          user op trips a false-positive check.
        - ``"off"`` skips validation entirely.
        """
        if self._validated:
            return
        raw_mode = self._options.auto_validation
        mode = raw_mode.lower() if isinstance(raw_mode, str) else raw_mode
        if mode not in ("strict", "warn", "off"):
            raise ValueError(
                f"auto_validation must be one of 'strict', 'warn', 'off' "
                f"(case-insensitive); got {raw_mode!r}"
            )
        if mode == "off":
            self._validated = True
            return
        report = self.validate(strict=False)
        if report.issues:
            warnings.warn(report.format(), stacklevel=2)
        if mode == "strict" and not report.ok:
            from zephon.validation import (
                ValidationError as _ValidationError,
            )

            raise _ValidationError(report)
        self._validated = True

    def __iter__(self) -> Iterator[Any]:
        if self._iterating:
            raise RuntimeError(
                "Pipeline is already being iterated. "
                "Close the existing iterator before starting a new one."
            )
        # Validation runs eagerly so the failure surfaces at iter() time, not
        # after the first next().
        self._run_auto_validation()
        self._prime_worksource()
        return self._iter_body()

    def _iter_body(self) -> Iterator[Any]:
        try:
            self._iterating = True
            if self._options.mtp_mode:
                import multiprocessing as _mp

                if _mp.current_process().daemon:
                    import warnings

                    warnings.warn(
                        "[zephon] mtp_mode=True but running inside a daemon "
                        "process (e.g. PyTorch DataLoader worker). Falling back "
                        "to inline mode. Set mtp_mode=False explicitly to "
                        "silence this warning.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    yield from self._iter_inline()
                else:
                    yield from self._iter_mtp()
            else:
                yield from self._iter_inline()
        finally:
            self._iterating = False

    def _build_raw_iter(
        self,
        restore_ckpt: dict[str, Any] | None = None,
    ) -> tuple[Engine, Iterator[StreamItem], bool]:
        """Build the raw engine iterator without notification wrapping.

        Returns ``(engine, iterator, use_monotone)``.  The iterator includes
        final-prefetch buffering when configured.  Used by both the inline
        path and the MTP subprocess worker.
        """
        self._ensure()
        engine = self._engine
        assert engine is not None
        if restore_ckpt is not None:
            engine.load_state_dict(restore_ckpt, replay=True)
        assert self._plan is not None
        use_monotone = self._plan.preserves_cursor_order
        iterator: Iterator[StreamItem] = engine.build_iter()
        final_prefetch = resolve_prefetch_batches(self._options)
        if final_prefetch > 0:
            iterator = buffered_iterable(iterator, final_prefetch, on_stop=engine.close)
        return engine, iterator, use_monotone

    def _iter_inline(self) -> Iterator[Any]:
        """Original inline iteration path — Engine runs in this process."""
        ckpt = self._pending_restore
        self._pending_restore = None
        engine, iterator, _ = self._build_raw_iter(restore_ckpt=ckpt)
        try:
            yield from self._yield_while_notifying(iterator)
        finally:
            engine.close()

    def _iter_mtp(self) -> Iterator[Any]:
        """MTP iteration path — Engine runs in a child process."""
        from zephon._internal.mtp import MTPPipeline

        self._last_state = None
        sp = MTPPipeline(
            self,
            buffer_size=resolve_mtp_buffer(self._options),
            buffer_bytes=self._options.mtp_buffer_bytes,
            transport=self._options.ipc_transport,
            prefetch=self._options.mtp_prefetch,
            restore_ckpt=self._pending_restore,
        )
        self._sp = sp
        self._pending_restore = None
        _completed = False
        try:
            yield from sp
            _completed = True
        finally:
            # Inner try/finally ensures sp.close() always runs even if
            # the checkpoint attempt raises (timeout, dead child, etc.).
            try:
                if not sp._closed:
                    if _completed:
                        # Normal completion: subprocess is in _wait_for_shutdown
                        # and can always handle CHECKPOINT.
                        if self._options.mtp_auto_checkpoint:
                            self._last_state = sp.checkpoint()
                    else:
                        # Early break: subprocess may be blocked on data_q.put()
                        # or in next(iterator).  Best-effort capture by draining
                        # data_q (without ACKs) to unblock the put-retry loop,
                        # which calls _drain_ctrl and sees our CHECKPOINT.
                        #
                        # NOTE: This works when the subprocess is blocked on
                        # data_q.put (the common case — queue is full because
                        # main stopped consuming).  It does NOT work when the
                        # subprocess is blocked inside next(iterator) doing slow
                        # computation, because it can't reach _drain_ctrl until
                        # the item is produced.  In that case capture_final_state
                        # times out and _last_state retains whatever was cached
                        # from any prior explicit checkpoint() call (or None).
                        # TODO: To handle the stuck-in-next(iterator) case, we'd
                        # need a background thread in the subprocess listening on
                        # ctrl_conn, or a signal-based interrupt mechanism.
                        sp.capture_final_state()
                        self._last_state = sp._last_state
            finally:
                sp.close()
                self._sp = None

    def _yield_while_notifying(
        self, source: Iterable[StreamItem]
    ) -> Iterator[StreamItem]:
        """Wrap an iterable that yields stream elements, notifying the engine.

        Uses the shared ``_notify_item`` helper (also called by the subprocess
        ACK path) so the bookkeeping logic is never duplicated.
        """
        from zephon._internal.notify import _notify_item, is_sentinel

        engine = self._engine
        assert engine is not None
        assert self._plan is not None
        use_monotone = self._plan.preserves_cursor_order
        for item in source:
            # Flush sentinels carry dummy cursor data — skip notification to
            # avoid corrupting offset_done bitmaps and cursor pinning.
            if not (isinstance(item, SampleRecord) and item.meta.is_flush_sentinel):
                _notify_item(engine, item, use_monotone)
            if not is_sentinel(item):
                yield item

        # After drain, flush cursor-pinned chunks.  During iteration,
        # notify() pins the record_cursor's chunk to keep it in inflight
        # for checkpoint correctness.  A final notify with record_cursor=None
        # releases the pin and lets fully-completed chunks evict.
        if not use_monotone:
            for lane_id in engine.inflight_chunks_per_lane:
                engine.notify(lane_id, [], record_cursor=None)

        # Pipeline fully drained — evict all remaining inflight chunks.
        # All accumulators have been force-flushed and every record has been
        # delivered and notified.  No further checkpoints will be taken, so
        # eviction is unconditionally safe.  Mid-stream eviction is gated by
        # the epoch-floor watermark, but end-of-stream is pure cleanup.
        for lane_id in list(engine.inflight_chunks_per_lane):
            engine.inflight_chunks_per_lane[lane_id].clear()

    def explain(self) -> str:
        spec = self._compile()
        assert self._plan is not None
        parts: list[str] = [self._plan.explain]
        if self._engine is not None:
            runtime = self._engine.explain()
        else:
            runtime = spec.explain(self._plan)
        if runtime:
            parts.append("")
            parts.append("Execution Graph:")
            parts.append(runtime)
        if self._options.mtp_mode:
            parts.append("GIL-isolation=mtp (Engine runs in child process)")
        return "\n".join(parts)

    def _eval_one(self, sample_id: Any) -> Any:
        self._ensure()
        assert self._engine is not None
        value: EngineSample | SampleId | Any = sample_id
        try:
            if (
                isinstance(sample_id, tuple)
                and len(sample_id) == 3
                and all(isinstance(x, int) for x in sample_id)
            ):
                sid = cast(SampleId, sample_id)
                value = cast(EngineSample, (sid, 0, 0, 0, 0))  # component_id=0
        except Exception:
            pass
        return self._engine.eval_one(value)

    def mtp_queue_stats(self) -> MTPQueueStats | None:
        """Return occupancy of the MTP hand-off queue (see ``MTPQueueStats``).

        Returns None when no MTP subprocess is live: ``mtp_mode`` is off,
        iteration has not started or has finished, or the daemon fallback
        chose inline mode.  Cheap to call once per training step.
        """
        if self._sp is None:
            return None
        return self._sp.queue_stats()

    def inflight_summary(self) -> dict[int, int]:
        """Return per-lane inflight chunk counts: ``{lane_id: count}``.

        In MTP mode this reads from shared memory (non-blocking, zero IPC).
        In inline mode it reads the engine dict directly.
        Returns ``{}`` if the engine is not running.
        """
        if self._sp is not None:
            return self._sp.inflight_summary()
        if self._engine is not None:
            return self._engine.inflight_summary()
        return {}

    def metrics_snapshot(self) -> PipelineSummary | None:
        """Return node metrics, or ``None`` unless inline tracking is active."""
        if self._engine is not None:
            return self._engine.metrics_snapshot()
        return None

    def fetch_timing_snapshot(self) -> FetchTimingSummary | None:
        """Return fetch timings for the inline engine, or ``None`` if absent.

        Data is collected only in ``ExecutionTrackingMode.NODES``.
        """
        if self._engine is not None:
            return self._engine.fetch_timing_snapshot()
        return None

    def prefetch_timing_snapshot(self) -> PrefetchTimingSummary | None:
        """Return prefetch timings for the inline engine, or ``None`` if absent.

        Data is collected only in ``ExecutionTrackingMode.NODES``.
        """
        if self._engine is not None:
            return self._engine.prefetch_timing_snapshot()
        return None

    def checkpoint(self) -> dict[str, Any]:
        # Live subprocess — ask it for a checkpoint.
        if self._sp is not None:
            return self._sp.checkpoint()
        # restore() was called but iteration hasn't started yet —
        # the pending checkpoint supersedes any stale engine / _last_state.
        if self._pending_restore is not None:
            return dict(self._pending_restore)
        # Inline engine (live or closed) — state_dict() works either way.
        # Checked before _last_state so a subsequent inline iteration
        # takes precedence over a stale cached subprocess checkpoint.
        if self._engine is not None:
            return self._engine.state_dict()
        # Cached state from a completed subprocess iteration.
        if self._last_state is not None:
            return self._last_state
        # The only engine build that bypasses __iter__: prime first, or the
        # checkpoint serializes unprimed token lanes no restore can prime.
        self._prime_worksource()
        self._ensure()
        assert self._engine is not None
        return self._engine.state_dict()

    def restore(self, ckpt: dict[str, Any]) -> None:
        from zephon._internal.checkpoint import EngineStateV2

        # Fail-fast: construct the schema purely for its validation side
        # effects. The Engine reloads it later when iteration starts.
        EngineStateV2.load(ckpt)
        # Always stash — applied lazily at the start of iteration in both
        # _iter_inline() and _iter_mtp().  This keeps restore()
        # lightweight and avoids eagerly building an Engine.
        self._pending_restore = ckpt

    def __getstate__(self) -> dict[str, Any]:
        """
        Pickle guard for DataLoader/StatefulDataLoader worker bootstrap.

        Why this exists
        ----------------
        - When num_workers > 0 and the start method is SPAWN (macOS/Windows, or Linux if set),
          PyTorch/torchdata must *pickle the entire Dataset object* to send it into each worker.
          The IterableDataset holds a reference to this Pipeline, so the Pipeline is pickled too.
        - That pickling happens during **worker startup**, *before* any iteration and *before*
          your dataset’s `state_dict()` is consulted. In other words, checkpoint/resume APIs do
          not affect how the Dataset/Pipeline objects themselves are transferred to workers.
        - If the Pipeline already holds a live Engine (thread pools, locks, file handles, etc.),
          serialization either fails (not picklable) or, under FORK, produces an unsafe snapshot
          of a partially initialized thread pool in the child process.

        Strategy
        --------
        - Strip all process-local runtime from the pickled representation,
          setting it to None in the pickled state:

          * `_engine` : the live runtime (threads, queues, locks)
          * `_plan`   : the derived execution plan (rebuildable from the graph)

        - Keep only pure data/config: graph, options, work source. Workers will rebuild the
          plan/engine lazily upon first iteration.

        How this interacts with resume
        -------------------------------
        - For resume with torchdata.StatefulDataLoader, the dataset should expose its own
          `state_dict()` / `load_state_dict()` and *stash* any Zephon checkpoint there.
          Apply that checkpoint inside the worker (e.g., in `__iter__`) by calling
          `pipeline.restore(ckpt)`, which will lazily `_ensure()` and load state.
        - We *intentionally* do not serialize a live Engine here. Resume state should be
          passed explicitly via the dataset’s state API, not implicitly by pickling.

        Effects
        -------
        - SPAWN: safe — no Engine is ever serialized across processes.
        - FORK: safe — the child won’t inherit an already-started thread pool; the Engine
          will be created post-fork inside the worker.
        """
        d = self.__dict__.copy()
        d["_engine"] = None
        d["_plan"] = None
        d["_runtime_spec"] = None
        d["_sp"] = None
        d["_iterating"] = False
        d["_last_state"] = None
        # Keep _pending_restore — it's a plain dict (JSON-serializable checkpoint
        # data) that should survive pickle so restore() + pickle doesn't silently
        # lose the stashed checkpoint.
        return d

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Unpickle guard that complements __getstate__.

        - Ensures plan/engine are rebuilt lazily in the receiving process by resetting
          the runtime fields to None.
        - Leaves graph/options/work source intact (pure data).
        - If you’re using a stateful IterableDataset, that dataset should apply any
          previously stashed checkpoint from its own `load_state_dict()` by calling
          `pipeline.restore(...)` *inside the worker* (e.g., in `__iter__`), not here.
        """
        self.__dict__.update(state)
        # _engine, _plan, _runtime_spec, _sp are already None in the
        # pickled state (set by __getstate__).  _pending_restore is kept.


__all__ = ["Pipeline"]
