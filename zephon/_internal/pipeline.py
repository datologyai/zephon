# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""User-facing pipeline wrapper that layers ergonomics atop core planning."""

import importlib.util as _importlib_util
import warnings
import weakref
from collections.abc import Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
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

from zephon._internal.branching import Router
from zephon._internal.checkpoint._aggregation import (
    aggregate_checkpoint,
    requires_aggregation,
)
from zephon._internal.coordination import CheckpointCoordinator, make_coordinator
from zephon._internal.engine import Engine, get_torch_worker_info
from zephon._internal.functional_op import _is_user_op
from zephon._internal.graph import Graph, Merge, Node, Plan
from zephon._internal.ops.batch import Batch
from zephon._internal.ops.ensure_mixture import EnsureMixture
from zephon._internal.ops.fetch import FetchOp
from zephon._internal.ops.identity import Identity
from zephon._internal.ops.map_transform import MapBatchTransform
from zephon._internal.ops.materialize import Materialize
from zephon._internal.ops.pack_sequences import PackSequences
from zephon._internal.ops.prefetch import PrefetchOp
from zephon._internal.ops.shuffle_buffer import ShuffleBuffer
from zephon._internal.ops.tokenize_base import TokenizeBase
from zephon._internal.planner import Planner
from zephon._internal.record_ops import Branch, _BranchOp, _mutates_graph, _RecordOps
from zephon._internal.runtime_spec import (
    RuntimeSpec,
    resolve_mtp_buffer,
    resolve_prefetch_batches,
    resolve_runtime_spec,
)
from zephon._internal.stream import EngineSample
from zephon._internal.utils import buffered_iterable
from zephon._internal.utils.gc_freeze import GCFreeze, resolve_gc_freeze
from zephon._internal.utils.torch_compat import detect_loader_kind
from zephon.io.options import StoreOptions
from zephon.observability import ExecutionTrackingMode, MetricsSinkConfig
from zephon.observability.mtp_stats import MTPQueueStats
from zephon.observability.stats import (
    FetchTimingSummary,
    PipelineSummary,
    PrefetchTimingSummary,
)
from zephon.ops.base import BaseOp
from zephon.ops.config import PackingAlgorithm
from zephon.ops.grouping import DomainGroups
from zephon.options import RuntimeOptions
from zephon.types import SampleBatch, SampleId, SamplePayload, SampleRecord, StreamItem
from zephon.validation import ValidationError, preflight_tokenizers
from zephon.work import WorkSource
from zephon.work.token_estimation import _PreTokenizeReplay, _UnreplayableOp

# TypeVar for stateful_transform state type
_S = TypeVar("_S")

_OpT = TypeVar("_OpT")


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

        # A loaded checkpoint not yet applied (__iter__ applies it lazily) is
        # the resume position; it supersedes any live or cached runtime state
        # left over from a previous iteration.
        if self._pending_ckpt is not None:
            return {"engine": self._pending_ckpt}

        # If an iteration exists (inline engine or live subprocess), or a
        # finished MTP iteration cached its final state, use pipeline.checkpoint().
        p = self._pipeline
        if (
            p._engine is not None
            or p._sp is not None
            or p._last_local_state is not None
            or p._last_state is not None
        ):
            return {"engine": p.checkpoint()}
        # Nothing loaded and no iteration has run in this process yet.
        return {"engine": None}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        if not self._stateful:
            raise AttributeError(
                "This dataset is not stateful. Pass stateful=True in to_torch_dataset()."
            )
        # Do NOT call pipeline.restore() here — this might be running in the parent.
        # Just stash it; __iter__ in the worker will apply it before building the engine.
        self._pending_ckpt = sd.get("engine")


class Pipeline(_RecordOps):
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
        self._last_local_state: dict[str, Any] | None = None
        self._final_mtp_coordinator: CheckpointCoordinator | None = None
        self._mtp_started = False
        self._iterating: bool = False
        # Track iterator wrappers (including MTP) without keeping them alive.
        self._iterators: weakref.WeakSet[Generator[Any, None, None]] = weakref.WeakSet()
        self._validated: bool = False
        self._options = RuntimeOptions()
        self._fetch_node: Node[FetchOp] = self._graph.add(
            "fetch", FetchOp(), placement="local"
        )
        # The vertex the next op takes as input: the last op, or the merge right
        # after branch().
        self._tail: Node[Any] | Merge = self._fetch_node
        # Optional prefetch node inserted before fetch
        self._prefetch_node: Node[PrefetchOp] | None = None

    def _attach(
        self,
        name: str,
        op: BaseOp,
        *,
        placement: str,
        parallelism: Optional[int],
    ) -> None:
        """Append ``op`` after the current tail and make it the tail."""
        self._tail = self._graph.add(
            name, op, self._tail, placement=placement, parallelism=parallelism
        )

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
    def branch(
        self,
        by: Callable[[SamplePayload], str],
        branches: Mapping[str, Callable[["Branch"], "Branch"]],
        *,
        default: str | None = None,
    ) -> "Pipeline":
        """Route each record to one branch, then merge the branches back.

        ``by`` picks a branch for each record from its payload. Each branch
        holds at most one stateless, record-local op (``map_transform``,
        ``add_op``, ``decode_text``, ``tokenize``, ``tokenize_chat`` or
        ``expand_chat``); ``lambda b: b`` passes records through unchanged.
        The branches merge back in input order before the next op, so the
        output is the same as a chain whose op picks per record with an
        if/else. Branches route records, so call ``branch()`` before
        ``batch()``.

        Args:
            by: Returns a branch name for a record's payload. It must be
                deterministic and read only the payload.
            branches: Branch name to a function that receives an empty
                `Branch` and returns it, with at most one op added.
            default: Branch for keys that name no branch. Without it, an
                unknown key raises while the pipeline runs.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: On an empty or invalid branch name, an unknown
                ``default``, a branch with more than one op, or a ``branch()``
                right after another one.
            TypeError: If ``by`` is not callable or a branch function does not
                return the `Branch` it was given.

        Example:
            >>> pipeline.branch(
            ...     by=lambda p: "chat" if "messages" in p else "text",
            ...     branches={
            ...         "text": lambda b: b.tokenize(tokenizer_id="gpt2", field="text"),
            ...         "chat": lambda b: b.tokenize_chat(tokenizer_id="gpt2"),
            ...     },
            ... )
        """
        if not callable(by):
            raise TypeError(f"branch() needs a callable `by`, got {type(by).__name__}")
        if isinstance(self._tail, Merge):
            raise ValueError(
                "branch() cannot follow another branch() directly; add an op "
                "after the first one to merge its branches"
            )
        upstream = self._tail
        if not branches:
            raise ValueError("branch() needs at least one branch")
        for name in branches:
            if not isinstance(name, str) or not name:  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ValueError(
                    f"branch names must be non-empty strings, got {name!r}"
                )
        if default is not None and default not in branches:
            raise ValueError(
                f"default={default!r} names no branch; branches: {list(branches)}"
            )

        # Build every branch before touching the graph, so a failing branch
        # function leaves the pipeline as it was.
        held: dict[str, _BranchOp] = {}
        for name, build in branches.items():
            builder = Branch(name)
            if build(builder) is not builder:
                raise TypeError(
                    f"branch {name!r}: the function must return the Branch it "
                    "received, e.g. `lambda b: b.tokenize(...)`"
                )
            # A passthrough branch is an ordinary branch op that returns its
            # input, so every branch has an op for the merge to read.
            held[name] = builder._held or _BranchOp(
                "passthrough", Identity(), placement="auto", parallelism=None
            )
        router = Router(by=by, branches=tuple(held), default=default)
        split = self._graph.add_split(f"{upstream.name}_split", upstream, router)
        branch_ops = {
            name: self._graph.add(
                op.name,
                op.op,
                split,
                placement=op.placement,
                parallelism=op.parallelism,
            )
            for name, op in held.items()
        }
        self._tail = self._graph.add_merge(f"{upstream.name}_merge", split, branch_ops)
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
        self._attach(
            "map_batch",
            op,
            placement=placement,
            parallelism=parallelism,
        )
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
        self._attach(
            name,
            op,
            placement=placement,
            parallelism=parallelism,
        )
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
        self._attach(
            "shuffle_buffer",
            op,
            placement=placement,
            parallelism=parallelism,
        )
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

        An end-of-data (EOD) sentinel signals that a component's source has run out
        in this lane; buffered records may still arrive. For components listed by
        ``WorkSource.continues_after_exhaustion()``, this removes their weight from the
        target, including explicit overrides, and renormalizes the remaining weights.
        ``StaticMixtureWorkSource`` lists components with ``exhausted_policy="redistribute"``.
        Late records follow ``obsolete_drain_rate`` in both modes.
        The final EOD leaves the target unchanged. Previously retired components
        continue through controlled drainage to avoid a burst of buffered records
        at final EOD.

        Args:
            max_buffer_size: Samples to hold while waiting for the component the
                target mixture needs next. Default 1000; when the buffer fills, the
                operator emits what it has, so the output mixture may drift if a
                component stays scarce. Pass ``None`` to instead **drop** the surplus
                it cannot place on-target — at every epoch boundary and at end of
                stream. This enforces the mixture at the cost of dropping data.
                Use strict mode when an exact mixture matters more than keeping
                every sample. How much is dropped depends on the stream's skew
                and on ``flush_every_k_chunks`` (smaller → more dropped).
                ensure_mixture is always non-monotonic, so that cadence defaults
                to 8; in strict mode the buffer is then discarded on roughly
                every flush. If drops are too frequent, raise
                ``flush_every_k_chunks`` above the default to give scarce components
                more time to absorb the buffer (at the cost of holding more in memory
                between flushes).
            drain_target_ratio: When forced to emit (buffer hits max_buffer_size), drain
                the buffer down to this fraction of max_buffer_size before stopping.
                Default is 0.8 (drain to 80% of max).  Ignored when ``max_buffer_size=None``.
            obsolete_drain_rate: Emit one obsolete record per ``int(1 / rate)`` normal
                output records, regardless of token count. The effective share is
                ``1 / (int(1 / rate) + 1)``: default 0.1 means one in eleven;
                values above 0.5 alternate (50%); 0 disables periodic drainage.
                A record is obsolete if it has any positive off-target contribution
                or no component in the target. Its on-target contributions still
                count toward the mixture. At most one slot is saved while the queue
                is empty. Bounded forced drains and flushes may exceed this rate;
                strict flushes discard leftovers.
                Within an epoch, components removed by one target change drain in
                proportion to their former weights. Separate EODs drain in notice
                order when their records are available. Other off-target records
                follow in arrival order.
            weight_by: How to compute sample weights. Options:

                - ``auto`` (default): Auto-detect token field from common names
                  (input_ids, tokens, token_ids, ids). Raises if not found.
                - ``samples``: Each sample has weight 1.
                - Explicit field name (e.g., ``input_ids``): Use that field's length.
                - Callable: Custom function taking SampleRecord, returning float.
                  Must not modify the record.
            warn_tolerance: If set, warn when mixture drift exceeds this value (0.05 = ±5%).
                If None (default), no warnings are emitted.
            mixture: Explicit mixture target {component_name: float}. If None, derived
                from chunk mixture via engine context. EOD sentinel handling also applies
                to this override.
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
        self._attach(
            "ensure_mixture",
            op,
            placement=placement,
            parallelism=parallelism,
        )
        return self

    @_mutates_graph
    def materialize(
        self, placement: str = "auto", parallelism: Optional[int] = None
    ) -> "Pipeline":
        self._attach(
            "materialize",
            Materialize(),
            placement=placement,
            parallelism=parallelism,
        )
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
        self._attach("batch", op, placement=placement, parallelism=None)
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
        (default keeps it as-is). When a callback returns a merged mapping of
        unshifted token/media fields, use ``SampleBatch.to_training`` to convert
        the batched packs into training tensors.
        For built-in concatenation/padding of token-aligned fields with
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
        self._attach(
            "pack_sequences",
            op,
            placement=placement,
            parallelism=parallelism,
        )
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
        self._attach("pack_flat", op, placement=placement, parallelism=parallelism)
        return self

    def enable_observability(
        self,
        tracking: ExecutionTrackingMode | str = ExecutionTrackingMode.NODES,
        *,
        sink: MetricsSinkConfig | None = None,
        measure_bytes: bool | None = None,
    ) -> "Pipeline":
        """Enable runtime metrics collection for the compiled pipeline.

        Args:
            tracking: How much execution detail to collect.
            sink: Where metrics are exported; defaults to logs.
            measure_bytes: Size each node's consumed/produced payloads under
                ``NODES`` tracking. ``True`` or ``False`` forces it for every
                stage. ``None`` (default) keeps any value already set via
                :meth:`options`; if none was set, only runners that ship
                payloads between processes or hosts (process and remote
                stages) measure.
        """
        mode = ExecutionTrackingMode.from_value(tracking)
        self._options.execution_tracking = mode
        if measure_bytes is not None:
            self._options.measure_bytes = measure_bytes
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
        self._attach(
            "delay",
            op,
            placement=placement,
            parallelism=parallelism,
        )
        return self

    def options(self, **hints: Any) -> "Pipeline":
        # TODO(MaxiBoether): Support in addition to dict options just typed options using dataclasses.
        for old, new in (
            ("shm_min_size", "shm_min_item_bytes"),
            ("coalesce_tensors", "shm_coalesce"),
        ):
            if old in hints:
                raise TypeError(f"'{old}' is no longer supported. Use '{new}' instead.")
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
        can change what it counts. An op without calibration replay support
        yields a marker that priming rejects if it actually measures.
        """
        fetch_index = self._graph.nodes.index(self._fetch_node)
        ops: list[Any] = []
        for node in self._graph.nodes[fetch_index + 1 : tokenize_index]:
            if not node.op.traits().calibration_replayable:
                is_custom = _is_user_op(node.op)
                return _UnreplayableOp(
                    node.name if is_custom else type(node.op).__name__,
                    is_custom=is_custom,
                )
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

        Each tokenizer operator is initialized through a deep copy. Successful
        template checks are retained for worker reuse, while the original
        tokenizer stays lazy. Changed configurations are checked again in the
        worker. This may download tokenizer files.

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
        iterator = self._iter_body()
        self._iterators.add(iterator)
        return iterator

    def close(self) -> None:
        """Close iterators and runtimes; safe to repeat.

        Call between iteration steps from the owning thread.
        MTP saves acknowledged progress locally for a later checkpoint.
        Save and restore a checkpoint before resuming a closed stream.
        """
        error: BaseException | None = None
        # Iterator cleanup captures MTP progress before the child shuts down.
        for resource in [*self._iterators, self._sp, self._engine]:
            if resource is not None:
                try:
                    resource.close()
                except BaseException as exc:
                    if error is None:
                        error = exc
        if error is not None:
            raise error

    def _iter_body(self) -> Generator[Any, None, None]:
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
        """Run the engine in the current process."""
        # GC freezing affects the whole interpreter, so inline mode defaults off.
        # This also applies to daemon workers falling back from MTP.
        freezer = GCFreeze(resolve_gc_freeze(self._options.gc_freeze, mtp=False))
        try:
            ckpt = self._pending_restore
            self._pending_restore = None
            self._last_local_state = None
            self._final_mtp_coordinator = None
            self._mtp_started = False
            engine, iterator, _ = self._build_raw_iter(restore_ckpt=ckpt)
            try:
                yield from self._yield_while_notifying(iterator)
            finally:
                engine.close()
        finally:
            freezer.close()

    def _iter_mtp(self) -> Iterator[Any]:
        """MTP iteration path — Engine runs in a child process."""
        from zephon._internal.mtp import MTPPipeline

        self._last_state = None
        self._last_local_state = None
        self._final_mtp_coordinator = None
        self._mtp_started = False
        # The child owns stream progress; release any checkpoint-only parent engine.
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        sp = MTPPipeline(
            self,
            buffer_size=resolve_mtp_buffer(self._options),
            buffer_bytes=self._options.mtp_buffer_bytes,
            transport=self._options.ipc_transport,
            prefetch=self._options.mtp_prefetch,
            restore_ckpt=self._pending_restore,
        )
        self._sp = sp
        self._mtp_started = True
        self._pending_restore = None
        completed = False
        try:
            yield from sp
            completed = True
        finally:
            # Inner try/finally ensures sp.close() always runs even if
            # the checkpoint attempt raises (timeout, dead child, etc.).
            try:
                if not sp._closed:
                    if completed:
                        # Normal completion: subprocess is in _wait_for_shutdown
                        # and can always handle LOCAL_CHECKPOINT.
                        if self._options.mtp_auto_checkpoint:
                            self._last_local_state = sp.checkpoint(local=True)
                    else:
                        # Early break: subprocess may be blocked on data_q.put()
                        # or in next(iterator).  Best-effort capture by draining
                        # data_q (without ACKs) to unblock the put-retry loop,
                        # which calls _drain_ctrl and sees our LOCAL_CHECKPOINT.
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
                        sp.capture_final_state(local=True)
                        if sp._last_state_is_local:
                            self._last_local_state = sp._last_state
                        else:
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
        from zephon._internal.notify import _notify_item, is_sentinel, is_tombstone

        engine = self._engine
        assert engine is not None
        assert self._plan is not None
        use_monotone = self._plan.preserves_cursor_order
        for item in source:
            sentinel = is_sentinel(item)
            # Only data and tombstones have real offsets; flush and EOD have dummy cursors.
            if not sentinel or is_tombstone(item):
                _notify_item(engine, item, use_monotone)
            if not sentinel:
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

    @overload
    def checkpoint(self, *, as_bytes: Literal[False] = False) -> dict[str, Any]: ...

    @overload
    def checkpoint(self, *, as_bytes: Literal[True]) -> bytes: ...

    def checkpoint(self, *, as_bytes: bool = False) -> dict[str, Any] | bytes:
        """Collect resumable pipeline progress, optionally as one encoded value.

        Args:
            as_bytes: Return versioned, zstd-compressed MessagePack bytes instead
                of the default dictionary. This lets checkpoint systems such as
                PyTorch DCP preserve dynamic replay entries as one value without
                requiring integrations to pickle the state. All participating
                ranks must still call checkpoint at the same training step.
                Custom work-source state must use MessagePack-compatible values.

        Returns:
            The complete checkpoint dictionary, or its encoded bytes. Encoding
            occurs after collection and any distributed aggregation.
        """
        from zephon._internal.checkpoint import encode_checkpoint

        state = self._checkpoint_state()
        return encode_checkpoint(state) if as_bytes else state

    def _checkpoint_state(self) -> dict[str, Any]:
        # Live subprocess — ask it for a checkpoint.
        if self._sp is not None:
            return self._sp.checkpoint()
        # restore() was called but iteration hasn't started yet —
        # the pending checkpoint supersedes any stale engine / _last_state.
        if self._pending_restore is not None:
            return dict(self._pending_restore)
        if self._last_local_state is not None:
            return self._aggregate_final_mtp_state()
        # Inline engine (live or closed) — state_dict() works either way.
        # Checked before _last_state so a subsequent inline iteration
        # takes precedence over a stale cached subprocess checkpoint.
        if self._engine is not None:
            return self._engine.state_dict()
        # Cached state from a completed subprocess iteration.
        if self._last_state is not None:
            return self._last_state
        if self._mtp_started:
            raise RuntimeError(
                "MTP iteration ended without a saved checkpoint. "
                + "Call checkpoint() before closing or exhausting the iterator, "
                + "or set mtp_auto_checkpoint=True for normal exhaustion."
            )
        # The only engine build that bypasses __iter__: prime first, or the
        # checkpoint serializes unprimed token lanes no restore can prime.
        self._prime_worksource()
        self._ensure()
        assert self._engine is not None
        return self._engine.state_dict()

    def _aggregate_final_mtp_state(self) -> dict[str, Any]:
        """Aggregate the finished child's cached progress."""
        assert self._last_local_state is not None
        local = self._last_local_state
        world = local["world"]
        if requires_aggregation(local) and self._final_mtp_coordinator is None:
            self._ensure_plan()
            assert self._plan is not None
            self._final_mtp_coordinator = make_coordinator(
                self._options,
                global_rank=int(world["global_rank"]),
                world_size=int(world["world_size"]),
                plan_id=self._plan.plan_id,
                worker_info=get_torch_worker_info,
            )
        coordinator = self._final_mtp_coordinator
        worker_id, _ = get_torch_worker_info()
        try:
            if coordinator is not None:
                coordinator.last_round_id = local.get("last_round_id")
                coordinator.reload_count = local.get("checkpoint_reload_count", 0)
                coordinator.check_ready()
            result = aggregate_checkpoint(
                local,
                coordinator,
                worker_id=worker_id,
                merge=Engine._merge_state_dicts,
            )
            if coordinator is not None:
                # Keep closed and live ranks on the same next checkpoint round.
                self._last_local_state = dict(
                    local, last_round_id=coordinator.last_round_id
                )
            return result
        finally:
            if coordinator is not None:
                try:
                    coordinator.cleanup()
                except Exception as exc:
                    warnings.warn(
                        f"[zephon] Failed to clean up checkpoint aggregation: {exc}",
                        stacklevel=2,
                    )

    def restore(self, ckpt: dict[str, Any] | bytes) -> None:
        """Validate saved progress and restore it lazily when iteration starts.

        Accepts a checkpoint dictionary or bytes from ``checkpoint(as_bytes=True)``.
        Byte decoding precedes the same schema validation and migration used for
        dictionaries. Malformed or unsupported byte formats raise ValueError
        before a runtime is constructed.
        """
        from zephon._internal.checkpoint import EngineStateV2, decode_checkpoint

        if isinstance(ckpt, bytes):
            ckpt = decode_checkpoint(ckpt)
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
        d.pop("_iterators", None)
        d["_last_state"] = None
        d["_last_local_state"] = None
        d["_final_mtp_coordinator"] = None
        d["_mtp_started"] = False
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
        self._iterators = weakref.WeakSet()
        # _engine, _plan, _runtime_spec, _sp are already None in the
        # pickled state (set by __getstate__).  _pending_restore is kept.


__all__ = ["Pipeline"]
