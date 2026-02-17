# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""User-facing pipeline wrapper that layers ergonomics atop core planning."""

import importlib.util as _importlib_util
import warnings
from collections.abc import Iterable
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
)

from zephon.core.constants import (
    ContributorRef,
    EngineSample,
    SampleBatch,
    SampleCursor,
    SampleId,
    SamplePayload,
    SampleRecord,
    StreamItem,
)
from zephon.core.engine import Engine, RuntimeOptions
from zephon.core.graph import Graph, Node, Plan
from zephon.core.planner import Planner
from zephon.io.options import StoreOptions
from zephon.observability import ExecutionTrackingMode, MetricsSinkConfig
from zephon.ops import (
    Batch,
    DecodeText,
    FetchOp,
    MapTransform,
    Materialize,
    PackSequences,
    PrefetchOp,
    ShuffleBuffer,
    TokenizeText,
)
from zephon.utils import buffered_iterable
from zephon.utils.torch_compat import detect_loader_kind
from zephon.work import WorkSource

# TypeVar for stateful_transform state type
_S = TypeVar("_S")


# ---------- Private protocol fallbacks (non-public => no D101) ----------
class _IterableDatasetProto(Protocol):
    def __iter__(self) -> Iterator[Any]: ...


class _DatasetProto(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...


# ---------- Public type aliases (unified across branches) ----------
if TYPE_CHECKING:
    try:
        from torch.utils.data import Dataset as _TDataset
        from torch.utils.data import IterableDataset as _TIterable
    except Exception:
        _TIterable = _IterableDatasetProto
        _TDataset = _DatasetProto
    TorchIterableDatasetType: TypeAlias = _TIterable  # type: ignore[assignment]
    TorchDatasetType: TypeAlias = _TDataset  # type: ignore[assignment]
else:
    TorchIterableDatasetType: TypeAlias = _IterableDatasetProto
    TorchDatasetType: TypeAlias = _DatasetProto

# ---------- Runtime base (used for inheritance / isinstance) ----------
_RTIterableDatasetBase: type[Any]
try:
    from torch.utils.data import IterableDataset as _RTIterableDatasetBase
except Exception:
    # Provide a concrete runtime base so type checkers accept the subclass definition.
    _RTIterableDatasetBase = object  # type: ignore[assignment]


class TorchPipelineIterableDataset(_RTIterableDatasetBase):
    """Adapter that just yields from the Zephon Pipeline; no sharding here."""

    def __init__(self, pipeline: "Pipeline", *, stateful: bool = False) -> None:
        self._pipeline = pipeline
        self._stateful = stateful
        self._pending_ckpt: dict | None = None  # only used when stateful=True

    def __iter__(self):
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

    def state_dict(self):
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

        # If engine already exists (typical after iteration started), use pipeline.checkpoint().
        eng = self._pipeline._engine
        if eng is not None:
            return {"engine": self._pipeline.checkpoint()}
        # If iteration hasn’t begun in this process, return whatever pending state we have (or None).
        return {"engine": self._pending_ckpt}

    def load_state_dict(self, sd: dict[Any, Any]):
        if not self._stateful:
            raise AttributeError(
                "This dataset is not stateful. Pass stateful=True in to_torch_dataset()."
            )
        # Do NOT call pipeline.restore() here — this might be running in the parent.
        # Just stash it; __iter__ in the worker will apply it before building the engine.
        self._pending_ckpt = sd.get("engine")


class Pipeline:
    """Fluent builder that compiles user ops into an executable pipeline."""

    def __init__(self, work_source: WorkSource) -> None:
        self.ws = work_source
        self._graph = Graph()
        self._plan: Plan | None = None
        self._engine: Engine | None = None
        self._options = RuntimeOptions()
        self._fetch_node: Node[FetchOp] = self._graph.add(
            "fetch", FetchOp(), placement="local"
        )
        self._tail: Node[Any] = self._fetch_node
        # Optional prefetch node inserted before fetch
        self._prefetch_node: Node[PrefetchOp] | None = None

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
            ...     .tokenize()
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
    ) -> "Pipeline":
        """Add a stateful transformation with custom accumulation logic.

        This is a higher-level alternative to implementing the full Op protocol.
        State management runs on the pump thread (serial); the optional transform
        runs in parallel workers for expensive computation.

        Execution model:
        - push/flush: Run on pump thread (serial) for state management
        - transform: Runs in parallel workers for expensive per-item processing

        This split enables patterns like "deduplicate (serial) then encode (parallel)".

        The state lifecycle:
        1. State is lazily initialized on first batch via init_state()
        2. Each batch calls push(state, items) -> (new_state, outputs)
        3. If should_flush returns True, flush is called and state is reset
        4. On stream end, flush() emits any remaining buffered items
        5. Each output item goes through transform (if provided) in parallel

        Args:
            name: Operator name for debugging/metrics.
            init_state: Factory that creates initial state (called once per worker).
            push: Called with (state, batch) -> (new_state, outputs).
                Outputs are emitted immediately; state carries forward.
            flush: Optional. Called at end-of-stream to emit remaining buffered items.
            should_flush: Optional. If returns True, triggers early flush and state reset.
            transform: Optional. Batch-level transform that runs in parallel workers.
                Receives the full batch from the accumulator, preserving batch structure
                for efficient GPU processing, vectorized ops, etc.
            placement: Placement hint for this operator.
            parallelism: Worker parallelism. Use >1 when transform is expensive.
            indexable: Whether this operator preserves indexability (default False).
                Set True only if the transform is 1:1 and deterministic.

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
        from zephon.ops.stateful_transform import StatefulTransformOp

        op = StatefulTransformOp(
            init_state=init_state,
            push_fn=push,
            flush_fn=flush,
            should_flush_fn=should_flush,
            transform_fn=transform,
            parallelism=parallelism,
            indexable=indexable,
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

    def tokenize(
        self,
        tokenizer: Any | None = None,
        tokenizer_id: str | None = None,
        *,
        field: str = "text",
        add_attention_mask: bool = True,
        max_length: int | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        return_tensors: str | None = None,
        split_long_samples: bool = False,
        use_fast: bool | None = None,
        preserve_upstream_payload: bool = False,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> "Pipeline":
        op = TokenizeText(
            tokenizer,
            tokenizer_id,
            field=field,
            add_attention_mask=add_attention_mask,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            split_long_samples=split_long_samples,
            use_fast=use_fast,
            preserve_upstream_payload=preserve_upstream_payload,
        )
        node = self._graph.add(
            "tokenize", op, self._tail, placement=placement, parallelism=parallelism
        )
        self._tail = node
        return self

    def shuffle(
        self,
        buffer_size: int,
        *,
        seed: int = 0,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> "Pipeline":
        """Insert a deterministic shuffle buffer."""
        op = ShuffleBuffer(buffer_size=buffer_size, seed=seed)
        node = self._graph.add(
            "shuffle_buffer",
            op,
            self._tail,
            placement=placement,
            parallelism=parallelism,
        )
        self._tail = node
        return self

    def ensure_mixture(
        self,
        *,
        max_buffer_size: int = 1000,
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
            max_buffer_size: Maximum samples to buffer before forcing emission. Only
                reached when the desired component isn't available. Default is 1000.
            drain_target_ratio: When forced to emit (buffer hits max_buffer_size), drain
                the buffer down to this fraction of max_buffer_size before stopping.
                Default is 0.8 (drain to 80% of max).
            obsolete_drain_rate: Fraction of emissions reserved for draining obsolete
                components (those no longer in the current mixture target). Default is
                0.1 (10%), meaning 1 in every 10 emissions drains an obsolete sample.
            weight_by: How to compute sample weights. Options:
                - "auto" (default): Auto-detect token field from common names
                  (input_ids, tokens, token_ids, ids). Raises if not found.
                - "samples": Each sample has weight 1.
                - Explicit field name (e.g., "input_ids"): Use that field's length.
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
            # Token-level (default) - place after tokenize
            pipeline.fetch().tokenize(...).ensure_mixture()  # Uses defaults

            # Sample-level enforcement (after filter)
            pipeline.fetch().filter(...).ensure_mixture(weight_by="samples")

            # With explicit token field
            pipeline.fetch().tokenize(...).ensure_mixture(weight_by="input_ids")

            # With warnings for drift (warn if >5% deviation)
            pipeline.fetch().tokenize(...).ensure_mixture(warn_tolerance=0.05)

            # Explicit mixture target (override chunk mixture)
            pipeline.fetch().tokenize(...).ensure_mixture(
                mixture={"English": 0.7, "German": 0.3}
            )
        """
        from zephon.ops.ensure_mixture import EnsureMixture

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

    def pack_sequences(
        self,
        max_length: int,
        num_bins: int,
        length_fn: Callable[[SampleRecord], int] | Literal["auto"] | str = "auto",
        algorithm: Literal["first_fit", "best_fit"] = "first_fit",
        *,
        drop_oversized: bool = True,
        shuffle_strategy: Literal["random", "length", None] = None,
        shuffle_seed: Optional[int] = None,
        flush_strategy: Literal["fifo", "fullest"] = "fifo",
        pack_payloads: str | Callable[[list[Any]], Any] = "keep_list",
        placement: str = "auto",
    ) -> "Pipeline":
        """Add a sequence packing operator to the pipeline.

        Supports lambda functions for length_fn and pack_payloads parameters,
        which work seamlessly with multiprocessing-based runners.

        Args:
            max_length: Maximum length for packed bins.
            num_bins: Number of bins to maintain per lane.
            length_fn: How to extract sequence length. Options:
                - "auto" (default): Auto-detect from common token fields
                  (input_ids, tokens, token_ids, ids).
                - Explicit field name (e.g., "input_ids"): Use that field.
                - Callable: Custom function taking SampleRecord, returning int.
                  Can be a lambda (e.g., lambda r: len(r.payload["tokens"])).
            algorithm: Packing algorithm to use ("first_fit" or "best_fit").
            drop_oversized: If True, drop sequences longer than max_length.
            shuffle_strategy: Strategy for ordering sequences before packing ("random", "length", or None).
            shuffle_seed: Seed for random shuffling when shuffle_strategy="random".
            flush_strategy: Strategy for flushing bins when num_bins limit is reached.
                "fifo" flushes oldest bins first (default), "fullest" flushes bins with smallest
                remaining capacity first (better packing efficiency).
            pack_payloads: How to combine payloads from multiple samples in a bin.
                "keep_list" keeps payloads as a list (default),
                "torch_tensor" concatenates PyTorch tensors along the first dimension,
                "numpy_array" concatenates NumPy arrays along the first axis,
                or a custom callable (including lambdas) that takes list[Any] and returns Any.
            placement: Placement strategy for this operator.
        """
        op = PackSequences(
            max_length=max_length,
            length_fn=length_fn,
            algorithm=algorithm,
            drop_oversized=drop_oversized,
            shuffle_strategy=shuffle_strategy,
            shuffle_seed=shuffle_seed,
            num_bins=num_bins,
            flush_strategy=flush_strategy,
            pack_payloads=pack_payloads,
        )
        node = self._graph.add("pack_sequences", op, self._tail, placement=placement)
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
    def _delay(
        self,
        *,
        max_delay_ms: float = 2.0,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> "Pipeline":
        from zephon.ops.delay import (
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
        return self

    def _ensure(self) -> None:
        if self._plan is None:
            plan = Planner().make_plan(self._graph)
            self._plan = plan
            self._engine = Engine(plan, self._options, self.ws)

    def to_torch_dataset(self, stateful: bool = True) -> TorchIterableDatasetType:
        if _importlib_util.find_spec("torch.utils.data") is None:
            raise RuntimeError("to_torch_dataset requires 'torch' to be installed.")
        return TorchPipelineIterableDataset(self, stateful=stateful)

    def to_indexable_torch_dataset(self) -> TorchDatasetType:
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
        self._ensure()
        assert self._plan is not None
        supports = getattr(self.ws, "supports_indexing", lambda: False)()
        return self._plan.indexable and supports

    def __iter__(self) -> Iterator[Any]:
        self._ensure()
        assert self._engine is not None
        iterator = self._engine.build_iter()
        final_prefetch = self._options.prefetch_batches or 0
        if final_prefetch > 0:
            iterator = buffered_iterable(
                iterator, final_prefetch, on_stop=self._engine.close
            )
        try:
            yield from self._yield_while_notifying(iterator)
        finally:
            self._engine.close()

    def _yield_while_notifying(
        self, source: Iterable[StreamItem]
    ) -> Iterator[StreamItem]:
        """Wrap an iterable that yields stream elements (SampleRecord or SampleBatch).

        - If item is SampleBatch -> yield item.
        - If item is SampleRecord -> yield the item.
        - Otherwise -> raise TypeError.
        """
        engine = self._engine
        assert engine is not None
        assert self._plan is not None
        use_monotone_notify = self._plan.preserves_cursor_order
        for item in source:
            if isinstance(item, SampleBatch):
                if not item.records:
                    raise TypeError("SampleBatch must contain at least one record")
                lane_id = item.lane_ids[0]
                if use_monotone_notify:
                    # Compute cursor count and max in a single pass —
                    # no list allocation, just scalar comparisons.
                    chunk_ids = item.chunk_ids
                    max_cid = chunk_ids[0]
                    n_cursors = 0
                    max_cursor: SampleCursor | None = None
                    for i, r in enumerate(item.records):
                        cid = chunk_ids[i]
                        if cid > max_cid:
                            max_cid = cid
                            n_cursors = 1
                            max_cursor = r.meta.cursor
                        elif cid == max_cid:
                            n_cursors += 1
                            c = r.meta.cursor
                            if max_cursor is None or c > max_cursor:
                                max_cursor = c
                    engine.notify_monotone(lane_id, max_cid, n_cursors, max_cursor)
                    yield item
                    continue

                contributors: list[ContributorRef] = []
                record_cursor: SampleCursor | None = item.records[-1].meta.cursor
                for record in item.records:
                    contributors.extend(record.meta.contribution_refs())
                engine.notify(lane_id, contributors, record_cursor=record_cursor)
                yield item
            elif isinstance(item, SampleRecord):  # pyright: ignore[reportUnnecessaryIsInstance]
                lane_id = item.meta.lane_id
                if use_monotone_notify:
                    engine.notify_monotone(
                        lane_id,
                        item.meta.chunk_id,
                        1,
                        item.meta.cursor,
                    )
                    if not item.meta.tombstone:
                        yield item
                    continue

                record_cursor = item.meta.cursor
                refs = item.meta.contribution_refs()
                if item.meta.tombstone:
                    engine.notify(lane_id, refs, record_cursor=record_cursor)
                    continue
                engine.notify(lane_id, refs, record_cursor=record_cursor)
                yield item
            else:
                raise TypeError(
                    f"Unsupported element type: {type(item)!r}; "
                    + "expected SampleBatch or SampleRecord"
                )

    def explain(self) -> str:
        self._ensure()
        assert self._plan is not None
        # Compose static plan plus execution graph.
        parts: list[str] = [self._plan.explain]
        if self._engine is not None:
            runtime = self._engine.explain()
            if runtime:
                parts.append("")
                parts.append("Execution Graph:")
                parts.append(runtime)
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

    def checkpoint(self) -> dict[str, Any]:
        self._ensure()
        assert self._engine is not None
        return self._engine.state_dict()

    def restore(self, ckpt: dict[str, Any]) -> None:
        self._ensure()
        assert self._engine is not None
        self._engine.load_state_dict(ckpt, replay=True)

    def __getstate__(self):
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
        - Strip all process-local runtime from the pickled representation:
            * `_engine` : the live runtime (threads, queues, locks)
            * `_plan`   : the derived execution plan (rebuildable from the graph)
          These are set to None in the pickled state.
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
        return d

    def __setstate__(self, state: dict[Any, Any]):
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
        self._engine = None
        self._plan = None
