# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""User-facing pipeline wrapper that layers ergonomics atop core planning."""

import importlib.util as _importlib_util
from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Optional,
    Protocol,
    TypeAlias,
    cast,
)

from zephon.core.constants import (
    EngineSample,
    SampleBatch,
    SampleId,
    SamplePayload,
    SampleRecord,
    StreamItem,
)
from zephon.core.engine import Engine, RuntimeOptions
from zephon.core.graph import Graph, Plan
from zephon.core.planner import Planner
from zephon.io.options import StoreOptions
from zephon.observability import (
    ExecutionTrackingMode,
    MetricsSinkConfig,
)
from zephon.ops import (
    Batch,
    DecodeText,
    FetchOp,
    MapTransform,
    Materialize,
    TokenizeText,
)
from zephon.utils import buffered_iterable
from zephon.utils.torch_compat import detect_loader_kind
from zephon.work import WorkSource


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
        self._fetch_node = self._graph.add("fetch", FetchOp(), placement="local")
        self._tail = self._fetch_node

    def fetch_parallelism(self, parallelism: int | None) -> "Pipeline":
        """Override the implicit FetchOp parallelism."""
        if parallelism is None:
            parallelism = max(1, self._fetch_node.op.traits().parallelism)
        elif parallelism < 1:
            raise ValueError("Fetch parallelism must be >= 1.")
        self._fetch_node.parallelism = parallelism
        return self

    def fetch(self, parallelism: int | None) -> "Pipeline":
        return self.fetch_parallelism(parallelism)

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

        Args:
            transform_fn: Callable that transforms the payload.
                If it returns None and drop_none=True, the sample is filtered out.
            drop_none: If True, drop samples where transform_fn returns None.
            placement: Placement hint for this operator.
            parallelism: Override default parallelism for this operator.

        Returns:
            Self for method chaining.
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

    def tokenize(
        self,
        tokenizer: Any | None = None,
        tokenizer_id: str | None = None,
        *,
        field: str = "text",
        add_attention_mask: bool = True,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> "Pipeline":
        op = TokenizeText(
            tokenizer, tokenizer_id, field=field, add_attention_mask=add_attention_mask
        )
        node = self._graph.add(
            "tokenize", op, self._tail, placement=placement, parallelism=parallelism
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
        for item in source:
            if isinstance(item, SampleBatch):
                assert len(list(set(item.lane_ids))) == 1
                lane_id = item.lane_ids[0]
                max_chunk_id = max(item.chunk_ids)
                progress_cursors = [
                    record.meta.cursor
                    for record in item.records
                    if record.meta.chunk_id == max_chunk_id
                ]
            elif isinstance(item, SampleRecord):  # pyright: ignore[reportUnnecessaryIsInstance]
                lane_id = item.meta.lane_id
                max_chunk_id = item.meta.chunk_id
                progress_cursors = [item.meta.cursor]
            else:
                raise TypeError(
                    f"Unsupported element type: {type(item)!r}; "
                    + "expected SampleBatch or SampleRecord"
                )

            engine.notify(lane_id, max_chunk_id, progress_cursors)
            yield item

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
                value = cast(EngineSample, (sid, 0, 0, 0))
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
