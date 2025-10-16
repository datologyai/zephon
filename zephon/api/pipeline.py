# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""User-facing pipeline wrapper that layers ergonomics atop core planning."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Iterator, Optional, Protocol, TypeAlias


class _IterableDatasetProto(Protocol):
    def __iter__(self) -> Iterator[Any]: ...


class _DatasetProto(_IterableDatasetProto, Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...


if TYPE_CHECKING:
    try:
        from torch.utils.data import Dataset as _TorchDataset
        from torch.utils.data import IterableDataset as _TorchIterableDataset
    except Exception:  # typing fallback when torch absent
        _TorchIterableDataset = _IterableDatasetProto
        _TorchDataset = _DatasetProto
else:
    _TorchIterableDataset = _IterableDatasetProto
    _TorchDataset = _DatasetProto

TorchIterableDatasetType: TypeAlias = _TorchIterableDataset  # pyright: ignore[reportInvalidTypeForm]
TorchDatasetType: TypeAlias = _TorchDataset  # pyright: ignore[reportInvalidTypeForm]

from zephon.core.constants import SampleBatch, SampleRecord
from zephon.core.engine import Engine, RuntimeOptions
from zephon.core.graph import Graph, Plan
from zephon.core.planner import Planner
from zephon.io.options import StoreOptions
from zephon.ops import Batch, DecodeText, FetchOp, Materialize, TokenizeText
from zephon.utils import buffered_iterable
from zephon.work import WorkSource


class Pipeline:
    """Fluent builder that compiles user ops into an executable pipeline."""

    def __init__(self, work_source: WorkSource) -> None:
        self.ws = work_source
        self._graph = Graph()
        self._plan: Plan | None = None
        self._engine: Engine | None = None
        self._options = RuntimeOptions()
        self._tail = self._graph.add("fetch", FetchOp(), placement="local")

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
        parallelism: Optional[int] = None,
    ) -> "Pipeline":
        op = Batch(microbatch_size, drop_last=drop_last)
        node = self._graph.add(
            "batch", op, self._tail, placement=placement, parallelism=parallelism
        )
        self._tail = node
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

    def to_torch_dataset(self) -> TorchIterableDatasetType:
        try:
            from torch.utils.data import IterableDataset
        except ModuleNotFoundError as exc:
            msg = "to_torch_dataset requires 'torch' to be installed."
            raise RuntimeError(msg) from exc

        pipeline = self

        class _Dataset(IterableDataset):
            def __iter__(self) -> Iterator[Any]:
                yield from pipeline

        return _Dataset()

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
        self, source: Iterable[SampleRecord | SampleBatch]
    ) -> Iterator[SampleRecord | SampleBatch]:
        """Wrap an iterable of SampleBatch | SampleRecord.

        - If item is SampleBatch -> yield item.to_training().
        - If item is SampleRecord -> yield the item unchanged.
        - Otherwise -> raise TypeError.
        """
        engine = self._engine
        assert engine is not None
        for item in source:
            if isinstance(item, SampleBatch):
                assert len(list(set(item.lane_ids))) == 1
                lane_id = item.lane_ids[0]
                max_chunk_id = max(item.chunk_ids)
                max_chunk_samples = [
                    record.meta.sample_id
                    for record in item.records
                    if record.meta.chunk_id == max_chunk_id
                ]
            elif isinstance(item, SampleRecord):  # pyright: ignore[reportUnnecessaryIsInstance]
                lane_id = item.meta.lane_id
                max_chunk_id = item.meta.chunk_id
                max_chunk_samples = [item.meta.sample_id]
            else:
                raise TypeError(
                    f"Unsupported element type: {type(item)!r}; "
                    + "expected SampleBatch or SampleRecord"
                )

            if engine.notify(lane_id, max_chunk_id, max_chunk_samples):
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
        return self._engine.eval_one(sample_id)

    def checkpoint(self) -> dict[str, Any]:
        self._ensure()
        assert self._engine is not None
        return self._engine.state_dict()

    def restore(self, ckpt: dict[str, Any]) -> None:
        self._ensure()
        assert self._engine is not None
        self._engine.load_state_dict(ckpt, replay=True)
