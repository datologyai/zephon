# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Pyright fixture using ONLY the supported public API (``make typecheck-public``).

The repo's default pyright run covers only ``zephon/**``; this file guards the
public typing surface and is checked by its own invocation.
"""

from __future__ import annotations

from zephon import (
    Dataset,
    InMemoryShard,
    MixtureSpec,
    Pipeline,
    SampleRecord,
    StaticMixtureWorkSource,
    WorkSource,
)
from zephon.build_index import build_index
from zephon.debug import DatasetInspector
from zephon.io import CacheOptions, ParquetRGCacheOptions, StoreOptions
from zephon.observability import (
    ExecutionTrackingMode,
    FetchTimingSummary,
    MTPQueueStats,
    PrefetchTimingSummary,
)
from zephon.ops import BaseOp, DomainGroups, OpContext, OpTraits, StageInfo
from zephon.options import IpcTransport, RuntimeOptions
from zephon.types import SampleBatch, SampleId
from zephon.validation import ValidationReport


def build_pipeline() -> Pipeline:
    ds: Dataset = Dataset.from_dict("d", {0: InMemoryShard([{"text": "x"}])})
    ws: WorkSource = StaticMixtureWorkSource([ds], mixture=MixtureSpec({"d": 1.0}))
    transport: IpcTransport = "socketpair"
    opts = RuntimeOptions(max_workers=4, ipc_transport=transport)
    store: StoreOptions = opts.io_options
    _ = (
        store,
        CacheOptions(),
        ParquetRGCacheOptions(),
        ExecutionTrackingMode.OFF,
        opts,
    )
    return Pipeline(ws).options(max_workers=4, ipc_transport=transport)


def training_batch(batch: SampleBatch) -> None:
    converted = batch.to_training(
        return_labels=True,
        flatten=True,
        exclude_fields=("ids", "texts"),
        return_num_valid_tokens=True,
        return_loss_mask=True,
        return_padding_mask=True,
        return_cu_seqlens=True,
        eos_mask_loss=True,
        eos_token_id=2,
        position_mode="sequence",
        rename_fields={"input_ids": "input"},
    )
    count: int = converted["num_valid_tokens"]
    _ = count


def inspect_dataset() -> None:
    dataset = Dataset.from_dict("d", {0: InMemoryShard([{"text": "x"}])})
    inspector: DatasetInspector = DatasetInspector(dataset)
    with inspector:
        row = inspector.read(shard_id=0, sample_index=0)
        rows = inspector.read_many(shard_id=0, sample_indices=[0])
        _ = (row, rows)


def inspect_stats(pipe: Pipeline) -> MTPQueueStats | None:
    return pipe.mtp_queue_stats()


def inspect_timing(pipe: Pipeline) -> None:
    fetch: FetchTimingSummary | None = pipe.fetch_timing_snapshot()
    if fetch is not None:
        for stage in fetch.iter_stages():
            for shard_key, totals in stage.shard_totals.items():
                key: tuple[int, int] = shard_key
                samples: int = totals.samples
                _ = (key, samples)

    prefetch: PrefetchTimingSummary | None = pipe.prefetch_timing_snapshot()
    if prefetch is not None:
        success_rate: float = prefetch.success_rate
        _ = success_rate


def report(pipe: Pipeline) -> ValidationReport:
    return pipe.validate()


def preflight(pipe: Pipeline) -> ValidationReport:
    return pipe.preflight_tokenizers(strict=False)


def make_index(dataset_dir: str) -> str:
    return str(build_index("parquet", dataset_dir))


def pack_grouped(pipe: Pipeline) -> Pipeline:
    groups = DomainGroups({"code": ["python", "java"]})
    members: dict[str, str] = groups.to_member_map()
    _ = members
    return pipe.pack_sequences(
        max_length=8, homogeneity="group", groups=groups, max_sequences_per_bin=2
    )


def pack_training(pipe: Pipeline) -> Pipeline:
    return pipe.pack_flat(
        max_length=9, algorithm="wrap", pad_token_id=0, max_sequences_per_bin=2
    ).batch(4)


class IdentityOp(BaseOp):
    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True)

    def setup(self, ctx: OpContext) -> None:
        super().setup(ctx)
        info: StageInfo = ctx.stage_info
        _ = (info.stage_index, info.stage_name, info.op_index, info.collect_stats)

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        for e in elems:
            sid: SampleId = e.meta.sample_id
            _ = sid
        return elems
