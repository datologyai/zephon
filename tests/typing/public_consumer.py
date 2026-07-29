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
from zephon.io import CacheOptions, StoreOptions
from zephon.observability import ExecutionTrackingMode, MTPQueueStats
from zephon.ops import BaseOp, OpContext, OpTraits, StageInfo
from zephon.options import IpcTransport, RuntimeOptions
from zephon.types import SampleId
from zephon.validation import ValidationReport


def build_pipeline() -> Pipeline:
    ds: Dataset = Dataset.from_dict("d", {0: InMemoryShard([{"text": "x"}])})
    ws: WorkSource = StaticMixtureWorkSource([ds], mixture=MixtureSpec({"d": 1.0}))
    transport: IpcTransport = "socketpair"
    opts = RuntimeOptions(max_workers=4, ipc_transport=transport)
    store: StoreOptions = opts.io_options
    _ = (store, CacheOptions(), ExecutionTrackingMode.OFF, opts)
    return Pipeline(ws).options(max_workers=4, ipc_transport=transport)


def inspect_stats(pipe: Pipeline) -> MTPQueueStats | None:
    return pipe.mtp_queue_stats()


def report(pipe: Pipeline) -> ValidationReport:
    return pipe.validate()


def make_index(dataset_dir: str) -> str:
    return str(build_index("parquet", dataset_dir))


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
