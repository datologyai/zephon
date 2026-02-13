# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Runtime execution engine that wires plans to concrete stage runners.

Canonical scheduling and mapping strategies
-------------------------------------------
The engine builds a canonical schedule over a fixed number of logical data
parallel replicas (``canonical_replicas``). At runtime, those replicas are
mapped to the set of physical ranks in this run via a mapping strategy. This
decoupling enables elastic continuation: we can resume with a different number
of ranks while keeping the canonical order.
Two built-in mapping strategies are supported via ``RuntimeOptions.mapping_strategy``:
- "contiguous" (default)
  - Assigns each rank a contiguous block of canonical replicas.
    Example: canonical_replicas=8, num_ranks=3 → rank→lanes:
      r0: [0,1,2], r1: [3,4,5], r2: [6,7]
  - Why: maximizes locality for chunk/block caches and minimizes cross-rank
    interleaving. Useful when shards/blocks benefit from staying together.
- "interleaved"
  - Assigns replica i to rank (i % num_ranks). Example with 8 and 3 →
    r0: [0,3,6], r1: [1,4,7], r2: [2,5]
  - Why: balances progress more evenly across ranks and often reduces tail
    variance. Helpful when ranks scale up/down between phases or when shards
    are heterogeneous.
Both strategies are deterministic and equivalent with respect to the canonical
order; only the per-rank partitioning differs. BOTH strategies will lead to identical
global batches the mapping strategy only decides which rank processes which parts.
The engine also inserts a lane-merging distributor at the tail only when a rank owns multiple replicas
and batching is present, so 1:1 rank↔replica runs remain a clean, single-lane dataflow.
"""

import json
import math
import multiprocessing as mp
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import warnings
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, TypeVar, cast

from zephon.io.storage import RouterStorageBackend

T = TypeVar("T")

from zephon.core.constants import (
    ChunkId,
    ContributorRef,
    EngineSample,
    LaneId,
    LanePtr,
    RunnerStageOut,
    SampleBatch,
    SampleCursor,
    SampleId,
    SampleRecord,
    StreamItem,
)
from zephon.core.graph import Plan, Stage
from zephon.core.replay import ReplayConfigService
from zephon.core.world import World
from zephon.io.options import StoreOptions
from zephon.observability import ExecutionTrackingMode, MetricsSinkConfig
from zephon.observability.collector import CollectorConfig, PipelineCollector
from zephon.observability.emitter import MetricsReporter
from zephon.ops.replay_filter import ReplayFilter
from zephon.runners.inline import InlineStageRunner
from zephon.runners.process import ProcessStageRunner
from zephon.runners.threads import ThreadStageRunner
from zephon.work import MixtureReadConfig, WorkSource
from zephon.work.base import WorkChunk


def inside_torch_worker() -> bool:
    """Return True if the current process is a PyTorch DataLoader worker."""
    try:
        from torch.utils.data import get_worker_info

        return get_worker_info() is not None
    except Exception:
        return False


def get_torch_worker_info() -> tuple[int, int]:
    """Return (worker_id, workers_per_rank) if torch is available; else (0,1)."""
    try:
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        if info is None:
            return 0, 1
        return int(info.id), int(info.num_workers)
    except Exception:
        return 0, 1


def _extract_lane_id(item: StreamItem) -> int:
    if isinstance(item, SampleRecord):
        return int(item.meta.lane_id)
    assert isinstance(item, SampleBatch)
    lids = item.lane_ids
    if not lids:
        raise RuntimeError("Empty batch has no lane_id")
    # Pipeline guarantees one lane per batch at the tail
    return int(lids[0])


def _noop(*args: Any, **kwargs: Any) -> None:
    pass


class OffsetBitmap:
    """Fast bitmap using 1 byte per offset (0 or 1)."""

    __slots__ = ("size", "_bits")

    def __init__(self, size: int) -> None:
        self.size = size
        self._bits = bytearray(size)

    def set(self, offset: int) -> None:
        self._bits[offset] = 1

    def is_set(self, offset: int) -> bool:
        return self._bits[offset] == 1


def _call_engine_cleanup(engine_ref: weakref.ReferenceType["Engine"]):
    """Cleanup finalizer for Engine - runs at GC or interpreter shutdown.

    This ensures semaphore cleanup happens before the resource tracker
    checks for leaks, even in free-threaded Python where GC is deferred.

    Calls close() first (to clean up runners, queues, and semaphores),
    then _clean_merged() (to clean up temporary files).
    """
    try:
        engine = engine_ref()
        if engine is None:
            return
        try:
            # Close runners first (handles semaphore cleanup)
            close_fn = getattr(engine, "close", None)
            if callable(close_fn):
                close_fn()
        finally:
            # Always clean merged files, even if close() fails
            cm = getattr(engine, "_clean_merged", None)
            if callable(cm):
                cm()
    except Exception:
        pass


DEFAULT_RUN_ID = "default_run_id"
_RELOAD_SUFFIX = re.compile(r"-(\d+)$")


@dataclass
class RuntimeOptions:
    """User-tunable knobs that influence how the engine constructs runners."""

    runner: str | None = None  # Default runner. Typically auto-inferred.
    run_id: str = DEFAULT_RUN_ID
    per_stage_runner: dict[int, str] = field(
        default_factory=dict
    )  # Manual override for runner per-stage. Mostly useful for debugging and advanced usage.
    allow_subprocess_in_worker: bool = False  # TODO(MaxiBoether): Implement this.
    mp_context: Any = mp.get_context("spawn")
    worker_allocation: Literal[
        "fit_to_ops", "per_stage_fixed", "global", "autotune"
    ] = "fit_to_ops"
    stage_weighting: Literal["equal", "by_declared_parallelism"] = (
        "by_declared_parallelism"
    )
    max_workers: int = (
        8  # max_workers per stage OR global, depending on worker_allocation.
    )
    deterministic: bool = True
    prefetch_batches: int | None = None
    default_stage_prefetch: int = 0
    per_stage_prefetch: dict[int, int] = field(default_factory=dict)
    op_queue_capacity: int = 16  # maximum size of inflight items between ops.
    mixture_config: MixtureReadConfig | None = None
    io_options: StoreOptions = field(default_factory=StoreOptions)
    # Expert knob:
    # Keep latency-based flush in deterministic mode when True unless a stage contains
    # a batch-shape sensitive operator (in which case we auto-disable it for that stage).
    # When False, latency flush is always disabled in deterministic mode.
    allow_latency_flush_in_deterministic: bool = True

    # === Global Coordination ===
    # Total number of ranks (GPUs) in the distributed job.
    world_size: int = 1
    # Unique identifier for this rank (0 to world_size-1).
    global_rank: int = 0

    # === Data Partitioning ===
    # Number of data parallel groups (data partitions).
    # Defaults to world_size (1D parallelism) if not specified.
    dp_degree: int | None = None
    # Which data partition this rank reads (0 to dp_degree-1).
    # Defaults to global_rank (1D parallelism) if not specified.
    dp_group_id: int | None = None

    # === Logical Parallelism ===
    # Number of canonical lanes (for elasticity). Defaults to dp_degree.
    canonical_replicas: int | None = None
    # How to map canonical replicas to dp groups:
    # - 'contiguous': dp groups own contiguous blocks of replicas (locality-friendly)
    # - 'interleaved': replicas are round-robin across dp groups (balanced progress)
    mapping_strategy: Literal["contiguous", "interleaved"] | None = None

    # Autotune placeholders (intentionally not implemented yet)
    autotune_config: dict[str, Any] | None = None  # e.g., {"target_util": 0.3, ...}
    # Where all workers/ranks dump their local state. For multi-node, must be a shared filesystem
    # (e.g., NFS) or cloud storage (s3://bucket/path or gs://bucket/path).
    aggregate_dir: str | None = None
    # How long to wait for all contributors and for the merged file.
    aggregate_timeout_s: float = 30.0
    # Observability controls.
    execution_tracking: ExecutionTrackingMode = ExecutionTrackingMode.OFF
    metrics_sink_config: MetricsSinkConfig | None = None


# =============================================================================
# Distributed Parallelism Model
# =============================================================================
#
# Zephon separates two orthogonal concerns for distributed training:
#
# 1. GLOBAL COORDINATION (world_size, global_rank)
#    - Used for: leader election, checkpoint aggregation, state file naming
#    - global_rank must be unique per rank
#    - Leader is global_rank == 0
#
# 2. DATA PARTITIONING (dp_degree, dp_group_id)
#    - Used for: determining which data each rank should read
#    - Multiple ranks can share the same dp_group_id (they get same data)
#    - Lanes assigned based on dp_group_id, not global_rank
#
# With 3D parallelism (DP x TP x PP), ranks are organized as:
#    - dp_degree groups, each needing different data
#    - mp_degree (= TP x PP) ranks per group, all needing SAME data
#    - world_size = dp_degree x mp_degree
#
# Example: PP=2, DP=2, TP=2 (8 GPUs)
#    GPU 0: global_rank=0, dp_group_id=0  --+
#    GPU 1: global_rank=1, dp_group_id=0    | All read same data (lane 0)
#    GPU 4: global_rank=4, dp_group_id=0    |
#    GPU 5: global_rank=5, dp_group_id=0  --+
#    GPU 2: global_rank=2, dp_group_id=1  --+
#    GPU 3: global_rank=3, dp_group_id=1    | All read same data (lane 1)
#    GPU 6: global_rank=6, dp_group_id=1    |
#    GPU 7: global_rank=7, dp_group_id=1  --+
#
# CURRENT IMPLEMENTATION: DP-only awareness
#    - All ranks with same dp_group_id independently read same lanes
#    - Redundant I/O (mp_degree x reads per DP group) but correct
#    - Simple and data loading rarely the bottleneck
#
# FUTURE OPTIMIZATION: MP awareness (not implemented)
#    If I/O becomes a bottleneck, could add:
#    - mp_degree: int  # TP x PP, ranks per DP group
#    - mp_rank: int    # unique ID within DP group (0 to mp_degree-1)
#
#    This would enable:
#    - Only mp_rank==0 actually reads data
#    - Others yield None, training framework broadcasts
#    - Reduces I/O by factor of mp_degree
#
#    Additional consideration for PP stages:
#    - First PP stage needs input tokens
#    - Last PP stage needs labels
#    - But both derive from same token window, so can't separate reads
#    - Within each TP group (same PP stage), only one needs to read
#
#    Complexity: requires coordination with training framework for broadcast.
#    Only implement if I/O proven to be bottleneck.
# =============================================================================


class Engine:
    """Bind a `Plan` to concrete runners and orchestrate streaming execution."""

    def __init__(self, plan: Plan, opts: RuntimeOptions, work: WorkSource) -> None:
        """Initialize stage runners and prepare to stream work items."""
        self._plan = plan
        self._preserves_cursor_order = bool(plan.preserves_cursor_order)
        base_ctx: dict[str, Any] = {
            "datasets_by_id": work.datasets_by_id,
            "io_options": opts.io_options,
        }
        # Mixture query service for EnsureMixture operator
        base_ctx["get_chunk_mixture"] = self._get_chunk_mixture
        base_ctx["get_component_name"] = self._get_component_name
        base_ctx["get_component_id"] = self._get_component_id
        self._replay_config = ReplayConfigService()
        self._ctx = base_ctx
        self._ctx["replay_state_service"] = self._replay_config
        self._opts = opts
        self._mp_context = self._resolve_mp_context(self._opts.mp_context)
        self._world = self._build_world()
        self.inflight_chunks_per_lane: dict[LaneId, dict[ChunkId, Any]] = defaultdict(
            dict
        )
        self._lane_progress: dict[LaneId, LanePtr] = defaultdict(LanePtr)
        self._lane_last_cursor: dict[LaneId, SampleCursor | None] = {}
        self._offset_done: dict[LaneId, dict[ChunkId, OffsetBitmap]] = defaultdict(dict)
        self._offset_done_count: dict[LaneId, dict[ChunkId, int]] = defaultdict(dict)

        self._lane_next_cid: dict[LaneId, int] = defaultdict(int)
        self._warned_once_about_runid = False
        self._checkpoint_reload_count = 0
        self._checkpoint_lock = threading.Lock()
        self._rr_next_idx: dict[str, int] = {}

        # Component ID mapping for mixture tracking (string -> int)
        self._component_to_id: dict[str, int] = {}
        self._id_to_component: list[str] = []
        self._component_lock = threading.Lock()
        # Per-chunk mixture storage: (lane_id, chunk_id) -> {component_id: weight}
        self._chunk_mixtures: dict[tuple[LaneId, ChunkId], dict[int, float]] = {}
        self._mixture_lock = threading.Lock()

        self._work = work
        self._lane_ws: dict[LaneId, WorkSource] = {}
        for lane in self._world.lanes_for_dp_group[self._world.dp_group_id]:
            self._lane_ws[lane] = self._work.clone_for_lane(
                lane, canonical_replicas=self._world.canonical_replicas
            )
            self._lane_last_cursor.setdefault(lane, None)

        self._publish_replay_snapshot()

        tracking_mode = self._opts.execution_tracking
        self._collector: PipelineCollector | None = None
        self._metrics_reporter: MetricsReporter | None = None
        self._metrics_started = False
        self._ctx["emit_fetch_metrics"] = _noop
        self._ctx["emit_prefetch_metrics"] = _noop
        self._ctx["emit_backpressure_metrics"] = _noop
        self._ctx["record_node_metrics"] = _noop

        if tracking_mode != ExecutionTrackingMode.OFF:
            metrics_sink_config = self._opts.metrics_sink_config
            report_interval = (
                metrics_sink_config.flush_interval_s
                if metrics_sink_config is not None
                else 5.0
            )
            collector_config = CollectorConfig(
                tracking_mode=tracking_mode,
                sink=metrics_sink_config,
                report_interval_s=report_interval,
                plan_id=self._plan.plan_id,
            )
            self._collector = PipelineCollector(collector_config)
            worker_id, _ = get_torch_worker_info()
            self._metrics_reporter = MetricsReporter(
                self._collector,
                metrics_sink_config,
                rank_id=self._world.global_rank,
                worker_id=worker_id,
            )
            if self._collector.tracking_mode.collects_nodes:
                self._ctx["emit_fetch_metrics"] = self._emit_fetch_metrics
                self._ctx["emit_prefetch_metrics"] = self._emit_prefetch_metrics
                self._ctx["emit_backpressure_metrics"] = self._emit_backpressure_metrics
            self._ctx["record_node_metrics"] = self._collector.record

        # Stage runners are stored heterogeneously.
        self._runners: list[
            ThreadStageRunner | InlineStageRunner | ProcessStageRunner
        ] = []
        self._build_runners()

        self._using_fresh_tmp = False
        if self._world.world_size == 1:
            # Single-node: auto if not provided
            base = self._opts.aggregate_dir or os.path.join(
                tempfile.gettempdir(), f"zephon_state_r{self._world.global_rank}"
            )
            self._using_fresh_tmp = (
                self._opts.aggregate_dir is None or self._opts.aggregate_dir == ""
            )
        else:
            # Multi-node: must be provided (shared FS or cloud storage)
            if not self._opts.aggregate_dir:
                raise RuntimeError(
                    "RuntimeOptions.aggregate_dir must be set for multi-node checkpoint aggregation."
                )
            base = self._opts.aggregate_dir

        # Initialize storage backend for checkpoint I/O
        self._agg_backend = RouterStorageBackend()

        # Detect cloud storage and adjust timeouts for higher latency
        base = str(base)  # Handle Path objects
        self._agg_is_cloud = self._agg_backend.is_cloud_path(base)
        if self._agg_is_cloud:
            self._agg_timeout_s = max(self._opts.aggregate_timeout_s, 60.0)
            self._agg_poll_base = 0.5  # 500ms base poll interval
        else:
            self._agg_timeout_s = self._opts.aggregate_timeout_s
            self._agg_poll_base = 0.05  # 50ms base poll interval
            # Resolve local paths; cloud paths stay as-is
            base = str(Path(base).resolve())

        self._agg_base = base.rstrip("/")
        self._agg_backend.mkdir(self._agg_base, parents=True, exist_ok=True)
        self._last_round_id: str | None = None
        self._previous_merged_file: str | None = None

        # Use a weakref finalizer so cleanup runs without pinning the engine until interpreter exit.
        # This calls close() to clean up runners/queues/semaphores, then _clean_merged() for temp files.
        self._cleanup_finalizer = weakref.finalize(
            self, _call_engine_cleanup, weakref.ref(self)
        )

        bs = self._plan.batch_size_hint
        if bs is not None:
            cs = self._work.chunk_size_hint()
            assert cs is not None, (
                "No chunk size hint for current work source, determinism breaks potentially"
            )

    @property
    def _round_file(self) -> str:
        return f"{self._agg_dir}/round.current"

    @property
    def _agg_dir(self) -> str:
        def _normalize_run_dir_name(
            name: str,
        ) -> str:  # Drop a trailing "-<int>" if present
            m = _RELOAD_SUFFIX.search(name)
            return name[: m.start()] if m else name

        run_id = self._opts.run_id
        using_default = False
        # Names (not full paths) of subdirectories in _agg_base, used to warn
        # about potential run ID collisions when using the default run ID.
        existing_subdir_names: list[str] = []

        if run_id.startswith(DEFAULT_RUN_ID):
            existing_subdir_names = self._agg_backend.listdir(self._agg_base)
            using_default = True
            # If users don't provide a run ID, we want to help them not shoot themselves in the foot.
            run_id = f"{run_id}-{self._opts.canonical_replicas}-{self._world.world_size}-{self._plan.plan_id}"

        # Avoid having to use a new run id with every reload
        run_id = f"{run_id}-{self._checkpoint_reload_count}"
        current_run_dir = f"{self._agg_base}/{run_id}"
        cur_norm = _normalize_run_dir_name(run_id)
        result = f"{current_run_dir}/.zephon_agg"

        if (
            not self._using_fresh_tmp
            and using_default
            and not self._warned_once_about_runid
            and existing_subdir_names
        ):
            existing_norms = {
                _normalize_run_dir_name(name) for name in existing_subdir_names
            }
            if existing_norms != {cur_norm}:
                # Potentially we can also only warn if the plan id or canonical replicas change since that probably really causes a semantic change but we better just tell the user early this is not the. best idea.
                print(
                    f"Warning! No run id has been supplied. This can cause issues in checkpointing if the same aggregate_dir ({self._agg_base}) is used across multiple runs. Your current supplied directory contains data from other runs (or your world changed), which might indicate that you re-use that directory (or use it for other purposes as well). Zephon adjusted the run id to {run_id} to avoid problems, but if you choose to run exactly the same pipeline twice in the samed directory, issues might still occur without providing a run id.\n\nOffending subdirs: {existing_subdir_names}",
                    file=sys.stderr,
                )
                self._warned_once_about_runid = True

        return result

    def __del__(self) -> None:
        try:
            self._clean_merged()
        except Exception:
            pass

    def _clean_merged(self) -> None:
        if self._previous_merged_file is not None:
            self._log(
                f"We are cleaning up the previous checkpoint synchronization file ({self._previous_merged_file}). If you face a timeout error after this, it means that your main rank/worker exits before all workers have consumed the checkpoint. You should ensure all workers have completed the checkpoint (e.g., via a barrier) if you checkpoint at the end of training. If no error occurs, all is well."
            )
            self._agg_backend.delete(self._previous_merged_file)
            self._previous_merged_file = None

    def __getstate__(self):
        # We rather fail explicitly here for now to avoid problems with runners.
        raise RuntimeError("Engine must not be pickled; build it inside a worker.")

    def _resolve_mp_context(self, ctx_spec: BaseContext | str | None) -> BaseContext:
        if ctx_spec is None:
            return mp.get_context("spawn")
        if isinstance(ctx_spec, str):
            return mp.get_context(ctx_spec)
        return ctx_spec

    def _resolve_parallelism_params(self) -> None:
        """Resolve parallelism parameters with defaults for 1D parallelism.

        For 1D data parallelism, users only need to set world_size and global_rank;
        dp_* params auto-derive to avoid redundant configuration.
        """
        opts = self._opts

        # Validate world_size before using it for defaults
        if opts.world_size < 1:
            raise ValueError(f"world_size ({opts.world_size}) must be >= 1")

        # Default dp params to 1D parallelism if not specified
        if opts.dp_degree is None:
            opts.dp_degree = opts.world_size
        if opts.dp_group_id is None:
            opts.dp_group_id = opts.global_rank

        # Default canonical_replicas to dp_degree
        if opts.canonical_replicas is None:
            opts.canonical_replicas = opts.dp_degree

        # === Strict Validation (errors) ===

        if not (0 <= opts.global_rank < opts.world_size):
            raise ValueError(
                f"global_rank ({opts.global_rank}) must be in [0, world_size ({opts.world_size}))"
            )

        if opts.dp_degree < 1:
            raise ValueError(f"dp_degree ({opts.dp_degree}) must be >= 1")

        if opts.dp_degree > opts.world_size:
            raise ValueError(
                f"dp_degree ({opts.dp_degree}) cannot exceed world_size ({opts.world_size})"
            )

        if opts.world_size % opts.dp_degree != 0:
            raise ValueError(
                f"world_size ({opts.world_size}) must be divisible by dp_degree ({opts.dp_degree}). "
                + "In 3D parallelism, mp_degree (= TP × PP = world_size / dp_degree) must be an integer."
            )

        if not (0 <= opts.dp_group_id < opts.dp_degree):
            raise ValueError(
                f"dp_group_id ({opts.dp_group_id}) must be in [0, dp_degree ({opts.dp_degree}))"
            )

        if opts.canonical_replicas < 1:
            raise ValueError(
                f"canonical_replicas ({opts.canonical_replicas}) must be >= 1"
            )

        if opts.canonical_replicas < opts.dp_degree:
            raise ValueError(
                f"canonical_replicas ({opts.canonical_replicas}) must be >= dp_degree ({opts.dp_degree})"
            )

        # === Warnings (valid but unusual configurations) ===

        mp_degree = opts.world_size // opts.dp_degree
        if mp_degree > 1 and (mp_degree & (mp_degree - 1)) != 0:
            # mp_degree is not a power of 2 (and > 1)
            warnings.warn(
                f"[zephon] mp_degree (= world_size / dp_degree = {opts.world_size} / "
                + f"{opts.dp_degree} = {mp_degree}) is not a power of 2. In 3D parallelism, "
                + "mp_degree = TP × PP is typically a power of 2. "
                + "This may indicate a configuration error.",
                RuntimeWarning,
                stacklevel=4,
            )

        if opts.canonical_replicas % opts.dp_degree != 0:
            warnings.warn(
                f"[zephon] canonical_replicas ({opts.canonical_replicas}) is not divisible by "
                + f"dp_degree ({opts.dp_degree}). This will result in uneven lane distribution "
                + "across DP groups, which may cause load imbalance.",
                RuntimeWarning,
                stacklevel=4,
            )

    def _build_world(self) -> World:
        worker_id, workers_per_rank = get_torch_worker_info()

        # Resolve defaults for parallelism parameters
        self._resolve_parallelism_params()

        # After _resolve_parallelism_params(), these are guaranteed to be int
        assert self._opts.dp_degree is not None
        assert self._opts.dp_group_id is not None
        assert self._opts.canonical_replicas is not None

        dp_degree: int = self._opts.dp_degree
        dp_group_id: int = self._opts.dp_group_id
        canonical_replicas: int = self._opts.canonical_replicas

        strategy = (self._opts.mapping_strategy or "contiguous").lower()

        # Build lane mapping based on dp_group_id (not global_rank)
        mapping: dict[int, list[int]] = {}
        if strategy == "interleaved":
            for dp_id in range(dp_degree):
                mapping[dp_id] = [
                    lane
                    for lane in range(canonical_replicas)
                    if lane % dp_degree == dp_id
                ]
        else:  # contiguous
            base = canonical_replicas // dp_degree
            rem = canonical_replicas % dp_degree
            start = 0
            for dp_id in range(dp_degree):
                count = base + (1 if dp_id < rem else 0)
                mapping[dp_id] = list(range(start, start + count))
                start += count

        return World(
            canonical_replicas=canonical_replicas,
            worker_id=worker_id,
            workers_per_rank=workers_per_rank,
            # Global coordination
            world_size=self._opts.world_size,
            global_rank=self._opts.global_rank,
            # Data partitioning
            dp_degree=dp_degree,
            dp_group_id=dp_group_id,
            lanes_for_dp_group=mapping,
        )

    def explain(self) -> str:
        """ASCII execution graph showing where buffers exist and their sizes.

        Visualizes:
        - Per-op input queues inside stages (as -[in_q=Q]-> between ops)
        - Stage output queue (as --[stage_out=Q]-->)
        - Boundary prefetch between stages (as ==[prefetch=Q]==>)
        - Final pipeline prefetch to the consumer
        """
        lines: list[str] = []

        mode = self._opts.worker_allocation
        if mode == "global":
            lines.append(
                f"Allocation=global total={self._opts.max_workers} weighting={self._opts.stage_weighting}"
            )
        elif mode == "per_stage_fixed":
            lines.append(
                f"Allocation=per_stage_fixed per_stage={self._opts.max_workers}"
            )
        else:
            lines.append("Allocation=fit_to_ops (cap=sum(node.parallelism), min 1/op)")
        bookkeeping = (
            "simple chunk-watermark (preserves_cursor_order=True)"
            if self._preserves_cursor_order
            else "contributor-aware (packing/shuffle-safe)"
        )
        lines.append(f"Bookkeeping={bookkeeping}")

        for idx, (stage, runner) in enumerate(zip(self._plan.stages, self._runners)):
            runner_kind = "unknown"
            op_in_q: int | None = None
            stage_prefetch: int | None = None
            cap: int | None = None
            forward_mode = (
                "microbatches"
                if getattr(runner, "_emit_microbatches", False)
                else "stream_items"
            )

            if isinstance(runner, ThreadStageRunner):  # pyright: ignore[reportUnnecessaryIsInstance]
                runner_kind = "threads"
                op_in_q = getattr(runner, "_queue_capacity", None)
                stage_prefetch = getattr(runner, "_prefetch_capacity", None)
                cap = getattr(runner, "_max_workers", None)  # <- show cap
            elif isinstance(runner, ProcessStageRunner):  # pyright: ignore[reportUnnecessaryIsInstance]
                runner_kind = "process"
                op_in_q = getattr(runner, "_queue_capacity", None)
                stage_prefetch = getattr(runner, "_prefetch_capacity", None)
                cap = getattr(runner, "_max_workers", None)
            elif isinstance(runner, InlineStageRunner):  # pyright: ignore[reportUnnecessaryIsInstance]
                runner_kind = "inline"
                stage_prefetch = getattr(runner, "_prefetch_capacity", None)
                cap = getattr(runner, "_max_workers", None)

            # Header with placement and runner only (buffers are shown inline)
            header = (
                f"Stage[{idx}] place={stage.placement} runner={runner_kind} "
                + f"cap={cap} mode={forward_mode}"
            )
            if isinstance(runner, ProcessStageRunner):  # pyright: ignore[reportUnnecessaryIsInstance]
                ipc = getattr(runner, "_ipc_batch_size", None)
                direct = getattr(runner, "_single_op_direct_ipc", False)
                header += (
                    f" first_op_ipc_batch={ipc} "
                    + f"direct_ipc={'enabled' if direct else 'disabled'}"
                )
            lines.append(header)

            # Inside-stage ops with input buffers between them
            nodes = [f"{nd.name}@p{nd.parallelism}" for nd in stage.nodes]
            if not nodes:
                lines.append("  [empty stage]")
            else:
                if len(nodes) == 1:
                    lines.append(f"  {nodes[0]}")
                else:
                    if op_in_q is not None:
                        connector = f" -[in_q={op_in_q}]-> "
                    else:
                        connector = " -> "
                    lines.append("  " + connector.join(nodes))

            # Stage output queue capacity follows ThreadStageRunner logic
            # out_capacity = max(1, stage_prefetch or op_in_q)
            out_q: int | None = None
            if op_in_q is not None:
                sp = stage_prefetch or 0
                out_q = max(1, sp or op_in_q)
            elif stage_prefetch and stage_prefetch > 0:
                out_q = max(1, stage_prefetch)

            # Boundary: to next stage or pipeline end
            if out_q is not None:
                lines.append(f"  --[stage_out={out_q}]-->")
            else:
                lines.append("  --[stage_out=?]-->")

            is_last = idx + 1 == len(self._runners)
            if not is_last:
                if stage_prefetch and stage_prefetch > 0:
                    lines.append(f"  ==[prefetch={stage_prefetch}]==> Stage[{idx + 1}]")
                else:
                    lines.append(f"  ==> Stage[{idx + 1}]")
            else:
                final_prefetch = self._opts.prefetch_batches or 0
                if final_prefetch > 0:
                    lines.append(
                        f"  ==[final_prefetch={final_prefetch}]==> pipeline_end"
                    )
                else:
                    lines.append("  ==> pipeline_end")

        return "\n".join(lines)

    @staticmethod
    def _apportion(total: int, weights: list[int]) -> list[int]:
        """Split `total` into integer parts proportional to `weights`.

        Uses largest-remainder (Hamilton) method; guarantees sum(parts) == total.
        If all weights are non-positive, falls back to an equal split.
        """
        if total <= 0 or not weights:
            return [0] * len(weights)
        wpos = [max(0, int(w)) for w in weights]
        wsum = sum(wpos)
        n = len(wpos)
        if wsum == 0:
            base = total // n
            parts = [base] * n
            for i in range(total - base * n):
                parts[i] += 1
            return parts
        quotas = [total * (w / wsum) for w in wpos]
        floors = [int(math.floor(q)) for q in quotas]
        remaining = total - sum(floors)
        order = sorted(range(n), key=lambda i: quotas[i] - floors[i], reverse=True)
        for i in range(remaining):
            floors[order[i]] += 1
        return floors

    def _build_runners(self) -> None:
        """Instantiate per-stage runners according to placement and options."""
        inside_worker = inside_torch_worker()
        default_runner = self._opts.runner or "auto"
        mode = self._opts.worker_allocation
        num_stages = len(self._plan.stages)
        if mode == "autotune":
            raise NotImplementedError(
                "worker_allocation='autotune' is reserved for future auto-tuning. "
                + "Use 'per_stage_fixed' or 'global' for now."
            )

        if mode == "global":
            total = self._opts.max_workers
            if total < num_stages:
                warnings.warn(
                    f"[zephon] max_workers_total={total} < number of stages={num_stages}; "
                    + f"bumping to {num_stages} (1 thread per stage).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                total = num_stages

            if self._opts.stage_weighting == "equal":
                weights = [1 for _ in range(num_stages)]
            else:  # by_declared_parallelism
                weights = [
                    self._stage_parallelism(stage) for stage in self._plan.stages
                ]
            per_stage_caps = self._apportion(total, weights)
        elif mode == "per_stage_fixed":
            # per_stage_fixed: same cap for each stage
            cap = self._opts.max_workers
            per_stage_caps = [cap for _ in range(num_stages)]
        else:
            per_stage_caps = [
                self._stage_parallelism(stage) for stage in self._plan.stages
            ]

        for idx, stage in enumerate(self._plan.stages):
            is_last_stage = idx == num_stages - 1
            cap_for_stage = per_stage_caps[idx]

            prefetch = self._opts.per_stage_prefetch.get(
                idx, self._opts.default_stage_prefetch
            )
            chosen = self._opts.per_stage_runner.get(idx, default_runner)
            if chosen == "auto":
                if inside_worker and not self._opts.allow_subprocess_in_worker:
                    chosen = "threads"
                elif stage.placement == "remote":
                    chosen = "remote"
                else:
                    chosen = "threads"
            if (
                inside_worker
                and chosen == "process"
                and not self._opts.allow_subprocess_in_worker
            ):
                chosen = "threads"

            if getattr(stage, "runner_hint", None) == "inline":
                chosen = "inline"
            allow_latency = self._opts.allow_latency_flush_in_deterministic
            if self._opts.deterministic:
                has_sensitive = any(
                    getattr(nd.op.traits(), "batch_shape_sensitive", False)
                    for nd in stage.nodes
                )
                if has_sensitive and allow_latency:
                    print(
                        "Deterministic mode: disabling time-based flush for Stage[%d] due to batch-shape sensitive op.",
                        idx,
                    )
                    allow_latency = False

            runner_tracking_mode = (
                self._collector.tracking_mode
                if self._collector is not None
                else ExecutionTrackingMode.OFF
            )
            output_mode = "stream_items" if is_last_stage else "microbatches"

            if chosen == "threads":
                self._runners.append(
                    ThreadStageRunner(
                        stage,
                        self._ctx,
                        cap_for_stage,
                        prefetch_capacity=prefetch,
                        deterministic=self._opts.deterministic,
                        allow_latency_flush_in_deterministic=allow_latency,
                        queue_capacity=self._opts.op_queue_capacity,
                        stage_index=idx,
                        tracking_mode=runner_tracking_mode,
                        stage_output_mode=output_mode,
                    )
                )
            elif chosen == "inline":
                self._runners.append(
                    InlineStageRunner(
                        stage,
                        self._ctx,
                        cap_for_stage,
                        prefetch_capacity=prefetch,
                        deterministic=self._opts.deterministic,
                        allow_latency_flush_in_deterministic=allow_latency,
                        stage_index=idx,
                        tracking_mode=runner_tracking_mode,
                        stage_output_mode=output_mode,
                    )
                )
            elif chosen == "remote":
                raise NotImplementedError("Remote workers are not yet implemented.")
            elif chosen == "process":
                self._runners.append(
                    ProcessStageRunner(
                        stage,
                        self._ctx,
                        cap_for_stage,
                        prefetch_capacity=prefetch,
                        deterministic=self._opts.deterministic,
                        allow_latency_flush_in_deterministic=allow_latency,
                        queue_capacity=self._opts.op_queue_capacity,
                        stage_index=idx,
                        tracking_mode=runner_tracking_mode,
                        stage_output_mode=output_mode,
                        mp_context=self._mp_context,
                    )
                )
            else:
                raise ValueError(f"Unknown runner '{chosen}'")

    def metrics_snapshot(self):
        """Return a clone of the current pipeline metrics summary when enabled."""
        if self._collector is None:
            return None
        return self._collector.snapshot()

    def _emit_fetch_metrics(self, delta: Any) -> None:
        """Forward fetch timings to the collector when tracking is enabled."""
        if self._collector is not None and self._collector.tracking_mode.collects_nodes:
            self._collector.record_fetch(delta)

    def _emit_prefetch_metrics(self, delta: Any) -> None:
        """Forward prefetch timings to the collector when tracking is enabled."""
        if self._collector is not None and self._collector.tracking_mode.collects_nodes:
            self._collector.record_prefetch(delta)

    def _emit_backpressure_metrics(self, delta: Any) -> None:
        """Forward backpressure metrics to the collector when tracking is enabled."""
        if self._collector is not None and self._collector.tracking_mode.collects_nodes:
            self._collector.record_backpressure(delta)

    def _get_component_id(self, component_name: str) -> int:
        """Get or assign a stable integer ID for a component name.

        Used by EnsureMixture to convert user-provided explicit weights from
        string names to integer IDs for efficient internal tracking.

        Thread-safe: uses a lock to ensure consistent ID assignment across
        concurrent lane streams.
        """
        # Fast path: already assigned
        cid = self._component_to_id.get(component_name)
        if cid is not None:
            return cid

        # Slow path: assign new ID under lock
        with self._component_lock:
            # Double-check after acquiring lock
            cid = self._component_to_id.get(component_name)
            if cid is not None:
                return cid

            cid = len(self._id_to_component)
            self._component_to_id[component_name] = cid
            self._id_to_component.append(component_name)
            return cid

    def _get_component_name(self, component_id: int) -> str | None:
        """Look up component name by ID for human-readable warning messages.

        Returns None if ID is unknown.
        """
        if 0 <= component_id < len(self._id_to_component):
            return self._id_to_component[component_id]
        return None

    def _get_chunk_mixture(
        self, lane_id: LaneId, chunk_id: ChunkId
    ) -> dict[int, float]:
        """Get the target mixture weights for a specific chunk.

        Used by EnsureMixture to determine the target component ratios for
        SWRR-based sample reordering.

        Returns a dict mapping component_id -> normalized weight.

        Raises:
            KeyError: If the chunk mixture is not available.
        """
        key = (lane_id, chunk_id)
        with self._mixture_lock:
            if key not in self._chunk_mixtures:
                raise KeyError(
                    f"Chunk mixture not available for lane={lane_id}, chunk={chunk_id}"
                )
            return self._chunk_mixtures[key]

    def _store_chunk_mixture(
        self, lane_id: LaneId, chunk_id: ChunkId, chunk: WorkChunk
    ) -> None:
        """Store the mixture weights for a chunk, converting names to IDs."""
        mixture = chunk.mixture  # dict[str, float] normalized
        if not mixture:
            return

        id_mixture: dict[int, float] = {}
        for name, weight in mixture.items():
            comp_id = self._get_component_id(name)
            id_mixture[comp_id] = weight

        with self._mixture_lock:
            self._chunk_mixtures[(lane_id, chunk_id)] = id_mixture

    def _lane_stream(self, lane_id: LaneId) -> Iterator[EngineSample]:
        """Yield EngineSamples for a single lane, fetching chunks lazily.

        Maintains inflight_chunks_per_lane[lane_id][chunk_id] -> chunk_obj.

        Each EngineSample is a 5-tuple:
            (sample_id, lane_id, chunk_id, chunk_offset, component_id)

        WorkChunk yields (sample_id, component_name) tuples; we convert
        component_name to component_id for efficient downstream processing.
        """
        inflight_lane = self.inflight_chunks_per_lane[lane_id]
        # Phase 1: replay restored inflight chunks first (ascending chunk_id)
        for cid in sorted(inflight_lane.keys()):
            chunk = inflight_lane[cid]
            # Store mixture for restored chunks (may already exist, but idempotent)
            self._store_chunk_mixture(lane_id, cid, chunk)
            for offset, (sample_id, component_name) in enumerate(chunk):
                component_id = self._get_component_id(component_name)
                # Note that we yield the _entire_ chunk here. This can break with elastic continuation in case a batch is cross-chunk boundaries.
                yield (sample_id, lane_id, int(cid), offset, component_id)

        # Phase 2: fetch new chunks and assign stable per-lane ids
        ws = self._lane_ws[lane_id]
        while True:
            with self._checkpoint_lock:
                # Since internally we prefetch, this could overlap with a checkpointing call.
                # We need to ensure that we are not prefetching while updating the lane state.
                chunk = ws.next_chunk()
                if chunk is None:
                    return  # lane exhausted

                cid = int(self._lane_next_cid[lane_id])
                self._lane_next_cid[lane_id] = cid + 1
                inflight_lane[cid] = chunk
                # Store mixture weights for this chunk
                self._store_chunk_mixture(lane_id, cid, chunk)

            for offset, (sample_id, component_name) in enumerate(chunk):
                component_id = self._get_component_id(component_name)
                yield (sample_id, lane_id, cid, offset, component_id)

    def _active_workers(self, num_workers: int, lanes_all: list[int]) -> int:
        L = len(lanes_all)
        a = min(num_workers, L)
        while a > 1 and (L % a) != 0:
            a -= 1
        return a  # at least 1

    def _refresh_rr_from_progress(self) -> None:
        """Recompute the tail round-robin (RR) pointer for THIS owner (rank + DataLoader worker) from durable per-lane progress.

        Scope: physical vs logical
        - Physical-scoped (ephemeral): the RR pointer is keyed by the current topology and
        the exact owned lane set:
            "{physical_rank}:{worker_id}/{active}:{','.join(str(l) for l in lanes)}"
        It means "which lane index should this owner emit from next?". When topology
        changes (ranks/workers/mapping), the key changes, and any stale pointer is ignored.
        - Logical-scoped (durable): per-lane state keyed by canonical lane_id:
            * LanePtr(chunk_id, offset)
            * inflight chunks
            * last replay cursor
            * next chunk id
        This state is the source of truth across checkpoints and elastic remaps and
        guarantees no skips/duplicates globally.

        What this method does
        - For the current owner, determine the set of owned lanes under the current topology.
        - If the owner is idle (worker_id >= active), return.
        - If there is ≤ 1 owned lane, set the RR index to 0.
        - Otherwise, choose the "least-advanced" lane among the owned lanes using:
            (progress.chunk_id, progress.offset, lane_id)
        and set the RR pointer to that lane's index within the owned lane list.
        This is deterministic and tends to preserve fairness.

        Guarantees
        - No skips/duplicates globally: ensured by durable per-lane progress and inflight state.
        - Deterministic local emission within the same topology: the RR pointer is persisted
        under the physical key and reused.
        - Multiset equality per global window when checkpointing at window boundaries:
        lane progress + one-lane-per-batch invariant ensure the same set of samples per
        window (order may be permuted).

        Non-goals
        - Exact cross-topology, sample-by-sample interleaving is not preserved. After a
        remap, the RR pointer is recomputed from progress and may differ from the prior
        owner's next turn. If you were to checkpoint mid-window, the composition of the
        remainder of that window could permute (still no skips/dups overall). Therefore,
        checkpoint at window boundaries if you require window-level set semantics.

        Where it's used
        - Called during load_state_dict() (after restoring per-lane progress) to seed the
        RR pointer for the new topology.
        - Called inside state_dict() before local state is written, ensuring RR state is
        consistent with observed progress.
        - If this method has not been called for a fresh run/topology, _lane_rr_iter()
        falls back to RR index 0 for the current key.

        Bottom line
        - RR pointer: local, physical-world-scoped fairness hint; recomputed or reused per key.
        - Per-lane progress: durable, topology-agnostic correctness state; guarantees continuity.
        """
        lanes_all = self._world.lanes_for_dp_group[self._world.dp_group_id]
        worker_id, workers_per_rank = get_torch_worker_info()
        active = self._active_workers(workers_per_rank, lanes_all)
        if worker_id >= active:
            return
        lanes = self._owned_lanes
        key = f"{self._opts.global_rank}:{worker_id}/{active}:{','.join(str(l) for l in lanes)}"
        if len(lanes) <= 1:
            self._rr_next_idx[key] = 0
            return

        def lane_progress_tuple(lane: int) -> tuple[int, int, int]:
            ptr = self._lane_progress.get(lane, LanePtr())
            return (int(ptr.chunk_id), int(ptr.offset), int(lane))

        next_lane = min(lanes, key=lane_progress_tuple)
        self._rr_next_idx[key] = lanes.index(next_lane)

    def _publish_replay_snapshot(self) -> None:
        snapshot: dict[int, SampleCursor | None] = {}
        for lane, cursor in self._lane_last_cursor.items():
            if cursor is None:
                # No prior record delivered on this lane (fresh run or evicted sentinel).
                snapshot[lane] = None
                continue

            inflight_lane = self.inflight_chunks_per_lane[lane]
            if cursor.chunk_id not in inflight_lane:
                # Target record will not be replayed (its chunk already evicted), so emit all.
                snapshot[lane] = None
            else:
                snapshot[lane] = cursor
        self._replay_config.set_snapshot(snapshot)

    def _stage_parallelism(self, stage: Stage) -> int:
        total = 0
        filter_parallelism = 0
        for nd in stage.nodes:
            dop = max(1, (nd.parallelism or 1))
            if isinstance(nd.op, ReplayFilter):
                # TODO: if we ever fuse ReplayFilter into Batch, revisit how we account for its DOP.
                filter_parallelism = max(filter_parallelism, dop)
                continue
            total += dop
        if total == 0:
            total = filter_parallelism
        else:
            total = max(total, filter_parallelism)
        return max(1, total)

    @property
    def _owned_lanes(self) -> list[LaneId]:
        lanes_all = self._world.lanes_for_dp_group[
            self._world.dp_group_id
        ]  # canonical order
        worker_id, workers_per_rank = get_torch_worker_info()
        active = self._active_workers(workers_per_rank, lanes_all)

        if worker_id >= active:
            return []  # this worker is idle / no lanes assigned

        return [
            lane for idx, lane in enumerate(lanes_all) if (idx % active) == worker_id
        ]

    def _source_stream(self) -> Iterator[EngineSample]:
        """Yield sample identifiers from the backing work source."""
        owned = self._owned_lanes

        # Build one generator per lane
        gens: dict[LaneId, Iterator[EngineSample]] = {
            lane: self._lane_stream(lane) for lane in owned
        }

        # Round-robin over the lane generators
        rr = deque(gens.items())  # holds (lane_id, gen)
        while rr:
            lane_id, gen = rr.popleft()
            try:
                yield next(gen)  # emit one sample from this lane
                rr.append((lane_id, gen))  # put it back for fair round-robin
            except StopIteration:
                # This lane is finished; don't re-append
                pass

    def _lane_rr_iter(
        self,
        upstream: Iterable[StreamItem],
    ) -> Iterator[StreamItem]:
        """Tail round-robin multiplexer over the lanes owned by THIS DataLoader worker.

        Purpose
        - Maintain the "one lane per batch" invariant at the tail:
        drain the upstream into per-lane buffers and emit in round-robin order
        across the owned lanes.

        Physical key scoping
        - The RR pointer is keyed by:
            "{physical_rank}:{worker_id}/{active}:{','.join(str(l) for l in lanes)}"
        which binds the pointer to this owner and its current lane assignment.
        When topology changes, the key changes and the stale pointer is ignored.

        Initialization and interaction with RR refresh
        - This iterator reads the saved RR pointer for the current key; if missing,
        it defaults to 0.
        - _refresh_rr_from_progress() should be called after loading a checkpoint (and is
        called by load_state_dict()) to provide a progress-derived starting point.
        For brand-new runs with no checkpoint, defaulting to 0 is fine.

        Round-robin algorithm (high level)
        - If this worker is idle (worker_id >= active), drain upstream (expected empty) and return.
        - Otherwise:
        1) Initialize idx from the saved RR pointer (or 0).
        2) Drain upstream into per-lane deques (routing by item.lane_id).
        3) Emit exactly one item from lane owned_lanes[idx], then advance:
            idx = (idx + 1) % len(owned_lanes), and persist the updated pointer
            under the same physical key.
        4) Continue until upstream ends and all per-lane buffers are drained.

        Guarantees and limits
        - No skips/duplicates: ensured by durable per-lane progress in the engine.
        - One lane per batch at the tail: the pipeline enforces that batches carry a
        single lane id; this mux preserves that invariant on emission.
        - Deterministic local interleaving in the same topology: the saved pointer is reused.
        - Across topology changes, the exact interleaving may differ; if you checkpoint
        at window boundaries, the multiset of samples per window remains identical.

        Best practices
        - Checkpoint at window boundaries to preserve window-level set semantics across
        elastic remaps.
        - Rely on per-lane progress for correctness; the RR pointer is a local fairness hint.
        """
        warn_threshold = 10000

        # Derive the lanes owned by THIS DataLoader worker (same logic as _source_stream)
        lanes_all = self._world.lanes_for_dp_group[self._world.dp_group_id]
        worker_id, workers_per_rank = get_torch_worker_info()
        active = self._active_workers(workers_per_rank, lanes_all)

        # Idle worker: upstream will be empty, but we still drain it to let the
        # ThreadStageRunner stop cleanly (propagate stop token, join threads).
        if worker_id >= active:
            upstr = list(upstream)
            assert len(upstr) == 0
            return

        lanes = self._owned_lanes
        if len(lanes) <= 1:  # Simple case: only one owned lane
            key = f"{self._opts.global_rank}:{worker_id}/{active}:{','.join(str(l) for l in lanes)}"
            self._rr_next_idx[key] = 0
            yield from upstream
            return

        buffers: dict[int, deque[StreamItem]] = {lane: deque() for lane in lanes}
        it = iter(upstream)
        upstream_ended = False
        key = f"{self._opts.global_rank}:{worker_id}/{active}:{','.join(str(l) for l in lanes)}"
        idx = self._rr_next_idx.get(key, 0) % len(lanes)

        # soft warning thresholds (double each time they’re tripped)
        per_lane_next_warn: dict[int, int] = {}
        total_next_warn = warn_threshold
        lane_warn_base = warn_threshold

        def warn_after_enqueue(lane: int) -> None:
            nonlocal total_next_warn
            if lane_warn_base:
                nxt = per_lane_next_warn.get(lane, lane_warn_base)
                sz = len(buffers[lane])
                if sz >= nxt:
                    warnings.warn(
                        f"[zephon] Tail RR buffer for lane {lane} reached {sz} items.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    per_lane_next_warn[lane] = max(nxt * 2, nxt + 1)
            if total_next_warn:
                total_sz = sum(len(q) for q in buffers.values())
                if total_sz >= total_next_warn:
                    warnings.warn(
                        f"[zephon] Total Tail RR buffered items reached {total_sz}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    total_next_warn = max(total_next_warn * 2, total_next_warn + 1)

        while not upstream_ended:
            lane = lanes[idx]
            # Block-fill until current lane has something or upstream ends.
            while not buffers[lane]:
                try:
                    item = next(it)  # blocks until data or StopIteration
                except StopIteration:
                    upstream_ended = True
                    break
                got_lane = _extract_lane_id(item)
                assert got_lane in buffers
                buffers[got_lane].append(item)
                warn_after_enqueue(got_lane)

            if upstream_ended:
                break  # move to drain regime

            # Emit exactly one from the current lane, then advance RR pointer.
            yield buffers[lane].popleft()
            idx = (idx + 1) % len(lanes)
            self._rr_next_idx[key] = idx

        # -------- DRAIN REGIME: upstream ended, flush everything in RR ----------
        while any(buffers[l] for l in lanes):
            # Find next non-empty lane starting at idx (skip empties).
            rotated = 0
            emitted = False
            while rotated < len(lanes):
                lane = lanes[idx]
                if buffers[lane]:
                    yield buffers[lane].popleft()
                    idx = (idx + 1) % len(lanes)
                    self._rr_next_idx[key] = idx
                    emitted = True
                    break
                idx = (idx + 1) % len(lanes)
                rotated += 1
            if not emitted:
                # All empty (defensive; the outer while should break next iteration).
                break
        self._rr_next_idx[key] = idx

    def build_iter(self) -> Iterator[StreamItem]:
        """Return an iterator that threads the work stream through all stages."""
        if self._metrics_reporter is not None and not self._metrics_started:
            self._metrics_reporter.start()
            self._metrics_started = True
        source_iter = self._source_stream()

        # Construct overall pipeline by chaining runners
        stage_stream: Iterable[RunnerStageOut] = self._runners[0].run(source_iter)
        for runner in self._runners[1:]:
            stage_stream = runner.run(stage_stream)

        final_stream = cast(Iterable[StreamItem], stage_stream)
        final_stream = self._lane_rr_iter(final_stream)

        yield from final_stream

    def notify_monotone(
        self, lane_id: int, max_chunk_id: int, cursors: Iterable[SampleCursor]
    ) -> None:
        """Cursor-ordered notify path (no cross-chunk reordering or packing).

        Mirrors the pre-contributors behavior: evict chunks older than
        ``max_chunk_id``, advance lane progress by the number of cursors provided
        from that chunk, and track the max cursor for replay.
        """
        inflight_lane = self.inflight_chunks_per_lane[lane_id]
        cursor_list = cursors if isinstance(cursors, list) else list(cursors)

        # 1) Evict older inflight chunks (no bitmap bookkeeping in this path).
        cids_to_evict = [cid for cid in list(inflight_lane) if cid < max_chunk_id]
        for cid in cids_to_evict:
            inflight_lane.pop(cid, None)
        if cids_to_evict:
            with self._mixture_lock:
                for cid in cids_to_evict:
                    self._chunk_mixtures.pop((lane_id, cid), None)

        add_k = len(cursor_list)
        cur = self._lane_progress.get(lane_id, LanePtr())
        seen_offset = cur.offset if (cur.chunk_id == max_chunk_id) else 0
        self._lane_progress[lane_id] = LanePtr(max_chunk_id, seen_offset + add_k)

        if cursor_list:
            max_cursor = max(cursor_list)
            previous = self._lane_last_cursor.get(lane_id)
            if previous is None or max_cursor > previous:
                self._lane_last_cursor[lane_id] = max_cursor

    def notify(
        self,
        lane_id: int,
        entries: Iterable[ContributorRef],
        record_cursor: SampleCursor | None = None,
    ) -> None:
        """Record delivery progress for ``lane_id`` and evict completed chunks.

        ``entries`` describe contributors that have been emitted. A chunk can be
        evicted once every base offset in that chunk has produced exactly one
        contributor (or tombstone) with ``is_last_child=True``. ``record_cursor``
        is the replay identity of the delivered training record and is stored as
        the per-lane sentinel for equality-based replay.
        """
        inflight_lane = self.inflight_chunks_per_lane[lane_id]
        # Per-chunk bitmaps track which offsets are closed; counts are popcounts
        # so we can check "chunk complete?" in O(1) instead of re-counting bits.
        done = self._offset_done[lane_id]
        done_count = self._offset_done_count[lane_id]

        # 1) Update per-offset completion state
        for entry in entries:
            cursor = entry.cursor
            cid = cursor.chunk_id
            off = cursor.chunk_offset

            chunk = inflight_lane.get(cid)
            if chunk is None:
                continue  # chunk already evicted or not tracked

            if cid not in done:
                done[cid] = OffsetBitmap(len(chunk))
                done_count[cid] = 0

            if entry.is_last_child and not done[cid].is_set(off):
                done[cid].set(off)
                done_count[cid] += 1

        # 2) Evict fully-completed chunks in cid order
        last_completed_ptr: LanePtr | None = None
        cids_to_evict: list[int] = []
        # Chunk IDs increase monotonically per lane; dict preserves insertion order.
        for cid in list(inflight_lane.keys()):
            chunk = inflight_lane[cid]
            if cid not in done:
                break
            if done_count[cid] >= len(chunk):
                last_completed_ptr = LanePtr(cid, len(chunk))
                cids_to_evict.append(cid)
            else:
                break  # earliest incomplete chunk blocks later evictions (keep inflight contiguous)

        for cid in cids_to_evict:
            inflight_lane.pop(cid, None)
            done.pop(cid, None)
            done_count.pop(cid, None)
        if cids_to_evict:
            with self._mixture_lock:
                for cid in cids_to_evict:
                    self._chunk_mixtures.pop((lane_id, cid), None)

        # 3) Maintain lane progress for fairness diagnostics
        if inflight_lane:
            front_cid = min(inflight_lane.keys())
            # A chunk may be inflight without any completions yet; default completed to 0.
            self._lane_progress[lane_id] = LanePtr(
                front_cid, done_count.get(front_cid, 0)
            )
        elif last_completed_ptr is not None:
            self._lane_progress[lane_id] = last_completed_ptr
        else:
            self._lane_progress.setdefault(lane_id, LanePtr())

        # 4) Track latest record-level cursor for replay
        if record_cursor is not None:
            self._lane_last_cursor[lane_id] = record_cursor

    def eval_one(self, sample: SampleId | EngineSample) -> Any:
        """Synchronously evaluate a single element through every stage runner."""
        value: Any = sample
        for runner in self._runners:
            value = runner.run_one(value)
        return value

    def close(self) -> None:
        """Close all stage runners, suppressing teardown errors."""
        for runner in self._runners:
            try:
                runner.close()
            except Exception:
                print(
                    f"Error closing runner {runner!r}.",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
        if self._metrics_reporter is not None and self._metrics_started:
            try:
                self._metrics_reporter.stop()
            except Exception:
                print(
                    "Error stopping metrics reporter.",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
        self._metrics_started = False

    def _state_dict_local(self) -> dict[str, Any]:
        """Serializable snapshot of engine runtime state (no plan/op state)."""
        with self._checkpoint_lock:
            owned = self._owned_lanes
            # print(f"node {self._world.global_rank}/{self._world.world_size} w{worker_id}/{workers_per_rank} owns {len(owned)} lanes.")

            self._refresh_rr_from_progress()

            for purge_candidate_str in [
                "_lane_ws",
                "inflight_chunks_per_lane",
                "_lane_progress",
                "_lane_next_cid",
                "_lane_last_cursor",
                "_offset_done",
                "_offset_done_count",
            ]:
                # print(f"length of {purge_candidate_str} is {len(getattr(self, purge_candidate_str))}")
                for lane in list(getattr(self, purge_candidate_str)):
                    if lane not in owned:
                        del getattr(self, purge_candidate_str)[lane]

            for lane in owned:
                self._lane_ws.setdefault(
                    lane,
                    self._work.clone_for_lane(
                        lane, canonical_replicas=self._world.canonical_replicas
                    ),
                )
                if lane not in self._lane_progress:
                    self._lane_progress[lane] = LanePtr(0, 0)
                if lane not in self._lane_next_cid:
                    self._lane_next_cid[lane] = 0
                self._lane_last_cursor.setdefault(lane, None)

            inflight: dict[int, dict[int, dict[str, Any]]] = {}
            for lane, by_chunk in self.inflight_chunks_per_lane.items():
                inflight[int(lane)] = {
                    int(cid): ch.state_dict() for cid, ch in by_chunk.items()
                }

            progress = {
                int(l): {"chunk_id": int(p.chunk_id), "offset": int(p.offset)}
                for l, p in self._lane_progress.items()
            }

            world = {
                "canonical_replicas": int(self._world.canonical_replicas),
                "world_size": int(self._world.world_size),
                "global_rank": int(self._world.global_rank),
                "dp_degree": int(self._world.dp_degree),
                "dp_group_id": int(self._world.dp_group_id),
                "mapping": {
                    int(dp_id): [int(x) for x in lanes]
                    for dp_id, lanes in self._world.lanes_for_dp_group.items()
                },
            }
            for lane, by_chunk in self.inflight_chunks_per_lane.items():
                if by_chunk:
                    mx = max(by_chunk)
                    assert self._lane_next_cid[lane] == mx + 1
                    if self._lane_next_cid[lane] < mx + 1:
                        # Either bump silently or assert in debug
                        self._lane_next_cid[lane] = mx + 1  # or: assert False

            lane_next = {int(l): int(n) for l, n in self._lane_next_cid.items()}
            lane_ws_state = {
                int(l): self._lane_ws[l].state_dict() for l in self._lane_ws
            }
            rr_next_idx = dict(self._rr_next_idx)

            replay_cursors: dict[int, Any] = {}
            for lane, cursor in self._lane_last_cursor.items():
                replay_cursors[int(lane)] = (
                    cursor.as_key() if cursor is not None else None
                )

            return {
                "version": 1,
                "world": world,
                "inflight": inflight,
                "progress": progress,
                "lane_next_cid": lane_next,
                "work_source": self._work.state_dict(),
                "lane_ws_state": lane_ws_state,
                "last_round_id": self._last_round_id,
                "checkpoint_reload_count": self._checkpoint_reload_count,
                "rr_next_idx": rr_next_idx,
                "replay_cursors": replay_cursors,
            }

    # ---------- FS utilities ----------
    def _read_text(self, path: str) -> str | None:
        try:
            with self._agg_backend.open(path, "r") as f:
                return str(f.read())
        except Exception:
            return None

    def _read_json(self, path: str) -> dict | None:
        try:
            with self._agg_backend.open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _wait_until(self, pred: Callable[[], bool], timeout: float) -> bool:
        deadline = time.time() + timeout
        delay = self._agg_poll_base
        while time.time() < deadline:
            if pred():
                return True
            time.sleep(min(delay, max(0, deadline - time.time())))
            # Exponential backoff for cloud, capped at 5s
            if self._agg_is_cloud:
                delay = min(delay * 1.5, 5.0)
        return False

    # ---------- Round / paths ----------

    def _publish_new_round_id(self) -> str:
        rid = str(int(time.time() * 1e9))  # monotonic-ish
        self._agg_backend.put(self._round_file, rid.encode("utf-8"))
        self._last_round_id = rid
        return rid

    def _read_open_round_id(self) -> str | None:
        rid = (self._read_text(self._round_file) or "").strip()
        if not rid:
            return None

        # This is a practically relevant guard in case users do neither supply a run id nor a fresh aggregate directory. With this check we at least avoid stale state if it is obviously old.
        try:
            stat = self._agg_backend.stat(self._round_file)
            age_s = time.time() - stat["mtime"]
        except OSError:
            return None  # File doesn't exist or access denied
        if age_s > 120:
            return None

        # Guard: if we already consumed this rid, skip if our own shard for
        # this rid exists. Under fork, a child can inherit _last_round_id==rid
        # from the parent before it has published; in that case we must proceed.
        if self._last_round_id is not None and rid == self._last_round_id:
            try:
                my_fp = self._state_file_path(rid)
            except Exception:
                my_fp = None
            if not my_fp or self._agg_backend.exists(my_fp):
                return None

        # only "open" if merged for this rid doesn't exist yet
        if not self._agg_backend.exists(self._merged_file_path(rid)):
            return rid
        return None

    def _wait_value(
        self,
        supplier: Callable[[], T | None],
        timeout: float,
    ) -> T | None:
        deadline = time.time() + timeout
        delay = self._agg_poll_base
        while time.time() < deadline:
            val = supplier()
            if val is not None:
                return val
            time.sleep(min(delay, max(0, deadline - time.time())))
            if self._agg_is_cloud:
                delay = min(delay * 1.5, 5.0)
        return None

    def _open_round_id(self, is_leader: bool) -> str:
        if is_leader:
            return self._publish_new_round_id()

        rid = self._wait_value(self._read_open_round_id, self._agg_timeout_s)
        if rid is None:
            raise RuntimeError(f"Timeout waiting for round id at {self._round_file}")
        return rid

    def _state_file_path(self, round_id: str) -> str:
        wid, _ = get_torch_worker_info()
        pid = os.getpid()
        return f"{self._agg_dir}/state_r{self._world.global_rank}_w{wid}_p{pid}_{round_id}.json"

    def _merged_file_path(self, round_id: str) -> str:
        return f"{self._agg_dir}/merged_{round_id}.json"

    def _list_state_files(self, round_id: str) -> list[str]:
        pattern = f"{self._agg_dir}/state_r*_w*_p*_{round_id}.json"
        return self._agg_backend.glob(pattern)

    def _read_states_for_round(
        self, round_id: str, printt: bool = False
    ) -> tuple[list[dict], set[int]]:
        states: list[dict] = []
        covered: set[int] = set()
        for fp in self._list_state_files(round_id):
            st = self._read_json(fp)
            fname = fp.rsplit("/", 1)[-1]  # Get filename from path
            if st is None:
                if printt:
                    self._log(f"{fname} covers nothing!")
                continue
            states.append(st)
            for k in st.get("progress", {}).keys():
                if printt:
                    self._log(f"{fname} covers lane {k}!")
                covered.add(int(k))
        return states, covered

    def _log(self, msg: str) -> None:
        worker_id, workers_per_rank = get_torch_worker_info()
        print(
            f"[PR {self._world.global_rank}][PID {os.getpid()}][Worker {worker_id}/{workers_per_rank - 1}] {msg}",
            file=sys.stderr,
        )

    def state_dict(self) -> dict[str, Any]:
        # Fast path: single node & single active worker → just return local
        local = self._state_dict_local()
        if len(self._owned_lanes) == 0:
            # Idle workers just return their local state
            # In the best case, they would also read the aggregate checkpoint to have a complete return
            # The problem is that we cannot ensure this:
            # Leader (rank 0, worker 0) publishes round_id
            # Worker 0 & 1 successfully read the round_id, write their state files
            # Leader sees all lanes covered, merges states, writes merged file, deletes round file
            # Worker 2 (slow to call state_dict()) tries to read round_id, but:
            # The round file might be deleted already, OR
            # The round file exists but the merged file ALSO exists, causing _read_open_round_id() to return None due to this check:
            #
            # The thing is that for torchdata in the end we only return worker 0 results anyways due to our torchdata_compat.py
            # so it doesn't really matter. For regular torch dataloader, state_dict() does not work anyways by design (it does not forward to workers)
            # And in regular Zephon without a DL, we don't have multiple workers per rank. Hence, this is fine, but if there is a better solution we should improve this.
            return local

        lanes_all = self._world.lanes_for_dp_group[self._world.dp_group_id]
        worker_id, workers_per_rank = get_torch_worker_info()
        active_here = self._active_workers(workers_per_rank, lanes_all)
        if self._world.world_size == 1 and active_here == 1:
            return local

        # Round setup
        is_leader = self._world.global_rank == 0 and worker_id == 0
        round_id = self._open_round_id(is_leader)
        self._last_round_id = round_id
        my_path = self._state_file_path(round_id)
        self._agg_backend.put(my_path, json.dumps(local).encode("utf-8"))

        merged_path = self._merged_file_path(round_id)
        if is_leader:
            # Wait for complete lane coverage
            expected_lanes = set(range(int(self._world.canonical_replicas)))

            def _have_full_coverage() -> bool:
                states, covered = self._read_states_for_round(round_id)
                return bool(states) and expected_lanes.issubset(covered)

            if not self._wait_until(_have_full_coverage, self._agg_timeout_s):
                states, covered = self._read_states_for_round(round_id)
                missing = sorted(expected_lanes - covered)
                raise RuntimeError(
                    f"[PID {os.getpid()}] Aggregation timeout after {self._agg_timeout_s}s. "
                    + f"Missing lanes={missing}; files={len(self._list_state_files(round_id))}"
                )

            # Merge and publish
            states, _ = self._read_states_for_round(round_id, printt=False)
            merged = self._merge_state_dicts(states)
            assert not self._agg_backend.exists(merged_path)
            self._agg_backend.put(merged_path, json.dumps(merged).encode("utf-8"))
            self._agg_backend.delete(my_path)
            if self._previous_merged_file is not None:
                self._agg_backend.delete(self._previous_merged_file)
            self._previous_merged_file = merged_path
            self._agg_backend.delete(
                self._round_file
            )  # can also clean this up since we know everybody consumed it.
            return merged

        # Followers: wait for merged file
        ok = self._wait_until(
            lambda: self._agg_backend.exists(merged_path), self._agg_timeout_s
        )
        if not ok:
            raise RuntimeError(
                f"[PID {os.getpid()}] Timed out after {self._agg_timeout_s}s waiting for merged checkpoint at {merged_path}"
            )
        merged = self._read_json(merged_path)
        self._agg_backend.delete(my_path)
        if merged is None:
            raise RuntimeError(f"Failed to read merged checkpoint {merged_path}")
        return merged

    def _merge_state_dicts(self, states: list[dict[str, Any]]) -> dict[str, Any]:
        assert states
        C = int(states[0]["world"]["canonical_replicas"])
        for s in states[1:]:
            if int(s["world"]["canonical_replicas"]) != C:
                raise RuntimeError("canonical_replicas mismatch")

        merged_world_size: int | None = None
        for s in states:
            w = s.get("world", {})
            if "world_size" in w and w["world_size"] is not None:
                n = int(w["world_size"])
                if merged_world_size is None:
                    merged_world_size = n
                elif merged_world_size != n:
                    raise RuntimeError("world_size mismatch across state shards")

        inflight, progress, lane_next, lane_ws_state = {}, {}, {}, {}
        for st in states:
            for lane_s, by_chunk in st.get("inflight", {}).items():
                lane = int(lane_s)
                if lane in inflight:
                    raise RuntimeError(f"duplicate inflight for lane {lane}")
                inflight[lane] = {
                    int(cid): payload for cid, payload in by_chunk.items()
                }

            for lane_s, p in st.get("progress", {}).items():
                lane = int(lane_s)
                if lane in progress:
                    raise RuntimeError(f"duplicate progress for lane {lane}")
                progress[lane] = {
                    "chunk_id": int(p["chunk_id"]),
                    "offset": int(p["offset"]),
                }

            for lane_s, nxt in st.get("lane_next_cid", {}).items():
                lane = int(lane_s)
                if lane in lane_next:
                    raise RuntimeError(f"duplicate lane_next_cid for lane {lane}")
                lane_next[lane] = int(nxt)

            for lane_s, ws in st.get("lane_ws_state", {}).items():
                lane = int(lane_s)
                if lane in lane_ws_state:
                    raise RuntimeError(f"duplicate lane_ws_state for lane {lane}")
                lane_ws_state[lane] = ws

        work_config = next(
            (s.get("work_config") for s in states if s.get("work_config")), None
        )
        last_round_ids = {s.get("last_round_id") for s in states}
        assert len(last_round_ids) == 1

        checkpoint_reload_counts = {s.get("checkpoint_reload_count") for s in states}
        assert len(checkpoint_reload_counts) == 1

        rr_next_idx: dict[str, int] = {}
        for st in states:
            for key, idx in st.get("rr_next_idx", {}).items():
                if key in rr_next_idx and rr_next_idx[key] != int(idx):
                    raise RuntimeError(f"duplicate rr_next_idx for key {key}")
                rr_next_idx[key] = int(idx)

        replay_cursors: dict[int, Any] = {}
        for st in states:
            for lane_s, key in st.get("replay_cursors", {}).items():
                lane = int(lane_s)
                if lane in replay_cursors:
                    raise RuntimeError(
                        f"duplicate replay_cursors entry for lane {lane}"
                    )
                replay_cursors[lane] = key

        return {
            "version": 1,
            "world": {"canonical_replicas": C, "world_size": merged_world_size},
            "inflight": inflight,
            "progress": progress,
            "lane_next_cid": lane_next,
            "work_config": work_config,
            "lane_ws_state": lane_ws_state,
            "last_round_id": list(last_round_ids)[0],
            "checkpoint_reload_count": list(checkpoint_reload_counts)[0],
            "rr_next_idx": rr_next_idx,
            "replay_cursors": replay_cursors,
        }

    def load_state_dict(self, state: dict[str, Any], *, replay: bool = True) -> None:
        """Restore engine & WorkSource; enter replay mode if 'replay' is True."""
        if int(state["world"]["canonical_replicas"]) != self._world.canonical_replicas:
            raise RuntimeError("canonical_replicas changed; migration required")

        bs = self._plan.batch_size_hint
        if bs is not None:
            cs = self._work.chunk_size_hint()
            assert cs is not None, (
                "No chunk size hint for current work source, determinism breaks potentially"
            )

        # This is FOR ALL WORKERS on that node. So we restore a bit more than we have to because we cannot be certain whether load_state_dict is called before workers are instantiated or not.

        owned = set(self._world.lanes_for_dp_group[self._world.dp_group_id])
        base_ws = self._work
        # TODO(MaxiBoether): in the future support sometihng like this
        # if state.get("work_config") and hasattr(type(self._work), "from_config"):
        #    try:
        # ensure identity parity if user passed a mismatched base WorkSource
        #        base_ws = type(self._work).from_config(state["work_config"])
        #    except Exception:
        #        pass

        # (Re)build lane WorkSources for currently owned lanes
        self._lane_ws.clear()
        for lane in owned:
            self._lane_ws[lane] = base_ws.clone_for_lane(
                lane, canonical_replicas=self._world.canonical_replicas
            )

        # Restore inflight & pointers
        self.inflight_chunks_per_lane.clear()
        self._offset_done.clear()
        self._offset_done_count.clear()
        inflight_all = state.get("inflight", {})
        for lane_s, by_chunk in inflight_all.items():
            lane = int(lane_s)
            if lane not in owned:
                continue
            self.inflight_chunks_per_lane[lane] = {}
            for cid_s, payload in by_chunk.items():
                cid = int(cid_s)
                self.inflight_chunks_per_lane[lane][cid] = WorkChunk.from_state(payload)

        self._lane_progress.clear()
        progress_all = state.get("progress", {})
        for lane in owned:
            p = progress_all.get(str(lane)) or progress_all.get(int(lane))
            if p is not None:
                self._lane_progress[lane] = LanePtr(
                    int(p["chunk_id"]), int(p["offset"])
                )
            else:
                raise RuntimeError(
                    f"Lane {lane} does not have progress state in the checkpoint!"
                )

        self._lane_next_cid.clear()
        lane_next_all = state.get("lane_next_cid", {})
        for lane in owned:
            saved_next = lane_next_all.get(str(lane)) or lane_next_all.get(int(lane))
            if saved_next is None:
                raise RuntimeError(
                    f"Lane {lane} does not have next cid in the checkpoint"
                )
            self._lane_next_cid[lane] = int(saved_next)

        # Restore lane WS state for owned lanes
        lane_ws_state_all = state.get("lane_ws_state", {})
        for lane, ws in self._lane_ws.items():
            st = lane_ws_state_all.get(str(lane)) or lane_ws_state_all.get(int(lane))
            if st is not None:
                ws.load_state_dict(dict(st))
            else:
                raise RuntimeError(
                    f"Lane {lane} does not have WorkSource state in the checkpoint"
                )

        self._last_round_id = state["last_round_id"]
        self._checkpoint_reload_count = state["checkpoint_reload_count"] + 1
        self._agg_backend.mkdir(self._agg_dir, parents=True, exist_ok=True)

        rr_next_idx_raw = state.get("rr_next_idx", {}) or {}
        self._rr_next_idx = {str(k): int(v) for k, v in rr_next_idx_raw.items()}

        replay_raw = state.get("replay_cursors", {}) or {}
        self._lane_last_cursor = dict.fromkeys(owned)
        if replay:
            for lane in owned:
                payload = replay_raw.get(str(lane)) or replay_raw.get(int(lane))
                if payload is None:
                    self._lane_last_cursor[lane] = None
                elif isinstance(payload, SampleCursor):
                    self._lane_last_cursor[lane] = payload
                else:
                    self._lane_last_cursor[lane] = SampleCursor.from_key(payload)
        else:
            for lane in owned:
                self._lane_last_cursor[lane] = None

        self._refresh_rr_from_progress()
        self._publish_replay_snapshot()
