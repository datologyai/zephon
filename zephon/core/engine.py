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

import ctypes
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

from zephon.core.checkpoint import (
    ENGINE_VERSION,
    AggregationCodec,
    EngineStateV1,
)
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
    SampleMeta,
    SampleRecord,
    StreamItem,
)
from zephon.core.graph import Plan
from zephon.core.notify import is_sentinel
from zephon.core.replay import ReplayConfigService
from zephon.core.runtime_spec import RuntimeSpec
from zephon.core.world import World
from zephon.io.options import StoreOptions
from zephon.io.storage import RouterStorageBackend
from zephon.io.stores.multi import finalize_dataset_catalogs, has_cacheable_dataset
from zephon.observability import ExecutionTrackingMode, MetricsSinkConfig
from zephon.observability.collector import CollectorConfig, PipelineCollector
from zephon.observability.emitter import MetricsReporter
from zephon.runners.inline import InlineStageRunner
from zephon.runners.process import ProcessStageRunner
from zephon.runners.threads import ThreadStageRunner
from zephon.utils.disk import check_cache_disk_space
from zephon.utils.ipc import (
    DEFAULT_IPC_BUFFER_BYTES,
    DEFAULT_IPC_TRANSPORT,
    DEFAULT_MTP_BUFFER_BYTES,
    IpcTransport,
)
from zephon.utils.rank import rank_ctx
from zephon.utils.shm_coalesce import DEFAULT_SHM_MIN_SIZE
from zephon.work import MixtureReadConfig, WorkSource
from zephon.work.base import WorkChunk

T = TypeVar("T")

# Set ZEPHON_DEBUG_EVICT=1 to log per-eviction diagnostics from the Engine
# hot path. Useful for debugging chunk-eviction / lane-progress issues; off
# by default because it fires on every eviction.
_DEBUG_EVICT = bool(os.environ.get("ZEPHON_DEBUG_EVICT"))


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

# Per-rank state-dict fields that must be dict-equal across same-DP-group,
# same-owned-lane-set peers at checkpoint time — the caller is expected to
# barrier all ranks (e.g. torch.distributed.barrier()) before
# pipe.checkpoint(). Production-side fields (inflight, lane_next_cid,
# lane_ws_state, epoch_boundaries) may legitimately differ across peers due
# to async prefetch and are taken wholesale from one representative.
_DELIVERY_SYNCED_FIELDS: tuple[str, ...] = ("progress", "replay_cursors")


@dataclass
class RuntimeOptions:
    """User-tunable knobs that influence how the engine constructs runners.

    Numeric tuning knobs default to ``None``, resulting in sensible
    defaults via  ``resolve_*`` helpers that form a dependency cascade.
    Each resolver calls the ones above it when its own field is unset::

        max_workers              ← os.cpu_count(), clamped [4, 16]
            └── prefetch_batches             = max(8, 2 × max_workers)
                    ├── op_queue_capacity        = max(8, prefetch_batches)
                    ├── mtp_buffer               = max(4, prefetch_batches // 2)
                    │       └── mtp_prefetch         = max(4, mtp_buffer // 10)
                    └── default_stage_prefetch   = max(4, prefetch_batches // 4)  [process]
                                                   max(8, prefetch_batches // 2)  [threads/inline]

    For example, ``max_workers=24`` naturally bumps the
    prefetch / queue / MTP buffers without the user having to set them too,
    while ``prefetch_batches=64`` overrides the tail buffer alone and lets
    the rest auto-derive from it.

    By default, ``max_workers`` is derived from the host's CPU count,
    but is clamped to a range of ``[4, 16]``. For any large deployment
    this will hit the upper limit, but the clamp allows Zephon to run
    reasonably on small machines as well.
    """

    runner: str | None = None  # Default runner. Typically auto-inferred.
    run_id: str = DEFAULT_RUN_ID
    per_stage_runner: dict[int, str] = field(
        default_factory=dict
    )  # Manual override for runner per-stage. Mostly useful for debugging and advanced usage.
    mp_context: Any = mp.get_context("spawn")
    # "autotune" is reserved for a future probe-based tuner; passing it today
    # raises NotImplementedError in resolve_runtime_spec.
    worker_allocation: Literal[
        "fit_to_ops", "per_stage_fixed", "global", "autotune"
    ] = "fit_to_ops"
    # max_workers per stage OR global, depending on worker_allocation.
    # None = auto-derived from os.cpu_count(), clamped to [4, 16].
    max_workers: int | None = None
    deterministic: bool = True
    # Consumer-side prefetch depth at the pipeline tail.
    # None = auto-derived from resolved max_workers (max(8, 2 * max_workers)).
    prefetch_batches: int | None = None
    # Inter-stage prefetch buffer depth.
    # None = auto-derived (4 for thread/inline runners, 2 for process runners).
    default_stage_prefetch: int | None = None
    per_stage_prefetch: dict[int, int] = field(default_factory=dict)
    # Maximum in-flight items in the queue between ops within a stage.
    # None = auto-derived from resolved prefetch_batches (max(8, prefetch_batches)).
    op_queue_capacity: int | None = None
    # Transport under process-runner and MTP IPC queues. "socketpair"
    # (default) is exempt from the shared per-UID pipe budget and honors
    # the *_buffer_bytes requests below; "pipe" is stock mp.Queue.
    ipc_transport: IpcTransport = DEFAULT_IPC_TRANSPORT
    # Kernel buffer request per process-runner IPC queue (socketpair only).
    # Best-effort: Linux clamps to 2 * net.core.wmem_max (416 KiB stock),
    # macOS to kern.ipc.maxsockbuf.
    ipc_buffer_bytes: int = DEFAULT_IPC_BUFFER_BYTES
    mixture_config: MixtureReadConfig | None = None
    io_options: StoreOptions = field(default_factory=StoreOptions)
    # Expert knob:
    # Keep latency-based flush in deterministic mode when True unless a stage contains
    # a batch-shape sensitive operator (in which case we auto-disable it for that stage).
    # When False, latency flush is always disabled in deterministic mode.
    allow_latency_flush_in_deterministic: bool = True

    # === Epoch flush ===
    # Flush sentinel cadence: inject a flush sentinel every K chunks per lane.
    # Forces history-dependent accumulators (preserves_cursor_order=False) to flush,
    # creating clean epoch boundaries for deterministic replay after eviction.
    # None = auto (8 for non-monotonic pipelines, 0 for monotone).
    # Explicitly setting 0 for non-monotonic pipelines is an error.
    flush_every_k_chunks: int | None = None

    # === Shutdown ===
    # "graceful" (default) waits generously for threads/processes to finish.
    # "hard" slashes all join timeouts for fast exit (useful for benchmarks).
    shutdown_mode: Literal["graceful", "hard"] = "graceful"

    # === Auto-validation ===
    # Controls how Pipeline.__iter__ handles the auto-validation harness.
    # "strict" (default) raises ValidationError on any error-severity issue.
    # "warn" runs validation and surfaces the full report via warnings.warn
    # but does not raise — escape hatch for cases where the validator's
    # generic checks produce a false positive against a user op.
    # "off" skips the validator entirely; reserved for last-resort overrides.
    auto_validation: Literal["strict", "warn", "off"] = "strict"

    # === MTP Mode (GIL isolation) ===
    # When True, the Engine runs in a non-daemon subprocess for GIL isolation.
    # The main process only dequeues finished batches via IPC.
    mtp_mode: bool = False
    # Bounded IPC queue depth for MTP mode.
    # None = auto-derived from resolved prefetch_batches (max(4, prefetch_batches // 2)).
    mtp_buffer: int | None = None
    # Same as ipc_buffer_bytes but for the MTP data queue; larger so the
    # subprocess feeder can serialize ahead of consumer demand instead of
    # stalling on the subprocess GIL at every next().  Stock Linux clamps
    # this to 2 * net.core.wmem_max (~416 KiB); raise that sysctl for the
    # full run-ahead.
    mtp_buffer_bytes: int = DEFAULT_MTP_BUFFER_BYTES
    # Size (items) of the main-process buffer that a low-priority thread
    # prefetches from the MTP data queue, hiding IPC recv + unpickle from
    # next().  None = auto (max(4, mtp_buffer // 10)); 0 disables the thread.
    mtp_prefetch: int | None = None
    # Automatically capture a checkpoint from the MTP subprocess after normal
    # iteration completion.  Set to False when multi-rank aggregation is not
    # available (e.g. no aggregate_dir, or ranks run sequentially rather than
    # in parallel).
    mtp_auto_checkpoint: bool = True

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

    # === IPC serialization ===
    # Coalesce all tensors in a microbatch by dtype into a single SHM
    # buffer before serialization. Reduces POSIX SHM segments from N to K
    # (K = number of distinct dtypes). Only consumed by process runners.
    coalesce_tensors: bool = True
    shm_min_size: int = DEFAULT_SHM_MIN_SIZE

    # ProcessStageRunner only.  Re-dispatches per crashed seq before
    # giving up; 0 disables.  In non-deterministic mode an exhausted
    # seq is dropped instead of failing the pipeline.
    max_worker_retries: int = 3

    # Where all workers/ranks dump their local state. For multi-node, must be a shared filesystem
    # (e.g., NFS) or cloud storage (s3://bucket/path or gs://bucket/path).
    aggregate_dir: str | None = None
    # How long to wait for all contributors and for the merged file.
    aggregate_timeout_s: float = 180.0
    # Serialization format for intermediate aggregation files ("json" or "msgpack").
    aggregate_serializer: str = "msgpack"
    # Compression for intermediate aggregation files ("none" or "zstd").
    aggregate_compressor: str = "zstd"
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

    def __init__(
        self,
        plan: Plan,
        opts: RuntimeOptions,
        work: WorkSource,
        runtime_spec: RuntimeSpec,
    ) -> None:
        """Initialize stage runners and prepare to stream work items."""
        self._runtime_spec = runtime_spec
        self._plan = plan
        self._preserves_cursor_order = bool(plan.preserves_cursor_order)
        base_ctx: dict[str, Any] = {
            "datasets_by_id": work.datasets_by_id,
            "io_options": opts.io_options,
        }
        cache_opts = opts.io_options.cache
        if cache_opts.enabled and has_cacheable_dataset(work.datasets_by_id):
            check_cache_disk_space(
                Path(cache_opts.root).expanduser(), cache_opts.limit_bytes
            )
        # Mixture query service for EnsureMixture operator
        base_ctx["get_chunk_mixture"] = self._get_chunk_mixture
        base_ctx["get_component_name"] = self._get_component_name
        base_ctx["get_component_id"] = self._get_component_id
        self._replay_config = ReplayConfigService()
        self._ctx = base_ctx
        self._ctx["replay_state_service"] = self._replay_config
        # Build each dataset's node-local catalog before runners spawn (the
        # source-key lock in finalize() elects one builder per node; the rest
        # mmap the file), baking the handle fingerprint that ships in ctx so
        # workers attach the shared mapping instead of unpickling a per-shard
        # metadata copy.
        if datasets_by_id := self._ctx["datasets_by_id"]:
            finalize_dataset_catalogs(datasets_by_id, opts.io_options)
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
        self._epoch_boundaries: dict[LaneId, list[int]] = defaultdict(list)
        # Per-lane consumer-delivered item counts at the tail; observability only.
        self._lane_emitted: dict[LaneId, int] = defaultdict(int)
        # False = unknown baseline (pre-counter restore): neither serialized nor warned on.
        self._lane_emitted_valid: bool = True
        # Resolve flush_every_k_chunks: auto-detect for non-monotonic pipelines.
        _flush_k = opts.flush_every_k_chunks
        if _flush_k is None:
            self._flush_every_k_chunks: int = 0 if self._preserves_cursor_order else 8
        elif not isinstance(_flush_k, int) or _flush_k < 0:
            raise ValueError(
                f"flush_every_k_chunks must be a non-negative integer, got {_flush_k!r}"
            )
        elif _flush_k == 0 and not self._preserves_cursor_order:
            raise ValueError(
                "flush_every_k_chunks=0 is not allowed for pipelines with "
                "preserves_cursor_order=False operators (e.g. PackSequences, "
                "ShuffleBuffer). These operators create cross-chunk accumulator "
                "coupling that requires flush sentinels for safe eviction. "
                "Remove the explicit flush_every_k_chunks=0 to use the default, "
                "or set a positive value (typical: 4-16)."
            )
        else:
            if _flush_k > 0 and self._preserves_cursor_order:
                warnings.warn(
                    f"flush_every_k_chunks={_flush_k} ignored for monotonic "
                    "pipeline (all operators have preserves_cursor_order=True). "
                    "Flush sentinels are only needed for non-monotonic operators "
                    "(e.g. PackSequences, ShuffleBuffer, EnsureMixture). "
                    "Setting to 0.",
                    UserWarning,
                    stacklevel=2,
                )
                _flush_k = 0
            self._flush_every_k_chunks = _flush_k
        self._warned_once_about_runid = False
        self._checkpoint_reload_count = 0
        self._checkpoint_lock = threading.Lock()
        self._inflight_shm: ctypes.Array[ctypes.c_int] | None = None
        self._rr_next_idx: dict[str, int] = {}

        # Component ids come from the source's declared vocabulary so they are
        # a function of config, not encounter order (which differs across
        # resumes and topologies).
        declared: dict[Any, Any] = dict(work.component_ids())
        id_to_component: dict[int, str] = {}
        for name, component_id in declared.items():
            if (
                not isinstance(name, str)
                or isinstance(component_id, bool)
                or not isinstance(component_id, int)
                or component_id < 0
                or component_id in id_to_component
            ):
                raise ValueError(
                    "WorkSource.component_ids() must map string component names to "
                    + f"unique non-negative ints, got {declared!r}"
                )
            id_to_component[component_id] = name
        self._component_to_id: dict[str, int] = declared
        self._id_to_component = id_to_component
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
        self._ctx["emit_pump_metrics"] = _noop
        self._ctx["record_node_metrics"] = _noop
        self._ctx["pump_flush_interval_s"] = 5.0

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
                self._ctx["emit_pump_metrics"] = self._emit_pump_metrics
            self._ctx["record_node_metrics"] = self._collector.record
            self._ctx["pump_flush_interval_s"] = report_interval

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

        # Initialize storage backend and aggregation codec for checkpoint I/O
        self._agg_backend = RouterStorageBackend()
        self._agg_codec = AggregationCodec(
            serializer=self._opts.aggregate_serializer,
            compressor=self._opts.aggregate_compressor,
        )

        # Detect cloud storage and adjust timeouts for higher latency
        base = str(base)  # Handle Path objects
        self._agg_is_cloud = self._agg_backend.is_cloud_path(base)
        if self._agg_is_cloud:
            # Floor: guard against user-provided values that are too low for cloud I/O.
            self._agg_timeout_s = max(self._opts.aggregate_timeout_s, 90.0)
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
        self._closed = False

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

        Delegates to :meth:`RuntimeSpec.explain` for the base output, then
        augments ``process`` stage headers with live runner details
        (``ipc_batch_size``, ``direct_ipc``) that are only available after
        runner construction.
        """
        base = self._runtime_spec.explain(self._plan)

        # Augment process-runner stage headers with IPC details
        process_annotations: dict[int, str] = {}
        for idx, runner in enumerate(self._runners):
            if isinstance(runner, ProcessStageRunner):  # pyright: ignore[reportUnnecessaryIsInstance]
                ipc = getattr(runner, "_ipc_batch_size", None)
                direct = getattr(runner, "_single_op_direct_ipc", False)
                process_annotations[idx] = (
                    f" first_op_ipc_batch={ipc} "
                    f"direct_ipc={'enabled' if direct else 'disabled'}"
                )

        if not process_annotations:
            return base

        lines = base.split("\n")
        for i, line in enumerate(lines):
            for stage_idx, annotation in process_annotations.items():
                prefix = f"Stage[{stage_idx}] "
                if line.startswith(prefix):
                    lines[i] = line + annotation
        return "\n".join(lines)

    def _build_runners(self) -> None:
        """Instantiate per-stage runners from the pre-computed RuntimeSpec.

        All decision logic (runner type, worker cap, queue depth) lives in
        :func:`~zephon.core.runtime_spec.resolve_runtime_spec`.  This method
        only does resource allocation.
        """
        runner_tracking_mode = (
            self._collector.tracking_mode
            if self._collector is not None
            else ExecutionTrackingMode.OFF
        )

        for stage, spec in zip(self._plan.stages, self._runtime_spec.stages):
            common = dict(
                prefetch_capacity=spec.prefetch_capacity,
                deterministic=self._opts.deterministic,
                allow_latency_flush_in_deterministic=spec.allow_latency_flush,
                stage_index=spec.stage_index,
                tracking_mode=runner_tracking_mode,
                stage_output_mode=spec.output_mode,
            )

            if spec.runner_type == "threads":
                self._runners.append(
                    ThreadStageRunner(
                        stage,
                        self._ctx,
                        spec.worker_cap,
                        queue_capacity=spec.queue_capacity,
                        **common,
                    )
                )
            elif spec.runner_type == "inline":
                self._runners.append(
                    InlineStageRunner(
                        stage,
                        self._ctx,
                        spec.worker_cap,
                        **common,
                    )
                )
            elif spec.runner_type == "process":
                self._runners.append(
                    ProcessStageRunner(
                        stage,
                        self._ctx,
                        spec.worker_cap,
                        queue_capacity=spec.queue_capacity,
                        mp_context=self._mp_context,
                        coalesce_tensors=spec.coalesce_tensors,
                        shm_min_size=spec.shm_min_size,
                        max_worker_retries=spec.max_worker_retries,
                        ipc_transport=spec.ipc_transport,
                        ipc_buffer_bytes=spec.ipc_buffer_bytes,
                        **common,
                    )
                )
            elif spec.runner_type == "remote":
                from zephon.runners.ray import RemoteStageRunner

                self._runners.append(
                    RemoteStageRunner(
                        stage,
                        self._ctx,
                        spec.worker_cap,
                        queue_capacity=spec.queue_capacity,
                        **common,
                    )
                )
            else:
                raise ValueError(f"Unknown runner '{spec.runner_type}'")

    # ------------------------------------------------------------------
    # Shared-memory inflight counter (used by MTP mode for zero-copy reads)
    # ------------------------------------------------------------------

    def attach_inflight_counter(self, shm: ctypes.Array[ctypes.c_int]) -> None:
        """Wire up a shared ``multiprocessing.RawArray`` for inflight counts.

        Called by the MTP worker after engine construction.  The array is
        indexed by lane_id; the engine writes ``len(inflight_chunks_per_lane[lane])``
        after every chunk admission or eviction.  The main process reads
        at any time with no IPC round-trip.
        """
        self._inflight_shm = shm

    def _sync_inflight_shm(self, lane_id: int) -> None:
        """Update the shared-memory counter for *lane_id*."""
        shm = self._inflight_shm
        if shm is None:
            return
        if 0 <= lane_id < len(shm):
            shm[lane_id] = len(self.inflight_chunks_per_lane.get(lane_id, {}))
            return
        warnings.warn(
            f"inflight shm counter skipped: lane_id={lane_id} out of range "
            f"[0, {len(shm)}); inflight_summary() will not include this lane.",
            RuntimeWarning,
            stacklevel=2,
        )

    def inflight_summary(self) -> dict[int, int]:
        """Return the number of inflight chunks per lane.

        Lightweight alternative to ``state_dict()`` for monitoring — returns
        only ``{lane_id: chunk_count}`` without serialising chunk contents.
        Safe to call from the main thread while workers are running; if a
        lane mutates concurrently we skip just that lane rather than dropping
        the whole summary.
        """
        try:
            lanes = list(self.inflight_chunks_per_lane)
        except RuntimeError:
            return {}
        result: dict[int, int] = {}
        for lane in lanes:
            try:
                chunks = self.inflight_chunks_per_lane.get(lane)
                if chunks is not None:
                    result[int(lane)] = len(chunks)
            except RuntimeError:
                continue
        return result

    def metrics_snapshot(self):
        """Return a clone of the current pipeline metrics summary when enabled."""
        if self._collector is None:
            return None
        return self._collector.snapshot()

    def pump_metrics_snapshot(self):
        """Return a clone of the current pump-timing summary when enabled."""
        if self._collector is None:
            return None
        return self._collector.snapshot_pump_timing()

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

    def _emit_pump_metrics(self, delta: Any) -> None:
        """Forward pump-thread timing metrics to the collector when tracking is enabled."""
        if self._collector is not None and self._collector.tracking_mode.collects_nodes:
            self._collector.record_pump_timing(delta)

    def _get_component_id(self, component_name: str) -> int:
        """Return the source-declared ID for a component name."""
        cid = self._component_to_id.get(component_name)
        if cid is None:
            raise ValueError(
                f"Unknown mixture component {component_name!r}: not in the "
                + "work source's component_ids() vocabulary "
                + f"{sorted(self._component_to_id)}. The vocabulary must cover "
                + "every component the source ever emits."
            )
        return cid

    def _get_component_name(self, component_id: int) -> str | None:
        """Look up component name by ID for human-readable warning messages.

        Returns None if ID is unknown.
        """
        return self._id_to_component.get(component_id)

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
        """Store a chunk's normalized mixture as component-id weights.

        A stamped ``target_mixture`` takes precedence over the counted
        composition (see :attr:`WorkChunk.target_mixture`); an unset (``None``)
        target falls back to the counts. Both are normalized on the chunk, so
        this only maps component names to ids.
        """
        mixture = chunk.target_mixture if chunk.target_mixture else chunk.mixture
        if not mixture:
            return

        id_mixture = {
            self._get_component_id(name): weight for name, weight in mixture.items()
        }
        with self._mixture_lock:
            self._chunk_mixtures[(lane_id, chunk_id)] = id_mixture

    @staticmethod
    def _make_flush_sentinel(lane_id: LaneId, *, boundary_cid: int = 0) -> SampleRecord:
        """Create a flush sentinel record for the given lane.

        Args:
            lane_id: The lane this sentinel belongs to.
            boundary_cid: First chunk_id of the new epoch.  All chunks below
                this value belong to the completed epoch and are eligible for
                eviction (subject to the all_done check).
        """
        meta = SampleMeta(
            sample_id=(0, 0, 0),
            lane_id=lane_id,
            chunk_id=0,
            chunk_offset=0,
            tags={"_flush_sentinel": True, "_boundary_cid": boundary_cid},
        )
        return SampleRecord(meta=meta, payload={})

    def _lane_stream(self, lane_id: LaneId) -> Iterator[EngineSample | SampleRecord]:
        """Yield EngineSamples for a single lane, fetching chunks lazily.

        Maintains inflight_chunks_per_lane[lane_id][chunk_id] -> chunk_obj.

        Each EngineSample is a 5-tuple:
            (sample_id, lane_id, chunk_id, chunk_offset, component_id)

        WorkChunk yields (sample_id, component_name) tuples; we convert
        component_name to component_id for efficient downstream processing.

        When ``flush_every_k_chunks > 0``, a flush sentinel SampleRecord is
        injected after every K-th chunk.  In Phase 1 (replay), sentinels are
        re-injected at stored epoch boundary positions.
        """
        inflight_lane = self.inflight_chunks_per_lane[lane_id]

        # Phase 1: replay restored inflight chunks first (ascending chunk_id).
        # Re-inject flush sentinels at stored epoch boundary positions so that
        # accumulators see the same flush points as the original run.
        # The consumer thread evicts (pops) concurrently, so an unguarded scan
        # raises under free-threaded Python; retry rather than lock — this is a
        # cold path (once per lane at stream start), not the hot eviction path.
        epoch_boundary_set = set(self._epoch_boundaries.get(lane_id, []))
        while True:
            try:
                sorted_inflight_cids = sorted(inflight_lane.keys())
                break
            except RuntimeError:
                pass
        for cid in sorted_inflight_cids:
            if cid in epoch_boundary_set:
                yield self._make_flush_sentinel(lane_id, boundary_cid=cid)
            chunk = inflight_lane[cid]
            # Store mixture for restored chunks (may already exist, but idempotent)
            self._store_chunk_mixture(lane_id, cid, chunk)
            for offset, (sample_id, component_name) in enumerate(chunk):
                component_id = self._get_component_id(component_name)
                # Note that we yield the _entire_ chunk here. This can break with elastic continuation in case a batch is cross-chunk boundaries.
                yield (sample_id, lane_id, int(cid), offset, component_id)

        # Trailing boundary: if a boundary exceeds all inflight cids, the
        # sentinel between the last Phase 1 chunk and the first Phase 2 chunk
        # must still fire so the accumulator flushes at the epoch edge.
        if sorted_inflight_cids and epoch_boundary_set:
            max_inflight = sorted_inflight_cids[-1]
            for b in sorted(b for b in epoch_boundary_set if b > max_inflight):
                yield self._make_flush_sentinel(lane_id, boundary_cid=b)

        # Phase 2: fetch new chunks and assign stable per-lane ids.
        # Inject a flush sentinel every K chunks per lane.
        ws = self._lane_ws[lane_id]
        K = self._flush_every_k_chunks
        # Seed the open epoch's chunk count from the admission counter, not
        # from live inflight. The open epoch starts at last_boundary (0 before
        # the first boundary); _lane_next_cid is one past the last admitted
        # chunk, so next_cid - last_boundary is how many chunks the epoch holds
        # — including ones admitted before the checkpoint and already evicted,
        # which a live inflight count would miss.
        phase1_boundaries = self._epoch_boundaries.get(lane_id, [])
        if K > 0:
            last_boundary = max(phase1_boundaries) if phase1_boundaries else 0
            chunks_in_epoch = max(0, self._lane_next_cid[lane_id] - last_boundary)
        else:
            chunks_in_epoch = 0
        while True:
            with self._checkpoint_lock:
                # Since internally we prefetch, this could overlap with a checkpointing call.
                # We need to ensure that we are not prefetching while updating the lane state.
                chunk = ws.next_chunk()
                if chunk is None:
                    break  # lane exhausted → emit final sentinel below

                cid = int(self._lane_next_cid[lane_id])
                self._lane_next_cid[lane_id] = cid + 1
                inflight_lane[cid] = chunk
                self._sync_inflight_shm(lane_id)
                # Store mixture weights for this chunk
                self._store_chunk_mixture(lane_id, cid, chunk)

                # Epoch accounting and boundary append in the SAME lock block
                # as chunk admission.  If they were separate, a checkpoint
                # between the two blocks would capture the K-th chunk but not
                # its boundary, breaking deterministic replay on restore.
                chunks_in_epoch += 1
                # boundary_cid = first chunk of the new epoch
                emit_boundary: int | None = None
                if K > 0 and chunks_in_epoch >= K:
                    emit_boundary = cid + 1
                    # Record boundary under checkpoint lock BEFORE yielding
                    # the sentinel.  This ensures any checkpoint snapshot that
                    # includes the chunks also includes the boundary —
                    # critical for the thread/process runners where the feeder
                    # and checkpoint run on different threads.
                    self._epoch_boundaries[lane_id].append(emit_boundary)
                    chunks_in_epoch = 0

            for offset, (sample_id, component_name) in enumerate(chunk):
                component_id = self._get_component_id(component_name)
                yield (sample_id, lane_id, cid, offset, component_id)

            if emit_boundary is not None:
                yield self._make_flush_sentinel(lane_id, boundary_cid=emit_boundary)

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
            ptr = self._lane_progress[lane]  # defaultdict auto-creates
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

    def _source_stream(self) -> Iterator[EngineSample | SampleRecord]:
        """Yield sample identifiers from the backing work source."""
        owned = self._owned_lanes

        # Build one generator per lane
        gens: dict[LaneId, Iterator[EngineSample | SampleRecord]] = {
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
            # Sentinels (tombstones, etc.) are transparent to RR scheduling —
            # they must not consume a slot, otherwise the interleaving order
            # diverges from the pre-checkpoint baseline after a replay.
            item = buffers[lane].popleft()
            yield item
            if not is_sentinel(item):
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
                    item = buffers[lane].popleft()
                    yield item
                    if not is_sentinel(item):
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
        self,
        lane_id: int,
        max_chunk_id: int,
        add_k: int,
        max_cursor: SampleCursor | None,
    ) -> None:
        """Cursor-ordered notify path (no cross-chunk reordering or packing).

        Evict chunks older than *max_chunk_id*, advance lane progress by
        *add_k* delivered records, and update the replay cursor.

        The caller pre-computes *add_k* (number of records whose chunk_id
        equals *max_chunk_id*) and *max_cursor* (greatest cursor among those
        records) so this method performs zero allocations on the common path.
        """
        inflight_lane = self.inflight_chunks_per_lane[lane_id]

        # 1) Evict older inflight chunks — O(1) fast-path skip.
        if inflight_lane:
            # Deliberately no _checkpoint_lock on this per-sample hot path: we
            # keep eviction lock-free and pay the cost on the cold reader
            # instead — _lane_stream scans inflight once per lane at stream
            # start and retries there on a racing pop.
            # list() can still race a feeder dispatch insert under free-threaded
            # Python; skipping is safe (retried on the next delivered sample).
            try:
                snapshot = list(inflight_lane)
            except RuntimeError:
                snapshot = []
            if snapshot and snapshot[0] < max_chunk_id:
                to_evict = [cid for cid in snapshot if cid < max_chunk_id]
                for cid in to_evict:
                    inflight_lane.pop(cid, None)
                self._sync_inflight_shm(lane_id)
                with self._mixture_lock:
                    for cid in to_evict:
                        self._chunk_mixtures.pop((lane_id, cid), None)
                if _DEBUG_EVICT:
                    # Compute remaining from local snapshots — reading
                    # ``inflight_lane`` here would race with concurrent feeder
                    # mutations under free-threaded Python.
                    remaining = len(snapshot) - len(to_evict)
                    print(
                        f"[zephon.evict.monotone] lane={lane_id} cids={to_evict} "
                        f"remaining_inflight~={remaining} "
                        f"max_chunk_id={max_chunk_id} {rank_ctx()}",
                        file=sys.stderr,
                        flush=True,
                    )

        # 2) Lane progress — mutate LanePtr in-place (mutable dataclass).
        cur = self._lane_progress[lane_id]  # defaultdict auto-creates
        if cur.chunk_id == max_chunk_id:
            cur.offset += add_k
        else:
            cur.chunk_id = max_chunk_id
            cur.offset = add_k

        # 3) Cursor tracking.
        if max_cursor is not None:
            previous = self._lane_last_cursor.get(lane_id)
            if previous is None or max_cursor > previous:
                self._lane_last_cursor[lane_id] = max_cursor

    def _accumulator_eviction_floor(self) -> int | None:
        """Return the global minimum epoch floor across all runners.

        The epoch floor is the lowest chunk_id that influenced any
        ``preserves_cursor_order=False`` accumulator since the last sentinel
        flush.  Chunks below this floor are candidates for atomic eviction
        (subject to the ``all_done`` bitmap check).

        Returns None if no records have entered any accumulator yet or if
        the floor was just reset after a sentinel flush.
        """
        floor: int | None = None
        for runner in self._runners:
            wm = runner.epoch_floor()
            if wm is not None:
                floor = min(floor, wm) if floor is not None else wm
        return floor

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

        # 2) Per-epoch eviction: walk epoch boundaries bottom-up, evict
        #    completed epochs contiguously from the lowest.
        #
        #    The pump (feeder thread) runs ahead of delivery, so the
        #    accumulator epoch floor may span many epoch boundaries.
        #    Instead of requiring ALL chunks below the floor to be done
        #    (which never passes when the pump is ahead), we evaluate
        #    each epoch independently via _epoch_boundaries.
        #
        #    Epoch i covers [boundaries[i-1], boundaries[i]).
        #    We stop at the cursor's epoch or the first incomplete epoch
        #    to keep eviction contiguous from the bottom.
        last_completed_cid = -1
        last_completed_offset = 0
        cids_to_evict: list[int] = []
        accum_floor = self._accumulator_eviction_floor()
        # Snapshot inflight keys and boundaries under the same lock so the
        # two views are consistent.  The feeder thread mutates both under
        # _checkpoint_lock; without the lock, free-threaded Python can
        # observe mid-iteration appends, causing premature epoch eviction.
        with self._checkpoint_lock:
            try:
                cid_snapshot = list(inflight_lane.keys())
            except RuntimeError:
                cid_snapshot = []
            boundaries = list(self._epoch_boundaries.get(lane_id, []))

        if boundaries and cid_snapshot:
            # Per-epoch eviction: walk boundaries ascending, evaluate each
            # epoch independently.
            cursor_cid = record_cursor.chunk_id if record_cursor is not None else None
            already_queued: set[int] = set()

            for boundary_cid in sorted(boundaries):
                epoch_cids = [
                    cid
                    for cid in cid_snapshot
                    if cid < boundary_cid
                    and cid not in already_queued
                    and inflight_lane.get(cid) is not None
                ]
                if not epoch_cids:
                    continue  # already evicted or empty

                # Cursor pinning: stop at the cursor's epoch.
                if cursor_cid is not None and cursor_cid in epoch_cids:
                    break

                # all_done per epoch — the load-bearing safety invariant.
                if not all(
                    cid in done and done_count[cid] >= len(inflight_lane[cid])
                    for cid in epoch_cids
                ):
                    break  # first incomplete epoch stops contiguous eviction

                cids_to_evict.extend(epoch_cids)
                already_queued.update(epoch_cids)
                last_completed_cid = max(epoch_cids)
                last_completed_offset = len(inflight_lane[last_completed_cid])

        elif accum_floor is not None and cid_snapshot:
            # Fallback: no epoch boundaries (flush_every_k_chunks=0 or no
            # sentinel fired yet).  All chunks below accum_floor are in a
            # single epoch — use the original atomic "all below floor" check.
            below = [
                cid
                for cid in cid_snapshot
                if cid < accum_floor and inflight_lane.get(cid) is not None
            ]
            all_below_done = all(
                cid in done and done_count[cid] >= len(inflight_lane[cid])
                for cid in below
            )
            if all_below_done and below:
                # Cursor pinning: abort if cursor is in the eviction set.
                if record_cursor is not None and record_cursor.chunk_id in below:
                    pass  # don't evict
                else:
                    cids_to_evict = below
                    last_completed_cid = max(below)
                    last_completed_offset = len(inflight_lane[last_completed_cid])

        # Deliberately no lock on this hot-path pop (see notify_monotone): we
        # keep eviction lock-free and let the cold reader _lane_stream retry on
        # a racing resize instead. The snapshot above takes _checkpoint_lock
        # only for a consistent decision view vs dispatch, not to guard this pop.
        for cid in cids_to_evict:
            inflight_lane.pop(cid, None)
            done.pop(cid, None)
            done_count.pop(cid, None)
        if cids_to_evict:
            self._sync_inflight_shm(lane_id)
            with self._mixture_lock:
                for cid in cids_to_evict:
                    self._chunk_mixtures.pop((lane_id, cid), None)
            # Prune stale epoch boundaries below evicted range.
            # Must hold _checkpoint_lock so the read-filter-replace is atomic
            # w.r.t. feeder appends and checkpoint reads.
            if lane_id in self._epoch_boundaries:
                max_evicted = max(cids_to_evict)
                with self._checkpoint_lock:
                    self._epoch_boundaries[lane_id] = [
                        b for b in self._epoch_boundaries[lane_id] if b > max_evicted
                    ]
            if _DEBUG_EVICT:
                _rc_cid = record_cursor.chunk_id if record_cursor is not None else None
                # Compute remaining from local snapshots — reading
                # ``inflight_lane`` here would race with concurrent feeder
                # mutations under free-threaded Python.
                remaining = len(cid_snapshot) - len(cids_to_evict)
                print(
                    f"[zephon.evict.non-monotone] lane={lane_id} cids={cids_to_evict} "
                    f"remaining_inflight~={remaining} "
                    f"accum_floor={accum_floor} record_cursor_cid={_rc_cid} {rank_ctx()}",
                    file=sys.stderr,
                    flush=True,
                )

        # 3) Maintain lane progress for fairness diagnostics — mutate in-place.
        cur = self._lane_progress[lane_id]  # defaultdict auto-creates
        # Free-threaded Python: dict iteration can race; stale progress is harmless.
        try:
            front_cid = next(iter(inflight_lane), None)
        except RuntimeError:
            front_cid = None
        if front_cid is not None:
            cur.chunk_id = front_cid
            cur.offset = done_count.get(front_cid, 0)
        elif last_completed_cid >= 0:
            cur.chunk_id = last_completed_cid
            cur.offset = last_completed_offset
        # else: cur stays at default (chunk_id=-1, offset=0)

        # 4) Track latest record-level cursor for replay
        if record_cursor is not None:
            self._lane_last_cursor[lane_id] = record_cursor

    def record_delivery(self, lane_id: int) -> None:
        """Count one consumer-delivered item at the pipeline tail for *lane_id*.

        Called from the tail notify path (``_apply_notify_args``) once per
        item actually yielded to the consumer — a batch when batching is
        present, a record otherwise. Tombstones and flush sentinels are
        excluded upstream (they are notified but never delivered), which is
        what keeps replayed-and-dropped records after a checkpoint restore
        from double-counting.

        Purely observability: the counters feed the mid-window checkpoint
        warning and never influence scheduling, replay, or RR emission.
        """
        self._lane_emitted[lane_id] += 1

    @staticmethod
    def _complete_lane_counts(
        lane_emitted: dict[int, int], canonical_replicas: int
    ) -> list[int] | None:
        """Counts for every canonical lane, or None if any lane is unknown."""
        counts: list[int] = []
        for lane in range(canonical_replicas):
            count = lane_emitted.get(lane)
            if count is None:
                return None
            counts.append(count)
        return counts

    @staticmethod
    def _check_mid_window_counts(
        lane_emitted: dict[int, int], canonical_replicas: int
    ) -> None:
        """Warn loudly when per-lane delivery counts indicate a mid-window cut.

        Only fires when counts are known for EVERY canonical lane (partial
        coverage means another shard holds the rest, or the counters are
        unknown because the run was resumed from a pre-counter checkpoint).
        """
        counts = Engine._complete_lane_counts(lane_emitted, canonical_replicas)
        if counts is None:
            return
        lo, hi = min(counts), max(counts)
        if lo == hi:
            return
        warnings.warn(
            f"[zephon] Checkpoint taken mid-window: emitted-batch counts per "
            f"canonical lane are unequal (range [{lo}..{hi}] across "
            f"{canonical_replicas} lanes). Resuming this checkpoint into a "
            f"DIFFERENT topology will permute the remainder of the current "
            f"global window (no samples are lost or duplicated). To avoid "
            f"this, checkpoint at window boundaries: the number of global "
            f"batches per pooled step must be a multiple of "
            f"canonical_replicas ({canonical_replicas}).",
            RuntimeWarning,
            stacklevel=3,
        )

    def _topology_differs(self, ckpt_world: dict[str, Any]) -> bool:
        """Best-effort check for a lane-ownership topology change vs *ckpt_world*.

        Merged checkpoints carry only ``canonical_replicas`` and
        ``world_size``; single-shard (fast-path) checkpoints additionally
        carry ``dp_degree`` and the full lane ``mapping``. Compare whatever
        is present; missing information conservatively counts as "same" so
        the load-time mid-window warning never fires spuriously.
        """
        ws = ckpt_world.get("world_size")
        if ws is not None and int(ws) != int(self._world.world_size):
            return True
        dp = ckpt_world.get("dp_degree")
        if dp is not None and int(dp) != int(self._world.dp_degree):
            return True
        mapping = ckpt_world.get("mapping")
        if mapping:
            current = {
                int(dp_id): [int(x) for x in lanes]
                for dp_id, lanes in self._world.lanes_for_dp_group.items()
            }
            saved = {
                int(dp_id): [int(x) for x in lanes] for dp_id, lanes in mapping.items()
            }
            if saved != current:
                return True
        return False

    def eval_one(self, sample: SampleId | EngineSample) -> Any:
        """Synchronously evaluate a single element through every stage runner."""
        value: Any = sample
        for runner in self._runners:
            value = runner.run_one(value)
        return value

    def close(self) -> None:
        """Close all stage runners, suppressing teardown errors."""
        if self._closed:
            return
        self._closed = True
        hard = self._opts.shutdown_mode == "hard"
        for runner in self._runners:
            try:
                runner.close(hard=hard)
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

            self._refresh_rr_from_progress()

            for purge_candidate_str in [
                "_lane_ws",
                "inflight_chunks_per_lane",
                "_lane_progress",
                "_lane_next_cid",
                "_lane_last_cursor",
                "_offset_done",
                "_offset_done_count",
                "_epoch_boundaries",
                "_lane_emitted",
            ]:
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
            # rr_next_idx is owner-keyed, not lane-keyed: don't add it to the
            # purge loop above (its `lane not in owned` check would drop our own
            # key). Peer keys are filtered out on load instead.
            rr_next_idx = dict(self._rr_next_idx)

            replay_cursors: dict[int, Any] = {}
            for lane, cursor in self._lane_last_cursor.items():
                replay_cursors[int(lane)] = (
                    cursor.as_key() if cursor is not None else None
                )

            epoch_boundaries: dict[int, list[int]] = {}
            for lane, boundaries in self._epoch_boundaries.items():
                if boundaries:
                    epoch_boundaries[int(lane)] = [int(cid) for cid in boundaries]

            lane_emitted: dict[int, int] = {}
            if self._lane_emitted_valid:
                lane_emitted = {lane: self._lane_emitted.get(lane, 0) for lane in owned}

            state = EngineStateV1(
                version=ENGINE_VERSION,
                world=world,
                inflight=inflight,
                progress=progress,
                lane_next_cid=lane_next,
                work_source=self._work.state_dict(),
                lane_ws_state=lane_ws_state,
                last_round_id=self._last_round_id,
                checkpoint_reload_count=self._checkpoint_reload_count,
                rr_next_idx=rr_next_idx,
                replay_cursors=replay_cursors,
                epoch_boundaries=epoch_boundaries,
                lane_emitted=lane_emitted,
            )
            return state.to_dict()

    # ---------- FS utilities ----------
    def _read_text(self, path: str) -> str | None:
        try:
            with self._agg_backend.open(path, "r") as f:
                return str(f.read())
        except Exception:
            return None

    def _read_aggregation_state(self, path: str) -> dict | None:
        try:
            with self._agg_backend.open(path, "rb") as f:
                raw = f.read()
        except Exception:
            return None
        try:
            return self._agg_codec.decode(raw)
        except Exception as exc:
            self._log(f"Failed to decode aggregation state at {path}: {exc!r}")
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
        if age_s > self._agg_timeout_s + 30:
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
        return f"{self._agg_dir}/state_r{self._world.global_rank}_w{wid}_p{pid}_{round_id}.ckpt"

    def _merged_file_path(self, round_id: str) -> str:
        return f"{self._agg_dir}/merged_{round_id}.ckpt"

    def _list_state_files(self, round_id: str) -> list[str]:
        pattern = f"{self._agg_dir}/state_r*_w*_p*_{round_id}.ckpt"
        return self._agg_backend.glob(pattern)

    def _read_states_for_round(
        self,
        round_id: str,
        *,
        seen: dict[str, dict[str, Any]] | None = None,
        covered: set[int] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], set[int]]:
        """Read per-rank state files for *round_id*.

        When *seen* and *covered* are supplied, only newly appeared files are
        fetched — previously read files are skipped.  This turns repeated
        polling from O(N * polls) GETs into O(N) total.
        """
        if seen is None:
            seen = {}
        if covered is None:
            covered = set()
        for fp in self._list_state_files(round_id):
            if fp in seen:
                continue
            st = self._read_aggregation_state(fp)
            if st is None:
                continue
            seen[fp] = st
            for k in st.get("progress", {}):
                covered.add(int(k))
        return seen, covered

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
            # Fast path has no merge; this owner covers every lane, so check locally.
            self._check_mid_window_counts(
                local.get("lane_emitted") or {},
                int(self._world.canonical_replicas),
            )
            return local

        # Round setup
        is_leader = self._world.global_rank == 0 and worker_id == 0
        round_id = self._open_round_id(is_leader)
        self._last_round_id = round_id
        my_path = self._state_file_path(round_id)
        self._agg_backend.put(my_path, self._agg_codec.encode(local))

        merged_path = self._merged_file_path(round_id)
        if is_leader:
            expected_lanes = set(range(int(self._world.canonical_replicas)))
            expected_ranks = set(range(int(self._world.world_size)))
            seen: dict[str, dict[str, Any]] = {}
            covered: set[int] = set()

            def _have_full_coverage() -> bool:
                self._read_states_for_round(round_id, seen=seen, covered=covered)
                ranks_seen = {int(s["world"]["global_rank"]) for s in seen.values()}
                return ranks_seen == expected_ranks and expected_lanes.issubset(covered)

            if not self._wait_until(_have_full_coverage, self._agg_timeout_s):
                ranks_seen = {int(s["world"]["global_rank"]) for s in seen.values()}
                missing_ranks = sorted(expected_ranks - ranks_seen)
                missing_lanes = sorted(expected_lanes - covered)
                raise RuntimeError(
                    f"[PID {os.getpid()}] Aggregation timeout after "
                    f"{self._agg_timeout_s}s. Missing ranks={missing_ranks}; "
                    f"missing lanes={missing_lanes}; "
                    f"files={len(self._list_state_files(round_id))}"
                )

            # States already cached from polling above.
            merged = self._merge_state_dicts(list(seen.values()))
            assert not self._agg_backend.exists(merged_path)
            merged_bytes = self._agg_codec.encode(merged)
            self._agg_backend.put(merged_path, merged_bytes)
            # Roundtrip so the leader's return value is byte-identical to
            # what followers read back (json would otherwise stringify keys).
            merged = self._agg_codec.decode(merged_bytes)
            self._agg_backend.delete(my_path)
            if self._previous_merged_file is not None:
                self._agg_backend.delete(self._previous_merged_file)
            self._previous_merged_file = merged_path
            # Safe: by the coverage check above every rank has already read
            # round.current and is now waiting on merged_path.
            self._agg_backend.delete(self._round_file)
            return merged

        # Followers: wait for merged file
        ok = self._wait_until(
            lambda: self._agg_backend.exists(merged_path), self._agg_timeout_s
        )
        if not ok:
            raise RuntimeError(
                f"[PID {os.getpid()}] Timed out after {self._agg_timeout_s}s waiting for merged checkpoint at {merged_path}"
            )
        merged = self._read_aggregation_state(merged_path)
        self._agg_backend.delete(my_path)
        if merged is None:
            raise RuntimeError(f"Failed to read merged checkpoint {merged_path}")
        return merged

    def _dedupe_dp_group_peers(
        self, states: list[EngineStateV1]
    ) -> list[EngineStateV1]:
        """Collapse same-DP-group, same-owned-lane-set duplicates to one rep.

        Under 3D parallelism (``world_size > dp_degree``) every rank in a DP
        group owns the same canonical lanes and writes a state file; their
        lane-keyed entries would collide in the downstream merge. One rep
        per ``(dp_group_id, owned-lane-set)`` group survives, with
        ``_DELIVERY_SYNCED_FIELDS`` strict-checked across the group.

        Operates on typed ``EngineStateV1`` so callers can run schema
        migration and cross-shard global-invariant checks BEFORE dedupe —
        otherwise a divergence in a universal field (``world.world_size``,
        ``last_round_id``, ``checkpoint_reload_count``) between same-DP-group
        peers would be silently hidden when peers are dropped.

        Why the lane-set component (not just ``dp_group_id``): with
        ``workers_per_rank > 1`` multiple DataLoader workers on the same
        rank split their DP group's lanes into disjoint subsets (see
        ``_owned_lanes``); those state files are complementary partitions,
        not duplicates, and must not be collapsed. At
        ``workers_per_rank=1`` the lane-set is redundant and grouping
        collapses to ``dp_group_id``.
        """
        groups: dict[tuple[int, frozenset[int]], list[EngineStateV1]] = {}
        for st in states:
            dp_id = int(st.world["dp_group_id"])
            lane_set = frozenset(int(k) for k in st.progress)
            groups.setdefault((dp_id, lane_set), []).append(st)

        chosen: list[EngineStateV1] = []
        for (dp_id, lane_set), group in groups.items():
            # Deterministic representative: lowest global_rank wins.
            rep = min(group, key=lambda st: int(st.world["global_rank"]))
            for st in group:
                if st is rep:
                    continue
                for fname in _DELIVERY_SYNCED_FIELDS:
                    if getattr(st, fname) != getattr(rep, fname):
                        raise RuntimeError(
                            f"DP group {dp_id} delivery-state divergence on "
                            f"{fname!r} (lanes={sorted(lane_set)}) between rank "
                            f"{int(rep.world['global_rank'])} and rank "
                            f"{int(st.world['global_rank'])}. Did all ranks "
                            f"barrier before calling pipe.checkpoint()?"
                        )
            chosen.append(rep)
        return chosen

    def _merge_state_dicts(self, states: list[dict[str, Any]]) -> dict[str, Any]:
        assert states
        # Load + migrate every shard up front. Same-DP-group peer dedupe runs
        # AFTER the cross-shard global-invariant checks below; otherwise a
        # divergence in a universal field between same-DP-group peers would
        # be silently dropped together with the peer.
        typed = [EngineStateV1.load(s) for s in states]
        errors: list[str] = []

        C = int(typed[0].world["canonical_replicas"])
        for idx, st in enumerate(typed[1:], start=1):
            if int(st.world["canonical_replicas"]) != C:
                errors.append(
                    f"states[{idx}].world.canonical_replicas="
                    f"{st.world['canonical_replicas']!r} != states[0]={C}"
                )

        merged_world_size: int | None = None
        for idx, st in enumerate(typed):
            w = st.world
            if "world_size" in w and w["world_size"] is not None:
                n = int(w["world_size"])
                if merged_world_size is None:
                    merged_world_size = n
                elif merged_world_size != n:
                    errors.append(
                        f"states[{idx}].world.world_size={n} mismatches "
                        f"previously-seen={merged_world_size}"
                    )

        last_round_ids = {st.last_round_id for st in typed}
        if len(last_round_ids) != 1:
            errors.append(f"last_round_id mismatch across shards: {last_round_ids!r}")

        checkpoint_reload_counts = {st.checkpoint_reload_count for st in typed}
        if len(checkpoint_reload_counts) != 1:
            errors.append(
                f"checkpoint_reload_count mismatch across shards: "
                f"{checkpoint_reload_counts!r}"
            )

        # Fail fast on universal-field divergence so it isn't masked by dedupe
        # dropping the divergent peer.
        if errors:
            raise RuntimeError(
                f"cannot merge {len(states)} aggregation state shards:\n  - "
                + "\n  - ".join(errors)
            )
        merged_last_round_id = next(iter(last_round_ids))
        merged_reload_count = next(iter(checkpoint_reload_counts))

        typed = self._dedupe_dp_group_peers(typed)

        inflight, progress, lane_next, lane_ws_state = {}, {}, {}, {}
        for idx, st in enumerate(typed):
            for lane_s, by_chunk in st.inflight.items():
                lane = int(lane_s)
                if lane in inflight:
                    errors.append(f"states[{idx}]: duplicate inflight for lane {lane}")
                    continue
                inflight[lane] = {
                    int(cid): payload for cid, payload in by_chunk.items()
                }

            for lane_s, p in st.progress.items():
                lane = int(lane_s)
                if lane in progress:
                    errors.append(f"states[{idx}]: duplicate progress for lane {lane}")
                    continue
                progress[lane] = {
                    "chunk_id": int(p["chunk_id"]),
                    "offset": int(p["offset"]),
                }

            for lane_s, nxt in st.lane_next_cid.items():
                lane = int(lane_s)
                if lane in lane_next:
                    errors.append(
                        f"states[{idx}]: duplicate lane_next_cid for lane {lane}"
                    )
                    continue
                lane_next[lane] = int(nxt)

            for lane_s, ws in st.lane_ws_state.items():
                lane = int(lane_s)
                if lane in lane_ws_state:
                    errors.append(
                        f"states[{idx}]: duplicate lane_ws_state for lane {lane}"
                    )
                    continue
                lane_ws_state[lane] = ws

        work_config = next((st.work_config for st in typed if st.work_config), None)

        rr_next_idx: dict[str, int] = {}
        for idx, st in enumerate(typed):
            for key, val in st.rr_next_idx.items():
                if key in rr_next_idx and rr_next_idx[key] != int(val):
                    errors.append(
                        f"states[{idx}]: rr_next_idx[{key!r}]={int(val)} conflicts "
                        f"with previously-merged value {rr_next_idx[key]}"
                    )
                    continue
                rr_next_idx[key] = int(val)

        replay_cursors: dict[int, Any] = {}
        for idx, st in enumerate(typed):
            for lane_s, key in st.replay_cursors.items():
                lane = int(lane_s)
                if lane in replay_cursors:
                    errors.append(
                        f"states[{idx}]: duplicate replay_cursors for lane {lane}"
                    )
                    continue
                replay_cursors[lane] = key

        epoch_boundaries: dict[int, list[int]] = {}
        for idx, st in enumerate(typed):
            for lane_s, boundaries in st.epoch_boundaries.items():
                lane = int(lane_s)
                if lane in epoch_boundaries:
                    errors.append(
                        f"states[{idx}]: duplicate epoch_boundaries for lane {lane}"
                    )
                    continue
                epoch_boundaries[lane] = [int(cid) for cid in boundaries]

        # Lanes are disjoint across shards (peers deduped above): merge by union.
        # Unknown shards contribute nothing; the check stays silent without full coverage.
        lane_emitted: dict[int, int] = {}
        for idx, st in enumerate(typed):
            for lane_s, count in st.lane_emitted.items():
                lane = int(lane_s)
                if lane in lane_emitted:
                    errors.append(
                        f"states[{idx}]: duplicate lane_emitted for lane {lane}"
                    )
                    continue
                lane_emitted[lane] = int(count)

        if errors:
            raise RuntimeError(
                f"cannot merge {len(states)} aggregation state shards:\n  - "
                + "\n  - ".join(errors)
            )

        self._check_mid_window_counts(lane_emitted, C)

        merged = EngineStateV1(
            version=ENGINE_VERSION,
            world={"canonical_replicas": C, "world_size": merged_world_size},
            inflight=inflight,
            progress=progress,
            lane_next_cid=lane_next,
            work_config=work_config,
            lane_ws_state=lane_ws_state,
            last_round_id=merged_last_round_id,
            checkpoint_reload_count=merged_reload_count,
            rr_next_idx=rr_next_idx,
            replay_cursors=replay_cursors,
            epoch_boundaries=epoch_boundaries,
            lane_emitted=lane_emitted,
        )
        return merged.to_dict()

    def load_state_dict(self, state: dict[str, Any], *, replay: bool = True) -> None:
        """Restore engine & WorkSource; enter replay mode if 'replay' is True."""
        ckpt = EngineStateV1.load(state)
        # ckpt.work_source is intentionally unread: it is captured on write for
        # debugging only. The authoritative WorkSource is the one constructed
        # in-process; per-lane state is restored from ckpt.lane_ws_state below.
        if int(ckpt.world["canonical_replicas"]) != self._world.canonical_replicas:
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
        for lane_s, by_chunk in ckpt.inflight.items():
            lane = int(lane_s)
            if lane not in owned:
                continue
            self.inflight_chunks_per_lane[lane] = {}
            for cid_s, payload in by_chunk.items():
                cid = int(cid_s)
                self.inflight_chunks_per_lane[lane][cid] = WorkChunk.from_state(payload)
            self._sync_inflight_shm(lane)

        self._lane_progress.clear()
        for lane in owned:
            p = ckpt.progress.get(str(lane)) or ckpt.progress.get(int(lane))
            if p is not None:
                self._lane_progress[lane] = LanePtr(
                    int(p["chunk_id"]), int(p["offset"])
                )
            else:
                raise RuntimeError(
                    f"Lane {lane} does not have progress state in the checkpoint!"
                )

        self._lane_next_cid.clear()
        for lane in owned:
            saved_next = ckpt.lane_next_cid.get(str(lane)) or ckpt.lane_next_cid.get(
                int(lane)
            )
            if saved_next is None:
                raise RuntimeError(
                    f"Lane {lane} does not have next cid in the checkpoint"
                )
            self._lane_next_cid[lane] = int(saved_next)

        # Restore lane WS state for owned lanes
        for lane, ws in self._lane_ws.items():
            st = ckpt.lane_ws_state.get(str(lane)) or ckpt.lane_ws_state.get(int(lane))
            if st is not None:
                ws.load_state_dict(dict(st))
            else:
                raise RuntimeError(
                    f"Lane {lane} does not have WorkSource state in the checkpoint"
                )

        self._last_round_id = ckpt.last_round_id
        self._checkpoint_reload_count = ckpt.checkpoint_reload_count + 1
        self._agg_backend.mkdir(self._agg_dir, parents=True, exist_ok=True)

        # rr_next_idx is owner-keyed ("{global_rank}:..."); keep only our own so
        # peers' stale pointers can't collide in the next merge (_refresh below
        # re-derives our key from progress).
        own_prefix = f"{self._opts.global_rank}:"
        rr_raw = ckpt.rr_next_idx or {}
        self._rr_next_idx = {
            k: int(v) for k, v in rr_raw.items() if k.startswith(own_prefix)
        }

        replay_raw = ckpt.replay_cursors or {}
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

        # Restore epoch boundaries for flush sentinel re-injection on replay.
        self._epoch_boundaries.clear()
        for lane_s, boundaries in (ckpt.epoch_boundaries or {}).items():
            lane = int(lane_s)
            if lane in owned:
                self._epoch_boundaries[lane] = [int(cid) for cid in boundaries]

        # Pre-counter checkpoints lack entries for owned lanes -> baseline unknown.
        emitted_raw = ckpt.lane_emitted or {}
        restored_emitted = {int(lane_s): int(c) for lane_s, c in emitted_raw.items()}
        self._lane_emitted.clear()
        self._lane_emitted_valid = all(lane in restored_emitted for lane in owned)
        if self._lane_emitted_valid:
            for lane in owned:
                self._lane_emitted[lane] = restored_emitted[lane]

        C = int(self._world.canonical_replicas)
        counts = self._complete_lane_counts(restored_emitted, C)
        if (
            counts is not None
            and min(counts) != max(counts)
            and self._topology_differs(ckpt.world)
        ):
            warnings.warn(
                f"[zephon] Resuming a mid-window checkpoint into a different "
                f"topology: emitted-batch counts per canonical lane are "
                f"unequal (range [{min(counts)}..{max(counts)}] across {C} "
                f"lanes) and the lane-ownership topology changed. The "
                f"remainder of the current global window will be permuted "
                f"(no samples are lost or duplicated). To avoid this, "
                f"checkpoint at window boundaries: the number of global "
                f"batches per pooled step must be a multiple of "
                f"canonical_replicas ({C}).",
                RuntimeWarning,
                stacklevel=2,
            )

        self._refresh_rr_from_progress()
        self._publish_replay_snapshot()
